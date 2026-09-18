"""Benchmark grouped fused NVFP4 QDQ against the per-expert loop on identical inputs.

Kernel-only paths (reference-only stacking never enters the timed region):
  per_expert_qdq       Miles ``fused_nvfp4_qdq`` per expert; contiguous views and amax precomputed
  per_expert_amax_qdq  per-expert ``compute_nvfp4_amax`` + ``fused_nvfp4_qdq``
  grouped_qdq          ``fused_grouped_nvfp4_qdq`` with precomputed per-group amax
  grouped_amax_qdq     ``compute_grouped_nvfp4_amax`` + ``fused_grouped_nvfp4_qdq``
  te_per_expert_qdq    production TE ``NVFP4Quantizer`` (rowwise, no RHT/2D/SR) quantize + dequantize per expert
Adapter paths (complete ``maybe_fake_quantize_nvfp4_weight_tensors`` latency on a TE GroupedLinear):
  adapter_packed       single grouped weight: amax + grouped QDQ + GroupedTensor wrap + STE
  adapter_discrete     G discrete weight Parameters through the per-expert STE

Every path reports eager and CUDA-graph replay latency (CUDA events, median/p10/p90
microseconds), CUDA launch counts from torch.profiler, and the peak allocation delta
of one call. Grouped launch counts are checked to stay constant across G. Run inside
the Miles container on one SM10x GPU, e.g.::

    NVTE_USE_FAST_MATH=0 NVTE_GROUPED_LINEAR_SINGLE_PARAM=1 \\
        python tools/benchmark_grouped_nvfp4_qdq.py --out /work/results/bench
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import statistics
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from typing import Any

import torch  # torch before transformer_engine: TE core links a different NCCL.
import transformer_engine
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.module.grouped_linear import GroupedLinear

from miles.utils.fused_grouped_nvfp4_qdq import compute_grouped_nvfp4_amax, fused_grouped_nvfp4_qdq
from miles.utils.fused_nvfp4_qdq import (
    NVFP4QDQConfig,
    NVFP4QDQErrorMode,
    compute_nvfp4_amax,
    current_nvfp4_qdq_config,
    fused_nvfp4_qdq,
)
from miles.utils.nvfp4_fake_qat import NVFP4_FAKE_QAT_FLAG, maybe_fake_quantize_nvfp4_weight_tensors

_KERNEL_PATHS = ("per_expert_qdq", "per_expert_amax_qdq", "grouped_qdq", "grouped_amax_qdq", "te_per_expert_qdq")
_ADAPTER_PATHS = {"adapter_packed": True, "adapter_discrete": False}
# name -> (baseline path, candidate path); speedup = baseline / candidate.
_SPEEDUPS = {
    "speedup_qdq": ("per_expert_qdq", "grouped_qdq"),
    "speedup_amax_qdq": ("per_expert_amax_qdq", "grouped_amax_qdq"),
    "speedup_te_vs_grouped": ("te_per_expert_qdq", "grouped_amax_qdq"),
    "speedup_adapter": ("adapter_discrete", "adapter_packed"),
}
# config name -> (config, TE environment that makes current_nvfp4_qdq_config() and TE's
# 4over6 candidate-error math agree with it; None unsets the variable).
_CONFIGS: dict[str, tuple[NVFP4QDQConfig, dict[str, str | None]]] = {
    "nvfp4": (
        NVFP4QDQConfig(),
        {
            "NVTE_NVFP4_4OVER6": None,
            "NVTE_NVFP4_4OVER6_E4M3_USE_256": None,
            "NVTE_NVFP4_4OVER6_ERR_MODE": None,
            "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH": None,
        },
    ),
    "w4a16_4over6": (
        NVFP4QDQConfig(
            use_4over6=True, e4m3_max=448, error_mode=NVFP4QDQErrorMode.MSE, error_use_fast_math=True
        ),
        {
            "NVTE_NVFP4_4OVER6": "all",
            "NVTE_NVFP4_4OVER6_E4M3_USE_256": "none",
            "NVTE_NVFP4_4OVER6_ERR_MODE": "MSE",
            "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH": "1",
        },
    ),
}


@dataclass(frozen=True)
class _Case:
    label: str
    num_groups: int
    rows: int
    cols: int
    dtype: torch.dtype


def _shape_sets() -> dict[str, list[_Case]]:
    small = [_Case("small", g, 128, 256, torch.bfloat16) for g in (1, 3, 8, 16, 32)]
    mid = [_Case("mid", g, 1024, 2048, torch.bfloat16) for g in (1, 3, 8, 16, 32)]
    mid.append(_Case("mid", 8, 1024, 2048, torch.float16))
    # GLM-5.2 744B-A40B per-rank expert weights at ETP=1: hidden 6144, moe_ffn 2048
    # (gated FC1 out 4096), 256 experts over EP32/16/8/4.
    glm = [_Case("glm_fc1", g, 4096, 6144, torch.bfloat16) for g in (8, 16, 32, 64)]
    glm += [_Case("glm_fc2", g, 6144, 2048, torch.bfloat16) for g in (8, 16, 32, 64)]
    return {"small": small, "mid": mid, "glm": glm}


def _make_input(case: _Case) -> torch.Tensor:
    torch.manual_seed(case.num_groups * 100003 + case.rows * 7 + case.cols)
    x = torch.randn(case.num_groups, case.rows, case.cols, dtype=case.dtype, device="cuda")
    for g in range(case.num_groups):
        x[g] *= 10.0 ** ((g % 5) - 2)
    return x


def _apply_env(env: dict[str, str | None]) -> None:
    for name, value in env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _nvidia_smi(fields: str) -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"
    return completed.stdout.strip()


def _versions() -> dict[str, Any]:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    try:
        cutlass_dsl = metadata.version("nvidia-cutlass-dsl")
    except metadata.PackageNotFoundError:
        cutlass_dsl = "unknown"
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformer_engine": transformer_engine.__version__,
        "cutlass_dsl": cutlass_dsl,
        "cublasLt": tex.get_cublasLt_version(),
        "gpu": props.name,
        "sm_count": props.multi_processor_count,
        "compute_capability": f"{props.major}.{props.minor}",
        "nvidia_smi": _nvidia_smi("name,driver_version,clocks.sm,clocks.applications.graphics,clocks.max.sm"),
    }


def _stats_us(samples_us: list[float]) -> dict[str, float]:
    ordered = sorted(samples_us)
    deciles = statistics.quantiles(ordered, n=10)
    quartiles = statistics.quantiles(ordered, n=4)
    return {
        "median_us": statistics.median(ordered),
        "p10_us": deciles[0],
        "p90_us": deciles[-1],
        "iqr_us": quartiles[2] - quartiles[0],
        "min_us": ordered[0],
        "n": len(ordered),
    }


def _time_eager(fn: Callable[[], Any], warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    wall_start = time.perf_counter()
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    stats = _stats_us([start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)])
    stats["wall_mean_us"] = (time.perf_counter() - wall_start) * 1e6 / iters
    return stats


def _time_graph(fn: Callable[[], Any], warmup: int, iters: int) -> dict[str, Any]:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            fn()
    except Exception as exc:  # noqa: BLE001 - report non-capturable paths instead of aborting the sweep
        torch.cuda.synchronize()
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}
    return _time_eager(graph.replay, warmup, iters)


def _count_launches(fn: Callable[[], Any]) -> dict[str, Any]:
    fn()
    torch.cuda.synchronize()
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as prof:
        fn()
        torch.cuda.synchronize()
    kernel_names: dict[str, int] = {}
    copies = 0
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        if event.name.startswith(("Memcpy", "Memset")):
            copies += 1
        else:
            kernel_names[event.name[:80]] = kernel_names.get(event.name[:80], 0) + 1
    return {"kernels": sum(kernel_names.values()), "memcpy_memset": copies, "kernel_names": kernel_names}


def _peak_alloc_delta_bytes(fn: Callable[[], Any]) -> int:
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del out
    return peak


def _measure(fn: Callable[[], Any], output_bytes: int, args: argparse.Namespace, graph: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"launches": _count_launches(fn)}
    result["peak_alloc_delta_bytes"] = _peak_alloc_delta_bytes(fn)
    result["temp_bytes"] = result["peak_alloc_delta_bytes"] - output_bytes
    result["eager"] = _time_eager(fn, args.warmup, args.iters)
    if graph:
        result["graph"] = _time_graph(fn, args.warmup, args.iters)
    return result


def _make_te_quantizer(config: NVFP4QDQConfig) -> te.NVFP4Quantizer:
    """Production TE rowwise quantizer with the same contract as the Miles kernels."""
    return te.NVFP4Quantizer(
        rowwise=True,
        columnwise=False,
        with_amax_reduction=False,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
        row_scaled_nvfp4=False,
        nvfp4_use_4over6=config.use_4over6,
        nvfp4_e4m3_max=config.e4m3_max,
        nvfp4_4over6_err_mode=config.error_mode.name,
        with_random_sign_mask=False,
    )


def _kernel_paths(
    x: torch.Tensor, config: NVFP4QDQConfig, quantizer: te.NVFP4Quantizer
) -> dict[str, Callable[[], Any]]:
    views = [x[g] for g in range(x.shape[0])]
    amax_list = [compute_nvfp4_amax(view) for view in views]
    amax = compute_grouped_nvfp4_amax(x)
    dtype = x.dtype
    return {
        "per_expert_qdq": lambda: [fused_nvfp4_qdq(view, a, config) for view, a in zip(views, amax_list)],
        "per_expert_amax_qdq": lambda: [fused_nvfp4_qdq(view, compute_nvfp4_amax(view), config) for view in views],
        "grouped_qdq": lambda: fused_grouped_nvfp4_qdq(x, amax, config),
        "grouped_amax_qdq": lambda: fused_grouped_nvfp4_qdq(x, compute_grouped_nvfp4_amax(x), config),
        "te_per_expert_qdq": lambda: [quantizer.quantize(view).dequantize(dtype=dtype) for view in views],
    }


def _bitwise_checks(x: torch.Tensor, paths: dict[str, Callable[[], Any]]) -> dict[str, bool]:
    """Integer-view equality of the grouped output against both per-expert oracles (untimed)."""
    grouped = paths["grouped_qdq"]()
    per_expert = torch.stack(paths["per_expert_qdq"]())
    te_out = torch.stack(paths["te_per_expert_qdq"]())
    per_expert_amax = torch.stack([compute_nvfp4_amax(x[g]) for g in range(x.shape[0])])
    return {
        "grouped_eq_per_expert": torch.equal(grouped.view(torch.int16), per_expert.view(torch.int16)),
        "grouped_eq_te": torch.equal(grouped.view(torch.int16), te_out.view(torch.int16)),
        "grouped_amax_eq_per_expert": torch.equal(compute_grouped_nvfp4_amax(x), per_expert_amax),
    }


def _build_grouped_linear(x: torch.Tensor, packed: bool) -> GroupedLinear:
    num_groups, rows, cols = x.shape
    kwargs: dict[str, Any] = {"single_grouped_weight": True, "use_grouped_tensor": True} if packed else {}
    module = GroupedLinear(num_groups, cols, rows, bias=False, params_dtype=x.dtype, device="cuda", **kwargs)
    with torch.no_grad():
        if packed:
            module.weight.rowwise_data.view_as(x).copy_(x)
        else:
            for g in range(num_groups):
                getattr(module, f"weight{g}").copy_(x[g])
    return module


def _measure_adapter(x: torch.Tensor, packed: bool, output_bytes: int, args: argparse.Namespace) -> dict[str, Any]:
    module = _build_grouped_linear(x, packed)
    weight_tensors = module._get_weight_tensors()
    fn = functools.partial(maybe_fake_quantize_nvfp4_weight_tensors, weight_tensors)
    result = _measure(fn, output_bytes, args, graph=args.graph_adapter)
    result["launches"]["weight_tensors"] = len(weight_tensors)
    del fn, weight_tensors, module
    return result


def _speedups(row: dict[str, Any]) -> dict[str, float | None]:
    speedups: dict[str, float | None] = {}
    for name, (baseline, candidate) in _SPEEDUPS.items():
        if baseline not in row or candidate not in row:
            continue
        speedups[name] = row[baseline]["eager"]["median_us"] / row[candidate]["eager"]["median_us"]
        graph_base, graph_cand = row[baseline].get("graph", {}), row[candidate].get("graph", {})
        if "median_us" in graph_base and "median_us" in graph_cand:
            speedups[f"{name}_graph"] = graph_base["median_us"] / graph_cand["median_us"]
    return speedups


def _case_fields(case: _Case, config_name: str) -> dict[str, Any]:
    return {
        "label": case.label,
        "num_groups": case.num_groups,
        "rows": case.rows,
        "cols": case.cols,
        "dtype": str(case.dtype).removeprefix("torch."),
        "config": config_name,
    }


def _run_case(
    case: _Case, config_name: str, config: NVFP4QDQConfig, quantizer: te.NVFP4Quantizer, args: argparse.Namespace
) -> dict[str, Any]:
    x = _make_input(case)
    output_bytes = x.numel() * x.element_size()
    paths = _kernel_paths(x, config, quantizer)
    row = _case_fields(case, config_name)
    row["output_bytes"] = output_bytes
    row["bitwise"] = _bitwise_checks(x, paths)
    for name in _KERNEL_PATHS:
        if name == "te_per_expert_qdq" and args.no_te:
            continue
        row[name] = _measure(paths[name], output_bytes, args, graph=not args.no_graph)
    if not args.no_adapter:
        for name, packed in _ADAPTER_PATHS.items():
            row[name] = _measure_adapter(x, packed, output_bytes, args)
    row.update(_speedups(row))
    del paths, x
    torch.cuda.empty_cache()
    return row


def _launch_checks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Grouped launch counts must not depend on G; grouped QDQ must be exactly one kernel."""
    groups: dict[tuple, dict[str, set[int]]] = {}
    for row in rows:
        if "grouped_qdq" not in row:
            continue
        key = (row["label"], row["rows"], row["cols"], row["dtype"], row["config"])
        counts = groups.setdefault(key, {"grouped_qdq": set(), "grouped_amax_qdq": set()})
        for path in counts:
            counts[path].add(row[path]["launches"]["kernels"] + row[path]["launches"]["memcpy_memset"])
    checks = []
    for key, counts in groups.items():
        checks.append(
            {
                "case": key,
                "grouped_qdq_launches": sorted(counts["grouped_qdq"]),
                "grouped_amax_qdq_launches": sorted(counts["grouped_amax_qdq"]),
                "passed": counts["grouped_qdq"] == {1} and len(counts["grouped_amax_qdq"]) == 1,
            }
        )
    return checks


def _fmt_us(entry: dict[str, Any] | None, key: str = "eager") -> str:
    if entry is None:
        return "-"
    stats = entry.get(key, {})
    return f"{stats['median_us']:.1f}" if "median_us" in stats else "n/a"


def _fmt_launches(entry: dict[str, Any] | None) -> str:
    if entry is None:
        return "-"
    launches = entry["launches"]
    return f"{launches['kernels']}+{launches['memcpy_memset']}"


def _fmt_ratio(value: float | None) -> str:
    return f"{value:.2f}x" if value is not None else "-"


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| shape | G | N | K | dtype | config | per-exp QDQ us | grouped QDQ us | speedup | "
        "per-exp amax+QDQ us | grouped amax+QDQ us | speedup | TE per-exp us | "
        "launches per-exp/grouped/grouped amax+QDQ | graph per-exp/grouped QDQ us | "
        "adapter discrete/packed us | speedup | bitwise |"
    )
    lines = [header, "|" + "---|" * 18]
    for row in rows:
        if "skipped" in row:
            lines.append(
                f"| {row['label']} | {row['num_groups']} | {row['rows']} | {row['cols']} | {row['dtype']} | "
                f"{row['config']} | {row['skipped']} |" + " |" * 11
            )
            continue
        bitwise = row["bitwise"]
        lines.append(
            f"| {row['label']} | {row['num_groups']} | {row['rows']} | {row['cols']} | {row['dtype']} | {row['config']} "
            f"| {_fmt_us(row.get('per_expert_qdq'))} | {_fmt_us(row.get('grouped_qdq'))} | {_fmt_ratio(row.get('speedup_qdq'))} "
            f"| {_fmt_us(row.get('per_expert_amax_qdq'))} | {_fmt_us(row.get('grouped_amax_qdq'))} "
            f"| {_fmt_ratio(row.get('speedup_amax_qdq'))} | {_fmt_us(row.get('te_per_expert_qdq'))} "
            f"| {_fmt_launches(row.get('per_expert_qdq'))} / {_fmt_launches(row.get('grouped_qdq'))} "
            f"/ {_fmt_launches(row.get('grouped_amax_qdq'))} "
            f"| {_fmt_us(row.get('per_expert_qdq'), 'graph')} / {_fmt_us(row.get('grouped_qdq'), 'graph')} "
            f"| {_fmt_us(row.get('adapter_discrete'))} / {_fmt_us(row.get('adapter_packed'))} "
            f"| {_fmt_ratio(row.get('speedup_adapter'))} "
            f"| pe={bitwise['grouped_eq_per_expert']} te={bitwise['grouped_eq_te']} amax={bitwise['grouped_amax_eq_per_expert']} |"
        )
    return "\n".join(lines)


def _flatten(value: Any, prefix: str = "", out: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {} if out is None else out
    if isinstance(value, dict) and not prefix.endswith("kernel_names"):
        for key, item in value.items():
            _flatten(item, f"{prefix}.{key}" if prefix else key, out)
    else:
        out[prefix] = json.dumps(value) if isinstance(value, dict) else value
    return out


def _write_results(out_dir: str, results: dict[str, Any]) -> None:
    with open(os.path.join(out_dir, "results.json"), "w") as handle:
        json.dump(results, handle, indent=1, default=str)
    flat_rows = [_flatten(row) for row in results["rows"]]
    columns = sorted({key for row in flat_rows for key in row}, key=lambda k: (k.count("."), k))
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(flat_rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=str, required=True, help="Directory for results.json, results.csv, summary.md.")
    parser.add_argument("--shape-sets", nargs="+", default=["small", "mid", "glm"], choices=list(_shape_sets()))
    parser.add_argument("--configs", nargs="+", default=list(_CONFIGS), choices=list(_CONFIGS))
    parser.add_argument("--warmup", type=int, default=10, help="Untimed iterations per path (default: 10).")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations per path (default: 50).")
    parser.add_argument("--no-graph", action="store_true", help="Skip CUDA-graph replay timing of kernel paths.")
    parser.add_argument("--no-te", action="store_true", help="Skip the TE per-expert quantize/dequantize baseline.")
    parser.add_argument("--no-adapter", action="store_true", help="Skip the GroupedLinear adapter paths.")
    parser.add_argument(
        "--graph-adapter",
        action="store_true",
        help="Also try to CUDA-graph the adapter paths; autograd may make them non-capturable.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.out, exist_ok=True)
    # The adapter is a no-op without the flag; this benchmark measures the enabled path.
    os.environ[NVFP4_FAKE_QAT_FLAG] = "1"
    results: dict[str, Any] = {"args": vars(args), "versions": _versions(), "clocks_sm_after_row": [], "rows": []}
    print(json.dumps(results["versions"], indent=1), flush=True)
    cases = [case for shape_set in args.shape_sets for case in _shape_sets()[shape_set]]
    for config_name in args.configs:
        config, env = _CONFIGS[config_name]
        _apply_env(env)
        if current_nvfp4_qdq_config() != config:
            raise RuntimeError(f"Environment for {config_name} resolves to {current_nvfp4_qdq_config()}, expected {config}.")
        quantizer = _make_te_quantizer(config)
        for case in cases:
            print(f"[{config_name}] {case}", flush=True)
            try:
                row = _run_case(case, config_name, config, quantizer, args)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                row = {**_case_fields(case, config_name), "skipped": f"OOM: {exc}"[:200]}
            results["rows"].append(row)
            results["clocks_sm_after_row"].append(_nvidia_smi("clocks.sm"))
            _write_results(args.out, results)
    results["checks"] = _launch_checks(results["rows"])
    _write_results(args.out, results)
    table = _markdown_table(results["rows"])
    with open(os.path.join(args.out, "summary.md"), "w") as handle:
        handle.write(table + "\n")
    print(table, flush=True)
    for check in results["checks"]:
        print(f"launch check {'PASS' if check['passed'] else 'FAIL'}: {check}", flush=True)
    if not all(check["passed"] for check in results["checks"]):
        raise SystemExit("grouped launch counts are not G-independent or grouped QDQ is not a single launch")


if __name__ == "__main__":
    main()
