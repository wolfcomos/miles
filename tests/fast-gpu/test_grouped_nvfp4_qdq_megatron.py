"""Packed NVFP4 fake QAT through Megatron TEGroupedMLP: EP/EDP, DDP accumulation, checkpoints, step time."""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=900,
    suite="stage-c-8-gpu-b200",
    labels=["precision", "megatron"],
    hardware=["blackwell"],
)

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_WORKER = Path(__file__).with_name("_grouped_nvfp4_qdq_megatron_worker.py")
_REPO_ROOT = Path(__file__).parents[2]
_STEP_TIMES_JSON = os.environ.get("GROUPED_NVFP4_QDQ_STEP_TIMES_JSON")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="packed NVFP4 fake QAT requires SM10x",
)


def _run_worker(case: str, nproc: int, recipe: str, *extra: str) -> str:
    if torch.cuda.device_count() < nproc:
        pytest.skip(f"{case} needs {nproc} GPUs")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_REPO_ROOT), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={nproc}",
            str(_WORKER),
            "--case",
            case,
            "--recipe",
            recipe,
            *extra,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=1500,
    )
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    assert f"PASS {case} {recipe}" in result.stdout, result.stdout[-4000:]
    print("\n".join(line for line in result.stdout.splitlines() if line.startswith("[rank ")))
    return result.stdout


def test_baseline_packed_weights_without_fake_qat() -> None:
    _run_worker("baseline", 1, "standard")


@pytest.mark.parametrize("recipe", ["standard", "w4a16"])
def test_ep1_packed_matches_discrete_per_expert_fake_qdq(recipe: str) -> None:
    _run_worker("ep1_parity", 1, recipe)


@pytest.mark.parametrize("recipe", ["standard", "w4a16"])
def test_ddp_accumulation_and_optimizer_steps(recipe: str) -> None:
    _run_worker("accumulation", 1, recipe)


def test_ep2_matches_ep1_reference() -> None:
    _run_worker("ep2", 2, "w4a16")


def test_edp2_grad_reduction_and_param_sync() -> None:
    _run_worker("edp2", 4, "w4a16")


def test_checkpoint_save_reload(tmp_path: Path) -> None:
    _run_worker("checkpoint", 1, "w4a16", "--ckpt-dir", str(tmp_path / "ckpt"))


def test_step_time_probe(tmp_path: Path) -> None:
    output = _STEP_TIMES_JSON or str(tmp_path / "megatron_step_times.json")
    stdout = _run_worker("step_time", 1, "w4a16", "--output", output)
    assert "fake_qat_packed" in stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
