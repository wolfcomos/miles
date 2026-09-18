"""Grouped fused CuTe DSL NVFP4 quantize-dequantize for packed expert weights.

One launch fake-quantizes the packed weights of all local experts. The input is
an ordinary contiguous ``[G, N, K]`` tensor whose group ``g`` starts at element
offset ``g * N * K``, exactly how Transformer Engine packs ``weight0, weight1,
...`` into a single grouped parameter. Every expert is quantized against its own
FP32 per-tensor amax; nothing is reduced across experts or ranks.

The kernel shares every numerical helper with :mod:`miles.utils.fused_nvfp4_qdq`
and therefore inherits its contract: 1x16 rowwise E4M3 block scaling, round to
nearest, standard NVFP4 plus the full Four Over Six matrix, and no RHT, 2D
scaling, stochastic rounding, or transpose output. Expert ``g`` of the grouped
output is bit-identical to ``fused_nvfp4_qdq(x[g], amax[g])``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from miles.utils.fused_nvfp4_qdq import (
    _4OVER6_BLOCKS_PER_SM,
    _4OVER6_THREADS,
    _FP4_BLOCK_SIZE,
    _INT32_MAX,
    _STANDARD_GRID_BLOCKS_PER_SM,
    _STANDARD_MIN_BLOCKS_PER_SM,
    _STANDARD_THREADS,
    NVFP4QDQConfig,
    _block_amax,
    _dequantize_store,
    _device_info,
    _fdiv_rn,
    _four_over_six_quantize,
    _get_ptr,
    _global_encode_scale,
    _input_values,
    _load_v4_u32,
    _standard_quantize,
    current_nvfp4_qdq_config,
)
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

# Each 1x16 block is 32 bytes for both BF16 and FP16.
_BLOCK_BYTES = 2 * _FP4_BLOCK_SIZE
# Experts are selected by the second grid dimension.
_MAX_GROUPS = 65535


class _GroupedNVFP4QDQKernel:
    """One thread processes one contiguous 1x16 block; ``blockIdx.y`` selects the expert."""

    def __init__(self, is_bfloat16: bool, config: NVFP4QDQConfig) -> None:
        self.is_bfloat16 = is_bfloat16
        self.config = config
        if config.use_4over6:
            self.threads = _4OVER6_THREADS
            self.min_blocks_per_sm = _4OVER6_BLOCKS_PER_SM
            self.grid_blocks_per_sm = _4OVER6_BLOCKS_PER_SM
        else:
            self.threads = _STANDARD_THREADS
            self.min_blocks_per_sm = _STANDARD_MIN_BLOCKS_PER_SM
            self.grid_blocks_per_sm = _STANDARD_GRID_BLOCKS_PER_SM

    @cute.jit
    def __call__(
        self,
        input_tensor: cute.Tensor,
        output_tensor: cute.Tensor,
        group_amax: cute.Tensor,
        blocks_per_group: Int32,
        ctas_per_group: Int32,
        num_groups: Int32,
        stream,
    ) -> None:
        self.kernel(input_tensor, output_tensor, group_amax, blocks_per_group).launch(
            grid=[ctas_per_group, num_groups, 1],
            block=[self.threads, 1, 1],
            max_number_threads=[self.threads, 1, 1],
            min_blocks_per_mp=self.min_blocks_per_sm,
            smem=0,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        input_tensor: cute.Tensor,
        output_tensor: cute.Tensor,
        group_amax: cute.Tensor,
        blocks_per_group: Int32,
    ) -> None:
        """Quantize and immediately dequantize grid-stride 1x16 blocks of one expert."""
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, group_idx, _ = cute.arch.block_idx()
        grid_dim, _, _ = cute.arch.grid_dim()

        amax = Float32(group_amax[group_idx])
        global_encode_scale = _global_encode_scale(amax, self.config.e4m3_max)
        global_decode_scale = _fdiv_rn(Float32(1.0), global_encode_scale)

        # Byte addressing in 64-bit: the packed payload of all experts may exceed
        # 2**31 elements even though each expert is bounded by _INT32_MAX.
        group_bytes = Int64(group_idx) * Int64(blocks_per_group) * Int64(_BLOCK_BYTES)
        input_base = _get_ptr(input_tensor, Int32(0)) + group_bytes
        output_base = _get_ptr(output_tensor, Int32(0)) + group_bytes
        block = block_idx * Int32(self.threads) + thread_idx
        stride = grid_dim * Int32(self.threads)
        while block < blocks_per_group:
            block_bytes = Int64(block) * Int64(_BLOCK_BYTES)
            ptr0 = input_base + block_bytes
            w0, w1, w2, w3 = _load_v4_u32(ptr0)
            w4, w5, w6, w7 = _load_v4_u32(ptr0 + Int64(_BLOCK_BYTES // 2))
            words = (w0, w1, w2, w3, w4, w5, w6, w7)
            block_amax = _block_amax(words, self.is_bfloat16)

            if cutlass.const_expr(self.config.use_4over6):
                values = _input_values(words, self.is_bfloat16)
                scale, lo, hi = _four_over_six_quantize(
                    values, block_amax, amax, global_encode_scale, global_decode_scale, self.config
                )
            else:
                scale, lo, hi = _standard_quantize(
                    words, block_amax, global_encode_scale, global_decode_scale, self.is_bfloat16
                )

            out_ptr0 = output_base + block_bytes
            _dequantize_store(
                out_ptr0,
                out_ptr0 + Int64(_BLOCK_BYTES // 2),
                lo,
                hi,
                scale,
                amax,
                self.config.e4m3_max,
                self.is_bfloat16,
            )
            block = block + stride


@dataclass(frozen=True)
class _GroupedNVFP4QDQSpecialization:
    """Compiled callable and its statically selected launch geometry."""

    launch: Any
    threads: int
    grid_blocks_per_sm: int


_GROUPED_KERNEL_CACHE: dict[tuple[Any, ...], _GroupedNVFP4QDQSpecialization] = {}


def _validate_grouped_input(x: torch.Tensor, amax: torch.Tensor) -> tuple[int, tuple[int, int], int]:
    if not x.is_cuda:
        raise ValueError("Grouped fused NVFP4 QDQ requires a CUDA tensor.")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"Grouped fused NVFP4 QDQ supports BF16 and FP16, got {x.dtype}.")
    if x.ndim != 3:
        raise ValueError(f"Grouped fused NVFP4 QDQ requires a rank-3 [G, N, K] tensor, got shape {tuple(x.shape)}.")
    if not x.is_contiguous():
        raise ValueError("Grouped fused NVFP4 QDQ requires a contiguous tensor.")
    if x.data_ptr() % 16 != 0:
        raise ValueError("Grouped fused NVFP4 QDQ requires a 16-byte-aligned input tensor.")
    num_groups, rows, cols = x.shape
    if num_groups <= 0 or rows <= 0 or cols <= 0:
        raise ValueError(f"Grouped fused NVFP4 QDQ requires positive dimensions, got shape {tuple(x.shape)}.")
    if num_groups > _MAX_GROUPS:
        raise ValueError(f"Grouped fused NVFP4 QDQ supports at most {_MAX_GROUPS} groups, got {num_groups}.")
    if cols % _FP4_BLOCK_SIZE != 0:
        raise ValueError(f"Grouped fused NVFP4 QDQ requires K divisible by {_FP4_BLOCK_SIZE}, got {cols}.")
    if rows * cols > _INT32_MAX:
        raise ValueError(
            f"Grouped fused NVFP4 QDQ supports at most {_INT32_MAX} elements per group, got {rows * cols}."
        )
    if not amax.is_cuda or amax.device != x.device:
        raise ValueError("The FP32 per-group amax must be on the input tensor's CUDA device.")
    if amax.dtype != torch.float32 or amax.shape != (num_groups,):
        raise TypeError(
            f"The per-group amax must be an FP32 tensor of shape ({num_groups},), got {amax.dtype} {tuple(amax.shape)}."
        )
    device_index = x.device.index
    if device_index is None:
        raise RuntimeError("CUDA tensor does not have a concrete device index.")
    capability, multiprocessors = _device_info(device_index)
    if capability[0] != 10:
        raise ValueError(f"Grouped fused NVFP4 QDQ requires SM10x, got compute capability {capability}.")
    return device_index, capability, multiprocessors


def _compile_grouped_specialization(dtype: torch.dtype, config: NVFP4QDQConfig) -> _GroupedNVFP4QDQSpecialization:
    """Compile one dtype/config specialization outside the steady-state path."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up grouped fused NVFP4 QDQ before CUDA graph capture.")
    kernel = _GroupedNVFP4QDQKernel(dtype == torch.bfloat16, config)
    element_type = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
    # Row-major rank-2 [G, N*K] fake tensors keep every extent below 2**31; the
    # kernel only takes their base pointers and addresses the payload in 64-bit bytes.
    input_fake = cute.runtime.make_fake_compact_tensor(
        element_type, (cute.sym_int(), cute.sym_int()), stride_order=(1, 0), assumed_align=16
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        element_type, (cute.sym_int(), cute.sym_int()), stride_order=(1, 0), assumed_align=16
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(),), assumed_align=4)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        kernel,
        input_fake,
        output_fake,
        amax_fake,
        Int32(1),
        Int32(1),
        Int32(1),
        stream_fake,
        options="--enable-tvm-ffi",
    )
    return _GroupedNVFP4QDQSpecialization(
        launch=compiled, threads=kernel.threads, grid_blocks_per_sm=kernel.grid_blocks_per_sm
    )


def _grouped_launch_geometry(
    blocks_per_group: int, num_groups: int, threads: int, grid_blocks_per_sm: int, multiprocessors: int
) -> int:
    """Share the single-kernel CTA budget across experts instead of multiplying it by G."""
    ctas_to_cover_group = (blocks_per_group + threads - 1) // threads
    budget_per_group = max(1, (multiprocessors * grid_blocks_per_sm + num_groups - 1) // num_groups)
    return min(ctas_to_cover_group, budget_per_group)


def _launch_fused_grouped_nvfp4_qdq(
    x: torch.Tensor,
    amax: torch.Tensor,
    config: NVFP4QDQConfig,
    capability: tuple[int, int],
    multiprocessors: int,
) -> torch.Tensor:
    """Launch on the current CUDA device with cached static dispatch."""
    key = (capability, x.dtype, config)
    specialization = _GROUPED_KERNEL_CACHE.get(key)
    if specialization is None:
        specialization = _compile_grouped_specialization(x.dtype, config)
        _GROUPED_KERNEL_CACHE[key] = specialization

    num_groups, rows, cols = x.shape
    output = torch.empty_like(x)
    input_2d = x.detach().view(num_groups, rows * cols)
    output_2d = output.view(num_groups, rows * cols)
    amax_flat = amax.detach().view(num_groups)
    blocks_per_group = (rows * cols) // _FP4_BLOCK_SIZE
    ctas_per_group = _grouped_launch_geometry(
        blocks_per_group, num_groups, specialization.threads, specialization.grid_blocks_per_sm, multiprocessors
    )
    specialization.launch(input_2d, output_2d, amax_flat, blocks_per_group, ctas_per_group, num_groups)
    return output


def compute_grouped_nvfp4_amax(x: torch.Tensor) -> torch.Tensor:
    """Compute one TE-compatible FP32 per-tensor amax per group of a ``[G, N, K]`` tensor."""
    if x.ndim != 3:
        raise ValueError(f"Grouped NVFP4 amax requires a rank-3 [G, N, K] tensor, got shape {tuple(x.shape)}.")
    if x.numel() == 0:
        raise ValueError("Cannot compute NVFP4 amax for an empty tensor.")
    # The reduction upcasts on the fly; no FP32 copy of the payload is materialized.
    return torch.linalg.vector_norm(x.detach().reshape(x.shape[0], -1), ord=float("inf"), dim=1, dtype=torch.float32)


def fused_grouped_nvfp4_qdq(x: torch.Tensor, amax: torch.Tensor, config: NVFP4QDQConfig | None = None) -> torch.Tensor:
    """Run one register-resident NVFP4 QDQ launch over all groups and return a detached tensor."""
    if config is None:
        config = current_nvfp4_qdq_config()
    device_index, capability, multiprocessors = _validate_grouped_input(x, amax)

    if torch.cuda.current_device() == device_index:
        return _launch_fused_grouped_nvfp4_qdq(x, amax, config, capability, multiprocessors)
    with torch.cuda.device(device_index):
        return _launch_fused_grouped_nvfp4_qdq(x, amax, config, capability, multiprocessors)


def _packed_weight_shape(weight: GroupedTensor) -> tuple[int, int, int]:
    """Validate a BF16/FP16 single grouped weight and return its ``[G, N, K]`` payload shape."""
    if not isinstance(weight, GroupedTensor):
        raise TypeError(f"Packed NVFP4 fake QAT requires a TE GroupedTensor, got {type(weight).__name__}.")
    if weight.quantizer is not None:
        raise ValueError("Packed NVFP4 fake QAT requires a high-precision grouped weight (quantizer=None).")
    if not weight.all_same_shape() or not weight.tensor_shapes or len(weight.tensor_shapes[0]) != 2:
        raise ValueError("Packed NVFP4 fake QAT requires uniform rank-2 expert weights.")
    payload = weight.rowwise_data
    if payload is None or not payload.is_contiguous():
        raise ValueError("Packed NVFP4 fake QAT requires a contiguous rowwise payload.")
    if payload.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"Packed NVFP4 fake QAT supports BF16 and FP16 payloads, got {payload.dtype}.")
    num_groups = weight.num_tensors
    rows, cols = weight.tensor_shapes[0]
    if payload.numel() != num_groups * rows * cols or tuple(weight.shape) != (num_groups, rows, cols):
        raise ValueError(
            f"Packed NVFP4 fake QAT payload/shape mismatch: payload {payload.numel()} elements, "
            f"wrapper shape {tuple(weight.shape)}, members {num_groups} x {(rows, cols)}."
        )
    return num_groups, rows, cols


def fused_grouped_nvfp4_qdq_packed_weight(
    weight: GroupedTensor, config: NVFP4QDQConfig | None = None
) -> GroupedTensor:
    """Fake-quantize the current payload of a packed grouped weight into a new GroupedTensor."""
    num_groups, rows, cols = _packed_weight_shape(weight)
    # Read the live payload every call: optimizer steps, DDP rebinding, and checkpoint
    # loads all write through weight.rowwise_data.
    x = weight.rowwise_data.view(num_groups, rows, cols)
    output = fused_grouped_nvfp4_qdq(x, compute_grouped_nvfp4_amax(x), config)
    return GroupedTensor.make_grouped_tensor_from_rowwise_data(
        num_tensors=num_groups,
        tensor_shape=(rows, cols),
        rowwise_data=output.view(-1),
        dtype=output.dtype,
        internal=False,
    )


class _FusedGroupedNVFP4QDQSTE(torch.autograd.Function):
    """Identity backward around grouped QDQ of a registered single grouped weight."""

    @staticmethod
    def forward(ctx: Any, weight: GroupedTensor, config: NVFP4QDQConfig) -> GroupedTensor:
        """Return an autograd-connected fake-quantized GroupedTensor over a fresh payload."""
        del ctx
        return fused_grouped_nvfp4_qdq_packed_weight(weight, config)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Propagate the packed weight gradient to the original grouped parameter unchanged."""
        del ctx
        return grad_output, None


def fake_grouped_nvfp4_quantization_ste(weight: GroupedTensor, config: NVFP4QDQConfig | None = None) -> GroupedTensor:
    """Apply grouped QDQ to a packed grouped weight with a straight-through estimator."""
    if config is None:
        config = current_nvfp4_qdq_config()
    return _FusedGroupedNVFP4QDQSTE.apply(weight, config)


__all__ = [
    "compute_grouped_nvfp4_amax",
    "fake_grouped_nvfp4_quantization_ste",
    "fused_grouped_nvfp4_qdq",
    "fused_grouped_nvfp4_qdq_packed_weight",
]
