"""Megatron-facing adapter for fused NVFP4 fake QAT."""

from __future__ import annotations

import os

import torch

NVFP4_FAKE_QAT_FLAG = "OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG"


def _is_packed_grouped_weight(weight_tensors: list[torch.Tensor]) -> bool:
    """True for TE's single-grouped-weight layout: one GroupedTensor packing every local expert."""
    if len(weight_tensors) != 1:
        return False
    from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

    return isinstance(weight_tensors[0], GroupedTensor)


def maybe_fake_quantize_nvfp4_weight_tensors(
    weight_tensors: list[torch.Tensor],
    *,
    fuse_wgrad_accumulation: bool = False,
    delay_wgrad_compute: bool = False,
) -> list[torch.Tensor]:
    """Apply env-gated fused NVFP4 fake QAT to TE grouped-linear weights.

    Discrete per-expert weights are fake-quantized one tensor at a time. A single
    packed grouped weight is fake-quantized by one grouped launch and returned as a
    one-element list holding an autograd-connected GroupedTensor.
    """
    if os.getenv(NVFP4_FAKE_QAT_FLAG, "0") != "1":
        return weight_tensors

    # Keep CuTe DSL optional for every process that does not enable this path.
    from miles.utils.fused_nvfp4_qdq import current_nvfp4_qdq_config, fake_nvfp4_quantization_ste

    qdq_config = current_nvfp4_qdq_config()
    if _is_packed_grouped_weight(weight_tensors):
        if fuse_wgrad_accumulation or delay_wgrad_compute:
            raise NotImplementedError(
                "Packed NVFP4 fake QAT requires gradient_accumulation_fusion=False and "
                "delay_wgrad_compute=False: TE writes fused/delayed weight gradients onto the "
                "weight object it receives, which would be the fake-quantized copy."
            )
        from miles.utils.fused_grouped_nvfp4_qdq import fake_grouped_nvfp4_quantization_ste

        return [fake_grouped_nvfp4_quantization_ste(weight_tensors[0], qdq_config)]
    return [fake_nvfp4_quantization_ste(weight, qdq_config) for weight in weight_tensors]


__all__ = ["NVFP4_FAKE_QAT_FLAG", "maybe_fake_quantize_nvfp4_weight_tensors"]
