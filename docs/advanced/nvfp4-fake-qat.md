---
title: NVFP4 Fake Quantization-Aware Training
description: Fake-quantize routed MoE expert weights to NVFP4 in Megatron with one fused CuTe DSL kernel per grouped GEMM, including Transformer Engine's packed single grouped weight.
---

miles NVFP4 fake QAT keeps actor parameters in Megatron's training dtype (BF16
by default) and applies NVFP4 quantize-dequantize (QDQ) to routed MoE expert
weights immediately before each Transformer Engine (TE) `GroupedLinear` GEMM.
Backward uses a straight-through estimator (STE), so the optimizer updates the
original trainable weight. Parameters, optimizer state, gradients, and
activations are never stored in FP4.

The QDQ kernel mirrors TE's 1D, 1x16, per-tensor NVFP4 numerics and the full
Four Over Six matrix (MAE/MSE error metric, E4M3 maximum 256/448, exact or
FP16 candidate-error math). The E4M3 block scales and E2M1 values stay in
registers; only the BF16/FP16 result is written.

## Two weight layouts, one numerical contract

| Megatron layout | Parameters per `GroupedLinear` | QDQ launches per forward |
|---|---|---|
| Discrete expert weights (default) | `weight0 ... weight{G-1}` | `G` (one per expert) |
| Packed single grouped weight (`--moe-single-grouped-weight`) | one TE `GroupedTensor` named `weight` | 1 |

Expert `g` of the packed output is bit-identical to running the discrete kernel
on that expert with its own FP32 amax. Nothing is reduced across experts or
ranks.

## Enabling

Environment (set through `--extra-env-vars` or the launcher):

```json
{
  "OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG": "1",
  "OPEN_TRAINING_INT4_FAKE_QAT_FLAG": "0",
  "NVTE_USE_FAST_MATH": "0",
  "NVTE_NVFP4_4OVER6": "all",
  "NVTE_NVFP4_4OVER6_E4M3_USE_256": "none",
  "NVTE_NVFP4_4OVER6_ERR_MODE": "MSE",
  "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH": "1"
}
```

The four `NVTE_NVFP4_4OVER6*` variables select the W4A16 rollout recipe (Four
Over Six, MSE, E4M3 max 448, FP16 candidate-error math). Leave them unset for
standard NVFP4. `NVTE_USE_FAST_MATH=0` is required; ordinary quant fast math
is outside the kernel's contract.

Megatron flags for the packed layout (BF16 training):

```text
--moe-grouped-gemm --moe-use-grouped-tensor --moe-single-grouped-weight
--no-gradient-accumulation-fusion --disable-bias-linear
```

together with `NVTE_GROUPED_LINEAR_SINGLE_PARAM=1` in the environment. Leave
`--use-transformer-engine-op-fuser` and `--delay-wgrad-compute` off. The
discrete layout needs only the environment variables above.

## Minimal packed BF16 example

The adapter is reached through `TEGroupedLinear._get_weight_tensors` in the
miles Megatron fork; the same call works on a bare TE module:

```python
import os

os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
os.environ["OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG"] = "1"

import torch  # import torch before transformer_engine
from transformer_engine.pytorch import GroupedLinear

from miles.utils.nvfp4_fake_qat import maybe_fake_quantize_nvfp4_weight_tensors


class FakeQATGroupedLinear(GroupedLinear):
    def _get_weight_tensors(self):
        return maybe_fake_quantize_nvfp4_weight_tensors(
            super()._get_weight_tensors(),
            fuse_wgrad_accumulation=self.fuse_wgrad_accumulation,
            delay_wgrad_compute=False,
        )


experts = FakeQATGroupedLinear(
    num_gemms=8, in_features=6144, out_features=4096, bias=False,
    params_dtype=torch.bfloat16, device="cuda",
    single_grouped_weight=True, use_grouped_tensor=True,
)
tokens_per_expert = torch.tensor([512, 0, 256, 768, 256, 512, 256, 512], dtype=torch.int64, device="cuda")
x = torch.randn(int(tokens_per_expert.sum()), 6144, dtype=torch.bfloat16, device="cuda", requires_grad=True)
experts(x, tokens_per_expert).sum().backward()
assert experts.weight.grad.shape == (8, 4096, 6144)  # gradient lands on the packed parameter
```

`experts.weight` is the registered `GroupedTensor`; its `rowwise_data` payload
is read on every forward, so optimizer updates, checkpoint loads, and DDP
buffer binding are picked up without caching.

## Supported configurations

- SM10x GPUs (Blackwell); TE 2.19 or newer for the packed layout
  (`use_grouped_tensor` and `single_grouped_weight` in `GroupedLinear`), with
  cuBLASLt 13.3 or newer so TE's native grouped-GEMM gate
  `is_module_grouped_tensor_path_supported(None, torch.bfloat16)` is true.
- BF16 or FP16 parameters and GEMMs; FP8/FP4 compute and `--fp4-param` off.
- Expert weights with `K` divisible by 16, at most `2**31 - 1` elements per
  expert, and at most 65535 local experts; the packed payload may exceed
  `2**31` elements.
- Standard NVFP4 and every Four Over Six combination exposed by the
  `NVTE_NVFP4_4OVER6*` variables.
- `gradient_accumulation_fusion=False` and `delay_wgrad_compute=False` for the
  packed layout. Normal Megatron DDP accumulation into `main_grad` works.

## Limitations

- The packed layout rejects `--gradient-accumulation-fusion` and
  `--delay-wgrad-compute`: TE writes fused or delayed weight gradients onto
  the weight object it receives, which would be the fake-quantized copy.
- Rollout weight updates and the Hugging Face exporters expect per-expert
  `weight{N}` parameter names. The packed single grouped weight is a training
  side layout; wiring it into miles weight synchronization is separate work.
- No RHT, 2D scaling, stochastic rounding, transpose output, or bias.
- The QDQ output of each grouped GEMM is materialized once per forward
  (same as the discrete layout).
