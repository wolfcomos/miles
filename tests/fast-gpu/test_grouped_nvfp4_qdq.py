from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    suite="stage-c-8-gpu-b200",
    labels=["precision"],
    hardware=["blackwell"],
)


import os
import sys

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch.module.grouped_linear import GroupedLinear, is_module_grouped_tensor_path_supported
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

import miles.utils.fused_grouped_nvfp4_qdq as grouped_qdq_module
import miles.utils.nvfp4_fake_qat as nvfp4_qat
from miles.utils.fused_grouped_nvfp4_qdq import (
    compute_grouped_nvfp4_amax,
    fake_grouped_nvfp4_quantization_ste,
    fused_grouped_nvfp4_qdq,
    fused_grouped_nvfp4_qdq_packed_weight,
)
from miles.utils.fused_nvfp4_qdq import (
    NVFP4QDQConfig,
    NVFP4QDQErrorMode,
    compute_nvfp4_amax,
    current_nvfp4_qdq_config,
    fused_nvfp4_qdq,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="grouped fused NVFP4 QDQ requires SM10x",
)


@pytest.fixture(scope="module", autouse=True)
def _select_local_cuda_device() -> None:
    """Keep torchrun workers on their assigned GPUs without initializing collectives."""
    if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


# The feature matrix mirrors NVFP4_QDQ_CONFIGS in tests/fast-gpu/test_nvfp4_quantizer.py.
NVFP4_QDQ_CONFIGS = [pytest.param(NVFP4QDQConfig(), id="nvfp4")]
for _error_mode in (NVFP4QDQErrorMode.MAE, NVFP4QDQErrorMode.MSE):
    for _e4m3_max in (448, 256):
        for _error_use_fast_math in (False, True):
            NVFP4_QDQ_CONFIGS.append(
                pytest.param(
                    NVFP4QDQConfig(
                        use_4over6=True,
                        e4m3_max=_e4m3_max,
                        error_mode=_error_mode,
                        error_use_fast_math=_error_use_fast_math,
                    ),
                    id=(
                        f"4over6-{_error_mode.name.lower()}-e4m3-{_e4m3_max}-"
                        f"{'fp16-error' if _error_use_fast_math else 'exact-error'}"
                    ),
                )
            )

W4A16_CONFIG = NVFP4QDQConfig(
    use_4over6=True, e4m3_max=448, error_mode=NVFP4QDQErrorMode.MSE, error_use_fast_math=True
)
# Standard NVFP4 plus the W4A16 recipe: the two configurations trained in practice.
RECIPE_CONFIGS = [pytest.param(NVFP4QDQConfig(), id="nvfp4"), pytest.param(W4A16_CONFIG, id="w4a16")]

DTYPES = [pytest.param(torch.bfloat16, id="bf16"), pytest.param(torch.float16, id="fp16")]

# [G, N, K]: minimum-K and odd-N tails crossed with the full feature matrix.
NVFP4_GROUPED_MATRIX_SHAPES = [(1, 7, 16), (3, 33, 48), (8, 16, 1024)]
# Larger expert counts and the recipe-sized K values, swept with the two recipe configs.
NVFP4_GROUPED_SWEEP_SHAPES = [(1, 64, 16), (3, 7, 2048), (8, 33, 1024), (64, 7, 16), (64, 16, 48), (3, 128, 2048)]


def _shape_id(shape: tuple[int, int, int]) -> str:
    return "x".join(str(dim) for dim in shape)


def _make_grouped_input(shape: tuple[int, int, int], dtype: torch.dtype, init_data: str) -> torch.Tensor:
    """Pack G experts of the data patterns used by the single-tensor test, at G distinct magnitudes."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    num_groups, m, n = shape
    if init_data == "random":
        x = torch.randn(shape, dtype=torch.float32, device="cuda")
        if m > 1:
            x[:, 0].zero_()
    elif init_data == "boundary":
        base = torch.linspace(-12.0, 12.0, steps=n // 2, dtype=torch.float32, device="cuda")
        eps = torch.full_like(base, 1e-3)
        eps = torch.maximum(eps, torch.full_like(base, 1e-4))
        row = torch.empty(n, dtype=torch.float32, device="cuda")
        row[0::2] = base - eps
        row[1::2] = base + eps
        x = row.expand(num_groups, m, n).clone()
    elif init_data == "zeros":
        # Alternate signed zeros so the integer-view equality exercises the sign bit of zero blocks.
        return (
            torch.tensor([-0.0, 0.0], dtype=torch.float32, device="cuda").repeat(num_groups, m, n // 2).to(dtype=dtype)
        )
    elif init_data == "maxes":
        return torch.full(shape, torch.finfo(dtype).max, dtype=dtype, device="cuda")
    else:
        raise ValueError(f"Unknown init_data: {init_data}")
    # Distinct per-expert magnitudes make a shared amax visible; one expert is all zeros when G > 1.
    magnitudes = torch.tensor([10.0 ** ((g % 5) - 2) for g in range(num_groups)], dtype=torch.float32, device="cuda")
    x = x * magnitudes.view(num_groups, 1, 1)
    if num_groups > 1:
        x[num_groups // 2].zero_()
    return x.to(dtype=dtype)


def _per_expert_reference(x: torch.Tensor, amax: torch.Tensor, config: NVFP4QDQConfig) -> torch.Tensor:
    """Stack the existing single-tensor kernel over the experts of a [G, N, K] tensor."""
    return torch.stack([fused_nvfp4_qdq(x[g].contiguous(), amax[g], config) for g in range(x.shape[0])])


def _assert_bitwise(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Integer views distinguish signed zero; tolerance-zero floating comparison does not."""
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    actual_bits = actual.contiguous().view(torch.int16)
    expected_bits = expected.contiguous().view(torch.int16)
    assert torch.equal(
        actual_bits, expected_bits
    ), f"bit mismatch count: {torch.count_nonzero(actual_bits != expected_bits).item()}"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", NVFP4_GROUPED_MATRIX_SHAPES, ids=_shape_id)
@pytest.mark.parametrize("init_data", ["random", "boundary", "zeros", "maxes"])
@pytest.mark.parametrize("config", NVFP4_QDQ_CONFIGS)
@torch.inference_mode()
def test_fused_grouped_nvfp4_qdq_is_bit_exact_with_per_expert_kernel(
    dtype: torch.dtype, shape: tuple[int, int, int], init_data: str, config: NVFP4QDQConfig
) -> None:
    """Cover BF16/FP16 x data patterns x the full supported feature matrix against the single-tensor kernel."""
    x = _make_grouped_input(shape, dtype, init_data)
    amax = compute_grouped_nvfp4_amax(x)
    actual = fused_grouped_nvfp4_qdq(x, amax, config)
    _assert_bitwise(actual, _per_expert_reference(x, amax, config))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", NVFP4_GROUPED_SWEEP_SHAPES, ids=_shape_id)
@pytest.mark.parametrize("init_data", ["random", "boundary"])
@pytest.mark.parametrize("config", RECIPE_CONFIGS)
@torch.inference_mode()
def test_fused_grouped_nvfp4_qdq_shape_sweep_is_bit_exact_with_per_expert_kernel(
    dtype: torch.dtype, shape: tuple[int, int, int], init_data: str, config: NVFP4QDQConfig
) -> None:
    x = _make_grouped_input(shape, dtype, init_data)
    amax = compute_grouped_nvfp4_amax(x)
    actual = fused_grouped_nvfp4_qdq(x, amax, config)
    _assert_bitwise(actual, _per_expert_reference(x, amax, config))


def _make_te_qdq_quantizer(config: NVFP4QDQConfig):
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


def _te_qdq_reference(x: torch.Tensor, config: NVFP4QDQConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Native TE quantize-dequantize of one expert, the oracle of test_nvfp4_quantizer.py."""
    m, n = x.shape
    padded_m = ((m + 15) // 16) * 16
    if padded_m == m:
        x_padded = x.contiguous()
    else:
        padding = torch.zeros((padded_m - m, n), dtype=x.dtype, device=x.device)
        x_padded = torch.cat((x.contiguous(), padding), dim=0)

    quantized = _make_te_qdq_quantizer(config).quantize(x_padded)
    reference = quantized.dequantize(dtype=x.dtype)[:m, :n].contiguous()
    assert quantized._amax_rowwise is not None
    return reference, quantized._amax_rowwise.reshape(1)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(3, 33, 48), (8, 7, 64)], ids=_shape_id)
@pytest.mark.parametrize("init_data", ["random", "boundary", "zeros", "maxes"])
@pytest.mark.parametrize("config", NVFP4_QDQ_CONFIGS)
@torch.inference_mode()
def test_fused_grouped_nvfp4_qdq_is_bit_exact_with_te_per_expert(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    init_data: str,
    config: NVFP4QDQConfig,
) -> None:
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "0")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH", "1" if config.error_use_fast_math else "0")
    x = _make_grouped_input(shape, dtype, init_data)
    amax = compute_grouped_nvfp4_amax(x)
    actual = fused_grouped_nvfp4_qdq(x, amax, config)
    for g in range(shape[0]):
        expected, te_amax = _te_qdq_reference(x[g], config)
        assert torch.equal(amax[g].reshape(1).view(torch.int32), te_amax.view(torch.int32))
        _assert_bitwise(actual[g], expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("config", RECIPE_CONFIGS)
@torch.inference_mode()
def test_fused_grouped_nvfp4_qdq_preserves_signed_zero(dtype: torch.dtype, config: NVFP4QDQConfig) -> None:
    num_groups, m, n = 3, 4, 32
    x = torch.zeros((num_groups, m, n), dtype=dtype, device="cuda")
    x[:, :, 0] = -0.0
    # Tiny negatives beside a large value in the same block round to -0 after scaling.
    x[:, :, 1] = -1e-3
    x[:, :, 2] = -torch.finfo(dtype).tiny
    x[:, :, 3] = 64.0
    x[1] = -0.0
    negative = torch.signbit(x)
    assert negative.any() and (x[1] == 0).all()

    amax = compute_grouped_nvfp4_amax(x)
    actual = fused_grouped_nvfp4_qdq(x, amax, config)

    assert (actual[:, :, :3] == 0).all() and (actual[1] == 0).all()
    assert torch.equal(torch.signbit(actual), negative)
    _assert_bitwise(actual, _per_expert_reference(x, amax, config))
    for g in range(num_groups):
        _assert_bitwise(actual[g], _te_qdq_reference(x[g], config)[0])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("config", RECIPE_CONFIGS)
@torch.inference_mode()
def test_fused_grouped_nvfp4_qdq_uses_per_expert_amax_and_expert_order(
    dtype: torch.dtype, config: NVFP4QDQConfig
) -> None:
    torch.manual_seed(7)
    num_groups, m, n = 5, 24, 128
    magnitudes = torch.tensor([1e-3, 0.0, 1e3, 1.0, 30.0], device="cuda")
    x = (torch.randn((num_groups, m, n), device="cuda") * magnitudes.view(num_groups, 1, 1)).to(dtype=dtype)
    amax = compute_grouped_nvfp4_amax(x)
    assert amax[1] == 0 and amax[2] > amax[4] > amax[3] > amax[0] > 0

    actual = fused_grouped_nvfp4_qdq(x, amax, config)
    _assert_bitwise(actual, _per_expert_reference(x, amax, config))
    _assert_bitwise(actual[1], x[1])
    # Quantizing an expert against a neighbour's amax changes its bits, so equality above proves per-expert scaling.
    assert not torch.equal(actual[3], fused_nvfp4_qdq(x[3].contiguous(), amax[2], config))
    assert not torch.equal(actual[0], fused_nvfp4_qdq(x[0].contiguous(), amax[3], config))

    permutation = torch.tensor([2, 0, 4, 1, 3], device="cuda")
    permuted = fused_grouped_nvfp4_qdq(x[permutation].contiguous(), amax[permutation].contiguous(), config)
    _assert_bitwise(permuted, actual[permutation])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(1, 7, 16), (3, 33, 48), (64, 16, 48)], ids=_shape_id)
@pytest.mark.parametrize("init_data", ["random", "boundary", "zeros", "maxes"])
def test_compute_grouped_nvfp4_amax_matches_stacked_per_expert_amax(
    dtype: torch.dtype, shape: tuple[int, int, int], init_data: str
) -> None:
    x = _make_grouped_input(shape, dtype, init_data).requires_grad_(True)
    actual = compute_grouped_nvfp4_amax(x)
    expected = torch.stack([compute_nvfp4_amax(x[g]) for g in range(shape[0])])

    assert actual.dtype == torch.float32 and actual.device == x.device and actual.shape == (shape[0],)
    assert not actual.requires_grad
    assert torch.equal(actual, expected)


def _misaligned_grouped_input() -> torch.Tensor:
    storage = torch.randn(65, dtype=torch.bfloat16, device="cuda")
    x = storage[1:].view(2, 2, 16)
    assert x.is_contiguous() and x.data_ptr() % 16 != 0
    return x


def _oversized_expert_input() -> torch.Tensor:
    # One expert of 2**31 + 2**20 elements: uninitialized, so only the allocation costs.
    try:
        return torch.empty((1, 2**16, 2**15 + 16), dtype=torch.bfloat16, device="cuda")
    except torch.OutOfMemoryError:
        pytest.skip("the per-expert element limit needs a 4.3 GB allocation")


@pytest.mark.parametrize(
    ("make_input", "make_amax", "error_type", "match"),
    [
        pytest.param(
            lambda: torch.randn((4, 16), dtype=torch.bfloat16, device="cuda"), None, ValueError, "rank-3", id="rank-2"
        ),
        pytest.param(
            lambda: torch.randn((2, 16, 4), dtype=torch.bfloat16, device="cuda").transpose(1, 2),
            None,
            ValueError,
            "contiguous",
            id="non-contiguous",
        ),
        pytest.param(
            lambda: torch.randn((2, 4, 24), dtype=torch.bfloat16, device="cuda"),
            None,
            ValueError,
            "K divisible by 16",
            id="k-tail",
        ),
        pytest.param(
            lambda: torch.randn((2, 4, 16), dtype=torch.float32, device="cuda"),
            None,
            TypeError,
            "supports BF16 and FP16",
            id="fp32",
        ),
        pytest.param(lambda: torch.randn((2, 4, 16), dtype=torch.bfloat16), None, ValueError, "CUDA tensor", id="cpu"),
        pytest.param(_misaligned_grouped_input, None, ValueError, "16-byte-aligned", id="misaligned"),
        pytest.param(
            lambda: torch.empty((65536, 1, 16), dtype=torch.bfloat16, device="cuda"),
            None,
            ValueError,
            "at most 65535 groups",
            id="too-many-groups",
        ),
        pytest.param(_oversized_expert_input, None, ValueError, "elements per group", id="oversized-expert"),
        pytest.param(
            lambda: torch.empty((0, 4, 16), dtype=torch.bfloat16, device="cuda"),
            None,
            ValueError,
            "positive dimensions",
            id="zero-groups",
        ),
        pytest.param(
            lambda: torch.empty((2, 0, 16), dtype=torch.bfloat16, device="cuda"),
            None,
            ValueError,
            "positive dimensions",
            id="zero-rows",
        ),
        pytest.param(
            None, lambda x: torch.ones((3,), dtype=torch.float32, device="cuda"), TypeError, "shape", id="amax-shape"
        ),
        pytest.param(
            None, lambda x: torch.ones((2,), dtype=torch.bfloat16, device="cuda"), TypeError, "FP32", id="amax-dtype"
        ),
        pytest.param(None, lambda x: torch.ones((2,), dtype=torch.float32), ValueError, "CUDA device", id="amax-cpu"),
    ],
)
def test_fused_grouped_nvfp4_qdq_rejects_invalid_inputs(
    make_input, make_amax, error_type: type[Exception], match: str
) -> None:
    x = torch.randn((2, 4, 16), dtype=torch.bfloat16, device="cuda") if make_input is None else make_input()
    amax = torch.ones((x.shape[0],), dtype=torch.float32, device="cuda") if make_amax is None else make_amax(x)
    with pytest.raises(error_type, match=match):
        fused_grouped_nvfp4_qdq(x, amax, NVFP4QDQConfig())


def test_compute_grouped_nvfp4_amax_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="rank-3"):
        compute_grouped_nvfp4_amax(torch.randn((4, 16), dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(ValueError, match="empty"):
        compute_grouped_nvfp4_amax(torch.empty((2, 0, 16), dtype=torch.bfloat16, device="cuda"))


def _make_packed_weight(x: torch.Tensor) -> GroupedTensor:
    """Wrap a fresh copy of a [G, N, K] payload as a leaf TE grouped weight."""
    num_groups, rows, cols = x.shape
    weight = GroupedTensor.make_grouped_tensor_from_rowwise_data(
        num_tensors=num_groups,
        tensor_shape=(rows, cols),
        rowwise_data=x.detach().clone().view(-1),
        dtype=x.dtype,
        internal=False,
    )
    weight.requires_grad_(True)
    return weight


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("config", RECIPE_CONFIGS)
def test_fake_grouped_nvfp4_quantization_ste_is_identity_for_nonuniform_gradients(
    dtype: torch.dtype, config: NVFP4QDQConfig
) -> None:
    shape = (4, 33, 64)
    x = _make_grouped_input(shape, dtype, "random")
    weight = _make_packed_weight(x)
    expected = fused_grouped_nvfp4_qdq(x, compute_grouped_nvfp4_amax(x), config)

    output = fake_grouped_nvfp4_quantization_ste(weight, config)

    assert isinstance(output, GroupedTensor) and output.requires_grad and output.grad_fn is not None
    assert output.quantizer is None and output.num_tensors == shape[0] and tuple(output.shape) == shape
    assert output.rowwise_data.data_ptr() != weight.rowwise_data.data_ptr()
    _assert_bitwise(output.rowwise_data.view(shape), expected)
    assert not torch.equal(output.rowwise_data.view(shape), x)

    torch.manual_seed(1)
    grad_output = torch.randn(shape, dtype=dtype, device="cuda")
    output.backward(grad_output)

    assert weight.grad is not None and tuple(weight.grad.shape) == shape
    assert torch.equal(weight.grad, grad_output)
    _assert_bitwise(weight.rowwise_data.view(shape), x)


@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_grouped_nvfp4_qdq_packed_weight_rejects_non_grouped_inputs(dtype: torch.dtype) -> None:
    x = _make_grouped_input((2, 4, 16), dtype, "random")
    with pytest.raises(TypeError, match="requires a TE GroupedTensor"):
        fused_grouped_nvfp4_qdq_packed_weight(x)


def _set_recipe_env(monkeypatch: pytest.MonkeyPatch, config: NVFP4QDQConfig) -> None:
    """Feed the config through the TE environment contract that current_nvfp4_qdq_config resolves."""
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "0")
    if config.use_4over6:
        monkeypatch.setenv("NVTE_NVFP4_4OVER6", "all")
        monkeypatch.setenv("NVTE_NVFP4_4OVER6_E4M3_USE_256", "all" if config.e4m3_max == 256 else "none")
        monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_MODE", config.error_mode.name)
        monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH", "1" if config.error_use_fast_math else "0")
    else:
        for name in (
            "NVTE_NVFP4_4OVER6",
            "NVTE_NVFP4_4OVER6_E4M3_USE_256",
            "NVTE_NVFP4_4OVER6_ERR_MODE",
            "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH",
        ):
            monkeypatch.delenv(name, raising=False)
    assert current_nvfp4_qdq_config() == config


@pytest.mark.parametrize("config", RECIPE_CONFIGS)
def test_nvfp4_fake_qat_adapter_routes_packed_and_discrete_weights(
    monkeypatch: pytest.MonkeyPatch, config: NVFP4QDQConfig
) -> None:
    _set_recipe_env(monkeypatch, config)
    shape = (3, 8, 64)
    x = _make_grouped_input(shape, torch.bfloat16, "random")
    packed = [_make_packed_weight(x)]
    discrete = [torch.nn.Parameter(x[g].clone()) for g in range(shape[0])]
    expected = fused_grouped_nvfp4_qdq(x, compute_grouped_nvfp4_amax(x), config)

    monkeypatch.setenv(nvfp4_qat.NVFP4_FAKE_QAT_FLAG, "0")
    assert nvfp4_qat.maybe_fake_quantize_nvfp4_weight_tensors(packed) is packed
    assert nvfp4_qat.maybe_fake_quantize_nvfp4_weight_tensors(discrete) is discrete

    monkeypatch.setenv(nvfp4_qat.NVFP4_FAKE_QAT_FLAG, "1")
    actual_packed = nvfp4_qat.maybe_fake_quantize_nvfp4_weight_tensors(packed)
    actual_discrete = nvfp4_qat.maybe_fake_quantize_nvfp4_weight_tensors(discrete)

    assert (
        len(actual_packed) == 1
        and isinstance(actual_packed[0], GroupedTensor)
        and actual_packed[0].grad_fn is not None
    )
    _assert_bitwise(actual_packed[0].rowwise_data.view(shape), expected)
    assert len(actual_discrete) == shape[0]
    for g, weight in enumerate(actual_discrete):
        assert not isinstance(weight, GroupedTensor) and weight.grad_fn is not None
        _assert_bitwise(weight, expected[g])

    for kwargs in ({"fuse_wgrad_accumulation": True}, {"delay_wgrad_compute": True}):
        with pytest.raises(NotImplementedError, match="gradient_accumulation_fusion=False"):
            nvfp4_qat.maybe_fake_quantize_nvfp4_weight_tensors(packed, **kwargs)


class _FakeQATGroupedLinear(GroupedLinear):
    """Mirror of the Megatron TEGroupedLinear._get_weight_tensors hook."""

    def _get_weight_tensors(self):
        weight_tensors = super()._get_weight_tensors()
        return nvfp4_qat.maybe_fake_quantize_nvfp4_weight_tensors(
            weight_tensors, fuse_wgrad_accumulation=self.fuse_wgrad_accumulation
        )


def _grouped_linear_reference(
    x: torch.Tensor, m_splits: torch.Tensor, w_packed: torch.Tensor, config: NVFP4QDQConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-expert Miles fused QDQ then torch matmul (FP32 accumulation, output in the activation dtype)."""
    w_qdq = _per_expert_reference(w_packed, compute_grouped_nvfp4_amax(w_packed), config)
    x_splits = torch.split(x, m_splits.tolist())
    y = torch.cat([x_splits[g] @ w_qdq[g].t() for g in range(w_packed.shape[0])])
    return y, w_qdq


def _grouped_linear_backward_reference(
    x: torch.Tensor, dy: torch.Tensor, m_splits: torch.Tensor, w_qdq: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """dgrad per expert against the fake-quantized weights, wgrad per expert in FP32."""
    x_splits = torch.split(x, m_splits.tolist())
    dy_splits = torch.split(dy, m_splits.tolist())
    dgrad = torch.cat([dy_splits[g] @ w_qdq[g] for g in range(w_qdq.shape[0])])
    wgrad = torch.stack([dy_splits[g].float().t() @ x_splits[g].float() for g in range(w_qdq.shape[0])])
    return dgrad, wgrad


@pytest.mark.parametrize("config", RECIPE_CONFIGS)
def test_te_grouped_linear_single_grouped_weight_fake_qat_forward_backward_and_updates(
    monkeypatch: pytest.MonkeyPatch, config: NVFP4QDQConfig
) -> None:
    if not is_module_grouped_tensor_path_supported(None, torch.bfloat16):
        pytest.skip("TE native grouped GEMM path is unavailable on this device/cuBLASLt")
    _set_recipe_env(monkeypatch, config)
    monkeypatch.setenv(nvfp4_qat.NVFP4_FAKE_QAT_FLAG, "1")
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
    # GEMM outputs are compared with TE's own grouped-GEMM tolerances; every QDQ payload stays bit exact.
    gemm_tolerance = {"rtol": 1e-2, "atol": 5e-3}
    torch.manual_seed(0)
    num_groups, in_features, out_features = 4, 256, 512
    m_splits = torch.tensor([300, 0, 128, 84], dtype=torch.int64, device="cuda")
    tokens = int(m_splits.sum())
    module = _FakeQATGroupedLinear(
        num_groups,
        in_features,
        out_features,
        bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
        single_grouped_weight=True,
        use_grouped_tensor=True,
    )
    assert (
        isinstance(module.weight, GroupedTensor)
        and module.weight.quantizer is None
        and not module.fuse_wgrad_accumulation
    )
    w_shape = (num_groups, out_features, in_features)
    with torch.no_grad():
        w0 = torch.randn(w_shape, dtype=torch.bfloat16, device="cuda")
        w0 *= torch.tensor([10.0 ** (g - 1) for g in range(num_groups)], device="cuda").view(num_groups, 1, 1)
        module.weight.rowwise_data.view(w_shape).copy_(w0)
    payload_ptr = module.weight.rowwise_data.data_ptr()

    hooked = module._get_weight_tensors()
    assert len(hooked) == 1 and isinstance(hooked[0], GroupedTensor) and hooked[0].grad_fn is not None
    _assert_bitwise(
        hooked[0].rowwise_data.view(w_shape), _per_expert_reference(w0, compute_grouped_nvfp4_amax(w0), config)
    )
    _assert_bitwise(module.weight.rowwise_data.view(w_shape), w0)

    # Two microbatches accumulate into the ORIGINAL grouped parameter's .grad.
    wgrad_ref = torch.zeros(w_shape, dtype=torch.float32, device="cuda")
    for microbatch in range(2):
        x = torch.randn((tokens, in_features), dtype=torch.bfloat16, device="cuda", requires_grad=True)
        dy = torch.randn((tokens, out_features), dtype=torch.bfloat16, device="cuda") * (1 + microbatch)
        y = module(x, m_splits)
        y.backward(dy)
        y_ref, w_qdq = _grouped_linear_reference(x.detach(), m_splits, w0, config)
        dgrad_ref, wgrad_mb = _grouped_linear_backward_reference(x.detach(), dy, m_splits, w_qdq)
        wgrad_ref += wgrad_mb
        torch.testing.assert_close(y.float(), y_ref.float(), **gemm_tolerance)
        torch.testing.assert_close(x.grad.float(), dgrad_ref.float(), **gemm_tolerance)
    grad = module.weight.grad
    assert grad is not None and tuple(grad.shape) == w_shape and not isinstance(grad, GroupedTensor)
    # TE emits each microbatch's wgrad in BF16 and autograd accumulates in BF16: two roundings of values up to
    # |wgrad|max, hence the absolute term scales with the reference magnitude.
    torch.testing.assert_close(grad.float(), wgrad_ref, rtol=1e-2, atol=1e-2 * wgrad_ref.abs().max().item())
    assert torch.equal(grad[1], torch.zeros_like(grad[1]))
    assert not torch.equal(grad[0], torch.zeros_like(grad[0]))

    optimizer = torch.optim.SGD([module.weight], lr=1e-3)
    previous = w0
    for _step in range(3):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updated = module.weight.rowwise_data.view(w_shape).clone()
        assert module.weight.rowwise_data.data_ptr() == payload_ptr
        assert not torch.equal(updated, previous)
        x = torch.randn((tokens, in_features), dtype=torch.bfloat16, device="cuda", requires_grad=True)
        y = module(x, m_splits)
        y_ref, _ = _grouped_linear_reference(x.detach(), m_splits, updated, config)
        torch.testing.assert_close(y.float(), y_ref.float(), **gemm_tolerance)
        assert not torch.allclose(
            y.float(), _grouped_linear_reference(x.detach(), m_splits, previous, config)[0].float(), **gemm_tolerance
        )
        y.backward(torch.randn_like(y))
        assert module.weight.grad is not None and not torch.equal(
            module.weight.grad[0], torch.zeros_like(module.weight.grad[0])
        )
        previous = updated


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("config", RECIPE_CONFIGS)
def test_fused_grouped_nvfp4_qdq_replays_in_cuda_graph(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, config: NVFP4QDQConfig
) -> None:
    shape = (4, 64, 256)
    static_x = _make_grouped_input(shape, dtype, "random")
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        # Compilation happens here; the capture below must only record launches.
        for _ in range(2):
            fused_grouped_nvfp4_qdq(static_x, compute_grouped_nvfp4_amax(static_x), config)
    torch.cuda.current_stream().wait_stream(side_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side_stream):
        static_amax = compute_grouped_nvfp4_amax(static_x)
        static_y = fused_grouped_nvfp4_qdq(static_x, static_amax, config)

    for replay in range(3):
        torch.manual_seed(100 + replay)
        magnitudes = torch.tensor([10.0 ** ((g + replay) % 4 - 1) for g in range(shape[0])], device="cuda")
        x = (torch.randn(shape, device="cuda") * magnitudes.view(shape[0], 1, 1)).to(dtype=dtype)
        static_x.copy_(x)
        graph.replay()
        torch.cuda.synchronize()
        eager_amax = compute_grouped_nvfp4_amax(x)
        assert torch.equal(static_amax, eager_amax)
        _assert_bitwise(static_y, fused_grouped_nvfp4_qdq(x, eager_amax, config))

    # An uncompiled specialization must refuse to compile inside a capture instead of recording garbage.
    monkeypatch.setattr(grouped_qdq_module, "_GROUPED_KERNEL_CACHE", {})
    with pytest.raises(RuntimeError, match="Warm up"):
        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=side_stream):
            fused_grouped_nvfp4_qdq(static_x, compute_grouped_nvfp4_amax(static_x), config)


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
