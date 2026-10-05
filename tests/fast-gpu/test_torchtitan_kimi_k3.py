"""K3 checkpoint conversion, packed boundaries, and expert reduction precision."""

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci
from torchtitan.models.common.token_dispatcher import LocalDispatchMetadata

from miles.backends.torchtitan_utils.models.kimi_k3.model import KimiK3Model
from miles.backends.torchtitan_utils.models.kimi_k3.ops import KimiTokenDispatcher, packed_metadata
from miles.backends.torchtitan_utils.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter

register_cuda_ci(est_time=20, suite="stage-b-2-gpu-h200", labels=["torchtitan"], hardware=["hopper"])


def test_packed_documents_and_rows_do_not_attend_to_each_other():
    tokens = torch.zeros(2, 4, dtype=torch.long)
    positions = torch.tensor([[0, 1, 0, 1], [0, 1, 2, 3]])
    cumulative, mask = packed_metadata(tokens, positions)
    assert cumulative.tolist() == [0, 2, 4, 8]
    for query in range(8):
        start = 0 if query < 2 else 2 if query < 4 else 4
        assert mask[query].nonzero().flatten().tolist() == list(range(start, query + 1))


def test_explicit_boundaries_keep_padding_together_and_single_token_documents():
    tokens = torch.zeros(1, 8, dtype=torch.long)
    positions = torch.tensor([[0, 1, 2, 0, 0, 0, 0, 0]])
    boundaries = torch.tensor([0, 3, 4, 8], dtype=torch.int32)
    cumulative, mask = packed_metadata(tokens, positions, boundaries)
    assert cumulative.tolist() == [0, 3, 4, 8]
    assert mask[3].nonzero().flatten().tolist() == [3]
    assert mask[7].nonzero().flatten().tolist() == [4, 5, 6, 7]
    _, inferred_mask = packed_metadata(tokens, positions)
    torch.testing.assert_close(mask[:4], inferred_mask[:4])


def test_hf_roundtrip_preserves_padded_kda_and_grouped_experts():
    adapter = KimiK3StateDictAdapter(KimiK3Model.Config(num_experts=2), None)
    state = {
        "layers.0.self_attn.core.A_log": torch.arange(128, dtype=torch.float32),
        "layers.0.self_attn.core.dt_bias": torch.arange(8, dtype=torch.float32),
        "layers.1.moe.gate.e_score_correction_bias": torch.tensor([0.25, -0.25]),
        "tok_embeddings.weight": torch.randn(8, 4),
        "lm_head.weight": torch.randn(8, 4),
    }
    for name, shape in (("w1_EFD", (2, 3, 4)), ("w2_EDF", (2, 4, 3)), ("w3_EFD", (2, 3, 4))):
        state["layers.1.moe.routed_experts.inner_experts." + name] = torch.randn(shape)
    hf = adapter.to_hf(state)
    assert hf["language_model.model.layers.0.self_attn.A_log"].shape == (128,)
    actual = adapter.from_hf(hf)
    assert actual.keys() == state.keys()
    for name in state:
        torch.testing.assert_close(actual[name], state[name], rtol=0, atol=0)
    hf.pop("language_model.model.layers.1.block_sparse_moe.experts.1.w2.weight")
    with pytest.raises(ValueError, match="Incomplete K3 expert groups"):
        adapter.from_hf(hf)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA scatter reduction")
def test_expert_sum_rounds_once_and_preserves_router_and_expert_gradients():
    dispatcher = KimiTokenDispatcher.Config(num_experts=3, top_k=3).build()
    values = torch.tensor([[512, 1], [1, 1], [-512, 1]], device="cuda", dtype=torch.bfloat16, requires_grad=True)
    scores = torch.ones(3, device="cuda", requires_grad=True)
    metadata = LocalDispatchMetadata(
        token_indices_experts_sorted_N=torch.zeros(3, device="cuda", dtype=torch.long),
        topk_scores_experts_sorted_N=scores,
    )
    output = dispatcher.combine(values, metadata, values.new_zeros(1, 2))
    torch.testing.assert_close(
        output.float(), torch.tensor([[1, 3]], device="cuda"), rtol=0, atol=0, check_dtype=False
    )
    output.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values), rtol=0, atol=0)
    torch.testing.assert_close(scores.grad, torch.tensor([513.0, 2.0, -511.0], device="cuda"), rtol=0, atol=0)
