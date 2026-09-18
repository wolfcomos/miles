"""torchrun worker for packed NVFP4 fake QAT through Megatron TEGroupedMLP (see the driver test)."""

import argparse
import gc
import json
import os
import statistics
import sys
import tempfile
import time
import traceback

os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
os.environ["OPEN_TRAINING_INT4_FAKE_QAT_FLAG"] = "0"
os.environ.setdefault("NVTE_USE_FAST_MATH", "0")

import torch  # noqa: E402  torch must be imported before transformer_engine
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor  # noqa: E402

from megatron.core import parallel_state  # noqa: E402
from megatron.core.enums import ModelType  # noqa: E402
from megatron.core.models.gpt.gpt_layer_specs import (  # noqa: E402
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.models.gpt.gpt_model import GPTModel  # noqa: E402
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer.module import Float16Module  # noqa: E402
from megatron.core.transformer.moe.experts import TEGroupedMLP  # noqa: E402
from megatron.core.transformer.moe.moe_layer import MoELayer  # noqa: E402
from megatron.core.transformer.moe.router import TopKRouter  # noqa: E402
from megatron.core.transformer.spec_utils import get_submodules  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args  # noqa: E402
from megatron.training.checkpointing import load_checkpoint, save_checkpoint  # noqa: E402
from megatron.training.global_vars import destroy_global_vars, get_args, set_args, set_global_variables  # noqa: E402
import megatron.training.training as megatron_training  # noqa: E402
from megatron.training.training import setup_model_and_optimizer  # noqa: E402

import miles.utils.fused_grouped_nvfp4_qdq as grouped_qdq_module  # noqa: E402
from miles.utils.fused_grouped_nvfp4_qdq import compute_grouped_nvfp4_amax, fused_grouped_nvfp4_qdq  # noqa: E402
from miles.utils.fused_nvfp4_qdq import compute_nvfp4_amax, current_nvfp4_qdq_config, fused_nvfp4_qdq  # noqa: E402
from miles.utils.nvfp4_fake_qat import NVFP4_FAKE_QAT_FLAG  # noqa: E402

_SEED = 1234
_FORCE_ROUTER = [True]  # read by _model_provider (setup_model_and_optimizer owns the provider call)
_W4A16_ENV = {
    "NVTE_NVFP4_4OVER6": "all",
    "NVTE_NVFP4_4OVER6_E4M3_USE_256": "none",
    "NVTE_NVFP4_4OVER6_ERR_MODE": "MSE",
    "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH": "1",
}
# GEMM outputs/grads: the packed grouped GEMM and the per-expert GEMMs accumulate in a different
# order and emit BF16, so these are compared with a tolerance; QDQ payloads are compared bitwise.
_GEMM_TOL = {"rtol": 1e-2, "atol": 5e-3}


def _log(msg):
    print(f"[rank {dist.get_rank()}] {msg}", flush=True)


def _payload(t):
    return t.rowwise_data if isinstance(t, GroupedTensor) else t


def _set_recipe(recipe):
    for key in _W4A16_ENV:
        os.environ.pop(key, None)
    if recipe == "w4a16":
        os.environ.update(_W4A16_ENV)


def _set_fake_qat(enabled):
    os.environ[NVFP4_FAKE_QAT_FLAG] = "1" if enabled else "0"


class _QdqCapture:
    """Records (input data_ptr, input payload, output payload) of every packed STE call."""

    def __init__(self):
        self.calls = []
        self._orig = grouped_qdq_module.fake_grouped_nvfp4_quantization_ste

    def install(self):
        def wrapped(weight, config=None):
            out = self._orig(weight, config)
            self.calls.append(
                (
                    weight.rowwise_data.data_ptr(),
                    weight.rowwise_data.detach().clone(),
                    out.rowwise_data.detach().clone(),
                )
            )
            return out

        grouped_qdq_module.fake_grouped_nvfp4_quantization_ste = wrapped
        return self

    def remove(self):
        grouped_qdq_module.fake_grouped_nvfp4_quantization_ste = self._orig


def _per_expert_qdq(payload, num_local, cfg):
    w = payload.view(num_local, -1, payload.shape[-1]) if payload.dim() == 2 else payload
    return torch.stack([fused_nvfp4_qdq(w[g].contiguous(), compute_nvfp4_amax(w[g]), cfg) for g in range(w.shape[0])])


def _assert_bitwise(actual, expected, what):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape, what
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), f"{what}: payload differs bitwise"


# ----------------------------------------------------------------------------------------------
# Direct TEGroupedMLP (Megatron MoE spec, no DDP): cases baseline / ep1_parity
# ----------------------------------------------------------------------------------------------
def _build_experts(packed, num_experts, hidden, moe_ffn, seed):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=4,
        num_moe_experts=num_experts,
        moe_ffn_hidden_size=moe_ffn,
        use_cpu_initialization=False,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        moe_router_dtype="fp32",
        moe_router_load_balancing_type="sinkhorn",
        moe_router_topk=1,
        moe_grouped_gemm=True,
        moe_use_grouped_tensor=packed,
        moe_single_grouped_weight=packed,
        use_transformer_engine_op_fuser=False,
        gradient_accumulation_fusion=False,
        delay_wgrad_compute=False,
    )
    torch.manual_seed(seed)
    submodules = get_submodules(
        get_gpt_layer_with_transformer_engine_submodules(num_experts, moe_grouped_gemm=True).mlp
    )
    layer = Float16Module(config, MoELayer(config, submodules)).module
    layer.cuda()
    assert isinstance(layer.experts, TEGroupedMLP)
    return layer.experts


def _expert_weights(linear, num_local):
    """Per-expert (G, out, in) view of a packed or discrete TEGroupedLinear weight (no copy for packed)."""
    if linear.single_grouped_weight:
        return linear.weight.rowwise_data.view(num_local, linear.out_features, linear.in_features)
    return torch.stack([getattr(linear, f"weight{g}").detach() for g in range(num_local)])


def _fill_expert_weights(linear, num_local, first_global_expert, tag):
    """Deterministic per-global-expert weights with spread magnitudes so per-expert amax matters."""
    with torch.no_grad():
        for g in range(num_local):
            e = first_global_expert + g
            gen = torch.Generator().manual_seed(7919 * e + tag)
            w = torch.randn(linear.out_features, linear.in_features, generator=gen) * 0.02 * 10.0 ** ((e % 3) - 1)
            w = w.to(device="cuda", dtype=torch.bfloat16)
            if linear.single_grouped_weight:
                _expert_weights(linear, num_local)[g].copy_(w)
            else:
                getattr(linear, f"weight{g}").copy_(w)


def _weight_grad(linear, num_local):
    if linear.single_grouped_weight:
        assert linear.weight.grad is not None, "packed weight received no gradient"
        return _payload(linear.weight.grad).view(num_local, linear.out_features, linear.in_features).float()
    return torch.stack([getattr(linear, f"weight{g}").grad for g in range(num_local)]).float()


def _record_m_splits(module, records):
    module.linear_fc1.register_forward_pre_hook(lambda _m, args: records.append(args[1]))


def case_baseline(opts):
    """Fake QAT OFF: one packed GroupedTensor per FC and native grouped BF16 fwd/bwd."""
    _set_fake_qat(False)
    G, H, FFN = 4, 256, 512
    mlp = _build_experts(True, G, H, FFN, _SEED)
    names = dict(mlp.named_parameters())
    assert set(names) == {"linear_fc1.weight", "linear_fc2.weight"}, sorted(names)
    for linear in (mlp.linear_fc1, mlp.linear_fc2):
        assert isinstance(linear.weight, GroupedTensor) and linear.weight.num_tensors == G
        assert linear.weight.rowwise_data.dtype == torch.bfloat16
    capture = _QdqCapture().install()
    records = []
    _record_m_splits(mlp, records)
    tokens_per_expert = torch.tensor([300, 0, 128, 84], dtype=torch.int64, device="cuda")
    x = torch.randn(512, H, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    probs = torch.rand(512, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    out, _ = mlp(x, tokens_per_expert, probs)
    out.backward(torch.randn_like(out))
    m_splits = records[0]
    assert isinstance(m_splits, torch.Tensor) and m_splits.is_cuda and m_splits.dtype == torch.int64, m_splits
    assert int(m_splits.sum()) >= 512 and len(m_splits) == G, m_splits  # 256-aligned padding on the grouped path
    _log(f"m_splits reaching TE: {m_splits.tolist()} (unpadded {tokens_per_expert.tolist()})")
    assert x.grad is not None and torch.isfinite(out).all()
    for linear in (mlp.linear_fc1, mlp.linear_fc2):
        assert _weight_grad(linear, G).abs().sum() > 0
    assert not capture.calls, "fake QAT ran while the flag was off"
    capture.remove()


def case_ep1_parity(opts):
    """Fake QAT ON: packed TEGroupedMLP == discrete TEGroupedMLP fed by per-expert fused_nvfp4_qdq."""
    _set_fake_qat(True)
    cfg = current_nvfp4_qdq_config()
    G, H, FFN = 4, 256, 512
    reference = _build_experts(False, G, H, FFN, _SEED)
    packed = _build_experts(True, G, H, FFN, _SEED + 1)
    for tag, attr in ((1, "linear_fc1"), (2, "linear_fc2")):
        _fill_expert_weights(getattr(reference, attr), G, 0, tag)
        _fill_expert_weights(getattr(packed, attr), G, 0, tag)
        assert torch.equal(_expert_weights(getattr(reference, attr), G), _expert_weights(getattr(packed, attr), G))
    tokens_per_expert = torch.tensor([300, 0, 128, 84], dtype=torch.int64, device="cuda")
    T = int(tokens_per_expert.sum())
    base_x = 0.5 * torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    base_probs = torch.rand(T, dtype=torch.bfloat16, device="cuda")
    grad_out = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")

    def run(mlp):
        x = base_x.clone().requires_grad_(True)
        probs = base_probs.clone().requires_grad_(True)
        out, _ = mlp(x, tokens_per_expert, probs)
        out.backward(grad_out)
        return out, x.grad, probs.grad

    capture = _QdqCapture().install()
    ref_out, ref_dx, ref_dp = run(reference)
    assert not capture.calls  # discrete weights take the per-tensor path
    out, dx, dp = run(packed)
    assert len(capture.calls) == 2, len(capture.calls)  # fc1 + fc2, one packed launch each
    for (ptr, w_in, w_out), linear in zip(capture.calls, (packed.linear_fc1, packed.linear_fc2)):
        assert ptr == linear.weight.rowwise_data.data_ptr()
        _assert_bitwise(
            w_out.view_as(_expert_weights(linear, G)), _per_expert_qdq(_expert_weights(linear, G), G, cfg), "QDQ"
        )
        assert torch.equal(w_in, linear.weight.rowwise_data), "STE modified the parameter payload"
    capture.remove()
    torch.testing.assert_close(out, ref_out, **_GEMM_TOL)
    torch.testing.assert_close(dx, ref_dx, **_GEMM_TOL)
    torch.testing.assert_close(dp, ref_dp, **_GEMM_TOL)
    for attr in ("linear_fc1", "linear_fc2"):
        ref_g = _weight_grad(getattr(reference, attr), G)
        g = _weight_grad(getattr(packed, attr), G)
        assert torch.equal(g[1], torch.zeros_like(g[1])), "zero-token expert has a nonzero wgrad"
        torch.testing.assert_close(g, ref_g, rtol=1e-2, atol=1e-2 * ref_g.abs().max().item())
    _log(f"parity OK (config {cfg})")


# ----------------------------------------------------------------------------------------------
# GPTModel + Megatron DDP + DistributedOptimizer: cases accumulation / ep2 / edp2 / checkpoint / step_time
# ----------------------------------------------------------------------------------------------
def _cleanup():
    parallel_state.destroy_model_parallel()
    destroy_global_vars()
    destroy_num_microbatches_calculator()
    gc.collect()
    torch.cuda.empty_cache()


def _make_args(*, packed, ep, num_experts, hidden, moe_ffn, seq, mbs, ckpt_dir=None):
    sys.argv = ["_grouped_nvfp4_qdq_megatron_worker.py"]
    args = parse_args()
    args.num_layers = 1
    args.vocab_size = 1024
    args.hidden_size = hidden
    args.ffn_hidden_size = moe_ffn
    args.num_attention_heads = 8
    args.max_position_embeddings = seq
    args.seq_length = seq
    args.micro_batch_size = mbs
    args.global_batch_size = mbs * dist.get_world_size()
    args.create_attention_mask_in_dataloader = True
    args.tensor_model_parallel_size = 1
    args.pipeline_model_parallel_size = 1
    args.context_parallel_size = 1
    args.expert_model_parallel_size = ep
    args.train_iters = 100
    args.lr = 1e-3
    args.bf16 = True
    args.attention_backend = "unfused"
    args.add_bias_linear = False
    args.hidden_dropout = 0.0
    args.attention_dropout = 0.0
    args.swiglu = True
    args.gradient_accumulation_fusion = False
    args.use_distributed_optimizer = True
    args.use_transformer_engine_op_fuser = False
    args.overlap_param_gather = False
    args.overlap_grad_reduce = False
    args.accumulate_allreduce_grads_in_fp32 = False
    args.ddp_bucket_size = 40960
    args.num_experts = num_experts
    args.moe_layer_freq = 1
    args.moe_grouped_gemm = True
    args.moe_use_grouped_tensor = packed
    args.moe_single_grouped_weight = packed
    args.moe_token_dispatcher_type = "alltoall"
    args.moe_router_topk = 1
    args.moe_router_pre_softmax = True
    args.moe_router_load_balancing_type = "none"
    args.moe_router_dtype = "fp32"
    args.moe_aux_loss_coeff = 0.0
    args.moe_ffn_hidden_size = moe_ffn
    args.moe_mlp_glu_interleave_size = 32
    if ckpt_dir is not None:
        args.save = ckpt_dir
        args.load = ckpt_dir
        args.save_interval = 1000
        args.no_save_optim = True
        args.no_load_optim = True
        args.no_save_rng = True
        args.no_load_rng = True
        args.load_main_params_from_ckpt = True
    validate_args(args)
    set_global_variables(args, False)
    set_args(args)
    return args


def _model_provider(pre_process=True, post_process=True, config=None, pg_collection=None, vp_stage=None):
    model_parallel_cuda_manual_seed(_SEED)
    args = get_args()
    if config is None:
        config = core_transformer_config_from_args(args)
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=args.num_experts, moe_grouped_gemm=args.moe_grouped_gemm
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=args.vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=True,
        share_embeddings_and_output_weights=True,
        position_embedding_type=args.position_embedding_type,
        pg_collection=pg_collection,
        vp_stage=vp_stage,
    )
    # Fill before DDP/DistributedOptimizer wrap: the optimizer snapshots main params at construction.
    _fill_model(model, args.num_experts, args.hidden_size, _FORCE_ROUTER[0])
    return model


def _fill_model(model, num_experts, hidden, force_router):
    """Deterministic per-global-expert weights; optionally force the last expert to get no tokens."""
    experts = [m for m in model.modules() if isinstance(m, TEGroupedMLP)]
    assert len(experts) == 1
    num_local = num_experts // parallel_state.get_expert_model_parallel_world_size()
    first = parallel_state.get_expert_model_parallel_rank() * num_local
    for tag, attr in ((1, "linear_fc1"), (2, "linear_fc2")):
        _fill_expert_weights(getattr(experts[0], attr), num_local, first, tag)
    if force_router:
        # Router logits <h, W[e]>: W[0] = v, W[1] = -v guarantee max_e > 0 = <h, W[E-1]>, so the
        # last expert never receives a token (zero-token expert on its owner rank).
        router = [m for m in model.modules() if isinstance(m, TopKRouter)][0]
        gen = torch.Generator().manual_seed(11)
        w = torch.randn(num_experts, hidden, generator=gen)
        w[1] = -w[0]
        w[-1] = 0.0
        with torch.no_grad():
            router.weight.copy_(w.to(router.weight.dtype))


def _build_gpt(
    *, packed, ep, num_experts=4, hidden=256, moe_ffn=256, seq=128, mbs=2, ckpt_dir=None, force_router=True
):
    """Fresh (model, optimizer, scheduler, experts) with deterministic expert weights and a forced router."""
    _cleanup()
    # The miles-megatron checkout carries megatron/post_training as an empty submodule; ModelOpt
    # checkpoint detection is irrelevant to these runs.
    megatron_training.has_nvidia_modelopt = False
    _make_args(
        packed=packed,
        ep=ep,
        num_experts=num_experts,
        hidden=hidden,
        moe_ffn=moe_ffn,
        seq=seq,
        mbs=mbs,
        ckpt_dir=ckpt_dir,
    )
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=ep)
    torch.manual_seed(_SEED)
    _FORCE_ROUTER[0] = force_router
    model, optimizer, scheduler = setup_model_and_optimizer(
        model_type=ModelType.encoder_or_decoder, model_provider_func=_model_provider
    )
    experts = [m for m in model[0].modules() if isinstance(m, TEGroupedMLP)]
    return model, optimizer, scheduler, experts[0], num_experts // ep


def _batch(seq, mbs, shift=0):
    args = get_args()
    data = (torch.arange(seq, dtype=torch.int64, device="cuda") + shift) % args.vocab_size
    input_ids = data.repeat((mbs, 1))
    labels = ((data + 1) % args.vocab_size).repeat((mbs, 1))
    position_ids = torch.arange(seq, dtype=torch.int64, device="cuda").repeat((mbs, 1))
    attention_mask = torch.ones((mbs, 1, seq, seq), dtype=torch.bool, device="cuda")
    loss_mask = torch.ones((mbs, seq), dtype=torch.float32, device="cuda")
    return input_ids, labels, position_ids, attention_mask, loss_mask


def _forward_backward(model, batches):
    """zero_grad + microbatch fwd/bwd + grad sync (no optimizer step); returns per-token losses."""
    model[0].zero_grad_buffer()
    outs = []
    for i, b in enumerate(batches):
        if i == 0:
            model[0].set_is_first_microbatch()
        out = model[0].forward(input_ids=b[0], labels=b[1], position_ids=b[2], attention_mask=b[3], loss_mask=b[4])
        assert torch.isfinite(out).all()
        out.mean().backward()
        outs.append(out.detach().float())
    model[0].finish_grad_sync()
    return torch.stack(outs)


def _main_grad(linear, num_local):
    return _payload(linear.weight.main_grad).view(num_local, linear.out_features, linear.in_features).float().clone()


def _owned_range(optimizer, param):
    """Elements of `param` whose reduce-scattered gradient this rank owns (DistributedOptimizer shard)."""
    for opt in getattr(optimizer, "chained_optimizers", [optimizer]):
        if param in opt.model_param_gbuf_map:
            r = opt._get_model_param_range_map(param)["param"]
            return r.start, r.end
    raise KeyError("parameter is not managed by a DistributedOptimizer")


def _assert_owned_shard_close(optimizer, linear, expected):
    """main_grad == expected on the owned shard (the rest of the buffer is not reduced on this rank)."""
    start, end = _owned_range(optimizer, linear.weight)
    assert end > start, "rank owns no elements of the packed weight"
    got = _payload(linear.weight.main_grad).reshape(-1)[start:end].float()
    ref = expected.reshape(-1)[start:end].float()
    assert got.abs().sum() > 0
    torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2 * ref.abs().max().item())
    return end - start


def _step(model, optimizer, batches):
    optimizer.zero_grad()
    outs = _forward_backward(model, batches)
    ok, _, _ = optimizer.step()
    assert ok
    return outs


def _hook_microbatch_grads(linear, sink):
    linear.weight.register_hook(lambda g: sink.append(_payload(g).detach().clone().float()))


def _pre_qdq(linear, num_local):
    """Expected fake-quantized payload of the CURRENT parameter payload, per-expert kernel oracle."""
    return _per_expert_qdq(_expert_weights(linear, num_local), num_local, current_nvfp4_qdq_config())


def case_accumulation(opts):
    """World 1: DDP main_grad == sum of microbatch grads; weights move; next QDQ uses the updated payload."""
    _set_fake_qat(True)
    model, optimizer, _, mlp, G = _build_gpt(packed=True, ep=1)
    fc1 = mlp.linear_fc1
    batches = [_batch(128, 2, shift=0), _batch(128, 2, shift=37)]
    capture = _QdqCapture().install()
    grads = []
    fc1.weight.register_hook(lambda g: grads.append(_payload(g).detach().clone()))
    for step in range(3):
        grads.clear()
        before = fc1.weight.rowwise_data.clone()
        capture.calls.clear()
        optimizer.zero_grad()
        _forward_backward(model, batches)
        assert len(grads) == 2, len(grads)
        main_grad = _payload(fc1.weight.main_grad)
        assert main_grad.abs().sum() > 0
        # same dtype (fp32 main grads) and order as the DDP backward hook's main_grad.add_
        manual = grads[0].to(main_grad.dtype).add_(grads[1])
        assert torch.equal(
            main_grad, manual
        ), f"main_grad != sum of microbatch grads, max diff {(main_grad - manual).abs().max()}"
        # every microbatch QDQ (fc1 + fc2 per microbatch) read the live payload and equals the grouped kernel on it
        assert len(capture.calls) == 4, len(capture.calls)
        for ptr, w_in, w_out in capture.calls[0::2]:
            assert ptr == fc1.weight.rowwise_data.data_ptr() and torch.equal(w_in, before)
            grouped = fused_grouped_nvfp4_qdq(
                w_in.view(G, fc1.out_features, fc1.in_features),
                compute_grouped_nvfp4_amax(w_in.view(G, -1, fc1.in_features)),
            )
            _assert_bitwise(w_out.view_as(grouped), grouped, "fc1 QDQ vs fused_grouped_nvfp4_qdq(payload)")
            _assert_bitwise(w_out.view_as(grouped), _pre_qdq(fc1, G), "fc1 QDQ vs per-expert oracle")
        ok, _, _ = optimizer.step()
        assert ok
        after = fc1.weight.rowwise_data
        assert not torch.equal(after, before), f"step {step}: packed weight did not change"
        _log(
            f"step {step}: |main_grad|={main_grad.float().abs().mean():.3e} |dW|={(after.float() - before.float()).abs().mean():.3e}"
        )
    capture.remove()


def _local_experts_of(mlp, num_local):
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    return list(range(ep_rank * num_local, (ep_rank + 1) * num_local))


def case_ep2(opts):
    """World 2: EP=2 packed run matches the EP=1 run with identical weights/tokens; zero-token expert on rank 1."""
    _set_fake_qat(True)
    E = 4
    model, optimizer, _, mlp, _ = _build_gpt(packed=True, ep=1, num_experts=E)
    batches = [_batch(128, 2, shift=0)]
    # Identical batches on both DP ranks: the pre-reduce local grad IS the single-rank reference.
    ref_grads = {attr: [] for attr in ("linear_fc1", "linear_fc2")}
    for attr in ref_grads:
        _hook_microbatch_grads(getattr(mlp, attr), ref_grads[attr])
    optimizer.zero_grad()
    ref_out = _forward_backward(model, batches)
    ref_grads = {attr: g[0].view(E, getattr(mlp, attr).out_features, -1) for attr, g in ref_grads.items()}
    ref_weights = {attr: _expert_weights(getattr(mlp, attr), E).clone() for attr in ("linear_fc1", "linear_fc2")}

    model, optimizer, _, mlp, G = _build_gpt(packed=True, ep=2, num_experts=E)
    assert G == 2 and parallel_state.get_expert_model_parallel_world_size() == 2
    local = _local_experts_of(mlp, G)
    for attr in ("linear_fc1", "linear_fc2"):
        linear = getattr(mlp, attr)
        assert linear.weight.num_tensors == G
        assert linear.weight.rowwise_data.numel() == G * linear.out_features * linear.in_features
        assert torch.equal(_expert_weights(linear, G), ref_weights[attr][local])
    unpadded = []
    mlp.register_forward_pre_hook(lambda _m, args: unpadded.append(args[1].tolist()))
    optimizer.zero_grad()
    out = _forward_backward(model, batches)
    torch.testing.assert_close(out, ref_out, **_GEMM_TOL)
    # Each expert now sees both ranks' (identical) tokens and DDP scales expert grads by 1/DP=1/2,
    # so the reduced main_grad equals the single-rank reference; only the owned shard is reduced here.
    for attr in ("linear_fc1", "linear_fc2"):
        n = _assert_owned_shard_close(optimizer, getattr(mlp, attr), ref_grads[attr][local])
        _log(f"{attr}: {n} owned grad elements match the EP1 reference")
    counts = torch.tensor(unpadded[0], device="cuda")
    gathered = [torch.zeros_like(counts) for _ in range(2)]
    dist.all_gather(gathered, counts)
    _log(f"tokens_per_expert per rank: {[t.tolist() for t in gathered]}")
    assert gathered[1][-1].item() == 0, "expected a zero-token expert on rank 1"
    assert all(t.sum().item() > 0 for t in gathered)


def case_edp2(opts):
    """World 4 (EP2 x EDP2): reduced main_grad == scaled sum over EDP ranks; payload identical after step."""
    _set_fake_qat(True)
    model, optimizer, _, mlp, G = _build_gpt(packed=True, ep=2, num_experts=4)
    edp_group = parallel_state.get_expert_data_parallel_group()
    edp = parallel_state.get_expert_data_parallel_world_size()
    dp = parallel_state.get_data_parallel_world_size(with_context_parallel=True)
    assert edp == 2 and dp == 4, (edp, dp)
    fc1 = mlp.linear_fc1
    rank = dist.get_rank()
    batches = [_batch(128, 2, shift=5 * rank), _batch(128, 2, shift=5 * rank + 101)]
    grads = []
    handle = fc1.weight.register_hook(lambda g: grads.append(_payload(g).detach().clone().float()))
    optimizer.zero_grad()
    _forward_backward(model, batches)
    handle.remove()
    assert len(grads) == 2
    local_sum = grads[0] + grads[1]
    gathered = [torch.zeros_like(local_sum) for _ in range(edp)]
    dist.all_gather(gathered, local_sum, group=edp_group)
    # Megatron DDP (average_in_collective=False): expert grads are summed over the expert-DP group
    # and scaled by 1 / data_parallel_world_size.
    expected = torch.stack(gathered).sum(0) / dp
    n = _assert_owned_shard_close(optimizer, fc1, expected)
    assert not torch.allclose(gathered[0], gathered[1]), "EDP ranks saw identical grads; batches must differ"
    ok, _, _ = optimizer.step()
    assert ok
    payload = fc1.weight.rowwise_data.clone()
    payloads = [torch.zeros_like(payload) for _ in range(edp)]
    dist.all_gather(payloads, payload, group=edp_group)
    assert all(torch.equal(p, payloads[0]) for p in payloads), "packed payload differs across EDP ranks after step"
    capture = _QdqCapture().install()
    optimizer.zero_grad()
    _forward_backward(model, batches[:1])
    ptr, w_in, w_out = capture.calls[0]
    assert ptr == fc1.weight.rowwise_data.data_ptr(), "forward reads a payload other than weight.rowwise_data"
    assert torch.equal(w_in, fc1.weight.rowwise_data) and torch.equal(w_in, payload)
    _assert_bitwise(w_out.view(G, fc1.out_features, fc1.in_features), _pre_qdq(fc1, G), "post-step QDQ")
    capture.remove()
    _log(f"edp2 OK: {n} owned grad elements of {expected.numel()} verified")


def case_checkpoint(opts):
    """World 1: torch_dist save -> fresh model load; next fwd/bwd equals continuing; keys unchanged by QAT."""
    _set_fake_qat(True)
    ckpt_dir = opts.ckpt_dir or tempfile.mkdtemp(prefix="grouped_qdq_ckpt_")
    model, optimizer, scheduler, mlp, G = _build_gpt(packed=True, ep=1, ckpt_dir=ckpt_dir)
    fc1 = mlp.linear_fc1
    batches = [_batch(128, 2, shift=0), _batch(128, 2, shift=37)]
    for _ in range(2):
        _step(model, optimizer, batches)
    saved_payload = fc1.weight.rowwise_data.clone()
    keys_qat = sorted(model[0].sharded_state_dict().keys())
    save_checkpoint(2, model, optimizer, scheduler, 0)
    capture = _QdqCapture().install()
    optimizer.zero_grad()
    cont_out = _forward_backward(model, batches)
    cont_grad = _main_grad(fc1, G)
    cont_qdq = capture.calls[0][2]
    capture.remove()

    model, optimizer, scheduler, mlp, G = _build_gpt(packed=True, ep=1, ckpt_dir=ckpt_dir)
    fc1 = mlp.linear_fc1
    assert not torch.equal(fc1.weight.rowwise_data, saved_payload), "fresh model already equals the checkpoint"
    iteration, _ = load_checkpoint(model, optimizer, scheduler, strict=True)
    assert iteration == 2
    assert torch.equal(fc1.weight.rowwise_data, saved_payload), "reloaded packed payload differs bitwise"
    capture = _QdqCapture().install()
    optimizer.zero_grad()
    out = _forward_backward(model, batches)
    capture.remove()
    _assert_bitwise(capture.calls[0][2], cont_qdq, "QDQ after reload")
    torch.testing.assert_close(out, cont_out, **_GEMM_TOL)
    grad = _main_grad(fc1, G)
    torch.testing.assert_close(grad, cont_grad, rtol=1e-2, atol=1e-2 * cont_grad.abs().max().item())
    _log(
        f"reload: outputs bitwise equal={torch.equal(out, cont_out)}, grads bitwise equal={torch.equal(grad, cont_grad)}"
    )

    _set_fake_qat(False)
    model, *_ = _build_gpt(packed=True, ep=1, ckpt_dir=ckpt_dir)
    assert sorted(model[0].sharded_state_dict().keys()) == keys_qat, "fake QAT changed checkpoint keys"


def _time_steps(model, optimizer, batches, warmup, steps):
    for _ in range(warmup):
        _step(model, optimizer, batches)
    times = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _step(model, optimizer, batches)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def _kernel_counts(model, optimizer, batches):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        _step(model, optimizer, batches)
    counts = {}
    for evt in prof.events():
        if evt.device_type == torch.autograd.DeviceType.CUDA:
            counts[evt.name] = counts.get(evt.name, 0) + 1
    return counts


def case_step_time(opts):
    """World 1 small MoE (hidden 1024, ffn 512, 32 experts, 2048 tokens): step time for four arms."""
    arms = {
        "no_fake_qat_discrete": dict(packed=False, qat=False),
        "no_fake_qat_packed": dict(packed=True, qat=False),
        "fake_qat_discrete": dict(packed=False, qat=True),
        "fake_qat_packed": dict(packed=True, qat=True),
    }
    result = {
        "config": {
            "hidden": 1024,
            "moe_ffn": 512,
            "num_experts": 32,
            "tokens": 2048,
            "topk": 1,
            "steps": 20,
            "warmup": 5,
        }
    }
    for name, arm in arms.items():
        _set_fake_qat(arm["qat"])
        model, optimizer, _, _, _ = _build_gpt(
            packed=arm["packed"], ep=1, num_experts=32, hidden=1024, moe_ffn=512, seq=1024, mbs=2, force_router=False
        )
        batches = [_batch(1024, 2, shift=0)]
        times = _time_steps(model, optimizer, batches, warmup=5, steps=20)
        kernels = _kernel_counts(model, optimizer, batches)
        qdq = {k: v for k, v in kernels.items() if "nvfp4_qdq" in k or "AbsMaxOps" in k}
        result[name] = {
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "qdq_kernel_launches": sum(qdq.values()),
            "qdq_kernels": {k.split("(")[0][:80]: v for k, v in qdq.items()},
        }
        _log(
            f"{name}: median {result[name]['median_ms']:.3f} ms, {result[name]['qdq_kernel_launches']} QDQ/amax kernel launches per step"
        )
    for name in ("fake_qat_discrete", "fake_qat_packed"):
        base = result["no_fake_qat_" + name.split("_")[-1]]
        result[name]["ratio_vs_same_layout_no_fake_qat"] = result[name]["median_ms"] / base["median_ms"]
    result["fake_qat_packed"]["ratio_vs_fake_qat_discrete"] = (
        result["fake_qat_packed"]["median_ms"] / result["fake_qat_discrete"]["median_ms"]
    )
    result["note"] = "small single-GPU model, launch-bound; not a training speedup claim"
    if opts.output:
        with open(opts.output, "w") as f:
            json.dump(result, f, indent=2)
    _log(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=1))


_CASES = {
    "baseline": case_baseline,
    "ep1_parity": case_ep1_parity,
    "accumulation": case_accumulation,
    "ep2": case_ep2,
    "edp2": case_edp2,
    "checkpoint": case_checkpoint,
    "step_time": case_step_time,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=sorted(_CASES))
    parser.add_argument("--recipe", default="standard", choices=["standard", "w4a16"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--ckpt-dir", default=None)
    opts = parser.parse_args()
    _set_recipe(opts.recipe)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    try:
        _CASES[opts.case](opts)
    except BaseException:
        # Report immediately and exit hard: waiting in destroy_process_group would deadlock peers.
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    dist.barrier()
    _log(f"PASS {opts.case} {opts.recipe}")
    _cleanup()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
