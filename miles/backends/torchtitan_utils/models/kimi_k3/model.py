"""Kimi K3 text model for four-layer training validation."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, Partial
from torchtitan.models.common import Embedding
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module

from miles.backends.torchtitan_utils.models.kimi_k3.attention import DeltaAttention, MLAAttention
from miles.backends.torchtitan_utils.models.kimi_k3.ops import (
    DenseMLP,
    KimiTokenDispatcher,
    RMSNorm,
    SituExperts,
    attention_residual,
    linear,
    packed_metadata,
)


class LayerDict(nn.ModuleDict, Module):
    pass


class LatentMoE(Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts, self.top_k = config.num_experts, config.top_k
        self.gate = linear(config.dim, config.num_experts)
        self.gate.register_buffer("e_score_correction_bias", torch.zeros(config.num_experts, dtype=torch.float32))
        self.routed_expert_down_proj = linear(config.dim, config.latent_dim)
        self.routed_expert_up_proj = linear(config.latent_dim, config.dim)
        self.routed_expert_norm = RMSNorm(config.latent_dim)
        self.shared_experts = DenseMLP(config.dim, config.moe_intermediate * 2)
        self.routed_experts = RoutedExperts.Config(
            inner_experts=SituExperts.Config(
                dim=config.latent_dim, hidden_dim=config.moe_intermediate, num_experts=config.num_experts
            ),
            token_dispatcher=KimiTokenDispatcher.Config(num_experts=config.num_experts, top_k=config.top_k),
        ).build()

    def _init_self_buffers(self, *, buffer_device=None):
        self.gate.e_score_correction_bias.zero_()

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = F.linear(x.float(), self.gate.weight.float()).sigmoid()
        choice = scores + self.gate.e_score_correction_bias.float()
        indices = choice.topk(self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(-1, indices)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        local_indices = indices.to_local() if isinstance(indices, DTensor) else indices
        counts = torch.bincount(local_indices.flatten(), minlength=self.num_experts)
        if isinstance(indices, DTensor):
            counts = DTensor.from_local(counts, indices.device_mesh, [Partial()], run_check=False)
        latent = self.routed_expert_down_proj(x)
        routed = self.routed_experts(latent, weights, indices, counts)
        return self.routed_expert_up_proj(self.routed_expert_norm(routed)) + self.shared_experts(x)


class KimiBlock(Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx, self.block_size = layer_idx, config.attn_res_block_size
        self.self_attn = DeltaAttention(config) if (layer_idx + 1) % 4 else MLAAttention(config)
        self.input_layernorm = RMSNorm(config.dim)
        self.post_attention_layernorm = RMSNorm(config.dim)
        self.self_attention_res_norm, self.mlp_res_norm = RMSNorm(config.dim), RMSNorm(config.dim)
        self.self_attention_res_proj, self.mlp_res_proj = linear(config.dim, 1), linear(config.dim, 1)
        self.moe_enabled = layer_idx >= config.dense_layers
        if self.moe_enabled:
            self.moe = LatentMoE(config)
        else:
            self.mlp = DenseMLP(config.dim, config.dense_intermediate)

    def forward(self, x, bank, cu_seqlens, mask):
        prefix = x
        if bank.shape[-2]:
            x = attention_residual(prefix, bank, self.self_attention_res_norm, self.self_attention_res_proj)
        if self.layer_idx % self.block_size == 0:
            bank = torch.cat((bank, prefix.unsqueeze(-2)), dim=-2)
            prefix = None
        x = self.self_attn(self.input_layernorm(x), cu_seqlens, mask)
        prefix = x if prefix is None else prefix + x
        x = attention_residual(prefix, bank, self.mlp_res_norm, self.mlp_res_proj)
        x = self.post_attention_layernorm(x)
        return prefix + (self.moe(x) if self.moe_enabled else self.mlp(x)), bank


class KimiK3Model(BaseModel):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        dim: int = 7168
        vocab_size: int = 163840
        n_layers: int = 4
        n_heads: int = 96
        head_dim: int = 128
        num_experts: int = 64
        top_k: int = 16
        latent_dim: int = 3584
        dense_layers: int = 1
        dense_intermediate: int = 33792
        moe_intermediate: int = 3072
        attn_res_block_size: int = 12
        enable_weight_tying: bool = False

        def update_from_config(self, *, config, **kwargs):
            p = config.parallelism
            if any(
                value != 1
                for value in (
                    p.context_parallel_degree,
                    p.pipeline_parallel_degree,
                )
            ):
                raise ValueError("Kimi K3 currently requires CP=PP=1")
            if self.n_heads % p.tensor_parallel_degree or self.num_experts % p.expert_parallel_degree:
                raise ValueError("K3 heads/experts must divide evenly across TP/EP")
            if p.tensor_parallel_degree > 1 and p.expert_parallel_degree != p.tensor_parallel_degree:
                raise ValueError("K3 tensor parallel training currently requires EP=TP")
            with (Path(config.hf_assets_path) / "config.json").open() as source:
                hf = json.load(source)
            hf = hf.get("text_config", hf)
            expected = {
                "hidden_size": self.dim,
                "num_hidden_layers": self.n_layers,
                "num_experts": self.num_experts,
                "num_experts_per_token": self.top_k,
                "vocab_size": self.vocab_size,
                "routed_expert_hidden_size": self.latent_dim,
                "mla_use_nope": True,
                "activation_situ_beta": 4.0,
                "activation_situ_linear_beta": 25.0,
            }
            for name, value in expected.items():
                if hf.get(name) != value:
                    raise ValueError(f"Unsupported K3 checkpoint: {name}={hf.get(name)!r}, expected {value!r}")
            if "quantization_config" in hf:
                raise ValueError("Kimi K3 trainer needs the BF16 dequantized checkpoint")

        def get_nparams_and_flops(self, model, seq_len):
            total = sum(p.numel() for p in model.parameters())
            experts = sum(p.numel() for n, p in model.named_parameters() if "inner_experts" in n)
            active = total - experts + experts * self.top_k // self.num_experts
            return total, 6 * active + 12 * (self.n_layers // 4) * self.n_heads * 192 * seq_len

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.enable_weight_tying = False
        self._skip_lm_head = False
        self.tok_embeddings = Embedding.Config(num_embeddings=config.vocab_size, embedding_dim=config.dim).build()
        self.layers = LayerDict({str(i): KimiBlock(config, i) for i in range(config.n_layers)})
        self.output_attn_res_norm = RMSNorm(config.dim)
        self.output_attn_res_proj = linear(config.dim, 1)
        self.norm = RMSNorm(config.dim)
        self.lm_head = linear(config.dim, config.vocab_size)

    def forward(self, tokens, *, positions=None, cu_seqlens=None, **kwargs):
        cu_seqlens, mask = packed_metadata(tokens, positions, cu_seqlens)
        x = self.tok_embeddings(tokens).reshape(1, -1, self.config.dim)
        bank = x.new_empty(1, x.shape[1], 0, x.shape[2])
        for layer in self.layers.values():
            x, bank = layer(x, bank, cu_seqlens, mask)
        x = attention_residual(x, bank, self.output_attn_res_norm, self.output_attn_res_proj)
        x = self.norm(x).reshape(*tokens.shape, -1)
        return x if self._skip_lm_head else self.lm_head(x)
