"""KDA and gated NoPE MLA, with document boundaries preserved during training."""

import torch
import torch.nn.functional as F
from fla.ops.kda import chunk_kda
from torch import nn
from torch.distributed.tensor import DTensor
from torchtitan.protocols.module import Module

from miles.backends.torchtitan_utils.models.kimi_k3.ops import GatedRMSNorm, RMSNorm, ShortConv, linear, local_tensor


class KDACore(Module):
    def __init__(self, heads, width):
        super().__init__()
        self.heads = heads
        self.tp_rank = 0
        # The released checkpoint pads A_log to 128 heads; preserve those bytes on export.
        self.A_log = nn.Parameter(torch.zeros(128, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(width, dtype=torch.float32))

    def reset_parameters(self):
        nn.init.zeros_(self.A_log)
        nn.init.zeros_(self.dt_bias)

    def forward(self, q, k, v, gate, beta, cu_seqlens):
        distributed_q = q if isinstance(q, DTensor) else None
        q, k, v, gate, beta = [local_tensor(value) for value in (q, k, v, gate, beta)]
        heads = q.shape[-2]
        first_head = self.tp_rank * heads
        output, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=gate,
            beta=beta.float().sigmoid(),
            A_log=local_tensor(self.A_log, partial_grad=distributed_q is not None)[first_head : first_head + heads],
            dt_bias=local_tensor(self.dt_bias),
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
            state_v_first=True,
            cu_seqlens=cu_seqlens,
        )
        if distributed_q is not None:
            return DTensor.from_local(output, distributed_q.device_mesh, distributed_q.placements, run_check=False)
        return output


class DeltaAttention(Module):
    def __init__(self, config):
        super().__init__()
        dim, heads, head_dim = config.dim, config.n_heads, config.head_dim
        self.heads, self.head_dim = heads, head_dim
        width = heads * head_dim
        self.q_proj, self.k_proj, self.v_proj = [linear(dim, width) for _ in range(3)]
        self.q_conv1d, self.k_conv1d, self.v_conv1d = [
            ShortConv(hidden_size=width, kernel_size=4, activation="silu") for _ in range(3)
        ]
        self.core = KDACore(heads, width)
        self.f_a_proj, self.f_b_proj = linear(dim, head_dim), linear(head_dim, width)
        self.b_proj, self.g_proj = linear(dim, heads), linear(dim, width)
        self.o_norm, self.o_proj = GatedRMSNorm(head_dim), linear(width, dim)

    def forward(self, x, cu_seqlens, mask):
        del mask
        q, k, v = [
            conv(proj(x), cu_seqlens=cu_seqlens, output_final_state=False)[0].unflatten(
                -1, (self.heads, self.head_dim)
            )
            for proj, conv in (
                (self.q_proj, self.q_conv1d),
                (self.k_proj, self.k_conv1d),
                (self.v_proj, self.v_conv1d),
            )
        ]
        gate = self.f_b_proj(self.f_a_proj(x)).unflatten(-1, (self.heads, self.head_dim))
        output = self.core(q, k, v, gate, self.b_proj(x), cu_seqlens)
        output = self.o_norm(output, self.g_proj(x).unflatten(-1, (self.heads, self.head_dim)))
        return self.o_proj(output.flatten(-2))


class MLAAttention(Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.n_heads
        self.qk_dim, self.v_dim = 192, 128
        self.q_a_proj = linear(config.dim, 1536)
        self.q_a_layernorm = RMSNorm(1536, eps=1e-6)
        self.q_b_proj = linear(1536, self.heads * 192)
        self.kv_a_proj_with_mqa = linear(config.dim, 512 + 64)
        self.kv_a_layernorm = RMSNorm(512, eps=1e-6)
        self.kv_b_proj = linear(512, self.heads * 256)
        self.g_proj = linear(config.dim, self.heads * 128)
        self.o_proj = linear(self.heads * 128, config.dim)

    def forward(self, x, cu_seqlens, mask):
        del cu_seqlens
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x))).unflatten(-1, (self.heads, 192))
        latent, key_shared = self.kv_a_proj_with_mqa(x).split((512, 64), dim=-1)
        key, value = self.kv_b_proj(self.kv_a_layernorm(latent)).unflatten(-1, (self.heads, 256)).split(128, dim=-1)
        key = torch.cat((key, key_shared.unsqueeze(-2).expand(*key.shape[:-1], 64)), dim=-1)
        distributed_q = q if isinstance(q, DTensor) else None
        q, key, value = [local_tensor(tensor) for tensor in (q, key, value)]
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=mask,
            dropout_p=0.0,
            scale=192**-0.5,
        ).transpose(1, 2)
        if distributed_q is not None:
            output = DTensor.from_local(output, distributed_q.device_mesh, distributed_q.placements, run_check=False)
        output = output.flatten(-2)
        return self.o_proj(output * self.g_proj(x).sigmoid())
