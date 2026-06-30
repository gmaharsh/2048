"""Multi-head Latent Attention (MLA), following DeepSeek-V2.

Key idea: keys and values are jointly compressed into a low-rank latent
``c_kv`` (dim ``d_c``) that is the *only* large thing cached, plus a small
shared decoupled-RoPE key ``k_rope`` (dim ``d_rope``). The KV cache is therefore
``(d_c + d_rope)`` elements per token, independent of the number of heads --
the source of MLA's large memory savings over MHA.

Per-head query/key content dims are ``d_nope`` (rotation-free) and ``d_rope``
(rotary). The attention logit for a head is
    q_nope . k_nope  +  q_rope . k_rope
where ``k_rope`` is shared across heads.

This is a faithful reference implementation (decompress-then-attend). The
inference-time "weight absorption" trick that avoids materialising k_nope/v is
a compute optimisation that does not change the cache footprint or outputs, so
we omit it for clarity.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..rope import apply_rope, build_rope_cache
from .base import SequenceMixer


class MLAMixer(SequenceMixer):
    is_attention = True

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        mk = config.mixer_kwargs
        self.d_model = config.d_model
        self.n_heads = int(mk.get("n_heads", config.n_heads))
        d_head = config.resolved_d_head()
        self.d_nope = int(mk.get("d_nope", d_head))
        self.d_rope = int(mk.get("d_rope", max(2, d_head // 2)))
        self.d_v = int(mk.get("d_v", d_head))
        self.d_c = int(mk.get("d_c", 4 * d_head))       # KV latent (cached)
        self.q_lora_rank: Optional[int] = mk.get("q_lora_rank", None)

        self.d_qk = self.d_nope + self.d_rope
        self.scale = self.d_qk ** -0.5

        # ----- query path (optionally low-rank) -----
        if self.q_lora_rank:
            self.q_down = nn.Linear(self.d_model, self.q_lora_rank, bias=False)
            self.q_norm = nn.RMSNorm(self.q_lora_rank) if hasattr(nn, "RMSNorm") else nn.LayerNorm(self.q_lora_rank)
            self.q_up = nn.Linear(self.q_lora_rank, self.n_heads * self.d_qk, bias=False)
        else:
            self.q_proj = nn.Linear(self.d_model, self.n_heads * self.d_qk, bias=False)

        # ----- key/value path: joint low-rank compression -----
        self.kv_down = nn.Linear(self.d_model, self.d_c, bias=False)              # -> c_kv (cached)
        self.kv_norm = nn.RMSNorm(self.d_c) if hasattr(nn, "RMSNorm") else nn.LayerNorm(self.d_c)
        self.kv_up = nn.Linear(self.d_c, self.n_heads * (self.d_nope + self.d_v), bias=False)
        self.k_rope_proj = nn.Linear(self.d_model, self.d_rope, bias=False)        # shared k_rope (cached)

        self.o_proj = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=False)
        self.dropout = config.dropout

        cos, sin = build_rope_cache(config.max_seq_len, self.d_rope, config.rope_theta, device="cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _query(self, x: torch.Tensor) -> torch.Tensor:
        if self.q_lora_rank:
            q = self.q_up(self.q_norm(self.q_down(x)))
        else:
            q = self.q_proj(x)
        B, T, _ = x.shape
        return q.view(B, T, self.n_heads, self.d_qk).transpose(1, 2)  # (B,H,T,d_qk)

    def _kv(self, c_kv: torch.Tensor):
        B, T, _ = c_kv.shape
        kv = self.kv_up(c_kv).view(B, T, self.n_heads, self.d_nope + self.d_v).transpose(1, 2)
        k_nope, v = kv.split([self.d_nope, self.d_v], dim=-1)
        return k_nope, v  # (B,H,T,d_nope), (B,H,T,d_v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        cos = self.rope_cos.to(x.device)
        sin = self.rope_sin.to(x.device)

        q = self._query(x)
        q_nope, q_rope = q.split([self.d_nope, self.d_rope], dim=-1)
        q_rope = apply_rope(q_rope, cos, sin)

        c_kv = self.kv_norm(self.kv_down(x))               # (B,T,d_c)  <- cached
        k_rope = self.k_rope_proj(x).unsqueeze(1)          # (B,1,T,d_rope)  <- cached
        k_rope = apply_rope(k_rope, cos, sin)
        k_nope, v = self._kv(c_kv)

        q_full = torch.cat([q_nope, q_rope], dim=-1)
        k_full = torch.cat([k_nope, k_rope.expand(-1, self.n_heads, -1, -1)], dim=-1)

        y = F.scaled_dot_product_attention(
            q_full, k_full, v, is_causal=True, scale=self.scale,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).reshape(B, T, self.n_heads * self.d_v)
        return self.o_proj(y)

    def analytic_state_bytes(self, seq_len: int, dtype_bytes: int = 2) -> int:
        # Only the latent c_kv (d_c) and shared k_rope (d_rope) are cached.
        return seq_len * (self.d_c + self.d_rope) * dtype_bytes

    def init_cache(self, batch_size: int, device, dtype=torch.float32):
        return {"c_kv": [], "k_rope": []}

    def step(self, x_t: torch.Tensor, cache, pos: int):
        B = x_t.shape[0]
        cos = self.rope_cos.to(x_t.device)
        sin = self.rope_sin.to(x_t.device)

        q = self._query(x_t)
        q_nope, q_rope = q.split([self.d_nope, self.d_rope], dim=-1)
        q_rope = apply_rope(q_rope, cos, sin, offset=pos)

        c_kv_t = self.kv_norm(self.kv_down(x_t))            # (B,1,d_c)
        k_rope_t = self.k_rope_proj(x_t).unsqueeze(1)       # (B,1,1,d_rope)
        k_rope_t = apply_rope(k_rope_t, cos, sin, offset=pos)
        cache["c_kv"].append(c_kv_t)
        cache["k_rope"].append(k_rope_t)

        c_kv = torch.cat(cache["c_kv"], dim=1)              # (B,S,d_c)
        k_rope = torch.cat(cache["k_rope"], dim=2)          # (B,1,S,d_rope)
        k_nope, v = self._kv(c_kv)

        q_full = torch.cat([q_nope, q_rope], dim=-1)
        k_full = torch.cat([k_nope, k_rope.expand(-1, self.n_heads, -1, -1)], dim=-1)
        y = F.scaled_dot_product_attention(q_full, k_full, v, is_causal=False, scale=self.scale)
        y = y.transpose(1, 2).reshape(B, 1, self.n_heads * self.d_v)
        return self.o_proj(y), cache
