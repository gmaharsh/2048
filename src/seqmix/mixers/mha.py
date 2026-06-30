"""Multi-Head Attention (MHA): the dense O(n^2) gold-standard control.

Supports an optional sliding window so that the same implementation backs both
MHA (window=None) and SWA (window=W).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..rope import apply_rope, build_rope_cache
from .base import SequenceMixer


class MHAMixer(SequenceMixer):
    is_attention = True

    def __init__(self, config: ModelConfig, window: Optional[int] = None):
        super().__init__(config)
        self.n_heads = int(config.mixer_kwargs.get("n_heads", config.n_heads))
        self.d_head = int(config.mixer_kwargs.get("d_head", config.resolved_d_head()))
        self.window = window if window is not None else config.mixer_kwargs.get("window", None)
        self.d_model = config.d_model
        inner = self.n_heads * self.d_head

        self.q_proj = nn.Linear(self.d_model, inner, bias=False)
        self.k_proj = nn.Linear(self.d_model, inner, bias=False)
        self.v_proj = nn.Linear(self.d_model, inner, bias=False)
        self.o_proj = nn.Linear(inner, self.d_model, bias=False)
        self.dropout = config.dropout

        cos, sin = build_rope_cache(config.max_seq_len, self.d_head, config.rope_theta, device="cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,T,d)

    def _maybe_window_mask(self, T: int, device) -> Optional[torch.Tensor]:
        if self.window is None:
            return None
        # Allow position i to attend to (i-window, i]. Combined with causal.
        idx = torch.arange(T, device=device)
        rel = idx[None, :] - idx[:, None]  # key - query
        allowed = (rel <= 0) & (rel > -self.window)
        mask = torch.zeros(T, T, device=device)
        mask = mask.masked_fill(~allowed, float("-inf"))
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))
        cos = self.rope_cos.to(x.device)
        sin = self.rope_sin.to(x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn_mask = self._maybe_window_mask(T, x.device)
        is_causal = attn_mask is None
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).reshape(B, T, self.n_heads * self.d_head)
        return self.o_proj(y)

    def analytic_state_bytes(self, seq_len: int, dtype_bytes: int = 2) -> int:
        eff = seq_len if self.window is None else min(seq_len, self.window)
        # K and V, each (heads * d_head) per cached token.
        return 2 * eff * self.n_heads * self.d_head * dtype_bytes

    def init_cache(self, batch_size: int, device, dtype=torch.float32):
        return {"k": [], "v": []}

    def step(self, x_t: torch.Tensor, cache, pos: int):
        B = x_t.shape[0]
        q = self._shape(self.q_proj(x_t))
        k = self._shape(self.k_proj(x_t))
        v = self._shape(self.v_proj(x_t))
        cos = self.rope_cos.to(x_t.device)
        sin = self.rope_sin.to(x_t.device)
        q = apply_rope(q, cos, sin, offset=pos)
        k = apply_rope(k, cos, sin, offset=pos)

        cache["k"].append(k)
        cache["v"].append(v)
        if self.window is not None and len(cache["k"]) > self.window:
            cache["k"] = cache["k"][-self.window:]
            cache["v"] = cache["v"][-self.window:]
        K = torch.cat(cache["k"], dim=2)
        V = torch.cat(cache["v"], dim=2)
        y = F.scaled_dot_product_attention(q, K, V, is_causal=False)
        y = y.transpose(1, 2).reshape(B, 1, self.n_heads * self.d_head)
        return self.o_proj(y), cache


class SWAMixer(MHAMixer):
    """Sliding-Window Attention: MHA restricted to a local window."""

    def __init__(self, config: ModelConfig):
        window = int(config.mixer_kwargs.get("window", 64))
        super().__init__(config, window=window)
