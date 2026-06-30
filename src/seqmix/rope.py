"""Rotary position embeddings (RoPE), shared by the attention-family mixers.

MLA uses a *decoupled* RoPE: only a small per-head ``d_rope`` slice carries
positional information, while the bulk of the key/value content is rotation
free and stored compressed in the latent cache.
"""

from __future__ import annotations

import torch


def build_rope_cache(seq_len: int, dim: int, theta: float, device, dtype=torch.float32):
    """Return (cos, sin) tables of shape (seq_len, dim).

    ``dim`` must be even. The same angle is repeated for the two halves so we
    can apply rotation with the standard rotate-half trick.
    """
    assert dim % 2 == 0, "RoPE dimension must be even"
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (seq_len, dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0):
    """Apply RoPE to ``x`` of shape (B, H, T, D).

    ``cos``/``sin`` are (max_seq, D); ``offset`` selects the start position for
    incremental decoding.
    """
    T = x.shape[-2]
    cos_t = cos[offset:offset + T].unsqueeze(0).unsqueeze(0)  # (1,1,T,D)
    sin_t = sin[offset:offset + T].unsqueeze(0).unsqueeze(0)
    return x * cos_t + rotate_half(x) * sin_t
