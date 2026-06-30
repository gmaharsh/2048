"""Shared decoder-only language-model backbone.

The ONLY thing that changes between experimental conditions is the sequence
mixer inside each block. Everything else (embeddings, MLP, norms, residual
structure, init) is held fixed so differences are attributable to the mixer.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .mixers import build_mixer


def make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "rmsnorm":
        if hasattr(nn, "RMSNorm"):
            return nn.RMSNorm(dim)
        return RMSNorm(dim)
    if kind == "layernorm":
        return nn.LayerNorm(dim)
    raise ValueError(f"Unknown norm '{kind}'")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = int(config.mlp_ratio * config.d_model)
        self.fc1 = nn.Linear(config.d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, config.d_model, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = make_norm(config.norm, config.d_model)
        self.mixer = build_mixer(config)
        self.norm2 = make_norm(config.norm, config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def step(self, x_t: torch.Tensor, cache, pos: int):
        y, cache = self.mixer.step(self.norm1(x_t), cache, pos)
        x_t = x_t + y
        x_t = x_t + self.mlp(self.norm2(x_t))
        return x_t, cache


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm_f = make_norm(config.norm, config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.drop(self.embed(idx))
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def analytic_state_bytes(self, seq_len: int, dtype_bytes: int = 2) -> int:
        """Total KV-cache / recurrent-state bytes across all layers."""
        return sum(b.mixer.analytic_state_bytes(seq_len, dtype_bytes) for b in self.blocks)

    @torch.no_grad()
    def decode_step_init(self, batch_size: int, device, dtype=torch.float32):
        return [b.mixer.init_cache(batch_size, device, dtype) for b in self.blocks]

    @torch.no_grad()
    def decode_step(self, idx_t: torch.Tensor, caches, pos: int):
        """Single-token incremental decode used by the efficiency benchmark."""
        x = self.embed(idx_t)
        for i, block in enumerate(self.blocks):
            x, caches[i] = block.step(x, caches[i], pos)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits, caches
