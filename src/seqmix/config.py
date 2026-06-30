"""Configuration dataclasses for models and experiments.

A single ``ModelConfig`` drives the shared backbone. Mixer-specific options
live in ``mixer_kwargs`` so the backbone stays agnostic to which sequence
mixer is plugged in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    # Vocabulary / sequence
    vocab_size: int = 256
    max_seq_len: int = 1024

    # Shared backbone shape (held constant across mixers for fair comparison)
    d_model: int = 256
    n_layers: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    norm: str = "rmsnorm"  # {"rmsnorm", "layernorm"}
    tie_embeddings: bool = True

    # Which sequence mixer to plug in: {"mha", "mla", "swa", "mamba"}
    mixer: str = "mha"
    mixer_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Attention defaults (used by mha / mla / swa). May be overridden per-mixer.
    n_heads: int = 4
    d_head: Optional[int] = None  # defaults to d_model // n_heads
    rope_theta: float = 10000.0

    def resolved_d_head(self) -> int:
        if self.d_head is not None:
            return self.d_head
        assert self.d_model % self.n_heads == 0, "d_model must divide n_heads"
        return self.d_model // self.n_heads


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 32
    seq_len: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup: int = 100
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    eval_every: int = 200
    eval_iters: int = 50
    seed: int = 0
    device: Optional[str] = None
    amp: bool = False
    log_every: int = 50
    compile: bool = False
