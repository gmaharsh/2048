"""Common interface for sequence mixers.

Every mixer is an ``nn.Module`` that maps (B, T, D) -> (B, T, D) causally and
additionally exposes:

* ``analytic_state_bytes(seq_len, dtype_bytes)`` -- the per-sequence cache/state
  footprint (KV cache for attention, recurrent state for SSMs). This is the
  *memory* axis of the recall-memory-compute frontier.
* ``init_cache`` / ``step`` -- an incremental single-token decode path used by
  the efficiency benchmarks to measure decode latency/throughput.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from ..config import ModelConfig


class SequenceMixer(nn.Module):
    is_attention: bool = False

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    # ---- memory accounting -------------------------------------------------
    def analytic_state_bytes(self, seq_len: int, dtype_bytes: int = 2) -> int:
        """Bytes of cache/state required to decode the next token at ``seq_len``."""
        raise NotImplementedError

    # ---- incremental decode ------------------------------------------------
    def init_cache(self, batch_size: int, device, dtype=torch.float32) -> Any:
        raise NotImplementedError

    def step(self, x_t: torch.Tensor, cache: Any, pos: int):
        """Process one token. ``x_t`` is (B, 1, D); returns (y_t, cache)."""
        raise NotImplementedError
