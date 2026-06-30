"""Synthetic capability tasks for probing recall vs. memory.

These tasks isolate specific capabilities and are cheap enough to sweep on a
single GPU/CPU. We follow the now-standard formulations used by Zoology / MAD:

* ``mqar``           -- Multi-Query Associative Recall. The central recall test:
                        memorise key->value pairs, then answer queries. Token
                        positions that are not queried are masked (-100).
* ``selective_copy`` -- copy scattered content tokens, in order, into an output
                        region (the Mamba selective-copying task).
* ``induction``      -- in-context bigram completion (induction-head task).

All generators return integer tensors ``(inputs, targets)`` of shape (B, L);
``targets`` uses -100 (ignore_index) everywhere the position is not scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

BLANK = 0  # reserved gap/pad token (never a key or value)


@dataclass
class SyntheticConfig:
    task: str = "mqar"
    seq_len: int = 128
    vocab_size: int = 64
    num_kv_pairs: int = 8       # mqar
    num_queries: Optional[int] = None  # mqar (defaults to num_kv_pairs)
    num_tokens: int = 16        # selective_copy: number of content tokens
    needle_depth_frac: Optional[float] = None  # passkey: fixed depth in [0,1], else random
    seed: int = 0

    def resolved_queries(self) -> int:
        return self.num_queries if self.num_queries is not None else self.num_kv_pairs


# ---------------------------------------------------------------------------
# MQAR
# ---------------------------------------------------------------------------

def make_mqar(batch_size: int, cfg: SyntheticConfig, rng: np.random.Generator):
    L, V, N = cfg.seq_len, cfg.vocab_size, cfg.num_kv_pairs
    Q = cfg.resolved_queries()
    assert 2 * N + Q <= L, "sequence too short for #pairs and #queries"
    assert N + 1 < V, "vocab too small for #pairs"

    inputs = np.full((batch_size, L), BLANK, dtype=np.int64)
    targets = np.full((batch_size, L), -100, dtype=np.int64)

    for b in range(batch_size):
        symbols = rng.choice(np.arange(1, V), size=N, replace=False)
        values = rng.choice(np.arange(1, V), size=N, replace=True)
        # key/value pairs packed at the front
        inputs[b, 0:2 * N:2] = symbols
        inputs[b, 1:2 * N:2] = values
        # queries placed at random positions in the remainder
        positions = rng.choice(np.arange(2 * N, L), size=Q, replace=False)
        which = rng.integers(0, N, size=Q)
        inputs[b, positions] = symbols[which]
        targets[b, positions] = values[which]
    return torch.from_numpy(inputs), torch.from_numpy(targets)


# ---------------------------------------------------------------------------
# Selective copying
# ---------------------------------------------------------------------------

def make_selective_copy(batch_size: int, cfg: SyntheticConfig, rng: np.random.Generator):
    L, V, K = cfg.seq_len, cfg.vocab_size, cfg.num_tokens
    # Layout: [memorise region (L - K - 1)] [marker] [output region (K)]
    marker = V - 1                      # dedicated copy-trigger token
    content_vocab = np.arange(1, V - 1)  # exclude BLANK and marker
    mem_len = L - K - 1
    assert mem_len >= K, "memorise region too short"

    inputs = np.full((batch_size, L), BLANK, dtype=np.int64)
    targets = np.full((batch_size, L), -100, dtype=np.int64)
    for b in range(batch_size):
        content = rng.choice(content_vocab, size=K, replace=True)
        slots = np.sort(rng.choice(np.arange(mem_len), size=K, replace=False))
        inputs[b, slots] = content
        inputs[b, mem_len] = marker
        out_start = mem_len + 1
        # teacher forcing: feed the marker then previously emitted tokens
        inputs[b, out_start:out_start + K - 1] = content[:-1]
        targets[b, mem_len:mem_len + K] = content  # predict each content token in order
    return torch.from_numpy(inputs), torch.from_numpy(targets)


# ---------------------------------------------------------------------------
# Induction heads
# ---------------------------------------------------------------------------

def make_induction(batch_size: int, cfg: SyntheticConfig, rng: np.random.Generator):
    """Bigram completion: a cue token appears twice; predict the token that
    followed it the first time. Tests the induction-head mechanism."""
    L, V = cfg.seq_len, cfg.vocab_size
    cue = V - 1
    body_vocab = np.arange(1, V - 1)

    inputs = np.full((batch_size, L), BLANK, dtype=np.int64)
    targets = np.full((batch_size, L), -100, dtype=np.int64)
    for b in range(batch_size):
        seq = rng.choice(body_vocab, size=L, replace=True)
        first = rng.integers(1, L - 2)
        answer = seq[first + 1]
        seq[first] = cue
        seq[L - 1] = cue           # second occurrence at the end
        inputs[b] = seq
        targets[b, L - 1] = answer  # predict what followed the cue the first time
    return torch.from_numpy(inputs), torch.from_numpy(targets)


def make_passkey(batch_size: int, cfg: SyntheticConfig, rng: np.random.Generator):
    """Needle-in-a-haystack: a (key,value) needle sits at some depth inside a
    field of random distractor tokens; the last position queries the key and
    must recall the value. Used to probe long-context retrieval vs. depth."""
    L, V = cfg.seq_len, cfg.vocab_size
    key_tok = V - 1                      # dedicated needle key
    filler_vocab = np.arange(1, V - 1)

    inputs = np.empty((batch_size, L), dtype=np.int64)
    targets = np.full((batch_size, L), -100, dtype=np.int64)
    for b in range(batch_size):
        inputs[b] = rng.choice(filler_vocab, size=L, replace=True)
        value = rng.integers(1, V - 1)
        if cfg.needle_depth_frac is None:
            depth = rng.integers(0, L - 3)
        else:
            depth = int(cfg.needle_depth_frac * (L - 3))
        inputs[b, depth] = key_tok
        inputs[b, depth + 1] = value
        inputs[b, L - 1] = key_tok       # query at the end
        targets[b, L - 1] = value
    return torch.from_numpy(inputs), torch.from_numpy(targets)


_GENERATORS = {
    "mqar": make_mqar,
    "selective_copy": make_selective_copy,
    "induction": make_induction,
    "passkey": make_passkey,
}


def get_batch(cfg: SyntheticConfig, batch_size: int, rng: np.random.Generator
              ) -> Tuple[torch.Tensor, torch.Tensor]:
    if cfg.task not in _GENERATORS:
        raise ValueError(f"Unknown task '{cfg.task}'. Options: {sorted(_GENERATORS)}")
    return _GENERATORS[cfg.task](batch_size, cfg, rng)


@torch.no_grad()
def scored_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Token accuracy over scored (target != -100) positions."""
    mask = targets != -100
    if mask.sum() == 0:
        return float("nan")
    pred = logits.argmax(dim=-1)
    correct = (pred[mask] == targets[mask]).float().mean().item()
    return correct
