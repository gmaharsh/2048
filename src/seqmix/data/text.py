"""Byte/character-level text data for language-model perplexity experiments.

Loads a real corpus from disk when available (e.g. enwik8, WikiText). When no
file is supplied it synthesises a corpus with deliberate long-range repetition
so that recall-capable mixers can separate from local ones even at tiny scale
and without internet access.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


class ByteDataset:
    def __init__(self, data: np.ndarray, vocab_size: int = 256):
        self.data = data.astype(np.int64)
        self.vocab_size = vocab_size

    @classmethod
    def from_file(cls, path: str, max_bytes: Optional[int] = None) -> "ByteDataset":
        with open(path, "rb") as f:
            raw = f.read() if max_bytes is None else f.read(max_bytes)
        return cls(np.frombuffer(raw, dtype=np.uint8).copy(), vocab_size=256)

    @classmethod
    def synthetic_recall_corpus(cls, n_bytes: int = 1_000_000, vocab_size: int = 64,
                                seed: int = 0, repeat_prob: float = 0.15,
                                phrase_len: int = 8) -> "ByteDataset":
        """A stream of random 'words' where phrases recur after long gaps.

        Recurrence injects long-range dependencies: a model that can recall the
        earlier occurrence predicts the continuation better than a purely local
        one, which is exactly the axis we study.
        """
        rng = np.random.default_rng(seed)
        out = np.empty(n_bytes, dtype=np.int64)
        bank = [rng.integers(1, vocab_size, size=phrase_len) for _ in range(64)]
        i = 0
        while i < n_bytes:
            if rng.random() < repeat_prob and i + phrase_len < n_bytes:
                phrase = bank[rng.integers(0, len(bank))]
                out[i:i + phrase_len] = phrase
                i += phrase_len
            else:
                out[i] = rng.integers(1, vocab_size)
                i += 1
        return cls(out, vocab_size=vocab_size)

    def split(self, frac: float = 0.9) -> Tuple["ByteDataset", "ByteDataset"]:
        n = int(len(self.data) * frac)
        return ByteDataset(self.data[:n], self.vocab_size), ByteDataset(self.data[n:], self.vocab_size)

    def get_batch(self, batch_size: int, seq_len: int, rng: np.random.Generator,
                  device="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        ix = rng.integers(0, len(self.data) - seq_len - 1, size=batch_size)
        x = np.stack([self.data[i:i + seq_len] for i in ix])
        y = np.stack([self.data[i + 1:i + 1 + seq_len] for i in ix])
        return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
