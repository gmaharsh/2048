"""Phase 3: long-context recall.

Two probes:
* ``passkey_accuracy_by_depth`` -- train on the passkey task (random depth) and
  measure recall accuracy as a function of needle depth and context length.
  This is where local (SWA) and fixed-state (Mamba) mixers should diverge from
  full/latent attention (MHA/MLA).
* ``ppl_by_length`` -- validation perplexity of a trained LM at increasing
  evaluation context lengths.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List

import numpy as np
import torch

from ..config import ModelConfig, TrainConfig
from ..data.synthetic import SyntheticConfig, get_batch, scored_accuracy
from ..data.text import ByteDataset
from ..model import LanguageModel
from ..train import train_synthetic
from ..utils import get_device


@torch.no_grad()
def passkey_accuracy_by_depth(model: LanguageModel, seq_len: int, vocab_size: int,
                              depths: List[float], batch_size: int = 64,
                              seed: int = 1234, device=None) -> List[Dict]:
    device = device or get_device()
    model = model.to(device).eval()
    out = []
    for d in depths:
        cfg = SyntheticConfig(task="passkey", seq_len=seq_len, vocab_size=vocab_size,
                              needle_depth_frac=d)
        rng = np.random.default_rng(seed + int(d * 1000))
        x, y = get_batch(cfg, batch_size, rng)
        logits, _ = model(x.to(device), None)
        out.append({"depth_frac": d, "accuracy": scored_accuracy(logits, y.to(device))})
    return out


def train_passkey_model(model_cfg: ModelConfig, seq_len: int, vocab_size: int,
                        train_cfg: TrainConfig) -> LanguageModel:
    syn = SyntheticConfig(task="passkey", seq_len=seq_len, vocab_size=vocab_size)
    res = train_synthetic(model_cfg, syn, train_cfg)
    return res["model"]


@torch.no_grad()
def ppl_by_length(model: LanguageModel, dataset: ByteDataset, lengths: List[int],
                  batch_size: int = 8, iters: int = 20, seed: int = 0, device=None) -> List[Dict]:
    device = device or get_device()
    model = model.to(device).eval()
    rng = np.random.default_rng(seed)
    out = []
    for L in lengths:
        if L > model.config.max_seq_len:
            continue
        losses = []
        for _ in range(iters):
            x, y = dataset.get_batch(batch_size, L, rng, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out.append({"seq_len": L, "val_ppl": math.exp(float(np.mean(losses)))})
    return out
