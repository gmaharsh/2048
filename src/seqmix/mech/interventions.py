"""Causal interventions to explain *how* each mixer solves recall.

Two complementary probes, both following the logic of mechanistic evaluations
of attention vs. SSMs (e.g. arXiv:2505.15105):

* ``layer_knockout`` -- zero a single block's mixer contribution and measure the
  accuracy drop. A mechanism that lives in one layer (direct retrieval, as in
  Mamba) degrades differently than a two-layer mechanism (induction, as in
  attention) where two specific layers are jointly necessary.

* ``residual_patch_recovery`` -- run a clean and a corrupted input, then patch
  the residual stream after one block (at the recall position) from clean into
  corrupted. The fraction of accuracy recovered localises where the association
  is read out.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from ..data.synthetic import SyntheticConfig, get_batch, scored_accuracy
from ..model import LanguageModel
from ..utils import get_device


@torch.no_grad()
def _accuracy_on_batch(model, x, y, device) -> float:
    logits, _ = model(x.to(device), None)
    return scored_accuracy(logits, y.to(device))


@torch.no_grad()
def layer_knockout(model: LanguageModel, syn_cfg: SyntheticConfig,
                   batch_size: int = 128, seed: int = 7, device=None) -> Dict:
    """Per-layer accuracy when that layer's mixer output is zeroed."""
    device = device or get_device()
    model = model.to(device).eval()
    rng = np.random.default_rng(seed)
    x, y = get_batch(syn_cfg, batch_size, rng)

    baseline = _accuracy_on_batch(model, x, y, device)
    per_layer = []
    for li, block in enumerate(model.blocks):
        orig = block.mixer.forward

        def zero_forward(inp, _orig=orig):
            return torch.zeros_like(_orig(inp))

        block.mixer.forward = zero_forward
        acc = _accuracy_on_batch(model, x, y, device)
        block.mixer.forward = orig
        per_layer.append({"layer": li, "accuracy_when_knocked_out": acc,
                          "drop": baseline - acc})
    return {"baseline_accuracy": baseline, "per_layer": per_layer}


@torch.no_grad()
def block_outputs(model: LanguageModel, idx: torch.Tensor) -> List[torch.Tensor]:
    """Capture the residual stream after each block via forward hooks."""
    captured: List[torch.Tensor] = []
    handles = []
    for block in model.blocks:
        handles.append(block.register_forward_hook(
            lambda m, i, o: captured.append(o.detach())))
    model(idx)
    for h in handles:
        h.remove()
    return captured


@torch.no_grad()
def residual_patch_recovery(model: LanguageModel, syn_cfg: SyntheticConfig,
                            batch_size: int = 128, seed: int = 7, device=None) -> Dict:
    """Patch each block's output (at scored positions) from a clean run into a
    corrupted run; report fraction of accuracy recovered per layer."""
    device = device or get_device()
    model = model.to(device).eval()
    rng = np.random.default_rng(seed)

    x_clean, y = get_batch(syn_cfg, batch_size, rng)
    x_clean, y = x_clean.to(device), y.to(device)
    # corruption: shuffle the key/value pair region so associations break
    x_corrupt = x_clean.clone()
    perm = torch.randperm(x_corrupt.shape[1], device=device)
    x_corrupt = x_corrupt[:, perm]

    clean_acts = block_outputs(model, x_clean)
    acc_clean = scored_accuracy(model(x_clean)[0], y)
    acc_corrupt = scored_accuracy(model(x_corrupt)[0], y)

    scored_mask = (y != -100)
    per_layer = []
    for li, block in enumerate(model.blocks):
        clean_act = clean_acts[li]

        def patch_hook(m, i, o, _ca=clean_act, _mask=scored_mask):
            o = o.clone()
            o[_mask] = _ca[_mask]
            return o

        handle = block.register_forward_hook(patch_hook)
        acc_patched = scored_accuracy(model(x_corrupt)[0], y)
        handle.remove()
        denom = max(1e-6, acc_clean - acc_corrupt)
        per_layer.append({"layer": li, "accuracy_patched": acc_patched,
                          "recovery": (acc_patched - acc_corrupt) / denom})
    return {"acc_clean": acc_clean, "acc_corrupt": acc_corrupt, "per_layer": per_layer}
