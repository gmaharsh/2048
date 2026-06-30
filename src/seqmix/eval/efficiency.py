"""Compute/memory axis: analytic cache size, prefill & decode latency,
throughput, and peak memory as a function of context length.

Latency is wall-clock and therefore hardware dependent; the analytic
cache-bytes curve is hardware independent and is the cleanest cross-machine
memory metric. Peak GPU memory is only meaningful on CUDA.
"""

from __future__ import annotations

import time
from typing import Dict, List

import torch

from ..config import ModelConfig
from ..model import LanguageModel
from ..utils import get_device


def analytic_cache_curve(model: LanguageModel, seq_lens: List[int],
                         dtype_bytes: int = 2) -> List[Dict]:
    return [{"seq_len": L, "state_bytes": model.analytic_state_bytes(L, dtype_bytes)}
            for L in seq_lens]


@torch.no_grad()
def benchmark_prefill(model: LanguageModel, seq_len: int, batch_size: int = 1,
                      device=None, repeats: int = 5, warmup: int = 2) -> float:
    device = device or get_device()
    model = model.to(device).eval()
    idx = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    for _ in range(warmup):
        model(idx)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        model(idx)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


@torch.no_grad()
def benchmark_decode(model: LanguageModel, prompt_len: int, gen_len: int,
                     batch_size: int = 1, device=None) -> Dict:
    device = device or get_device()
    model = model.to(device).eval()
    # keep all decode positions within the model's positional range
    max_pos = model.config.max_seq_len
    prompt_len = max(1, min(prompt_len, max_pos - gen_len))
    idx = torch.randint(0, model.config.vocab_size, (batch_size, prompt_len), device=device)

    caches = model.decode_step_init(batch_size, device)
    # prefill token-by-token to populate caches (reference path)
    for t in range(prompt_len):
        logits, caches = model.decode_step(idx[:, t:t + 1], caches, t)
    nxt = logits[:, -1].argmax(-1, keepdim=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for t in range(gen_len):
        logits, caches = model.decode_step(nxt, caches, prompt_len + t)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"decode_s_per_tok": dt / gen_len,
            "decode_tok_per_s": gen_len * batch_size / dt}


def peak_memory_mb(device=None) -> float:
    device = device or get_device()
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    return float("nan")  # not tracked on CPU


def profile_model(model_cfg: ModelConfig, seq_lens: List[int], batch_size: int = 1,
                  gen_len: int = 32, device=None) -> Dict:
    device = device or get_device()
    model = LanguageModel(model_cfg)
    out = {"mixer": model_cfg.mixer, "analytic": analytic_cache_curve(model, seq_lens),
           "prefill": [], "decode": []}
    for L in seq_lens:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        pf = benchmark_prefill(model, L, batch_size, device)
        dec = benchmark_decode(model, min(L, max(seq_lens)), gen_len, batch_size, device)
        out["prefill"].append({"seq_len": L, "prefill_s": pf, "peak_mb": peak_memory_mb(device)})
        out["decode"].append({"prompt_len": L, **dec})
    return out
