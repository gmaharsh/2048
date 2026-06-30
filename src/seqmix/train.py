"""Training loops for synthetic-task and language-model experiments.

Kept deliberately small and dependency-light: AdamW + cosine schedule with
warmup, gradient clipping, periodic evaluation. The same backbone/optimizer is
used for every mixer so the comparison stays controlled.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .config import ModelConfig, TrainConfig
from .data.synthetic import SyntheticConfig, get_batch, scored_accuracy
from .data.text import ByteDataset
from .model import LanguageModel
from .utils import count_non_embedding_parameters, get_device, set_seed


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    progress = min(1.0, progress)
    return 0.5 * cfg.lr * (1.0 + math.cos(math.pi * progress))


def _run_step(model, optimizer, x, y, cfg, step) -> float:
    lr = lr_at(step, cfg)
    for g in optimizer.param_groups:
        g["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    if cfg.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    return loss.item()


def train_synthetic(model_cfg: ModelConfig, syn_cfg: SyntheticConfig,
                    train_cfg: TrainConfig, verbose: bool = False) -> Dict:
    set_seed(train_cfg.seed)
    device = get_device(train_cfg.device)
    model = LanguageModel(model_cfg).to(device)
    opt = build_optimizer(model, train_cfg)
    rng = np.random.default_rng(train_cfg.seed)
    eval_rng = np.random.default_rng(train_cfg.seed + 10_000)

    history: List[Dict] = []
    for step in range(train_cfg.steps):
        x, y = get_batch(syn_cfg, train_cfg.batch_size, rng)
        x, y = x.to(device), y.to(device)
        model.train()
        loss = _run_step(model, opt, x, y, train_cfg, step)
        if verbose and step % train_cfg.log_every == 0:
            print(f"  step {step:5d}  loss {loss:.4f}  lr {lr_at(step, train_cfg):.2e}")
        if (step + 1) % train_cfg.eval_every == 0 or step == train_cfg.steps - 1:
            acc = evaluate_synthetic(model, syn_cfg, train_cfg, eval_rng, device)
            history.append({"step": step + 1, "loss": loss, "accuracy": acc})
            if verbose:
                print(f"  [eval] step {step + 1}  acc {acc:.4f}")

    final_acc = history[-1]["accuracy"] if history else float("nan")
    return {
        "final_accuracy": final_acc,
        "history": history,
        "non_embedding_params": count_non_embedding_parameters(model),
        "analytic_state_bytes": model.analytic_state_bytes(syn_cfg.seq_len),
        "model_cfg": model_cfg,
        "syn_cfg": syn_cfg,
        "model": model,  # not JSON-serialisable; strip before saving
    }


@torch.no_grad()
def evaluate_synthetic(model, syn_cfg, train_cfg, rng, device) -> float:
    model.eval()
    accs = []
    for _ in range(max(1, train_cfg.eval_iters // 10)):
        x, y = get_batch(syn_cfg, train_cfg.batch_size, rng)
        logits, _ = model(x.to(device), None)
        accs.append(scored_accuracy(logits, y.to(device)))
    return float(np.nanmean(accs))


def train_lm(model_cfg: ModelConfig, dataset: ByteDataset, train_cfg: TrainConfig,
             verbose: bool = False) -> Dict:
    set_seed(train_cfg.seed)
    device = get_device(train_cfg.device)
    train_ds, val_ds = dataset.split(0.9)
    model = LanguageModel(model_cfg).to(device)
    opt = build_optimizer(model, train_cfg)
    rng = np.random.default_rng(train_cfg.seed)
    eval_rng = np.random.default_rng(train_cfg.seed + 10_000)

    history: List[Dict] = []
    for step in range(train_cfg.steps):
        x, y = train_ds.get_batch(train_cfg.batch_size, train_cfg.seq_len, rng, device)
        model.train()
        loss = _run_step(model, opt, x, y, train_cfg, step)
        if verbose and step % train_cfg.log_every == 0:
            print(f"  step {step:5d}  loss {loss:.4f}")
        if (step + 1) % train_cfg.eval_every == 0 or step == train_cfg.steps - 1:
            val_loss = evaluate_lm(model, val_ds, train_cfg, eval_rng, device)
            history.append({"step": step + 1, "train_loss": loss,
                            "val_loss": val_loss, "val_ppl": math.exp(val_loss)})
            if verbose:
                print(f"  [eval] step {step + 1}  val_ppl {math.exp(val_loss):.3f}")

    return {
        "final_val_ppl": history[-1]["val_ppl"] if history else float("nan"),
        "history": history,
        "non_embedding_params": count_non_embedding_parameters(model),
        "analytic_state_bytes": model.analytic_state_bytes(train_cfg.seq_len),
        "model_cfg": model_cfg,
        "model": model,  # not JSON-serialisable; strip before saving
    }


@torch.no_grad()
def evaluate_lm(model, val_ds, train_cfg, rng, device) -> float:
    model.eval()
    losses = []
    for _ in range(train_cfg.eval_iters):
        x, y = val_ds.get_batch(train_cfg.batch_size, train_cfg.seq_len, rng, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    return float(np.mean(losses))
