"""Shared utilities: seeding, devices, parameter counting, JSON logging."""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    params: Iterable[torch.nn.Parameter] = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def count_non_embedding_parameters(model: torch.nn.Module) -> int:
    """Parameter count excluding token-embedding / tied-LM-head weights.

    Non-embedding parameters are the fair axis for comparing capacity across
    mixers because the embedding tables are identical for all of them.
    """
    total = count_parameters(model, trainable_only=True)
    emb = 0
    for name, p in model.named_parameters():
        if "embed" in name.lower() or name.endswith("lm_head.weight"):
            emb += p.numel()
    return total - emb


def human_readable(n: float) -> str:
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000.0:
            return f"{n:.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}P"


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_jsonable(data), f, indent=2)


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(_to_jsonable(record)) + "\n")


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


@contextmanager
def timer():
    """Context manager yielding a callable that returns elapsed seconds."""
    start = time.perf_counter()
    elapsed = {"value": 0.0}

    def read() -> float:
        return elapsed["value"]

    try:
        yield read
    finally:
        elapsed["value"] = time.perf_counter() - start
