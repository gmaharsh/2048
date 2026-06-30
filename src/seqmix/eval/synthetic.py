"""Phase 1 sweeps: trace each mixer across its memory knob to build the
recall-memory frontier.

For every mixer we vary the single hyperparameter that controls its cache /
state footprint and record (state-bytes, params, accuracy) on a fixed synthetic
task. Plotting accuracy vs. state-bytes yields the recall-memory Pareto plot
that is the headline of the study.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..config import ModelConfig, TrainConfig
from ..data.synthetic import SyntheticConfig
from ..train import train_synthetic
from ..utils import append_jsonl


# The knob that controls each mixer's cache/state footprint.
MEMORY_KNOB = {
    "mha": "n_heads",          # cache ~ n_heads * d_head * seq_len
    "swa": "window",           # cache ~ window
    "mla": "d_c",              # cache ~ d_c + d_rope
    "mamba": "d_state",        # state ~ d_inner * d_state
}


@dataclass
class SweepPoint:
    mixer: str
    knob_value: int
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


def build_model_cfg(base: ModelConfig, point: SweepPoint) -> ModelConfig:
    cfg = copy.deepcopy(base)
    cfg.mixer = point.mixer
    knob = MEMORY_KNOB[point.mixer]
    mk = dict(point.extra_kwargs)
    mk[knob] = point.knob_value
    cfg.mixer_kwargs = mk
    return cfg


def run_point(base_cfg: ModelConfig, syn_cfg: SyntheticConfig,
              train_cfg: TrainConfig, point: SweepPoint) -> Dict:
    model_cfg = build_model_cfg(base_cfg, point)
    res = train_synthetic(model_cfg, syn_cfg, train_cfg)
    return {
        "task": syn_cfg.task,
        "mixer": point.mixer,
        "knob": MEMORY_KNOB[point.mixer],
        "knob_value": point.knob_value,
        "accuracy": res["final_accuracy"],
        "state_bytes": res["analytic_state_bytes"],
        "params": res["non_embedding_params"],
        "seq_len": syn_cfg.seq_len,
        "num_kv_pairs": syn_cfg.num_kv_pairs,
        "seed": train_cfg.seed,
    }


def run_sweep(base_cfg: ModelConfig, syn_cfg: SyntheticConfig, train_cfg: TrainConfig,
              points: List[SweepPoint], out_path: str, verbose: bool = True) -> List[Dict]:
    records = []
    for pt in points:
        rec = run_point(base_cfg, syn_cfg, train_cfg, pt)
        records.append(rec)
        append_jsonl(out_path, rec)
        if verbose:
            print(f"[{rec['task']}] {rec['mixer']:6s} {rec['knob']}={rec['knob_value']:<4} "
                  f"acc={rec['accuracy']:.3f} state_bytes={rec['state_bytes']:>7d} "
                  f"params={rec['params']}")
    return records
