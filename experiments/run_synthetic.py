"""Phase 1 driver: sweep each mixer's memory knob on a synthetic task and write
the recall-memory frontier to results/synthetic/<task>.jsonl.

Usage:
    python experiments/run_synthetic.py --task mqar --scale demo
    python experiments/run_synthetic.py --task mqar --scale full --device cuda

The "demo" scale is tuned to run on CPU in minutes; "full" is meant for a GPU
(install mamba-ssm for fast Mamba). All knobs are CLI-overridable.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from seqmix.config import ModelConfig, TrainConfig  # noqa: E402
from seqmix.data.synthetic import SyntheticConfig  # noqa: E402
from seqmix.eval.synthetic import SweepPoint, run_sweep  # noqa: E402

# Memory-knob sweep values per mixer at each scale.
SWEEPS = {
    "demo": {
        "mha": [1, 2, 4],
        "swa": [4, 8, 16, 32],
        "mla": [8, 16, 32, 64],
        "mamba": [16],
    },
    "full": {
        "mha": [1, 2, 4, 8],
        "swa": [8, 16, 32, 64, 128],
        "mla": [8, 16, 32, 64, 128, 256],
        "mamba": [4, 8, 16, 32, 64],
    },
}

SCALE_DEFAULTS = {
    # demo is sized to finish on a single CPU (pure-PyTorch Mamba is the bottleneck).
    "demo": dict(d_model=96, n_layers=2, seq_len=64, vocab=20, kv=3, steps=1200, batch=32),
    "full": dict(d_model=256, n_layers=4, seq_len=256, vocab=64, kv=16, steps=8000, batch=64),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mqar", choices=["mqar", "selective_copy", "induction"])
    ap.add_argument("--scale", default="demo", choices=["demo", "full"])
    ap.add_argument("--mixers", nargs="+", default=["mha", "swa", "mla", "mamba"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = SCALE_DEFAULTS[args.scale]
    seq_len = args.seq_len or d["seq_len"]
    steps = args.steps or d["steps"]
    out = args.out or f"results/synthetic/{args.task}_{args.scale}.jsonl"
    if os.path.exists(out):
        os.remove(out)

    base = ModelConfig(vocab_size=d["vocab"], max_seq_len=seq_len, d_model=d["d_model"],
                       n_layers=d["n_layers"], n_heads=4)
    syn = SyntheticConfig(task=args.task, seq_len=seq_len, vocab_size=d["vocab"],
                          num_kv_pairs=d["kv"], num_queries=d["kv"], seed=args.seed)
    tc = TrainConfig(steps=steps, batch_size=d["batch"], lr=1e-3, warmup=max(50, steps // 20),
                     eval_every=steps, eval_iters=40, seed=args.seed, device=args.device)

    points = [SweepPoint(mixer=m, knob_value=v)
              for m in args.mixers for v in SWEEPS[args.scale][m]]
    print(f"Running {len(points)} points -> {out}")
    run_sweep(base, syn, tc, points, out)
    print(f"Done. Results in {out}")


if __name__ == "__main__":
    main()
