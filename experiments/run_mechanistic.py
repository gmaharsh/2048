"""Phase 4 driver: train MHA and Mamba on MQAR, then run layer-knockout and
residual-patching interventions, plus the Mamba short-convolution ablation.
Writes results/mech/<scale>.json.

The contrast we expect: attention implements recall as a two-layer induction
mechanism (two specific layers jointly necessary), whereas Mamba performs
single-layer direct retrieval that critically depends on its short convolution.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from seqmix.config import ModelConfig, TrainConfig  # noqa: E402
from seqmix.data.synthetic import SyntheticConfig  # noqa: E402
from seqmix.mech import layer_knockout, residual_patch_recovery  # noqa: E402
from seqmix.train import train_synthetic  # noqa: E402
from seqmix.utils import save_json  # noqa: E402


def train_model(mixer, syn, vocab, seq_len, steps, device, mixer_kwargs=None):
    mc = ModelConfig(vocab_size=vocab, max_seq_len=seq_len, d_model=64, n_layers=2,
                     n_heads=4, mixer=mixer, mixer_kwargs=mixer_kwargs or {})
    tc = TrainConfig(steps=steps, batch_size=32, lr=1e-3, warmup=max(50, steps // 20),
                     eval_every=steps, eval_iters=40, seed=0, device=device)
    return train_synthetic(mc, syn, tc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="demo", choices=["demo", "full"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seq_len, vocab = (48, 32) if args.scale == "demo" else (128, 64)
    steps = 900 if args.scale == "demo" else 8000
    syn = SyntheticConfig(task="mqar", seq_len=seq_len, vocab_size=vocab,
                          num_kv_pairs=4, num_queries=4, seed=0)
    out = args.out or f"results/mech/{args.scale}.json"

    report = {"task": "mqar", "seq_len": seq_len, "models": {}}
    for mixer in ["mha", "mamba"]:
        res = train_model(mixer, syn, vocab, seq_len, steps, args.device)
        model = res["model"]
        report["models"][mixer] = {
            "final_accuracy": res["final_accuracy"],
            "layer_knockout": layer_knockout(model, syn, device=args.device),
            "residual_patch": residual_patch_recovery(model, syn, device=args.device),
        }
        print(f"{mixer}: acc={res['final_accuracy']:.3f} "
              f"knockout_drops={[round(p['drop'],3) for p in report['models'][mixer]['layer_knockout']['per_layer']]}")

    # Mamba short-conv ablation
    abl = {}
    for use_conv in [True, False]:
        res = train_model("mamba", syn, vocab, seq_len, steps, args.device,
                          mixer_kwargs={"use_conv": use_conv, "d_conv": 4})
        abl[f"use_conv={use_conv}"] = res["final_accuracy"]
        print(f"mamba use_conv={use_conv}: acc={res['final_accuracy']:.3f}")
    report["mamba_conv_ablation"] = abl

    save_json(out, report)
    print(f"Done. Results in {out}")


if __name__ == "__main__":
    main()
