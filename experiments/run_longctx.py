"""Phase 3 driver: train each mixer on the passkey (needle-in-a-haystack) task,
then measure recall accuracy as a function of needle depth. Writes
results/longctx/<scale>.jsonl.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from seqmix.config import ModelConfig, TrainConfig  # noqa: E402
from seqmix.eval.longctx import passkey_accuracy_by_depth, train_passkey_model  # noqa: E402
from seqmix.utils import append_jsonl  # noqa: E402

MIXER_KWARGS = {"mha": {}, "swa": {"window": 32}, "mla": {"d_c": 64}, "mamba": {"d_state": 16}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="demo", choices=["demo", "full"])
    ap.add_argument("--mixers", nargs="+", default=["mha", "swa", "mla", "mamba"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.scale == "demo":
        seq_len, vocab, steps, d_model, n_layers = 64, 48, 700, 96, 2
    else:
        seq_len, vocab, steps, d_model, n_layers = 1024, 64, 8000, 256, 4

    depths = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]
    out = args.out or f"results/longctx/{args.scale}.jsonl"
    if os.path.exists(out):
        os.remove(out)

    for mixer in args.mixers:
        mc = ModelConfig(vocab_size=vocab, max_seq_len=seq_len, d_model=d_model,
                         n_layers=n_layers, n_heads=4, mixer=mixer,
                         mixer_kwargs=dict(MIXER_KWARGS[mixer]))
        tc = TrainConfig(steps=steps, batch_size=32, lr=1e-3, warmup=max(50, steps // 20),
                         eval_every=steps, eval_iters=40, seed=args.seed, device=args.device)
        model = train_passkey_model(mc, seq_len, vocab, tc)
        by_depth = passkey_accuracy_by_depth(model, seq_len, vocab, depths, device=args.device)
        for row in by_depth:
            append_jsonl(out, {"mixer": mixer, "seq_len": seq_len, **row})
        avg = sum(r["accuracy"] for r in by_depth) / len(by_depth)
        print(f"{mixer:6s} mean passkey acc over depths = {avg:.3f}")
    print(f"Done. Results in {out}")


if __name__ == "__main__":
    main()
