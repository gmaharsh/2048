"""Phase 2 driver: train tiny LMs (one per mixer, identical backbone shape) and
report validation perplexity alongside parameter count and analytic cache size.

Reports are written to results/lm/<scale>.jsonl. Because mixers differ slightly
in parameter count at a fixed backbone shape, we record params and cache-bytes
with every point so the analysis can present ppl-vs-params (compute-matched)
and ppl-vs-cache (memory-matched) Pareto views.

By default it trains on an offline synthetic recall corpus so it runs without
internet; pass --text-file to use a real corpus (e.g. enwik8).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from seqmix.config import ModelConfig, TrainConfig  # noqa: E402
from seqmix.data.text import ByteDataset  # noqa: E402
from seqmix.train import train_lm  # noqa: E402
from seqmix.utils import append_jsonl  # noqa: E402

SCALE = {
    "demo": dict(d_model=128, n_layers=3, seq_len=96, steps=1200, batch=32, n_bytes=300_000, vocab=64),
    "full": dict(d_model=512, n_layers=8, seq_len=512, steps=20000, batch=32, n_bytes=50_000_000, vocab=256),
}

MIXER_KWARGS = {"mha": {}, "swa": {"window": 32}, "mla": {"d_c": 64}, "mamba": {"d_state": 16}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="demo", choices=["demo", "full"])
    ap.add_argument("--mixers", nargs="+", default=["mha", "swa", "mla", "mamba"])
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = SCALE[args.scale]
    out = args.out or f"results/lm/{args.scale}.jsonl"
    if os.path.exists(out):
        os.remove(out)

    if args.text_file:
        dataset = ByteDataset.from_file(args.text_file, max_bytes=d["n_bytes"])
        vocab = 256
    else:
        dataset = ByteDataset.synthetic_recall_corpus(n_bytes=d["n_bytes"], vocab_size=d["vocab"], seed=args.seed)
        vocab = d["vocab"]

    for mixer in args.mixers:
        mc = ModelConfig(vocab_size=vocab, max_seq_len=d["seq_len"], d_model=d["d_model"],
                         n_layers=d["n_layers"], n_heads=4, mixer=mixer,
                         mixer_kwargs=dict(MIXER_KWARGS[mixer]))
        tc = TrainConfig(steps=d["steps"], batch_size=d["batch"], seq_len=d["seq_len"],
                         lr=3e-4, warmup=max(100, d["steps"] // 20), eval_every=d["steps"],
                         eval_iters=40, seed=args.seed, device=args.device)
        res = train_lm(mc, dataset, tc)
        rec = {"mixer": mixer, "val_ppl": res["final_val_ppl"],
               "params": res["non_embedding_params"],
               "state_bytes": res["analytic_state_bytes"], "seq_len": d["seq_len"],
               "scale": args.scale, "seed": args.seed}
        append_jsonl(out, rec)
        print(f"{mixer:6s} ppl={rec['val_ppl']:.3f} params={rec['params']} "
              f"cache_bytes@{d['seq_len']}={rec['state_bytes']}")
    print(f"Done. Results in {out}")


if __name__ == "__main__":
    main()
