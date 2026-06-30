"""Compute/memory driver: analytic cache curves + prefill/decode latency and
throughput vs. context length for each mixer. Writes results/efficiency/<scale>.json.

The analytic cache curve is hardware independent (the cleanest memory metric);
latency/throughput are wall-clock and reflect whatever device you run on.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from seqmix.config import ModelConfig  # noqa: E402
from seqmix.eval.efficiency import profile_model  # noqa: E402
from seqmix.utils import get_device, save_json  # noqa: E402

MIXER_KWARGS = {"mha": {}, "swa": {"window": 128}, "mla": {"d_c": 128}, "mamba": {"d_state": 16}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="demo", choices=["demo", "full"])
    ap.add_argument("--mixers", nargs="+", default=["mha", "swa", "mla", "mamba"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.scale == "demo":
        seq_lens = [64, 128, 256, 512]
        d_model, n_layers, gen_len = 128, 3, 16
    else:
        seq_lens = [512, 1024, 2048, 4096, 8192]
        d_model, n_layers, gen_len = 512, 8, 64

    device = get_device(args.device)
    out = args.out or f"results/efficiency/{args.scale}.json"
    results = []
    for mixer in args.mixers:
        mc = ModelConfig(vocab_size=256, max_seq_len=max(seq_lens), d_model=d_model,
                         n_layers=n_layers, n_heads=4, mixer=mixer,
                         mixer_kwargs=dict(MIXER_KWARGS[mixer]))
        prof = profile_model(mc, seq_lens, batch_size=1, gen_len=gen_len, device=device)
        results.append(prof)
        last = prof["analytic"][-1]
        print(f"{mixer:6s} cache@{last['seq_len']}={last['state_bytes']} bytes  "
              f"prefill@{seq_lens[-1]}={prof['prefill'][-1]['prefill_s']*1e3:.1f} ms  "
              f"decode={prof['decode'][-1]['decode_tok_per_s']:.1f} tok/s")
    save_json(out, {"device": device.type, "seq_lens": seq_lens, "models": results})
    print(f"Done. Results in {out}")


if __name__ == "__main__":
    main()
