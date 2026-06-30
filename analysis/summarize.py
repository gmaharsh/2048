"""Print compact summary tables from the results/ directory.

Useful for filling in the paper's results section and for a quick textual view
of every phase without opening the figures.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def summarize_frontier(task, scale):
    rows = load_jsonl(os.path.join(ROOT, "results", "synthetic", f"{task}_{scale}.jsonl"))
    if not rows:
        return
    print(f"\n## Frontier: {task} ({scale})")
    print(f"{'mixer':6} {'knob':8} {'value':>6} {'acc':>6} {'state_B':>9} {'params':>9}")
    for r in rows:
        print(f"{r['mixer']:6} {r['knob']:8} {r['knob_value']:>6} {r['accuracy']:>6.3f} "
              f"{r['state_bytes']:>9} {r['params']:>9}")


def summarize_lm(scale, variant=""):
    suffix = f"_{variant}" if variant else ""
    rows = load_jsonl(os.path.join(ROOT, "results", "lm", f"{scale}{suffix}.jsonl"))
    if not rows:
        return
    label = variant or "synthetic"
    print(f"\n## Language modeling ({scale}, {label})")
    print(f"{'mixer':6} {'val_ppl':>8} {'params':>9} {'cache_B':>9}")
    for r in rows:
        print(f"{r['mixer']:6} {r['val_ppl']:>8.3f} {r['params']:>9} {r['state_bytes']:>9}")


def summarize_efficiency(scale):
    data = load_json(os.path.join(ROOT, "results", "efficiency", f"{scale}.json"))
    if not data:
        return
    print(f"\n## Efficiency ({scale}, device={data['device']})")
    print(f"{'mixer':6} {'cache@max_B':>12} {'prefill_ms':>11} {'decode_tok/s':>13}")
    for m in data["models"]:
        cache = m["analytic"][-1]["state_bytes"]
        pf = m["prefill"][-1]["prefill_s"] * 1e3
        dec = m["decode"][-1]["decode_tok_per_s"]
        print(f"{m['mixer']:6} {cache:>12} {pf:>11.1f} {dec:>13.1f}")


def summarize_longctx(scale):
    rows = load_jsonl(os.path.join(ROOT, "results", "longctx", f"{scale}.jsonl"))
    if not rows:
        return
    by = defaultdict(list)
    for r in rows:
        by[r["mixer"]].append(r["accuracy"])
    print(f"\n## Long-context passkey ({scale}) — mean accuracy over depths")
    for mixer, accs in by.items():
        print(f"{mixer:6} {sum(accs)/len(accs):.3f}")


def summarize_mech(scale):
    data = load_json(os.path.join(ROOT, "results", "mech", f"{scale}.json"))
    if not data:
        return
    print(f"\n## Mechanistic ({scale})")
    for mixer, info in data["models"].items():
        drops = [round(p["drop"], 3) for p in info["layer_knockout"]["per_layer"]]
        print(f"{mixer:6} acc={info['final_accuracy']:.3f} knockout_drops={drops}")
    if "mamba_conv_ablation" in data:
        print("mamba conv ablation:", data["mamba_conv_ablation"])


def main():
    scale = sys.argv[1] if len(sys.argv) > 1 else "demo"
    for task in ["mqar", "selective_copy", "induction"]:
        summarize_frontier(task, scale)
    summarize_lm(scale)
    summarize_lm(scale, variant="tinystories")
    summarize_efficiency(scale)
    summarize_longctx(scale)
    summarize_mech(scale)


if __name__ == "__main__":
    main()
