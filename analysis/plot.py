"""Generate all paper figures from the results/ directory.

Each function is guarded by file existence so you can run it after any subset
of phases. Figures are written to paper/figures/.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG_DIR = os.path.join(ROOT, "paper", "figures")
MIXER_COLORS = {"mha": "#1f77b4", "mla": "#2ca02c", "swa": "#ff7f0e", "mamba": "#d62728"}
MIXER_MARKERS = {"mha": "o", "mla": "s", "swa": "^", "mamba": "D"}


def _load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _ensure_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def plot_frontier(task: str, scale: str = "demo"):
    rows = _load_jsonl(os.path.join(ROOT, "results", "synthetic", f"{task}_{scale}.jsonl"))
    if not rows:
        return None
    by_mixer = defaultdict(list)
    for r in rows:
        by_mixer[r["mixer"]].append(r)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for mixer, pts in by_mixer.items():
        pts = sorted(pts, key=lambda p: p["state_bytes"])
        ax.plot([p["state_bytes"] for p in pts], [p["accuracy"] for p in pts],
                marker=MIXER_MARKERS.get(mixer, "o"), color=MIXER_COLORS.get(mixer),
                label=mixer.upper(), linewidth=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("Cache / state bytes per sequence (log)")
    ax.set_ylabel(f"{task.upper()} accuracy")
    ax.set_title(f"Recall-memory frontier: {task.upper()} ({scale})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"frontier_{task}_{scale}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_lm(scale: str = "demo", variant: str = ""):
    suffix = f"_{variant}" if variant else ""
    rows = _load_jsonl(os.path.join(ROOT, "results", "lm", f"{scale}{suffix}.jsonl"))
    if not rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for r in rows:
        c = MIXER_COLORS.get(r["mixer"])
        axes[0].scatter(r["params"], r["val_ppl"], color=c, label=r["mixer"].upper(),
                        marker=MIXER_MARKERS.get(r["mixer"], "o"), s=60)
        axes[1].scatter(r["state_bytes"], r["val_ppl"], color=c,
                        marker=MIXER_MARKERS.get(r["mixer"], "o"), s=60)
    axes[0].set_xlabel("Non-embedding params")
    axes[0].set_ylabel("Val perplexity")
    axes[0].set_title("Quality vs. compute")
    axes[1].set_xlabel("Cache/state bytes per sequence")
    axes[1].set_title("Quality vs. memory")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    title = f"_{variant}" if variant else ""
    fig.suptitle(f"Language modeling{(' (' + variant + ')') if variant else ''}", y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"lm_{scale}{title}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_efficiency(scale: str = "demo"):
    data = _load_json(os.path.join(ROOT, "results", "efficiency", f"{scale}.json"))
    if not data:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for m in data["models"]:
        mixer = m["mixer"]
        c = MIXER_COLORS.get(mixer)
        sl = [a["seq_len"] for a in m["analytic"]]
        kb = [a["state_bytes"] / 1024 for a in m["analytic"]]
        axes[0].plot(sl, kb, marker=MIXER_MARKERS.get(mixer, "o"), color=c, label=mixer.upper())
        dl = [(d["prompt_len"], d["decode_tok_per_s"]) for d in m["decode"]]
        axes[1].plot([p[0] for p in dl], [p[1] for p in dl],
                     marker=MIXER_MARKERS.get(mixer, "o"), color=c, label=mixer.upper())
    axes[0].set_xlabel("Context length")
    axes[0].set_ylabel("Cache/state (KiB per sequence)")
    axes[0].set_title("Memory growth vs. context")
    axes[1].set_xlabel("Prompt length")
    axes[1].set_ylabel("Decode throughput (tok/s)")
    axes[1].set_title(f"Decode throughput ({data['device']})")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"efficiency_{scale}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_longctx(scale: str = "demo"):
    rows = _load_jsonl(os.path.join(ROOT, "results", "longctx", f"{scale}.jsonl"))
    if not rows:
        return None
    by_mixer = defaultdict(list)
    for r in rows:
        by_mixer[r["mixer"]].append(r)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for mixer, pts in by_mixer.items():
        pts = sorted(pts, key=lambda p: p["depth_frac"])
        ax.plot([p["depth_frac"] for p in pts], [p["accuracy"] for p in pts],
                marker=MIXER_MARKERS.get(mixer, "o"), color=MIXER_COLORS.get(mixer),
                label=mixer.upper(), linewidth=1.8)
    ax.set_xlabel("Needle depth (fraction of context)")
    ax.set_ylabel("Passkey recall accuracy")
    ax.set_title(f"Long-context retrieval vs. depth ({scale})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"longctx_{scale}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_mechanistic(scale: str = "demo"):
    data = _load_json(os.path.join(ROOT, "results", "mech", f"{scale}.json"))
    if not data:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for mixer, color in [("mha", MIXER_COLORS["mha"]), ("mamba", MIXER_COLORS["mamba"])]:
        if mixer not in data["models"]:
            continue
        ko = data["models"][mixer]["layer_knockout"]["per_layer"]
        layers = [p["layer"] for p in ko]
        drops = [p["drop"] for p in ko]
        axes[0].bar([l + (0.0 if mixer == "mha" else 0.35) for l in layers], drops,
                    width=0.35, color=color, label=mixer.upper())
    axes[0].set_xlabel("Layer knocked out")
    axes[0].set_ylabel("Accuracy drop")
    axes[0].set_title("Layer knockout (which layers carry recall)")
    axes[0].legend(fontsize=8)

    abl = data.get("mamba_conv_ablation", {})
    if abl:
        keys = list(abl.keys())
        axes[1].bar(keys, [abl[k] for k in keys], color=MIXER_COLORS["mamba"])
        axes[1].set_ylabel("MQAR accuracy")
        axes[1].set_title("Mamba short-conv ablation")
    for ax in axes:
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"mechanistic_{scale}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    scale = sys.argv[1] if len(sys.argv) > 1 else "demo"
    _ensure_dir()
    made = []
    for task in ["mqar", "selective_copy", "induction"]:
        made.append(plot_frontier(task, scale))
    made.append(plot_lm(scale))
    made.append(plot_lm(scale, variant="tinystories"))
    made.append(plot_efficiency(scale))
    made.append(plot_longctx(scale))
    made.append(plot_mechanistic(scale))
    for m in made:
        if m:
            print("wrote", m)


if __name__ == "__main__":
    main()
