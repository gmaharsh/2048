# SeqMixer: MLA vs MHA vs SWA vs Mamba on the Recall–Memory–Compute Frontier

A controlled, reproducible study that swaps **only** the sequence-mixing layer
inside a fixed decoder backbone and compares four mixers along three axes:

| Mixer | What it is | Memory knob |
|-------|------------|-------------|
| **MHA** | dense multi-head attention (the O(n²) control) | `n_heads` |
| **MLA** | Multi-head *Latent* Attention (DeepSeek-V2): low-rank KV-cache compression + decoupled RoPE | `d_c` (latent dim) |
| **SWA** | Sliding-Window Attention (local receptive field) | `window` |
| **Mamba** | selective state-space model (subquadratic recurrence, fixed state) | `d_state` |

The goal is a workshop-paper-grade comparison: **recall quality** vs. **memory
(KV-cache / state bytes per token)** vs. **compute (FLOPs, latency, throughput)**,
plus a **mechanistic explanation** of *why* each mixer lands where it does.

## Why this is more than a benchmark

Plain leaderboard comparisons of these mixers exist. The under-explored gap is
placing **MLA** on the controlled recall–memory–compute frontier *next to* Mamba
and sliding-window attention at small, matched scale, with a mechanistic account
of the failure modes. That is the contribution here. See `paper/` for the draft.

## Run on Google Colab (recommended for GPU)

The pure-PyTorch reference Mamba is slow on CPU; a GPU fixes this. Open
[`notebooks/colab_quickstart.ipynb`](notebooks/colab_quickstart.ipynb) in Colab
(set **Runtime -> GPU**) and run top to bottom. It clones the repo, installs
deps, runs the phases at `--scale full --device cuda`, and displays the figures
inline. To load it: in Colab, **File -> Open notebook -> GitHub**, paste the repo
URL, and pick the notebook (or upload the `.ipynb`).

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -r requirements.txt
```

The reference Mamba is pure PyTorch and runs on CPU. For fast GPU Mamba,
optionally `pip install mamba-ssm causal-conv1d` (CUDA only); the analytic
state-size accounting and outputs are unchanged.

## Reproduce

Everything runs at two scales: `demo` (CPU, minutes) and `full` (GPU).

```bash
# end-to-end demo + figures
bash experiments/run_demo_all.sh

# or individual phases
PYTHONPATH=src python experiments/run_synthetic.py  --task mqar --scale demo   # Phase 1 frontier
PYTHONPATH=src python experiments/run_lm.py         --scale demo               # Phase 2 perplexity (synthetic corpus)
PYTHONPATH=src python experiments/run_lm.py         --scale demo --dataset tinystories  # Phase 2 on real TinyStories text
PYTHONPATH=src python experiments/run_efficiency.py --scale demo               # compute/memory
PYTHONPATH=src python experiments/run_longctx.py    --scale demo               # Phase 3 passkey
PYTHONPATH=src python experiments/run_mechanistic.py --scale demo              # Phase 4 interventions
PYTHONPATH=src python analysis/plot.py demo                                    # figures -> paper/figures/
```

Scale up on a GPU with `--scale full --device cuda`.

## Layout

```
src/seqmix/
  config.py            ModelConfig / TrainConfig
  model.py             shared decoder backbone (only the mixer changes)
  rope.py              rotary embeddings (+ MLA decoupled RoPE)
  mixers/              mha.py, mla.py, mamba.py (+ SWA), registry, base interface
  data/                synthetic.py (MQAR, selective-copy, induction, passkey),
                       text.py (synthetic recall corpus, local files, TinyStories loader)
  train.py             AdamW + cosine training loop
  eval/                synthetic.py (frontier sweep), lm.py, efficiency.py, longctx.py
  mech/                interventions.py (layer knockout, residual patching)
notebooks/             colab_quickstart.ipynb (GPU-ready end-to-end run)
experiments/           one driver per phase + run_demo_all.sh
analysis/plot.py       builds all figures from results/
tests/                 correctness tests (decode equivalence, causality, window, ...)
paper/                 workshop paper draft + figures
```

## Tests

```bash
PYTHONPATH=src python -m pytest -q
```

Covers decode/parallel equivalence, causality, the SWA receptive field, MLA
cache-footprint accounting, the Mamba short-conv ablation, task generators, and
a fast end-to-end learning check.

## Status / scaling notes

The committed `demo` results are CPU-scale smoke runs that establish the
framework and the qualitative ordering. Quantitative claims for a paper should
use `--scale full` on a GPU (and the official Mamba kernel for speed). The
analytic memory curves are exact and hardware independent at any scale.
