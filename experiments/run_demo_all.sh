#!/usr/bin/env bash
# Run the full demo-scale study end-to-end and generate all figures.
# CPU-friendly; the same scripts accept --scale full --device cuda for GPU runs.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src

echo "=== Phase 1: synthetic recall-memory frontier (MQAR) ==="
python3 experiments/run_synthetic.py --task mqar --scale demo

echo "=== Phase 1b: induction frontier ==="
python3 experiments/run_synthetic.py --task induction --scale demo

echo "=== Phase 2: language modeling on synthetic recall corpus ==="
python3 experiments/run_lm.py --scale demo

echo "=== Phase 2b: language modeling on TinyStories ==="
python3 experiments/run_lm.py --scale demo --dataset tinystories

echo "=== compute/memory profiling ==="
python3 experiments/run_efficiency.py --scale demo

echo "=== Phase 3: long-context passkey by depth ==="
python3 experiments/run_longctx.py --scale demo

echo "=== Phase 4: mechanistic interventions ==="
python3 experiments/run_mechanistic.py --scale demo

echo "=== Generating figures ==="
python3 analysis/plot.py demo

echo "=== DEMO COMPLETE ==="
