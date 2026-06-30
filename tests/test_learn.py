"""Fast functional test: attention learns MQAR well above chance quickly.

Kept tiny so it runs on CPU in a few seconds; it guards against pipeline
regressions (data + model + optimizer wired correctly), not against research
conclusions.
"""

from seqmix.config import ModelConfig, TrainConfig
from seqmix.data.synthetic import SyntheticConfig
from seqmix.train import train_synthetic


def test_mha_learns_mqar_above_chance():
    syn = SyntheticConfig(task="mqar", seq_len=32, vocab_size=24, num_kv_pairs=3, num_queries=3, seed=0)
    mc = ModelConfig(vocab_size=24, max_seq_len=32, d_model=64, n_layers=2, n_heads=4, mixer="mha")
    tc = TrainConfig(steps=300, batch_size=32, lr=2e-3, warmup=30, eval_every=300, eval_iters=20, seed=0)
    res = train_synthetic(mc, syn, tc)
    chance = 1.0 / 24
    assert res["final_accuracy"] > 5 * chance, res["final_accuracy"]
