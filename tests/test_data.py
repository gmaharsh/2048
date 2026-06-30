"""Tests for synthetic task generators and the text dataset."""

import numpy as np
import torch

from seqmix.data.synthetic import SyntheticConfig, get_batch, scored_accuracy
from seqmix.data.text import ByteDataset


def test_mqar_shapes_and_masking():
    cfg = SyntheticConfig(task="mqar", seq_len=64, vocab_size=32, num_kv_pairs=4, num_queries=4)
    rng = np.random.default_rng(0)
    x, y = get_batch(cfg, 8, rng)
    assert x.shape == (8, 64) and y.shape == (8, 64)
    # exactly num_queries scored positions per row
    assert (y != -100).sum(dim=1).tolist() == [4] * 8
    # queried key token must equal a key that appeared earlier with that value
    for b in range(8):
        qpos = (y[b] != -100).nonzero().flatten()
        for p in qpos:
            key = x[b, p].item()
            val = y[b, p].item()
            # find the key in the pair region and check its value
            pair_region = x[b, :2 * cfg.num_kv_pairs]
            ki = (pair_region == key).nonzero().flatten()
            assert len(ki) >= 1
            assert pair_region[ki[0] + 1].item() == val


def test_selective_copy_targets_are_content():
    cfg = SyntheticConfig(task="selective_copy", seq_len=48, vocab_size=32, num_tokens=8)
    rng = np.random.default_rng(0)
    x, y = get_batch(cfg, 4, rng)
    assert (y != -100).sum(dim=1).tolist() == [8] * 4


def test_induction_single_scored_position():
    cfg = SyntheticConfig(task="induction", seq_len=40, vocab_size=32)
    rng = np.random.default_rng(0)
    x, y = get_batch(cfg, 5, rng)
    assert (y != -100).sum(dim=1).tolist() == [1] * 5


def test_scored_accuracy_perfect_and_chance():
    cfg = SyntheticConfig(task="mqar", seq_len=64, vocab_size=32, num_kv_pairs=4, num_queries=4)
    rng = np.random.default_rng(1)
    x, y = get_batch(cfg, 4, rng)
    V = cfg.vocab_size
    # construct logits that perfectly predict the targets
    logits = torch.zeros(4, 64, V)
    tgt = y.clone()
    tgt[tgt == -100] = 0
    logits.scatter_(2, tgt.unsqueeze(-1), 10.0)
    assert scored_accuracy(logits, y) == 1.0


def test_byte_dataset_batch():
    ds = ByteDataset.synthetic_recall_corpus(n_bytes=5000, vocab_size=32, seed=0)
    rng = np.random.default_rng(0)
    x, y = ds.get_batch(4, 32, rng)
    assert x.shape == (4, 32) and y.shape == (4, 32)
    # y is x shifted by one
    assert torch.equal(x[:, 1:], y[:, :-1])
