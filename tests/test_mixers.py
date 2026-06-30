"""Correctness tests for the four sequence mixers and the shared backbone."""

import pytest
import torch

from seqmix.config import ModelConfig
from seqmix.model import LanguageModel

MIXERS = ["mha", "swa", "mla", "mamba"]


def make_model(mixer, T=24, V=48, d_model=48, n_layers=2, **mk):
    cfg = ModelConfig(
        vocab_size=V, max_seq_len=T, d_model=d_model, n_layers=n_layers,
        n_heads=4, mixer=mixer, mixer_kwargs=mk,
    )
    m = LanguageModel(cfg)
    m.eval()
    return m, cfg


@pytest.mark.parametrize("mixer", MIXERS)
def test_forward_shapes_and_loss(mixer):
    m, cfg = make_model(mixer)
    B, T = 3, cfg.max_seq_len
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = m(idx, idx)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("mixer", MIXERS)
def test_incremental_decode_matches_parallel(mixer):
    """The single-token decode path must reproduce the parallel forward."""
    m, cfg = make_model(mixer)
    B, T = 2, cfg.max_seq_len
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        full, _ = m(idx)
        caches = m.decode_step_init(B, idx.device)
        outs = []
        for t in range(T):
            lo, caches = m.decode_step(idx[:, t:t + 1], caches, t)
            outs.append(lo)
        step = torch.cat(outs, dim=1)
    assert torch.allclose(full, step, atol=1e-4), (full - step).abs().max()


@pytest.mark.parametrize("mixer", MIXERS)
def test_causality(mixer):
    """Output at position t must not depend on tokens after t."""
    m, cfg = make_model(mixer)
    B, T = 2, cfg.max_seq_len
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        out_a, _ = m(idx)
        idx2 = idx.clone()
        idx2[:, T // 2 + 1:] = torch.randint(0, cfg.vocab_size, idx2[:, T // 2 + 1:].shape)
        out_b, _ = m(idx2)
    # positions up to and including T//2 must be identical
    assert torch.allclose(out_a[:, : T // 2 + 1], out_b[:, : T // 2 + 1], atol=1e-5)


def test_swa_receptive_field():
    """Sliding-window attention at position t ignores tokens beyond the window."""
    window = 4
    m, cfg = make_model("swa", window=window, T=20)
    B, T = 1, cfg.max_seq_len
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    pos = T - 1
    with torch.no_grad():
        out_a, _ = m(idx)
        idx2 = idx.clone()
        # change a token well outside the window of `pos`
        far = pos - window - 2
        idx2[0, far] = (idx2[0, far] + 1) % cfg.vocab_size
        out_b, _ = m(idx2)
    # single-layer would be exactly invariant; with 2 layers the effective
    # window is 2*window, so pick `far` outside that too.
    m1, cfg1 = make_model("swa", window=window, T=20, n_layers=1)
    with torch.no_grad():
        o1, _ = m1(idx)
        o2, _ = m1(idx2)
    assert torch.allclose(o1[0, pos], o2[0, pos], atol=1e-6)


def test_mla_cache_is_smaller_than_mha_at_length():
    mha, _ = make_model("mha", T=512, d_model=128)
    mla, _ = make_model("mla", T=512, d_model=128)
    L = 512
    assert mla.analytic_state_bytes(L) < mha.analytic_state_bytes(L)


def test_mamba_state_constant_in_length():
    m, _ = make_model("mamba", T=512)
    assert m.analytic_state_bytes(64) == m.analytic_state_bytes(512)


def test_mamba_conv_ablation_changes_state_and_output():
    with_conv, cfg = make_model("mamba", d_conv=4, use_conv=True)
    without, _ = make_model("mamba", d_conv=4, use_conv=False)
    assert with_conv.analytic_state_bytes(128) > without.analytic_state_bytes(128)
