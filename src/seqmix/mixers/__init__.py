"""Registry of sequence mixers keyed by name."""

from __future__ import annotations

from ..config import ModelConfig
from .base import SequenceMixer
from .mamba import MambaMixer
from .mha import MHAMixer, SWAMixer
from .mla import MLAMixer

MIXER_REGISTRY = {
    "mha": MHAMixer,
    "mla": MLAMixer,
    "swa": SWAMixer,
    "mamba": MambaMixer,
}


def build_mixer(config: ModelConfig) -> SequenceMixer:
    name = config.mixer.lower()
    if name not in MIXER_REGISTRY:
        raise ValueError(f"Unknown mixer '{name}'. Options: {sorted(MIXER_REGISTRY)}")
    return MIXER_REGISTRY[name](config)


__all__ = ["MIXER_REGISTRY", "build_mixer", "SequenceMixer",
           "MHAMixer", "SWAMixer", "MLAMixer", "MambaMixer"]
