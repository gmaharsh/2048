"""SeqMixer: a controlled study of MLA, MHA, SWA, and Mamba sequence mixers.

The package provides a single shared decoder backbone in which only the
sequence-mixing layer is swapped, so that quality, memory, and compute can be
compared under matched budgets (param-, FLOP-, and KV-cache/state-matched).
"""

from .config import ModelConfig
from .model import LanguageModel
from .mixers import build_mixer, MIXER_REGISTRY

__all__ = ["ModelConfig", "LanguageModel", "build_mixer", "MIXER_REGISTRY"]
