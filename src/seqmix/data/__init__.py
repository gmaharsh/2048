"""Data generators: synthetic capability tasks and text corpora."""

from .synthetic import SyntheticConfig, get_batch, scored_accuracy
from .text import ByteDataset

__all__ = ["SyntheticConfig", "get_batch", "scored_accuracy", "ByteDataset"]
