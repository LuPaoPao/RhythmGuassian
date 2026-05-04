"""Dataset loading utilities + per-dataset trainer variants.

Re-exports the core dataset class and helpers so callers can write::

    from datasets import Data_DG, getIndex

Per-dataset trainer variants (MMPD, MR, Phys, UCLA, VV100) live alongside as
runnable modules; launch them with ``python -m datasets.<name>`` from the
project root.
"""
from .MyDataset import Data_DG, getIndex, CrossValidation

__all__ = ["Data_DG", "getIndex", "CrossValidation"]
