"""Dataset loading and per-target trainer / evaluation scripts.

Re-exports the core dataset class and helpers for clean imports::

    from datasets import Data_DG, getIndex

The directory also bundles runnable scripts that share the same
project layout:

* ``datasets/<MMPD|MR|Phys|UCLA|VV100>.py`` — per-target training variants;
  launch via ``python -m datasets.<name>`` from the project root.
* ``datasets/Eval.py``    — per-video HR aggregation + final metrics.
* ``datasets/dataSort.py`` — sort raw BVP/HR result mats into per-subject files.
"""
from .MyDataset import CrossValidation, Data_DG, getIndex

__all__ = ["Data_DG", "getIndex", "CrossValidation"]
