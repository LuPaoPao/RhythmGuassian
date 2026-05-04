"""Network modules for RhythmGaussian.

Exposes the main entry points so callers can write::

    from models import BaseNet, GaussianRenderer

instead of touching the individual files.
"""
from .model import BaseNet
from .gs import GaussianRenderer

__all__ = ["BaseNet", "GaussianRenderer"]
