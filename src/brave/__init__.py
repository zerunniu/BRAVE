"""
BRAVE: Block-wise Structural Regularization via Controlled Evidence Feedback

This package exposes BRAVE components for blockwise label aggregation.

Implementation entry point:
  - `BRAVE` in `src/brave/brave.py`
"""

from .brave import BRAVE
from .global_update import BRAVEGlobalUpdater, BRAVEUpdate
from .block import BRAVEBlock
from .partition import partition_annotations

__all__ = [
    "BRAVE",
    "BRAVEGlobalUpdater",
    "BRAVEUpdate",
    "BRAVEBlock",
    "partition_annotations",
]
