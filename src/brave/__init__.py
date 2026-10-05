"""
BRAVE: Block-wise Structural Regularization via Controlled Evidence Feedback

This package exposes BRAVE components for blockwise label aggregation.

Implementation entry point:
  - `BRAVE` in `src/brave/brave.py`
"""

from .brave import BRAVE
from .global_update import BRAVEGlobalUpdater, BRAVEUpdate
from .block import BRAVEBlock
from .diagnostics import WARNING_TEXT, compute_feedback_gap
from .partition import partition_annotations

__all__ = [
    "BRAVE",
    "BRAVEGlobalUpdater",
    "BRAVEUpdate",
    "BRAVEBlock",
    "WARNING_TEXT",
    "compute_feedback_gap",
    "partition_annotations",
]

