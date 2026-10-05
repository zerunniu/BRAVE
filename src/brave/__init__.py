"""
BRAVE: Blockwise Reliability-Aware Variational EM

This package exposes BRAVE components for blockwise label aggregation.

Implementation entry point:
  - `BRAVE` in `src/brave/brave.py`
"""

from .brave import BRAVE
from .server import BRAVEServer, BRAVEUpdate
from .block import BRAVEBlock
from .diagnostics import WARNING_TEXT, compute_feedback_gap
from .partition import simulate_federated_scenario

__all__ = [
    "BRAVE",
    "BRAVEServer",
    "BRAVEUpdate",
    "BRAVEBlock",
    "WARNING_TEXT",
    "compute_feedback_gap",
    "simulate_federated_scenario",
]

