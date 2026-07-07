"""Probe->gene mapping and log2 normalization. [Day 2]

Stubs only today. See TICKETS.md Epic 2 and CLAUDE.md for the settled behavior:
- ``map_probes`` collapses many-probes-to-one-gene, drops unmapped features.
- ``to_log2`` carries a double-log2 guard (max < ~30 -> already logged, skip).
"""

from __future__ import annotations

import pandas as pd


def map_probes(
    expr: pd.DataFrame,
    platform_id: str,
    collapse: str = "max",
) -> pd.DataFrame:
    """Map probe ids to gene symbols, returning a gene x sample matrix. [Day 2]"""
    raise NotImplementedError("map_probes is implemented on Day 2 (Epic 2).")


def to_log2(expr: pd.DataFrame) -> pd.DataFrame:
    """log2-transform an expression matrix, guarding against double-log2. [Day 2]"""
    raise NotImplementedError("to_log2 is implemented on Day 2 (Epic 2).")
