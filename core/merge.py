"""Merge per-series gene x sample matrices onto a shared gene set. [Day 2 end]

Stub only today. See TICKETS.md Epic 3: ``merge`` intersects genes across
series and returns the combined matrix plus a per-sample batch label (its source
accession — what ComBat removes and the PCA colours by).
"""

from __future__ import annotations

import pandas as pd


def merge(matrices: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.Series]:
    """Merge {accession: gene x sample matrix} -> (merged matrix, batch labels)."""
    raise NotImplementedError("merge is implemented at Day 2 end (Epic 3).")
