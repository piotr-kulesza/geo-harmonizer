"""ComBat batch correction, wrapping an existing implementation. [Day 2 end / 3]

Stub only today. See CLAUDE.md: existing ComBat only (inmoose ``pycombat_norm`` /
neuroCombat), OFF by default, framed as an option. The scientifically hard part
is deliberately cut — we do not implement a new batch-correction method.
"""

from __future__ import annotations

import pandas as pd


def combat(merged: pd.DataFrame, batch: pd.Series) -> pd.DataFrame:
    """Batch-correct a merged gene x sample matrix given per-sample batch labels."""
    raise NotImplementedError("combat is implemented at Day 2 end / Day 3 (Epic 3).")
