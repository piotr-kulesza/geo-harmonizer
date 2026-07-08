"""ComBat batch correction, wrapping an existing implementation. [Day 2 end / 3 — Epic 3.3]

Wraps inmoose's ``pycombat_norm`` (imported lazily) to remove the batch signal —
the source-accession effect — from a merged gene x sample matrix. Per CLAUDE.md
this is existing ComBat only; we do not implement a new batch-correction method.

This function ALWAYS corrects when called. "OFF by default" is an app-layer
choice (the web/MCP wrapper decides whether to call it), not something this core
function knows about — iron rule.

Data convention: ``merged`` is features (genes, rows) x samples (GSM columns)
from :func:`core.merge.merge`; ``batch`` is a per-sample Series of source
accession aligned to ``merged.columns``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def combat(merged: pd.DataFrame, batch: pd.Series) -> pd.DataFrame:
    """Batch-correct a merged gene x sample matrix given per-sample batch labels.

    Args:
        merged: genes (rows) x samples (GSM columns) expression matrix.
        batch: per-sample Series of source-accession labels. Aligned to
            ``merged.columns`` (reindexed defensively).

    Returns:
        A batch-corrected DataFrame with the same columns as the input and the
        same gene index minus any rows dropped for containing non-finite values.

    Raises:
        ValueError: if fewer than 2 distinct batches, any batch has < 2 samples,
            or the whole matrix is all-NaN.
    """
    # Align batch labels to the matrix's sample order.
    batch = batch.reindex(merged.columns)
    if batch.isna().any():
        missing = list(batch.index[batch.isna()])
        raise ValueError(
            f"No batch label for samples: {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}."
        )

    # Precondition: >= 2 distinct batches.
    counts = batch.value_counts()
    if counts.size < 2:
        raise ValueError(
            f"ComBat needs >=2 distinct batches, got {counts.size} "
            f"({list(counts.index)})."
        )

    # Precondition: every batch has >= 2 samples.
    singletons = counts[counts < 2]
    if not singletons.empty:
        raise ValueError(
            "ComBat needs >=2 samples per batch; these have too few: "
            f"{singletons.to_dict()}."
        )

    # Precondition: not entirely NaN.
    if merged.size == 0 or not np.isfinite(merged.to_numpy(dtype=float)).any():
        raise ValueError("Cannot run ComBat on an empty or all-NaN matrix.")

    # Drop gene rows with any non-finite value (ComBat needs a dense matrix).
    finite_rows = np.isfinite(merged.to_numpy(dtype=float)).all(axis=1)
    n_dropped = int((~finite_rows).sum())
    if n_dropped:
        logger.info(
            "combat: dropping %d/%d gene rows with non-finite values.",
            n_dropped,
            merged.shape[0],
        )
    clean = merged.loc[finite_rows]
    if clean.empty:
        raise ValueError("No gene rows remain after dropping non-finite values.")

    # inmoose is a heavy dep; import lazily so core stays importable offline.
    from inmoose.pycombat import pycombat_norm

    batch_list = batch.loc[clean.columns].tolist()
    corrected = pycombat_norm(clean, batch_list)

    # Normalize back to a DataFrame with the cleaned input's labels/shape.
    result = pd.DataFrame(
        np.asarray(corrected),
        index=clean.index,
        columns=clean.columns,
    )
    result.index.name = merged.index.name or "gene"
    result.columns.name = merged.columns.name or "sample"
    logger.info(
        "combat: corrected %d genes x %d samples across %d batches.",
        result.shape[0],
        result.shape[1],
        counts.size,
    )
    return result
