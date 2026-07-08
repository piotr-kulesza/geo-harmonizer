"""Merge per-series gene x sample matrices onto a shared gene set. [Day 2 end — Epic 3]

Takes the harmonized (gene-indexed) matrices from several GEO series and stacks
their samples side by side on the gene set they all share. The per-sample batch
label — a sample's source accession — is what ComBat later removes and the PCA
colours by (CLAUDE.md data conventions).

Iron rule: pure logic, no web/MCP/UI; nothing to stdout — return the data.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def merge(matrices: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.Series]:
    """Merge ``{accession: genes x samples}`` onto a shared gene set.

    Intersects gene symbols across all inputs, then concatenates their samples
    column-wise on that shared set.

    Args:
        matrices: mapping of accession -> genes (rows) x samples (GSM columns).
            Each matrix is expected to be gene-indexed (post ``map_probes``).

    Returns:
        A ``(merged, batch)`` tuple:
        - ``merged``: shared-genes (rows) x all-samples (GSM columns) DataFrame,
          rows sorted for determinism, index name ``"gene"``.
        - ``batch``: a Series indexed by GSM, value = source accession, name
          ``"batch"``.

    Raises:
        ValueError: if ``matrices`` is empty or the gene intersection is empty
            (shouldn't happen for same-modality series — fail loudly if it does).
    """
    if not matrices:
        raise ValueError("merge requires at least one matrix.")

    # Shared gene set = intersection of every matrix's index.
    shared: set[str] | None = None
    for matrix in matrices.values():
        genes = set(matrix.index)
        shared = genes if shared is None else (shared & genes)
    shared_genes = sorted(shared or set())

    if not shared_genes:
        raise ValueError(
            "No genes are shared across all input series — cannot merge. "
            "Check that every matrix was mapped to gene symbols first."
        )

    # Concatenate samples column-wise on the shared gene set, tracking batches.
    pieces: list[pd.DataFrame] = []
    batch_labels: dict[str, str] = {}
    for accession, matrix in matrices.items():
        piece = matrix.loc[shared_genes]
        pieces.append(piece)
        for gsm in piece.columns:
            batch_labels[str(gsm)] = accession

    merged = pd.concat(pieces, axis=1)
    merged.index.name = "gene"
    merged.columns.name = "sample"

    batch = pd.Series(batch_labels, name="batch")
    batch.index.name = "sample"
    # Keep batch aligned to the merged column order.
    batch = batch.reindex(merged.columns)

    logger.info(
        "merge: %d shared genes across %d series -> %d samples.",
        len(shared_genes),
        len(matrices),
        merged.shape[1],
    )
    return merged, batch
