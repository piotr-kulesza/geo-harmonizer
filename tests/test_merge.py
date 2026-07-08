"""Offline tests for core.merge — shared-gene intersection + batch labels."""

from __future__ import annotations

import pandas as pd
import pytest

from core.merge import merge


def _frame(genes, cols, fill):
    data = {c: [fill + i for i in range(len(genes))] for c in cols}
    return pd.DataFrame(data, index=genes)


def test_merge_intersects_genes_and_labels_batches():
    a = _frame(["GENEA", "GENEB", "GENEC"], ["GSM1", "GSM2"], 1.0)
    b = _frame(["GENEB", "GENEC", "GENED"], ["GSM3"], 10.0)

    merged, batch = merge({"GSE1": a, "GSE2": b})

    # Shared genes = intersection, sorted.
    assert list(merged.index) == ["GENEB", "GENEC"]
    assert merged.index.name == "gene"
    # All samples concatenated column-wise.
    assert list(merged.columns) == ["GSM1", "GSM2", "GSM3"]
    assert merged.shape == (2, 3)

    # Batch labels: each GSM -> its source accession.
    assert batch.name == "batch"
    assert batch["GSM1"] == "GSE1"
    assert batch["GSM2"] == "GSE1"
    assert batch["GSM3"] == "GSE2"
    assert list(batch.index) == list(merged.columns)


def test_merge_values_preserved_on_shared_genes():
    a = _frame(["GENEA", "GENEB"], ["GSM1"], 1.0)  # GENEA=1, GENEB=2
    b = _frame(["GENEB", "GENEA"], ["GSM2"], 5.0)  # GENEB=5, GENEA=6
    merged, _ = merge({"GSE1": a, "GSE2": b})

    assert merged.loc["GENEA", "GSM1"] == 1.0
    assert merged.loc["GENEB", "GSM1"] == 2.0
    # b's rows realigned to the shared-gene order, not b's original order.
    assert merged.loc["GENEA", "GSM2"] == 6.0
    assert merged.loc["GENEB", "GSM2"] == 5.0


def test_merge_empty_intersection_raises():
    a = _frame(["GENEA"], ["GSM1"], 1.0)
    b = _frame(["GENEZ"], ["GSM2"], 1.0)
    with pytest.raises(ValueError):
        merge({"GSE1": a, "GSE2": b})


def test_merge_empty_input_raises():
    with pytest.raises(ValueError):
        merge({})
