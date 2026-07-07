"""Offline tests for core.fetch — no network, no real GEOparse.

We install a fake ``GEOparse`` module into ``sys.modules`` so ``fetch_gse`` runs
its full logic without touching the network. The fakes mimic just the surface
``fetch_gse`` uses: ``get_GEO``, ``gse.gsms``, ``gsm.metadata``, ``gsm.table``,
and ``gse.pivot_samples``.
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from core.fetch import (
    FetchResult,
    fetch_gse,
    load_matrix_from_file,
    _build_metadata_frame,
    _build_expression_matrix,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeGSM:
    """Minimal stand-in for a GEOparse GSM sample."""

    def __init__(self, metadata: dict, table: pd.DataFrame | None):
        self.metadata = metadata
        self.table = table


class FakeGSE:
    """Minimal stand-in for a GEOparse GSE series."""

    def __init__(self, gsms: dict[str, FakeGSM]):
        self.gsms = gsms

    def pivot_samples(self, value_column: str) -> pd.DataFrame:
        """Build a features x samples frame from each sample's table."""
        series = {}
        for gsm_id, gsm in self.gsms.items():
            table = gsm.table
            series[gsm_id] = pd.Series(
                table[value_column].values, index=table["ID_REF"].values
            )
        return pd.DataFrame(series)


def _make_sample(gsm_id: str, stage: str, ids, values) -> FakeGSM:
    table = pd.DataFrame({"ID_REF": ids, "VALUE": values})
    metadata = {
        "title": [f"tumor {gsm_id}"],
        "source_name_ch1": ["ovarian tumor"],
        "platform_id": ["GPL570"],
        "characteristics_ch1": [f"stage: {stage}", "grade: 3"],
    }
    return FakeGSM(metadata, table)


def _install_fake_geoparse(monkeypatch, gse=None, raise_on_get: bool = False):
    """Install a fake GEOparse module whose get_GEO returns ``gse`` (or raises)."""
    fake = types.ModuleType("GEOparse")

    def get_GEO(geo=None, destdir=None, silent=False, **kwargs):
        if raise_on_get:
            raise RuntimeError("simulated network failure")
        return gse

    fake.get_GEO = get_GEO
    monkeypatch.setitem(sys.modules, "GEOparse", fake)
    return fake


# --------------------------------------------------------------------------- #
# load_matrix_from_file
# --------------------------------------------------------------------------- #
def test_load_matrix_from_file_csv(tmp_path):
    csv = tmp_path / "matrix.csv"
    csv.write_text("gene,GSM1,GSM2\nA,1.0,2.0\nB,3.0,4.0\n")

    matrix = load_matrix_from_file(str(csv))

    assert list(matrix.columns) == ["GSM1", "GSM2"]
    assert list(matrix.index) == ["A", "B"]
    assert matrix.loc["B", "GSM2"] == 4.0


# --------------------------------------------------------------------------- #
# metadata frame
# --------------------------------------------------------------------------- #
def test_metadata_frame_expands_characteristics_and_preserves_raw():
    gse = FakeGSE(
        {
            "GSM1": _make_sample("GSM1", "III", ["p1", "p2"], [1.0, 2.0]),
            "GSM2": _make_sample("GSM2", "IV", ["p1", "p2"], [3.0, 4.0]),
        }
    )

    meta = _build_metadata_frame(gse)

    # char:: split for human inspection
    assert meta.loc["GSM1", "char::stage"] == "III"
    assert meta.loc["GSM2", "char::stage"] == "IV"
    assert meta.loc["GSM1", "char::grade"] == "3"
    # raw string preserved verbatim for Day 3's Claude pass
    assert meta.loc["GSM1", "characteristics_ch1"] == "stage: III || grade: 3"
    assert meta.index.name == "sample"


# --------------------------------------------------------------------------- #
# expression pivot
# --------------------------------------------------------------------------- #
def test_expression_pivot_is_features_by_samples():
    gse = FakeGSE(
        {
            "GSM1": _make_sample("GSM1", "III", ["p1", "p2"], [1.0, 2.0]),
            "GSM2": _make_sample("GSM2", "IV", ["p1", "p2"], [3.0, 4.0]),
        }
    )

    matrix = _build_expression_matrix(gse, value_column="VALUE")

    assert list(matrix.columns) == ["GSM1", "GSM2"]  # samples are columns
    assert list(matrix.index) == ["p1", "p2"]  # features are rows
    assert matrix.loc["p2", "GSM1"] == 2.0


def test_no_value_column_returns_none():
    # tables lack a VALUE column
    gsm = FakeGSM(
        {"platform_id": ["GPL570"], "characteristics_ch1": ["stage: III"]},
        pd.DataFrame({"ID_REF": ["p1"], "COUNT": [5]}),
    )
    gse = FakeGSE({"GSM1": gsm})

    assert _build_expression_matrix(gse, value_column="VALUE") is None


# --------------------------------------------------------------------------- #
# fetch_gse — fallbacks and happy path
# --------------------------------------------------------------------------- #
def test_get_geo_raising_falls_back_to_manual_upload(monkeypatch, tmp_path):
    _install_fake_geoparse(monkeypatch, raise_on_get=True)

    result = fetch_gse("gse9891", cache_dir=str(tmp_path))

    assert isinstance(result, FetchResult)
    assert result.status == "needs_manual_upload"
    assert not result.ok
    assert result.accession == "GSE9891"  # normalized
    assert "upload" in result.message.lower()
    assert result.expression is None


def test_parsed_but_no_matrix_keeps_metadata(monkeypatch, tmp_path):
    # samples parse fine (metadata present) but carry no VALUE column
    gsm = FakeGSM(
        {
            "title": ["tumor 1"],
            "platform_id": ["GPL11154"],
            "characteristics_ch1": ["stage: III"],
        },
        pd.DataFrame({"ID_REF": ["p1"], "COUNT": [5]}),
    )
    gse = FakeGSE({"GSM1": gsm})
    _install_fake_geoparse(monkeypatch, gse=gse)

    result = fetch_gse("GSE_RNASEQ", cache_dir=str(tmp_path))

    assert result.status == "needs_manual_upload"
    assert result.expression is None
    # metadata is still carried through
    assert result.metadata is not None
    assert result.n_samples == 1
    assert "GPL11154" in result.platform_ids
    assert "metadata" in result.message.lower()


def test_happy_path_returns_ok(monkeypatch, tmp_path):
    gse = FakeGSE(
        {
            "GSM1": _make_sample("GSM1", "III", ["p1", "p2", "p3"], [1.0, 2.0, 3.0]),
            "GSM2": _make_sample("GSM2", "IV", ["p1", "p2", "p3"], [4.0, 5.0, 6.0]),
        }
    )
    _install_fake_geoparse(monkeypatch, gse=gse)

    result = fetch_gse("GSE9891", cache_dir=str(tmp_path))

    assert result.status == "ok"
    assert result.ok
    assert result.n_features == 3
    assert result.n_samples == 2
    assert result.platform_ids == ["GPL570"]
    assert result.expression is not None
    assert result.metadata is not None
    assert "ok" in result.summary()
