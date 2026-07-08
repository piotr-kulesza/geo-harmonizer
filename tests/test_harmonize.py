"""Offline tests for core.harmonize — no network; mygene is monkeypatched.

Covers the log2 double-guard, probe->gene collapse (max/mean, unmapped dropped),
and the annotation cache short-circuiting the mygene network call.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from core import harmonize
from core.harmonize import map_probes, to_log2, _query_symbols


# --------------------------------------------------------------------------- #
# to_log2 — double-log2 guard
# --------------------------------------------------------------------------- #
def test_to_log2_transforms_linear_matrix():
    # Linear-scale matrix (max in the thousands) -> log2.
    matrix = pd.DataFrame(
        {"GSM1": [1024.0, 16.0], "GSM2": [4096.0, 256.0]},
        index=["p1", "p2"],
    )

    out = to_log2(matrix)

    assert out.loc["p1", "GSM1"] == pytest.approx(10.0)  # log2(1024)
    assert out.loc["p2", "GSM2"] == pytest.approx(8.0)  # log2(256)
    assert out.shape == matrix.shape


def test_to_log2_leaves_already_log_matrix_unchanged():
    # Already log-scale (max < 30) -> returned unchanged.
    matrix = pd.DataFrame(
        {"GSM1": [10.0, 4.0], "GSM2": [12.5, 6.0]},
        index=["p1", "p2"],
    )

    out = to_log2(matrix)

    pd.testing.assert_frame_equal(out, matrix)


def test_to_log2_floors_at_one_no_neg_inf():
    matrix = pd.DataFrame({"GSM1": [0.0, 5000.0]}, index=["p1", "p2"])
    out = to_log2(matrix)
    assert np.isfinite(out.to_numpy()).all()
    assert out.loc["p1", "GSM1"] == pytest.approx(0.0)  # log2(clip(0->1)) = 0


# --------------------------------------------------------------------------- #
# map_probes — collapse + drop unmapped
# --------------------------------------------------------------------------- #
def _expr():
    # p1 and p2 both map to GENEA; p3 -> GENEB; p4 -> unmapped.
    return pd.DataFrame(
        {
            "GSM1": [10.0, 2.0, 7.0, 99.0],
            "GSM2": [20.0, 8.0, 1.0, 99.0],
        },
        index=["p1", "p2", "p3", "p4"],
    )


def _patch_symbols(monkeypatch, mapping):
    monkeypatch.setattr(
        harmonize,
        "_query_symbols",
        lambda probes, platform_id, cache_dir="x": {
            p: mapping[p] for p in probes if p in mapping
        },
    )


def test_map_probes_collapse_max(monkeypatch):
    _patch_symbols(monkeypatch, {"p1": "GENEA", "p2": "GENEA", "p3": "GENEB"})

    genes = map_probes(_expr(), platform_id="GPL570", collapse="max")

    assert list(genes.index) == ["GENEA", "GENEB"]  # sorted, unmapped p4 dropped
    assert genes.index.name == "gene"
    assert genes.shape == (2, 2)
    # GENEA = max(p1, p2) per sample
    assert genes.loc["GENEA", "GSM1"] == 10.0
    assert genes.loc["GENEA", "GSM2"] == 20.0
    assert genes.loc["GENEB", "GSM1"] == 7.0


def test_map_probes_collapse_mean(monkeypatch):
    _patch_symbols(monkeypatch, {"p1": "GENEA", "p2": "GENEA", "p3": "GENEB"})

    genes = map_probes(_expr(), platform_id="GPL570", collapse="mean")

    assert genes.loc["GENEA", "GSM1"] == pytest.approx(6.0)  # mean(10, 2)
    assert genes.loc["GENEA", "GSM2"] == pytest.approx(14.0)  # mean(20, 8)


def test_map_probes_rejects_bad_collapse(monkeypatch):
    _patch_symbols(monkeypatch, {"p1": "GENEA"})
    with pytest.raises(ValueError):
        map_probes(_expr(), platform_id="GPL570", collapse="median")


# --------------------------------------------------------------------------- #
# _query_symbols — cache short-circuits the network
# --------------------------------------------------------------------------- #
def test_query_symbols_uses_cache_without_network(monkeypatch, tmp_path):
    # Pre-write a full cache for the platform.
    cache_dir = tmp_path / "annotations"
    cache_dir.mkdir()
    (cache_dir / "GPL570.json").write_text(
        json.dumps({"p1": "GENEA", "p2": "GENEB", "p3": "notfound"})
    )

    # If mygene is touched, fail loudly.
    def _boom(*args, **kwargs):
        raise AssertionError("mygene should not be called when fully cached")

    monkeypatch.setattr(harmonize, "_fetch_symbols_from_mygene", _boom)

    result = _query_symbols(["p1", "p2", "p3"], "GPL570", str(cache_dir))

    assert result == {"p1": "GENEA", "p2": "GENEB"}  # notfound dropped


def test_query_symbols_queries_only_missing_and_rewrites(monkeypatch, tmp_path):
    cache_dir = tmp_path / "annotations"
    cache_dir.mkdir()
    (cache_dir / "GPL570.json").write_text(json.dumps({"p1": "GENEA"}))

    calls = {}

    def _fake_fetch(probes, platform_id):
        calls["probes"] = list(probes)
        return {"p2": "GENEB"}

    monkeypatch.setattr(harmonize, "_fetch_symbols_from_mygene", _fake_fetch)

    result = _query_symbols(["p1", "p2"], "GPL570", str(cache_dir))

    # Only the missing probe was queried.
    assert calls["probes"] == ["p2"]
    assert result == {"p1": "GENEA", "p2": "GENEB"}
    # Cache rewritten with the merged map.
    rewritten = json.loads((cache_dir / "GPL570.json").read_text())
    assert rewritten == {"p1": "GENEA", "p2": "GENEB"}
