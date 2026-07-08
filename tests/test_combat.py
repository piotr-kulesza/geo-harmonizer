"""Offline tests for core.combat — inmoose is monkeypatched; no network."""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from core.combat import combat


def _install_fake_inmoose(monkeypatch, transform=None):
    """Install a fake inmoose.pycombat.pycombat_norm.

    Default stub is shape-preserving identity (returns the input values). A
    ``transform`` callable may customize the returned array.
    """
    calls = {}

    def pycombat_norm(matrix, batch, *args, **kwargs):
        calls["batch"] = list(batch)
        calls["shape"] = np.asarray(matrix).shape
        values = np.asarray(matrix, dtype=float)
        return transform(values) if transform else values

    pkg = types.ModuleType("inmoose")
    sub = types.ModuleType("inmoose.pycombat")
    sub.pycombat_norm = pycombat_norm
    pkg.pycombat = sub
    monkeypatch.setitem(sys.modules, "inmoose", pkg)
    monkeypatch.setitem(sys.modules, "inmoose.pycombat", sub)
    return calls


def _merged(genes=("G1", "G2", "G3"), samples=("A1", "A2", "B1", "B2")):
    data = {s: [float(i + j) for j in range(len(genes))] for i, s in enumerate(samples)}
    return pd.DataFrame(data, index=list(genes))


def _batch(samples=("A1", "A2", "B1", "B2"), accs=("GSE_A", "GSE_A", "GSE_B", "GSE_B")):
    return pd.Series(dict(zip(samples, accs)), name="batch")


def test_combat_preserves_index_and_columns(monkeypatch):
    calls = _install_fake_inmoose(monkeypatch)
    merged, batch = _merged(), _batch()

    out = combat(merged, batch)

    assert list(out.index) == list(merged.index)
    assert list(out.columns) == list(merged.columns)
    assert out.shape == merged.shape
    # batch passed in column order
    assert calls["batch"] == ["GSE_A", "GSE_A", "GSE_B", "GSE_B"]


def test_combat_requires_two_batches(monkeypatch):
    _install_fake_inmoose(monkeypatch)
    merged = _merged(samples=("A1", "A2", "A3"))
    batch = _batch(samples=("A1", "A2", "A3"), accs=("GSE_A", "GSE_A", "GSE_A"))
    with pytest.raises(ValueError, match="distinct batches"):
        combat(merged, batch)


def test_combat_rejects_singleton_batch(monkeypatch):
    _install_fake_inmoose(monkeypatch)
    merged = _merged(samples=("A1", "A2", "B1"))
    batch = _batch(samples=("A1", "A2", "B1"), accs=("GSE_A", "GSE_A", "GSE_B"))
    with pytest.raises(ValueError, match="samples per batch"):
        combat(merged, batch)


def test_combat_drops_nonfinite_rows(monkeypatch):
    calls = _install_fake_inmoose(monkeypatch)
    merged = _merged()
    merged.loc["G2", "B1"] = np.nan  # one bad gene row

    out = combat(merged, _batch())

    assert list(out.index) == ["G1", "G3"]  # G2 dropped
    assert out.shape == (2, 4)
    assert calls["shape"] == (2, 4)  # inmoose saw the cleaned matrix


def test_combat_raises_when_all_rows_nonfinite(monkeypatch):
    _install_fake_inmoose(monkeypatch)
    merged = _merged()
    merged.iloc[:, :] = np.nan
    with pytest.raises(ValueError):
        combat(merged, _batch())
