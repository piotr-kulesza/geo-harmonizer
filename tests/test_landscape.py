"""Offline tests for core.landscape — synthetic data with a planted signal; no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.landscape import (
    HeightLayer,
    LandscapeModel,
    cv_cindex,
    embed,
    fit_risk,
    gene_layer,
    predict_risk,
    risk_layer,
    signature_layer,
    surface,
)


# --------------------------------------------------------------------------- #
# Synthetic substrate with a planted survival signal
# --------------------------------------------------------------------------- #
def _planted(n_samples: int = 60, n_genes: int = 200, seed: int = 0):
    """genes x samples matrix + survival, sharing one latent axis z.

    z drives both expression (first 20 genes load on it) and survival (higher z
    => shorter time). A risk model should recover z and score C-index >> 0.5.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n_samples)

    data = np.empty((n_genes, n_samples))
    for i in range(20):  # signal genes
        data[i] = 3.0 * z + rng.normal(scale=0.5, size=n_samples)
    for i in range(20, n_genes):  # noise genes
        data[i] = rng.normal(size=n_samples)

    genes = [f"g{i}" for i in range(n_genes)]
    samples = [f"GSM{i}" for i in range(n_samples)]
    matrix = pd.DataFrame(data, index=genes, columns=samples)

    durations = np.clip(500.0 * np.exp(-0.8 * z) + rng.normal(scale=15, size=n_samples), 10, None)
    events = np.ones(n_samples, dtype=int)  # fully observed
    survival = pd.DataFrame({"survival_days": durations, "event": events}, index=samples)
    return matrix, survival


# --------------------------------------------------------------------------- #
# embed
# --------------------------------------------------------------------------- #
def test_embed_shape_and_determinism():
    matrix, _ = _planted()
    a = embed(matrix, n_components=2)
    b = embed(matrix, n_components=2)

    assert a.shape == (matrix.shape[1], 2)
    assert list(a.columns) == ["x", "y"]
    assert list(a.index) == list(matrix.columns)
    assert a.index.name == "sample"
    pd.testing.assert_frame_equal(a, b)  # deterministic


# --------------------------------------------------------------------------- #
# risk model
# --------------------------------------------------------------------------- #
def test_predict_risk_returns_per_sample_series():
    matrix, survival = _planted()
    model = fit_risk(matrix, survival, n_pca=15)
    risk = predict_risk(model, matrix)

    assert isinstance(risk, pd.Series)
    assert list(risk.index) == list(matrix.columns)  # ALL samples scored
    assert risk.name == "risk"
    assert np.isfinite(risk.to_numpy()).all()


def test_cv_cindex_recovers_planted_signal():
    matrix, survival = _planted()
    c = cv_cindex(matrix, survival, folds=5, n_pca=15)
    assert c > 0.6  # clearly better than chance on the planted signal


def test_fit_risk_only_uses_samples_with_survival():
    # Survival known for only a subset; predict still covers all.
    matrix, survival = _planted()
    partial = survival.iloc[:30]  # only 30/60 have survival
    model = fit_risk(matrix, partial, n_pca=15)
    risk = predict_risk(model, matrix)
    assert risk.shape[0] == matrix.shape[1]  # all 60 draped


# --------------------------------------------------------------------------- #
# height layers
# --------------------------------------------------------------------------- #
def test_signature_layer_is_mean_of_present_genes():
    matrix, _ = _planted()
    layer = signature_layer(matrix, ["g0", "g1", "absent"], name="sig")
    assert isinstance(layer, HeightLayer)
    assert layer.name == "sig"
    expected = matrix.loc[["g0", "g1"]].mean(axis=0)
    pd.testing.assert_series_equal(layer.values, expected, check_names=False)


def test_gene_layer_single_gene():
    matrix, _ = _planted()
    layer = gene_layer(matrix, "g5")
    pd.testing.assert_series_equal(layer.values, matrix.loc["g5"], check_names=False)


def test_gene_layer_absent_gene_is_nan():
    matrix, _ = _planted()
    layer = gene_layer(matrix, "nope")
    assert layer.values.isna().all()
    assert layer.values.shape[0] == matrix.shape[1]


def test_risk_layer_wires_through():
    matrix, survival = _planted()
    model = fit_risk(matrix, survival, n_pca=15)
    layer = risk_layer(model, matrix)
    assert layer.name == "risk"
    assert layer.values.shape[0] == matrix.shape[1]


# --------------------------------------------------------------------------- #
# surface
# --------------------------------------------------------------------------- #
def test_surface_shape_and_determinism():
    matrix, survival = _planted()
    coords = embed(matrix, n_components=2)
    model = fit_risk(matrix, survival, n_pca=15)
    layer = risk_layer(model, matrix)

    XX1, YY1, ZZ1 = surface(coords, layer, grid=25, method="rbf")
    XX2, YY2, ZZ2 = surface(coords, layer, grid=25, method="rbf")

    assert XX1.shape == (25, 25)
    assert ZZ1.shape == (25, 25)
    assert np.isfinite(ZZ1).all()
    np.testing.assert_allclose(ZZ1, ZZ2)  # deterministic


def test_landscape_model_produces_surface_for_a_layer():
    matrix, survival = _planted()
    coords = embed(matrix, n_components=2)
    model = fit_risk(matrix, survival, n_pca=15)
    landscape = LandscapeModel(coords=coords, risk_model=model)
    landscape.add_layer(risk_layer(model, matrix))

    XX, YY, ZZ = landscape.surface("risk", grid=20)
    assert ZZ.shape == (20, 20)
    with pytest.raises(KeyError):
        landscape.surface("does-not-exist")
