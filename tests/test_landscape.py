"""Offline tests for core.landscape — synthetic data with a planted signal; no network."""

from __future__ import annotations

import importlib.util
from pathlib import Path

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
    height_subpayload,
    landscape_payload,
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


def _planted_with_nuisance(n_samples: int = 90, n_genes: int = 400, seed: int = 1):
    """Like ``_planted`` but the risk axis is buried under dominant nuisance axes.

    Two nuisance latents (uncorrelated with survival) drive many high-variance
    genes, so the top-2 principal components capture nuisance and MISS the risk
    axis ``z`` — it sits around PC3. This is the realistic batch-vs-biology case:
    unsupervised 2D PCA cannot show the prognosis gradient, but a supervised
    embedding, guided toward outcome, can pull ``z`` out. Returns
    ``(matrix, survival, z)`` (``z`` = the planted per-sample risk latent).
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n_samples)  # risk latent

    data = np.zeros((n_genes, n_samples))
    for i in range(20):  # risk genes -> a mid-rank PC
        data[i] = 3.0 * z + rng.normal(scale=0.4, size=n_samples)
    idx = 20
    for _ in range(2):  # two dominant nuisance blocks (batch-like)
        w = rng.normal(size=n_samples)
        for _ in range(30):
            data[idx] = 3.0 * w + rng.normal(scale=0.4, size=n_samples)
            idx += 1
    for i in range(idx, n_genes):  # noise genes
        data[i] = rng.normal(size=n_samples)

    genes = [f"g{i}" for i in range(n_genes)]
    samples = [f"GSM{i}" for i in range(n_samples)]
    matrix = pd.DataFrame(data, index=genes, columns=samples)

    durations = np.clip(500.0 * np.exp(-0.8 * z) + rng.normal(scale=15, size=n_samples), 10, None)
    events = np.ones(n_samples, dtype=int)
    survival = pd.DataFrame({"survival_days": durations, "event": events}, index=samples)
    return matrix, survival, pd.Series(z, index=samples)


def _best_axis_corr(coords: pd.DataFrame, target: pd.Series) -> float:
    """Max |corr| of either embedding axis with a per-sample target."""
    t = target.reindex(coords.index).to_numpy()
    return max(abs(np.corrcoef(coords[c].to_numpy(), t)[0, 1]) for c in ["x", "y"])


def _load_check_script():
    """Import scripts/landscape_check.py by path (offline; runs no main)."""
    path = Path(__file__).resolve().parent.parent / "scripts" / "landscape_check.py"
    spec = importlib.util.spec_from_file_location("landscape_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _survival_frame(days, events, datasets):
    """Build a standardized-style frame + matching batch Series from parallel lists."""
    idx = pd.Index([f"GSM{i}" for i in range(len(days))], name="sample")
    std = pd.DataFrame(
        {"survival_days": days, "event": events, "dataset": datasets}, index=idx
    )
    batch = pd.Series(datasets, index=idx)
    return std, batch


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


def test_supervised_embedding_organizes_by_outcome():
    # Risk axis is buried under dominant nuisance: unsupervised 2D PCA misses it,
    # the outcome-supervised map recovers it.
    matrix, survival, z = _planted_with_nuisance()

    pca_coords = embed(matrix, n_components=2, method="pca")
    sup_coords = embed(matrix, n_components=2, method="supervised", survival=survival)

    pca_corr = _best_axis_corr(pca_coords, z)
    sup_corr = _best_axis_corr(sup_coords, z)

    assert pca_corr < 0.4  # nuisance dominates the top-2 PCs
    assert sup_corr > pca_corr + 0.2  # supervision clearly recovers the risk axis


def test_supervised_embedding_is_deterministic():
    matrix, survival, _ = _planted_with_nuisance()
    a = embed(matrix, n_components=2, method="supervised", survival=survival)
    b = embed(matrix, n_components=2, method="supervised", survival=survival)
    pd.testing.assert_frame_equal(a, b)


def test_supervised_embedding_places_unlabeled_by_signal():
    # Withhold survival for half the samples: they are still embedded (no crash)
    # and land near same-signal labeled samples (projected by expression only).
    matrix, survival, z = _planted_with_nuisance()
    labeled = list(matrix.columns[: matrix.shape[1] * 2 // 3])
    unlabeled = [g for g in matrix.columns if g not in set(labeled)]
    partial = survival.loc[labeled]

    coords = embed(matrix, n_components=2, method="supervised", survival=partial)

    assert list(coords.index) == list(matrix.columns)  # all samples embedded
    assert np.isfinite(coords.to_numpy()).all()
    # The held-out samples' placement still tracks their planted risk.
    assert _best_axis_corr(coords.loc[unlabeled], z.loc[unlabeled]) > 0.4


def test_embed_supervised_requires_survival():
    matrix, _, _ = _planted_with_nuisance()
    with pytest.raises(ValueError):
        embed(matrix, method="supervised")


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

    XX1, YY1, ZZ1 = surface(coords, layer, grid=25, method="rbf", extrapolate=True)
    XX2, YY2, ZZ2 = surface(coords, layer, grid=25, method="rbf", extrapolate=True)

    assert XX1.shape == (25, 25)
    assert ZZ1.shape == (25, 25)
    assert np.isfinite(ZZ1).all()  # extrapolate=True fills the whole grid
    np.testing.assert_allclose(ZZ1, ZZ2)  # deterministic


def test_surface_clips_to_data_hull():
    # Default (extrapolate=False): finite inside the sample hull, NaN outside.
    matrix, survival = _planted()
    coords = embed(matrix, n_components=2)
    model = fit_risk(matrix, survival, n_pca=15)
    layer = risk_layer(model, matrix)

    _, _, ZZ_clip = surface(coords, layer, grid=30, method="rbf")
    _, _, ZZ_full = surface(coords, layer, grid=30, method="rbf", extrapolate=True)

    assert np.isnan(ZZ_clip).any()  # some cells fall outside the hull
    assert np.isfinite(ZZ_clip).any()  # but the interior is filled
    # Clipping never invents values: inside cells match the unclipped grid.
    inside = ~np.isnan(ZZ_clip)
    np.testing.assert_allclose(ZZ_clip[inside], ZZ_full[inside])

    # Deterministic (NaN-aware).
    _, _, ZZ_clip2 = surface(coords, layer, grid=30, method="rbf")
    np.testing.assert_allclose(ZZ_clip, ZZ_clip2, equal_nan=True)


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


# --------------------------------------------------------------------------- #
# landscape_check.py — held-out / supervised reporting (count-based, no drift)
# --------------------------------------------------------------------------- #
def test_report_flags_only_series_with_zero_usable_survival():
    lc = _load_check_script()
    # A: all four samples usable. B: none (all-NA survival).
    std, batch = _survival_frame(
        days=[365.0, 700.0, 500.0, 900.0, np.nan, np.nan],
        events=[1, 0, 1, 1, np.nan, np.nan],
        datasets=["A", "A", "A", "A", "B", "B"],
    )
    usable, per_series, held_out = lc._survival_supervision(std, batch, std.index)

    assert held_out == ["B"]  # ONLY the series with zero usable survival
    per = dict((s, (n, tot)) for s, n, tot in per_series)
    assert per["A"][0] > 0  # A supervised the map
    assert per["A"] == (4, 4)
    # Per-series supervised counts sum to the usable-survival mask exactly.
    assert sum(n for _, n, _ in per_series) == len(usable)
    assert set(usable) == {"GSM0", "GSM1", "GSM2", "GSM3"}


def test_report_counts_partial_nan_series_as_contributing():
    lc = _load_check_script()
    # A has partial NaN / a zero-duration row (unusable), but SOME usable samples,
    # so it must count as contributing — never flagged held-out.
    std, batch = _survival_frame(
        days=[365.0, np.nan, 0.0, 800.0, np.nan],
        events=[1, 1, 1, 0, np.nan],
        datasets=["A", "A", "A", "A", "B"],
    )
    usable, per_series, held_out = lc._survival_supervision(std, batch, std.index)

    per = dict((s, (n, tot)) for s, n, tot in per_series)
    assert "A" not in held_out  # partial NaN != held out
    assert per["A"] == (2, 4)  # GSM0 + GSM3 usable; NaN-day and 0-day excluded
    assert held_out == ["B"]  # B is the only zero-usable series
    assert sum(n for _, n, _ in per_series) == len(usable) == 2


def test_usable_survival_matches_align_survival_criterion():
    # The report's mask must equal the samples the risk model actually fits.
    from core.landscape import _align_survival

    lc = _load_check_script()
    std, _ = _survival_frame(
        days=[365.0, np.nan, 0.0, 800.0],
        events=[1, 1, 1, 0],
        datasets=["A", "A", "A", "A"],
    )
    matrix = pd.DataFrame(
        np.random.default_rng(0).normal(size=(10, 4)),
        index=[f"g{i}" for i in range(10)],
        columns=std.index,
    )
    report_usable = set(lc._usable_survival(std, matrix.columns))
    expr, _, _ = _align_survival(matrix, std[["survival_days", "event"]])
    assert report_usable == set(expr.index)


# --------------------------------------------------------------------------- #
# landscape_payload — the JSON contract the 3D frontend consumes
# --------------------------------------------------------------------------- #
def _payload_fixture():
    """A small model + metadata + height layers. Dataset A has survival, B none."""
    matrix, survival = _planted(n_samples=40, n_genes=60)
    # Split into two datasets; B's survival is withheld (all-NA) so it is held out.
    ids = list(matrix.columns)
    datasets = ["A"] * 28 + ["B"] * (len(ids) - 28)
    meta = pd.DataFrame({"dataset": datasets}, index=ids)
    meta["survival_days"] = survival["survival_days"]
    meta["event"] = survival["event"]
    meta.loc[[g for g, d in zip(ids, datasets) if d == "B"], ["survival_days", "event"]] = np.nan

    surv_used = meta.loc[meta["dataset"] == "A", ["survival_days", "event"]]
    coords = embed(matrix, n_components=2)
    model = fit_risk(matrix, surv_used, n_pca=10)
    c = cv_cindex(matrix, surv_used, folds=4, n_pca=10)
    lm = LandscapeModel(coords=coords, risk_model=model, cindex=c)
    layers = {
        "risk": ("Predicted risk", "risk", risk_layer(model, matrix)),
        "prolif": ("Proliferation", "signature", signature_layer(matrix, ["g0", "g1", "g2"], "prolif")),
        "g5": ("g5", "gene", gene_layer(matrix, "g5")),
    }
    return lm, meta, layers, matrix


def test_landscape_payload_shape_and_keys():
    import json

    lm, meta, layers, _ = _payload_fixture()
    payload = landscape_payload(lm, meta, layers, grid=15)

    assert set(payload) == {"samples", "surfaces", "height_options", "meta"}
    # samples: ordered by GSM, each with the full height set
    assert [s["id"] for s in payload["samples"]] == sorted(lm.coords.index)
    s0 = payload["samples"][0]
    assert set(s0) == {"id", "x", "y", "dataset", "heights"}
    assert set(s0["heights"]) == {"risk", "prolif", "g5"}
    # height_options non-empty and well-formed
    assert payload["height_options"]
    assert all(set(o) == {"key", "label", "kind"} for o in payload["height_options"])
    # surfaces: one per layer, grid-shaped
    assert set(payload["surfaces"]) == {"risk", "prolif", "g5"}
    surf = payload["surfaces"]["risk"]
    assert len(surf["gx"]) == 15 and len(surf["gy"]) == 15
    assert len(surf["z"]) == 15 and all(len(row) == 15 for row in surf["z"])
    # meta
    assert payload["meta"]["n_samples"] == lm.coords.shape[0]
    assert isinstance(payload["meta"]["cindex"], float)
    # fully JSON-serializable (no NaN/inf leaks)
    json.dumps(payload)


def test_landscape_payload_nan_surface_cells_are_null():
    lm, meta, layers, _ = _payload_fixture()
    payload = landscape_payload(lm, meta, layers, grid=20)
    z = payload["surfaces"]["risk"]["z"]
    flat = [cell for row in z for cell in row]
    assert any(cell is None for cell in flat)  # outside the hull -> null
    assert any(isinstance(cell, float) for cell in flat)  # inside -> finite float
    assert all(cell is None or isinstance(cell, float) for cell in flat)


def test_landscape_payload_is_deterministic():
    lm, meta, layers, _ = _payload_fixture()
    a = landscape_payload(lm, meta, layers, grid=12)
    b = landscape_payload(lm, meta, layers, grid=12)
    import json

    assert json.dumps(a) == json.dumps(b)


def test_landscape_payload_meta_holds_out_only_zero_usable_series():
    lm, meta, layers, _ = _payload_fixture()
    payload = landscape_payload(lm, meta, layers, grid=12)
    assert payload["meta"]["held_out"] == ["B"]  # only the zero-usable series
    assert payload["meta"]["n_supervised"] == 28  # dataset A's samples


def test_height_subpayload_shape():
    lm, meta, layers, matrix = _payload_fixture()
    sub = height_subpayload(lm.coords, "gene:g7", "g7", "gene", gene_layer(matrix, "g7"), grid=12)
    assert set(sub) == {"key", "label", "kind", "heights", "surface"}
    assert set(sub["heights"]) == set(lm.coords.index)
    assert set(sub["surface"]) == {"gx", "gy", "z"}
