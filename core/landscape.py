"""Disease risk landscape on the harmonized substrate. [Landscape v1]

Built ON TOP of the existing pipeline — it does not touch fetch/harmonize/merge/
combat/metadata. The ComBat-corrected merged matrix (genes x samples) is the
substrate. The idea:

- **Terrain (fixed 2D map).** Embed every tumor into a fixed 2D plane
  (:func:`embed`) — the map is the same regardless of which height you drape.
- **Height (pluggable per-sample scalar).** A :class:`HeightLayer` is any
  per-GSM scalar. The validated default is predicted survival risk
  (:func:`risk_layer`); a gene signature (:func:`signature_layer`) or a single
  gene (:func:`gene_layer`) work on the same map. "Choose the height while
  analyzing."
- **Surface.** :func:`surface` smooths a layer's per-sample values over a grid
  spanning the map, giving meshgrids for a 3D render.

Data realities encoded here:
- Survival exists only in some cohorts. So the risk model is FIT and
  CROSS-VALIDATED only on samples that have ``survival_days`` + ``event`` (from
  the Day-3 standardized metadata), then used to PREDICT risk for ALL samples so
  the surface can be draped everywhere.
- We never fit a Cox model on ~13k genes (p >> n). :func:`fit_risk` reduces to a
  few dozen PCA components first, then fits a penalized Cox on those.

Iron rule: pure logic — no web/MCP/UI, no ``print``. The 3D *render* lives in
``scripts/landscape_check.py``, not here. Heavy libs (scikit-learn, lifelines,
scipy, umap) are lazy-imported inside functions so ``core`` imports without them.
Determinism: same input -> same embedding and same surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 2D embedding (the terrain)
# --------------------------------------------------------------------------- #
def embed(
    matrix: pd.DataFrame,
    n_components: int = 2,
    method: str = "pca",
    survival: Optional[pd.DataFrame] = None,
    n_pca: int = 50,
    random_state: int = 0,
) -> pd.DataFrame:
    """Embed samples into a fixed low-D map from a genes x samples matrix.

    Args:
        matrix: genes (rows) x samples (GSM columns) — typically the
            ComBat-corrected merged matrix.
        n_components: output dimensions (2 for the terrain plane).
        method: how to lay out the map.
            - ``"pca"`` (default, deterministic): top principal components.
            - ``"umap"``: unsupervised UMAP on the top PCA components.
            - ``"supervised"``: outcome-supervised UMAP toward survival — the
              map organizes by prognosis-relevant biology (see below).
        survival: required for ``method="supervised"``. A DataFrame indexed by
            GSM with ``survival_days`` + ``event`` for the cohorts that carry
            survival. Ignored by the other methods.
        n_pca: PCA components reduced to before UMAP / used as the supervised
            feature space.
        random_state: seed for reproducibility.

    Supervised embedding (honesty note): expression is reduced to ``n_pca``
    deterministic PCA components, then a UMAP is fit on the samples that HAVE
    survival toward an **out-of-fold** Cox risk target (the same PCA+Cox refit on
    training folds and predicted for held-out samples). Using out-of-fold risk —
    not the in-sample risk we later drape as height — keeps the supervision from
    being the very signal it is meant to reveal. Samples WITHOUT survival (e.g.
    GSE9891) are held out of the supervised fit (masked / unlabeled) and then
    projected into the learned space by expression structure alone, so their
    landing on the risk terrain is a genuine generalization check, not baked in.

    This produces an **outcome-structured representation for visualization**. It
    is NOT the validation: the risk model's independent cross-validated C-index
    (:func:`cv_cindex`) remains the gate. We are not claiming the risk gradient
    is emergent from unsupervised structure.

    Returns:
        A samples (GSM index) x ``[x, y, ...]`` DataFrame, deterministic for a
        given input.
    """
    samples_by_genes, _, _ = _standardize_columns_as_samples(matrix)
    gsm_index = pd.Index(matrix.columns, name="sample")

    if method == "pca":
        coords = _pca_transform(samples_by_genes, n_components, random_state)
    elif method == "umap":
        coords = _umap_transform(samples_by_genes, n_components, random_state)
    elif method == "supervised":
        if survival is None:
            raise ValueError("embed(method='supervised') requires a survival frame.")
        coords = _supervised_transform(matrix, survival, n_components, n_pca, random_state)
    else:
        raise ValueError(
            f"embed method must be 'pca', 'umap' or 'supervised', got {method!r}"
        )

    labels = ["x", "y", "z"][:n_components] or [f"dim{i}" for i in range(n_components)]
    if n_components > 3:
        labels = [f"dim{i}" for i in range(n_components)]
    frame = pd.DataFrame(coords, index=gsm_index, columns=labels[:n_components])
    logger.info("embed: %d samples -> %d-D map via %s.", frame.shape[0], n_components, method)
    return frame


# --------------------------------------------------------------------------- #
# Survival risk model (the validated default height)
# --------------------------------------------------------------------------- #
@dataclass
class RiskModel:
    """A fitted PCA + penalized Cox that scores any sample's survival risk.

    Attributes:
        pca: the fitted ``sklearn.decomposition.PCA`` (genes -> components).
        cox: the fitted ``lifelines.CoxPHFitter`` (components -> risk).
        genes: gene order the PCA was fit on (for reindexing new matrices).
        center: per-gene mean subtracted before the PCA transform.
        component_labels: PCA component column names used to fit the Cox.
    """

    pca: object
    cox: object
    genes: pd.Index
    center: np.ndarray
    component_labels: list[str]


def fit_risk(
    matrix: pd.DataFrame,
    survival: pd.DataFrame,
    n_pca: int = 30,
    penalizer: float = 0.1,
) -> RiskModel:
    """Fit a survival-risk model on the samples that HAVE survival.

    Reduces expression to ``n_pca`` PCA components, then fits a penalized Cox on
    them. p >> n is avoided by the PCA step.

    Args:
        matrix: genes (rows) x samples (GSM columns).
        survival: DataFrame indexed by GSM with ``survival_days`` + ``event``
            (0/1), for the samples that have it. Only samples present in BOTH
            ``matrix.columns`` and ``survival.index`` with a positive duration
            and a defined event are used.
        n_pca: number of PCA components (clamped to the sample count).
        penalizer: L2 penalty for ``CoxPHFitter``.

    Returns:
        A :class:`RiskModel` that can score any sample via :func:`predict_risk`.
    """
    expr, durations, events = _align_survival(matrix, survival)
    pca, center, comp_labels, components = _fit_pca_on(expr, n_pca)
    cox = _fit_cox(components, durations, events, comp_labels, penalizer)
    logger.info(
        "fit_risk: %d survival samples, %d PCA comps, penalizer=%.3g.",
        expr.shape[0],
        len(comp_labels),
        penalizer,
    )
    return RiskModel(
        pca=pca,
        cox=cox,
        genes=matrix.index,
        center=center,
        component_labels=comp_labels,
    )


def predict_risk(risk_model: RiskModel, matrix: pd.DataFrame) -> pd.Series:
    """Predict the Cox linear predictor (risk) for ALL samples in ``matrix``.

    Projects each sample through the fitted PCA, then applies the Cox model.
    Higher = worse prognosis. Returns a Series indexed by GSM (name ``"risk"``).
    """
    components = _project_components(risk_model, matrix)
    # Cox linear predictor (log partial hazard): higher => higher risk.
    risk = risk_model.cox.predict_log_partial_hazard(components)
    risk = pd.Series(np.asarray(risk).ravel(), index=matrix.columns, name="risk")
    risk.index.name = "sample"
    return risk


def cv_cindex(
    matrix: pd.DataFrame,
    survival: pd.DataFrame,
    folds: int = 5,
    n_pca: int = 30,
    penalizer: float = 0.1,
    random_state: int = 0,
) -> float:
    """K-fold cross-validated concordance index of the risk model.

    THE VALIDATION GATE. A landscape is only meaningful if this is comfortably
    above 0.5 (aim >= ~0.6). PCA + Cox are refit inside each fold on the training
    split, then scored on the held-out split, so there's no leakage.

    Returns:
        The mean per-fold C-index (``nan`` if no fold could be scored).
    """
    from lifelines.utils import concordance_index  # lazy
    from sklearn.model_selection import KFold  # lazy

    expr, durations, events = _align_survival(matrix, survival)
    n = expr.shape[0]
    k = min(folds, n)
    if k < 2:
        logger.warning("cv_cindex: only %d survival samples — cannot cross-validate.", n)
        return float("nan")

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    scores: list[float] = []
    for train_idx, test_idx in kf.split(np.arange(n)):
        tr_expr = expr.iloc[train_idx]
        te_expr = expr.iloc[test_idx]
        tr_dur, tr_evt = durations.iloc[train_idx], events.iloc[train_idx]
        te_dur, te_evt = durations.iloc[test_idx], events.iloc[test_idx]

        # A fold with no events in the test split has no admissible pairs.
        if te_evt.sum() < 1 or tr_evt.sum() < 1:
            continue

        pca, center, comp_labels, tr_comp = _fit_pca_on(tr_expr, n_pca)
        cox = _fit_cox(tr_comp, tr_dur, tr_evt, comp_labels, penalizer)

        te_std = te_expr.to_numpy(dtype=float) - center
        te_comp = pd.DataFrame(pca.transform(te_std), index=te_expr.index, columns=comp_labels)
        risk = np.asarray(cox.predict_log_partial_hazard(te_comp)).ravel()

        try:
            # concordance_index treats higher score as LONGER survival, so pass
            # -risk (higher risk => shorter survival).
            c = concordance_index(te_dur.to_numpy(), -risk, te_evt.to_numpy())
        except Exception as exc:  # e.g. no admissible pairs
            logger.info("cv_cindex: skipping a fold (%s).", exc)
            continue
        scores.append(float(c))

    if not scores:
        return float("nan")
    mean_c = float(np.mean(scores))
    logger.info("cv_cindex: %d/%d folds scored, mean C-index=%.3f.", len(scores), k, mean_c)
    return mean_c


# --------------------------------------------------------------------------- #
# Height layers (pluggable per-sample scalars)
# --------------------------------------------------------------------------- #
@dataclass
class HeightLayer:
    """A named per-sample scalar to drape over the fixed map.

    Attributes:
        name: layer identifier (e.g. ``"risk"``, ``"serous_signature"``).
        values: per-GSM scalar Series (index = GSM).
    """

    name: str
    values: pd.Series


def risk_layer(risk_model: RiskModel, matrix: pd.DataFrame) -> HeightLayer:
    """Height = predicted survival risk (the validated default)."""
    return HeightLayer(name="risk", values=predict_risk(risk_model, matrix))


def signature_layer(matrix: pd.DataFrame, genes, name: str) -> HeightLayer:
    """Height = mean expression of a gene set per sample.

    Genes absent from ``matrix`` are ignored; if none are present the layer is
    all-NaN (so the map still renders).
    """
    present = [g for g in genes if g in matrix.index]
    if not present:
        logger.warning("signature_layer(%s): none of the genes are present.", name)
        values = pd.Series(np.nan, index=matrix.columns, name=name)
    else:
        values = matrix.loc[present].mean(axis=0)
        values.name = name
    values.index.name = "sample"
    return HeightLayer(name=name, values=values)


def gene_layer(matrix: pd.DataFrame, gene: str) -> HeightLayer:
    """Height = single-gene expression per sample."""
    if gene in matrix.index:
        values = matrix.loc[gene].copy()
    else:
        logger.warning("gene_layer(%s): gene not present.", gene)
        values = pd.Series(np.nan, index=matrix.columns)
    values.name = gene
    values.index.name = "sample"
    return HeightLayer(name=gene, values=values)


# --------------------------------------------------------------------------- #
# Surface (smooth a layer over the map)
# --------------------------------------------------------------------------- #
def surface(
    coords_xy: pd.DataFrame,
    height_layer: HeightLayer,
    grid: int = 60,
    method: str = "rbf",
    extrapolate: bool = False,
):
    """Smooth a height layer over a grid spanning the 2D map.

    Args:
        coords_xy: samples (GSM index) x ``[x, y]`` from :func:`embed`.
        height_layer: the per-sample scalar to interpolate.
        grid: mesh resolution per axis.
        method: ``"rbf"`` (scipy radial-basis smoothing) or ``"griddata"``
            (scipy linear interpolation with nearest-fill).
        extrapolate: if ``False`` (default), grid cells OUTSIDE the convex hull
            of the samples are set to ``NaN`` — the surface exists only where
            there are actually tumors, so no phantom peaks are invented in empty
            corners. If ``True``, the interpolated grid is returned unclipped.

    Returns:
        ``(XX, YY, ZZ)`` meshgrids (each ``grid`` x ``grid``) for a 3D surface.
        With ``extrapolate=False``, ``ZZ`` is ``NaN`` outside the sample hull and
        finite inside. Pure and deterministic.
    """
    xy = coords_xy[["x", "y"]].copy()
    values = height_layer.values.reindex(xy.index)
    ok = values.notna().to_numpy()
    points = xy.to_numpy(dtype=float)[ok]
    z = values.to_numpy(dtype=float)[ok]
    if points.shape[0] < 3:
        raise ValueError("surface needs >=3 samples with finite height values.")

    pad_x = 0.05 * (points[:, 0].max() - points[:, 0].min() or 1.0)
    pad_y = 0.05 * (points[:, 1].max() - points[:, 1].min() or 1.0)
    xs = np.linspace(points[:, 0].min() - pad_x, points[:, 0].max() + pad_x, grid)
    ys = np.linspace(points[:, 1].min() - pad_y, points[:, 1].max() + pad_y, grid)
    XX, YY = np.meshgrid(xs, ys)
    mesh = np.column_stack([XX.ravel(), YY.ravel()])

    if method == "rbf":
        ZZ = _rbf_interpolate(points, z, mesh).reshape(XX.shape)
    elif method == "griddata":
        ZZ = _griddata_interpolate(points, z, mesh).reshape(XX.shape)
    else:
        raise ValueError(f"surface method must be 'rbf' or 'griddata', got {method!r}")

    if not extrapolate:
        ZZ = _clip_to_hull(points, mesh, ZZ)
    return XX, YY, ZZ


# --------------------------------------------------------------------------- #
# Assembled model (what the web/MCP layer consumes)
# --------------------------------------------------------------------------- #
@dataclass
class LandscapeModel:
    """The fixed map plus fitted risk model and any number of height layers.

    Attributes:
        coords: samples (GSM index) x ``[x, y]`` — the fixed terrain.
        risk_model: the fitted :class:`RiskModel`, or ``None`` if no survival.
        layers: height layers by name.
    """

    coords: pd.DataFrame
    risk_model: Optional[RiskModel] = None
    layers: dict[str, HeightLayer] = field(default_factory=dict)

    def add_layer(self, layer: HeightLayer) -> "LandscapeModel":
        """Register a height layer (same map, any scalar). Returns self."""
        self.layers[layer.name] = layer
        return self

    def surface(self, layer_name: str, grid: int = 60, method: str = "rbf"):
        """Build ``(XX, YY, ZZ)`` for one registered layer."""
        if layer_name not in self.layers:
            raise KeyError(f"no height layer named {layer_name!r}; have {list(self.layers)}")
        return surface(self.coords, self.layers[layer_name], grid=grid, method=method)


# --------------------------------------------------------------------------- #
# Private helpers
# --------------------------------------------------------------------------- #
def _standardize_columns_as_samples(matrix: pd.DataFrame):
    """Transpose genes x samples -> samples x genes and center each gene.

    Returns ``(samples_by_genes_centered, center, gene_order)``.
    """
    samples_by_genes = matrix.to_numpy(dtype=float).T
    center = np.nanmean(samples_by_genes, axis=0)
    return samples_by_genes - center, center, matrix.index


def _pca_transform(samples_by_genes_centered, n_components, random_state):
    from sklearn.decomposition import PCA  # lazy

    n_comp = min(n_components, *samples_by_genes_centered.shape)
    pca = PCA(n_components=n_comp, svd_solver="full", random_state=random_state)
    coords = pca.fit_transform(samples_by_genes_centered)
    if coords.shape[1] < n_components:  # pad if fewer comps than requested
        pad = np.zeros((coords.shape[0], n_components - coords.shape[1]))
        coords = np.hstack([coords, pad])
    return coords


def _umap_transform(samples_by_genes_centered, n_components, random_state):
    from sklearn.decomposition import PCA  # lazy
    import umap  # lazy

    pre = min(20, *samples_by_genes_centered.shape)
    pca = PCA(n_components=pre, svd_solver="full", random_state=random_state)
    reduced = pca.fit_transform(samples_by_genes_centered)
    reducer = umap.UMAP(n_components=n_components, random_state=random_state, n_jobs=1)
    return reducer.fit_transform(reduced)


def _supervised_transform(matrix, survival, n_components, n_pca, random_state):
    """Outcome-supervised UMAP: fit on survival cohorts, project the rest.

    Returns coords (n_samples x n_components) in ``matrix.columns`` order. See
    :func:`embed` for the design rationale (out-of-fold target, held-out
    unlabeled samples). Deterministic: fixed PCA + ``random_state`` + ``n_jobs=1``.
    """
    import umap  # lazy

    # Fixed, deterministic PCA feature space for ALL samples.
    samples_by_genes, _, _ = _standardize_columns_as_samples(matrix)
    n_comp = min(n_pca, *samples_by_genes.shape)
    comps_all = _pca_transform(samples_by_genes, n_comp, random_state)
    comps_all = pd.DataFrame(comps_all, index=pd.Index(matrix.columns, name="sample"))

    # Labeled = samples with usable survival; unlabeled = everything else.
    expr, durations, events = _align_survival(matrix, survival)
    labeled = list(expr.index)
    unlabeled = [g for g in matrix.columns if g not in set(labeled)]

    target = _oof_cox_risk(expr, durations, events, n_pca, penalizer=0.1,
                           folds=5, random_state=random_state)

    reducer = umap.UMAP(
        n_components=n_components,
        target_metric="l2",  # continuous supervision toward out-of-fold risk
        random_state=random_state,
        n_jobs=1,
    )
    reducer.fit(comps_all.loc[labeled].to_numpy(), y=target.loc[labeled].to_numpy())

    coords = pd.DataFrame(index=comps_all.index, columns=range(n_components), dtype=float)
    coords.loc[labeled] = reducer.embedding_
    if unlabeled:
        # Placed by expression structure alone (a generalization check).
        coords.loc[unlabeled] = reducer.transform(comps_all.loc[unlabeled].to_numpy())
    logger.info(
        "supervised embed: %d labeled (survival), %d unlabeled (projected).",
        len(labeled),
        len(unlabeled),
    )
    return coords.to_numpy(dtype=float)


def _oof_cox_risk(expr, durations, events, n_pca, penalizer, folds, random_state):
    """Out-of-fold Cox risk per labeled sample (PCA+Cox refit per training fold).

    Returns a Series of held-out risk indexed by GSM (``expr.index``). Any sample
    left unpredicted (e.g. a fold with no training events) is filled from a global
    fit so every labeled sample carries a target.
    """
    from sklearn.model_selection import KFold  # lazy

    n = expr.shape[0]
    oof = pd.Series(np.nan, index=expr.index, name="risk", dtype=float)
    k = min(folds, n)
    if k >= 2:
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        for tr_idx, te_idx in kf.split(np.arange(n)):
            tr_expr = expr.iloc[tr_idx]
            te_expr = expr.iloc[te_idx]
            tr_dur, tr_evt = durations.iloc[tr_idx], events.iloc[tr_idx]
            if tr_evt.sum() < 1:
                continue
            pca, center, labels, tr_comp = _fit_pca_on(tr_expr, n_pca)
            cox = _fit_cox(tr_comp, tr_dur, tr_evt, labels, penalizer)
            te_comp = pd.DataFrame(
                pca.transform(te_expr.to_numpy(dtype=float) - center),
                index=te_expr.index,
                columns=labels,
            )
            oof.iloc[te_idx] = np.asarray(cox.predict_log_partial_hazard(te_comp)).ravel()

    if oof.isna().any():
        # Fallback for unassigned samples (few labels / skipped folds).
        pca, center, labels, comp = _fit_pca_on(expr, n_pca)
        cox = _fit_cox(comp, durations, events, labels, penalizer)
        pred = pd.Series(
            np.asarray(cox.predict_log_partial_hazard(comp)).ravel(), index=expr.index
        )
        oof[oof.isna()] = pred[oof.isna()]
    return oof


def _align_survival(matrix: pd.DataFrame, survival: pd.DataFrame):
    """Select samples present in both, with positive duration + defined event.

    Returns ``(expr_samples_by_genes_df, durations, events)`` aligned by GSM.
    """
    if "survival_days" not in survival.columns or "event" not in survival.columns:
        raise ValueError("survival must have 'survival_days' and 'event' columns.")

    shared = [g for g in matrix.columns if g in survival.index]
    durations = pd.to_numeric(survival.loc[shared, "survival_days"], errors="coerce")
    events = pd.to_numeric(survival.loc[shared, "event"], errors="coerce")
    keep = durations.notna() & events.notna() & (durations > 0)
    kept = [g for g, k in zip(shared, keep) if k]
    if len(kept) < 2:
        raise ValueError(
            f"Only {len(kept)} samples have usable survival — cannot fit a risk model."
        )

    expr = pd.DataFrame(matrix[kept].to_numpy(dtype=float).T, index=pd.Index(kept, name="sample"), columns=matrix.index)
    return expr, durations.loc[kept], events.loc[kept].astype(int)


def _fit_pca_on(expr: pd.DataFrame, n_pca: int):
    """Center genes and fit a PCA on samples x genes. Returns (pca, center, labels, components_df)."""
    from sklearn.decomposition import PCA  # lazy

    X = expr.to_numpy(dtype=float)
    center = np.nanmean(X, axis=0)
    n_comp = min(n_pca, X.shape[0] - 1, X.shape[1])
    n_comp = max(n_comp, 1)
    pca = PCA(n_components=n_comp, svd_solver="full", random_state=0)
    comps = pca.fit_transform(X - center)
    labels = [f"PC{i + 1}" for i in range(n_comp)]
    components = pd.DataFrame(comps, index=expr.index, columns=labels)
    return pca, center, labels, components


def _fit_cox(components, durations, events, comp_labels, penalizer):
    """Fit a penalized Cox on PCA components. Returns the fitted CoxPHFitter."""
    from lifelines import CoxPHFitter  # lazy

    df = components.copy()
    df["_duration"] = durations.to_numpy()
    df["_event"] = events.to_numpy()
    cox = CoxPHFitter(penalizer=penalizer)
    cox.fit(df, duration_col="_duration", event_col="_event")
    return cox


def _project_components(risk_model: RiskModel, matrix: pd.DataFrame) -> pd.DataFrame:
    """Project a matrix's samples into the risk model's PCA component space."""
    aligned = matrix.reindex(risk_model.genes)
    missing = int(aligned.isna().all(axis=1).sum())
    if missing:
        logger.warning("predict_risk: %d model genes absent — filled with 0.", missing)
        aligned = aligned.fillna(0.0)
    X = aligned.to_numpy(dtype=float).T - risk_model.center
    comps = risk_model.pca.transform(X)
    return pd.DataFrame(comps, index=matrix.columns, columns=risk_model.component_labels)


def _rbf_interpolate(points, z, mesh):
    from scipy.interpolate import RBFInterpolator  # lazy

    # Smoothing keeps the surface readable rather than spiking at each point.
    interp = RBFInterpolator(points, z, smoothing=1.0, kernel="thin_plate_spline")
    return interp(mesh)


def _clip_to_hull(points, mesh, ZZ):
    """Set grid cells outside the convex hull of ``points`` to ``NaN``.

    Uses ``Delaunay.find_simplex`` as the inside test (>= 0 means inside). Falls
    back to the unclipped grid if the points are degenerate (collinear), where a
    hull is undefined.
    """
    from scipy.spatial import Delaunay, QhullError  # lazy

    try:
        tri = Delaunay(points)
    except QhullError:
        logger.warning("_clip_to_hull: degenerate hull — returning unclipped grid.")
        return ZZ
    inside = (tri.find_simplex(mesh) >= 0).reshape(ZZ.shape)
    clipped = ZZ.copy()
    clipped[~inside] = np.nan
    return clipped


def _griddata_interpolate(points, z, mesh):
    from scipy.interpolate import griddata  # lazy

    zz = griddata(points, z, mesh, method="linear")
    # Fill the convex-hull exterior (NaN from linear) with nearest.
    nan = np.isnan(zz)
    if nan.any():
        zz[nan] = griddata(points, z, mesh[nan], method="nearest")
    return zz
