"""Fixed-projection PCA over samples. [Epic 3.4 support — pure, reusable]

PCA here operates over SAMPLES: each sample is a point in gene space, so we
transpose a genes x samples matrix to samples x genes and reduce to 2 components.

The projection is FIXED (fit once, reused). This is the crux of the demo's
chaos->order beat (CLAUDE.md): if you refit PCA for every state — pre/post
ComBat, or each incremental subset — sign/axis ambiguity makes existing points
flip and rotate, so "adding a series" or "turning on ComBat" reads as random
motion. Instead we fit ONE basis and project every state into it, so points
*move* meaningfully within one coordinate frame.

Recommended demo basis: fit once on the full RAW merged matrix (all samples,
log2, pre-ComBat), then project both the raw merge and the ComBat-corrected merge
into that same basis — "before" shows clouds separated along the batch axis,
"after" shows them collapse along it.

Iron rule: pure logic, no web/MCP/UI. sklearn is imported lazily.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PCAModel:
    """A fitted, reusable PCA projection over samples.

    Holds everything needed to project any (sub)matrix into the same fixed basis:
    the fitted sklearn ``PCA``, the exact gene order the basis was built on, and
    the per-gene centering (and optional scaling) applied before the transform.

    Attributes:
        pca: the fitted ``sklearn.decomposition.PCA``.
        genes: gene order (matrix index) the basis was fit on.
        center: per-gene mean subtracted before transforming.
        scale: per-gene scale divided out (all ones if scaling was off).
        component_labels: column names for the projected coords (``PC1``, ...).
    """

    pca: object
    genes: pd.Index
    center: np.ndarray
    scale: np.ndarray
    component_labels: list[str]

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Fraction of variance captured by each retained component."""
        return np.asarray(self.pca.explained_variance_ratio_)


def fit_pca(
    matrix: pd.DataFrame,
    n_components: int = 2,
    scale: bool = False,
) -> PCAModel:
    """Fit a fixed PCA basis over the samples of a genes x samples matrix.

    Args:
        matrix: genes (rows) x samples (GSM columns).
        n_components: number of principal components to retain (default 2).
        scale: if True, standardize each gene to unit variance in addition to
            centering. Default False (center only).

    Returns:
        A :class:`PCAModel` capturing the fitted basis and the centering/scaling,
        so :func:`project` reproduces coordinates for any subset.
    """
    from sklearn.decomposition import PCA  # lazy import

    genes = matrix.index
    # samples x genes
    samples_by_genes = matrix.to_numpy(dtype=float).T

    center = np.nanmean(samples_by_genes, axis=0)
    if scale:
        std = np.nanstd(samples_by_genes, axis=0)
        std[std == 0] = 1.0
    else:
        std = np.ones(samples_by_genes.shape[1], dtype=float)

    standardized = (samples_by_genes - center) / std

    pca = PCA(n_components=n_components, svd_solver="full", random_state=0)
    pca.fit(standardized)

    labels = [f"PC{i + 1}" for i in range(n_components)]
    logger.info(
        "fit_pca: %d samples x %d genes -> %d components (var %s).",
        samples_by_genes.shape[0],
        samples_by_genes.shape[1],
        n_components,
        np.round(pca.explained_variance_ratio_, 3).tolist(),
    )
    return PCAModel(
        pca=pca,
        genes=genes,
        center=center,
        scale=std,
        component_labels=labels,
    )


def project(matrix: pd.DataFrame, model: PCAModel) -> pd.DataFrame:
    """Project a matrix's samples into a fixed PCA basis.

    ``matrix`` is reindexed to the model's gene order first, so any subset of
    samples (or a differently-ordered gene set) projects consistently.

    Args:
        matrix: genes (rows) x samples (GSM columns) to project. Must contain
            every gene the model was fit on.
        model: a :class:`PCAModel` from :func:`fit_pca`.

    Returns:
        samples (rows, indexed by GSM) x component columns (``PC1``, ``PC2``, ...).

    Raises:
        ValueError: if ``matrix`` is missing genes the model was fit on.
    """
    missing = model.genes.difference(matrix.index)
    if len(missing) > 0:
        raise ValueError(
            f"Matrix is missing {len(missing)} genes the PCA basis was fit on "
            f"(e.g. {list(missing[:5])}). Cannot project into a fixed basis."
        )

    aligned = matrix.reindex(model.genes)
    samples_by_genes = aligned.to_numpy(dtype=float).T
    standardized = (samples_by_genes - model.center) / model.scale

    coords = model.pca.transform(standardized)
    return pd.DataFrame(
        coords,
        index=pd.Index(aligned.columns, name="sample"),
        columns=model.component_labels,
    )
