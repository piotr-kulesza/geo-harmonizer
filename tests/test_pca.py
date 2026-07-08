"""Offline tests for core.pca — fixed-projection PCA over samples."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.pca import fit_pca, project


def _matrix(n_genes=8, n_samples=10, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_genes, n_samples))
    genes = [f"G{i}" for i in range(n_genes)]
    samples = [f"GSM{i}" for i in range(n_samples)]
    return pd.DataFrame(data, index=genes, columns=samples)


def test_fit_and_project_shape_and_labels():
    matrix = _matrix()
    model = fit_pca(matrix, n_components=2)
    coords = project(matrix, model)

    assert coords.shape == (matrix.shape[1], 2)  # samples x 2
    assert list(coords.columns) == ["PC1", "PC2"]
    assert list(coords.index) == list(matrix.columns)
    assert coords.index.name == "sample"


def test_projection_is_deterministic():
    matrix = _matrix()
    model = fit_pca(matrix)
    a = project(matrix, model)
    b = project(matrix, model)
    pd.testing.assert_frame_equal(a, b)


def test_subset_projects_into_same_fixed_basis():
    matrix = _matrix()
    model = fit_pca(matrix)
    full = project(matrix, model)

    subset = matrix[["GSM1", "GSM4", "GSM7"]]
    sub = project(subset, model)

    # Fixed basis: subset samples land at exactly the same coords as in the full
    # projection (projection does not depend on which samples are present).
    pd.testing.assert_frame_equal(sub, full.loc[["GSM1", "GSM4", "GSM7"]])


def test_gene_reordering_projects_consistently():
    matrix = _matrix()
    model = fit_pca(matrix)
    reordered = matrix.loc[matrix.index[::-1]]  # reverse gene order
    pd.testing.assert_frame_equal(project(reordered, model), project(matrix, model))


def test_project_raises_on_missing_genes():
    matrix = _matrix()
    model = fit_pca(matrix)
    with pytest.raises(ValueError, match="missing"):
        project(matrix.drop(index="G0"), model)
