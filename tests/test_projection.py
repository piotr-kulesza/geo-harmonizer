"""Offline tests for core.projection — the fixed-projection correctness gate. No network."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from core.projection import progressive_projection


def _planted(seed: int = 0):
    """3 datasets: a SHARED biological axis + a STRONG per-batch offset.

    Raw -> clouds separate by batch (high silhouette). A simulated ComBat (remove
    each batch's per-gene mean, restore the global mean) merges them (silhouette
    collapses). Returns (raw, corrected, batch).
    """
    rng = np.random.default_rng(seed)
    genes = [f"g{i}" for i in range(60)]
    frames, batch = [], {}
    for bi, acc in enumerate(["A", "B", "C"]):
        n = 40
        bio = rng.normal(size=n)  # shared biology (comparable across datasets)
        M = np.zeros((60, n))
        for i in range(20):
            M[i] = 2.0 * bio + rng.normal(scale=0.3, size=n)
        for i in range(20, 60):
            M[i] = rng.normal(scale=0.3, size=n)
        M += (bi - 1) * 6.0  # strong batch offset (all genes)
        cols = [f"{acc}{k}" for k in range(n)]
        for c in cols:
            batch[c] = acc
        frames.append(pd.DataFrame(M, index=genes, columns=cols))

    raw = pd.concat(frames, axis=1)
    gm = raw.mean(axis=1)
    corrected = raw.copy()
    for acc in ["A", "B", "C"]:
        cols = [c for c in raw.columns if batch[c] == acc]
        corrected[cols] = raw[cols].sub(raw[cols].mean(axis=1), axis=0).add(gm, axis=0)
    return raw, corrected, batch


def test_basis_is_fixed_sample_coords_identical_across_steps():
    # THE correctness gate: a sample present from step 1 keeps EXACTLY the same
    # projected coords as later datasets are added — proves the PCA basis is fixed
    # (no per-step refit, no flip/rotate).
    raw, corrected, batch = _planted()
    prog = progressive_projection(raw, batch, corrected=corrected)

    first_dataset = prog["order"][0]
    gsm = next(g for g, acc in prog["batch"].items() if acc == first_dataset)
    coords_per_step = [
        step["raw"]["coords"][gsm] for step in prog["steps"] if gsm in step["raw"]["coords"]
    ]
    assert len(coords_per_step) == len(prog["steps"])  # present at every step
    for c in coords_per_step[1:]:
        assert c == coords_per_step[0]  # identical, not merely close


def test_combat_reduces_batch_separation():
    raw, corrected, batch = _planted()
    prog = progressive_projection(raw, batch, corrected=corrected)
    last = prog["steps"][-1]
    assert last["raw"]["silhouette"] is not None
    assert last["combat"]["silhouette"] is not None
    # Raw clouds are far apart; ComBat collapses them toward ~0.
    assert last["raw"]["silhouette"] > last["combat"]["silhouette"]
    assert last["raw"]["silhouette"] > 0.3
    assert last["combat"]["silhouette"] < 0.2


def test_single_batch_step_has_no_silhouette():
    raw, corrected, batch = _planted()
    prog = progressive_projection(raw, batch, corrected=corrected)
    assert prog["steps"][0]["raw"]["silhouette"] is None  # one cloud -> undefined


def test_shape_and_serializable():
    raw, corrected, batch = _planted()
    prog = progressive_projection(raw, batch, corrected=corrected)
    assert set(prog) == {"axes", "order", "batch", "steps"}
    assert len(prog["axes"]["explained_variance"]) == 2
    assert prog["order"] == ["A", "B", "C"]  # equal sizes -> alphabetical
    assert len(prog["steps"]) == 3
    step = prog["steps"][-1]
    assert set(step) == {"k", "included", "raw", "combat"}
    assert set(step["raw"]) == {"coords", "silhouette"}
    json.dumps(prog)  # fully JSON-serializable (no NaN/np types)


def test_corrected_optional():
    raw, _, batch = _planted()
    prog = progressive_projection(raw, batch)  # no ComBat state
    assert all(s["combat"] is None for s in prog["steps"])


def test_deterministic_across_runs():
    raw, corrected, batch = _planted()
    a = progressive_projection(raw, batch, corrected=corrected)
    b = progressive_projection(raw, batch, corrected=corrected)
    assert json.dumps(a) == json.dumps(b)
