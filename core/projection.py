"""Progressive PCA into a FIXED basis — the harmonization hero ("Act 1"). [pure]

The signature demo (CLAUDE.md → "The signature demo moment"): add series one by
one and watch samples cluster into separate clouds BY BATCH; then turn on ComBat
and watch them merge BY BIOLOGY. Chaos -> order.

**The non-negotiable correctness rule — fixed-projection PCA.** PCA has sign/axis
ambiguity: refitting after each added dataset makes existing points flip and
rotate, so the animation reads as noise — the opposite of the thesis. So we fit
the PCA basis ONCE, on the final full merged matrix, and PROJECT every progressive
subset AND both ComBat states into that same fixed 2D frame. Every position is
precomputed here; the frontend animation is pure interpolation between these fixed
positions and can never drift.

Iron rule: pure logic, no web/UI imports, no ``print``. Heavy libs (scikit-learn)
are lazy-imported so ``core`` still imports without them. Deterministic: fixed seed
+ ``svd_solver="full"`` -> identical output across runs (fit-once also fixes the
sign, so a sample's coords are identical whichever step first reveals it).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def progressive_projection(
    raw_merged: pd.DataFrame,
    batch: "pd.Series | dict",
    corrected: Optional[pd.DataFrame] = None,
    order: Optional[list] = None,
    n_components: int = 2,
    random_state: int = 0,
) -> dict:
    """Project every cumulative subset (and both ComBat states) into ONE PCA basis.

    Args:
        raw_merged: genes (rows) x samples (GSM columns) — the uncorrected merged
            matrix. The PCA basis is fit ONCE on this (all samples).
        batch: per-sample source accession (GSM -> accession), a Series or dict.
        corrected: the ComBat-corrected matrix (same shape/labels) or ``None``.
            When given, its matching subset is projected into the SAME basis with
            the SAME centering, so the raw->ComBat morph lives in identical axes.
        order: reveal order of accessions. Default: descending sample count, then
            accession name — deterministic.
        n_components: PCA components (2 for the 2D scatter).
        random_state: PCA seed (determinism).

    Returns (JSON-serializable):
        ``{axes:{explained_variance:[pc1,pc2]}, order:[acc...], batch:{gsm:acc},
        steps:[{k, included:[acc...], raw:{coords:{gsm:[x,y]}, silhouette:float|None},
        combat:{...}|null}]}``. ``silhouette`` is the batch-separation score of the
        2D coords labelled by batch (high = clouds apart; ~0 = merged); ``None`` when
        it is undefined (a single batch at step 1).
    """
    from sklearn.decomposition import PCA  # lazy

    samples = list(raw_merged.columns)
    batch_s = pd.Series(dict(batch)) if isinstance(batch, dict) else pd.Series(batch)
    batch_s = batch_s.reindex(samples).astype(object)

    if order is None:
        counts = batch_s.value_counts()
        order = sorted(counts.index, key=lambda a: (-int(counts[a]), str(a)))
    else:
        order = [a for a in order if a in set(batch_s.values)]

    # Fit the ONE fixed basis on all samples (center per gene, no scaling — matches
    # the landscape PCA). transform() reuses this basis for every subset/mode.
    X = raw_merged.to_numpy(dtype=float).T
    center = np.nanmean(X, axis=0)
    Xc = np.nan_to_num(X - center)
    n_comp = max(1, min(n_components, *Xc.shape))
    pca = PCA(n_components=n_comp, svd_solver="full", random_state=random_state)
    pca.fit(Xc)

    raw_all = pca.transform(Xc)
    raw_coords = {g: [round(float(raw_all[i, 0]), 4), round(float(raw_all[i, 1]), 4)] for i, g in enumerate(samples)}

    combat_coords = None
    if corrected is not None:
        Xc2 = np.nan_to_num(corrected.reindex(columns=samples).to_numpy(dtype=float).T - center)
        comb_all = pca.transform(Xc2)
        combat_coords = {
            g: [round(float(comb_all[i, 0]), 4), round(float(comb_all[i, 1]), 4)]
            for i, g in enumerate(samples)
        }

    ev = pca.explained_variance_ratio_
    steps = []
    for k in range(1, len(order) + 1):
        included = list(order[:k])
        include_set = set(included)
        gsms = [g for g in samples if batch_s[g] in include_set]
        labels = [str(batch_s[g]) for g in gsms]

        raw_block = {
            "coords": {g: raw_coords[g] for g in gsms},
            "silhouette": _silhouette([raw_coords[g] for g in gsms], labels),
        }
        combat_block = None
        if combat_coords is not None:
            combat_block = {
                "coords": {g: combat_coords[g] for g in gsms},
                "silhouette": _silhouette([combat_coords[g] for g in gsms], labels),
            }
        steps.append({"k": k, "included": included, "raw": raw_block, "combat": combat_block})

    logger.info(
        "progressive_projection: %d samples, %d datasets, corrected=%s.",
        len(samples),
        len(order),
        corrected is not None,
    )
    return {
        "axes": {"explained_variance": [round(float(ev[0]), 4), round(float(ev[1]), 4) if len(ev) > 1 else 0.0]},
        "order": [str(a) for a in order],
        "batch": {g: str(batch_s[g]) for g in samples},
        "steps": steps,
    }


def _silhouette(coords: list, labels: list) -> Optional[float]:
    """Batch-separation of 2D coords labelled by dataset; ``None`` when undefined.

    silhouette_score needs ``2 <= n_labels <= n_samples - 1``; a single batch (step
    1) or a degenerate split returns ``None`` so the UI simply shows no number yet.
    """
    uniq = set(labels)
    if len(uniq) < 2 or len(uniq) > len(labels) - 1:
        return None
    from sklearn.metrics import silhouette_score  # lazy

    score = silhouette_score(np.asarray(coords, dtype=float), labels)
    return round(float(score), 4)
