#!/usr/bin/env python3
"""Epic 3.4 HARD GATE: does the demo set produce the chaos->order PCA?

Networked (via the cache); the author runs this, never the build sandbox. It
proves the hero visual before any front-end work: clouds must visibly separate by
dataset BEFORE ComBat and merge AFTER, in ONE fixed PCA basis.

Pipeline over the cached demo set (default GSE9891 GSE26712 GSE26193):
  fetch (cache) -> to_log2 -> map_probes -> merge -> (merged, batch)
  fit_pca(raw merged)            # the single fixed basis
  project(raw merged)            -> coords_before
  combat(merged, batch)          -> project -> coords_after   (same basis)

Outputs:
  - outputs/pca_before.png, outputs/pca_after.png (coloured by dataset, shared axes)
  - printed variance explained + silhouette(coords, batch) BEFORE vs AFTER.
    Silhouette should DROP after ComBat (less separable by dataset) — the
    quantitative proof behind the visual.

Usage:
    python scripts/pca_check.py [ACCESSION ...] [--serous-only]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import fetch_gse  # noqa: E402
from core.harmonize import map_probes, to_log2  # noqa: E402
from core.merge import merge  # noqa: E402
from core.combat import combat  # noqa: E402
from core.pca import fit_pca, project  # noqa: E402

DEFAULT_ACCESSIONS = ["GSE9891", "GSE26712", "GSE26193"]
OUTPUT_DIR = Path("outputs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accessions", nargs="*", default=DEFAULT_ACCESSIONS)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--collapse", choices=["max", "mean"], default="max")
    parser.add_argument(
        "--serous-only",
        action="store_true",
        help=(
            "Placeholder (no-op for now): later restrict to comparable malignant "
            "serous tumors for a cleaner biology-driven merge."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.serous_only:
        print("[--serous-only requested — no-op placeholder; using all samples]")

    accessions = args.accessions or DEFAULT_ACCESSIONS
    matrices = {}
    for accession in accessions:
        result = fetch_gse(accession, cache_dir=args.cache_dir)
        if not result.ok:
            print(f"{accession}: SKIPPED — {result.message}")
            continue
        platform = result.platform_ids[0] if result.platform_ids else "unknown"
        genes = map_probes(
            to_log2(result.expression), platform_id=platform, collapse=args.collapse
        )
        matrices[accession] = genes
        print(f"{accession}: {genes.shape[0]} genes x {genes.shape[1]} samples [{platform}]")

    if len(matrices) < 2:
        print("\nNeed >=2 harmonized series — stopping.")
        return 1

    merged, batch = merge(matrices)
    print(f"\nshared genes: {merged.shape[0]}  |  merged {merged.shape[0]} x {merged.shape[1]}")

    # One fixed basis, fit on the RAW merged matrix.
    model = fit_pca(merged, n_components=2)
    coords_before = project(merged, model)

    corrected = combat(merged, batch)
    coords_after = project(corrected, model)

    var = model.explained_variance_ratio
    sil_before = _silhouette(coords_before, batch)
    sil_after = _silhouette(coords_after, batch)

    print(f"\nvariance explained: PC1={var[0]:.1%}  PC2={var[1]:.1%}")
    print(f"batch silhouette  BEFORE ComBat: {sil_before:+.3f}")
    print(f"batch silhouette   AFTER ComBat: {sil_after:+.3f}")
    verdict = "PASS — clouds merge" if sil_after < sil_before else "CHECK — did not drop"
    print(f"separation dropped after ComBat? {verdict}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    _scatter(coords_before, batch, "Before ComBat", OUTPUT_DIR / "pca_before.png",
             _shared_limits(coords_before, coords_after))
    _scatter(coords_after, batch, "After ComBat", OUTPUT_DIR / "pca_after.png",
             _shared_limits(coords_before, coords_after))
    print(f"\nwrote {OUTPUT_DIR / 'pca_before.png'} and {OUTPUT_DIR / 'pca_after.png'}")
    return 0


def _silhouette(coords, batch):
    """silhouette_score of the 2D coords labelled by batch (dataset separability)."""
    from sklearn.metrics import silhouette_score

    labels = batch.reindex(coords.index).astype("category").cat.codes.to_numpy()
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(silhouette_score(coords.to_numpy(), labels))


def _shared_limits(a, b):
    both = a.to_numpy().tolist() + b.to_numpy().tolist()
    import numpy as np

    arr = np.asarray(both)
    pad_x = 0.05 * (arr[:, 0].max() - arr[:, 0].min() or 1)
    pad_y = 0.05 * (arr[:, 1].max() - arr[:, 1].min() or 1)
    return (
        (arr[:, 0].min() - pad_x, arr[:, 0].max() + pad_x),
        (arr[:, 1].min() - pad_y, arr[:, 1].max() + pad_y),
    )


def _scatter(coords, batch, title, path, limits):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = batch.reindex(coords.index)
    fig, ax = plt.subplots(figsize=(6, 5))
    for accession, group in coords.groupby(labels):
        ax.scatter(group["PC1"], group["PC2"], label=str(accession), s=18, alpha=0.8)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(title="dataset", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
