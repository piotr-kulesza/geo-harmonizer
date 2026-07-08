#!/usr/bin/env python3
"""Landscape validation + look gate (networked; the author runs it, never the sandbox).

Over the cached demo set: harmonize -> ComBat -> embed to a 2D map, fit + CV the
survival-risk model on the cohorts that have survival, predict risk for ALL
samples, and render a static 3D risk surface. The printed cross-validated
C-index is the GATE: >= 0.6 means the landscape is meaningful.

Usage:
    python scripts/landscape_check.py [ACCESSION ...]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import fetch_gse  # noqa: E402
from core.harmonize import map_probes, to_log2  # noqa: E402
from core.merge import merge  # noqa: E402
from core.combat import combat  # noqa: E402
from core.metadata import parse_metadata  # noqa: E402
from core.landscape import (  # noqa: E402
    embed,
    fit_risk,
    cv_cindex,
    predict_risk,
    risk_layer,
    surface,
    LandscapeModel,
)

DEFAULT_ACCESSIONS = ["GSE9891", "GSE26712", "GSE26193"]
OUTPUT_DIR = Path("outputs")
GATE = 0.60


def _load_dotenv() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accessions", nargs="*", default=DEFAULT_ACCESSIONS)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--collapse", choices=["max", "mean"], default="max")
    parser.add_argument(
        "--embed",
        choices=["pca", "umap", "supervised"],
        default="supervised",
        help="2D map layout (default: outcome-supervised UMAP toward survival)",
    )
    parser.add_argument("--no-combat", action="store_true", help="skip ComBat (debug)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    _load_dotenv()

    accessions = args.accessions or DEFAULT_ACCESSIONS
    matrices, raw_meta = {}, {}
    for accession in accessions:
        result = fetch_gse(accession, cache_dir=args.cache_dir)
        if not result.ok:
            print(f"{accession}: SKIPPED — {result.message}")
            continue
        platform = result.platform_ids[0] if result.platform_ids else "unknown"
        matrices[accession] = map_probes(
            to_log2(result.expression), platform_id=platform, collapse=args.collapse
        )
        if result.metadata is not None:
            raw_meta[accession] = result.metadata
        print(f"{accession}: {matrices[accession].shape[0]} genes x {matrices[accession].shape[1]} samples")

    if len(matrices) < 2:
        print("\nNeed >=2 harmonized series — stopping.")
        return 1

    merged, batch = merge(matrices)
    substrate = merged if args.no_combat else combat(merged, batch)
    print(f"\nsubstrate: {substrate.shape[0]} genes x {substrate.shape[1]} samples")

    # Standardized survival via the Day-3 metadata pass (needs ANTHROPIC_API_KEY).
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — cannot standardize survival. Aborting.")
        return 2
    mapping = parse_metadata(raw_meta)
    survival = mapping.standardized[["survival_days", "event"]].dropna()
    survival = survival[survival.index.isin(substrate.columns)]
    print(f"samples with usable survival: {survival.shape[0]}")

    if survival.shape[0] < 10:
        print("Too few survival samples to validate a risk model — stopping.")
        return 1

    # 2D map (fixed terrain) + risk model.
    coords = embed(substrate, n_components=2, method=args.embed, survival=survival)
    cindex = cv_cindex(substrate, survival)
    model = fit_risk(substrate, survival)
    risk = predict_risk(model, substrate)

    # Which samples supervised the map vs. were only projected onto it.
    supervised_gsm = [g for g in survival.index if g in substrate.columns]
    n_sup = len(supervised_gsm)
    n_unlabeled = substrate.shape[1] - n_sup
    print(f"\nembedding: {args.embed}")
    if args.embed == "supervised":
        print(
            f"  {n_sup} samples supervised the map; {n_unlabeled} were projected "
            f"by expression structure only."
        )
        unlabeled_series = sorted(
            set(batch.reindex([g for g in substrate.columns if g not in set(supervised_gsm)]).dropna())
        )
        if unlabeled_series:
            print(
                f"  series without survival (NOT used to supervise, placement tests "
                f"generalization): {', '.join(unlabeled_series)}"
            )

    print(f"\ncross-validated C-index: {cindex:.3f}")
    print(f"risk range: [{risk.min():.3f}, {risk.max():.3f}] over {risk.shape[0]} samples")
    if cindex >= GATE:
        print(f"GATE PASSED (>= {GATE}) — the landscape is meaningful.")
    else:
        print(f"GATE FAILED (< {GATE}) — risk not predictive; rethink before the interactive layer.")

    # Assemble + render.
    landscape = LandscapeModel(coords=coords, risk_model=model)
    landscape.add_layer(risk_layer(model, substrate))
    OUTPUT_DIR.mkdir(exist_ok=True)
    _render_3d(coords, risk, OUTPUT_DIR / "landscape_risk.png")
    _render_2d(coords, batch, risk, OUTPUT_DIR / "landscape_map.png")
    print(f"\nwrote {OUTPUT_DIR/'landscape_risk.png'} and {OUTPUT_DIR/'landscape_map.png'}")
    return 0


def _render_3d(coords, risk, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from core.landscape import surface, HeightLayer

    XX, YY, ZZ = surface(coords, HeightLayer("risk", risk), grid=60, method="rbf")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(XX, YY, ZZ, cmap="magma", alpha=0.7, linewidth=0, antialiased=True)
    r = risk.reindex(coords.index)
    ax.scatter(coords["x"], coords["y"], r, c=r, cmap="magma", s=16, depthshade=True)
    ax.set_xlabel("map-x")
    ax.set_ylabel("map-y")
    ax.set_zlabel("predicted risk")
    ax.set_title("Disease risk landscape (height = predicted survival risk)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _render_2d(coords, batch, risk, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    labels = batch.reindex(coords.index)
    for accession, grp in coords.groupby(labels):
        ax1.scatter(grp["x"], grp["y"], label=str(accession), s=16, alpha=0.8)
    ax1.legend(title="dataset", fontsize=8)
    ax1.set_title("Map coloured by dataset")

    sc = ax2.scatter(coords["x"], coords["y"], c=risk.reindex(coords.index), cmap="magma", s=16)
    fig.colorbar(sc, ax=ax2, label="predicted risk")
    ax2.set_title("Map coloured by risk")
    for ax in (ax1, ax2):
        ax.set_xlabel("map-x")
        ax.set_ylabel("map-y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
