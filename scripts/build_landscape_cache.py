#!/usr/bin/env python3
"""Precompute the landscape bundle the web backend serves (networked; author-run).

Runs the cached pipeline end to end — fetch(cached) -> harmonize -> merge ->
(ComBat) -> supervised embed -> fit_risk -> CV C-index — then persists a
network-free bundle so the FastAPI backend loads instantly:

    outputs/landscape_cache/bundle.pkl   fitted model + matrix + standardized meta
    outputs/landscape_payload.json       static payload (frontend fallback, no API)

Prints the cross-validated C-index and the held-out series (only cohorts with
ZERO usable-survival samples). NEVER run on the demo host — the sandbox/tests do
not touch the network; the author runs this once to bake the cache.

Usage:
    python scripts/build_landscape_cache.py [ACCESSION ...] [--out outputs/landscape_cache]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import fetch_gse  # noqa: E402
from core.harmonize import map_probes, to_log2  # noqa: E402
from core.merge import merge  # noqa: E402
from core.combat import combat  # noqa: E402
from core.metadata import parse_metadata  # noqa: E402
from core.landscape import (  # noqa: E402
    LandscapeModel,
    cv_cindex,
    embed,
    fit_risk,
    held_out_series,
    landscape_payload,
    usable_survival_ids,
)
from core.projection import progressive_projection  # noqa: E402
from web.api import Bundle, DEFAULT_GRID, _build_base_layers, save_bundle  # noqa: E402

DEFAULT_ACCESSIONS = ["GSE9891", "GSE26712", "GSE26193"]
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
    parser.add_argument("--cache-dir", default="data/cache", help="GEO fetch cache")
    parser.add_argument("--out", default=os.environ.get("LANDSCAPE_CACHE", "outputs/landscape_cache"))
    parser.add_argument("--collapse", choices=["max", "mean"], default="max")
    parser.add_argument("--no-combat", action="store_true", help="skip ComBat (debug)")
    parser.add_argument("--grid", type=int, default=DEFAULT_GRID)
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
    combat_substrate = None if args.no_combat else combat(merged, batch)
    substrate = merged if args.no_combat else combat_substrate
    print(f"\nsubstrate: {substrate.shape[0]} genes x {substrate.shape[1]} samples")

    # Act 1: fixed-basis progressive PCA (raw clouds separate by batch -> ComBat
    # merges them). Fit ONCE on the raw merged matrix; project both states.
    progression = progressive_projection(merged, batch, corrected=combat_substrate)
    last = progression["steps"][-1]
    print(
        "PCA progression: {} datasets, raw silhouette {} -> combat {}".format(
            len(progression["order"]),
            last["raw"]["silhouette"],
            last["combat"]["silhouette"] if last["combat"] else "n/a",
        )
    )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — cannot standardize survival. Aborting.")
        return 2
    mapping = parse_metadata(raw_meta)
    standardized = mapping.standardized

    usable = usable_survival_ids(standardized, substrate.columns)
    survival = standardized.loc[usable, ["survival_days", "event"]]
    print(f"samples with usable survival: {len(usable)}")
    if len(usable) < 10:
        print("Too few survival samples to validate a risk model — stopping.")
        return 1

    # Supervised map + validated risk model (the height).
    coords = embed(substrate, n_components=2, method="supervised", survival=survival)
    cindex = cv_cindex(substrate, survival)
    model = fit_risk(substrate, survival)
    landscape = LandscapeModel(coords=coords, risk_model=model, cindex=cindex)

    held = held_out_series(standardized, substrate.columns)
    print(f"\ncross-validated C-index: {cindex:.3f}")
    print(f"held out (no usable survival, placement tests generalization): {', '.join(held) or '(none)'}")
    if cindex >= GATE:
        print(f"GATE PASSED (>= {GATE}) — the landscape is meaningful.")
    else:
        print(f"GATE FAILED (< {GATE}) — risk not predictive; investigate before shipping.")

    # Persist the network-free bundle + static-file fallbacks.
    bundle = Bundle(model=landscape, matrix=substrate, samples_meta=standardized, pca=progression)
    bundle_path = save_bundle(bundle, args.out)

    base_layers = _build_base_layers(bundle)
    payload = landscape_payload(landscape, standardized, base_layers, grid=args.grid)
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_path = out_dir / "landscape_payload.json"
    payload_path.write_text(json.dumps(payload))
    pca_path = out_dir / "pca_progression.json"
    pca_path.write_text(json.dumps(progression))

    print(f"\nwrote {bundle_path}")
    print(f"wrote {payload_path} ({len(payload['samples'])} samples, "
          f"{len(payload['height_options'])} height options)")
    print(f"wrote {pca_path} ({len(progression['steps'])} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
