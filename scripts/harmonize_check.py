#!/usr/bin/env python3
"""Epic 2.5 / 3.2 checkpoint: harmonize the demo set and report the shared-gene count.

Runs against the cached demo series (network on a cold cache; the author runs
this, never the build sandbox). For each series: fetch (cache) -> to_log2 ->
map_probes -> collect. Then merge and report the shared-gene count — the figure
the PCA hero visual depends on.

Usage:
    python scripts/harmonize_check.py [ACCESSION ...]
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

DEFAULT_ACCESSIONS = ["GSE9891", "GSE26712", "GSE26193"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accessions", nargs="*", default=DEFAULT_ACCESSIONS)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--collapse", choices=["max", "mean"], default="max")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    accessions = args.accessions or DEFAULT_ACCESSIONS
    matrices = {}
    for accession in accessions:
        result = fetch_gse(accession, cache_dir=args.cache_dir)
        if not result.ok:
            print(f"{accession}: SKIPPED — {result.message}")
            continue
        platform = result.platform_ids[0] if result.platform_ids else "unknown"
        raw_features = result.n_features

        logged = to_log2(result.expression)
        genes = map_probes(logged, platform_id=platform, collapse=args.collapse)
        matrices[accession] = genes

        print(
            f"{accession}: {raw_features} probes -> {genes.shape[0]} genes "
            f"x {genes.shape[1]} samples  [{platform}]"
        )

    if len(matrices) < 2:
        print("\nNeed >=2 harmonized series to merge — stopping.")
        return 1

    merged, batch = merge(matrices)
    print("\n--- merged ---")
    print(f"shared genes: {merged.shape[0]}")
    print(f"merged shape: {merged.shape[0]} genes x {merged.shape[1]} samples")
    print("samples per batch:")
    for accession, count in batch.value_counts().items():
        print(f"  {accession}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
