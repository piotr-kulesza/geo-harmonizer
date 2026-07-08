#!/usr/bin/env python3
"""Day-3 Claude-Use checkpoint: parse + verify the demo set's metadata.

Networked — uses the REAL Anthropic API and spends a few grant credits. The
author runs this; the build sandbox never does. Needs ANTHROPIC_API_KEY (a local
.env, gitignored) and `pip install anthropic`.

For the cached demo series: load each one's metadata via fetch_gse, run
parse_metadata (one Claude call per dataset) then verify_mapping (one more),
and print each spec, the head of the unified standardized table, and the flags.

Usage:
    python scripts/metadata_check.py [ACCESSION ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import fetch_gse  # noqa: E402
from core.metadata import parse_metadata, verify_mapping, _spec_to_dict  # noqa: E402

DEFAULT_ACCESSIONS = ["GSE9891", "GSE26193", "GSE26712"]


def _load_dotenv() -> None:
    """Minimal .env loader (ANTHROPIC_API_KEY) — no python-dotenv dependency."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accessions", nargs="*", default=DEFAULT_ACCESSIONS)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set (put it in a local .env). Aborting.")
        return 2

    accessions = args.accessions or DEFAULT_ACCESSIONS
    raw_by_dataset = {}
    for accession in accessions:
        result = fetch_gse(accession, cache_dir=args.cache_dir)
        if result.metadata is None:
            print(f"{accession}: no metadata available — skipping.")
            continue
        raw_by_dataset[accession] = result.metadata
        print(f"{accession}: {result.metadata.shape[0]} samples x {result.metadata.shape[1]} raw fields")

    if not raw_by_dataset:
        print("No metadata to parse — stopping.")
        return 1

    # One Claude call per dataset (parse), then one more (verify).
    mapping = parse_metadata(raw_by_dataset)
    mapping = verify_mapping(mapping)

    print("\n=== per-dataset mapping specs ===")
    for accession, spec in mapping.specs.items():
        print(f"\n{accession}:")
        print(json.dumps(_spec_to_dict(spec), indent=2))

    print("\n=== unified standardized table (head) ===")
    with_opts = mapping.standardized
    print(with_opts.head(10).to_string())

    print("\n=== verifier flags ===")
    if not mapping.flags:
        print("(no flags — verifier found the mappings consistent)")
    for flag in mapping.flags:
        print(
            f"[{flag['severity'].upper()}] {flag['field']} "
            f"{flag['datasets']}: {flag['message']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
