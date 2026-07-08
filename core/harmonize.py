"""Probe->gene mapping and log2 normalization. [Day 2 — Epic 2]

Turns a per-series expression matrix into something comparable across platforms:
map platform probe IDs to HGNC gene symbols (so GPL96 and GPL570 speak the same
gene language), collapse many-probes-to-one-gene, and put every series on a log2
scale — with a guard against double-logging already-log data.

Settled decisions (CLAUDE.md / TICKETS.md):
- Probe->symbol source is **mygene** (ticket 2.1, decided), queried through the
  one seam ``_query_symbols`` that tests monkeypatch. It is the only place that
  touches the network, and its results are cached to ``{platform}.json`` so
  harmonize is offline-repeatable and demo-safe (ticket 2.7).
- Double-log2 guard (ticket 2.4): a matrix whose max is < ~30 is already
  log-scale and is returned untouched; double-logging squashes everything and
  quietly ruins the PCA.

Iron rule: pure logic, no web/MCP/UI knowledge; no ``print`` (uses ``logging``).
Data convention: expression is features (rows) x samples (GSM columns); after
:func:`map_probes` the rows are gene symbols.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Below this max value a matrix is treated as already log-scale (ticket 2.4).
_LOG2_GUARD_MAX = 30.0

# mygene "notfound" sentinel and empties we treat as unmapped.
_UNMAPPED = {"", "notfound", "nan", "none"}


def to_log2(matrix: pd.DataFrame) -> pd.DataFrame:
    """Put an expression matrix on a log2 scale, guarding against double-log2.

    If the matrix is already log-scale (``nanmax`` below ~30) it is returned
    unchanged. Otherwise ``log2`` is applied, flooring values at 1 first to avoid
    ``-inf`` from zeros/negatives. Same shape in, same shape out.

    Args:
        matrix: features (rows) x samples (columns) expression matrix.

    Returns:
        The (possibly) log2-transformed matrix, same shape and labels.
    """
    max_value = float(np.nanmax(matrix.to_numpy(dtype=float))) if matrix.size else 0.0

    if max_value < _LOG2_GUARD_MAX:
        logger.info(
            "to_log2: max %.3f < %.0f — already log-scale, leaving unchanged.",
            max_value,
            _LOG2_GUARD_MAX,
        )
        return matrix

    logger.info("to_log2: max %.3f — applying log2 (floor at 1).", max_value)
    return np.log2(matrix.clip(lower=1))


def map_probes(
    matrix: pd.DataFrame,
    platform_id: str,
    collapse: str = "max",
    cache_dir: str = "data/cache/annotations",
) -> pd.DataFrame:
    """Map probe IDs (the matrix index) to gene symbols and collapse duplicates.

    Args:
        matrix: features (rows = probe IDs) x samples (GSM columns).
        platform_id: GPL platform id, used to key the annotation cache.
        collapse: how to combine probes sharing a gene symbol — ``"max"``
            (default) or ``"mean"``.
        cache_dir: directory holding ``{platform_id}.json`` probe->symbol caches.

    Returns:
        A genes (rows, unique HGNC symbols, index name ``"gene"``) x samples
        matrix, rows sorted for deterministic order. Probes with no symbol are
        dropped.

    Raises:
        ValueError: if ``collapse`` is not ``"max"`` or ``"mean"``.
    """
    if collapse not in ("max", "mean"):
        raise ValueError(f"collapse must be 'max' or 'mean', got {collapse!r}")

    probes = [str(p) for p in matrix.index]
    symbols = _query_symbols(probes, platform_id, cache_dir)

    # Attach each probe's gene symbol; drop the unmapped ones.
    gene_of = matrix.index.to_series().astype(str).map(symbols)
    keep = gene_of.notna() & ~gene_of.astype(str).str.strip().str.lower().isin(
        _UNMAPPED
    )
    n_dropped = int((~keep).sum())
    if n_dropped:
        logger.info(
            "map_probes(%s): dropping %d/%d unmapped probes.",
            platform_id,
            n_dropped,
            len(probes),
        )

    mapped = matrix.loc[keep].copy()
    mapped.index = gene_of[keep].astype(str).values

    # Collapse many-probes-to-one-gene per sample.
    grouped = mapped.groupby(level=0)
    collapsed = grouped.max() if collapse == "max" else grouped.mean()

    collapsed = collapsed.sort_index()
    collapsed.index.name = "gene"
    collapsed.columns.name = matrix.columns.name or "sample"
    logger.info(
        "map_probes(%s): %d probes -> %d genes (collapse=%s).",
        platform_id,
        len(probes),
        collapsed.shape[0],
        collapse,
    )
    return collapsed


def _query_symbols(
    probes: Iterable[str],
    platform_id: str,
    cache_dir: str = "data/cache/annotations",
) -> dict[str, str]:
    """Return a probe->gene-symbol map, using a per-platform cache (mygene seam).

    This is the ONLY function that touches the network. Tests monkeypatch it.
    A ``{platform_id}.json`` cache is loaded if present; only probes missing from
    it are queried via mygene, then the merged map is rewritten. This makes
    harmonize offline-repeatable and demo-safe (ticket 2.7).

    Args:
        probes: platform probe IDs to resolve.
        platform_id: GPL id, used as the cache filename.
        cache_dir: directory holding the JSON caches.

    Returns:
        A dict mapping each requested probe to its gene symbol. Probes that
        resolve to nothing are simply absent from the dict.
    """
    probes = [str(p) for p in probes]
    cache_path = Path(cache_dir) / f"{platform_id}.json"

    cached: dict[str, str] = {}
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
        except Exception as exc:  # corrupt cache — treat as empty, re-query
            logger.warning("Ignoring unreadable annotation cache %s: %s", cache_path, exc)
            cached = {}

    missing = [p for p in probes if p not in cached]

    if missing:
        fetched = _fetch_symbols_from_mygene(missing, platform_id)
        cached.update(fetched)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cached, indent=0, sort_keys=True))
        logger.info(
            "Cached %d probe->symbol entries for %s (%d newly fetched).",
            len(cached),
            platform_id,
            len(fetched),
        )

    # Restrict to the requested probes; drop empties.
    result: dict[str, str] = {}
    for probe in probes:
        symbol = cached.get(probe)
        if symbol and str(symbol).strip().lower() not in _UNMAPPED:
            result[probe] = str(symbol)
    return result


def _fetch_symbols_from_mygene(probes: list[str], platform_id: str) -> dict[str, str]:
    """Query mygene for probe->symbol. Network side of the seam; imported lazily."""
    import mygene  # lazy import: keeps core importable offline

    client = mygene.MyGeneInfo()
    frame = client.querymany(
        probes,
        scopes="reporter",  # matches Affymetrix probe IDs (GPL96/GPL570)
        fields="symbol",
        species="human",
        as_dataframe=True,
    )

    mapping: dict[str, str] = {}
    if frame is None or getattr(frame, "empty", True):
        return mapping

    if "symbol" not in frame.columns:
        return mapping

    for probe, symbol in frame["symbol"].items():
        if symbol is None or (isinstance(symbol, float) and np.isnan(symbol)):
            continue
        # querymany may return several rows per probe; keep the first symbol.
        if probe not in mapping:
            mapping[str(probe)] = str(symbol)
    return mapping
