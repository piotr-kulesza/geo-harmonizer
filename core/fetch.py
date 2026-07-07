"""Fetch a GEO series into a comparable expression matrix + raw metadata frame.

This is the only real logic in the pipeline on Day 1. It obeys the iron rule in
CLAUDE.md: pure logic, no web/MCP/UI knowledge. It never writes a GEO parser
(GEOparse does that), never raises for expected network/parse/shape failures
(returns a structured fallback instead), and never prints (uses ``logging``).

Data conventions (see CLAUDE.md):
- Expression matrices are **features (rows) x samples (columns)**; GSM ids are
  the sample columns.
- Metadata frames are **samples (rows) x fields (columns)**, indexed by GSM.
  Metadata here is *raw* — no value standardization. Day 3's Claude pass reasons
  over the mess, so we preserve it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

logger = logging.getLogger(__name__)

Status = Literal["ok", "needs_manual_upload"]


@dataclass
class FetchResult:
    """Structured result of fetching one GEO series.

    Attributes:
        accession: The (normalized) GEO series accession, e.g. ``"GSE9891"``.
        status: ``"ok"`` when an expression matrix was built; otherwise
            ``"needs_manual_upload"`` (network/parse/shape failure, or a series
            whose counts live in supplementary files).
        message: Human-facing text explaining the result. On fallback this tells
            the user exactly what to do (upload the matrix). Empty on clean OK.
        platform_ids: Distinct GPL platform ids seen across the samples.
        expression: features (rows) x samples (GSM columns), or ``None``.
        metadata: samples (rows) x raw fields, indexed by GSM, or ``None``. May
            be present even when ``expression`` is ``None`` (metadata is usable,
            only the matrix needs manual upload).
    """

    accession: str
    status: Status
    message: str = ""
    platform_ids: list[str] = field(default_factory=list)
    expression: Optional[pd.DataFrame] = None
    metadata: Optional[pd.DataFrame] = None

    @property
    def ok(self) -> bool:
        """True when an expression matrix was successfully built."""
        return self.status == "ok"

    @property
    def n_samples(self) -> int:
        """Number of samples (matrix columns, else metadata rows, else 0)."""
        if self.expression is not None:
            return int(self.expression.shape[1])
        if self.metadata is not None:
            return int(self.metadata.shape[0])
        return 0

    @property
    def n_features(self) -> int:
        """Number of features/rows in the expression matrix (0 if none)."""
        if self.expression is not None:
            return int(self.expression.shape[0])
        return 0

    def summary(self) -> str:
        """One-line human-facing summary for smoke scripts and logs."""
        if self.ok:
            platforms = ", ".join(self.platform_ids) or "unknown platform"
            return (
                f"{self.accession}: ok — {self.n_features} features x "
                f"{self.n_samples} samples [{platforms}]"
            )
        return f"{self.accession}: needs manual upload — {self.message}"


def fetch_gse(
    accession: str,
    cache_dir: str = "data/cache",
    value_column: str = "VALUE",
) -> FetchResult:
    """Fetch a GEO series and return a :class:`FetchResult`.

    Args:
        accession: A GEO series accession (e.g. ``"GSE9891"``). Case/whitespace
            are normalized.
        cache_dir: Directory GEOparse downloads into; created if missing. Caching
            here is what lets the demo run without live-fetching on camera.
        value_column: The per-sample column to pivot into the expression matrix.
            ``"VALUE"`` is the GEO convention for processed intensities.

    Returns:
        A :class:`FetchResult`. Never raises for expected failure modes — a
        network error, an unparseable series, or a missing value column all come
        back as ``status="needs_manual_upload"`` with an actionable ``message``.
    """
    accession = _normalize_accession(accession)
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Import GEOparse lazily so this module imports even where it isn't installed
    # (the build sandbox is offline; tests monkeypatch sys.modules).
    try:
        import GEOparse  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-dependent
        logger.warning("GEOparse unavailable for %s: %s", accession, exc)
        return FetchResult(
            accession=accession,
            status="needs_manual_upload",
            message=(
                "The GEO fetcher isn't available in this environment. "
                "Upload the expression matrix file manually to continue."
            ),
        )

    try:
        gse = GEOparse.get_GEO(geo=accession, destdir=str(cache_path), silent=True)
    except Exception as exc:
        logger.warning("Failed to fetch %s from GEO: %s", accession, exc)
        return FetchResult(
            accession=accession,
            status="needs_manual_upload",
            message=(
                f"This series ({accession}) couldn't be fetched automatically. "
                "Upload the expression matrix file manually to continue."
            ),
        )

    metadata = _build_metadata_frame(gse)
    platform_ids = _platform_ids(metadata)
    expression = _build_expression_matrix(gse, value_column=value_column)

    if expression is None or expression.empty:
        # Common for RNA-seq series whose counts live in supplementary files:
        # the SOFT parsed fine and metadata is usable — only the matrix is missing.
        logger.info(
            "No usable '%s' matrix for %s; returning metadata-only fallback.",
            value_column,
            accession,
        )
        return FetchResult(
            accession=accession,
            status="needs_manual_upload",
            message=(
                f"Parsed {accession}'s sample metadata, but no processed "
                f"expression matrix (column '{value_column}') was found — its "
                "values likely live in supplementary files. Metadata is ready; "
                "upload the expression matrix file to continue."
            ),
            platform_ids=platform_ids,
            metadata=metadata,
        )

    return FetchResult(
        accession=accession,
        status="ok",
        message="",
        platform_ids=platform_ids,
        expression=expression,
        metadata=metadata,
    )


def load_matrix_from_file(
    path: str,
    sep: Optional[str] = None,
    index_col: int = 0,
) -> pd.DataFrame:
    """Load a manually-uploaded expression matrix (the fallback path).

    Returns a features (rows) x samples (columns) frame — the same shape as
    :func:`fetch_gse`, so downstream code never branches on the data's source.

    Args:
        path: Path to a ``.csv`` (comma) or other delimited (tab) file.
        sep: Delimiter override. If ``None``, inferred: ``","`` for ``.csv``
            files, tab otherwise.
        index_col: Column to use as the feature index (default first column).
    """
    if sep is None:
        sep = "," if str(path).lower().endswith(".csv") else "\t"
    matrix = pd.read_csv(path, sep=sep, index_col=index_col)
    return matrix


# --------------------------------------------------------------------------- #
# Private helpers
# --------------------------------------------------------------------------- #
def _normalize_accession(accession: str) -> str:
    """Uppercase and strip whitespace from a GEO accession."""
    return str(accession).strip().upper()


def _build_expression_matrix(
    gse,
    value_column: str = "VALUE",
) -> Optional[pd.DataFrame]:
    """Pivot the series' per-sample tables into features x samples on ``value_column``.

    Returns ``None`` if no sample carries the column or the pivot comes back
    empty (the caller turns that into a metadata-only fallback).
    """
    # Only pivot if at least one sample actually has the value column, otherwise
    # GEOparse raises deep inside pivot_samples.
    has_column = False
    for gsm in getattr(gse, "gsms", {}).values():
        table = getattr(gsm, "table", None)
        if table is not None and value_column in getattr(table, "columns", []):
            has_column = True
            break
    if not has_column:
        return None

    try:
        matrix = gse.pivot_samples(value_column)
    except Exception as exc:
        logger.warning("pivot_samples('%s') failed: %s", value_column, exc)
        return None

    if matrix is None or matrix.empty:
        return None

    # Rows are probe/feature ids, columns are GSM sample ids — the uniform shape.
    matrix.index.name = matrix.index.name or "ID_REF"
    matrix.columns.name = "sample"
    return matrix


def _build_metadata_frame(gse) -> pd.DataFrame:
    """Build a raw samples x fields metadata frame indexed by GSM.

    Preserves the full ``characteristics_ch1`` list as one raw string (Day 3's
    Claude pass reasons over it) and additionally splits each entry into a
    best-effort ``char::<key>`` column for human inspection. No value
    standardization happens here — that is deliberately Day 3's job.
    """
    rows: dict[str, dict[str, object]] = {}
    for gsm_id, gsm in getattr(gse, "gsms", {}).items():
        meta = getattr(gsm, "metadata", {}) or {}
        record: dict[str, object] = {
            "title": _first(meta.get("title")),
            "source_name_ch1": _first(meta.get("source_name_ch1")),
            "platform_id": _first(meta.get("platform_id")),
        }

        characteristics = meta.get("characteristics_ch1") or []
        if isinstance(characteristics, str):
            characteristics = [characteristics]
        # Preserve the raw mess verbatim — this is what Claude reasons over.
        record["characteristics_ch1"] = " || ".join(str(c) for c in characteristics)

        # Best-effort split on the first ':' for human inspection only.
        for entry in characteristics:
            entry = str(entry)
            if ":" in entry:
                key, _, val = entry.partition(":")
                col = f"char::{key.strip().lower()}"
                # Don't clobber a populated column if a key repeats.
                if col not in record or not record.get(col):
                    record[col] = val.strip()

        rows[gsm_id] = record

    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "sample"
    return frame


def _platform_ids(metadata: pd.DataFrame) -> list[str]:
    """Distinct, sorted GPL platform ids from a metadata frame."""
    if metadata is None or "platform_id" not in metadata.columns:
        return []
    values = (
        metadata["platform_id"].dropna().astype(str).str.strip()
    )
    values = values[values != ""]
    return sorted(values.unique().tolist())


def _first(value) -> Optional[str]:
    """Return the first element of a GEOparse metadata list, or ``None``.

    GEOparse stores every metadata field as a list of strings; the fields we
    surface are single-valued, so we take the first entry.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else None
    return str(value)
