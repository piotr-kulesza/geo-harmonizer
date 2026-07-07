"""Fetch a GEO series into a comparable expression matrix + raw metadata frame.

**Fetch strategy (series-matrix-first).** The primary source is the GEO *series
matrix* over HTTPS — the same compact, already-pivoted artifact that R's
``GEOquery::getGEO(GSEMatrix=TRUE)`` uses (probes x samples + the ``!Sample_*``
metadata header). It's one small file over a reliable protocol, so it avoids
GEOparse's default ``*_family.soft.gz`` pull over NCBI **FTP**, which truncates
on large series (e.g. GSE9891 -> "Downloaded size do not match"). It also removes
the Day-2 pivot step. Parsing series-matrix is a bounded TSV-with-header reader,
not a hand-rolled SOFT parser — a justified exception to CLAUDE.md's "no GEO
parser" rule. GEOparse/SOFT stays as the FALLBACK, and manual upload is the last
resort. The path that succeeds is logged at INFO level.

This module obeys the iron rule in CLAUDE.md: pure logic, no web/MCP/UI
knowledge. It never raises for expected network/parse/shape failures (returns a
structured fallback instead), and never prints (uses ``logging``).

Data conventions (see CLAUDE.md):
- Expression matrices are **features (rows) x samples (columns)**; GSM ids are
  the sample columns.
- Metadata frames are **samples (rows) x fields (columns)**, indexed by GSM.
  Metadata here is *raw* — no value standardization. Day 3's Claude pass reasons
  over the mess, so we preserve it.
"""

from __future__ import annotations

import gzip
import logging
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

logger = logging.getLogger(__name__)

Status = Literal["ok", "needs_manual_upload"]

_USER_AGENT = "geo-harmonizer/0.1 (+https://github.com/piotr-kulesza/geo-harmonizer)"


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

    Tries, in order: (1) the series matrix over HTTPS (primary), (2) GEOparse's
    SOFT path (fallback), (3) a ``needs_manual_upload`` result (last resort).

    Args:
        accession: A GEO series accession (e.g. ``"GSE9891"``). Case/whitespace
            are normalized.
        cache_dir: Directory downloads land in; created if missing. Caching here
            is what lets the demo run without live-fetching on camera.
        value_column: The per-sample column the SOFT fallback pivots into the
            expression matrix. ``"VALUE"`` is the GEO convention for processed
            intensities. (The series-matrix path ignores this — it's already
            pivoted.)

    Returns:
        A :class:`FetchResult`. Never raises for expected failure modes — a
        network error, a 404, a multi-platform series, or a parse error all come
        back through the fallback chain, ending in ``needs_manual_upload`` with
        an actionable ``message``.
    """
    accession = _normalize_accession(accession)
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # 1. Series matrix over HTTPS (primary).
    result = _fetch_via_series_matrix(accession, cache_path)
    if result is not None:
        return result

    # 2. GEOparse / SOFT (fallback). Returns a usable result (ok, or a
    #    metadata-only needs_manual_upload) or None if it can't fetch at all.
    result = _fetch_via_soft(accession, cache_path, value_column)
    if result is not None:
        return result

    # 3. Manual upload (last resort).
    logger.info("All fetch paths failed for %s; needs manual upload.", accession)
    return FetchResult(
        accession=accession,
        status="needs_manual_upload",
        message=(
            f"This series ({accession}) couldn't be fetched automatically. "
            "Upload the expression matrix file manually to continue."
        ),
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
# Primary path: series matrix over HTTPS
# --------------------------------------------------------------------------- #
def _fetch_via_series_matrix(
    accession: str, cache_path: Path
) -> Optional[FetchResult]:
    """Download + parse the series matrix over HTTPS. Returns None on any failure."""
    url = _series_matrix_url(accession)
    dest = cache_path / f"{accession}_series_matrix.txt.gz"

    try:
        _download_to_cache(url, dest)
    except Exception as exc:
        logger.info("Series-matrix download failed for %s: %s", accession, exc)
        return None

    try:
        expression, metadata, platform_ids = _parse_series_matrix(dest)
    except Exception as exc:
        logger.info("Series-matrix parse failed for %s: %s", accession, exc)
        return None

    if expression is None or expression.empty:
        logger.info("Series matrix for %s parsed to an empty matrix.", accession)
        return None

    logger.info("Fetched %s via series matrix (HTTPS).", accession)
    return FetchResult(
        accession=accession,
        status="ok",
        message="",
        platform_ids=platform_ids,
        expression=expression,
        metadata=metadata,
    )


def _series_matrix_url(accession: str) -> str:
    """Build the GEO series-matrix HTTPS URL for an accession.

    GEO groups series into directories by stripping the last 3 digits of the
    numeric part and replacing them with ``nnn`` (``GSE9891`` -> ``GSE9nnn``,
    ``GSE712`` -> ``GSEnnn``).
    """
    accession = _normalize_accession(accession)
    digits = "".join(ch for ch in accession if ch.isdigit())
    stub = f"GSE{digits[:-3]}nnn"
    return (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/"
        f"{stub}/{accession}/matrix/{accession}_series_matrix.txt.gz"
    )


def _download_to_cache(
    url: str,
    dest_path,
    retries: int = 3,
    timeout: int = 30,
) -> Path:
    """Download ``url`` to ``dest_path`` with retry + backoff. Skip if cached.

    Raises on failure (the caller turns that into a fallback). A non-empty file
    already at ``dest_path`` is treated as a cache hit and returned as-is.
    """
    dest_path = Path(dest_path)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info("Using cached %s", dest_path.name)
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                data = resp.read()
            tmp = dest_path.with_name(dest_path.name + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest_path)
            return dest_path
        except Exception as exc:  # network, HTTP error, timeout
            last_exc = exc
            logger.info(
                "Download attempt %d/%d failed for %s: %s",
                attempt,
                retries,
                url,
                exc,
            )
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError(f"Failed to download {url} after {retries} attempts: {last_exc}")


def _parse_series_matrix(path):
    """Parse a local series-matrix file into (expression, metadata, platform_ids).

    Reads a ``.txt.gz`` (or plain ``.txt``) with no network access. Returns the
    canonical shapes: expression is features (rows) x samples (GSM columns);
    metadata is samples (rows) x fields, indexed by GSM (index name ``"sample"``),
    with the same schema the SOFT path produces. Values are NOT transformed
    (log2 etc. is Day 2). Raises on a malformed/empty file.
    """
    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open

    sample_fields: dict[str, list[str]] = {}
    characteristics_lines: list[list[str]] = []
    table_lines: list[str] = []
    in_table = False

    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n").rstrip("\r")
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                in_table = False
                continue
            if in_table:
                if line.strip():
                    table_lines.append(line)
                continue
            if line.startswith("!Sample_"):
                parts = line.split("\t")
                key = parts[0].lstrip("!")
                values = [_strip_quotes(v) for v in parts[1:]]
                if key == "Sample_characteristics_ch1":
                    characteristics_lines.append(values)
                else:
                    sample_fields[key] = values

    gsm_ids = sample_fields.get("Sample_geo_accession")
    if not gsm_ids:
        raise ValueError("series matrix has no !Sample_geo_accession header")
    if not table_lines:
        raise ValueError("series matrix has no data table")

    metadata = _build_series_matrix_metadata(
        gsm_ids=gsm_ids,
        titles=sample_fields.get("Sample_title"),
        sources=sample_fields.get("Sample_source_name_ch1"),
        platforms=sample_fields.get("Sample_platform_id"),
        characteristics_lines=characteristics_lines,
    )
    platform_ids = _platform_ids(metadata)
    expression = _build_series_matrix_expression(table_lines)
    return expression, metadata, platform_ids


def _build_series_matrix_expression(table_lines: list[str]) -> pd.DataFrame:
    """Turn the ``!series_matrix_table_*`` block into features x samples."""
    header = [_strip_quotes(cell) for cell in table_lines[0].split("\t")]
    sample_columns = header[1:]  # header[0] is "ID_REF"

    index: list[str] = []
    rows: list[list[str]] = []
    for line in table_lines[1:]:
        cells = line.split("\t")
        index.append(_strip_quotes(cells[0]))
        rows.append([_strip_quotes(c) for c in cells[1:]])

    matrix = pd.DataFrame(rows, index=index, columns=sample_columns)
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    matrix.index.name = header[0] or "ID_REF"
    matrix.columns.name = "sample"
    return matrix


def _build_series_matrix_metadata(
    gsm_ids: list[str],
    titles: Optional[list[str]],
    sources: Optional[list[str]],
    platforms: Optional[list[str]],
    characteristics_lines: list[list[str]],
) -> pd.DataFrame:
    """Build the canonical metadata frame from series-matrix header fields."""
    rows: dict[str, dict[str, object]] = {}
    for i, gsm_id in enumerate(gsm_ids):
        characteristics = [
            line[i] for line in characteristics_lines if i < len(line)
        ]
        rows[gsm_id] = _build_metadata_record(
            title=_at(titles, i),
            source_name_ch1=_at(sources, i),
            platform_id=_at(platforms, i),
            characteristics=characteristics,
        )

    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "sample"
    return frame


# --------------------------------------------------------------------------- #
# Fallback path: GEOparse / SOFT
# --------------------------------------------------------------------------- #
def _fetch_via_soft(
    accession: str, cache_path: Path, value_column: str
) -> Optional[FetchResult]:
    """Fetch via GEOparse's SOFT path. Returns a usable result, or None if it
    can't fetch at all (import/network failure), so the caller can move on.
    """
    try:
        import GEOparse  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-dependent
        logger.info("GEOparse unavailable for %s: %s", accession, exc)
        return None

    try:
        gse = GEOparse.get_GEO(geo=accession, destdir=str(cache_path), silent=True)
    except Exception as exc:
        logger.info("SOFT fetch failed for %s: %s", accession, exc)
        return None

    metadata = _build_metadata_frame(gse)
    platform_ids = _platform_ids(metadata)
    expression = _build_expression_matrix(gse, value_column=value_column)

    if expression is None or expression.empty:
        # Common for RNA-seq series whose counts live in supplementary files:
        # the SOFT parsed fine and metadata is usable — only the matrix is missing.
        logger.info(
            "SOFT parsed %s but found no '%s' matrix; metadata-only fallback.",
            accession,
            value_column,
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

    logger.info("Fetched %s via SOFT (GEOparse fallback).", accession)
    return FetchResult(
        accession=accession,
        status="ok",
        message="",
        platform_ids=platform_ids,
        expression=expression,
        metadata=metadata,
    )


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
    """Build a raw samples x fields metadata frame indexed by GSM (SOFT path).

    Preserves the full ``characteristics_ch1`` list as one raw string (Day 3's
    Claude pass reasons over it) and additionally splits each entry into a
    best-effort ``char::<key>`` column for human inspection. No value
    standardization happens here — that is deliberately Day 3's job.
    """
    rows: dict[str, dict[str, object]] = {}
    for gsm_id, gsm in getattr(gse, "gsms", {}).items():
        meta = getattr(gsm, "metadata", {}) or {}
        rows[gsm_id] = _build_metadata_record(
            title=_first(meta.get("title")),
            source_name_ch1=_first(meta.get("source_name_ch1")),
            platform_id=_first(meta.get("platform_id")),
            characteristics=meta.get("characteristics_ch1"),
        )

    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "sample"
    return frame


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _build_metadata_record(
    title: Optional[str],
    source_name_ch1: Optional[str],
    platform_id: Optional[str],
    characteristics,
) -> dict[str, object]:
    """Build one sample's canonical metadata record from raw fields.

    Shared by both the series-matrix and SOFT paths so the two sources produce
    an identical schema: ``title``, ``source_name_ch1``, ``platform_id``, the
    raw ``characteristics_ch1`` (joined verbatim with ``" || "``), and
    best-effort ``char::<key>`` splits on the first ``:``.
    """
    record: dict[str, object] = {
        "title": title,
        "source_name_ch1": source_name_ch1,
        "platform_id": platform_id,
    }

    if isinstance(characteristics, str):
        characteristics = [characteristics]
    characteristics = [
        str(c) for c in (characteristics or []) if str(c).strip()
    ]

    # Preserve the raw mess verbatim — this is what Claude reasons over.
    record["characteristics_ch1"] = " || ".join(characteristics)

    # Best-effort split on the first ':' for human inspection only.
    for entry in characteristics:
        if ":" in entry:
            key, _, val = entry.partition(":")
            col = f"char::{key.strip().lower()}"
            # Don't clobber a populated column if a key repeats.
            if col not in record or not record.get(col):
                record[col] = val.strip()

    return record


def _normalize_accession(accession: str) -> str:
    """Uppercase and strip whitespace from a GEO accession."""
    return str(accession).strip().upper()


def _platform_ids(metadata: pd.DataFrame) -> list[str]:
    """Distinct, sorted GPL platform ids from a metadata frame."""
    if metadata is None or "platform_id" not in metadata.columns:
        return []
    values = metadata["platform_id"].dropna().astype(str).str.strip()
    values = values[values != ""]
    return sorted(values.unique().tolist())


def _strip_quotes(value: str) -> str:
    """Strip surrounding whitespace then surrounding double quotes."""
    value = value.strip()
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value


def _at(values: Optional[list[str]], index: int) -> Optional[str]:
    """Safely index into an optional per-sample list."""
    if values is None or index >= len(values):
        return None
    return values[index]


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
