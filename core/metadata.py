"""Claude-driven metadata parsing + a second-pass verifier. [Day 3 — Epic 4]

This is the 25% Claude-Use quadrant: reasoning over messy oncology free text, not
regex. The design keeps it cheap, auditable, and offline-testable:

**The LLM proposes a mapping SPEC; code applies it deterministically.** We never
ask the model for per-sample values. Once per dataset, we ask it for a small
mapping spec from that dataset's raw fields -> ``TARGET_SCHEMA``, then apply the
spec in pure Python to every sample. So each dataset costs exactly ONE llm call
for parsing (never per field or per sample — see :func:`parse_metadata`), and a
second single call for verification (:func:`verify_mapping`).

**The injectable ``llm`` seam** is the one deliberate exception to the iron rule:
metadata reasoning is where the caller matters (CLAUDE.md → "The MCP metadata
seam"). ``llm`` is a callable ``llm(system, user) -> str``. The web path lets
:func:`parse_metadata` call the internal :func:`_default_llm` (real Anthropic
API); the MCP path passes pre-built ``specs`` (the orchestrating Claude maps
natively) so no key is needed. Tests inject a fake ``llm`` — no network, no
credits, no key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The unified target schema the raw fields map onto (ticket 4.1).
TARGET_SCHEMA: tuple[str, ...] = (
    "stage",
    "grade",
    "histology",
    "survival_days",
    "event",
)

# Closed set of transforms the code applies to a chosen source column.
_TRANSFORMS: frozenset[str] = frozenset(
    {
        "identity",
        "uppercase",
        "lower",
        "value_map",
        "scale",
        "constant",
        "regex_extract",
        "missing",
    }
)

# Keys of a field-map entry that are NOT transform params.
_ENTRY_META_KEYS = {"source", "transform", "note", "confidence"}

# An LLM callable: (system, user) -> model text.
LLM = Callable[[str, str], str]

# Default model. NOTE: the current Anthropic model id is claude-opus-4-8 (the
# recommended default). Override per-deployment with ANTHROPIC_MODEL;
# "claude-sonnet-4-6" is the cheaper option, "claude-opus-4-8" the stronger one.
_DEFAULT_MODEL = "claude-opus-4-8"


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class FieldMap:
    """How one target field is produced from a dataset's raw fields.

    Attributes:
        source: the raw column the transform reads (``None`` for
            ``constant``/``missing``).
        transform: one of :data:`_TRANSFORMS`.
        params: transform parameters (e.g. ``{"map": {...}}``, ``{"factor": ...}``,
            ``{"value": ...}``, ``{"pattern": ..., "group": ...}``).
        note: the model's short rationale (audit trail).
        confidence: the model's 0..1 self-reported confidence.
    """

    source: Optional[str]
    transform: str
    params: dict = field(default_factory=dict)
    note: str = ""
    confidence: float = 0.0


@dataclass
class MappingSpec:
    """One dataset's full raw-fields -> ``TARGET_SCHEMA`` mapping."""

    dataset: Optional[str]
    fields: dict[str, FieldMap]


@dataclass
class MappingResult:
    """The standardized table plus its per-dataset specs and verifier flags.

    Attributes:
        standardized: samples (rows, GSM index) x ``TARGET_SCHEMA`` + a
            ``dataset`` column, concatenated across all datasets.
        specs: ``{accession: MappingSpec}`` — the audit trail of how each raw
            field mapped.
        flags: cross-dataset consistency flags from :func:`verify_mapping`
            (empty until it runs).
    """

    standardized: pd.DataFrame
    specs: dict[str, MappingSpec]
    flags: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_dataset_summary(
    raw_df: pd.DataFrame,
    accession: Optional[str],
    max_values: int = 40,
) -> dict:
    """Compact per-dataset view: candidate columns and their distinct values.

    This compact summary — NOT the full sample rows — is what makes one llm call
    per dataset sufficient. Candidate columns are the ``char::*`` splits plus
    ``source_name_ch1`` and ``title``; ``platform_id`` and the raw
    ``characteristics_ch1`` blob are skipped.
    """
    candidates = [
        c
        for c in raw_df.columns
        if c.startswith("char::") or c in ("source_name_ch1", "title")
    ]
    columns: dict[str, list[str]] = {}
    for col in candidates:
        values = raw_df[col].dropna().astype(str).str.strip()
        values = values[values != ""]
        columns[col] = sorted(set(values))[:max_values]
    return {"accession": accession, "columns": columns}


def build_mapping_spec(
    dataset_summary: dict,
    llm: LLM,
    cache_dir: str = "data/cache/meta",
) -> MappingSpec:
    """The ONE llm call for a dataset: raw fields -> mapping spec.

    The raw response is cached by a hash of the summary (ticket 4.6), so re-runs
    and the web path don't re-spend credits.
    """
    user = _render_parse_user(dataset_summary)
    data = _cached_json(llm, _PARSE_SYSTEM, user, cache_dir)
    return _spec_from_json(data, dataset_summary.get("accession"))


def apply_mapping(raw_df: pd.DataFrame, spec: MappingSpec) -> pd.DataFrame:
    """Apply a :class:`MappingSpec` to every sample. PURE — no llm.

    This is the seam's MCP-native entry point: given a spec (from the API or from
    an orchestrating Claude), it deterministically produces the standardized
    frame. Returns samples (GSM index) x ``TARGET_SCHEMA`` + a ``dataset`` column.
    """
    index = raw_df.index
    columns: dict[str, pd.Series] = {}
    for target in TARGET_SCHEMA:
        fm = spec.fields.get(target) or FieldMap(source=None, transform="missing")
        source = (
            raw_df[fm.source]
            if (fm.source and fm.source in raw_df.columns)
            else None
        )
        columns[target] = _apply_transform(source, fm, index)

    frame = pd.DataFrame(columns, index=index)
    frame = frame[list(TARGET_SCHEMA)]
    frame["dataset"] = spec.dataset
    frame.index.name = raw_df.index.name or "sample"
    return frame


def parse_metadata(
    raw_by_dataset: dict[str, pd.DataFrame],
    llm: Optional[LLM] = None,
    specs: Optional[dict[str, MappingSpec]] = None,
    cache_dir: str = "data/cache/meta",
) -> MappingResult:
    """Map every dataset's raw metadata onto ``TARGET_SCHEMA`` and concatenate.

    For each ``{accession: raw_df}``: use a provided ``specs[acc]`` (the MCP path,
    no llm) or :func:`build_mapping_spec` (the web path, ONE llm call). The result
    is one standardized table across datasets.

    Batching guarantee: exactly one llm call per dataset for parsing — the loop
    calls :func:`build_mapping_spec` once per accession, never per field/sample.
    """
    llm = llm or _default_llm
    out_specs: dict[str, MappingSpec] = {}
    frames: list[pd.DataFrame] = []

    for accession, raw_df in raw_by_dataset.items():
        if specs and accession in specs:
            spec = specs[accession]
        else:
            summary = build_dataset_summary(raw_df, accession)
            spec = build_mapping_spec(summary, llm, cache_dir)
        out_specs[accession] = spec
        frames.append(apply_mapping(raw_df, spec))

    if frames:
        standardized = pd.concat(frames)
    else:
        standardized = pd.DataFrame(columns=[*TARGET_SCHEMA, "dataset"])
    return MappingResult(standardized=standardized, specs=out_specs, flags=[])


def verify_mapping(
    result: MappingResult,
    llm: Optional[LLM] = None,
    cache_dir: str = "data/cache/meta",
) -> MappingResult:
    """Second Claude pass: check cross-dataset consistency, attach flags.

    Feeds the model the per-dataset specs + the resulting standardized
    vocabularies across datasets (compact, not per-sample) and asks whether
    equivalent-looking fields actually mean the same thing (grade 3 == G3?
    survival units match? staging vocab clash?). ONE llm call.
    """
    llm = llm or _default_llm
    user = _render_verify_user(result.specs, result.standardized)
    data = _cached_json(llm, _VERIFY_SYSTEM, user, cache_dir)
    result.flags = _flags_from_json(data)
    return result


# --------------------------------------------------------------------------- #
# Transform application (pure)
# --------------------------------------------------------------------------- #
def _apply_transform(
    source: Optional[pd.Series], fm: FieldMap, index: pd.Index
) -> pd.Series:
    """Apply one field's transform to its source column (or produce a NaN column)."""
    transform = fm.transform

    if transform == "missing":
        return pd.Series(np.nan, index=index, dtype="object")
    if transform == "constant":
        return pd.Series(fm.params.get("value"), index=index)

    # Everything below needs a source column.
    if source is None:
        return pd.Series(np.nan, index=index, dtype="object")

    if transform == "identity":
        return source.copy()
    if transform == "uppercase":
        return source.map(lambda v: v.upper() if isinstance(v, str) else v)
    if transform == "lower":
        return source.map(lambda v: v.lower() if isinstance(v, str) else v)
    if transform == "value_map":
        mapping = {str(k).strip(): v for k, v in (fm.params.get("map") or {}).items()}

        def _map(value):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return np.nan
            return mapping.get(str(value).strip(), np.nan)

        return source.map(_map)
    if transform == "scale":
        factor = float(fm.params.get("factor", 1.0))
        return pd.to_numeric(source, errors="coerce") * factor
    if transform == "regex_extract":
        pattern = re.compile(str(fm.params.get("pattern", "")))
        group = int(fm.params.get("group", 0))

        def _extract(value):
            if not isinstance(value, str):
                return np.nan
            match = pattern.search(value)
            if not match:
                return np.nan
            try:
                return match.group(group)
            except (IndexError, re.error):  # bad group index
                return np.nan

        return source.map(_extract)

    # Unknown transform (shouldn't happen — validated on ingest) -> NaN column.
    return pd.Series(np.nan, index=index, dtype="object")


# --------------------------------------------------------------------------- #
# Spec (de)serialization + validation
# --------------------------------------------------------------------------- #
def _spec_from_json(data: dict, accession: Optional[str]) -> MappingSpec:
    """Parse + validate the model's JSON into a :class:`MappingSpec`."""
    # Accept either {"fields": {target: entry}} or a flat {target: entry}.
    fields_json = data.get("fields", data) if isinstance(data, dict) else {}
    if not isinstance(fields_json, dict):
        fields_json = {}
    fields = {
        target: _fieldmap_from_entry(fields_json.get(target)) for target in TARGET_SCHEMA
    }
    return MappingSpec(dataset=accession, fields=fields)


def _fieldmap_from_entry(entry: Optional[dict]) -> FieldMap:
    """Build a validated :class:`FieldMap` from one field's JSON entry."""
    entry = entry if isinstance(entry, dict) else {}
    transform = entry.get("transform", "missing")
    if transform not in _TRANSFORMS:
        logger.warning("Unknown transform %r — treating as 'missing'.", transform)
        transform = "missing"
    params = {k: v for k, v in entry.items() if k not in _ENTRY_META_KEYS}
    confidence = entry.get("confidence")
    return FieldMap(
        source=entry.get("source"),
        transform=transform,
        params=params,
        note=str(entry.get("note", "")),
        confidence=float(confidence) if confidence is not None else 0.0,
    )


def _spec_to_dict(spec: MappingSpec) -> dict:
    """Serialize a spec for the verifier prompt / audit output."""
    return {
        target: {
            "source": fm.source,
            "transform": fm.transform,
            **fm.params,
            "note": fm.note,
            "confidence": fm.confidence,
        }
        for target, fm in spec.fields.items()
    }


def _flags_from_json(data: dict) -> list[dict]:
    """Validate/normalize the verifier's ``flags`` list."""
    raw = data.get("flags", []) if isinstance(data, dict) else []
    flags: list[dict] = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        severity = entry.get("severity", "info")
        if severity not in ("info", "warn", "error"):
            severity = "info"
        flags.append(
            {
                "severity": severity,
                "field": str(entry.get("field", "")),
                "datasets": list(entry.get("datasets", []) or []),
                "message": str(entry.get("message", "")),
            }
        )
    return flags


# --------------------------------------------------------------------------- #
# The injectable LLM seam
# --------------------------------------------------------------------------- #
def _default_llm(system: str, user: str) -> str:
    """Internal Anthropic call (the web path). Lazy-imports ``anthropic``.

    Reads ``ANTHROPIC_API_KEY`` from the environment; model from
    ``ANTHROPIC_MODEL`` (default :data:`_DEFAULT_MODEL`). Returns the model's text.
    """
    import anthropic  # lazy: keeps core importable offline and key-free

    client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY from env
    model = os.environ.get("ANTHROPIC_MODEL", _DEFAULT_MODEL)
    message = client.messages.create(
        model=model,
        max_tokens=8000,
        thinking={"type": "adaptive"},  # reasoning, not regex (the 25% quadrant)
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    )


def _cached_json(llm: LLM, system: str, user: str, cache_dir: str) -> dict:
    """Call the llm for JSON, using a hash-of-prompt cache (ticket 4.6)."""
    key = hashlib.sha256(f"{system}\x00{user}".encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:  # corrupt cache — re-query
            logger.warning("Ignoring unreadable meta cache %s.", path)

    data = _call_json(llm, system, user)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return data


def _call_json(llm: LLM, system: str, user: str) -> dict:
    """Call the llm and parse strict JSON, with ONE repair retry on failure."""
    text = llm(system, user)
    try:
        return _loads_json(text)
    except Exception:
        logger.info("First llm reply wasn't valid JSON — asking for a repair.")

    repair_user = (
        user
        + "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON "
        "object — no prose, no markdown, no code fences."
    )
    text = llm(system, repair_user)
    try:
        return _loads_json(text)
    except Exception as exc:
        raise ValueError(
            f"LLM did not return valid JSON after one repair attempt: {exc}"
        ) from exc


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def _loads_json(text: str) -> dict:
    """Parse model text as JSON, tolerating ```json fences."""
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    if match:
        stripped = match.group(1).strip()
    return json.loads(stripped)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
_SCHEMA_DESCRIPTION = (
    "stage: FIGO tumor stage (normalize casing, e.g. IIIC).\n"
    "grade: tumor grade (numeric 1-3, or from prose like 'high-grade').\n"
    "histology: histological subtype (e.g. serous, endometrioid, clear cell).\n"
    "survival_days: overall-survival time in DAYS (convert years/months).\n"
    "event: survival event / vital status (death=1/alive=0, or coded AWD/NED/DOD)."
)

_TRANSFORM_MENU = (
    "identity — passthrough.\n"
    "uppercase / lower — case-normalize (e.g. stage 'IIIc' -> 'IIIC').\n"
    "value_map {\"map\": {raw: target}} — recode; unmapped values become null.\n"
    "scale {\"factor\": float} — multiply a numeric column (years->days = 365.25).\n"
    "constant {\"value\": ...} — one value for the whole dataset (e.g. grade from "
    "the prose 'high-grade').\n"
    "regex_extract {\"pattern\": ..., \"group\": int} — pull a value out of prose.\n"
    "missing — this target field is absent for this dataset (null column)."
)

_PARSE_SYSTEM = (
    "You are a translational-oncology data curator harmonizing GEO ovarian-cancer "
    "metadata. You map one dataset's raw sample fields onto a fixed target schema. "
    "You reason about oncology vocabularies — FIGO stage casing (IIIC vs IIIc), "
    "tumor grade written numerically or embedded in prose, histology synonyms and "
    "abbreviations (Ser/PapSer = serous), survival units (years/months vs days) and "
    "OS-vs-PFS disambiguation (prefer overall survival), and event codes "
    "(AWD/NED/DOD, alive/dead, 0/1). You DO NOT invent values: you choose a source "
    "column and a transform, and code applies it to every sample. "
    "Return ONLY a JSON object — no prose, no code fences."
)


def _render_parse_user(dataset_summary: dict) -> str:
    """Build the per-dataset parsing prompt (schema + transforms + the summary)."""
    payload = {
        "accession": dataset_summary.get("accession"),
        "candidate_fields": dataset_summary.get("columns", {}),
    }
    return (
        "TARGET SCHEMA (target field -> meaning):\n"
        f"{_SCHEMA_DESCRIPTION}\n\n"
        "ALLOWED TRANSFORMS:\n"
        f"{_TRANSFORM_MENU}\n\n"
        "For EACH target field, return an entry:\n"
        '{"source": <raw column name or null>, "transform": <name>, '
        '<transform params...>, "note": <short rationale>, '
        '"confidence": <0..1>}\n'
        'Return: {"fields": {"stage": {...}, "grade": {...}, "histology": {...}, '
        '"survival_days": {...}, "event": {...}}}\n\n'
        "DATASET (candidate columns with their distinct values):\n"
        f"{json.dumps(payload, indent=2)}"
    )


_VERIFY_SYSTEM = (
    "You are a translational-oncology reviewer auditing a cross-dataset metadata "
    "harmonization for meta-analysis. You are given each dataset's mapping spec and "
    "the resulting standardized vocabularies per field across datasets. Your job is "
    "to decide whether equivalent-LOOKING values actually MEAN the same thing "
    "across datasets, and to flag the uncertain. Watch specifically for: survival "
    "unit mismatches (one dataset in days, another still in years/months); event "
    "encodings that don't line up (AWD/NED/DOD text vs 0/1); grade scales that "
    "differ (numeric 1-3 vs prose 'high-grade' vs G3); FIGO stage vocab/casing "
    "clashes; histology label mismatches; and cohorts that aren't comparable "
    "(borderline/LMP tumors or normal controls mixed with invasive cases). "
    "Return ONLY a JSON object — no prose, no code fences."
)

_VERIFY_USER_PREFIX = (
    "Review the harmonization below. Return "
    '{"flags": [{"severity": "info|warn|error", "field": <target field>, '
    '"datasets": [<accessions involved>], "message": <what is inconsistent and '
    'why it matters for meta-analysis>}]}. Return an empty flags list only if '
    "everything is genuinely consistent.\n\n"
)


def _render_verify_user(
    specs: dict[str, MappingSpec], standardized: pd.DataFrame
) -> str:
    """Build the verifier prompt: specs + standardized vocabularies (compact)."""
    specs_json = {acc: _spec_to_dict(spec) for acc, spec in specs.items()}

    vocab: dict[str, dict[str, list[str]]] = {}
    for target in TARGET_SCHEMA:
        per_dataset: dict[str, list[str]] = {}
        if target in standardized.columns and "dataset" in standardized.columns:
            for accession, sub in standardized.groupby("dataset"):
                values = sub[target].dropna().astype(str).str.strip()
                values = values[values != ""]
                per_dataset[str(accession)] = sorted(set(values))[:40]
        vocab[target] = per_dataset

    payload = {
        "target_schema": list(TARGET_SCHEMA),
        "specs": specs_json,
        "standardized_vocabularies": vocab,
    }
    return _VERIFY_USER_PREFIX + json.dumps(payload, indent=2)
