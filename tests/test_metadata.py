"""Offline tests for core.metadata — a FAKE llm; no network, no key, no credits."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from core.metadata import (
    FieldMap,
    MappingResult,
    MappingSpec,
    TARGET_SCHEMA,
    apply_mapping,
    build_dataset_summary,
    build_mapping_spec,
    parse_metadata,
    verify_mapping,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _raw(accession_seed: int = 0) -> pd.DataFrame:
    """A raw metadata frame shaped like core.fetch produces (GSM index)."""
    idx = pd.Index([f"GSM{accession_seed}{i}" for i in range(3)], name="sample")
    return pd.DataFrame(
        {
            "title": ["t1", "t2", "t3"],
            "source_name_ch1": ["ovarian tumor"] * 3,
            "platform_id": ["GPL570"] * 3,  # should be ignored by the summary
            "characteristics_ch1": ["blob"] * 3,  # raw blob, ignored by summary
            "char::stage": ["IIIc", "Ia", "IV"],
            "char::grade": ["3", "2", "3"],
            "char::histology": ["Ser", "PapSer", "Endo"],
            "char::os_years": ["1.0", "2.0", "0.5"],
            "char::status": ["DOD", "NED", "AWD"],
        },
        index=idx,
    )


# A full canned spec the fake llm returns (matches the raw frame above).
CANNED_SPEC = {
    "fields": {
        "stage": {"source": "char::stage", "transform": "uppercase", "note": "FIGO", "confidence": 0.9},
        "grade": {"source": "char::grade", "transform": "identity", "confidence": 0.8},
        "histology": {
            "source": "char::histology",
            "transform": "value_map",
            "map": {"Ser": "serous", "PapSer": "serous"},
            "confidence": 0.7,
        },
        "survival_days": {"source": "char::os_years", "transform": "scale", "factor": 365.25},
        "event": {"source": "char::status", "transform": "identity"},
    }
}


class CountingLLM:
    """A fake llm(system, user) -> str that records calls and returns canned JSON."""

    def __init__(self, response, fence: bool = False):
        self._response = response
        self._fence = fence
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        text = json.dumps(self._response)
        return f"```json\n{text}\n```" if self._fence else text


# --------------------------------------------------------------------------- #
# apply_mapping — each transform (pure, no llm)
# --------------------------------------------------------------------------- #
def _spec(**overrides) -> MappingSpec:
    fields = {t: FieldMap(source=None, transform="missing") for t in TARGET_SCHEMA}
    fields.update(overrides)
    return MappingSpec(dataset="GSE_TEST", fields=fields)


def test_apply_uppercase_stage():
    out = apply_mapping(_raw(), _spec(stage=FieldMap("char::stage", "uppercase")))
    assert list(out["stage"]) == ["IIIC", "IA", "IV"]


def test_apply_scale_years_to_days():
    out = apply_mapping(_raw(), _spec(survival_days=FieldMap("char::os_years", "scale", {"factor": 365.25})))
    assert out["survival_days"].iloc[0] == pytest.approx(365.25)
    assert out["survival_days"].iloc[2] == pytest.approx(182.625)


def test_apply_value_map_histology():
    fm = FieldMap("char::histology", "value_map", {"map": {"Ser": "serous", "PapSer": "serous"}})
    out = apply_mapping(_raw(), _spec(histology=fm))
    # Ser and PapSer -> serous; Endo is unmapped -> NaN
    assert out["histology"].iloc[0] == "serous"
    assert out["histology"].iloc[1] == "serous"
    assert pd.isna(out["histology"].iloc[2])


def test_apply_constant_grade():
    out = apply_mapping(_raw(), _spec(grade=FieldMap(None, "constant", {"value": "high"})))
    assert list(out["grade"]) == ["high", "high", "high"]


def test_apply_event_identity():
    out = apply_mapping(_raw(), _spec(event=FieldMap("char::status", "identity")))
    assert list(out["event"]) == ["DOD", "NED", "AWD"]


def test_apply_missing_gives_nan_column():
    out = apply_mapping(_raw(), _spec())  # everything missing
    assert out["stage"].isna().all()
    # schema columns present, plus dataset column
    assert list(out.columns) == [*TARGET_SCHEMA, "dataset"]
    assert (out["dataset"] == "GSE_TEST").all()
    assert out.index.name == "sample"


def test_apply_regex_extract():
    raw = _raw()
    raw["char::tissue"] = ["late-stage high-grade", "normal epithelium", "grade 3 tumor"]
    fm = FieldMap("char::tissue", "regex_extract", {"pattern": r"(high-grade|grade \d)", "group": 1})
    out = apply_mapping(raw, _spec(grade=fm))
    assert out["grade"].iloc[0] == "high-grade"
    assert pd.isna(out["grade"].iloc[1])
    assert out["grade"].iloc[2] == "grade 3"


def test_apply_mapping_is_pure_and_deterministic():
    spec = _spec(stage=FieldMap("char::stage", "uppercase"))
    a = apply_mapping(_raw(), spec)
    b = apply_mapping(_raw(), spec)
    pd.testing.assert_frame_equal(a, b)  # no llm involved, same output


# --------------------------------------------------------------------------- #
# build_dataset_summary
# --------------------------------------------------------------------------- #
def test_summary_uses_char_and_skips_platform_and_blob():
    summary = build_dataset_summary(_raw(), "GSE9891")
    cols = summary["columns"]
    assert "char::stage" in cols
    assert "source_name_ch1" in cols
    assert "title" in cols
    assert "platform_id" not in cols  # skipped
    assert "characteristics_ch1" not in cols  # raw blob skipped
    assert summary["accession"] == "GSE9891"
    assert cols["char::stage"] == ["IIIc", "IV", "Ia"]  # sorted (case-sensitive) distinct


# --------------------------------------------------------------------------- #
# parse_metadata — one llm call per dataset, and fence-tolerant parsing
# --------------------------------------------------------------------------- #
def test_parse_metadata_one_llm_call_per_dataset(tmp_path):
    llm = CountingLLM(CANNED_SPEC)
    raw_by_dataset = {"GSE1": _raw(1), "GSE2": _raw(2)}

    result = parse_metadata(raw_by_dataset, llm=llm, cache_dir=str(tmp_path))

    assert len(llm.calls) == 2  # exactly one per dataset, never per field/sample
    assert isinstance(result, MappingResult)
    # standardized concatenates both datasets: 3 + 3 samples
    assert result.standardized.shape[0] == 6
    assert set(result.standardized["dataset"]) == {"GSE1", "GSE2"}
    # spec applied: uppercase + scale worked
    assert "IIIC" in set(result.standardized["stage"])
    assert result.standardized["survival_days"].max() == pytest.approx(2 * 365.25)


def test_build_mapping_spec_survives_json_fences(tmp_path):
    llm = CountingLLM(CANNED_SPEC, fence=True)  # wraps reply in ```json fences
    summary = build_dataset_summary(_raw(), "GSE1")

    spec = build_mapping_spec(summary, llm, cache_dir=str(tmp_path))

    assert spec.fields["stage"].transform == "uppercase"
    assert spec.fields["histology"].params["map"]["PapSer"] == "serous"
    assert spec.fields["stage"].confidence == 0.9


def test_parse_metadata_uses_provided_specs_without_llm(tmp_path):
    # MCP path: caller supplies specs -> no llm call.
    llm = CountingLLM(CANNED_SPEC)
    spec = MappingSpec(
        dataset="GSE1",
        fields={t: FieldMap(None, "missing") for t in TARGET_SCHEMA}
        | {"stage": FieldMap("char::stage", "uppercase")},
    )
    result = parse_metadata({"GSE1": _raw(1)}, llm=llm, specs={"GSE1": spec}, cache_dir=str(tmp_path))

    assert len(llm.calls) == 0  # provided spec used; llm untouched
    assert "IIIC" in set(result.standardized["stage"])


# --------------------------------------------------------------------------- #
# verify_mapping — one call, surfaces flags
# --------------------------------------------------------------------------- #
def test_verify_mapping_surfaces_flags(tmp_path):
    verify_response = {
        "flags": [
            {
                "severity": "warn",
                "field": "survival_days",
                "datasets": ["GSE1", "GSE2"],
                "message": "GSE2 survival may still be in years.",
            },
            {"severity": "bogus", "field": "grade", "datasets": [], "message": "x"},
        ]
    }
    parse_llm = CountingLLM(CANNED_SPEC)
    result = parse_metadata({"GSE1": _raw(1), "GSE2": _raw(2)}, llm=parse_llm, cache_dir=str(tmp_path))

    verify_llm = CountingLLM(verify_response)
    result = verify_mapping(result, llm=verify_llm, cache_dir=str(tmp_path))

    assert len(verify_llm.calls) == 1  # exactly one verify call
    assert len(result.flags) == 2
    assert result.flags[0]["severity"] == "warn"
    assert result.flags[0]["field"] == "survival_days"
    assert result.flags[1]["severity"] == "info"  # invalid severity normalized
