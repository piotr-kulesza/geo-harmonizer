"""Offline tests for the web backend — synthetic bundle, fake llm; no network, no key."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")  # web layer needs fastapi; core does not
with warnings.catch_warnings():
    # Starlette's TestClient emits a third-party httpx deprecation at import; not ours.
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient  # noqa: E402

from core.landscape import LandscapeModel, cv_cindex, embed, fit_risk  # noqa: E402
from core.projection import progressive_projection  # noqa: E402
from web.api import Bundle, create_app  # noqa: E402


def _bundle_with_pca(seed: int = 0) -> Bundle:
    """A bundle carrying a precomputed PCA progression (for GET /api/pca)."""
    b = _bundle(seed=seed)
    batch = b.samples_meta["dataset"]
    # Simulated ComBat: remove each batch's per-gene mean, restore the global mean.
    gm = b.matrix.mean(axis=1)
    corrected = b.matrix.copy()
    for acc in batch.unique():
        cols = [c for c in b.matrix.columns if batch[c] == acc]
        corrected[cols] = b.matrix[cols].sub(b.matrix[cols].mean(axis=1), axis=0).add(gm, axis=0)
    prog = progressive_projection(b.matrix, batch, corrected=corrected)
    return Bundle(model=b.model, matrix=b.matrix, samples_meta=b.samples_meta, pca=prog)


# --------------------------------------------------------------------------- #
# Synthetic, network-free bundle
# --------------------------------------------------------------------------- #
def _bundle(n_samples: int = 44, seed: int = 0) -> Bundle:
    """A tiny fitted landscape: real marker symbols + a planted survival signal.

    Dataset A carries survival; dataset B does not (held out).
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n_samples)

    # A few real symbols the demo menu/signatures reference, plus noise genes.
    real = ["MKI67", "PCNA", "TOP2A", "EGFR", "CD8A", "TP53", "VIM", "PTPRC"]
    noise = [f"g{i}" for i in range(40)]
    genes = real + noise
    data = np.empty((len(genes), n_samples))
    for i in range(len(real)):  # signal genes track z
        data[i] = 3.0 * z + rng.normal(scale=0.5, size=n_samples)
    for i in range(len(real), len(genes)):
        data[i] = rng.normal(size=n_samples)

    samples = [f"GSM{i:02d}" for i in range(n_samples)]
    matrix = pd.DataFrame(data, index=genes, columns=samples)

    durations = np.clip(500.0 * np.exp(-0.8 * z) + rng.normal(scale=15, size=n_samples), 10, None)
    datasets = ["A"] * 30 + ["B"] * (n_samples - 30)
    meta = pd.DataFrame(
        {"dataset": datasets, "survival_days": durations, "event": 1}, index=samples
    )
    meta.loc[[s for s, d in zip(samples, datasets) if d == "B"], ["survival_days", "event"]] = np.nan

    surv = meta.loc[meta["dataset"] == "A", ["survival_days", "event"]]
    coords = embed(matrix, n_components=2)
    model = fit_risk(matrix, surv, n_pca=10)
    c = cv_cindex(matrix, surv, folds=4, n_pca=10)
    lm = LandscapeModel(coords=coords, risk_model=model, cindex=c)
    return Bundle(model=lm, matrix=matrix, samples_meta=meta)


class CountingLLM:
    """Fake llm(system, user) -> str returning canned JSON; records call count."""

    def __init__(self, response: dict):
        self._response = response
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return json.dumps(self._response)


def _client(bundle=None, llm=None, cache_dir="unused", tmp_path=None) -> TestClient:
    cache = str(tmp_path) if tmp_path is not None else cache_dir
    return TestClient(create_app(bundle=bundle, llm=llm, cache_dir=cache, grid=24))


def _has_null_and_finite(z) -> bool:
    flat = [cell for row in z for cell in row]
    return any(c is None for c in flat) and any(isinstance(c, float) for c in flat)


# --------------------------------------------------------------------------- #
# GET /api/landscape
# --------------------------------------------------------------------------- #
def test_get_landscape_returns_payload(tmp_path):
    client = _client(bundle=_bundle(), tmp_path=tmp_path)
    resp = client.get("/api/landscape")
    assert resp.status_code == 200
    payload = resp.json()
    assert set(payload) == {"samples", "surfaces", "height_options", "meta"}
    assert payload["height_options"]  # risk + signatures + example genes
    assert payload["meta"]["held_out"] == ["B"]
    assert payload["samples"][0]["id"] == "GSM00"


# --------------------------------------------------------------------------- #
# POST /api/height
# --------------------------------------------------------------------------- #
def test_post_height_gene(tmp_path):
    client = _client(bundle=_bundle(), tmp_path=tmp_path)
    resp = client.post("/api/height", json={"kind": "gene", "value": "EGFR"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "gene" and body["key"] == "gene:EGFR"
    assert set(body["heights"]) == set(f"GSM{i:02d}" for i in range(44))
    assert _has_null_and_finite(body["surface"]["z"])  # finite inside hull, null outside


def test_post_height_signature(tmp_path):
    client = _client(bundle=_bundle(), tmp_path=tmp_path)
    resp = client.post("/api/height", json={"kind": "signature", "value": "proliferation"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "signature"
    assert _has_null_and_finite(body["surface"]["z"])


def test_post_height_custom_signature_list(tmp_path):
    client = _client(bundle=_bundle(), tmp_path=tmp_path)
    resp = client.post("/api/height", json={"kind": "signature", "value": ["MKI67", "PCNA", "TOP2A"]})
    assert resp.status_code == 200
    assert resp.json()["kind"] == "signature"


def test_post_height_unknown_gene_is_422(tmp_path):
    client = _client(bundle=_bundle(), tmp_path=tmp_path)
    resp = client.post("/api/height", json={"kind": "gene", "value": "NOTAGENE"})
    assert resp.status_code == 422
    assert "not in the expression matrix" in resp.json()["detail"]


def test_post_height_empty_signature_is_422(tmp_path):
    client = _client(bundle=_bundle(), tmp_path=tmp_path)
    resp = client.post("/api/height", json={"kind": "signature", "value": ["FAKE1", "FAKE2"]})
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# POST /api/height/interpret — the Claude-Use beat (fake llm)
# --------------------------------------------------------------------------- #
def test_interpret_maps_query_to_height(tmp_path):
    llm = CountingLLM({"kind": "gene", "value": "EGFR", "label": "EGFR expression"})
    client = _client(bundle=_bundle(), llm=llm, tmp_path=tmp_path)

    resp = client.post("/api/height/interpret", json={"query": "EGFR signaling"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["selection"] == {"kind": "gene", "value": "EGFR", "label": "EGFR expression"}
    assert body["key"] == "gene:EGFR"
    assert set(body["heights"]) == set(f"GSM{i:02d}" for i in range(44))
    assert len(llm.calls) == 1  # exactly one llm call (cached seam, no per-sample)


def test_interpret_llm_called_at_most_once(tmp_path):
    llm = CountingLLM({"kind": "signature", "value": ["MKI67", "PCNA", "TOP2A"], "label": "Proliferation"})
    client = _client(bundle=_bundle(), llm=llm, tmp_path=tmp_path)
    resp = client.post("/api/height/interpret", json={"query": "how proliferative"})
    assert resp.status_code == 200
    assert resp.json()["selection"]["kind"] == "signature"
    assert len(llm.calls) <= 1


def test_interpret_hallucinated_gene_is_422(tmp_path):
    # Even if the model returns a gene not in the matrix, the route validates -> 422.
    llm = CountingLLM({"kind": "gene", "value": "GENE_NOT_PRESENT", "label": "nope"})
    client = _client(bundle=_bundle(), llm=llm, tmp_path=tmp_path)
    resp = client.post("/api/height/interpret", json={"query": "something"})
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# GET /api/pca — the harmonization progression (Act 1)
# --------------------------------------------------------------------------- #
def test_get_pca_returns_progression(tmp_path):
    client = _client(bundle=_bundle_with_pca(), tmp_path=tmp_path)
    resp = client.get("/api/pca")
    assert resp.status_code == 200
    prog = resp.json()
    assert set(prog) == {"axes", "order", "batch", "steps"}
    assert len(prog["axes"]["explained_variance"]) == 2
    assert len(prog["steps"]) == len(prog["order"])
    last = prog["steps"][-1]
    assert set(last) == {"k", "included", "raw", "combat"}
    assert set(last["raw"]) == {"coords", "silhouette"}
    # ComBat state present and reduces batch separation.
    assert last["combat"] is not None
    assert last["raw"]["silhouette"] > last["combat"]["silhouette"]


def test_get_pca_without_progression_is_503(tmp_path):
    # A bundle that predates Act 1 (no pca) -> 503 with a rebuild hint, not a 500.
    client = _client(bundle=_bundle(), tmp_path=tmp_path)
    resp = client.get("/api/pca")
    assert resp.status_code == 503
    assert "PCA progression" in resp.json()["detail"]


# --------------------------------------------------------------------------- #
# Missing-cache path — 503, no crash
# --------------------------------------------------------------------------- #
def test_missing_cache_returns_503(tmp_path):
    # No bundle injected and an empty cache dir -> load_bundle returns None.
    client = _client(bundle=None, tmp_path=tmp_path)
    for method, path, body in [
        ("get", "/api/landscape", None),
        ("get", "/api/pca", None),
        ("post", "/api/height", {"kind": "gene", "value": "EGFR"}),
        ("post", "/api/height/interpret", {"query": "x"}),
    ]:
        resp = getattr(client, method)(path, **({"json": body} if body else {}))
        assert resp.status_code == 503
        assert "cache not loaded" in resp.json()["detail"]
