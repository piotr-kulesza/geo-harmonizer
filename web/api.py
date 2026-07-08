"""Thin FastAPI backend for the interactive 3D disease-risk landscape. [web]

Loads a PRECOMPUTED bundle (fitted ``LandscapeModel`` + harmonized matrix +
standardized metadata) from a local cache dir — no network, ever. The handlers
stay thin: they call into ``core.landscape`` and shape JSON. Three routes:

- ``GET  /api/landscape``          — the baked payload (risk + a few signatures +
  example genes, each with a hull-clipped surface).
- ``POST /api/height``             — recompute one height selection (risk / gene /
  signature) from the in-memory matrix.
- ``POST /api/height/interpret``   — the Claude-Use beat: an injectable ``llm``
  maps a natural-language request onto a selection constrained to genes actually
  present, then returns the same height sub-shape.

Design constraints honored here:
- ``core`` stays pure — this module imports core, never the reverse.
- FastAPI/pydantic are lazy-imported inside :func:`create_app`, so ``import web``
  (and ``import core``) work without web deps installed.
- The Anthropic key lives only in the web layer's default llm (reusing the
  established seam in ``core.metadata``); tests inject a fake llm — no key, no
  network. The MCP path can supply the selection directly instead of the llm.

Serve for the dev frontend with::

    uvicorn "web.api:create_app" --factory --reload
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from core.landscape import (
    HeightLayer,
    LandscapeModel,
    gene_layer,
    height_subpayload,
    landscape_payload,
    risk_layer,
    signature_layer,
)

logger = logging.getLogger(__name__)

# Where the precompute script drops the bundle (overridable per-deploy).
DEFAULT_CACHE_DIR = os.environ.get("LANDSCAPE_CACHE", "outputs/landscape_cache")
BUNDLE_FILE = "bundle.pkl"

# The demo height menu baked into GET /api/landscape. Signatures are small,
# well-known marker sets; only their genes PRESENT in the matrix are used.
DEMO_SIGNATURES: dict[str, tuple[str, list[str]]] = {
    "proliferation": ("Proliferation", ["MKI67", "PCNA", "TOP2A", "CCNB1", "CCNB2", "BUB1", "AURKA"]),
    "immune": ("Immune activity", ["CD3D", "CD8A", "PTPRC", "GZMB", "CD2", "IL2RG", "CXCL9"]),
    "emt_stroma": ("EMT / stroma", ["VIM", "ZEB1", "FN1", "CDH2", "TWIST1", "SNAI2", "COL1A1"]),
}
EXAMPLE_GENES: list[str] = ["EGFR", "MKI67", "TP53", "CD8A", "VIM", "ERBB2"]

# Surface mesh resolution served to the frontend (per axis).
DEFAULT_GRID = 48


# --------------------------------------------------------------------------- #
# Bundle: the precomputed, network-free substrate the backend serves
# --------------------------------------------------------------------------- #
@dataclass
class Bundle:
    """Everything the backend needs, precomputed offline.

    Attributes:
        model: the fitted :class:`~core.landscape.LandscapeModel` (fixed map +
            risk model + CV C-index).
        matrix: harmonized/ComBat-corrected expression (genes x samples).
        samples_meta: standardized metadata (GSM index) — ``dataset`` + survival.
        pca: the precomputed fixed-basis progressive-PCA progression (Act 1), the
            JSON payload :func:`core.projection.progressive_projection` returns, or
            ``None`` if the cache predates it.
    """

    model: LandscapeModel
    matrix: pd.DataFrame
    samples_meta: pd.DataFrame
    pca: Optional[dict] = None


def save_bundle(bundle: Bundle, cache_dir: str = DEFAULT_CACHE_DIR) -> Path:
    """Persist a :class:`Bundle` as a single pickle (no network, no parquet dep)."""
    import pickle

    out = Path(cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / BUNDLE_FILE
    with path.open("wb") as fh:
        pickle.dump(bundle, fh)
    return path


def load_bundle(cache_dir: str = DEFAULT_CACHE_DIR) -> Optional[Bundle]:
    """Load the precomputed bundle, or ``None`` if the cache is absent/unreadable."""
    import pickle

    path = Path(cache_dir) / BUNDLE_FILE
    if not path.exists():
        logger.warning("landscape cache not found at %s — API will return 503.", path)
        return None
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except Exception as exc:  # corrupt/incompatible pickle — degrade, don't crash
        logger.error("failed to load landscape cache %s: %s", path, exc)
        return None


# --------------------------------------------------------------------------- #
# Selection -> height layer (pure; raises ValueError with a helpful message)
# --------------------------------------------------------------------------- #
class SelectionError(ValueError):
    """A bad height selection (unknown gene, empty signature, no risk model)."""


def _resolve_selection(bundle: Bundle, kind: str, value: Any):
    """Turn a ``{kind, value}`` selection into ``(key, label, kind, HeightLayer)``.

    Raises :class:`SelectionError` (mapped to HTTP 422) on an unusable selection,
    with a message that suggests valid symbols — never a 500.
    """
    matrix = bundle.matrix

    if kind == "risk":
        if bundle.model.risk_model is None:
            raise SelectionError("no survival-risk model is available for this landscape.")
        return "risk", "Predicted survival risk", "risk", risk_layer(bundle.model.risk_model, matrix)

    if kind == "gene":
        symbol = str(value or "").strip()
        if not symbol or symbol not in matrix.index:
            sample = ", ".join(list(matrix.index[:12]))
            raise SelectionError(
                f"gene {symbol!r} is not in the expression matrix. "
                f"Try a symbol like: {sample}."
            )
        return f"gene:{symbol}", symbol, "gene", gene_layer(matrix, symbol)

    if kind == "signature":
        if isinstance(value, str):
            if value not in DEMO_SIGNATURES:
                known = ", ".join(sorted(DEMO_SIGNATURES))
                raise SelectionError(
                    f"unknown signature {value!r}. Known signatures: {known}; "
                    "or pass a list of gene symbols."
                )
            label, genes = DEMO_SIGNATURES[value]
            name = value
        elif isinstance(value, (list, tuple)):
            genes, label, name = list(value), "Custom signature", "custom"
        else:
            raise SelectionError("signature 'value' must be a name or a list of gene symbols.")

        present = [g for g in genes if g in matrix.index]
        if not present:
            sample = ", ".join(list(matrix.index[:12]))
            raise SelectionError(
                "none of the signature's genes are in the matrix. "
                f"Available symbols include: {sample}."
            )
        key = f"sig:{name}"
        return key, label, "signature", signature_layer(matrix, present, key)

    raise SelectionError(f"unknown height kind {kind!r}; expected 'risk', 'gene' or 'signature'.")


def _build_base_layers(bundle: Bundle) -> dict:
    """The baked ``{key: (label, kind, HeightLayer)}`` menu for GET /api/landscape."""
    matrix = bundle.matrix
    layers: dict[str, tuple[str, str, HeightLayer]] = {}

    if bundle.model.risk_model is not None:
        layers["risk"] = ("Predicted survival risk", "risk", risk_layer(bundle.model.risk_model, matrix))

    for key, (label, genes) in DEMO_SIGNATURES.items():
        present = [g for g in genes if g in matrix.index]
        if present:
            layers[key] = (label, "signature", signature_layer(matrix, present, key))

    for gene in EXAMPLE_GENES:
        if gene in matrix.index:
            layers[f"gene:{gene}"] = (gene, "gene", gene_layer(matrix, gene))

    return layers


# --------------------------------------------------------------------------- #
# The Claude-Use beat: natural language -> selection (injectable llm seam)
# --------------------------------------------------------------------------- #
_INTERPRET_SYSTEM = (
    "You translate a cancer researcher's natural-language request for a landscape "
    "'height' into ONE concrete selection over an ovarian-cancer expression matrix. "
    "Choose exactly one:\n"
    "- kind='risk': the model's predicted survival risk (value = null).\n"
    "- kind='gene': a single gene symbol; value = that symbol.\n"
    "- kind='signature': a set of ~3-25 gene symbols capturing the concept "
    "(e.g. proliferation -> MKI67/PCNA/TOP2A...); value = the list of symbols.\n"
    "HARD CONSTRAINT: every gene symbol you output MUST appear verbatim in the "
    "provided AVAILABLE_GENES list. Prefer the KNOWN_SIGNATURES when one fits. "
    "Give a short human 'label' for the UI. Return ONLY a JSON object "
    '{"kind":..., "value":..., "label":...} — no prose, no code fences.'
)


def _render_interpret_user(query: str, genes: list[str], signatures: dict) -> str:
    payload = {
        "query": query,
        "known_signatures": {name: g for name, (_, g) in signatures.items()},
        "available_genes": genes,
    }
    return (
        "Map this request to a height selection. Only use gene symbols from "
        "available_genes.\n\n" + json.dumps(payload)
    )


def _interpret_query(
    bundle: Bundle,
    query: str,
    llm: Optional[Callable[[str, str], str]],
    cache_dir: str,
) -> dict:
    """One llm call: query -> ``{kind, value, label}`` (cached, offline-testable).

    Reuses the exact injectable+cached seam from ``core.metadata`` so tests pass a
    fake llm and the MCP path can bypass it. Returns a normalized selection dict.
    """
    from core.metadata import _cached_json, _default_llm  # reuse the seam

    llm = llm or _default_llm
    genes = sorted(bundle.matrix.index.astype(str))
    system = _INTERPRET_SYSTEM
    user = _render_interpret_user(query, genes, DEMO_SIGNATURES)
    data = _cached_json(llm, system, user, str(Path(cache_dir) / "interpret"))
    if not isinstance(data, dict):
        raise SelectionError("could not interpret the request.")
    return {"kind": data.get("kind"), "value": data.get("value"), "label": data.get("label")}


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app(
    *,
    bundle: Optional[Bundle] = None,
    llm: Optional[Callable[[str, str], str]] = None,
    cache_dir: Optional[str] = None,
    grid: int = DEFAULT_GRID,
):
    """Build the FastAPI app. Lazy-imports FastAPI so ``import web`` stays light.

    Args:
        bundle: inject a precomputed :class:`Bundle` (tests); if ``None``, load it
            from ``cache_dir``. A missing cache is NOT fatal — routes return 503.
        llm: inject a fake ``llm(system, user) -> str`` for the interpret route
            (tests / MCP). ``None`` uses the real Anthropic seam at call time.
        cache_dir: bundle + interpret-cache location (defaults to
            ``$LANDSCAPE_CACHE`` or ``outputs/landscape_cache``).
        grid: surface mesh resolution served to the frontend.
    """
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    state: dict[str, Any] = {
        "bundle": bundle if bundle is not None else load_bundle(cache_dir),
        "llm": llm,
        "cache_dir": cache_dir,
        "grid": grid,
        "base_payload": None,
    }

    app = FastAPI(title="GEO disease-risk landscape", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # permissive for the dev frontend
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _require_bundle() -> Bundle:
        b = state["bundle"]
        if b is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"landscape cache not loaded (looked in {state['cache_dir']!r}). "
                    "Run scripts/build_landscape_cache.py first."
                ),
            )
        return b

    def _height_or_422(b: Bundle, kind: str, value: Any) -> dict:
        try:
            key, label, resolved_kind, layer = _resolve_selection(b, kind, value)
        except SelectionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return height_subpayload(b.model.coords, key, label, resolved_kind, layer, grid=state["grid"])

    @app.get("/api/landscape")
    def get_landscape() -> dict:
        b = _require_bundle()
        if state["base_payload"] is None:
            layers = _build_base_layers(b)
            state["base_payload"] = landscape_payload(
                b.model, b.samples_meta, layers, grid=state["grid"]
            )
        return state["base_payload"]

    @app.get("/api/pca")
    def get_pca() -> dict:
        b = _require_bundle()
        prog = getattr(b, "pca", None)
        if not prog:
            raise HTTPException(
                status_code=503,
                detail=(
                    "PCA progression not in the cache. Re-run "
                    "scripts/build_landscape_cache.py to bake it."
                ),
            )
        return prog

    @app.post("/api/height")
    def post_height(payload: dict = Body(...)) -> dict:
        b = _require_bundle()
        return _height_or_422(b, payload.get("kind"), payload.get("value"))

    @app.post("/api/height/interpret")
    def post_interpret(payload: dict = Body(...)) -> dict:
        b = _require_bundle()
        query = str(payload.get("query") or "").strip()
        if not query:
            raise HTTPException(status_code=422, detail="'query' is required.")
        try:
            selection = _interpret_query(b, query, state["llm"], state["cache_dir"])
        except SelectionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        out = _height_or_422(b, selection["kind"], selection["value"])
        out["selection"] = {
            "kind": selection["kind"],
            "value": selection["value"],
            "label": selection.get("label") or out.get("label"),
        }
        return out

    # One-origin demo: if the frontend has been built, serve it at "/" so
    # `uvicorn web.api:app` hosts API + UI together (no CORS on camera). Mounted
    # LAST so /api/* routes above win; a no-op when the build is absent (tests
    # never build, so this changes nothing for them).
    dist = Path(__file__).resolve().parent / "frontend" / "dist"
    if dist.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")
        logger.info("mounted built frontend at / from %s", dist)

    return app


def __getattr__(name: str):
    """Lazily build ``web.api.app`` on first access (PEP 562).

    Lets ``uvicorn web.api:app`` work while keeping ``import web`` / ``import
    web.api`` free of FastAPI — the app (and thus FastAPI) is only constructed
    when something actually reads the ``app`` attribute.
    """
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
