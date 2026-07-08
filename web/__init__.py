"""Web layer — a thin FastAPI wrapper over ``core.landscape``. [web]

Iron rule: this package is a THIN adapter. All science lives in ``core``; the
handlers only marshal requests, call core, and shape JSON. FastAPI is
lazy-imported inside :func:`web.api.create_app` so ``import core`` (and anyone
who only needs the pure pipeline) never pays for web dependencies.
"""

from __future__ import annotations

from .api import Bundle, create_app, load_bundle, save_bundle

__all__ = ["create_app", "Bundle", "load_bundle", "save_bundle"]
