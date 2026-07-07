"""geo-harmonizer core — pure, interface-agnostic pipeline logic.

The iron rule (CLAUDE.md): no web/MCP/UI knowledge lives here. Only Day-1
symbols are real; the rest are stubs that raise ``NotImplementedError``.
"""

from core.fetch import FetchResult, fetch_gse, load_matrix_from_file

__all__ = ["FetchResult", "fetch_gse", "load_matrix_from_file"]
