"""Claude-driven metadata parsing + cross-dataset verification. [Day 3 — Claude]

Stubs only today. This is the Claude Use quadrant (see CLAUDE.md + TICKETS.md
Epic 4). Settled behavior for Day 3:
- ``parse_metadata`` makes ONE batched Anthropic call per dataset (never per
  field/sample) mapping that dataset's full raw field set -> TARGET_SCHEMA.
- ``verify_mapping`` is a second Claude pass checking cross-dataset consistency
  and flagging the uncertain.
- The Anthropic call must be INJECTABLE, not hardwired (the MCP metadata seam).

Do not hardwire an internal API call here in a way that forecloses dual-mode.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# The unified target schema the raw fields map onto. Finalized on Day 3 (Epic 4.1).
TARGET_SCHEMA: tuple[str, ...] = (
    "stage",
    "grade",
    "histology",
    "survival_days",
    "event",
)


def parse_metadata(metadata: pd.DataFrame, *args: Any, **kwargs: Any):
    """Map one dataset's raw metadata fields onto ``TARGET_SCHEMA``. [Day 3]"""
    raise NotImplementedError("parse_metadata is implemented on Day 3 (Epic 4).")


def verify_mapping(*args: Any, **kwargs: Any):
    """Second Claude pass: check cross-dataset consistency, flag mismatches. [Day 3]"""
    raise NotImplementedError("verify_mapping is implemented on Day 3 (Epic 4).")
