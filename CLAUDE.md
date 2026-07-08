# CLAUDE.md — build contract for this repo

Read this before writing any code. It encodes decisions already made; do not
reopen them without a clear reason.

## What this is
A tool that harmonizes GEO transcriptomic datasets: give it GEO series
accessions, get back a comparable expression matrix + a unified metadata table
in minutes. Named user: a researcher running cross-dataset meta-analyses (the
author does this weekly). Two entry points over one core: a web service (the
demo star) and an MCP server (Claude / Claude Science calls the same core).

Hackathon: *Built with Claude: Life Sciences — Builder Track*. Submission due
**Mon 13 July, 9:00 PM ET**. Deliverable is a **pitch** (3-min demo video +
public repo + 100–200 word description), not a README dump.

## Disease risk landscape (built on the harmonized substrate)
On top of the harmonized+ComBat matrix we build a **disease risk landscape**
(`core/landscape.py`): embed every tumor into a fixed 2D map, and drape a
**pluggable per-sample height** over it. The validated default height is
predicted survival risk (PCA→penalized Cox, fit + cross-validated ONLY on the
cohorts that carry survival, then predicted for all samples). Height is
swappable — a gene signature or single gene works on the same map ("choose the
height while analyzing"). The cross-validated C-index (≥ ~0.6) is the gate that
says the landscape is meaningful. Core stays pure: the 3D render lives in the
check script, not in `core/`.

## Iron rule — the core is interface-agnostic
`core/` is pure logic. Functions take arguments, return data. **Zero** web/MCP/UI
knowledge in the core. The web layer and the MCP layer are both thin wrappers.
This is the only reason two interfaces are possible in one week. If you find
yourself importing FastAPI or MCP types inside `core/`, stop.

## Architecture
```
core/         pure pipeline (no UI)
  fetch.py        fetch_gse(accession) -> FetchResult        [Day 1 ✓]
  harmonize.py    map_probes(), to_log2()                    [Day 2]
  merge.py        merge({acc: matrix}) -> (matrix, batch)    [Day 2 end]
  combat.py       combat(merged, batch)                      [Day 2 end/3]
  metadata.py     parse_metadata() + verify_mapping()        [Day 3 — Claude]
  landscape.py    embed(), fit_risk(), surface()             [Landscape v1]
web/          React + FastAPI thin layer                     [Day 4-5]
mcp/          MCP server thin layer                          [Day 4]
tests/        pytest
scripts/      local runnable scripts (smoke tests, demos)
docs/         series.md, design-brief.md
```

## Settled technical decisions (do not reopen)
- **Fetch = series matrix over HTTPS first, GEOparse/SOFT as fallback.** Primary
  path pulls `GSE…_series_matrix.txt.gz` over HTTPS (what R's `getGEO` uses):
  compact, already-pivoted, reliable. Parsing it is a bounded ~30-line TSV reader,
  the one justified exception to "no hand-rolled GEO parser" — GEOparse still
  handles the SOFT fallback (no SOFT parser of our own). Series that fail both
  return `needs_manual_upload`, not a crash. Honest limitation.
- **Existing ComBat only** (inmoose `pycombat_norm` / neuroCombat). OFF by
  default, framed as an option. The scientifically hard part is deliberately cut.
- **Claude parses free-text metadata**, and a **second Claude pass verifies** the
  mapping across datasets. This is the Claude Use quadrant (25%) — reasoning, not
  regex. **Batch the calls**: one call per dataset's full field set, never per
  field/sample (saves credits; scores as "pushed past first idea", Depth 20%).
- **Web-first.** The web front end gets the most attention. MCP is a thin second
  entry point shown briefly. One smooth demo beats two that break.
- **New Work Only.** Everything from scratch this week. No code from prior
  projects (the transcriptomic agent, pdf_extractor, etc.). Libraries are fine.

## The signature demo moment
Progressive PCA: add series one by one -> samples cluster into separate clouds by
dataset (the problem becomes visible) -> turn on ComBat -> clouds merge by
biology (chaos -> order). Build the STATIC version first (recompute + show after
each step). Animated transitions are polish, only if time remains.

**Fixed-projection PCA (do this or the animation reads as noise).** PCA has
sign/axis ambiguity: refitting after each added dataset makes existing points
flip, rotate, and reshuffle, so "adding a series" looks like every point jumping
randomly — the opposite of the thesis. Fit PCA ONCE on the final combined matrix
and project each step into that fixed space (or Procrustes-align each step to the
previous). This changes the data flow — decide it before building the animation.

## Demo reliability (review-derived; these protect the 30%)
- **Never live-fetch GEO on camera.** GEOparse pulls large SOFT/matrix files over
  a flaky endpoint; on a cold free-tier host that's a hang waiting to happen.
  Pre-fetch the 2–3 demo series into the cache and record against the cache. Cache
  probe/gene annotations (mygene/pybiomart) too. A live-fetch of one small series
  can be shown once; the money shot runs on cache.
- **Curate 2–3 demo datasets, not 5.** Every extra cloud is one more thing that
  can look wrong on video. The set must serve three things at once: (1) a clean
  PCA hero visual, (2) a real metadata mismatch for the verifier to catch, (3) a
  real batch-masked domain marker that appears after ComBat.
- **Cross-platform, but SAME MODALITY.** Cross-GPL is the product's pitch, kept in
  the demo — but all demo series must be the same modality (all log-intensity
  microarray; do NOT mix in RNA-seq counts). ComBat across modalities is garbage.
  The shared-gene count + PCA before/after check is a HARD GATE before committing
  the hero visual (Epic 3.4). If it doesn't look clean, swap series then.
- **Double-log2 guard.** Many GEO series are already log-transformed; log2-ing
  again squashes everything and quietly ruins the PCA. Heuristic before
  transforming: max value < ~30 → probably already logged, skip.
- **ComBat preconditions:** ≥2 batches, enough samples per batch, no covariate
  fully confounded with batch. Check at curation time.
- **Seed a real metadata mismatch** in the demo data (survival months vs days,
  "grade 3" vs "G3" vs "high grade", staging-vocab clash). Clean data → the
  verifier finds nothing → the 25% Claude-Use loop is invisible = unscored.
- **De-risk the back half:** smoke-test a throwaway deploy + one MCP tool plugged
  into Claude by end of Day 4 (not first attempt on Day 6/7). Full dry-run
  recording on Day 6, so Day 7 is a re-take, not first contact.

## The MCP metadata seam (decide on Day 3; keep it open until then)
The iron rule holds for fetch/merge/combat — they don't care who calls them. But
metadata reasoning is the one seam where the caller matters. Structure
`parse_metadata` so the Anthropic API call is INJECTABLE, not hardwired. Two
modes we choose between on Day 3 with real data:
  - web path: core makes the Anthropic call internally;
  - MCP path: expose raw metadata + target schema and let the orchestrating
    Claude (Claude Science) map/verify natively — more on-theme for Claude Use.
Leaning dual-mode. Do not hardwire an internal API call that forecloses it.

## Data conventions
- Expression matrices are **features (rows) x samples (columns)**. GSM ids are
  the sample columns. This shape is uniform whether data came from GEO or a
  manual upload, so downstream code never branches on the source.
- Batch label per sample = its source accession (what ComBat removes, what the
  PCA colours by).
- Metadata frames are **samples (rows) x fields (columns)**, indexed by GSM.

## Code conventions
- Python 3.10+, type hints, dataclasses for structured returns.
- Core functions never `print` and never raise for *expected* failure modes
  (network/parse/shape) — they return a structured result with a `message`.
  Diagnostics go through `logging`.
- Human-facing copy uses the researcher's language ("compare datasets",
  "download unified data"), not implementation language ("run pipeline").

## Judging criteria — weigh every decision against these
Demo 30% (works, cool to watch) · Claude Use 25% (metadata parse + verifier, MCP
in Claude Science) · Impact 25% (named user + product + Claude Science) · Depth
20% (batching, verifier, real craft). The deliverable sells the PROBLEM and the
VISION, not the architecture.

## Git workflow
Public GitHub repo (`geo-harmonizer`, MIT). **End every task/prompt by committing
with a clear message and pushing to `origin main`** — small, frequent commits, one
per ticket or coherent change. The commit history is part of the "New Work Only"
story, so keep it clean and honest.

## Working style (author)
Concise, direct, substance-first. No filler, no motivational phrasing. Give
critical feedback straight. Author brings the ~20% domain knowledge (what makes
data comparable); Claude writes the ~80% boilerplate. Expose that split in the
pitch — it is the thesis of the whole series.
