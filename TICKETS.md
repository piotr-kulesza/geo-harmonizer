# PRD + ticket breakdown

Break the whole thing into tickets **before** coding (Tekton's winning pattern).
Each ticket is a single, self-contained task with an acceptance check, sized so a
Claude Code workflow can take it end to end. Work top to bottom; parallelize
within an epic where the tickets don't touch the same files.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done · **(Claude)** = uses the
Claude API.

---

## PRD (one screen)

**Problem.** Before any cross-dataset transcriptomic meta-analysis, a researcher
loses days getting data from different GEO series into a comparable state:
different platforms, different probe IDs, expression on different scales, and
sample metadata written as inconsistent free text. It is the most tedious, least
intellectual part of the work.

**Product.** Give the tool a handful of GEO accessions. It returns (1) a single
comparable expression matrix on a shared gene set and (2) a unified, standardized
metadata table — in minutes, self-serve. Optional ComBat batch correction. The
same core is reachable two ways: a web service (primary) and an MCP server that
Claude / Claude Science drives (secondary).

**Named user.** A translational-oncology researcher running HGSOC meta-analyses
(the author). "I do this every week."

**Non-goals.** Not an analysis suite. Not enterprise (cf. Polly). Not a
clone-and-run repo. Not a new batch-correction method. Not agent orchestration.

**Success = the demo.** Progressive PCA shows the problem (clouds split by
dataset) and the fix (ComBat merges them by biology), legibly, in one take.

---

## Epic 0 — Repo & setup (Day 1)
- [x] 0.1 Empty repo, MIT license, `core/ web/ mcp/ tests/ scripts/ docs/`.
- [x] 0.2 `.gitignore`, `requirements.txt`, `README` stub, `CLAUDE.md`.
- [x] 0.3 This ticket list (PRD + breakdown) written before coding.
- [x] 0.4 `git init`, first commit, push to a **public** GitHub repo (MIT).
- [ ] 0.5 Create a venv locally and `pip install -r requirements.txt`; confirm it
      resolves on your machine (the build sandbox is offline — this must be local).
- [ ] 0.6 Redeem the $200 API credits on this account's console; set
      `ANTHROPIC_API_KEY` in a local `.env` (gitignored).

## Epic 1 — core.fetch (Day 1)
- [x] 1.1 `fetch_gse(accession) -> FetchResult` via GEOparse (matrix + metadata).
- [x] 1.2 Manual-upload fallback: `needs_manual_upload` status + `message`; no
      crashes on parse/network/shape failure.
- [x] 1.3 `load_matrix_from_file()` for the fallback path (CSV/TSV).
- [ ] 1.4 Curate 2–3 demo HGSOC series (see `docs/series.md`): cross-platform but
      **same modality** (all log-intensity microarray), serving all three demo
      jobs at once — clean PCA + seeded metadata mismatch + batch-masked marker.
      Confirm each fetches or falls back cleanly **locally**; check ComBat
      preconditions (≥2 batches, enough samples, no full confounding).
- [x] 1.5 `scripts/smoke_fetch.py`: run fetch over the set, print
      `FetchResult.summary()` for each. This is the Day-1 acceptance check.
- [x] 1.6 Unit test: fallback path returns a well-formed result (mock a bad GSE).
- [x] 1.7 `scripts/prewarm_cache.py`: fetch the demo series into `data/cache/` so
      the demo NEVER live-fetches on camera. Record against the cache.
- [x] 1.8 **Harden fetch download** (GEOparse's FTP truncates large series, e.g.
      GSE9891 → "Downloaded size do not match"). Two levers, decide later — do NOT
      block Day 1/2 on this (cache-once is the strategy):
        (a) Transport: fetch over HTTPS + retry-with-backoff into `data/cache/`,
            let GEOparse parse the local file (no FTP).
        (b) Source: prefer `GSE…_series_matrix.txt.gz` (genes×samples + metadata,
            ~10x smaller, far less truncation) over `family.soft.gz`; probe→gene
            mapping happens separately anyway.

## Epic 2 — core.harmonize (Day 2)
- [x] 2.1 **(Claude, tool-choice)** Decide probe->symbol source per platform
      (mygene vs pybiomart vs GPL table). Ask Claude to compare, don't just code.
- [x] 2.2 `map_probes(expr, platform_id, collapse)` -> gene x sample matrix.
- [x] 2.3 Many-probes-to-one-gene collapse (`max`/`mean`); drop unmapped.
- [x] 2.4 `to_log2()` with a **double-log2 guard**: if the matrix max is < ~30,
      treat it as already logged and skip; else log2. Never blindly transform —
      double-logging squashes everything and quietly ruins the PCA.
- [x] 2.5 End-to-end terminal test: list of GSE -> per-series gene x sample matrix.
- [x] 2.6 Unit test: collapse + log2 on a tiny synthetic matrix (both scales).
- [x] 2.7 Cache probe/gene annotations (mygene/pybiomart are network-flaky) so
      harmonize is offline-repeatable and demo-safe.

## Epic 3 — core.merge + combat (Day 2 end — pure mechanics, moved off Day 3)
- [x] 3.1 `merge({acc: matrix})` -> merged matrix on shared gene set + batch labels.
- [x] 3.2 Report shared-gene count and per-dataset sample counts (UI needs these).
- [ ] 3.3 `combat(merged, batch)` wrapping inmoose/neuroCombat; OFF by default.
- [ ] 3.4 **PCA validation (HARD GATE for the whole demo):** using a FIXED
      projection (fit PCA once on the final combined matrix, project each step into
      it — never refit per step), compute PCA before/after ComBat on the demo set.
      Confirm clouds visibly split by dataset then merge AND the shared-gene count
      is large enough to plot. If not clean, swap series NOW — not on Day 5.
- [ ] 3.5 Unit test: merge intersects genes correctly; combat preserves shape.

## Epic 4 — core.metadata + Claude (Day 3 — the Claude Use quadrant)
- [ ] 4.1 Define `TARGET_SCHEMA` (stage, grade, histology, survival_days, event).
- [ ] 4.2 **(Claude, MAP)** `parse_metadata()`: one batched call per dataset maps
      that dataset's full field set -> schema. Never one call per field/sample.
- [ ] 4.3 **(Claude, VERIFY)** `verify_mapping()`: second pass checks cross-dataset
      consistency (grade 3 == G3? survival units match?) and flags the uncertain.
- [ ] 4.4 Return an audit trail (`raw_to_schema`) + `flags` for user approval.
- [ ] 4.5 User approve/correct step (not blind automation) — surfaced in the UI later.
- [ ] 4.6 Prompt + response caching so re-runs don't re-spend credits.
- [ ] 4.7 **CORE FREEZE.** If time is short, trim schema fields — do NOT push core
      past today. Defense line before travel/async work.
- [ ] 4.8 Unit test: mapping on captured raw fields; verifier flags a seeded mismatch.
- [ ] 4.9 **MCP seam:** make the Anthropic call in `parse_metadata` INJECTABLE,
      not hardwired. Web path calls the API internally; MCP path can expose raw
      metadata + target schema for the calling Claude to map/verify natively.
      Decide dual-mode vs single here, with real data. (See CLAUDE.md → "The MCP
      metadata seam".)

## Epic 5 — FastAPI (Day 4)
- [ ] 5.1 `POST /fetch` — accessions in, per-series summaries out.
- [ ] 5.2 `POST /harmonize` — accessions (+combat flag) -> merged matrix + metadata
      + shared-gene count + PCA coordinates (before/after).
- [ ] 5.3 `POST /upload` — manual-upload fallback (multipart) into the same flow.
- [ ] 5.4 `GET /download` — unified matrix + metadata as CSV/zip.
- [ ] 5.5 CORS for the front end; typed request/response models; thin — no logic.
- [ ] 5.6 Smoke test the endpoints locally (httpie/pytest).
- [ ] 5.7 **De-risk deploy early:** push a throwaway deploy (empty FastAPI + one
      endpoint) to the target host by END OF DAY 4 — surface OOM / cold-start /
      CORS now, so Day 6 is re-deploying something known-good.

## Epic 6 — MCP server (Day 4 — thin, secondary)
- [ ] 6.1 `mcp/server.py`: wrap `fetch_gse`, `harmonize`, `merge` (+`combat`) as
      MCP tools with clear descriptions. Official `mcp` SDK.
- [ ] 6.2 Verify it connects and the tools are callable from Claude locally.
- [ ] 6.3 One scripted Claude conversation that drives the pipeline (for the demo).
      Stop at "it works and you can see Claude calling the tools." Don't over-polish.
- [ ] 6.4 **De-risk MCP early:** by END OF DAY 4, one tool callable from Claude
      end-to-end (handshake/config proven), so Day 6 is re-running known-good, not
      discovering an MCP config problem at 3 AM.

## Epic 7 — Web front end (Day 5 — the star)
- [ ] 7.0 Read `docs/design-brief.md` first. Avoid the three AI-default looks.
- [ ] 7.1 Enter accessions -> results: matrix preview, shared-gene count,
      per-dataset breakdown, metadata table.
- [ ] 7.2 Manual-upload fallback UI; error copy = what went wrong + how to fix.
- [ ] 7.3 Download (CSV/zip) of unified matrix + metadata.
- [ ] 7.4 Hero = the thesis (two mismatched datasets, promise of one), not a bare
      accession field. Copy in the researcher's language.
- [ ] 7.5 Responsive to mobile, visible keyboard focus, reduced-motion respected.

## Epic 8 — Signature PCA (Day 5 — spend all boldness here)
- [ ] 8.1 Static version with a **FIXED PCA projection** (fit once on the final
      combined matrix; project each step into it — do NOT refit per step or points
      flip/rotate and it reads as noise). Add series -> project -> show; each
      dataset its own colour; ComBat toggle -> project -> show. Tells the story alone.
- [ ] 8.2 Legibility pass: legend, axis, one-glance readability on video.
- [ ] 8.3 (Only if time) animated transitions between states. Static-that-works
      beats animated-that-stutters on the recording.
- [ ] 8.4 (Optional cherry) one HGSOC marker masked by batch pre-ComBat, visible
      post-ComBat. One closing beat, not an analysis module.

## Epic 9 — Deploy & polish (Day 6)
- [ ] 9.1 Confirm the rules: is a live URL required, or is repo + video enough? If
      not required, recording locally is safer — don't let deploy threaten the demo.
- [ ] 9.2 (If deploying) front on Vercel, FastAPI on Render/Railway/Fly; public URL.
- [ ] 9.3 MCP confirmed working plugged into Claude for the recording.
- [ ] 9.4 Name the product; consistent look; clean happy path end to end.
- [ ] 9.5 Decide video order: web as product (0:00–1:30), MCP as "same thing inside
      Claude" (1:30–2:45).
- [ ] 9.6 **Full dry-run recording on Day 6** (not just Day 7): surfaces bugs and
      banks a fallback take before the hard deadline.

## Epic 10 — Record & submit (Day 7 — don't code)
- [ ] 10.1 Code freeze in the morning.
- [ ] 10.2 Record demo <=3 min; rehearse once; no crashes.
- [ ] 10.3 100–200 word description — sells problem + vision, tuned from the app.
- [ ] 10.4 README: what it is, how to run web, how to plug in MCP; clean code.
- [ ] 10.5 **Submit via the CV platform before 9:00 PM ET** — not 8:55.

---

## Crisis priorities (read if "both interfaces" starts slipping)
1. Core works (Epics 1–4). Without it there is nothing. Non-negotiable.
2. One smooth demo. If Sunday both aren't landing, pick web, finish it 100%, show
   MCP briefly/statically. Crashes are punished harder than narrow scope.
3. Claude in metadata parsing. That's the 25% even if MCP slips.
