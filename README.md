# geo-harmonizer

Harmonize GEO transcriptomic datasets: give it a handful of GEO series
accessions, get back a single comparable expression matrix on a shared gene set
and a unified, standardized metadata table — in minutes. Optional ComBat batch
correction. Built for a researcher running cross-dataset meta-analyses who
otherwise loses days getting mismatched data into a comparable state.

The same core is reachable two ways: a **web service** (primary) and an **MCP
server** that Claude / Claude Science drives (secondary).

## Layout

```
core/      pure pipeline logic — no web/MCP/UI knowledge (the iron rule)
  fetch.py       fetch_gse(accession) -> FetchResult          [Day 1 ✓]
  harmonize.py   map_probes(), to_log2()                      [Day 2]
  merge.py       merge({acc: matrix}) -> (matrix, batch)      [Day 2]
  combat.py      combat(merged, batch)                        [Day 2/3]
  metadata.py    parse_metadata() + verify_mapping()          [Day 3, Claude]
web/       React + FastAPI thin layer                         [Day 4-5]
mcp/       MCP server thin layer                              [Day 4]
tests/     pytest (offline — no network)
scripts/   smoke_fetch.py, prewarm_cache.py
docs/      series.md, design-brief.md
```

`core/` is interface-agnostic pure logic: functions take arguments and return
data. The web and MCP layers are both thin wrappers over it.

## Data conventions

- Expression matrices are **features (rows) × samples (columns)**; GSM ids are
  the sample columns — uniform whether data came from GEO or a manual upload.
- Metadata frames are **samples (rows) × fields (columns)**, indexed by GSM.
- A sample's batch label is its source accession.

## Develop

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run the offline test suite (no network, no GEOparse needed):

```bash
pytest -q
```

Smoke-test fetching against real GEO (network; not part of the test suite):

```bash
python scripts/smoke_fetch.py GSE9891 GSE26712 GSE26193
```

Pre-warm the cache so the demo never live-fetches on camera:

```bash
python scripts/prewarm_cache.py GSE9891 GSE26712 GSE26193
```

## License

MIT © 2026 Piotr Kulesza
