# Disease-risk landscape — frontend

A small react-three-fiber client for the interactive 3D disease-risk landscape.
It's a thin view over the FastAPI backend (`web/api.py`); all science lives in
`core/`. Orbit or fly through the terrain, swap the height signal, and ask Claude
for a height in natural language.

## Develop (live backend)

Terminal 1 — the API (needs a precomputed cache; see
`scripts/build_landscape_cache.py`):

```bash
uvicorn web.api:app --reload         # serves /api/* on http://localhost:8000
```

Terminal 2 — the Vite dev server:

```bash
cd web/frontend
npm install
npm run dev                          # http://localhost:5173
```

Point the dev server at the API with a Vite env var (default is same-origin):

```bash
echo 'VITE_API_BASE=http://localhost:8000' > .env.local
```

## Demo (one origin, no CORS)

```bash
cd web/frontend && npm run build     # emits web/frontend/dist/
uvicorn web.api:app                  # mounts dist/ at / — API + UI on one origin
```

Open http://localhost:8000.

## Static / offline

With no backend reachable, the app renders from bundled snapshots and shows a
**STATIC** badge:

- **Landscape (Act 2):** `public/landscape_payload.json` (shape of `GET /api/landscape`).
  The gene/signature and Ask-Claude boxes are disabled with a hint.
- **Harmonize (Act 1):** `public/pca_progression.json` (shape of `GET /api/pca`).
  Reveal + the ComBat morph still animate off the snapshot.

Regenerate both from real data by running `scripts/build_landscape_cache.py`, then
copying `outputs/landscape_payload.json` over `public/landscape_payload.json` **and**
`outputs/pca_progression.json` over `public/pca_progression.json`.

## Two acts

A top switcher toggles **① Harmonize** (progressive PCA + ComBat morph) and
**② Landscape** (the 3D risk terrain); the choice persists across reloads.

## Test

```bash
npm test                             # vitest — pure helpers only, offline
```

`src/color.js` and `src/geometry.js` are pure (no three.js, no network) and carry
the unit tests; the 3D scene is exercised by eye.
