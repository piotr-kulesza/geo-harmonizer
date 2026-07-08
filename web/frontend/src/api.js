// Thin API client. LIVE when the FastAPI backend answers; otherwise falls back to
// the bundled static snapshot so the app renders with zero backend (demo-safe).
//
// API_BASE comes from a Vite env var (VITE_API_BASE); default is same-origin, so a
// one-origin deploy (uvicorn serving the built UI) needs no config.
const API_BASE = import.meta.env.VITE_API_BASE ?? ''

// Marker returned by mutating calls when we're running off the static snapshot.
export const BACKEND_NEEDED = { backendNeeded: true }

// loadLandscape(): try the live API, else the static file. Returns the payload
// augmented with `source: "live" | "static"`.
export async function loadLandscape() {
  try {
    const resp = await fetch(`${API_BASE}/api/landscape`, { headers: { Accept: 'application/json' } })
    if (resp.ok) {
      const payload = await resp.json()
      return { ...payload, source: 'live' }
    }
  } catch {
    // network/backend down — fall through to the static snapshot
  }
  const staticUrl = `${import.meta.env.BASE_URL}landscape_payload.json`
  const resp = await fetch(staticUrl)
  if (!resp.ok) throw new Error('could not load landscape (no backend and no static snapshot)')
  const payload = await resp.json()
  return { ...payload, source: 'static' }
}

// loadPca(): the fixed-projection progressive-PCA progression (Act 1). Tries the
// live API, else the bundled static snapshot. Mirrors loadLandscape's contract.
export async function loadPca() {
  try {
    const resp = await fetch(`${API_BASE}/api/pca`, { headers: { Accept: 'application/json' } })
    if (resp.ok) {
      const payload = await resp.json()
      return { ...payload, source: 'live' }
    }
  } catch {
    // backend down / no progression — fall through to the static snapshot
  }
  const staticUrl = `${import.meta.env.BASE_URL}pca_progression.json`
  const resp = await fetch(staticUrl)
  if (!resp.ok) throw new Error('could not load PCA progression (no backend and no static snapshot)')
  const payload = await resp.json()
  return { ...payload, source: 'static' }
}

// recomputeHeight({kind, value}): POST /api/height. Returns the height sub-shape on
// 200; on 422 returns { error: "<message>" }; in static mode returns BACKEND_NEEDED.
export async function recomputeHeight({ kind, value }, source) {
  if (source === 'static') return BACKEND_NEEDED
  return postHeight(`${API_BASE}/api/height`, { kind, value })
}

// interpretHeight(query): POST /api/height/interpret. Same return contract as
// recomputeHeight, plus a `selection` field ({kind,value,label}) when it succeeds.
export async function interpretHeight(query, source) {
  if (source === 'static') return BACKEND_NEEDED
  return postHeight(`${API_BASE}/api/height/interpret`, { query })
}

async function postHeight(url, body) {
  let resp
  try {
    resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
  } catch (e) {
    return { error: `backend unreachable: ${e.message}` }
  }
  if (resp.status === 422) {
    // Surface the helpful message (unknown gene / empty signature) — don't throw.
    let detail = 'invalid selection'
    try {
      const data = await resp.json()
      detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)
    } catch {
      /* ignore parse error, keep default */
    }
    return { error: detail }
  }
  if (!resp.ok) return { error: `request failed (${resp.status})` }
  return resp.json()
}
