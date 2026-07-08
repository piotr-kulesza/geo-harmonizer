import React, { useEffect, useMemo, useState } from 'react'
import Landscape from './scene/Landscape.jsx'
import { loadLandscape, recomputeHeight, interpretHeight } from './api.js'

// Build a {gsm: height} map for a baked height option straight from the payload.
function bakedHeights(payload, key) {
  const out = {}
  for (const s of payload.samples) out[s.id] = s.heights[key]
  return out
}

// Act 2 — the interactive 3D disease-risk landscape. Rendered inside App's shell,
// so it returns the canvas + panel directly (App owns the outer .app + switcher).
export default function LandscapeView() {
  const [payload, setPayload] = useState(null)
  const [loadError, setLoadError] = useState(null)
  const [active, setActive] = useState(null) // {key,label,kind,surface,heights}

  const [colorMode, setColorMode] = useState('height')
  const [controlMode, setControlMode] = useState('orbit')
  const [exaggeration, setExaggeration] = useState(1)

  const [geneValue, setGeneValue] = useState('')
  const [geneHint, setGeneHint] = useState(null)
  const [busy, setBusy] = useState(false)

  const [claudeQuery, setClaudeQuery] = useState('')
  const [claudePick, setClaudePick] = useState(null) // {label,kind}
  const [claudeHint, setClaudeHint] = useState(null)

  useEffect(() => {
    loadLandscape()
      .then((p) => {
        setPayload(p)
        const opt = p.height_options.find((o) => o.key === 'risk') || p.height_options[0]
        setActive({ ...opt, surface: p.surfaces[opt.key], heights: bakedHeights(p, opt.key) })
      })
      .catch((e) => setLoadError(e.message))
  }, [])

  const isStatic = payload?.source === 'static'
  const heldOut = useMemo(() => new Set(payload?.meta?.held_out ?? []), [payload])

  const sceneSamples = useMemo(() => {
    if (!payload || !active) return []
    return payload.samples.map((s) => {
      const v = active.heights[s.id]
      return { id: s.id, x: s.x, y: s.y, dataset: s.dataset, h: Number.isFinite(v) ? v : 0 }
    })
  }, [payload, active])

  // Switch among baked options — instant, no backend.
  function selectBaked(key) {
    const opt = payload.height_options.find((o) => o.key === key)
    setActive({ ...opt, surface: payload.surfaces[key], heights: bakedHeights(payload, key) })
    setClaudePick(null)
  }

  // Apply a recomputed/interpreted height sub-shape (surface + heights map).
  function applySub(sub) {
    setActive({
      key: sub.key,
      label: sub.label,
      kind: sub.kind,
      surface: sub.surface,
      heights: sub.heights,
    })
  }

  async function onGene(e) {
    e.preventDefault()
    const value = geneValue.trim()
    if (!value) return
    setBusy(true)
    setGeneHint(null)
    // A comma/space list -> signature; a single token -> gene.
    const tokens = value.split(/[\s,]+/).filter(Boolean)
    const req = tokens.length > 1 ? { kind: 'signature', value: tokens } : { kind: 'gene', value: tokens[0] }
    const res = await recomputeHeight(req, payload.source)
    setBusy(false)
    if (res.backendNeeded) return setGeneHint('Start the backend to compute new heights.')
    if (res.error) return setGeneHint(res.error)
    setClaudePick(null)
    applySub(res)
  }

  async function onClaude(e) {
    e.preventDefault()
    const q = claudeQuery.trim()
    if (!q) return
    setBusy(true)
    setClaudeHint(null)
    const res = await interpretHeight(q, payload.source)
    setBusy(false)
    if (res.backendNeeded) return setClaudeHint('Start the backend to ask Claude for a height.')
    if (res.error) return setClaudeHint(res.error)
    setClaudePick(res.selection)
    applySub(res)
  }

  if (loadError) {
    return (
      <div className="panel">
        <p className="title">Disease-risk landscape</p>
        <p className="hint error">Could not load data: {loadError}</p>
      </div>
    )
  }
  if (!payload || !active) {
    return (
      <div className="panel">
        <p className="spin">Loading landscape…</p>
      </div>
    )
  }

  const meta = payload.meta
  return (
    <>
      <div className="canvas-wrap">
        <Landscape
          surface={active.surface}
          samples={sceneSamples}
          heldOut={heldOut}
          colorMode={colorMode}
          controlMode={controlMode}
          exaggeration={exaggeration}
        />
      </div>

      <div className="panel">
        <p className="title">Disease-risk landscape</p>
        <p className="sub">
          Every tumour embedded on one map; height = the selected signal. Orbit, or
          switch to fly-through to move through the space.
        </p>

        <div className="badges">
          <span className={`badge ${payload.source}`}>
            {payload.source === 'live' ? '● LIVE backend' : '◐ STATIC snapshot'}
          </span>
          {meta.cindex != null && <span className="badge">C-index {meta.cindex.toFixed(2)}</span>}
          <span className="badge">{meta.n_samples} tumours</span>
          {meta.held_out?.length > 0 && <span className="badge">held out: {meta.held_out.join(', ')}</span>}
        </div>

        <div className="section">
          <h3>Height</h3>
          <label htmlFor="height">Baked signal (instant)</label>
          <select id="height" value={active.key} onChange={(e) => selectBaked(e.target.value)}>
            {payload.height_options.map((o) => (
              <option key={o.key} value={o.key}>
                {o.label} · {o.kind}
              </option>
            ))}
          </select>
        </div>

        <div className="section">
          <h3>Gene or signature</h3>
          <form onSubmit={onGene}>
            <input
              type="text"
              placeholder="e.g. EGFR — or MKI67, PCNA, TOP2A"
              value={geneValue}
              onChange={(e) => setGeneValue(e.target.value)}
              disabled={isStatic}
              title={isStatic ? 'Backend needed to recompute heights' : ''}
            />
            <div className="row" style={{ marginTop: 8 }}>
              <button type="submit" disabled={isStatic || busy}>
                {busy ? 'Computing…' : 'Set as height'}
              </button>
            </div>
          </form>
          {isStatic && <p className="hint">Static snapshot — start the backend to compute new heights.</p>}
          {geneHint && <p className="hint error">{geneHint}</p>}
        </div>

        <div className="section">
          <h3>Ask Claude for a height</h3>
          <div className="claude">
            <form onSubmit={onClaude}>
              <input
                type="text"
                placeholder='e.g. "proliferation", "immune activity", "EGFR signalling"'
                value={claudeQuery}
                onChange={(e) => setClaudeQuery(e.target.value)}
                disabled={isStatic}
                title={isStatic ? 'Backend needed to ask Claude' : ''}
              />
              <div className="row" style={{ marginTop: 8 }}>
                <button type="submit" className="primary" disabled={isStatic || busy}>
                  {busy ? 'Asking Claude…' : 'Ask Claude'}
                </button>
              </div>
            </form>
            {claudePick && (
              <p className="chose">
                Claude chose: <b>{claudePick.label}</b> ({claudePick.kind})
              </p>
            )}
            {isStatic && <p className="hint">Static snapshot — start the backend to ask Claude.</p>}
            {claudeHint && <p className="hint error">{claudeHint}</p>}
          </div>
        </div>

        <div className="section">
          <h3>View</h3>
          <label>Colour by</label>
          <div className="toggle">
            <button className={colorMode === 'height' ? 'on' : ''} onClick={() => setColorMode('height')}>
              Height
            </button>
            <button className={colorMode === 'dataset' ? 'on' : ''} onClick={() => setColorMode('dataset')}>
              Dataset
            </button>
          </div>
          <label style={{ marginTop: 10 }}>Camera</label>
          <div className="toggle">
            <button className={controlMode === 'orbit' ? 'on' : ''} onClick={() => setControlMode('orbit')}>
              Orbit
            </button>
            <button className={controlMode === 'fly' ? 'on' : ''} onClick={() => setControlMode('fly')}>
              Fly-through
            </button>
          </div>
          <label style={{ marginTop: 10 }}>Height exaggeration ×{exaggeration.toFixed(1)}</label>
          <input
            type="range"
            min="0.3"
            max="3"
            step="0.1"
            value={exaggeration}
            onChange={(e) => setExaggeration(parseFloat(e.target.value))}
            style={{ width: '100%' }}
          />
        </div>
      </div>
    </>
  )
}
