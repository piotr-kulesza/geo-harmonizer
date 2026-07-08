import React, { useEffect, useMemo, useRef, useState } from 'react'
import { loadPca } from '../api.js'
import { datasetHex } from '../palette.js'
import { easeInOutCubic, lerpPoint, lerp, formatSilhouette } from '../tween.js'

// Act 1 — the harmonization hero. Fixed-projection progressive PCA: reveal series
// one by one (clouds separate BY BATCH), then toggle ComBat (clouds merge BY
// BIOLOGY). All positions are precomputed in a fixed basis by the backend; here we
// only interpolate between them, so points GLIDE and never teleport/flip.

const REVEAL_MS = 900 // per-series reveal cadence when playing
const COMBAT_MS = 950 // raw <-> ComBat morph duration
const APPEAR_MS = 600 // fade+grow of newly revealed points

function domainOf(maps) {
  let xmin = Infinity
  let xmax = -Infinity
  let ymin = Infinity
  let ymax = -Infinity
  for (const m of maps) {
    if (!m) continue
    for (const k in m) {
      const [x, y] = m[k]
      if (x < xmin) xmin = x
      if (x > xmax) xmax = x
      if (y < ymin) ymin = y
      if (y > ymax) ymax = y
    }
  }
  if (!Number.isFinite(xmin)) return { xmin: -1, xmax: 1, ymin: -1, ymax: 1 }
  // pad 6%
  const px = (xmax - xmin || 1) * 0.06
  const py = (ymax - ymin || 1) * 0.06
  return { xmin: xmin - px, xmax: xmax + px, ymin: ymin - py, ymax: ymax + py }
}

export default function Harmonize() {
  const [data, setData] = useState(null)
  const [err, setErr] = useState(null)
  const [step, setStep] = useState(1)
  const [combatOn, setCombatOn] = useState(false)
  const [playing, setPlaying] = useState(false)

  const canvasRef = useRef(null)
  const wrapRef = useRef(null)
  const silRef = useRef(null)
  const anim = useRef({ combatT: 0, combatTarget: 0, alphas: new Map(), targets: new Map(), last: 0, size: [1, 1] })
  const viz = useRef(null)
  const dataRef = useRef(null)

  useEffect(() => {
    loadPca()
      .then((p) => {
        setData(p)
        dataRef.current = p
        const lastStep = p.steps[p.steps.length - 1]
        const rawMap = lastStep.raw.coords
        const combatMap = lastStep.combat ? lastStep.combat.coords : null
        viz.current = {
          rawMap,
          combatMap,
          gsms: Object.keys(p.batch),
          domain: domainOf([rawMap, combatMap]),
        }
        for (const g of Object.keys(p.batch)) anim.current.alphas.set(g, 0)
      })
      .catch((e) => setErr(e.message))
  }, [])

  const hasCombat = !!data?.steps?.[0]?.combat
  const nSteps = data?.steps?.length ?? 0

  // Reveal targets: points in the current step ease to alpha 1, the rest to 0.
  useEffect(() => {
    if (!data) return
    const included = data.steps[step - 1].raw.coords
    for (const g of Object.keys(data.batch)) anim.current.targets.set(g, g in included ? 1 : 0)
  }, [data, step])

  // ComBat target (0 = raw, 1 = corrected).
  useEffect(() => {
    anim.current.combatTarget = combatOn && hasCombat ? 1 : 0
  }, [combatOn, hasCombat])

  // Auto-play the reveal.
  useEffect(() => {
    if (!playing || !data) return
    if (step >= nSteps) {
      setPlaying(false)
      return
    }
    const id = setTimeout(() => setStep((s) => Math.min(nSteps, s + 1)), REVEAL_MS)
    return () => clearTimeout(id)
  }, [playing, step, nSteps, data])

  // The render loop — pure interpolation between fixed positions.
  useEffect(() => {
    if (!data) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    let raf = 0

    const resize = () => {
      const wrap = wrapRef.current
      const dpr = window.devicePixelRatio || 1
      const w = wrap.clientWidth
      const h = wrap.clientHeight
      canvas.width = Math.max(1, Math.floor(w * dpr))
      canvas.height = Math.max(1, Math.floor(h * dpr))
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      anim.current.size = [w, h]
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(wrapRef.current)

    const draw = (now) => {
      const a = anim.current
      const dt = a.last ? Math.min(0.05, (now - a.last) / 1000) : 0
      a.last = now

      // Ease combatT toward its target at a fixed rate; ease per-point alphas.
      const cd = 1000 / COMBAT_MS
      if (a.combatT < a.combatTarget) a.combatT = Math.min(a.combatTarget, a.combatT + dt * cd)
      else if (a.combatT > a.combatTarget) a.combatT = Math.max(a.combatTarget, a.combatT - dt * cd)
      const ad = 1000 / APPEAR_MS
      for (const [g, cur] of a.alphas) {
        const t = a.targets.get(g) ?? 0
        if (cur < t) a.alphas.set(g, Math.min(t, cur + dt * ad))
        else if (cur > t) a.alphas.set(g, Math.max(t, cur - dt * ad))
      }

      const d = dataRef.current
      const v = viz.current
      const [w, h] = a.size
      const pad = 64
      const { xmin, xmax, ymin, ymax } = v.domain
      const sx = (x) => pad + ((x - xmin) / (xmax - xmin || 1)) * (w - 2 * pad)
      const sy = (y) => h - pad - ((y - ymin) / (ymax - ymin || 1)) * (h - 2 * pad)

      ctx.clearRect(0, 0, w, h)

      // Faint zero axes.
      ctx.strokeStyle = 'rgba(255,255,255,0.06)'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(sx(0), pad * 0.4)
      ctx.lineTo(sx(0), h - pad * 0.4)
      ctx.moveTo(pad * 0.4, sy(0))
      ctx.lineTo(w - pad * 0.4, sy(0))
      ctx.stroke()

      // Points.
      const combatE = easeInOutCubic(a.combatT)
      for (const g of v.gsms) {
        const alpha = a.alphas.get(g) ?? 0
        if (alpha < 0.01) continue
        const raw = v.rawMap[g]
        const corr = v.combatMap ? v.combatMap[g] : null
        const p = corr ? lerpPoint(raw, corr, a.combatT) : raw
        const e = easeInOutCubic(alpha)
        ctx.globalAlpha = e
        ctx.fillStyle = datasetHex(d.batch[g], d.order)
        ctx.beginPath()
        ctx.arc(sx(p[0]), sy(p[1]), 3.2 + 2.2 * e, 0, Math.PI * 2)
        ctx.fill()
      }
      ctx.globalAlpha = 1

      // Axis labels with explained-variance %.
      const ev = d.axes.explained_variance
      ctx.fillStyle = 'rgba(232,232,240,0.75)'
      ctx.font = '13px ui-sans-serif, system-ui, sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(`PC1 · ${(ev[0] * 100).toFixed(1)}% variance`, w / 2, h - 20)
      ctx.save()
      ctx.translate(20, h / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.fillText(`PC2 · ${(ev[1] * 100).toFixed(1)}% variance`, 0, 0)
      ctx.restore()

      // Animate the silhouette readout (no React re-render per frame).
      if (silRef.current) {
        const stp = d.steps[Math.min(step, d.steps.length) - 1]
        const rawSil = stp.raw.silhouette
        const combatSil = stp.combat ? stp.combat.silhouette : null
        let shown = rawSil
        if (rawSil != null && combatSil != null) shown = lerp(rawSil, combatSil, combatE)
        silRef.current.textContent = formatSilhouette(shown)
        silRef.current.style.color = shown == null ? '#9aa' : shown > 0.3 ? '#e0b64f' : shown > 0.1 ? '#7ed08f' : '#7ee0a0'
      }

      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  }, [data, step])

  const currentStep = data?.steps?.[step - 1]
  const rawSil = currentStep?.raw?.silhouette
  const combatSil = currentStep?.combat?.silhouette

  if (err) {
    return (
      <div className="panel">
        <p className="title">Harmonize</p>
        <p className="hint error">Could not load PCA progression: {err}</p>
      </div>
    )
  }
  if (!data) {
    return (
      <div className="panel">
        <p className="spin">Loading harmonization…</p>
      </div>
    )
  }

  return (
    <>
      <div className="canvas-wrap" ref={wrapRef}>
        <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
      </div>

      <div className="panel">
        <p className="title">Harmonize · progressive PCA</p>
        <p className="sub">
          Add datasets one at a time — they land in separate clouds by batch. Turn on
          ComBat and the clouds collapse together by biology.
        </p>

        <div className="badges">
          <span className={`badge ${data.source}`}>
            {data.source === 'live' ? '● LIVE backend' : '◐ STATIC snapshot'}
          </span>
          <span className="badge">{Object.keys(data.batch).length} samples</span>
          <span className="badge">{data.order.length} datasets</span>
        </div>

        <div className="readout">
          <div className="readout-num" ref={silRef}>
            —
          </div>
          <div className="readout-label">batch separation (silhouette)</div>
          {hasCombat && rawSil != null && combatSil != null && (
            <div className="readout-ref">
              raw {formatSilhouette(rawSil)} → ComBat {formatSilhouette(combatSil)}
            </div>
          )}
        </div>

        <p className={`caption ${combatOn ? 'good' : ''}`}>
          {combatOn
            ? 'ComBat on — samples cluster by biology'
            : `${step} dataset${step > 1 ? 's' : ''}, uncorrected — samples cluster by batch`}
        </p>

        <div className="section">
          <h3>Reveal datasets</h3>
          <div className="row">
            <button onClick={() => setPlaying((p) => !p)} disabled={step >= nSteps && !playing}>
              {playing ? 'Pause' : step >= nSteps ? 'Revealed' : 'Play ▶'}
            </button>
            <button
              onClick={() => {
                setPlaying(false)
                setStep(1)
              }}
            >
              Reset
            </button>
          </div>
          <input
            type="range"
            min="1"
            max={nSteps}
            step="1"
            value={step}
            onChange={(e) => {
              setPlaying(false)
              setStep(parseInt(e.target.value, 10))
            }}
            style={{ width: '100%', marginTop: 10 }}
          />
          <div className="legend">
            {data.order.map((acc, i) => (
              <span key={acc} className="legend-item" style={{ opacity: i < step ? 1 : 0.3 }}>
                <span className="swatch" style={{ background: datasetHex(acc, data.order) }} />
                {acc}
              </span>
            ))}
          </div>
        </div>

        <div className="section">
          <h3>Correction</h3>
          <div className="toggle">
            <button className={!combatOn ? 'on' : ''} onClick={() => setCombatOn(false)}>
              Raw
            </button>
            <button
              className={combatOn ? 'on' : ''}
              onClick={() => setCombatOn(true)}
              disabled={!hasCombat}
              title={hasCombat ? '' : 'No ComBat state in this snapshot'}
            >
              ComBat
            </button>
          </div>
          {!hasCombat && <p className="hint">This snapshot has no ComBat state.</p>}
        </div>
      </div>
    </>
  )
}
