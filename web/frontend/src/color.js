// Pure, deterministic color + normalization helpers. No three.js, no DOM.
// Shared by the terrain mesh and the point cloud so height reads consistently.

// A small magma-like ramp (perceptually dark->bright, colour-blind friendly-ish).
// Control stops sampled from matplotlib's "magma", as {t, r, g, b} in [0,1].
const MAGMA_STOPS = [
  { t: 0.0, r: 0.001, g: 0.000, b: 0.014 },
  { t: 0.25, r: 0.231, g: 0.059, b: 0.439 },
  { t: 0.5, r: 0.549, g: 0.161, b: 0.506 },
  { t: 0.75, r: 0.871, g: 0.288, b: 0.408 },
  { t: 0.9, r: 0.988, g: 0.553, b: 0.349 },
  { t: 1.0, r: 0.987, g: 0.991, b: 0.75 },
]

function clamp01(t) {
  if (Number.isNaN(t)) return 0
  return t < 0 ? 0 : t > 1 ? 1 : t
}

// heightToColor(t): map t in [0,1] to {r,g,b} in [0,1] via linear interpolation
// between the magma stops. Deterministic and pure.
export function heightToColor(t) {
  const x = clamp01(t)
  let lo = MAGMA_STOPS[0]
  let hi = MAGMA_STOPS[MAGMA_STOPS.length - 1]
  for (let i = 0; i < MAGMA_STOPS.length - 1; i++) {
    if (x >= MAGMA_STOPS[i].t && x <= MAGMA_STOPS[i + 1].t) {
      lo = MAGMA_STOPS[i]
      hi = MAGMA_STOPS[i + 1]
      break
    }
  }
  const span = hi.t - lo.t || 1
  const f = (x - lo.t) / span
  return {
    r: lo.r + (hi.r - lo.r) * f,
    g: lo.g + (hi.g - lo.g) * f,
    b: lo.b + (hi.b - lo.b) * f,
  }
}

// "#rrggbb" form, handy for CSS swatches / point materials.
export function heightToHex(t) {
  const { r, g, b } = heightToColor(t)
  const h = (v) => Math.round(clamp01(v) * 255).toString(16).padStart(2, '0')
  return `#${h(r)}${h(g)}${h(b)}`
}

// normalize(values): compute [min,max] over the FINITE values and return a scaler
// that maps a value to [0,1]. Non-finite inputs are ignored. When all values are
// equal (or none are finite) the scaler returns 0.5 (a flat mid-tone), never NaN.
export function normalize(values) {
  let min = Infinity
  let max = -Infinity
  for (const v of values) {
    if (Number.isFinite(v)) {
      if (v < min) min = v
      if (v > max) max = v
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    min = 0
    max = 1
    return { min, max, scale: () => 0.5 }
  }
  const span = max - min
  const scale = (v) => {
    if (!Number.isFinite(v) || span === 0) return 0.5
    return clamp01((v - min) / span)
  }
  return { min, max, scale }
}
