// Pure animation helpers — no DOM, no three.js. Unit-tested. The harmonize view's
// motion (points gliding, ComBat morph, silhouette counting down) is all built
// from these, so smoothness is deterministic and testable.

export function clamp01(t) {
  if (Number.isNaN(t)) return 0
  return t < 0 ? 0 : t > 1 ? 1 : t
}

// Smooth ease for glides/morphs: flat-fast-flat, exact at the endpoints.
export function easeInOutCubic(t) {
  const x = clamp01(t)
  return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2
}

// Linear interpolation.
export function lerp(a, b, t) {
  return a + (b - a) * t
}

// Interpolate a 2D point [x,y] between two fixed positions (the fixed-projection
// coords from the backend), easing t. `from`/`to` may be the same array (a point
// that doesn't move); missing endpoints fall back to the other so nothing jumps.
export function lerpPoint(from, to, t) {
  const a = from || to
  const b = to || from
  if (!a || !b) return a || b || [0, 0]
  const e = easeInOutCubic(t)
  return [lerp(a[0], b[0], e), lerp(a[1], b[1], e)]
}

// Format a batch-separation (silhouette) value for the readout. `null`/NaN -> "—".
export function formatSilhouette(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  return v.toFixed(2)
}
