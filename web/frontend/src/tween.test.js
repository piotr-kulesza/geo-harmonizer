import { describe, it, expect } from 'vitest'
import { clamp01, easeInOutCubic, lerp, lerpPoint, formatSilhouette } from './tween.js'

describe('easeInOutCubic', () => {
  it('is exact at the endpoints and midpoint', () => {
    expect(easeInOutCubic(0)).toBe(0)
    expect(easeInOutCubic(1)).toBe(1)
    expect(easeInOutCubic(0.5)).toBeCloseTo(0.5, 10)
  })
  it('clamps out-of-range and NaN', () => {
    expect(easeInOutCubic(-1)).toBe(0)
    expect(easeInOutCubic(2)).toBe(1)
    expect(easeInOutCubic(NaN)).toBe(0)
  })
  it('is monotonic across the unit interval', () => {
    let prev = -Infinity
    for (let i = 0; i <= 10; i++) {
      const v = easeInOutCubic(i / 10)
      expect(v).toBeGreaterThanOrEqual(prev)
      prev = v
    }
  })
})

describe('lerp / clamp01', () => {
  it('lerp hits endpoints', () => {
    expect(lerp(2, 8, 0)).toBe(2)
    expect(lerp(2, 8, 1)).toBe(8)
    expect(lerp(2, 8, 0.5)).toBe(5)
  })
  it('clamp01 bounds the range', () => {
    expect(clamp01(-3)).toBe(0)
    expect(clamp01(3)).toBe(1)
    expect(clamp01(0.4)).toBe(0.4)
  })
})

describe('lerpPoint', () => {
  it('returns the fixed endpoints at t=0 and t=1', () => {
    expect(lerpPoint([0, 0], [10, -4], 0)).toEqual([0, 0])
    expect(lerpPoint([0, 0], [10, -4], 1)).toEqual([10, -4])
  })
  it('a stationary point never moves', () => {
    const p = [3, 7]
    expect(lerpPoint(p, p, 0.37)).toEqual([3, 7])
  })
  it('falls back when an endpoint is missing (no jump to origin)', () => {
    expect(lerpPoint(null, [5, 6], 0.5)).toEqual([5, 6])
    expect(lerpPoint([5, 6], null, 0.5)).toEqual([5, 6])
  })
  it('is deterministic', () => {
    expect(lerpPoint([0, 0], [10, 10], 0.42)).toEqual(lerpPoint([0, 0], [10, 10], 0.42))
  })
})

describe('formatSilhouette', () => {
  it('formats numbers to two decimals', () => {
    expect(formatSilhouette(0.9123)).toBe('0.91')
    expect(formatSilhouette(-0.0302)).toBe('-0.03')
  })
  it('renders an em dash for null/NaN', () => {
    expect(formatSilhouette(null)).toBe('—')
    expect(formatSilhouette(undefined)).toBe('—')
    expect(formatSilhouette(NaN)).toBe('—')
  })
})
