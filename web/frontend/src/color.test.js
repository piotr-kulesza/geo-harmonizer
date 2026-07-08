import { describe, it, expect } from 'vitest'
import { heightToColor, heightToHex, normalize } from './color.js'

describe('heightToColor', () => {
  it('is deterministic', () => {
    expect(heightToColor(0.42)).toEqual(heightToColor(0.42))
  })

  it('hits the ramp endpoints (dark low, bright high)', () => {
    const lo = heightToColor(0)
    const hi = heightToColor(1)
    expect(lo).toEqual({ r: 0.001, g: 0.0, b: 0.014 })
    expect(hi).toEqual({ r: 0.987, g: 0.991, b: 0.75 })
    // brightness increases across the ramp
    const sum = (c) => c.r + c.g + c.b
    expect(sum(hi)).toBeGreaterThan(sum(lo))
  })

  it('clamps out-of-range and NaN inputs into [0,1]', () => {
    expect(heightToColor(-5)).toEqual(heightToColor(0))
    expect(heightToColor(9)).toEqual(heightToColor(1))
    expect(heightToColor(NaN)).toEqual(heightToColor(0))
  })

  it('returns channels within [0,1]', () => {
    for (const t of [0, 0.1, 0.33, 0.5, 0.77, 1]) {
      const c = heightToColor(t)
      for (const ch of [c.r, c.g, c.b]) {
        expect(ch).toBeGreaterThanOrEqual(0)
        expect(ch).toBeLessThanOrEqual(1)
      }
    }
  })

  it('heightToHex yields a 7-char hex string', () => {
    expect(heightToHex(0.5)).toMatch(/^#[0-9a-f]{6}$/)
  })
})

describe('normalize', () => {
  it('reports [min,max] over finite values and scales into [0,1]', () => {
    const { min, max, scale } = normalize([2, 4, 6, NaN, null, Infinity])
    expect([min, max]).toEqual([2, 6])
    expect(scale(2)).toBe(0)
    expect(scale(6)).toBe(1)
    expect(scale(4)).toBeCloseTo(0.5)
  })

  it('returns a flat mid-tone when all values are equal', () => {
    const { scale } = normalize([3, 3, 3])
    expect(scale(3)).toBe(0.5)
  })

  it('handles an all-nonfinite / empty input without NaN', () => {
    const { scale } = normalize([NaN, Infinity, null])
    expect(scale(1)).toBe(0.5)
    expect(normalize([]).scale(0)).toBe(0.5)
  })
})
