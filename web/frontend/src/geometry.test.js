import { describe, it, expect } from 'vitest'
import { surfaceToMesh } from './geometry.js'

// A 3x3 grid -> 4 cells -> 8 triangles -> 24 indices when fully finite.
function grid3(z) {
  return { gx: [0, 1, 2], gy: [0, 1, 2], z }
}

describe('surfaceToMesh', () => {
  it('builds a full mesh for an all-finite grid', () => {
    const z = [
      [0, 1, 2],
      [1, 2, 3],
      [2, 3, 4],
    ]
    const { positions, colors, indices } = surfaceToMesh(grid3(z))
    expect(positions.length).toBe(9 * 3) // one vertex per node
    expect(colors.length).toBe(9 * 3)
    expect(indices.length).toBe(4 * 6) // 4 quads * 2 tris * 3 verts
  })

  it('omits exactly the quads touching a null corner node', () => {
    // Corner node (0,0) is null -> it touches only 1 cell -> 1 quad dropped.
    const z = [
      [null, 1, 2],
      [1, 2, 3],
      [2, 3, 4],
    ]
    const { indices } = surfaceToMesh(grid3(z))
    expect(indices.length).toBe(3 * 6) // 3 remaining quads
  })

  it('drops all four quads when the centre node is null', () => {
    const z = [
      [0, 1, 2],
      [1, null, 3],
      [2, 3, 4],
    ]
    const { indices } = surfaceToMesh(grid3(z))
    expect(indices.length).toBe(0) // centre touches every cell
  })

  it('never references a null vertex from the index buffer', () => {
    const z = [
      [0, 1, 2],
      [1, null, 3],
      [2, 3, 4],
    ]
    const { indices } = surfaceToMesh(grid3(z))
    const nullIndex = 1 * 3 + 1 // node (row1,col1)
    expect(indices.includes(nullIndex)).toBe(false)
  })

  it('places height on Y and the map plane on X/Z', () => {
    const z = [
      [10, 11, 12],
      [11, 12, 13],
      [12, 13, 14],
    ]
    const { positions } = surfaceToMesh(grid3(z))
    // First vertex = node (gy=0, gx=0): (x=0, height=10, z=0)
    expect(positions.slice(0, 3)).toEqual([0, 10, 0])
    // Last vertex = node (gy=2, gx=2): (x=2, height=14, z=2)
    expect(positions.slice(-3)).toEqual([2, 14, 2])
  })

  it('is deterministic in vertices and colours', () => {
    const z = [
      [0, 1, 2],
      [1, 2, 3],
      [2, 3, 4],
    ]
    const a = surfaceToMesh(grid3(z))
    const b = surfaceToMesh(grid3(z))
    expect(a.positions).toEqual(b.positions)
    expect(a.colors).toEqual(b.colors)
    expect(a.indices).toEqual(b.indices)
  })
})
