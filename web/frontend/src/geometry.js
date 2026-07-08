// Pure surface -> BufferGeometry arrays. No three.js import (plain arrays), so it
// unit-tests offline. The frontend feeds these into a THREE.BufferGeometry.
import { heightToColor, normalize } from './color.js'

function isFiniteNum(v) {
  return v !== null && v !== undefined && Number.isFinite(v)
}

// surfaceToMesh({gx, gy, z}) -> { positions, colors, indices }
//
// Grid layout matches the backend contract: `z` is row-major over gy (rows) x gx
// (cols), with `null` marking cells OUTSIDE the data hull. We emit a vertex per
// grid node (Y = height, X/Z = map plane) and a quad (two triangles) ONLY when all
// four of its corner heights are finite — so any cell touching a `null` is skipped
// and the mesh exists solely over the hull (no skirts into empty space).
//
// Vertex colours come from the shared magma colormap over the finite-z range.
// Deterministic: identical input -> identical arrays.
export function surfaceToMesh({ gx, gy, z }) {
  const nx = gx.length
  const ny = gy.length
  const positions = []
  const colors = []
  const indices = []

  // Colour scale over the finite heights only.
  const finite = []
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const v = z[j]?.[i]
      if (isFiniteNum(v)) finite.push(v)
    }
  }
  const { scale } = normalize(finite)

  // One vertex per node. Null nodes get height 0 and a dark colour; they are never
  // referenced by an index, so their exact values are cosmetic.
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const v = z[j]?.[i]
      const finiteV = isFiniteNum(v)
      positions.push(gx[i], finiteV ? v : 0, gy[j])
      const c = finiteV ? heightToColor(scale(v)) : { r: 0, g: 0, b: 0 }
      colors.push(c.r, c.g, c.b)
    }
  }

  const idx = (j, i) => j * nx + i

  // Emit quads whose four corners are all finite.
  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const a = z[j]?.[i]
      const b = z[j]?.[i + 1]
      const c = z[j + 1]?.[i]
      const d = z[j + 1]?.[i + 1]
      if (isFiniteNum(a) && isFiniteNum(b) && isFiniteNum(c) && isFiniteNum(d)) {
        const va = idx(j, i)
        const vb = idx(j, i + 1)
        const vc = idx(j + 1, i)
        const vd = idx(j + 1, i + 1)
        // Two triangles (winding is irrelevant — material is double-sided).
        indices.push(va, vc, vb, vb, vc, vd)
      }
    }
  }

  return { positions, colors, indices }
}
