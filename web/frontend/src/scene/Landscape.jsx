import React, { useMemo, useRef, useLayoutEffect } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, FlyControls } from '@react-three/drei'
import * as THREE from 'three'
import { surfaceToMesh } from '../geometry.js'
import { heightToColor, normalize } from '../color.js'
import { datasetHex } from '../palette.js'

function datasetColor(name, order) {
  return new THREE.Color(datasetHex(name, order))
}

function extent(arr) {
  let lo = Infinity
  let hi = -Infinity
  for (const v of arr) {
    if (Number.isFinite(v)) {
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
  }
  return Number.isFinite(lo) ? [lo, hi] : [0, 1]
}

// The scene contents (inside <Canvas>). Everything is centred on the origin so the
// camera and controls have a stable target regardless of the map's coordinates.
function Scene({ surface, samples, heldOut, colorMode, controlMode, exaggeration }) {
  const [gxLo, gxHi] = extent(surface.gx)
  const [gyLo, gyHi] = extent(surface.gy)
  const cx = (gxLo + gxHi) / 2
  const cy = (gyLo + gyHi) / 2
  const mapExtent = Math.max(gxHi - gxLo, gyHi - gyLo) || 1

  // Vertical scale: make the height amplitude a readable fraction of the map size,
  // then apply the user's exaggeration. Heights themselves (risk, expression) can
  // be tiny or large; this keeps the terrain legible either way.
  const zScale = useMemo(() => {
    const flat = []
    for (const row of surface.z) for (const v of row) if (Number.isFinite(v)) flat.push(v)
    const [lo, hi] = extent(flat)
    const span = hi - lo || 1
    return ((mapExtent * 0.35) / span) * exaggeration
  }, [surface, mapExtent, exaggeration])

  // Terrain geometry (hull-clipped mesh). Y positions scaled for visibility.
  const geometry = useMemo(() => {
    const { positions, colors, indices } = surfaceToMesh(surface)
    const g = new THREE.BufferGeometry()
    const pos = Float32Array.from(positions)
    for (let i = 1; i < pos.length; i += 3) pos[i] *= zScale
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    g.setAttribute('color', new THREE.BufferAttribute(Float32Array.from(colors), 3))
    g.setIndex(indices)
    g.computeVertexNormals()
    return g
  }, [surface, zScale])

  const pointSize = mapExtent * 0.013
  const datasetOrder = useMemo(
    () => Array.from(new Set(samples.map((s) => s.dataset))).sort(),
    [samples],
  )

  // Per-sample colour: by height (shared magma) or by dataset (categorical).
  const heightScale = useMemo(() => normalize(samples.map((s) => s.h)).scale, [samples])

  const pointsRef = useRef()
  useLayoutEffect(() => {
    const mesh = pointsRef.current
    if (!mesh) return
    const dummy = new THREE.Object3D()
    samples.forEach((s, i) => {
      dummy.position.set(s.x - cx, s.h * zScale, s.y - cy)
      dummy.scale.setScalar(pointSize)
      dummy.updateMatrix()
      mesh.setMatrixAt(i, dummy.matrix)
      const col =
        colorMode === 'dataset'
          ? datasetColor(s.dataset, datasetOrder)
          : (() => {
              const c = heightToColor(heightScale(s.h))
              return new THREE.Color(c.r, c.g, c.b)
            })()
      mesh.setColorAt(i, col)
    })
    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  }, [samples, colorMode, zScale, pointSize, cx, cy, datasetOrder, heightScale])

  // Held-out samples get a subtle ring so "these landed on the terrain without
  // supervising it" is visible.
  const heldSamples = useMemo(() => samples.filter((s) => heldOut.has(s.dataset)), [samples, heldOut])
  const ringsRef = useRef()
  useLayoutEffect(() => {
    const mesh = ringsRef.current
    if (!mesh) return
    const dummy = new THREE.Object3D()
    heldSamples.forEach((s, i) => {
      dummy.position.set(s.x - cx, s.h * zScale, s.y - cy)
      dummy.rotation.set(Math.PI / 2, 0, 0) // lay flat around the point
      dummy.scale.setScalar(pointSize * 2.1)
      dummy.updateMatrix()
      mesh.setMatrixAt(i, dummy.matrix)
    })
    mesh.instanceMatrix.needsUpdate = true
  }, [heldSamples, zScale, pointSize, cx, cy])

  return (
    <group>
      <ambientLight intensity={0.55} />
      <directionalLight position={[mapExtent, mapExtent * 1.5, mapExtent]} intensity={1.1} />
      <directionalLight position={[-mapExtent, mapExtent, -mapExtent]} intensity={0.3} />

      <gridHelper args={[mapExtent * 2, 20, '#26263a', '#181824']} position={[0, 0, 0]} />

      {/* Terrain */}
      <mesh geometry={geometry}>
        <meshStandardMaterial
          vertexColors
          side={THREE.DoubleSide}
          transparent
          opacity={0.82}
          roughness={0.85}
          metalness={0.05}
        />
      </mesh>

      {/* Tumours */}
      <instancedMesh
        key={samples.length}
        ref={pointsRef}
        args={[undefined, undefined, samples.length]}
      >
        <sphereGeometry args={[1, 12, 12]} />
        <meshStandardMaterial vertexColors roughness={0.4} metalness={0.1} />
      </instancedMesh>

      {/* Held-out markers */}
      {heldSamples.length > 0 && (
        <instancedMesh
          key={`held-${heldSamples.length}`}
          ref={ringsRef}
          args={[undefined, undefined, heldSamples.length]}
        >
          <torusGeometry args={[1, 0.14, 8, 24]} />
          <meshBasicMaterial color="#ffffff" transparent opacity={0.7} />
        </instancedMesh>
      )}

      {controlMode === 'fly' ? (
        <FlyControls movementSpeed={mapExtent * 0.6} rollSpeed={0.4} dragToLook />
      ) : (
        <OrbitControls makeDefault enablePan enableZoom target={[0, 0, 0]} />
      )}
    </group>
  )
}

export default function Landscape(props) {
  const { surface } = props
  const [gxLo, gxHi] = extent(surface.gx)
  const [gyLo, gyHi] = extent(surface.gy)
  const mapExtent = Math.max(gxHi - gxLo, gyHi - gyLo) || 1
  const camStart = [mapExtent * 0.05, mapExtent * 0.75, mapExtent * 1.15]

  return (
    <Canvas camera={{ position: camStart, fov: 50, near: 0.01, far: mapExtent * 40 }}>
      <color attach="background" args={['#0b0b12']} />
      <Scene {...props} />
    </Canvas>
  )
}
