import { Suspense, useCallback, useEffect, useMemo, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { Center, ContactShadows, Environment, OrbitControls, useGLTF } from '@react-three/drei'
import { AlertTriangle, Box, Loader2, RefreshCw } from 'lucide-react'
import { apiFetch } from '@/lib/auth'
import styles from './Mesh3dViewer.module.css'

const POLL_MS = 2500
const TERMINAL = new Set(['ready', 'failed', 'disabled'])

const srgb = (rgb) => (Array.isArray(rgb) && rgb.length >= 3
  ? rgb.map((c) => Math.min(1, Math.max(0, c / 255)))
  : null)

/**
 * Renders the GLB and enforces the colour scheme on every material by node
 * name (metal vs stone). The backend bakes gold/gem PBR materials, but a fully
 * metallic surface can wash out to white under a bright studio environment and
 * some pipelines drop the baked factors — so we re-assert the sampled colours
 * and a low, always-gold metalness here. Colours come from status.color when
 * available (the exact tones sampled from the photo), else sane gold/gem.
 */
function GlbScene({ url, color }) {
  const { scene } = useGLTF(url)
  const obj = useMemo(() => {
    const root = scene.clone(true)
    // New models bake photo-sampled colour + photographic shading into the mesh
    // as per-vertex colours (COLOR_0) with correct metalness per material — we
    // trust those and only tune the environment. Older flat-material models
    // (no vertex colours) fall back to enforcing colour + metalness by node
    // name from status.color so they still render gold/gem, never washed white.
    const goldRGB = srgb(color?.gold_rgb) || [0.83, 0.66, 0.24]
    const stoneRGB = srgb(color?.stone_rgb) || [0.62, 0.05, 0.11]
    root.traverse((c) => {
      if (!c.isMesh || !c.material) return
      const mats = Array.isArray(c.material) ? c.material : [c.material]
      const hasVertexColour = !!c.geometry?.attributes?.color
      const tag = `${c.name || ''} ${c.parent?.name || ''} ${mats[0]?.name || ''}`.toLowerCase()
      const isStone = /stone|gem|ruby|sapphire|emerald|red|blue|green/.test(tag)
      mats.forEach((m) => {
        m.envMapIntensity = 0.55
        if (hasVertexColour) {
          m.vertexColors = true          // baked colour + shading — leave as-is
          m.needsUpdate = true
          return
        }
        // Legacy flat model: enforce colour + metalness so it isn't white.
        if (isStone) {
          m.color?.setRGB(...stoneRGB)
          m.metalness = 0.0
          m.roughness = 0.14
          if (m.emissive) m.emissive.setRGB(stoneRGB[0] * 0.14, stoneRGB[1] * 0.14, stoneRGB[2] * 0.14)
        } else {
          m.color?.setRGB(...goldRGB)
          m.metalness = 0.55
          m.roughness = 0.38
          if (m.emissive) m.emissive.setRGB(0, 0, 0)
        }
        m.needsUpdate = true
      })
    })
    return root
  }, [scene, color])
  return (
    <Center>
      <primitive object={obj} />
    </Center>
  )
}

/**
 * Async TRELLIS 3D viewer — polls /api/mesh3d/{caseId} until ready/failed,
 * then renders an interactive GLB. Visual-only; does not affect loan risk.
 *
 * When `visible` is false the component still polls (so the model is often
 * ready by the time the officer opens the 3D Model tab) but stays hidden.
 */
export default function Mesh3dViewer({ caseId, initial, visible = true }) {
  const [status, setStatus] = useState(initial || { status: 'pending' })
  const [retrying, setRetrying] = useState(false)

  const refresh = useCallback(async () => {
    if (!caseId) return
    try {
      const res = await apiFetch(`/api/mesh3d/${caseId}`)
      if (!res.ok) return
      const data = await res.json()
      setStatus(data)
    } catch {
      // keep last known status
    }
  }, [caseId])

  useEffect(() => {
    if (initial) setStatus(initial)
  }, [initial])

  useEffect(() => {
    if (!caseId) return undefined
    const st = status?.status
    if (TERMINAL.has(st)) return undefined

    refresh()
    const id = setInterval(refresh, POLL_MS)
    return () => clearInterval(id)
  }, [caseId, status?.status, refresh])

  const onRetry = async () => {
    if (!caseId || retrying) return
    setRetrying(true)
    try {
      const res = await apiFetch(`/api/mesh3d/${caseId}/retry`, { method: 'POST' })
      const data = await res.json().catch(() => ({}))
      if (res.ok) setStatus(data)
      else setStatus(s => ({
        ...s,
        status: 'failed',
        error: data.detail || 'Retry failed',
      }))
    } catch (e) {
      setStatus(s => ({ ...s, status: 'failed', error: e.message }))
    } finally {
      setRetrying(false)
    }
  }

  const st = status?.status || 'pending'
  const glbUrl = status?.glb_url
  const error = status?.error || status?.message

  // Always poll while non-terminal (even when the tab isn't open), but only
  // paint the viewer UI when `visible` is true.
  if (!visible) return null

  return (
    <div className={styles.wrap}>
      {(st === 'pending' || st === 'generating') && (
        <div className={styles.stateBox}>
          <Loader2 size={22} className={styles.spin} />
          <div className={styles.stateText}>
            <strong>Generating 3D model…</strong>
            <span>
              TRELLIS runs on Hugging Face in the background. A cold Space or
              queue can take a few minutes — all other analysis results stay available.
            </span>
            {status?.message && <span className={styles.hint}>{status.message}</span>}
          </div>
        </div>
      )}

      {st === 'disabled' && (
        <div className={styles.stateBox}>
          <Box size={22} />
          <div className={styles.stateText}>
            <strong>3D generation unavailable</strong>
            <span>{error || 'Set HF_TOKEN and USE_MESH3D=1 on the server to enable TRELLIS.'}</span>
          </div>
        </div>
      )}

      {st === 'failed' && (
        <div className={styles.stateBox}>
          <AlertTriangle size={22} className={styles.warn} />
          <div className={styles.stateText}>
            <strong>3D generation failed</strong>
            <span>{error || 'The TRELLIS Space could not produce a model.'}</span>
            <button type="button" className={styles.retryBtn} onClick={onRetry} disabled={retrying}>
              {retrying ? <Loader2 size={13} className={styles.spin} /> : <RefreshCw size={13} />}
              Retry 3D generation
            </button>
          </div>
        </div>
      )}

      {st === 'ready' && glbUrl && (
        <div className={styles.canvasWrap}>
          <Canvas camera={{ position: [0, 0.4, 2.2], fov: 40 }} dpr={[1, 2]}>
            <color attach="background" args={['#0B1220']} />
            <ambientLight intensity={0.6} />
            <directionalLight position={[3, 4, 2]} intensity={1.2} />
            <directionalLight position={[-3, 1, -2]} intensity={0.5} />
            <Suspense fallback={null}>
              {/* Cache-bust on updated_at so a model that was re-coloured
                  (white → gold+gem, or a retry) never renders a stale copy. */}
              <GlbScene
                url={status?.updated_at ? `${glbUrl}?v=${encodeURIComponent(status.updated_at)}` : glbUrl}
                color={status?.color}
              />
              <Environment preset="studio" />
              <ContactShadows opacity={0.35} scale={6} blur={2.5} far={2} />
            </Suspense>
            <OrbitControls makeDefault enablePan autoRotate autoRotateSpeed={0.6} />
          </Canvas>
          <div className={styles.canvasHint}>Drag to orbit · scroll to zoom · colour-matched reconstruction</div>
        </div>
      )}
    </div>
  )
}
