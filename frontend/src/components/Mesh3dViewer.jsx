import { Suspense, useCallback, useEffect, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { Center, ContactShadows, Environment, OrbitControls, useGLTF } from '@react-three/drei'
import { AlertTriangle, Box, Loader2, RefreshCw } from 'lucide-react'
import { apiFetch } from '@/lib/auth'
import styles from './Mesh3dViewer.module.css'

const POLL_MS = 2500
const TERMINAL = new Set(['ready', 'failed', 'disabled'])

function GlbScene({ url }) {
  const { scene } = useGLTF(url)
  return (
    <Center>
      <primitive object={scene.clone()} />
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
            <ambientLight intensity={0.55} />
            <directionalLight position={[3, 4, 2]} intensity={1.1} />
            <Suspense fallback={null}>
              <GlbScene url={glbUrl} />
              <Environment preset="studio" />
              <ContactShadows opacity={0.35} scale={6} blur={2.5} far={2} />
            </Suspense>
            <OrbitControls makeDefault enablePan autoRotate autoRotateSpeed={0.6} />
          </Canvas>
          <div className={styles.canvasHint}>Drag to orbit · scroll to zoom · illustrative reconstruction</div>
        </div>
      )}
    </div>
  )
}
