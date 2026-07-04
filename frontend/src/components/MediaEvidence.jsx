import { useEffect, useRef, useState } from 'react'
import { Image as ImageIcon, Mic, Volume2, Play, Pause } from 'lucide-react'
import styles from './MediaEvidence.module.css'

// Build a URL that works in both dev (proxied) and prod (same origin)
function mediaUrl(path) {
  if (!path) return null
  // path = "data/cases/{id}/img_0.jpg" → "/cases/{id}/img_0.jpg"
  return '/' + path.replace(/^data\//, '')
}

// ── Waveform canvas ────────────────────────────────────────────────────
function WaveformCanvas({ src }) {
  const canvasRef  = useRef()
  const audioRef   = useRef()
  const animRef    = useRef()
  const [playing, setPlaying]   = useState(false)
  const [progress, setProgress] = useState(0)
  const [drawn, setDrawn]       = useState(false)
  const dataRef = useRef(null)   // amplitude data

  useEffect(() => {
    if (!src) return
    const ctx = new (window.AudioContext || window.webkitAudioContext)()
    fetch(src)
      .then(r => r.arrayBuffer())
      .then(buf => ctx.decodeAudioData(buf))
      .then(decoded => {
        const raw  = decoded.getChannelData(0)
        const cols = 400
        const hop  = Math.max(1, Math.floor(raw.length / cols))
        const env  = []
        for (let i = 0; i < cols; i++) {
          let peak = 0
          for (let j = 0; j < hop; j++) {
            const v = Math.abs(raw[i * hop + j] || 0)
            if (v > peak) peak = v
          }
          env.push(peak)
        }
        const max = Math.max(...env) || 1
        dataRef.current = env.map(v => v / max)
        drawWaveform(dataRef.current, 0)
        setDrawn(true)
        ctx.close()
      })
      .catch(() => {
        drawFlatWaveform()
        ctx.close()
      })
  }, [src])

  function drawWaveform(data, playPct) {
    const canvas = canvasRef.current
    if (!canvas) return
    const W   = canvas.width
    const H   = canvas.height
    const mid = H / 2
    const dpr = window.devicePixelRatio || 1
    canvas.width  = W
    canvas.height = H
    const c = canvas.getContext('2d')
    c.clearRect(0, 0, W, H)

    const bw = W / data.length
    const played = Math.floor(data.length * playPct)

    data.forEach((amp, i) => {
      const bh = Math.max(amp * mid * 0.88, 1)
      const x  = i * bw
      const isPast = i < played
      // colour: past = brand blue, future = grey, peak = red
      c.fillStyle = isPast
        ? (amp > 0.75 ? '#DC2626' : '#1E4E9C')
        : (amp > 0.75 ? '#FCA5A5' : '#D1D5DB')
      c.fillRect(x, mid - bh, bw * 0.72, bh * 2)
    })

    // Centre line
    c.strokeStyle = '#E5E7EB'
    c.lineWidth   = 0.5
    c.beginPath()
    c.moveTo(0, mid)
    c.lineTo(W, mid)
    c.stroke()
  }

  function drawFlatWaveform() {
    const canvas = canvasRef.current
    if (!canvas) return
    const W = canvas.width, H = canvas.height, mid = H / 2
    const c = canvas.getContext('2d')
    c.clearRect(0, 0, W, H)
    c.strokeStyle = '#D1D5DB'
    c.lineWidth   = 1
    c.setLineDash([6, 6])
    c.beginPath()
    c.moveTo(0, mid)
    c.lineTo(W, mid)
    c.stroke()
    c.setLineDash([])
    c.fillStyle    = '#9CA3AF'
    c.font         = '11px system-ui'
    c.textAlign    = 'center'
    c.fillText('Could not decode audio', W / 2, mid - 8)
  }

  function togglePlay() {
    const audio = audioRef.current
    if (!audio) return
    if (playing) {
      audio.pause()
      setPlaying(false)
      if (animRef.current) cancelAnimationFrame(animRef.current)
    } else {
      audio.play()
      setPlaying(true)
      tick()
    }
  }

  function tick() {
    const audio = audioRef.current
    if (!audio || !dataRef.current) return
    const pct = audio.duration ? audio.currentTime / audio.duration : 0
    setProgress(pct)
    drawWaveform(dataRef.current, pct)
    if (!audio.ended && !audio.paused) {
      animRef.current = requestAnimationFrame(tick)
    } else {
      setPlaying(false)
      setProgress(0)
      drawWaveform(dataRef.current, 0)
    }
  }

  return (
    <div className={styles.waveWrap}>
      <canvas ref={canvasRef} className={styles.waveCanvas} width={400} height={64} />
      {src && (
        <audio ref={audioRef} src={src} onEnded={() => { setPlaying(false); setProgress(0) }} />
      )}
      <div className={styles.waveFooter}>
        {src && drawn && (
          <button className={styles.playBtn} onClick={togglePlay} type="button">
            {playing ? <Pause size={12} /> : <Play size={12} />}
            {playing ? 'Pause' : 'Play recording'}
          </button>
        )}
        <span className={styles.waveLabel}>
          {src ? 'Acoustic ring fingerprint · Blue = normal · Red = anomalous'
                : 'No audio recording provided'}
        </span>
      </div>
    </div>
  )
}

// ── Main component ─────────────────────────────────────────────────────
export default function MediaEvidence({ caseData }) {
  const media   = caseData?.media || {}
  const caseId  = caseData?.case_id

  const images      = (media.images || []).filter(Boolean).slice(0, 4)
  const streakPath  = media.streak  || null
  const audioPath   = media.audio   || null

  const hasImages = images.length > 0 || streakPath
  const hasAudio  = Boolean(audioPath)

  if (!hasImages && !hasAudio) return null

  return (
    <div className={styles.card}>
      <h3 className={styles.title}>Physical Evidence</h3>

      {hasImages && (
        <div className={styles.section}>
          <div className={styles.sectionLabel}>
            <ImageIcon size={11} />
            Item photographs
            <span className={styles.count}>{images.length + (streakPath ? 1 : 0)}</span>
          </div>
          <div className={styles.photoGrid}>
            {images.map((p, i) => (
              <div key={i} className={styles.photoCell}>
                <img
                  src={mediaUrl(p)}
                  alt={`Photo ${i + 1}`}
                  className={styles.photo}
                  loading="lazy"
                  onError={e => { e.target.style.display = 'none' }}
                />
                <span className={styles.photoLabel}>Photo {i + 1}</span>
              </div>
            ))}
            {streakPath && (
              <div className={`${styles.photoCell} ${styles.photoCellStreak}`}>
                <img
                  src={mediaUrl(streakPath)}
                  alt="Touchstone streak"
                  className={styles.photo}
                  loading="lazy"
                  onError={e => { e.target.style.display = 'none' }}
                />
                <span className={styles.photoLabel}>Touchstone streak</span>
              </div>
            )}
          </div>
        </div>
      )}

      {(hasAudio || true) && (
        <div className={styles.section}>
          <div className={styles.sectionLabel}>
            <Mic size={11} />
            Acoustic ring test
            {hasAudio && <span className={styles.count}>waveform</span>}
          </div>
          <WaveformCanvas src={hasAudio ? mediaUrl(audioPath) : null} />
        </div>
      )}
    </div>
  )
}
