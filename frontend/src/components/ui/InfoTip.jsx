import { useRef, useState } from 'react'
import { Info } from 'lucide-react'
import styles from './InfoTip.module.css'

const TIP_WIDTH = 230
const MARGIN = 8

/**
 * Small (i) icon that reveals officer guidance on hover, focus, or tap.
 * The tooltip is rendered with viewport-fixed positioning and clamped to the
 * screen edges, so it can never be clipped by a scrolling column or an
 * adjacent panel. Usage: <InfoTip text="Tare the scale with the water cup on it…" />
 */
export default function InfoTip({ text }) {
  const iconRef = useRef(null)
  const [pos, setPos] = useState(null)   // null = closed

  const openTip = () => {
    const r = iconRef.current?.getBoundingClientRect()
    if (!r) return
    const left = Math.min(
      Math.max(r.left + r.width / 2 - TIP_WIDTH / 2, MARGIN),
      window.innerWidth - TIP_WIDTH - MARGIN,
    )
    setPos({ left, bottom: window.innerHeight - r.top + 7, arrowX: r.left + r.width / 2 - left })
  }

  if (!text) return null
  return (
    <span
      className={styles.wrap}
      onMouseEnter={openTip}
      onMouseLeave={() => setPos(null)}
    >
      <button
        ref={iconRef}
        type="button"
        className={styles.icon}
        aria-label="More information"
        onClick={e => { e.preventDefault(); pos ? setPos(null) : openTip() }}
        onBlur={() => setPos(null)}
      >
        <Info size={12} />
      </button>
      {pos && (
        <span
          role="tooltip"
          className={styles.tip}
          style={{ left: pos.left, bottom: pos.bottom, width: TIP_WIDTH }}
        >
          {text}
          <span className={styles.arrow} style={{ left: pos.arrowX }} />
        </span>
      )}
    </span>
  )
}
