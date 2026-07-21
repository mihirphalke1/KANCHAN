import { useEffect } from 'react'
import { X } from 'lucide-react'
import styles from './Lightbox.module.css'

/**
 * Fullscreen click-to-zoom viewer. Render once per screen; pass the src to
 * open, null to close. Closes on backdrop click, the X, or Escape.
 */
export default function Lightbox({ src, alt, onClose }) {
  useEffect(() => {
    if (!src) return
    const onKey = e => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    document.body.style.overflow = 'hidden'
    return () => {
      window.removeEventListener('keydown', onKey)
      document.body.style.overflow = ''
    }
  }, [src, onClose])

  if (!src) return null
  return (
    <div className={styles.overlay} onClick={onClose}>
      <button className={styles.close} onClick={onClose} aria-label="Close">
        <X size={22} />
      </button>
      <img
        src={src}
        alt={alt || ''}
        className={styles.img}
        onClick={e => e.stopPropagation()}
        onError={e => {
          e.target.replaceWith(Object.assign(document.createElement('div'), {
            textContent: 'Image unavailable',
            style: 'color:#e5e7eb;font:500 14px system-ui;padding:40px',
          }))
        }}
      />
    </div>
  )
}
