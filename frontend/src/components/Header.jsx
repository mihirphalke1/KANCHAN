import { History, Shield } from 'lucide-react'
import styles from './Header.module.css'

export default function Header({ onHistoryClick }) {
  return (
    <header className={styles.header}>
      <div className={styles.inner}>
        <div className={styles.brand}>
          <div className={styles.logoMark}>
            <Shield size={22} strokeWidth={2.5} />
          </div>
          <div className={styles.brandText}>
            <span className={styles.logoName}>KANCHAN<span className={styles.logoAI}>-AI</span></span>
            <span className={styles.tagline}>Spurious Gold Intelligence System</span>
          </div>
        </div>

        <div className={styles.right}>
          <span className={styles.badge}>
            <span className={styles.badgeDot} />
            Canara Bank · Gold Loan Division
          </span>
          <button className={styles.historyBtn} onClick={onHistoryClick} aria-label="View case history">
            <History size={17} />
            <span>History</span>
          </button>
        </div>
      </div>
    </header>
  )
}
