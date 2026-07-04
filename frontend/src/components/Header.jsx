import { History } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import styles from './Header.module.css'

export default function Header({ onHistoryClick }) {
  const navigate = useNavigate()
  return (
    <header className={styles.header}>
      <div className={styles.inner}>
        <button className={styles.brand} onClick={() => navigate('/')} title="Go to home page">
          <div className={styles.logoMark}>
            <img src="/logo.png" alt="KANCHAN-AI logo" />
          </div>
          <div className={styles.brandText}>
            <span className={styles.logoName}>KANCHAN<span className={styles.logoAI}>-AI</span></span>
            <span className={styles.tagline}>Spurious Gold Intelligence System</span>
          </div>
        </button>

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
