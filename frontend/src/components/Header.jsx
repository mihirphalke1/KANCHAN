import { History, LogOut, ShieldAlert, ShieldCheck } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { getEvaluator, clearSession } from '@/lib/auth'
import { ADMIN_DASHBOARD_ROLES } from '@/components/RequireRole'
import styles from './Header.module.css'

export default function Header({ onHistoryClick }) {
  const navigate  = useNavigate()
  const evaluator = getEvaluator()
  const canSeeAdmin = evaluator && ADMIN_DASHBOARD_ROLES.includes(evaluator.role)

  const logout = () => {
    clearSession()
    navigate('/login', { replace: true })
  }

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
          {evaluator && (
            <span className={styles.evaluatorBadge} title={`${evaluator.evaluator_id} · ${evaluator.branch_id}`}>
              <ShieldCheck size={13} />
              {evaluator.name}
            </span>
          )}
          <button className={styles.historyBtn} onClick={onHistoryClick} aria-label="View case history">
            <History size={17} />
            <span>History</span>
          </button>
          {canSeeAdmin && (
            <button className={styles.historyBtn} onClick={() => navigate('/admin')} aria-label="Regional admin dashboard">
              <ShieldAlert size={17} />
              <span>Admin</span>
            </button>
          )}
          {evaluator && (
            <button className={styles.logoutBtn} onClick={logout} aria-label="Log out">
              <LogOut size={15} />
            </button>
          )}
        </div>
      </div>
    </header>
  )
}
