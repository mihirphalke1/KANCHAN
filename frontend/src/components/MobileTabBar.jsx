import { useNavigate, useLocation } from 'react-router-dom'
import { Home, LayoutDashboard, Clock, ShieldAlert } from 'lucide-react'
import { getEvaluator } from '@/lib/auth'
import { ADMIN_DASHBOARD_ROLES } from '@/components/RequireRole'
import styles from './MobileTabBar.module.css'

const BASE_TABS = [
  { path: '/',          label: 'Home',      Icon: Home },
  { path: '/dashboard', label: 'Analyse',   Icon: LayoutDashboard },
  { path: '/history',   label: 'History',   Icon: Clock },
]
const ADMIN_TAB = { path: '/admin', label: 'Admin', Icon: ShieldAlert }

export default function MobileTabBar() {
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const evaluator = getEvaluator()
  const TABS = evaluator && ADMIN_DASHBOARD_ROLES.includes(evaluator.role)
    ? [...BASE_TABS, ADMIN_TAB] : BASE_TABS

  return (
    <nav className={`${styles.bar} ${TABS.length > 3 ? styles.compact : ''}`} aria-label="Primary">
      {TABS.map(({ path, label, Icon }) => {
        const active = pathname === path
        return (
          <button
            key={path}
            className={`${styles.tab} ${active ? styles.active : ''}`}
            onClick={() => navigate(path)}
            aria-current={active ? 'page' : undefined}
            aria-label={label}
          >
            <Icon size={17} strokeWidth={active ? 2.25 : 1.75} />
            <span>{label}</span>
          </button>
        )
      })}
    </nav>
  )
}
