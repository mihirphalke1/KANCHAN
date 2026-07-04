import { useNavigate, useLocation } from 'react-router-dom'
import { Home, LayoutDashboard, Clock } from 'lucide-react'
import styles from './MobileTabBar.module.css'

const TABS = [
  { path: '/',          label: 'Home',      Icon: Home },
  { path: '/dashboard', label: 'Analyse',   Icon: LayoutDashboard },
  { path: '/history',   label: 'History',   Icon: Clock },
]

export default function MobileTabBar() {
  const navigate = useNavigate()
  const { pathname } = useLocation()

  return (
    <nav className={styles.bar} aria-label="Primary">
      {TABS.map(({ path, label, Icon }) => {
        const active = pathname === path
        return (
          <button
            key={path}
            className={`${styles.tab} ${active ? styles.active : ''}`}
            onClick={() => navigate(path)}
            aria-current={active ? 'page' : undefined}
          >
            <Icon size={17} strokeWidth={active ? 2.25 : 1.75} />
            <span>{label}</span>
          </button>
        )
      })}
    </nav>
  )
}
