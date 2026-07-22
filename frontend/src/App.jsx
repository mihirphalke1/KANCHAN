import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import HeroPage from './components/HeroPage'
import RequireAuth from './components/RequireAuth'
import RequireRole, { ADMIN_DASHBOARD_ROLES } from './components/RequireRole'
import LoginPage from './pages/LoginPage'
import DashboardPage from './pages/DashboardPage'
import HistoryPage from './pages/HistoryPage'
import AdminDashboardPage from './pages/AdminDashboardPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"          element={<HeroPage />} />
        <Route path="/login"     element={<LoginPage />} />
        <Route path="/dashboard" element={<RequireAuth><DashboardPage /></RequireAuth>} />
        <Route path="/history"   element={<RequireAuth><HistoryPage /></RequireAuth>} />
        <Route path="/admin"     element={<RequireRole roles={ADMIN_DASHBOARD_ROLES}><AdminDashboardPage /></RequireRole>} />
        <Route path="*"          element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
