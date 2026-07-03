import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import HeroPage from './components/HeroPage'
import DashboardPage from './pages/DashboardPage'
import HistoryPage from './pages/HistoryPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"          element={<HeroPage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/history"   element={<HistoryPage />} />
        <Route path="*"          element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
