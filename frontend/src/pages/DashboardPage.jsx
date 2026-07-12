import { useRef, useState } from 'react'
import Header from '@/components/Header'
import AnalysisForm from '@/components/AnalysisForm'
import ResultsPanel from '@/components/ResultsPanel'
import HistoryDrawer from '@/components/HistoryDrawer'
import MobileTabBar from '@/components/MobileTabBar'
import styles from './DashboardPage.module.css'

export default function DashboardPage() {
  const [result, setResult]           = useState(null)
  const [localMedia, setLocalMedia]   = useState(null)
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState(null)
  const [showHistory, setShowHistory] = useState(false)
  const resultsRef                    = useRef(null)

  const handleAnalyze = async (formData, media) => {
    setLoading(true)
    setError(null)
    setResult(null)
    setLocalMedia(media || null)
    if (window.innerWidth < 768) {
      resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
    try {
      const res = await fetch('/api/analyze', { method: 'POST', body: formData })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail || 'Analysis failed')
      }
      const data = await res.json()
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.page}>
      <Header onHistoryClick={() => setShowHistory(true)} />
      <main className={styles.main}>
        <div className={styles.layout}>
          <aside className={styles.formCol}>
            <AnalysisForm onSubmit={handleAnalyze} loading={loading} hasResult={!!result} />
          </aside>
          <section className={styles.resultsCol} ref={resultsRef}>
            <ResultsPanel result={result} loading={loading} error={error} localMedia={localMedia} />
          </section>
        </div>
      </main>
      {showHistory && <HistoryDrawer onClose={() => setShowHistory(false)} />}
      <MobileTabBar />
    </div>
  )
}
