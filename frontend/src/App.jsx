import { useRef, useState } from 'react'
import Header from './components/Header'
import AnalysisForm from './components/AnalysisForm'
import ResultsPanel from './components/ResultsPanel'
import HistoryDrawer from './components/HistoryDrawer'
import HeroPage from './components/HeroPage'
import styles from './App.module.css'

export default function App() {
  const [showHero, setShowHero]       = useState(true)
  const [result, setResult]           = useState(null)
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState(null)
  const [showHistory, setShowHistory] = useState(false)
  const resultsRef                    = useRef(null)

  if (showHero) {
    return <HeroPage onLaunch={() => setShowHero(false)} />
  }

  const handleAnalyze = async (formData) => {
    setLoading(true)
    setError(null)
    setResult(null)
    // On mobile the results column is below the form — scroll to it immediately
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
    <div className={styles.app}>
      <Header onHistoryClick={() => setShowHistory(true)} />
      <main className={styles.main}>
        <div className={styles.layout}>
          <aside className={styles.formCol}>
            <AnalysisForm onSubmit={handleAnalyze} loading={loading} />
          </aside>
          <section className={styles.resultsCol} ref={resultsRef}>
            <ResultsPanel result={result} loading={loading} error={error} />
          </section>
        </div>
      </main>
      {showHistory && <HistoryDrawer onClose={() => setShowHistory(false)} />}
    </div>
  )
}
