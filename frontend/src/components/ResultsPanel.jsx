import { AlertTriangle } from 'lucide-react'
import VerdictCard from './VerdictCard'
import SignalBars from './SignalBars'
import ContradictionAlert from './ContradictionAlert'
import DensityDetails from './DensityDetails'
import SHAPBreakdown from './SHAPBreakdown'
import BenfordStatus from './BenfordStatus'
import EmptyState from './EmptyState'
import LoadingState from './LoadingState'
import styles from './ResultsPanel.module.css'

export default function ResultsPanel({ result, loading, error }) {
  if (loading) return <LoadingState />
  if (error)   return <ErrorState message={error} />
  if (!result) return <EmptyState />

  const { modality_scores, contradiction, fusion, benford, verdict, case_id } = result

  return (
    <div className={styles.panel}>
      <VerdictCard verdict={verdict} caseId={case_id} />
      <SignalBars scores={modality_scores} />
      {contradiction?.flags?.length > 0 && (
        <ContradictionAlert contradiction={contradiction} />
      )}
      <DensityDetails density={modality_scores?.density} />
      {fusion?.shap_values && <SHAPBreakdown shap={fusion.shap_values} />}
      <BenfordStatus benford={benford} />
    </div>
  )
}

function ErrorState({ message }) {
  return (
    <div className={styles.errorState}>
      <div className={styles.errorIcon}><AlertTriangle size={30} /></div>
      <h3>Analysis Failed</h3>
      <p>{message}</p>
      <p className={styles.errorHint}>Check that all required fields are filled and the backend is running.</p>
    </div>
  )
}
