import { Camera, Mic, Scale, Layers } from 'lucide-react'
import styles from './EmptyState.module.css'

const TIPS = [
  { Icon: Camera, text: '4–6 angle photos give the best visual score' },
  { Icon: Mic,    text: 'Tap the item with a stylus for a clear ring sound' },
  { Icon: Scale,  text: 'Weigh to 0.01 g precision for accurate density' },
  { Icon: Layers, text: 'A touchstone streak photo activates the streak module' },
]

export default function EmptyState() {
  return (
    <div className={styles.wrap}>
      <div className={styles.iconWrap}>
        <img src="/OG.svg" alt="KANCHAN-AI illustration" />
      </div>
      <div className={styles.textGroup}>
        <h2 className={styles.heading}>Ready to Analyse</h2>
        <p className={styles.body}>
          Enter item details and weight measurements on the left, then upload
          photos or a tap-test recording for the highest accuracy verdict.
        </p>
      </div>
      <div className={styles.tips}>
        {TIPS.map(({ Icon, text }) => (
          <div key={text} className={styles.tip}>
            <span className={styles.tipIcon}><Icon size={13} strokeWidth={2} /></span>
            <span className={styles.tipText}>{text}</span>
          </div>
        ))}
      </div>
      <div className={styles.novelties}>
        <span className={styles.noveltyChip}>Sound ring test</span>
        <span className={styles.noveltyChip}>Branch records check</span>
        <span className={styles.noveltyChip}>Tests cross-check each other</span>
      </div>
    </div>
  )
}
