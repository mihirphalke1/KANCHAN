import { Camera, Mic, Scale, Layers } from 'lucide-react'
import styles from './EmptyState.module.css'

const TIPS = [
  { Icon: Camera, text: 'Photos on a plain backdrop — the system finds the stones itself' },
  { Icon: Scale,  text: 'Weigh in air and in water, to 0.01 g' },
  { Icon: Mic,    text: 'Tap once with a stylus, record the ring in a quiet room' },
  { Icon: Layers, text: 'Optional: a touchstone streak photo adds a fourth check' },
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
          Three core tests, all required: photos of the item, the two weight
          readings, and a tap recording. No single test can approve an item
          on its own — that is how fakes have slipped through elsewhere.
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
