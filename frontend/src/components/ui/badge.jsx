import { cva } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const badgeVariants = cva(
  'inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors',
  {
    variants: {
      variant: {
        default:     'border-transparent bg-[var(--brand)] text-white',
        secondary:   'border-transparent bg-[var(--brand-muted)] text-[var(--brand)]',
        gold:        'border-transparent bg-[var(--canara-yellow-bg)] text-[var(--gold)] border-[var(--gold-border)]',
        outline:     'border-[var(--border-mid)] text-[var(--text-muted)]',
        success:     'border-transparent bg-[var(--ok-bg)] text-[var(--ok)] border-[var(--ok-border)]',
        warning:     'border-transparent bg-[var(--warn-bg)] text-[var(--warn)] border-[var(--warn-border)]',
        destructive: 'border-transparent bg-[var(--danger-bg)] text-[var(--danger)] border-[var(--danger-border)]',
      },
    },
    defaultVariants: { variant: 'default' },
  }
)

export function Badge({ className, variant, ...props }) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />
}

export { badgeVariants }
