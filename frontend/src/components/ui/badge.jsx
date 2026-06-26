import { cva } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const badgeVariants = cva(
  'inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors',
  {
    variants: {
      variant: {
        default:     'border-transparent bg-[var(--brand)] text-white',
        secondary:   'border-[var(--brand-border)] bg-[var(--brand-muted)] text-[var(--brand-dark)]',
        yellow:      'border-[var(--yellow-border)] bg-[var(--yellow-bg)] text-[var(--yellow-dark)]',
        outline:     'border-[var(--border-mid)] bg-transparent text-[var(--text-muted)]',
        success:     'border-[var(--ok-border)] bg-[var(--ok-bg)] text-[var(--ok)]',
        warning:     'border-[var(--warn-border)] bg-[var(--warn-bg)] text-[var(--warn)]',
        destructive: 'border-[var(--danger-border)] bg-[var(--danger-bg)] text-[var(--danger)]',
      },
    },
    defaultVariants: { variant: 'default' },
  }
)

export function Badge({ className, variant, ...props }) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />
}

export { badgeVariants }
