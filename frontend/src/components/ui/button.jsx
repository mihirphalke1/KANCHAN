import { Slot } from '@radix-ui/react-slot'
import { cva } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-[var(--r-sm)] text-sm font-semibold transition-all duration-[var(--base)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--brand)] focus-visible:ring-offset-1 disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      variant: {
        default:
          'bg-[var(--brand)] text-white shadow-sm hover:bg-[var(--brand-light)] active:scale-[.98]',
        secondary:
          'bg-[var(--brand-muted)] text-[var(--brand)] border border-[var(--brand-light)]/30 hover:bg-[var(--brand-light)]/15',
        gold:
          'bg-[var(--canara-yellow)] text-[var(--text)] shadow-sm hover:brightness-95 active:scale-[.98]',
        outline:
          'border border-[var(--border-mid)] bg-[var(--surface)] text-[var(--text)] hover:bg-[var(--bg)] hover:border-[var(--border-mid)]',
        ghost:
          'bg-transparent text-[var(--text-muted)] hover:bg-[var(--bg)] hover:text-[var(--text)]',
        destructive:
          'bg-[var(--danger)] text-white hover:bg-[var(--danger)]/90',
        link:
          'text-[var(--brand)] underline-offset-4 hover:underline',
      },
      size: {
        sm:      'h-8 px-3 text-xs',
        default: 'h-9 px-4 py-2',
        lg:      'h-11 px-6 text-base',
        icon:    'h-9 w-9',
      },
    },
    defaultVariants: { variant: 'default', size: 'default' },
  }
)

export function Button({ className, variant, size, asChild = false, ...props }) {
  const Comp = asChild ? Slot : 'button'
  return (
    <Comp className={cn(buttonVariants({ variant, size }), className)} {...props} />
  )
}

export { buttonVariants }
