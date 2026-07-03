import { Slot } from '@radix-ui/react-slot'
import { cva } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-[var(--r-sm)] text-sm font-semibold transition-all duration-[var(--base)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--brand)] focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      variant: {
        default:
          'bg-[var(--brand)] text-white shadow-sm hover:bg-[var(--brand-dark)] active:scale-[.98]',
        secondary:
          'bg-[var(--brand-muted)] text-[var(--brand-dark)] border border-[var(--brand-border)] hover:bg-[var(--brand-border)]/50',
        yellow:
          'bg-[var(--yellow)] text-[var(--text)] font-bold shadow-sm hover:bg-[var(--yellow-dark)] active:scale-[.98]',
        outline:
          'border border-[var(--border-mid)] bg-white text-[var(--text-mid)] hover:bg-[var(--bg)] hover:border-[var(--brand)]',
        ghost:
          'bg-transparent text-[var(--text-muted)] hover:bg-[var(--bg)] hover:text-[var(--text)]',
        destructive:
          'bg-[var(--danger)] text-white hover:bg-[var(--danger)]/90',
        link:
          'text-[var(--brand)] underline-offset-4 hover:underline p-0 h-auto',
      },
      size: {
        sm:      'h-8 px-3 text-xs',
        default: 'h-9 px-4 py-2',
        lg:      'h-11 px-8 text-[15px]',
        xl:      'h-13 px-10 text-[16px] rounded-[var(--r)]',
        icon:    'h-9 w-9 p-0',
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
