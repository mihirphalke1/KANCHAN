import { cn } from '@/lib/utils'

export function Input({ className, type, ...props }) {
  return (
    <input
      type={type}
      className={cn(
        'flex h-9 w-full rounded-[var(--r-sm)] border border-[var(--border-mid)] bg-[var(--surface)] px-3 py-1 text-sm text-[var(--text)] placeholder:text-[var(--text-faint)] transition-colors duration-[var(--fast)] file:border-0 file:bg-transparent file:text-sm file:font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--brand)] focus-visible:ring-offset-0 focus-visible:border-[var(--brand)] disabled:cursor-not-allowed disabled:opacity-50',
        className
      )}
      {...props}
    />
  )
}
