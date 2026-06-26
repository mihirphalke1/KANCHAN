import { cn } from '@/lib/utils'

export function Card({ className, ...props }) {
  return (
    <div
      className={cn(
        'rounded-[var(--r-lg)] border border-[var(--border)] bg-[var(--surface)] shadow-[var(--shadow-sm)]',
        className
      )}
      {...props}
    />
  )
}

export function CardHeader({ className, ...props }) {
  return (
    <div className={cn('flex flex-col gap-1 px-5 pt-5 pb-3', className)} {...props} />
  )
}

export function CardTitle({ className, ...props }) {
  return (
    <h3
      className={cn('font-semibold text-base leading-tight text-[var(--text)]', className)}
      {...props}
    />
  )
}

export function CardDescription({ className, ...props }) {
  return (
    <p className={cn('text-sm text-[var(--text-muted)]', className)} {...props} />
  )
}

export function CardContent({ className, ...props }) {
  return <div className={cn('px-5 pb-5', className)} {...props} />
}

export function CardFooter({ className, ...props }) {
  return (
    <div
      className={cn('flex items-center px-5 py-3 border-t border-[var(--border)]', className)}
      {...props}
    />
  )
}
