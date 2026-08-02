import clsx from 'clsx';

type State = 'active' | 'warning' | 'offline';

export function StatusDot({
  state,
  className,
}: {
  state: State;
  className?: string;
}) {
  const color =
    state === 'active'
      ? 'var(--color-success)'
      : state === 'warning'
        ? 'var(--color-warning)'
        : 'var(--gray-400)';
  return (
    <span
      aria-label={state}
      className={clsx('inline-block h-2 w-2 shrink-0 rounded-full', className)}
      style={{
        background: color,
        boxShadow:
          state === 'active'
            ? `0 0 6px ${color}`
            : state === 'warning'
              ? `0 0 4px ${color}`
              : 'none',
      }}
    />
  );
}
