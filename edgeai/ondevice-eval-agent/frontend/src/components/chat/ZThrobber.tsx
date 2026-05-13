/**
 * Thinking indicator: a throbbing ZEDEDA "Z" in brand cyan, paired
 * with a softly pulsing label. Replaces the generic three-dot typing
 * indicator.
 */
export function ZThrobber({ label = 'Thinking' }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 py-1">
      <span className="relative inline-flex h-6 w-6 items-center justify-center">
        <svg
          viewBox="0 0 24 24"
          className="h-5 w-5 z-throb"
          aria-hidden
          style={{
            color: 'var(--zededa-cyan)',
            filter: 'drop-shadow(0 0 6px var(--primary-40))',
          }}
        >
          <path
            d="M5 5H19L5 19H19"
            fill="none"
            stroke="currentColor"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </span>
      <span
        className="thinking-label text-sm font-medium"
        style={{ color: 'var(--gray-500)' }}
      >
        {label}
        <span className="thinking-dots">…</span>
      </span>
    </div>
  );
}
