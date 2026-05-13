import clsx from 'clsx';
import { FloatingCopyButton } from '../ui/FloatingCopyButton';

/**
 * Minimal code block:
 *   - No full-width header bar
 *   - Floating copy button in the top-right
 *   - Explicit dark palette (not theme-flipping CSS vars) so contrast
 *     stays high regardless of the user's light/dark preference
 */
export function CodeBlock({
  language,
  children,
  className,
}: {
  language?: string;
  children: string;
  className?: string;
}) {
  return (
    <div
      className={clsx('code-block group relative my-4 overflow-hidden', className)}
    >
      {language && (
        <span
          className="pointer-events-none absolute left-3 top-2 select-none font-mono text-[10px] font-semibold uppercase tracking-wider opacity-50"
          style={{ color: '#94a3b8' }}
        >
          {language}
        </span>
      )}
      <FloatingCopyButton text={children} tone="dark" />
      <pre className="m-0 overflow-x-auto px-4 pb-4 pt-7 font-mono text-[12.5px] leading-6">
        <code className={language ? `language-${language}` : undefined}>
          {children}
        </code>
      </pre>
    </div>
  );
}
