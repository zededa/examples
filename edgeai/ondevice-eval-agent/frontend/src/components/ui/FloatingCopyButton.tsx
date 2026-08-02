import { useState } from 'react';
import { Check, Copy } from 'lucide-react';
import clsx from 'clsx';

/**
 * Small unobtrusive copy button designed to float in the corner of a
 * code block, table, or similar content. Uses muted colors until
 * hovered so it doesn't fight with the content for attention.
 *
 * `tone` controls which palette to use:
 *   - "dark"  — for placement over a dark surface (code block)
 *   - "light" — for placement over a light surface (table card)
 */
export function FloatingCopyButton({
  text,
  tone = 'light',
  className,
  title,
}: {
  text: string | (() => string);
  tone?: 'dark' | 'light';
  className?: string;
  title?: string;
}) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      const v = typeof text === 'function' ? text() : text;
      await navigator.clipboard.writeText(v);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* noop */
    }
  };

  const Icon = copied ? Check : Copy;
  const isDark = tone === 'dark';

  return (
    <button
      type="button"
      onClick={copy}
      aria-label={copied ? 'Copied' : 'Copy'}
      title={title ?? (copied ? 'Copied' : 'Copy')}
      className={clsx(
        'absolute right-2 top-2 inline-flex h-7 w-7 items-center justify-center rounded-md opacity-60 transition hover:opacity-100',
        className,
      )}
      style={{
        background: isDark
          ? 'rgba(255, 255, 255, 0.06)'
          : 'var(--island-bg)',
        color: copied
          ? 'var(--color-success)'
          : isDark
            ? '#d4d4d8'
            : 'var(--gray-500)',
        border: isDark
          ? '1px solid rgba(255, 255, 255, 0.08)'
          : '1px solid var(--gray-200)',
        backdropFilter: isDark ? 'blur(4px)' : undefined,
      }}
    >
      <Icon className="h-3.5 w-3.5" />
    </button>
  );
}
