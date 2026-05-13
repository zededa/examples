import { AlertTriangle, X } from 'lucide-react';
import type { SessionWarning } from '../../hooks/useStreamingChat';

export function SessionWarningBanner({
  warning,
  onDismiss,
}: {
  warning: SessionWarning;
  onDismiss: () => void;
}) {
  const hard = warning.hard_limit_exceeded;
  const near = warning.near_limit_dimensions ?? [];
  const text = hard
    ? `Session limit reached${warning.exceeded_dimension ? ` (${warning.exceeded_dimension})` : ''}. Start a new chat to continue.`
    : near.length > 0
      ? `Session nearing its limit on: ${near.join(', ')}.`
      : 'Session warning.';

  return (
    <div
      className="mx-3 mb-2 flex items-start gap-2 rounded-lg border px-3 py-2 text-sm"
      style={{
        background: hard ? 'var(--color-error-light)' : 'var(--color-warning-light)',
        borderColor: hard ? 'rgba(239, 68, 68, 0.3)' : 'rgba(245, 158, 11, 0.3)',
        color: hard ? 'var(--color-error)' : 'var(--color-warning)',
      }}
      role="status"
    >
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <span className="flex-1">{text}</span>
      <button
        type="button"
        onClick={onDismiss}
        aria-label="Dismiss"
        className="opacity-70 hover:opacity-100"
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
