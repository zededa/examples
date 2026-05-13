import { StatusDot } from '../ui/StatusDot';
import type { AgentStatusResponse } from '../../lib/api';

/**
 * Top bar — the brand/logo lives in the sidebar now. The only thing
 * in the header is the LLM status pill, which doubles as a shortcut
 * to open Settings.
 */
export function Header({
  status,
  onOpenSettings,
}: {
  status: AgentStatusResponse | null;
  onOpenSettings: () => void;
}) {
  const active = status?.llm_router?.active_provider;
  const enabled = status?.enabled ?? false;

  const dotState: 'active' | 'warning' | 'offline' = enabled
    ? 'active'
    : status?.llm_router?.providers && status.llm_router.providers > 0
      ? 'warning'
      : 'offline';

  const label = enabled
    ? `${active ?? '?'} · ${status?.model ?? 'no model'}`
    : (status?.message ?? 'No LLM configured');

  return (
    <header
      className="flex h-14 shrink-0 items-center justify-end gap-3 border-b px-4"
      style={{
        background: 'var(--island-bg)',
        borderColor: 'var(--gray-100)',
      }}
    >
      <button
        type="button"
        onClick={onOpenSettings}
        className="flex max-w-[280px] items-center gap-2 rounded-full border px-3 py-1.5 text-xs"
        style={{
          borderColor: 'var(--gray-200)',
          background: 'var(--island-bg)',
          color: 'var(--gray-700)',
        }}
        title="Open LLM settings"
      >
        <StatusDot state={dotState} />
        <span className="max-w-[220px] truncate">{label}</span>
      </button>
    </header>
  );
}
