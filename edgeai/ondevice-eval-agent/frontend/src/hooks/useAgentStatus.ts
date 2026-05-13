import { useCallback, useEffect, useState } from 'react';
import { agentApi, type AgentStatusResponse } from '../lib/api';

/**
 * Polls /agent/status so the header badge reflects whether the router
 * has an active provider. Refetches on window focus and on demand via
 * the returned `refresh` callback — SettingsModal calls this after
 * adding/activating a credential.
 */
export function useAgentStatus(pollMs = 30000) {
  const [status, setStatus] = useState<AgentStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const s = await agentApi.status();
      setStatus(s);
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, pollMs);
    const onFocus = () => refresh();
    window.addEventListener('focus', onFocus);
    return () => {
      clearInterval(id);
      window.removeEventListener('focus', onFocus);
    };
  }, [refresh, pollMs]);

  return { status, error, refresh };
}
