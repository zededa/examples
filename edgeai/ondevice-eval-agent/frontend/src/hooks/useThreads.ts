import { useCallback, useEffect, useMemo, useState } from 'react';
import { threadStore } from '../lib/threadStore';
import type { Thread } from '../lib/types';

/**
 * Reactive view over threadStore. Components subscribe via this hook.
 * Returns threads sorted by updatedAt desc, plus the active thread id.
 */
export function useThreads() {
  // Bumped by the threadStore listener so the memoized `threads` list
  // re-reads after create/rename/remove. Without this dep, `threads`
  // stayed frozen at whatever threadStore.list() returned on first
  // mount — so a fresh install (empty localStorage) would see threads=[]
  // forever, `active` would stay null, and quick-start button clicks
  // would hit the `if (!threadId)` guard in useStreamingChat.send().
  const [version, force] = useState(0);

  useEffect(() => threadStore.subscribe(() => force((v) => v + 1)), []);

  const threads = useMemo(() => threadStore.list(), [version]);
  const activeId = threadStore.getActive();

  const active = useMemo<Thread | null>(() => {
    if (!activeId) return null;
    return threads.find((t) => t.id === activeId) ?? null;
  }, [threads, activeId]);

  const createAndActivate = useCallback(() => {
    const t = threadStore.create();
    threadStore.setActive(t.id);
    return t;
  }, []);

  const ensureActive = useCallback((): Thread => {
    const cur = threadStore.getActive();
    if (cur) {
      const t = threadStore.get(cur);
      if (t) return t;
    }
    const existing = threadStore.list();
    if (existing.length > 0) {
      threadStore.setActive(existing[0].id);
      return existing[0];
    }
    const t = threadStore.create();
    threadStore.setActive(t.id);
    return t;
  }, []);

  return {
    threads,
    active,
    activeId,
    setActive: threadStore.setActive,
    create: threadStore.create,
    createAndActivate,
    ensureActive,
    remove: threadStore.remove,
    rename: threadStore.rename,
    exportAll: threadStore.exportAll,
    importAll: threadStore.importAll,
  };
}
