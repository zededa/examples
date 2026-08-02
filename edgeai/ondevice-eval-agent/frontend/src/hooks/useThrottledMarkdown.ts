import { useEffect, useRef, useState } from 'react';

/**
 * Throttle the value passed to MarkdownRenderer during streaming.
 *
 * ReactMarkdown + remark-gfm + rehype-highlight is heavy — re-parsing
 * the entire message on every single token update (30-60 Hz) is what
 * makes streaming feel chunky: React can't schedule enough frames, so
 * updates land in visible bursts.
 *
 * This hook exposes a throttled copy of `content` that updates at most
 * every `intervalMs` while `streaming` is true, and flushes to the
 * latest value immediately when `streaming` flips false so the user
 * sees the fully-formed message at the end of the turn.
 *
 * 80 ms (≈ 12 Hz) is enough to feel alive while cutting parse cost by
 * ~5x vs the old useSmoothText path (which drove ~60 parses/sec).
 */
export function useThrottledMarkdown(
  content: string,
  streaming: boolean,
  intervalMs: number = 80,
): string {
  const [throttled, setThrottled] = useState(content);
  const lastFlushRef = useRef<number>(performance.now());
  const timeoutRef = useRef<number | null>(null);

  useEffect(() => {
    // Flush immediately on stream end (or when not streaming at all)
    // so the final, complete content is rendered with markdown parsed.
    if (!streaming) {
      if (timeoutRef.current !== null) {
        window.clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
      setThrottled(content);
      return;
    }

    const now = performance.now();
    const elapsed = now - lastFlushRef.current;

    if (elapsed >= intervalMs) {
      lastFlushRef.current = now;
      setThrottled(content);
      return;
    }

    // Schedule a catch-up flush at the remaining interval.
    if (timeoutRef.current !== null) {
      window.clearTimeout(timeoutRef.current);
    }
    timeoutRef.current = window.setTimeout(() => {
      lastFlushRef.current = performance.now();
      timeoutRef.current = null;
      setThrottled(content);
    }, intervalMs - elapsed);

    return () => {
      if (timeoutRef.current !== null) {
        window.clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, [content, streaming, intervalMs]);

  return throttled;
}
