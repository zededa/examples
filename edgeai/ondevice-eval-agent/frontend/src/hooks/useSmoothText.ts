import { useEffect, useRef, useState } from 'react';

/**
 * Smoothly reveal `target` a few characters per frame while `streaming`
 * is true, so backend token bursts (~70 chars each) don't visibly jump.
 *
 * When streaming flips false, snaps to `target` immediately so the final
 * message doesn't linger with a partial reveal.
 */
export function useSmoothText(target: string, streaming: boolean): string {
  const [displayed, setDisplayed] = useState(() => (streaming ? '' : target));
  const targetRef = useRef(target);
  targetRef.current = target;

  useEffect(() => {
    if (!streaming) {
      setDisplayed(targetRef.current);
      return;
    }

    let raf = 0;
    let stopped = false;

    const tick = () => {
      if (stopped) return;
      setDisplayed((cur) => {
        const t = targetRef.current;
        if (cur.length >= t.length) return cur;
        // Reveal ~2 chars/frame when close, faster when far behind
        // so we catch up if many tokens arrived in one chunk.
        const gap = t.length - cur.length;
        const step = Math.max(2, Math.ceil(gap / 12));
        return t.slice(0, cur.length + step);
      });
      raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);
    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
    };
  }, [streaming]);

  return displayed;
}
