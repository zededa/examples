import { useCallback, useEffect, useRef, useState } from 'react';
import { agentApi } from '../lib/api';
import { parseSSE } from '../lib/sse';
import type { Attachment, ChatMessage, ToolCall } from '../lib/types';
import { shortId } from '../lib/ids';
import { threadStore } from '../lib/threadStore';
import {
  buildWelcomeMessage,
  isAutoWelcome,
  getStoredSuggestions,
} from '../lib/welcomeMessage';
import type { DraftAttachment } from '../components/chat/Composer';

/**
 * Drives a single in-flight chat stream.
 *
 * Messages are sourced from and written back to `threadStore` keyed by
 * the active thread id. The component passes the active thread in; if
 * it changes mid-stream, we abort and start fresh.
 *
 * SSE event contract (from webapp/routes/agent.py::_generate_sse_events):
 *   event: start         { session_id, warnings? }
 *   event: warning       { ... }
 *   (message)            { token: string }
 *   event: tool_start    { name, id }
 *   event: tool_end      { name, result }
 *   event: done|complete { response, tool_calls, finish_reason, meta }
 *   event: error         { error }
 */

export interface SessionWarning {
  has_warnings?: boolean;
  near_limit_dimensions?: string[];
  hard_limit_exceeded?: boolean;
  exceeded_dimension?: string;
  [k: string]: unknown;
}

export interface UseStreamingChat {
  messages: ChatMessage[];
  isStreaming: boolean;
  warning: SessionWarning | null;
  /**
   * Context-aware follow-up suggestions shown below the auto-welcome
   * message. Populated by buildWelcomeMessage from real server state.
   * Cleared once the user sends their first real message.
   */
  suggestions: string[];
  send: (text: string, drafts?: DraftAttachment[]) => void;
  stop: () => void;
  clearWarning: () => void;
}

export function useStreamingChat(threadId: string | null): UseStreamingChat {
  const [messages, setMessages] = useState<ChatMessage[]>(() =>
    threadId ? (threadStore.get(threadId)?.messages ?? []) : [],
  );
  const [isStreaming, setIsStreaming] = useState(false);
  const [warning, setWarning] = useState<SessionWarning | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const abortRef = useRef<AbortController | null>(null);
  // Thread ids we've already attempted to auto-welcome, to avoid
  // re-firing on effect re-runs or message reloads.
  const welcomedRef = useRef<Set<string>>(new Set());

  // Reload messages whenever the active thread changes (switching threads).
  // Also restores suggestion chips from the persisted welcome message, so a
  // page refresh on a thread that still only has the auto-welcome keeps
  // showing the same 4 follow-ups without re-fetching.
  useEffect(() => {
    if (!threadId) {
      setMessages([]);
      setSuggestions([]);
      return;
    }
    abortRef.current?.abort();
    abortRef.current = null;
    setIsStreaming(false);
    const existing = threadStore.get(threadId)?.messages ?? [];
    setMessages(existing);
    if (existing.length === 1 && isAutoWelcome(existing[0])) {
      setSuggestions(getStoredSuggestions(existing[0]));
    } else {
      setSuggestions([]);
    }
  }, [threadId]);

  // Auto-inject a server-status welcome message on empty threads.
  // Runs once per threadId per browser session.
  useEffect(() => {
    if (!threadId) return;
    if (welcomedRef.current.has(threadId)) return;
    if (messages.length > 0) return;
    welcomedRef.current.add(threadId);

    let cancelled = false;
    (async () => {
      try {
        const welcome = await buildWelcomeMessage();
        if (cancelled) return;
        // Only inject if the thread is still empty (user may have typed
        // something during the fetch). Also re-check the active thread
        // hasn't changed under us by reading the latest from the store.
        setMessages((prev) => (prev.length === 0 ? [welcome.message] : prev));
        setSuggestions(welcome.suggestions);
      } catch {
        // buildWelcomeMessage already has its own fallback; this catch is
        // just belt-and-suspenders so a truly unexpected error doesn't
        // crash the hook.
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [threadId, messages.length]);

  // Persist messages to thread store whenever they change.
  useEffect(() => {
    if (!threadId) return;
    threadStore.setMessages(threadId, messages);
  }, [threadId, messages]);

  // Abort on unmount.
  useEffect(() => () => abortRef.current?.abort(), []);

  const updateMsg = useCallback(
    (id: string, patch: (m: ChatMessage) => ChatMessage) => {
      setMessages((prev) => prev.map((m) => (m.id === id ? patch(m) : m)));
    },
    [],
  );

  const send = useCallback(
    (text: string, drafts: DraftAttachment[] = []) => {
      const trimmed = text.trim();
      if (!trimmed && drafts.length === 0) return;
      if (!threadId || isStreaming) return;

      const userAttachments: Attachment[] = drafts.map(({ file: _file, ...rest }) => rest);
      const userMsg: ChatMessage = {
        id: shortId('u'),
        role: 'user',
        content: trimmed,
        toolCalls: [],
        attachments: userAttachments.length > 0 ? userAttachments : undefined,
        createdAt: Date.now(),
      };
      const asstId = shortId('a');
      const asstMsg: ChatMessage = {
        id: asstId,
        role: 'assistant',
        content: '',
        toolCalls: [],
        blocks: [],
        createdAt: Date.now(),
        streaming: true,
      };
      setMessages((prev) => [...prev, userMsg, asstMsg]);
      setIsStreaming(true);
      // First real user message — retire the welcome suggestions.
      setSuggestions([]);

      const firstImage = drafts.find((d) => d.kind === 'image');
      const ac = new AbortController();
      abortRef.current = ac;

      (async () => {
        try {
          // Same streaming path for text and image-upload turns —
          // streamChat switches to multipart when an image is passed.
          const res = await agentApi.streamChat(
            trimmed || (firstImage ? '(image)' : ''),
            threadId,
            ac.signal,
            firstImage?.file,
          );
          if (!res.ok) throw new Error(`stream failed (${res.status})`);

          for await (const evt of parseSSE(res, ac.signal)) {
            const data = evt.data as Record<string, unknown> | string;

            if (evt.event === 'start') {
              const d = data as { warnings?: SessionWarning };
              if (d.warnings?.has_warnings) setWarning(d.warnings);
              continue;
            }
            if (evt.event === 'warning') {
              setWarning(data as SessionWarning);
              continue;
            }
            if (evt.event === 'message') {
              if (typeof data === 'object' && data && 'token' in data) {
                const tok = String(
                  (data as { token: unknown }).token ?? '',
                );
                if (tok) {
                  updateMsg(asstId, (m) => {
                    // Append to the trailing text block if there is one;
                    // otherwise start a new text block so this prose
                    // renders AFTER any preceding tool calls rather than
                    // getting merged into an earlier text block up top.
                    const blocks = m.blocks ? [...m.blocks] : [];
                    const last = blocks[blocks.length - 1];
                    if (last && last.type === 'text') {
                      blocks[blocks.length - 1] = {
                        type: 'text',
                        text: last.text + tok,
                      };
                    } else {
                      blocks.push({ type: 'text', text: tok });
                    }
                    return {
                      ...m,
                      content: m.content + tok,
                      blocks,
                    };
                  });
                }
              }
              continue;
            }
            if (evt.event === 'tool_start' && typeof data === 'object' && data) {
              const d = data as { id?: string; name?: string };
              const tc: ToolCall = {
                id: d.id || shortId('tc'),
                name: d.name || 'tool',
                status: 'running',
                startedAt: Date.now(),
              };
              updateMsg(asstId, (m) => {
                const blocks = m.blocks ? [...m.blocks] : [];
                blocks.push({ type: 'tool', toolCallId: tc.id });
                return {
                  ...m,
                  toolCalls: [...m.toolCalls, tc],
                  blocks,
                };
              });
              continue;
            }
            if (evt.event === 'tool_end' && typeof data === 'object' && data) {
              const d = data as { name?: string; result?: unknown };
              updateMsg(asstId, (m) => {
                const idx = [...m.toolCalls]
                  .map((t, i) => ({ t, i }))
                  .reverse()
                  .find(
                    ({ t }) => t.name === d.name && t.status === 'running',
                  )?.i;
                if (idx === undefined) return m;
                const next = [...m.toolCalls];
                next[idx] = {
                  ...next[idx],
                  status: 'completed',
                  result: d.result,
                  endedAt: Date.now(),
                };
                return { ...m, toolCalls: next };
              });
              continue;
            }
            if (
              (evt.event === 'done' || evt.event === 'complete') &&
              typeof data === 'object' &&
              data
            ) {
              const d = data as {
                response?: string;
                finish_reason?: string;
                warnings?: SessionWarning;
              };
              updateMsg(asstId, (m) => ({
                ...m,
                content:
                  d.response && d.response.length > m.content.length
                    ? d.response
                    : m.content,
                streaming: false,
                finishReason: d.finish_reason,
              }));
              if (d.warnings?.has_warnings) setWarning(d.warnings);
              continue;
            }
            if (evt.event === 'error' && typeof data === 'object' && data) {
              const d = data as {
                error?: string;
                retry_after?: number | null;
                status_code?: number;
                error_code?: string;
              };
              // Rate-limit errors from the EdgeAI built-in provider (or
              // any OpenAI-compatible upstream) carry a retry_after hint.
              // Surface it inline so the user knows when to try again
              // instead of just seeing a generic failure.
              const isRateLimited =
                d.error_code === 'rate_limit_exceeded' ||
                d.status_code === 429;
              const errMsg = (() => {
                const base = d.error || 'Unknown error';
                if (isRateLimited && d.retry_after) {
                  const secs = Math.ceil(d.retry_after);
                  return `Rate limited — retry in ~${secs}s. (${base})`;
                }
                if (isRateLimited) {
                  return `Rate limited — please retry shortly. (${base})`;
                }
                return base;
              })();
              updateMsg(asstId, (m) => ({
                ...m,
                streaming: false,
                error: errMsg,
              }));
              continue;
            }
          }
        } catch (err) {
          if ((err as Error).name === 'AbortError') {
            updateMsg(asstId, (m) => ({ ...m, streaming: false }));
          } else {
            updateMsg(asstId, (m) => ({
              ...m,
              streaming: false,
              error: (err as Error).message,
            }));
          }
        } finally {
          if (abortRef.current === ac) abortRef.current = null;
          setIsStreaming(false);
          updateMsg(asstId, (m) => ({ ...m, streaming: false }));
        }
      })();
    },
    [isStreaming, threadId, updateMsg],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsStreaming(false);
  }, []);

  const clearWarning = useCallback(() => setWarning(null), []);

  return { messages, isStreaming, warning, suggestions, send, stop, clearWarning };
}
