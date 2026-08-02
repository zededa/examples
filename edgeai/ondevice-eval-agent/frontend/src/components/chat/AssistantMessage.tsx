import { AlertCircle } from 'lucide-react';
import type { ChatMessage, MessageBlock, ToolCall } from '../../lib/types';
import { MarkdownRenderer } from './MarkdownRenderer';
import { ToolStepsBlock, ToolStep } from './ToolStepsBlock';
import { ZThrobber } from './ZThrobber';
import { MessageCopyButton } from './MessageCopyButton';
import { useThrottledMarkdown } from '../../hooks/useThrottledMarkdown';

/**
 * Assistant response — no bubble, no avatar, flows like prose.
 *
 * Two rendering modes:
 *
 *   1. Block-aware (new). Messages streamed with useStreamingChat write a
 *      `blocks` array preserving the order text / tools actually arrived.
 *      We render each block in place, so a turn that goes
 *        "Let me check → run_inference → Here's what I found → view_image → ..."
 *      shows exactly like that instead of bunching every tool up top.
 *
 *   2. Legacy (fallback). Old persisted messages and image-upload turns
 *      don't have blocks; for those we render `ToolStepsBlock` above the
 *      text as before. Nothing in localStorage needs migration.
 *
 * Streaming note: raw content is throttled (~12 Hz) via useThrottledMarkdown
 * to keep react-markdown + rehype-highlight parse cost sane during long
 * token bursts. Final value flushes instantly when streaming flips off.
 */
export function AssistantMessage({ message }: { message: ChatMessage }) {
  const streaming = message.streaming ?? false;
  const displayed = useThrottledMarkdown(message.content, streaming);
  const hasContent = displayed.trim().length > 0;
  const hasTools = message.toolCalls.length > 0;
  const runningTool = message.toolCalls.find((t) => t.status === 'running');
  const hasBlocks = Array.isArray(message.blocks) && message.blocks.length > 0;

  // Throbber only while there is nothing to show yet in THIS turn — once
  // any block has rendered (text or a tool step) the activity is
  // visible via the tool marker or the text itself.
  const showThrobber = streaming && !hasContent && !hasTools;
  const throbberLabel = runningTool
    ? `Running ${runningTool.name}`
    : 'Thinking';

  return (
    <article className="flex flex-col gap-4 animate-message-in">
      {hasBlocks ? (
        <BlockList
          blocks={message.blocks!}
          toolCalls={message.toolCalls}
          streaming={streaming}
          throttledContent={displayed}
        />
      ) : (
        <>
          {hasTools && (
            <ToolStepsBlock
              toolCalls={message.toolCalls}
              isStreaming={streaming}
            />
          )}
          {hasContent && (
            <div className="prose-msg" style={{ color: 'var(--gray-800)' }}>
              <MarkdownRenderer content={displayed} />
            </div>
          )}
        </>
      )}

      {showThrobber && <ZThrobber label={throbberLabel} />}

      {message.error && (
        <div
          className="flex items-start gap-2 rounded-lg border p-3 text-sm"
          style={{
            background: 'var(--color-error-light)',
            borderColor: 'rgba(239, 68, 68, 0.3)',
            color: 'var(--color-error)',
          }}
        >
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>{message.error}</span>
        </div>
      )}

      {!streaming && (hasContent || hasBlocks) && (
        <div className="-ml-1 flex items-center">
          <MessageCopyButton text={message.content} />
        </div>
      )}
    </article>
  );
}

/**
 * Renders the ordered block list. Text blocks go through the throttled
 * markdown pipeline; tool blocks render as inline ToolStep rows.
 *
 * The LAST text block during streaming renders the `throttledContent`
 * from useThrottledMarkdown so updates feel smooth; earlier text blocks
 * are already sealed so they render their stored text as-is.
 */
function BlockList({
  blocks,
  toolCalls,
  streaming,
  throttledContent,
}: {
  blocks: MessageBlock[];
  toolCalls: ToolCall[];
  streaming: boolean;
  throttledContent: string;
}) {
  // Resolve tool refs by id once. O(N) lookup below is fine (N is tiny).
  const toolById = new Map(toolCalls.map((t) => [t.id, t]));

  // Find the index of the last text block so we can swap in the throttled
  // value for the one that's currently growing.
  const lastTextIdx = (() => {
    for (let i = blocks.length - 1; i >= 0; i--) {
      if (blocks[i].type === 'text') return i;
    }
    return -1;
  })();

  // Reconstruct the full content (before the last text block) so we can
  // subtract it from throttledContent and show just this block's share.
  // This matters when a turn has multiple text blocks separated by tool
  // calls: earlier blocks are sealed, only the last one streams.
  let consumedChars = 0;
  for (let i = 0; i < lastTextIdx; i++) {
    const b = blocks[i];
    if (b.type === 'text') consumedChars += b.text.length;
  }
  const streamingTail = streaming ? throttledContent.slice(consumedChars) : '';

  return (
    <div className="flex flex-col gap-3">
      {blocks.map((block, idx) => {
        if (block.type === 'text') {
          const isLastText = idx === lastTextIdx;
          const text =
            isLastText && streaming ? streamingTail : block.text;
          if (!text.trim()) return null;
          return (
            <div
              key={`text-${idx}`}
              className="prose-msg"
              style={{ color: 'var(--gray-800)' }}
            >
              <MarkdownRenderer content={text} />
            </div>
          );
        }

        const tc = toolById.get(block.toolCallId);
        if (!tc) return null;
        return (
          <ToolStep
            key={`tool-${block.toolCallId}`}
            tool={tc}
            isLast={idx === blocks.length - 1}
            streaming={streaming}
            inline
          />
        );
      })}
    </div>
  );
}
