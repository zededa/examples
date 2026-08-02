import { useEffect, useRef, useState } from 'react';
import type { ChatMessage } from '../../lib/types';
import { UserMessage } from './UserMessage';
import { AssistantMessage } from './AssistantMessage';
import { WelcomeScreen } from './WelcomeScreen';
import { ImageModal } from './ImageModal';
import { isAutoWelcome } from '../../lib/welcomeMessage';

interface Props {
  messages: ChatMessage[];
  /** Context-aware follow-up prompts shown below the auto-welcome message. */
  suggestions?: string[];
  onPickSuggestion: (text: string) => void;
}

export function ChatThread({ messages, suggestions, onPickSuggestion }: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  const [preview, setPreview] = useState<{ src: string; alt?: string } | null>(
    null,
  );

  const last = messages[messages.length - 1];
  const lastAssistantLen = last?.role === 'assistant' ? last.content.length : 0;
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages.length, lastAssistantLen]);

  if (messages.length === 0) {
    // The welcome message is fetched and injected by useStreamingChat, so
    // this branch only renders very briefly on a brand-new thread while
    // the /server-info + /models + /llm/status fetches are in flight.
    return (
      <div className="flex flex-1 items-center justify-center overflow-y-auto p-6">
        <WelcomeScreen onPick={onPickSuggestion} />
      </div>
    );
  }

  // Show suggestion chips only when the thread is still just the
  // auto-welcome (no user messages yet). Once the user sends anything,
  // useStreamingChat clears the suggestions list.
  const showSuggestions =
    (suggestions?.length ?? 0) > 0 &&
    messages.length === 1 &&
    isAutoWelcome(messages[0]);

  return (
    <div
      className="flex-1 overflow-y-auto scroll-smooth rounded-md border p-6"
      style={{
        margin: '0.75rem',
        background:
          'linear-gradient(to bottom, var(--island-bg), var(--island-bg), var(--gray-50))',
        borderColor: 'var(--gray-100)',
      }}
    >
      <div className="mx-auto flex w-full max-w-3xl flex-col gap-6">
        {messages.map((m) =>
          m.role === 'user' ? (
            <UserMessage
              key={m.id}
              message={m}
              onOpenImage={(src, alt) => setPreview({ src, alt })}
            />
          ) : (
            <AssistantMessage key={m.id} message={m} />
          ),
        )}

        {showSuggestions && (
          <div
            className="flex flex-wrap gap-2"
            aria-label="Suggested follow-up questions"
          >
            {suggestions!.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => onPickSuggestion(s)}
                className="inline-flex items-center rounded-full border bg-transparent px-4 py-2 text-sm font-medium transition-all hover:-translate-y-px hover:shadow-sm active:scale-[0.97]"
                style={{
                  borderColor: 'var(--primary-20)',
                  color: 'var(--gray-700)',
                }}
              >
                {s}
              </button>
            ))}
          </div>
        )}

        <div ref={endRef} />
      </div>
      {preview && (
        <ImageModal
          src={preview.src}
          alt={preview.alt}
          onClose={() => setPreview(null)}
        />
      )}
    </div>
  );
}
