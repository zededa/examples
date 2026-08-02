import { useRef, useState } from 'react';
import { ArrowUp, ImageIcon, Square, X } from 'lucide-react';
import { AutoResizeTextarea } from '../ui/AutoResizeTextarea';
import { AttachmentChip } from './AttachmentPreview';
import type { Attachment } from '../../lib/types';
import { shortId } from '../../lib/ids';

export interface DraftAttachment extends Attachment {
  file: File;
}

interface Props {
  onSubmit: (text: string, attachments: DraftAttachment[]) => void;
  onStop: () => void;
  isStreaming: boolean;
  disabled?: boolean;
}

const MAX_MB = 10;

export function Composer({ onSubmit, onStop, isStreaming, disabled }: Props) {
  const [text, setText] = useState('');
  const [drafts, setDrafts] = useState<DraftAttachment[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const canSend = (text.trim() || drafts.length > 0) && !isStreaming && !disabled;

  const submit = () => {
    if (!canSend) return;
    onSubmit(text, drafts);
    setText('');
    setDrafts([]);
    textareaRef.current?.focus();
  };

  const handleFiles = (files: FileList | null) => {
    if (!files) return;
    const next: DraftAttachment[] = [];
    for (const file of Array.from(files)) {
      if (file.size > MAX_MB * 1024 * 1024) {
        // Silently drop oversize; toast would require prop drilling.
        console.warn(`Dropped ${file.name}: exceeds ${MAX_MB}MB`);
        continue;
      }
      const kind: Attachment['kind'] = file.type.startsWith('image/')
        ? 'image'
        : 'file';
      const previewUrl =
        kind === 'image' ? URL.createObjectURL(file) : undefined;
      next.push({
        id: shortId('att'),
        kind,
        name: file.name,
        mimeType: file.type,
        previewUrl,
        file,
      });
    }
    setDrafts((prev) => [...prev, ...next]);
  };

  const removeDraft = (id: string) => {
    setDrafts((prev) => {
      const gone = prev.find((d) => d.id === id);
      if (gone?.previewUrl) URL.revokeObjectURL(gone.previewUrl);
      return prev.filter((d) => d.id !== id);
    });
  };

  return (
    <div
      className="p-4"
      style={{
        background: 'var(--island-bg)',
        borderTop: '1px solid var(--gray-100)',
      }}
      onDragOver={(e) => {
        e.preventDefault();
      }}
      onDrop={(e) => {
        e.preventDefault();
        handleFiles(e.dataTransfer.files);
      }}
    >
      <div className="mx-auto w-full max-w-3xl">
        {drafts.length > 0 && (
          <div className="mb-3 flex flex-wrap gap-2">
            {drafts.map((d) => (
              <AttachmentChip
                key={d.id}
                attachment={d}
                onRemove={() => removeDraft(d.id)}
                onOpen={() => {
                  if (d.previewUrl) window.open(d.previewUrl, '_blank');
                }}
              />
            ))}
            {drafts.length > 0 && (
              <button
                type="button"
                onClick={() =>
                  setDrafts((prev) => {
                    prev.forEach((d) => d.previewUrl && URL.revokeObjectURL(d.previewUrl));
                    return [];
                  })
                }
                className="flex items-center gap-1 rounded-full border px-2.5 py-1 text-[11px]"
                style={{
                  borderColor: 'var(--gray-200)',
                  color: 'var(--gray-500)',
                }}
              >
                <X className="h-3 w-3" /> Clear
              </button>
            )}
          </div>
        )}

        <div
          className="flex items-end gap-2 rounded-2xl border p-2 transition-colors focus-within:shadow-[var(--shadow-floating-focus)]"
          style={{
            background: 'var(--gray-50)',
            borderColor: 'rgba(0,0,0,0.06)',
          }}
        >
          <button
            type="button"
            aria-label="Attach image"
            onClick={() => fileRef.current?.click()}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg transition-colors"
            style={{
              color: drafts.length > 0 ? 'var(--zededa-cyan)' : 'var(--gray-500)',
              background:
                drafts.length > 0 ? 'var(--primary-10)' : 'transparent',
            }}
          >
            <ImageIcon className="h-4 w-4" />
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={(e) => {
              handleFiles(e.target.files);
              e.target.value = '';
            }}
          />

          <AutoResizeTextarea
            ref={textareaRef}
            value={text}
            rows={1}
            placeholder="Ask about models, inputs, outputs, or integration…"
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            className="min-h-[24px] flex-1 resize-none border-0 bg-transparent px-2 py-2 text-[15px] outline-none"
            style={{ color: 'var(--gray-900)' }}
          />

          {isStreaming ? (
            <button
              type="button"
              onClick={onStop}
              aria-label="Stop streaming"
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border transition-all"
              style={{
                background: 'var(--gray-100)',
                borderColor: 'var(--gray-200)',
                color: 'var(--color-error)',
              }}
            >
              <Square className="h-4 w-4" />
            </button>
          ) : (
            <button
              type="button"
              onClick={submit}
              disabled={!canSend}
              aria-label="Send message"
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border transition-all disabled:cursor-not-allowed"
              style={{
                background: canSend ? 'var(--primary-10)' : 'var(--gray-50)',
                borderColor: canSend
                  ? 'var(--zededa-cyan-border)'
                  : 'var(--gray-200)',
                color: canSend ? 'var(--zededa-cyan)' : 'var(--gray-300)',
              }}
            >
              <ArrowUp className="h-5 w-5" />
            </button>
          )}
        </div>
        <p
          className="mt-3 text-center text-[11px]"
          style={{ color: 'var(--gray-400)' }}
        >
          Enter to send · Shift+Enter for newline · drag-drop images
        </p>
      </div>
    </div>
  );
}
