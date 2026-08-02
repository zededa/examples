import { File as FileIcon, X } from 'lucide-react';
import type { Attachment } from '../../lib/types';

/** Inline thumbnail for a draft or sent attachment. */
export function AttachmentChip({
  attachment,
  onRemove,
  onOpen,
}: {
  attachment: Attachment;
  onRemove?: () => void;
  onOpen?: () => void;
}) {
  const isImage = attachment.kind === 'image';

  return (
    <div
      className="hover-ring relative flex items-center gap-2 r-input border px-2.5 py-1.5 text-xs"
      style={{
        background: 'var(--island-bg)',
        borderColor: 'var(--gray-200)',
        color: 'var(--gray-700)',
      }}
    >
      {isImage && attachment.previewUrl ? (
        <button
          type="button"
          onClick={onOpen}
          className="flex h-10 w-10 overflow-hidden rounded-md border"
          style={{ borderColor: 'var(--primary-20)' }}
        >
          <img
            src={attachment.previewUrl}
            alt={attachment.name}
            className="h-full w-full object-cover"
          />
        </button>
      ) : (
        <div
          className="flex h-10 w-10 items-center justify-center rounded-md"
          style={{
            background: 'var(--gray-100)',
            color: 'var(--gray-500)',
          }}
        >
          <FileIcon className="h-4 w-4" />
        </div>
      )}
      <span className="max-w-[160px] truncate">{attachment.name}</span>
      {onRemove && (
        <button
          type="button"
          onClick={onRemove}
          aria-label={`Remove ${attachment.name}`}
          className="absolute -right-1.5 -top-1.5 flex h-5 w-5 items-center justify-center rounded-full border-2 text-white"
          style={{
            background: 'var(--gray-600)',
            borderColor: 'var(--island-bg)',
          }}
        >
          <X className="h-2.5 w-2.5" />
        </button>
      )}
    </div>
  );
}
