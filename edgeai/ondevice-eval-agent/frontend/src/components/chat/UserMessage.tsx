import type { ChatMessage } from '../../lib/types';
import { UserAvatar } from '../ui/Avatar';
import { AttachmentChip } from './AttachmentPreview';

export function UserMessage({
  message,
  onOpenImage,
}: {
  message: ChatMessage;
  onOpenImage: (src: string, alt?: string) => void;
}) {
  const atts = message.attachments ?? [];
  return (
    <div className="flex flex-row-reverse gap-4 animate-message-in">
      <UserAvatar />
      <div className="flex max-w-[80%] flex-col items-end gap-2">
        {atts.length > 0 && (
          <div className="flex flex-wrap justify-end gap-2">
            {atts.map((a) => (
              <AttachmentChip
                key={a.id}
                attachment={a}
                onOpen={() => {
                  if (a.kind === 'image' && a.previewUrl) {
                    onOpenImage(a.previewUrl, a.name);
                  }
                }}
              />
            ))}
          </div>
        )}
        {message.content && (
          <div className="bubble-user">
            <div className="whitespace-pre-wrap break-words text-[15px] leading-[1.6]">
              {message.content}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
