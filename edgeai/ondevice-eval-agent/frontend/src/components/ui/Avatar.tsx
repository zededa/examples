import clsx from 'clsx';
import { Bot } from 'lucide-react';

export function AssistantAvatar({ className }: { className?: string }) {
  return (
    <div
      className={clsx(
        'flex h-10 w-10 shrink-0 items-center justify-center rounded-lg',
        'border text-[color:var(--zededa-cyan)]',
        className,
      )}
      style={{
        background: 'var(--gray-100)',
        borderColor: 'var(--primary-20)',
      }}
    >
      <Bot className="h-5 w-5" />
    </div>
  );
}

export function UserAvatar({
  initials = 'YOU',
  className,
}: {
  initials?: string;
  className?: string;
}) {
  return (
    <div
      className={clsx(
        'flex h-10 w-10 shrink-0 items-center justify-center rounded-lg',
        'text-[10px] font-bold tracking-[0.5px]',
        className,
      )}
      style={{
        background:
          'linear-gradient(135deg, var(--gray-200), var(--gray-100))',
        border: '1px solid rgba(0, 0, 0, 0.06)',
        color: 'var(--gray-600)',
      }}
    >
      {initials.toUpperCase().slice(0, 3)}
    </div>
  );
}
