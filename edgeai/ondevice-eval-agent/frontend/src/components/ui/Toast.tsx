import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { AlertCircle, CheckCircle2, Info, X, AlertTriangle } from 'lucide-react';

type ToastKind = 'success' | 'error' | 'warning' | 'info';

interface Toast {
  id: string;
  kind: ToastKind;
  text: string;
}

interface ToastCtx {
  push: (kind: ToastKind, text: string) => void;
  success: (text: string) => void;
  error: (text: string) => void;
  warning: (text: string) => void;
  info: (text: string) => void;
}

const Ctx = createContext<ToastCtx | null>(null);

export function useToast(): ToastCtx {
  const v = useContext(Ctx);
  if (!v) throw new Error('useToast must be used inside <ToastProvider>');
  return v;
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const idRef = useRef(0);

  const push = useCallback((kind: ToastKind, text: string) => {
    const id = `t${++idRef.current}`;
    setToasts((prev) => [...prev, { id, kind, text }]);
    window.setTimeout(
      () => setToasts((prev) => prev.filter((t) => t.id !== id)),
      4200,
    );
  }, []);

  const ctx: ToastCtx = {
    push,
    success: (t) => push('success', t),
    error: (t) => push('error', t),
    warning: (t) => push('warning', t),
    info: (t) => push('info', t),
  };

  return (
    <Ctx.Provider value={ctx}>
      {children}
      <ToastViewport
        toasts={toasts}
        onDismiss={(id) =>
          setToasts((prev) => prev.filter((t) => t.id !== id))
        }
      />
    </Ctx.Provider>
  );
}

const TONE: Record<
  ToastKind,
  { bg: string; border: string; icon: typeof Info }
> = {
  success: {
    bg: 'var(--color-success-light)',
    border: 'rgba(16, 185, 129, 0.4)',
    icon: CheckCircle2,
  },
  error: {
    bg: 'var(--color-error-light)',
    border: 'rgba(239, 68, 68, 0.4)',
    icon: AlertCircle,
  },
  warning: {
    bg: 'var(--color-warning-light)',
    border: 'rgba(245, 158, 11, 0.4)',
    icon: AlertTriangle,
  },
  info: {
    bg: 'var(--primary-10)',
    border: 'var(--zededa-cyan-border)',
    icon: Info,
  },
};

function ToastViewport({
  toasts,
  onDismiss,
}: {
  toasts: Toast[];
  onDismiss: (id: string) => void;
}) {
  return (
    <div className="pointer-events-none fixed right-4 bottom-4 z-50 flex flex-col gap-2">
      {toasts.map((t) => {
        const tone = TONE[t.kind];
        const Icon = tone.icon;
        return (
          <div
            key={t.id}
            className="pointer-events-auto flex max-w-sm items-start gap-2 rounded-lg border px-3 py-2 text-sm shadow-md"
            style={{
              background: tone.bg,
              borderColor: tone.border,
              color: 'var(--gray-800)',
              backdropFilter: 'blur(6px)',
            }}
            role="status"
          >
            <Icon
              className="mt-0.5 h-4 w-4 shrink-0"
              style={{ color: tone.border }}
            />
            <span className="flex-1 whitespace-pre-wrap break-words">{t.text}</span>
            <button
              type="button"
              onClick={() => onDismiss(t.id)}
              className="opacity-60 hover:opacity-100"
              aria-label="Dismiss"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        );
      })}
    </div>
  );
}

// A standalone effect for auto-invoking showError on unhandled promise
// rejections in development — only enabled when explicitly requested.
export function useUnhandledRejectionToast(enabled = false) {
  const { error } = useToast();
  useEffect(() => {
    if (!enabled) return;
    const handler = (e: PromiseRejectionEvent) => {
      error(String(e.reason?.message ?? e.reason ?? 'Unhandled error'));
    };
    window.addEventListener('unhandledrejection', handler);
    return () => window.removeEventListener('unhandledrejection', handler);
  }, [enabled, error]);
}
