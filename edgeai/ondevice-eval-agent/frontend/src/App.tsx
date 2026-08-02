import { useEffect, useRef, useState } from 'react';
import { Header } from './components/layout/Header';
import { Sidebar } from './components/layout/Sidebar';
import { ChatThread } from './components/chat/ChatThread';
import { Composer } from './components/chat/Composer';
import { SessionWarningBanner } from './components/chat/SessionWarningBanner';
import { SettingsModal } from './components/settings/SettingsModal';
import { ToastProvider, useToast } from './components/ui/Toast';
import { ErrorBoundary } from './components/ErrorBoundary';
import { useStreamingChat } from './hooks/useStreamingChat';
import { useThreads } from './hooks/useThreads';
import { useAgentStatus } from './hooks/useAgentStatus';

const SIDEBAR_KEY = 'ondevice-eval.sidebarCollapsed';

export default function App() {
  return (
    <ToastProvider>
      <ErrorBoundary>
        <Shell />
      </ErrorBoundary>
    </ToastProvider>
  );
}

function Shell() {
  const toast = useToast();
  const { active, activeId, ensureActive } = useThreads();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(
    () => localStorage.getItem(SIDEBAR_KEY) === 'true',
  );
  const [settingsOpen, setSettingsOpen] = useState(false);
  const { status, refresh } = useAgentStatus();

  // Ensure an active thread on first load.
  useEffect(() => {
    if (!activeId) ensureActive();
  }, [activeId, ensureActive]);

  // Auto-collapse sidebar on narrow viewports so chat stays usable on
  // small laptops / phones. User's explicit toggle still wins: we only
  // force-collapse when the viewport *becomes* narrow, not every render.
  useEffect(() => {
    const mq = window.matchMedia('(max-width: 767px)');
    const apply = (narrow: boolean) => {
      if (narrow) setSidebarCollapsed(true);
    };
    apply(mq.matches);
    const handler = (e: MediaQueryListEvent) => apply(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  // If the agent reports itself not-configured, nudge the user into Settings
  // on the first page load (not when they explicitly dismissed).
  const nudgedRef = useRef<boolean>(false);
  useEffect(() => {
    if (!status || nudgedRef.current) return;
    nudgedRef.current = true;
    if (!status.enabled) {
      toast.info('No LLM configured — add one in Settings.');
    }
  }, [status, toast]);

  const { messages, isStreaming, warning, suggestions, send, stop, clearWarning } =
    useStreamingChat(active?.id ?? null);

  const toggleSidebar = () => {
    setSidebarCollapsed((v) => {
      const n = !v;
      localStorage.setItem(SIDEBAR_KEY, String(n));
      return n;
    });
  };

  return (
    <div
      className="flex h-screen w-screen"
      style={{ background: 'var(--gray-50)' }}
    >
      {/* Sidebar spans full viewport height — header only covers the
          main column on the right. */}
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggleCollapsed={toggleSidebar}
        onOpenSettings={() => setSettingsOpen(true)}
      />
      <div className="flex min-w-0 flex-1 flex-col">
        <Header status={status} onOpenSettings={() => setSettingsOpen(true)} />
        <main className="flex min-h-0 flex-1 flex-col">
          {warning && (
            <SessionWarningBanner
              warning={warning}
              onDismiss={clearWarning}
            />
          )}
          <ChatThread
            messages={messages}
            suggestions={suggestions}
            onPickSuggestion={(t) => send(t, [])}
          />
          <Composer
            onSubmit={(text, drafts) => send(text, drafts)}
            onStop={stop}
            isStreaming={isStreaming}
            disabled={!active}
          />
        </main>
      </div>
      {/* Isolated boundary so a settings render error doesn't black out the chat. */}
      <ErrorBoundary resetLabel="Close">
        <SettingsModal
          open={settingsOpen}
          onClose={() => setSettingsOpen(false)}
          onChange={refresh}
        />
      </ErrorBoundary>
    </div>
  );
}

