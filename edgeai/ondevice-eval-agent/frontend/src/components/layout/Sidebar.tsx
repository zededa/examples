import { useMemo, useRef, useState } from 'react';
import clsx from 'clsx';
import {
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Download,
  MessageSquare,
  Moon,
  Pencil,
  Plus,
  PlusCircle,
  Search,
  Settings,
  Sun,
  Trash2,
  Upload,
} from 'lucide-react';
import type { Thread } from '../../lib/types';
import { useThreads } from '../../hooks/useThreads';
import { useToast } from '../ui/Toast';

const PAGE_KEY = 'ondevice-eval.sidebarPageSize';
const HISTORY_OPEN_KEY = 'ondevice-eval.sidebarHistoryOpen';

const SUPPORTED_PAGE_SIZES = [10, 20, 50] as const;
const DEFAULT_PAGE_SIZE = 10;

function normalizePageSize(value: unknown): number {
  const n = Number(value);
  return (SUPPORTED_PAGE_SIZES as readonly number[]).includes(n)
    ? n
    : DEFAULT_PAGE_SIZE;
}

interface Props {
  collapsed: boolean;
  onToggleCollapsed: () => void;
  onOpenSettings: () => void;
}

export function Sidebar({ collapsed, onToggleCollapsed, onOpenSettings }: Props) {
  const {
    threads,
    activeId,
    setActive,
    createAndActivate,
    remove,
    rename,
    exportAll,
    importAll,
  } = useThreads();
  const toast = useToast();

  const [query, setQuery] = useState('');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState<number>(
    () => normalizePageSize(localStorage.getItem(PAGE_KEY)),
  );
  const [historyOpen, setHistoryOpen] = useState(
    () => localStorage.getItem(HISTORY_OPEN_KEY) !== 'false',
  );
  const [editingId, setEditingId] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return threads;
    return threads.filter((t) => {
      if (t.title.toLowerCase().includes(q)) return true;
      return t.messages.some((m) => m.content.toLowerCase().includes(q));
    });
  }, [query, threads]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const safePage = Math.min(page, totalPages);
  const pageItems = filtered.slice((safePage - 1) * pageSize, safePage * pageSize);

  const toggleHistory = () => {
    setHistoryOpen((v) => {
      const n = !v;
      localStorage.setItem(HISTORY_OPEN_KEY, String(n));
      return n;
    });
  };

  const handleDelete = (t: Thread) => {
    if (
      !window.confirm(
        `Delete "${t.title}"?  ${t.messages.length} message${t.messages.length === 1 ? '' : 's'} will be lost.`,
      )
    )
      return;
    remove(t.id);
    toast.info(`Deleted "${t.title}"`);
  };

  const handleExport = () => {
    const blob = new Blob([exportAll()], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chats-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success('Exported chats');
  };

  const handleImportFile = async (f: File) => {
    const text = await f.text();
    const { imported, skipped } = importAll(text);
    if (imported > 0)
      toast.success(`Imported ${imported} thread${imported === 1 ? '' : 's'}`);
    if (skipped > 0 && imported === 0)
      toast.warning(`Skipped ${skipped} (duplicate ids)`);
  };

  // --------- collapsed rail ---------
  // Logo slot swaps Z ↔ expand-chevron based on hover over the whole
  // sidebar, not just the button itself. Below the logo sit minimal
  // action icons (+, settings, theme) so they're always reachable.
  if (collapsed) {
    return (
      <aside
        className="group/rail flex h-full w-12 shrink-0 flex-col items-center border-r py-3"
        style={{ background: 'var(--island-bg)', borderColor: 'var(--gray-100)' }}
      >
        <button
          type="button"
          onClick={onToggleCollapsed}
          aria-label="Expand sidebar"
          className="relative flex h-9 w-9 items-center justify-center rounded-lg"
          style={{ color: 'var(--gray-700)' }}
        >
          <img
            src="/static/z-symbol.png"
            alt="ZEDEDA"
            className="h-6 w-6 object-contain transition-opacity group-hover/rail:opacity-0"
          />
          <ChevronRight className="absolute h-5 w-5 opacity-0 transition-opacity group-hover/rail:opacity-100" />
        </button>

        <div className="mt-2 flex flex-col items-center gap-1">
          <RailIcon
            onClick={() => createAndActivate()}
            aria-label="New chat"
            icon={<Plus className="h-4 w-4" />}
          />
        </div>

        <div className="mt-auto flex flex-col items-center gap-1">
          <RailIcon
            onClick={onOpenSettings}
            aria-label="Settings"
            icon={<Settings className="h-4 w-4" />}
          />
          <RailThemeToggle />
        </div>
      </aside>
    );
  }

  // --------- expanded sidebar ---------
  return (
    <aside
      className="group/sidebar flex h-full w-72 shrink-0 flex-col border-r"
      style={{ background: 'var(--island-bg)', borderColor: 'var(--gray-100)' }}
    >
      {/* Brand row — logo + collapse button (collapse is hover-revealed). */}
      <div className="flex items-center justify-between gap-2 px-3 pb-2 pt-3">
        <div className="flex items-center gap-2 px-1">
          <img
            src="/static/logo-light.png"
            alt="ZEDEDA"
            className="block h-6 w-auto dark:hidden"
          />
          <img
            src="/static/logo-dark.png"
            alt="ZEDEDA"
            className="hidden h-6 w-auto dark:block"
          />
        </div>
        <button
          type="button"
          onClick={onToggleCollapsed}
          aria-label="Collapse sidebar"
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg opacity-0 transition-opacity group-hover/sidebar:opacity-100 focus-visible:opacity-100"
          style={{ color: 'var(--gray-500)' }}
        >
          <ChevronLeft className="h-4 w-4" />
        </button>
      </div>

      <div className="px-3 pb-2">
        <SidebarRow
          onClick={() => createAndActivate()}
          icon={<PlusCircle className="h-4 w-4" />}
          label="New chat"
          prominent
        />
      </div>

      <div className="px-3 pb-2">
        <div
          className="flex items-center gap-2 rounded-lg border px-2 py-1.5"
          style={{ borderColor: 'var(--gray-200)', background: 'var(--gray-50)' }}
        >
          <Search className="h-3.5 w-3.5" style={{ color: 'var(--gray-400)' }} />
          <input
            type="search"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setPage(1);
            }}
            placeholder="Search chats"
            className="flex-1 bg-transparent text-sm outline-none"
            style={{ color: 'var(--gray-900)' }}
          />
        </div>
      </div>

      {/* History section */}
      <button
        type="button"
        onClick={toggleHistory}
        className="mx-1 mb-1 flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-semibold uppercase tracking-wide"
        style={{ color: 'var(--gray-500)' }}
      >
        <ChevronDown
          className={clsx(
            'h-3 w-3 transition-transform',
            !historyOpen && '-rotate-90',
          )}
        />
        History
        <span className="ml-auto text-[10px] font-normal normal-case opacity-60">
          {threads.length > 0 && `${filtered.length} of ${threads.length}`}
        </span>
      </button>

      {historyOpen && (
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="min-h-0 flex-1 overflow-y-auto px-1">
            {pageItems.length === 0 ? (
              <p
                className="px-2 py-4 text-center text-xs"
                style={{ color: 'var(--gray-400)' }}
              >
                {threads.length === 0 ? 'No chats yet' : 'No matches'}
              </p>
            ) : (
              <ul className="flex flex-col">
                {pageItems.map((t) => (
                  <li key={t.id}>
                    <ThreadItem
                      thread={t}
                      active={t.id === activeId}
                      editing={editingId === t.id}
                      onSelect={() => setActive(t.id)}
                      onStartEdit={() => setEditingId(t.id)}
                      onFinishEdit={(next) => {
                        if (next !== null) rename(t.id, next);
                        setEditingId(null);
                      }}
                      onDelete={() => handleDelete(t)}
                    />
                  </li>
                ))}
              </ul>
            )}
          </div>

          {filtered.length > pageSize && (
            <div
              className="flex items-center justify-between gap-1 px-3 py-1 text-xs"
              style={{ color: 'var(--gray-500)' }}
            >
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={safePage === 1}
                  className="rounded px-1.5 py-0.5 disabled:opacity-40"
                >
                  Prev
                </button>
                <span>
                  {safePage} / {totalPages}
                </span>
                <button
                  type="button"
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={safePage === totalPages}
                  className="rounded px-1.5 py-0.5 disabled:opacity-40"
                >
                  Next
                </button>
              </div>
              <select
                value={pageSize}
                onChange={(e) => {
                  const n = normalizePageSize(e.target.value);
                  setPageSize(n);
                  localStorage.setItem(PAGE_KEY, String(n));
                  setPage(1);
                }}
                className="rounded border bg-transparent px-1 py-0.5"
                style={{ borderColor: 'var(--gray-200)' }}
              >
                {SUPPORTED_PAGE_SIZES.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
      )}

      {/* Footer: settings, theme, chat import/export */}
      <div
        className="mt-auto flex flex-col gap-1 border-t p-2"
        style={{ borderColor: 'var(--gray-100)' }}
      >
        <SidebarRow
          onClick={onOpenSettings}
          icon={<Settings className="h-4 w-4" />}
          label="Settings"
        />
        <ThemeRow />
        <div
          className="mt-1 grid grid-cols-2 gap-1 pt-1"
          style={{ borderTop: '1px dashed var(--gray-100)' }}
        >
          <SidebarRow
            onClick={handleExport}
            icon={<Download className="h-3.5 w-3.5" />}
            label="Export"
            dense
          />
          <SidebarRow
            onClick={() => fileRef.current?.click()}
            icon={<Upload className="h-3.5 w-3.5" />}
            label="Import"
            dense
          />
        </div>
        <input
          ref={fileRef}
          type="file"
          accept="application/json"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void handleImportFile(f);
            e.target.value = '';
          }}
        />
      </div>
    </aside>
  );
}

// ------------------ building blocks ------------------

function SidebarRow({
  onClick,
  icon,
  label,
  active,
  prominent,
  dense,
}: {
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
  active?: boolean;
  prominent?: boolean;
  dense?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx(
        'group flex w-full items-center gap-2 rounded-lg text-left transition-colors',
        dense ? 'px-2 py-1 text-xs' : 'px-2 py-1.5 text-sm',
        active
          ? 'bg-[var(--primary-10)]'
          : 'hover:bg-black/5 dark:hover:bg-white/5',
      )}
      style={{
        color: prominent
          ? 'var(--gray-900)'
          : active
            ? 'var(--zededa-cyan)'
            : 'var(--gray-700)',
      }}
    >
      <span
        className="flex h-5 w-5 items-center justify-center"
        style={{
          color: prominent
            ? 'var(--zededa-cyan)'
            : active
              ? 'var(--zededa-cyan)'
              : 'var(--gray-500)',
        }}
      >
        {icon}
      </span>
      <span className={clsx('truncate', prominent && 'font-medium')}>
        {label}
      </span>
    </button>
  );
}

function RailIcon({
  onClick,
  icon,
  ...rest
}: {
  onClick: () => void;
  icon: React.ReactNode;
  'aria-label': string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex h-9 w-9 items-center justify-center rounded-lg"
      style={{ color: 'var(--gray-500)' }}
      {...rest}
    >
      {icon}
    </button>
  );
}

function RailThemeToggle() {
  const [theme, setTheme] = useState<'light' | 'dark'>(
    () =>
      (document.documentElement.dataset.theme as 'light' | 'dark') || 'light',
  );
  const toggle = () => {
    const next = theme === 'light' ? 'dark' : 'light';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('theme', next);
    setTheme(next);
  };
  return (
    <RailIcon
      onClick={toggle}
      aria-label={theme === 'light' ? 'Dark mode' : 'Light mode'}
      icon={
        theme === 'light' ? (
          <Moon className="h-4 w-4" />
        ) : (
          <Sun className="h-4 w-4" />
        )
      }
    />
  );
}

function ThemeRow() {
  const [theme, setTheme] = useState<'light' | 'dark'>(
    () => (document.documentElement.dataset.theme as 'light' | 'dark') || 'light',
  );
  const toggle = () => {
    const next = theme === 'light' ? 'dark' : 'light';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('theme', next);
    setTheme(next);
  };
  return (
    <SidebarRow
      onClick={toggle}
      icon={
        theme === 'light' ? (
          <Moon className="h-4 w-4" />
        ) : (
          <Sun className="h-4 w-4" />
        )
      }
      label={theme === 'light' ? 'Dark mode' : 'Light mode'}
    />
  );
}

function ThreadItem({
  thread,
  active,
  editing,
  onSelect,
  onStartEdit,
  onFinishEdit,
  onDelete,
}: {
  thread: Thread;
  active: boolean;
  editing: boolean;
  onSelect: () => void;
  onStartEdit: () => void;
  onFinishEdit: (next: string | null) => void;
  onDelete: () => void;
}) {
  const [draft, setDraft] = useState(thread.title);

  return (
    <div
      className={clsx(
        'group flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm transition-colors',
        active ? 'bg-[var(--gray-100)]' : 'hover:bg-black/5 dark:hover:bg-white/5',
      )}
    >
      <MessageSquare
        className="h-3.5 w-3.5 shrink-0"
        style={{ color: active ? 'var(--zededa-cyan)' : 'var(--gray-400)' }}
      />
      {editing ? (
        <input
          autoFocus
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={() => onFinishEdit(draft)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') onFinishEdit(draft);
            if (e.key === 'Escape') onFinishEdit(null);
          }}
          className="flex-1 rounded border bg-transparent px-1 py-0.5 text-sm outline-none"
          style={{ borderColor: 'var(--zededa-cyan-border)' }}
        />
      ) : (
        <button
          type="button"
          onClick={onSelect}
          className="flex-1 truncate text-left"
          style={{ color: active ? 'var(--gray-900)' : 'var(--gray-700)' }}
          title={thread.title}
        >
          {thread.title}
        </button>
      )}
      {!editing && (
        <div className="flex items-center opacity-0 transition-opacity group-hover:opacity-100">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              setDraft(thread.title);
              onStartEdit();
            }}
            aria-label="Rename"
            className="rounded p-1"
            style={{ color: 'var(--gray-500)' }}
          >
            <Pencil className="h-3 w-3" />
          </button>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
            aria-label="Delete"
            className="rounded p-1"
            style={{ color: 'var(--gray-500)' }}
          >
            <Trash2 className="h-3 w-3" />
          </button>
        </div>
      )}
    </div>
  );
}
