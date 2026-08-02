import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Check,
  CheckCircle2,
  Download,
  Loader2,
  Play,
  Plus,
  RefreshCw,
  ShieldCheck,
  Trash2,
  Upload,
} from 'lucide-react';
import { llmApi, type Credential, type RouterStatus } from '../../lib/api';
import { Modal } from '../ui/Modal';
import { StatusDot } from '../ui/StatusDot';
import { useToast } from '../ui/Toast';

interface Props {
  open: boolean;
  onClose: () => void;
  /** Called after any mutation so the header badge refetches. */
  onChange?: () => void;
}

export function SettingsModal({ open, onClose, onChange }: Props) {
  const toast = useToast();
  const [creds, setCreds] = useState<Credential[]>([]);
  const [router, setRouter] = useState<RouterStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [showAdd, setShowAdd] = useState(false);
  const importRef = useRef<HTMLInputElement>(null);

  const reload = useCallback(async () => {
    setLoading(true);
    try {
      const [c, r] = await Promise.all([
        llmApi.listCredentials(),
        llmApi.routerStatus(),
      ]);
      setCreds(c.credentials ?? []);
      setRouter(r);
    } catch (e) {
      toast.error(`Failed to load: ${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    if (open) void reload();
  }, [open, reload]);

  // /llm/status returns `active_provider` as the full provider dict, not a
  // string. Extract the name for display + comparisons. Defensive in case
  // a future payload changes shape.
  const activeName =
    typeof router?.active_provider === 'string'
      ? (router.active_provider as string)
      : (router?.active_provider?.name ?? null);

  // When the deployment injects EIP_ACCESS_TOKEN, the router
  // auto-registers an "edgeai-builtin" openai-compatible provider tagged
  // `metadata.builtin = true`. When this provider is present, the agent
  // works out of the box with no user API key required — we surface a
  // banner and treat custom credentials as optional fallbacks.
  const builtinProvider = useMemo(
    () =>
      router?.providers?.find(
        (p) => p.metadata?.builtin === true,
      ) ?? null,
    [router],
  );
  const isBuiltinActive =
    builtinProvider != null && builtinProvider.name === activeName;

  const activate = async (name: string) => {
    try {
      await llmApi.activateCredential(name);
      toast.success(`Activated ${name}`);
      onChange?.();
      await reload();
    } catch (e) {
      toast.error(`Activate failed: ${(e as Error).message}`);
    }
  };

  const remove = async (name: string) => {
    if (!window.confirm(`Delete credential "${name}"?`)) return;
    try {
      await llmApi.deleteCredential(name);
      toast.info(`Deleted ${name}`);
      onChange?.();
      await reload();
    } catch (e) {
      toast.error(`Delete failed: ${(e as Error).message}`);
    }
  };

  const exportCreds = async () => {
    try {
      const resp = await llmApi.exportCredentials();
      // Save only the portable `bundle` — the rest is response metadata.
      const blob = new Blob([JSON.stringify(resp.bundle, null, 2)], {
        type: 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `credentials-${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
      URL.revokeObjectURL(url);
      if (resp.warning) toast.warning(resp.warning);
      else toast.success(`Exported ${resp.credential_count} credential(s)`);
    } catch (e) {
      toast.error(`Export failed: ${(e as Error).message}`);
    }
  };

  const importCreds = async (file: File) => {
    try {
      const parsed = JSON.parse(await file.text());
      const res = await llmApi.importCredentials(parsed);
      if (res.imported_count > 0) await llmApi.activateAll();
      const bits: string[] = [];
      if (res.imported_count)
        bits.push(`imported ${res.imported_count}`);
      if (res.skipped_count) bits.push(`skipped ${res.skipped_count}`);
      if (res.error_count) bits.push(`${res.error_count} error(s)`);
      toast.success(bits.join(' · ') || 'Nothing to import');
      onChange?.();
      await reload();
    } catch (e) {
      toast.error(`Import failed: ${(e as Error).message}`);
    }
  };

  return (
    <Modal
      open={open}
      onClose={onClose}
      wide
      title="Settings · LLM providers"
      footer={
        <>
          <button
            type="button"
            onClick={() => reload()}
            className="flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-sm"
            style={{
              borderColor: 'var(--gray-200)',
              color: 'var(--gray-600)',
            }}
          >
            <RefreshCw className="h-3.5 w-3.5" /> Reload
          </button>
          <button
            type="button"
            onClick={onClose}
            className="rounded-full px-4 py-1.5 text-sm font-medium"
            style={{
              background: 'var(--zededa-cyan)',
              color: '#000',
            }}
          >
            Done
          </button>
        </>
      }
    >
      {loading && (
        <div className="mb-3 flex items-center gap-2 text-xs" style={{ color: 'var(--gray-500)' }}>
          <Loader2 className="h-3 w-3 animate-spin" /> Loading…
        </div>
      )}

      {builtinProvider && (
        <section
          className="mb-5 rounded-lg border px-3 py-3"
          style={{
            borderColor: 'var(--zededa-cyan-border)',
            background: 'var(--primary-10)',
          }}
        >
          <div className="flex items-start gap-2">
            <ShieldCheck
              className="mt-0.5 h-4 w-4 shrink-0"
              style={{ color: 'var(--zededa-cyan)' }}
            />
            <div className="min-w-0 flex-1">
              <div
                className="text-sm font-semibold"
                style={{ color: 'var(--gray-900)' }}
              >
                EdgeAI built-in LLM
                {isBuiltinActive ? ' · active' : ' · available'}
              </div>
              <p
                className="mt-0.5 text-xs"
                style={{ color: 'var(--gray-600)' }}
              >
                This deployment ships with a managed OpenAI-compatible
                endpoint{builtinProvider.model ? ` (${builtinProvider.model})` : ''}
                {' '}authenticated by the platform — no API key needed. You
                can still register your own provider below to use as a
                fallback.
              </p>
            </div>
          </div>
        </section>
      )}

      {/* Router status */}
      <section className="mb-5">
        <h3
          className="mb-2 text-xs font-semibold uppercase tracking-wide"
          style={{ color: 'var(--gray-500)' }}
        >
          Router
        </h3>
        <div
          className="flex flex-wrap items-center gap-3 rounded-lg border px-3 py-2"
          style={{
            borderColor: 'var(--gray-200)',
            background: 'var(--gray-50)',
          }}
        >
          <StatusDot state={activeName ? 'active' : 'offline'} />
          <span className="text-sm" style={{ color: 'var(--gray-800)' }}>
            {activeName ? (
              <>
                Active: <strong>{activeName}</strong>
              </>
            ) : (
              'No active provider'
            )}
          </span>
          <span className="text-xs" style={{ color: 'var(--gray-500)' }}>
            {router?.providers?.length ?? 0} registered
            {router?.routing_strategy
              ? ` · strategy ${router.routing_strategy}`
              : ''}
          </span>
        </div>
      </section>

      {/* Credentials list */}
      <section className="mb-5">
        <div className="mb-2 flex items-center justify-between">
          <h3
            className="text-xs font-semibold uppercase tracking-wide"
            style={{ color: 'var(--gray-500)' }}
          >
            Credentials
          </h3>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setShowAdd((v) => !v)}
              className="flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs"
              style={{
                borderColor: 'var(--zededa-cyan-border)',
                color: 'var(--zededa-cyan)',
                background: 'var(--primary-10)',
              }}
            >
              <Plus className="h-3 w-3" /> Add
            </button>
            <button
              type="button"
              onClick={exportCreds}
              className="flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs"
              style={{ borderColor: 'var(--gray-200)', color: 'var(--gray-600)' }}
              title="Export credentials"
            >
              <Download className="h-3 w-3" /> Export
            </button>
            <button
              type="button"
              onClick={() => importRef.current?.click()}
              className="flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs"
              style={{ borderColor: 'var(--gray-200)', color: 'var(--gray-600)' }}
              title="Import credentials"
            >
              <Upload className="h-3 w-3" /> Import
            </button>
            <input
              ref={importRef}
              type="file"
              accept="application/json"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void importCreds(f);
                e.target.value = '';
              }}
            />
          </div>
        </div>

        {creds.length === 0 ? (
          <p
            className="rounded-lg border border-dashed px-3 py-6 text-center text-sm"
            style={{ borderColor: 'var(--gray-200)', color: 'var(--gray-500)' }}
          >
            {builtinProvider
              ? 'No additional credentials configured. The built-in EdgeAI provider above is active — adding one here is optional.'
              : 'No credentials yet. Add one to enable the agent.'}
          </p>
        ) : (
          <ul className="flex flex-col gap-2">
            {creds.map((c) => (
              <li
                key={c.name}
                className="hover-ring flex flex-wrap items-center gap-3 r-card border px-3 py-2 sm:flex-nowrap"
                style={{
                  borderColor:
                    c.name === activeName
                      ? 'var(--zededa-cyan-border)'
                      : 'var(--gray-200)',
                  background:
                    c.name === activeName
                      ? 'var(--primary-10)'
                      : 'var(--island-bg)',
                }}
              >
                <div className="min-w-0 flex-1">
                  <div
                    className="flex items-center gap-2 truncate text-sm font-medium"
                    style={{ color: 'var(--gray-900)' }}
                  >
                    {c.name}
                    {c.name === activeName && (
                      <span
                        className="inline-flex items-center gap-1 rounded-full px-1.5 py-0.5 text-[10px] font-semibold uppercase"
                        style={{
                          background: 'var(--color-success-light)',
                          color: 'var(--color-success)',
                        }}
                      >
                        <CheckCircle2 className="h-2.5 w-2.5" /> Active
                      </span>
                    )}
                  </div>
                  <div
                    className="truncate text-xs"
                    style={{ color: 'var(--gray-500)' }}
                  >
                    {c.provider_type} · {c.model ?? 'no model'}{' '}
                    {c.url ? `· ${c.url}` : ''}
                  </div>
                </div>
                {c.name !== activeName && (
                  <button
                    type="button"
                    onClick={() => activate(c.name)}
                    className="flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs"
                    style={{
                      borderColor: 'var(--zededa-cyan-border)',
                      color: 'var(--zededa-cyan)',
                    }}
                  >
                    <Play className="h-3 w-3" /> Activate
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => remove(c.name)}
                  className="rounded-full border p-1.5"
                  style={{
                    borderColor: 'var(--gray-200)',
                    color: 'var(--gray-500)',
                  }}
                  aria-label="Delete"
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      {showAdd && (
        <AddCredentialForm
          onCancel={() => setShowAdd(false)}
          onSaved={async () => {
            setShowAdd(false);
            onChange?.();
            await reload();
          }}
        />
      )}
    </Modal>
  );
}

function AddCredentialForm({
  onCancel,
  onSaved,
}: {
  onCancel: () => void;
  onSaved: () => void;
}) {
  const toast = useToast();
  const [name, setName] = useState('');
  const [url, setUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('');
  const [models, setModels] = useState<string[] | null>(null);
  const [fetching, setFetching] = useState(false);
  const [saving, setSaving] = useState(false);
  const [supportsTools, setSupportsTools] = useState(true);
  const [providerType, setProviderType] = useState('auto');

  const detectedType = useMemo(() => {
    if (providerType !== 'auto') return providerType;
    const u = url.toLowerCase();
    if (!u) return 'anthropic';
    if (u.includes('anthropic.com')) return 'anthropic';
    if (u.includes('openai.com')) return 'openai';
    if (u.includes('googleapis.com')) return 'google';
    if (u.includes('groq.com')) return 'groq';
    if (u.includes('11434') || u.includes('ollama')) return 'ollama';
    return 'openai-compatible';
  }, [url, providerType]);

  const fetchModels = async () => {
    setFetching(true);
    try {
      const { models: list } = await llmApi.fetchModels({
        provider_type: detectedType,
        url: url || undefined,
        api_key: apiKey || undefined,
      });
      setModels(list);
      if (list.length === 0) toast.warning('No models returned');
    } catch (e) {
      toast.error(`Fetch models failed: ${(e as Error).message}`);
    } finally {
      setFetching(false);
    }
  };

  const save = async () => {
    if (!name.trim()) {
      toast.warning('Name is required');
      return;
    }
    setSaving(true);
    try {
      await llmApi.saveCredential({
        name: name.trim(),
        provider_type: providerType === 'auto' ? undefined : providerType,
        url: url.trim() || undefined,
        api_key: apiKey.trim() || undefined,
        model: model.trim() || undefined,
        supports_tools: supportsTools,
        enabled: true,
      });
      await llmApi.activateCredential(name.trim());
      toast.success(`Saved & activated ${name.trim()}`);
      onSaved();
    } catch (e) {
      toast.error(`Save failed: ${(e as Error).message}`);
    } finally {
      setSaving(false);
    }
  };

  return (
    <section
      className="rounded-lg border p-4"
      style={{
        borderColor: 'var(--gray-200)',
        background: 'var(--gray-50)',
      }}
    >
      <h3 className="mb-3 text-sm font-semibold" style={{ color: 'var(--gray-900)' }}>
        New credential
      </h3>
      <div className="grid gap-3 sm:grid-cols-2">
        <Field label="Name">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. padraig-key"
            className="form-input"
          />
        </Field>
        <Field label="Provider type">
          <select
            value={providerType}
            onChange={(e) => setProviderType(e.target.value)}
            className="form-input"
          >
            <option value="auto">Auto from URL ({detectedType})</option>
            <option value="anthropic">Anthropic</option>
            <option value="openai">OpenAI</option>
            <option value="google">Google</option>
            <option value="groq">Groq</option>
            <option value="ollama">Ollama</option>
            <option value="openai-compatible">OpenAI-compatible</option>
          </select>
        </Field>
        <Field label="Server URL" hint="Optional for Anthropic/OpenAI native">
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://api.anthropic.com"
            className="form-input"
          />
        </Field>
        <Field label="API key">
          <input
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            type="password"
            placeholder="sk-…"
            className="form-input"
          />
        </Field>
        <Field label="Model" className="sm:col-span-2">
          <div className="flex gap-2">
            {models && models.length > 0 ? (
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="form-input flex-1"
              >
                <option value="">Select a model…</option>
                {models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            ) : (
              <input
                value={model}
                onChange={(e) => setModel(e.target.value)}
                placeholder="claude-sonnet-4-6"
                className="form-input flex-1"
              />
            )}
            <button
              type="button"
              disabled={fetching}
              onClick={fetchModels}
              className="flex items-center gap-1 rounded-full border px-3 text-xs"
              style={{
                borderColor: 'var(--zededa-cyan-border)',
                color: 'var(--zededa-cyan)',
                background: 'var(--primary-10)',
              }}
            >
              {fetching ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <RefreshCw className="h-3 w-3" />
              )}
              Fetch
            </button>
          </div>
        </Field>
      </div>
      <label
        className="mt-3 flex items-center gap-2 text-xs"
        style={{ color: 'var(--gray-600)' }}
      >
        <input
          type="checkbox"
          checked={supportsTools}
          onChange={(e) => setSupportsTools(e.target.checked)}
          style={{ accentColor: 'var(--zededa-cyan)' }}
        />
        Supports tool calling
      </label>
      <div className="mt-4 flex justify-end gap-2">
        <button
          type="button"
          onClick={onCancel}
          className="rounded-full border px-3 py-1.5 text-sm"
          style={{ borderColor: 'var(--gray-200)', color: 'var(--gray-600)' }}
        >
          Cancel
        </button>
        <button
          type="button"
          disabled={saving}
          onClick={save}
          className="flex items-center gap-1.5 rounded-full px-4 py-1.5 text-sm font-medium"
          style={{
            background: 'var(--zededa-cyan)',
            color: '#000',
          }}
        >
          {saving ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Check className="h-3.5 w-3.5" />
          )}
          Save & activate
        </button>
      </div>
    </section>
  );
}

function Field({
  label,
  hint,
  className,
  children,
}: {
  label: string;
  hint?: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <label className={className}>
      <div
        className="mb-1 flex items-center justify-between text-[11px] font-semibold uppercase tracking-wide"
        style={{ color: 'var(--gray-500)' }}
      >
        <span>{label}</span>
        {hint && <span className="font-normal normal-case">{hint}</span>}
      </div>
      {children}
    </label>
  );
}
