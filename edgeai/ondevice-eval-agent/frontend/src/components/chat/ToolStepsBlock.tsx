import { useState } from 'react';
import {
  Activity,
  AlertCircle,
  Boxes,
  ChevronDown,
  Compass,
  FileText,
  Hammer,
  Loader2,
  ScanSearch,
  Sparkles,
  Wrench,
  Workflow,
} from 'lucide-react';
import clsx from 'clsx';
import type { ToolCall } from '../../lib/types';

/**
 * Grouped timeline of tool calls — renders like Claude's "Completed N
 * steps" UI: a single collapsible header with a vertical line running
 * through per-step markers. Replaces the flat list of per-tool cards.
 */
export function ToolStepsBlock({
  toolCalls,
  isStreaming,
}: {
  toolCalls: ToolCall[];
  isStreaming: boolean;
}) {
  const [open, setOpen] = useState(true);
  if (toolCalls.length === 0) return null;

  const running = toolCalls.some((t) => t.status === 'running');
  const label = running
    ? `Working on ${toolCalls.length} step${toolCalls.length === 1 ? '' : 's'}`
    : `Completed ${toolCalls.length} step${toolCalls.length === 1 ? '' : 's'}`;

  return (
    <div className="text-sm">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="-ml-1 inline-flex items-center gap-1.5 rounded-md px-1.5 py-0.5 text-sm font-medium"
        style={{ color: 'var(--gray-700)' }}
      >
        {running && (
          <Loader2
            className="h-3.5 w-3.5 animate-spin"
            style={{ color: 'var(--zededa-cyan)' }}
          />
        )}
        <span>{label}</span>
        <ChevronDown
          className={clsx(
            'h-3.5 w-3.5 transition-transform',
            !open && '-rotate-90',
          )}
          style={{ color: 'var(--gray-400)' }}
        />
      </button>

      {open && (
        <div className="relative ml-1.5 mt-1 pl-6">
          {/* Vertical timeline line — runs between first and last marker. */}
          <span
            aria-hidden
            className="absolute top-3 bottom-3 w-px"
            style={{ left: '12px', background: 'var(--gray-200)' }}
          />
          {toolCalls.map((tc, i) => (
            <ToolStep
              key={tc.id}
              tool={tc}
              isLast={i === toolCalls.length - 1}
              streaming={isStreaming}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Single tool step. Exported so AssistantMessage can render it inline
 * between text blocks — with `inline=true` the absolute marker is
 * replaced by an inline icon so it doesn't depend on an ancestor
 * providing the timeline vertical line.
 */
export function ToolStep({
  tool,
  isLast: _isLast,
  streaming: _streaming,
  inline = false,
}: {
  tool: ToolCall;
  isLast: boolean;
  streaming: boolean;
  inline?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const visual = toolVisual(tool.name);
  const Icon = visual.icon;

  const running = tool.status === 'running';
  const errored = tool.status === 'error';

  const markerStyle = {
    background: 'var(--island-bg)',
    border: `1.5px solid ${running ? 'var(--zededa-cyan-border)' : errored ? 'rgba(239,68,68,0.4)' : 'var(--gray-200)'}`,
    color: running
      ? 'var(--zededa-cyan)'
      : errored
        ? 'var(--color-error)'
        : visual.color,
  };

  const markerInner = running ? (
    <Loader2 className="h-3 w-3 animate-spin" />
  ) : errored ? (
    <AlertCircle className="h-3 w-3" />
  ) : (
    <Icon className="h-3 w-3" />
  );

  return (
    <div className={clsx('relative', inline ? 'py-1' : 'py-1.5')}>
      {inline ? (
        // Inline form: icon sits next to the label, no timeline line.
        // Used by AssistantMessage when rendering tool blocks interleaved
        // with text so the tool doesn't need an ancestor container
        // providing the vertical line.
        <span
          aria-hidden
          className="mr-2 inline-flex h-5 w-5 items-center justify-center rounded-full align-[-4px]"
          style={markerStyle}
        >
          {markerInner}
        </span>
      ) : (
        // Absolute-positioned marker, vertically centered on its row.
        // Parent container has pl-6 (24px) and the timeline line sits
        // at left:12px, so a 20px marker at left:2px has its centre at
        // 12px — exactly on the line.
        <span
          aria-hidden
          className="absolute top-1/2 z-[1] flex h-5 w-5 -translate-y-1/2 items-center justify-center rounded-full"
          style={{ left: '-22px', ...markerStyle }}
        >
          {markerInner}
        </span>
      )}

      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1.5 rounded-md py-0.5 text-sm"
        style={{ color: 'var(--gray-700)' }}
      >
        <span className="font-medium">{prettyName(tool.name)}</span>
        <ChevronDown
          className={clsx(
            'h-3 w-3 transition-transform',
            !open && '-rotate-90',
          )}
          style={{ color: 'var(--gray-400)' }}
        />
      </button>

      {/* If this tool returned an image, always render it inline — hiding
          it behind the "expand JSON" toggle makes the most useful part of
          the output invisible by default (e.g. inference result overlays,
          view_image output, DETR visualizations). */}
      {(() => {
        const img = extractImageFromToolResult(tool.result);
        if (!img) return null;
        return (
          <div
            className="mt-2 overflow-hidden rounded-lg border"
            style={{
              borderColor: 'var(--gray-200)',
              background: 'var(--island-bg)',
              maxWidth: '640px',
            }}
          >
            <img
              src={img.src}
              alt={img.alt}
              className="block h-auto w-full"
              loading="lazy"
            />
            {img.caption && (
              <div
                className="px-3 py-1.5 text-[11px]"
                style={{
                  color: 'var(--gray-600)',
                  borderTop: '1px solid var(--gray-200)',
                }}
              >
                {img.caption}
              </div>
            )}
          </div>
        );
      })()}

      {open && (
        <div
          className="mt-1.5 space-y-2 rounded-lg border px-3 py-2 text-xs font-mono"
          style={{
            background: 'var(--gray-50)',
            borderColor: 'var(--gray-200)',
            color: 'var(--gray-700)',
          }}
        >
          {tool.args && Object.keys(tool.args).length > 0 && (
            <DetailRow label="Arguments" value={formatJson(tool.args)} />
          )}
          {tool.result !== undefined && (
            <DetailRow
              label="Result"
              value={formatJson(stripImageBase64(tool.result))}
            />
          )}
          {tool.args === undefined && tool.result === undefined && (
            <div style={{ color: 'var(--gray-500)' }}>
              {running ? 'Running…' : 'No details captured.'}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface ExtractedImage {
  src: string;
  alt: string;
  caption?: string;
}

/**
 * Look for an image payload in a tool result. Handles:
 *   { image_base64, mime_type, message? }           // view_image, run_inference
 *   { image, mime_type }                            // generic fallback
 *   { visualization: { image_base64, mime_type } }  // nested helpers
 * Returns null when no image is present.
 */
function extractImageFromToolResult(result: unknown): ExtractedImage | null {
  if (!result || typeof result !== 'object') return null;
  const r = result as Record<string, unknown>;

  // Recurse into common nested containers so we don't miss images that
  // live under `visualization` / `image` / `data`.
  for (const nested of ['visualization', 'image', 'data', 'output']) {
    const v = r[nested];
    if (v && typeof v === 'object') {
      const found = extractImageFromToolResult(v);
      if (found) return found;
    }
  }

  const b64 =
    (typeof r.image_base64 === 'string' && r.image_base64) ||
    (typeof r.image === 'string' && r.image) ||
    (typeof r.base64 === 'string' && r.base64);
  if (!b64) return null;

  const mime =
    (typeof r.mime_type === 'string' && r.mime_type) ||
    (typeof r.mimetype === 'string' && r.mimetype) ||
    'image/png';
  const src = b64.startsWith('data:') ? b64 : `data:${mime};base64,${b64}`;
  const alt =
    (typeof r.description === 'string' && r.description) ||
    (typeof r.message === 'string' && r.message) ||
    'Tool output image';
  const caption =
    typeof r.message === 'string' && r.message !== alt ? r.message : undefined;

  return { src, alt, caption };
}

/**
 * Return a copy of a tool result with any base64 image payloads replaced
 * with a short marker, so the raw JSON view stays readable. We only strip
 * the top-level and one level of nesting; this matches extractImageFromToolResult.
 */
function stripImageBase64(result: unknown): unknown {
  if (!result || typeof result !== 'object') return result;
  const src = result as Record<string, unknown>;
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(src)) {
    if (
      (k === 'image_base64' || k === 'image' || k === 'base64') &&
      typeof v === 'string' &&
      v.length > 200
    ) {
      out[k] = `[${v.length} chars of base64 — rendered above]`;
      continue;
    }
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      out[k] = stripImageBase64(v);
    } else {
      out[k] = v;
    }
  }
  return out;
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div
        className="mb-1 text-[10px] font-semibold uppercase tracking-wider"
        style={{ color: 'var(--gray-500)' }}
      >
        {label}
      </div>
      <pre
        className="m-0 max-h-60 overflow-auto rounded px-2 py-1.5 text-[10.5px] leading-5"
        style={{
          background: 'var(--island-bg)',
          color: 'var(--gray-800)',
          border: '1px solid var(--gray-200)',
        }}
      >
        {value}
      </pre>
    </div>
  );
}

function formatJson(v: unknown): string {
  if (v == null) return '';
  if (typeof v === 'string') return v;
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}

// ------------- tool icon + label mapping -------------

interface Visual {
  color: string;
  icon: typeof Sparkles;
}

function toolVisual(name: string): Visual {
  const n = name.toLowerCase();
  if (n.includes('list') && n.includes('model'))
    return { color: '#5B8DEF', icon: Boxes };
  if (n.includes('analyze') && n.includes('model'))
    return { color: '#A855F7', icon: ScanSearch };
  if (n.includes('metadata')) return { color: '#14B8A6', icon: FileText };
  if (n.includes('input')) return { color: '#06B6D4', icon: Workflow };
  if (n.includes('output') || n.includes('interpret'))
    return { color: '#F59E0B', icon: Compass };
  if (n.includes('integration') || n.includes('frontend'))
    return { color: '#EC4899', icon: Hammer };
  if (n.includes('recommend') || n.includes('next'))
    return { color: '#10B981', icon: Sparkles };
  if (n.includes('predict') || n.includes('infer'))
    return { color: '#6366F1', icon: Activity };
  return { color: '#6B7280', icon: Wrench };
}

function prettyName(raw: string): string {
  const overrides: Record<string, string> = {
    list_available_models: 'Listing available models',
    get_model_metadata: 'Fetching model metadata',
    analyze_model_type: 'Analysing model type',
    get_model_input_requirements: 'Checking input requirements',
    get_model_output_interpretation: 'Interpreting model output',
    get_frontend_integration_guide: 'Writing integration snippet',
    recommend_next_steps: 'Recommending next steps',
    get_server_status: 'Checking server status',
  };
  if (overrides[raw]) return overrides[raw];
  return raw.replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase());
}
