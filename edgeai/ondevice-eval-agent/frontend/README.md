# ondevice-eval-agent — frontend

React + TypeScript + Vite + Tailwind. Replaces the Jinja + vanilla JS UI in
`webapp/templates/` and `webapp/static/js/` with a proper SPA that consumes
the Flask backend's existing SSE stream at `POST /agent/chat/stream`.

Design tokens (colors, typography, radii, shadows) are ported verbatim from
`webapp/static/css/variables.css` into `src/index.css` and `tailwind.config.js`,
so the UI shares the ZEDEDA EPI theme with the legacy app.

## Dev

```bash
# in ondevice-eval-agent/
python webapp/app.py    # Flask on :8080

# in ondevice-eval-agent/frontend/
pnpm install            # or npm install
pnpm dev                # Vite on :5173, proxies /agent /llm /core /eval /static → :8080
```

Open http://localhost:5173.

## Build

```bash
pnpm build   # → dist/
```

Serve `dist/` from any static host, or have Flask serve it. Set
`VITE_API_BASE` at build time if the API is on a different origin.

## SSE event contract

Mirrors `webapp/routes/agent.py::_generate_sse_events`:

| event         | payload                                                    |
|---------------|------------------------------------------------------------|
| `start`       | `{ session_id, warnings? }`                                |
| `warning`     | `{ has_warnings, ... }`                                    |
| (default)     | `{ token: string }` — streaming token chunk                |
| `tool_start`  | `{ name, id }`                                             |
| `tool_end`    | `{ name, result }`                                         |
| `done`        | `{ response, tool_calls, finish_reason, meta, success }`   |
| `complete`    | same shape as `done`, used when streaming unavailable      |
| `error`       | `{ error, limit_exceeded?, enabled? }`                     |

Parsed in `src/lib/sse.ts`; reduced into `ChatMessage[]` in
`src/hooks/useStreamingChat.ts`.

## Layout

```
src/
  App.tsx                    — screen: Header + ChatThread + Composer
  index.css                  — EPI tokens + prose + hljs
  lib/
    api.ts                   — fetch wrappers
    sse.ts                   — fetch-based SSE parser
    types.ts                 — ChatMessage, ToolCall, AgentStatus
  hooks/
    useStreamingChat.ts      — send/stop/reset + reducer for SSE events
  components/
    layout/Header.tsx
    ui/{Avatar,AutoResizeTextarea,ThemeToggle}.tsx
    chat/
      ChatThread.tsx         — message list + auto-scroll
      Composer.tsx           — input + send/stop
      WelcomeScreen.tsx      — empty state + suggestion pills
      UserMessage.tsx
      AssistantMessage.tsx   — combines tool cards + markdown + cursor
      InlineToolCard.tsx     — per-tool color, expandable args/result
      MarkdownRenderer.tsx   — react-markdown + GFM + highlight
      CodeBlock.tsx          — code header + copy
      TypingIndicator.tsx
```
