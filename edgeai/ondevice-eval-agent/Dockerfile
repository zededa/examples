# Build context: ondevice-eval-agent/
#
# Multi-stage build:
#   stage 1 (node)   — compile the React SPA (frontend/ → dist/)
#   stage 2 (python) — install deps, copy backend, drop SPA dist into webapp/spa/
#
# The Python runtime serves both the API and the built SPA on :8080,
# so the whole app is a single image and a single port.

# ---------- Stage 1: build the React SPA ----------
# Pinned to BUILDPLATFORM so multi-arch builds compile the SPA natively
# (output is static JS/HTML/CSS — arch-neutral) instead of via qemu, which
# can crash esbuild.
FROM --platform=$BUILDPLATFORM node:20-alpine AS spa-builder

# Pin pnpm to a version that still supports Node 20. Without this, corepack
# auto-fetches the newest pnpm (11+) which requires Node 22's built-in
# `node:sqlite` and crashes on cold install.
RUN corepack enable && corepack prepare pnpm@9.15.0 --activate

WORKDIR /spa

COPY frontend/package.json ./
RUN pnpm install --no-frozen-lockfile

COPY frontend/ ./
# Same-origin API — the Flask server serves the SPA and the /agent/* routes.
ENV VITE_API_BASE=""
RUN pnpm build


# ---------- Stage 2: Python runtime ----------
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY client/ client/
COPY webapp/ webapp/

# Drop the built SPA where core.py (SPA_DIST) expects it.
COPY --from=spa-builder /spa/dist/ webapp/spa/

ENV MAX_STARTUP_WAIT=300 \
    HEALTH_CHECK_INTERVAL=10 \
    SPA_DIST=/app/webapp/spa \
    PYTHONUNBUFFERED=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8080/agent/status || exit 1

CMD ["python", "webapp/app.py"]
