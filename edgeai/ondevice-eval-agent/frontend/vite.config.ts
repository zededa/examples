import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * Dev server proxies API calls to the Flask backend.
 *
 * Proxy target resolves from DEV_API_TARGET (env). When running inside
 * Docker Desktop, set DEV_API_TARGET=http://host.docker.internal:8080
 * so the container can reach a Flask backend running on the host.
 * Outside Docker, the default localhost:8080 is fine.
 */
const API_TARGET = process.env.DEV_API_TARGET ?? 'http://localhost:8080';

export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // bind 0.0.0.0 so `docker run -p 5173:5173` works
    port: 5173,
    strictPort: true,
    proxy: {
      '/agent': { target: API_TARGET, changeOrigin: true },
      '/llm': { target: API_TARGET, changeOrigin: true },
      '/core': { target: API_TARGET, changeOrigin: true },
      '/eval': { target: API_TARGET, changeOrigin: true },
      '/static': { target: API_TARGET, changeOrigin: true },
      '/health': { target: API_TARGET, changeOrigin: true },
      '/server-info': { target: API_TARGET, changeOrigin: true },
      '/models': { target: API_TARGET, changeOrigin: true },
      '/predict': { target: API_TARGET, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
});
