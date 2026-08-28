import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

// `npm run build` bundles everything into dist/index.html and copies it to
// ../dashboard.html, the single self-contained file the Makefile embeds into
// ds4-server (dashboard_html.h). Commit the built file so `make` needs no node. Dev: `npm run dev` proxies /stats
// to a running server on :8000.
export default defineConfig({
  plugins: [react(), viteSingleFile()],
  build: { outDir: "dist" },
  server: { proxy: { "/stats": "http://127.0.0.1:8000", "/health": "http://127.0.0.1:8000", "/v1": "http://127.0.0.1:8000" } },
});
