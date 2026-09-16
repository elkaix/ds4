import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { viteSingleFile } from 'vite-plugin-singlefile';

// The server embeds dashboard.html as one C string (Makefile: xxd -i), so the
// build must produce a single file with every script and style inlined and no
// external requests — the page has to work offline on 127.0.0.1.
export default defineConfig({
  plugins: [react(), viteSingleFile()],
  build: {
    target: 'es2022',
    cssTarget: 'safari16',
    modulePreload: false,
    reportCompressedSize: false,
  },
  server: {
    proxy: {
      '/stats': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/v1': 'http://127.0.0.1:8000',
    },
  },
  preview: {
    proxy: {
      '/stats': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/v1': 'http://127.0.0.1:8000',
    },
  },
});
