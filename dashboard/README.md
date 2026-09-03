# ds4-server dashboard

Source for the live stats page served at `http://127.0.0.1:8000/dashboard`.

React 19 + TypeScript + Vite, built with `vite-plugin-singlefile` into **one**
self-contained HTML file: `../dashboard.html`. The C build (`make`) turns that
file into `dashboard_html.h` with `xxd -i` and `ds4_server` embeds it, so the
page works from any cwd, needs no `--cors` (same-origin `/stats`), and makes no
network requests other than to the server that served it.

`dashboard.html` is a **committed build artifact**: the C build never needs
Node. Rebuild it only when you change something under `src/`.

```sh
cd dashboard
npm ci            # once
npm run dev       # live-reload at http://localhost:5173/?url=http://127.0.0.1:8000
npm run build     # typecheck + bundle -> ../dashboard.html
npm run lint
```

Then `make ds4-server` from the repo root to embed the new page.

Query parameters: `?url=<server>` overrides the polled origin (needed when the
page is opened as a file or from the Vite dev server; the server then needs
`--cors`), `?ms=<interval>` changes the poll period (default 1000 ms).
