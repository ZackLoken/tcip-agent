# tcip-web

Browser-based GUI for TCIP. Replaces the partial VS Code extension panels with
a FastAPI backend + React frontend that both human operators and the Claude
agent drive through the same state store.

See [docs/gui_design.md](../../docs/gui_design.md) for architecture and
[docs/gui_implementation_plan.md](../../docs/gui_implementation_plan.md) for
the slice-by-slice plan.

## Layout

```
packages/tcip-web/
  src/tcip_web/
    app.py              # FastAPI app
    state.py            # in-memory GuiState + debounced .tcip/state/gui.json
    paths.py            # safe_join + image-root allow-list (traversal/LFI guards)
    plant_mapping.py    # EXIF GPS + sequence → plant_locations.csv matching
    routes/
      annotate.py       # label CRUD, Annotate-tab commands
      dataset.py        # tree + list + select + state snapshot
      images.py         # EXIF-oriented JPEG serving (+ downsample)
      review.py         # compute_matches, action, backup, save_gt
      training.py       # validate / launch / list / metrics / WS stream
      inference.py      # SAHI-tiled background jobs + progress WS
      results.py        # plant mapping + per-plant curves + onset dates + CSV
      tuning.py         # HPO launch + sweep listing
  frontend/             # Vite + React + TS + Tailwind + Zustand + Konva
  static/               # vite build output (served at /)
```

## Run

### Backend

```bash
# From repo root, in the tcip-agent conda env
conda activate tcip-agent
python -m tcip_web
```

On startup, the backend:

- Binds to `TCIP_WEB_HOST:TCIP_WEB_PORT` (defaults `127.0.0.1:8765`).
- Writes `.tcip/state/web_port.txt` so MCP tools can discover it.
- Replays the last `.tcip/state/gui.json` snapshot if present.

Set `TCIP_WEB_RELOAD=1` to run under uvicorn auto-reload during development.

Open `http://127.0.0.1:8765/` once the backend is up.

### Frontend (development mode)

When iterating on the UI, run Vite's dev server instead of hitting the
pre-built bundle:

```bash
cd packages/tcip-web/frontend
npm install
npm run dev        # http://127.0.0.1:5173  (proxies /api + /ws to :8765)
```

### Frontend (production build)

```bash
cd packages/tcip-web/frontend
npm install
npm run build      # emits into ../static/
```

`python -m tcip_web` then serves the built bundle from `/`.

## Agent ↔ GUI integration

MCP tools that previously wrote to `.tcip/events/*.json` now HTTP POST to
`POST /api/events/{panel}` on the backend. The backend broadcasts those
events to any browser subscribed to `/ws/panel/{panel}`. There is no longer
a VS Code file-watcher in the loop.

Port discovery inside MCP tools: `TCIP_WEB_PORT` env > `.tcip/state/web_port.txt`
> 8765. Host defaults to `127.0.0.1` unless `TCIP_WEB_HOST` is set.

## Keyboard map

`?` opens the full help overlay in the browser.

## VS Code extension

The legacy `packages/tcip-vscode/` extension was retired in favour of this web
GUI. See git history for the old webview-panel implementation if needed.

## Design & deployment decisions

Conscious calls for this single-operator desktop GUI (recorded so they're not
re-litigated):

- **Dark-only theme.** No light mode or theme toggle — the palette is the SI dark
  tokens + `color-scheme: dark`. (The old inert `class="dark"` was removed; no
  Tailwind `dark:` variants are used.)
- **Accessibility bar.** The core flows are keyboard-navigable (shortcut map above;
  focusable native controls). Full ARIA/screen-reader support is **not** targeted for
  this local single-user tool — revisit if it's ever shipped to broader users.
- **Touch / pen: not supported.** The annotation canvas is mouse-only (wheel-zoom,
  middle-drag pan, click-to-draw). Field-tablet (touch/pen) annotation is a known
  limitation and future work, not a regression.
- **Sourcemaps.** The shipped `static/` bundle is built **without** sourcemaps (leaner
  wheel). Use `npm run dev` (HMR + sourcemaps) to debug.
- **Trust boundary.** Loopback bind (default `127.0.0.1`) is frictionless with no auth;
  cross-site WS reads are blocked (Origin check) and DNS-rebinding is blocked
  (TrustedHost). Binding a non-loopback host is refused unless `TCIP_WEB_ALLOW_INSECURE=1`
  (it would expose filesystem browsing + writes with no login). Token auth for an
  intentionally-exposed GUI is a planned follow-on (needs the frontend to attach a token).

### Packaging the GUI into a wheel

`STATIC_DIR` resolves the built frontend from either the installed package
(`tcip_web/static/`) or the src-layout checkout (`packages/tcip-web/static/`). Running
from a checkout (`python -m tcip_web`) needs only `npm run build` (writes
`packages/tcip-web/static/`). To ship the GUI **inside** a wheel, build the frontend
first, then include the built `static/` at `tcip_web/static/` in the wheel (e.g. a
hatch `force-include`) — build the frontend before building the wheel, or the wheel will
ship API-only and `/` returns a 503 with build instructions.
