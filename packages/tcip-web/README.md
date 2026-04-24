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
    paths.py            # safe_join (traversal-safe relative joins)
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

Matches yolo-annotator; `?` opens the full help overlay in the browser.

## VS Code extension

`packages/tcip-vscode/` remains on disk but its webview panels are no longer
the primary GUI. Retirement decision deferred to post-Phase-1 (see
[docs/gui_design.md §10.10](../../docs/gui_design.md#10-open-risks-and-mitigations)).
