# tcip-web

Browser-based GUI for TCIP: a FastAPI backend + React frontend that both human
operators and the agent drive through the same state store.

## Layout

```
packages/tcip-web/
  src/tcip_web/
    app.py              # FastAPI app
    state.py            # the open project, its GuiState (written to the project's GUI snapshot record on change)
    paths.py            # safe_join + the always-on path guard (derived allow-set, identity containment)
    jobstore.py         # background job tracking (training/inference/tuning)
    terminal.py         # in-app agent terminal (spawns a provider table row's harness)
    routes/
      annotate.py       # the editor: label document load/save with gestures, proposals, review queue
      audit_gap.py      # the 409 a committed write whose audit line was lost answers
      canvas.py         # live canvas capture for the agent's own image-capable read tool
      dataset.py        # tree + select + nav position
      fs.py             # filesystem browsing for path pickers
      images.py         # EXIF-oriented JPEG serving (+ downsample, region-serving grid)
      inference.py      # SAHI-tiled background jobs + progress WS
      meta.py           # crop/project metadata
      projects.py       # project open/create/list
      results.py        # plant mapping + per-plant curves + onset dates + CSV
      sessions.py       # GUI session state
      subjects.py       # subject registry load/save
      terminal.py       # in-app agent terminal endpoints
      training.py       # validate / launch / list / metrics / WS stream
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

- Resolves the workspace `TCIP_WORKSPACE` names, once, and refuses to start when it is unset.
- Binds the loopback address `127.0.0.1` at `TCIP_WEB_PORT` (default `8765`).
- Records the port it bound in the workspace root's store so MCP tools can discover it, the
  one location every process on the machine resolves the same way regardless of which
  project each has open.
- Opens the project the workspace's last-opened pointer names (by the id in its project
  record), if that project is still in the workspace, and replays that project's GUI snapshot
  if present.

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

MCP tools HTTP POST to `POST /api/events/{panel}` on the backend, which broadcasts those
events to any browser subscribed to `/ws/panel/{panel}`.

Port discovery inside MCP tools: the workspace root's port record, the port the backend bound, always at `127.0.0.1`; with no record, no backend serves the workspace and an
event is not delivered.

## Keyboard map

`?` opens the full help overlay in the browser.

## Design & deployment decisions

Calls made for this single-operator desktop GUI:

- Dark-only theme. No light mode or theme toggle: the palette is the SI dark
  tokens plus `color-scheme: dark`. No Tailwind `dark:` variants are used.
- Accessibility bar. The core flows are keyboard-navigable (shortcut map above;
  focusable native controls). Full ARIA/screen-reader support is not targeted for
  this local single-user tool.
- Touch / pen: not supported. The annotation canvas is mouse-only (wheel-zoom,
  middle-drag pan, click-to-draw). Field-tablet (touch/pen) annotation is a known
  limitation and future work.
- Sourcemaps. The shipped `static/` bundle is built without sourcemaps. Use
  `npm run dev` (HMR + sourcemaps) to debug.
- Trust boundary. A connection from this machine (a loopback address) is served with no auth;
  a connection through a network address is refused, because the GUI hands its client filesystem
  reads and writes and the interactive agent terminal (keyboard access to a coding agent) with no
  login. The Host header must be a loopback name at the port the connection arrived on, and every
  WebSocket connect and every state-changing HTTP request (`POST`/`PUT`/`PATCH`/`DELETE`) either
  carries no Origin at all (the non-browser allowance; the MCP tools send none) or carries a
  loopback origin at any port. Another local server's page is admitted by this layer too; only
  the JSON-body guard's unanswered preflight stops its browser from mutating.

### Packaging the GUI into a wheel

`STATIC_DIR` resolves the built frontend from either the installed package
(`tcip_web/static/`) or the src-layout checkout (`packages/tcip-web/static/`). Running
from a checkout (`python -m tcip_web`) needs only `npm run build` (writes
`packages/tcip-web/static/`). To ship the GUI inside a wheel, build the frontend
first, then include the built `static/` at `tcip_web/static/` in the wheel (e.g. a
hatch `force-include`); the Dockerfile takes the same route with a build-time `COPY`
into `src/tcip_web/static/` before the pip install, rather than a hatch declaration.
Build the frontend before building the wheel, or the wheel will ship API-only and
`/` returns a 503 with build instructions.
