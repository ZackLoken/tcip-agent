# tcip-web

Browser-based GUI for TCIP: a FastAPI backend + React frontend that both human
operators and the Claude agent drive through the same state store.

## Layout

```
packages/tcip-web/
  src/tcip_web/
    app.py              # FastAPI app
    state.py            # in-memory GuiState + debounced .tcip/state/gui.json
    paths.py            # safe_join + the always-on path guard (derived allow-set, identity containment)
    identity.py         # current-user identity for created_by/accepted_by provenance stamping
    jobstore.py         # background job tracking (training/inference/tuning)
    terminal.py         # in-app agent terminal (spawns a hardened `claude` process)
    routes/
      annotate.py       # label CRUD, Annotate-tab commands
      canvas.py         # live canvas capture for the agent's own image-capable read tool
      classes.py        # subject/class CRUD
      coverage.py       # view-coverage: reference grid + served/swept per-cell record
      dataset.py        # tree + select + nav position
      fs.py             # filesystem browsing for path pickers
      images.py         # EXIF-oriented JPEG serving (+ downsample)
      inference.py      # SAHI-style tiled background jobs + progress WS
      meta.py           # crop/project metadata
      projects.py       # project open/create/list
      results.py        # plant mapping + per-plant curves + onset dates + CSV
      review.py         # compute_matches, action, backup, priority queue
      sessions.py       # GUI session state
      terminal.py       # in-app agent terminal endpoints
      training.py       # validate / launch / list / metrics / WS stream
      tuning.py         # HPO launch + sweep listing
      validation.py     # validate_reference: a review's verdicts into COCO evaluation records
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
- Writes the workspace root's `.tcip/state/web_port.txt` so MCP tools can discover it, the
  one location every process on the machine resolves the same way regardless of which
  project each has open.
- Pins its own platform-state root from the workspace's active-project marker, so it starts
  on the project the GUI has open.
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

MCP tools HTTP POST to `POST /api/events/{panel}` on the backend, which broadcasts those
events to any browser subscribed to `/ws/panel/{panel}`.

Port discovery inside MCP tools: the workspace root's `.tcip/state/web_port.txt` (the port
bound) > `TCIP_WEB_PORT` env (read only when no record parses) > 8765. Host
defaults to `127.0.0.1` unless `TCIP_WEB_HOST` is set.

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
  a connection through a network address is refused, whatever the bind, until the operator sets
  `TCIP_WEB_ALLOW_INSECURE=1`, because an exposed GUI hands a network client filesystem reads
  and writes and the interactive agent terminal (keyboard access to Claude Code) with no login.
  The Host header must name this backend as reached (its arrival address, its own hostname, or
  an entry of `TCIP_WEB_ADVERTISED_HOSTS`, consulted only under the opt-in; never a wildcard),
  and every WebSocket connect and every state-changing HTTP request (`POST`/`PUT`/`PATCH`/
  `DELETE`) either carries no Origin at all (the non-browser allowance; the MCP tools send
  none) or must carry one the backend serves: the request's own origin, a loopback origin at
  any port on a local arrival, or an advertised authority under the opt-in. On a
  loopback arrival another local server's page is admitted by this layer too; only the
  JSON-body guard's unanswered preflight stops its browser from mutating. The port
  compared is the Origin's own, with its scheme's default port filled in when none is written,
  so a same-machine reverse proxy that terminates HTTPS and forwards a bare `Host: gui.example`
  needs two `TCIP_WEB_ADVERTISED_HOSTS` entries (`gui.example:80` for the Host check,
  `gui.example:443` for the Origin), while one that forwards `Host: gui.example:443` needs only
  that one; an advertised name is scheme-blind, so it admits an `http` or `https` Origin alike.
  Token auth for an intentionally exposed GUI is a planned follow-on (needs the frontend to
  attach a token).

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
