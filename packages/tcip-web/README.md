# tcip-web

Browser-based GUI for TCIP: a FastAPI backend + React frontend that both human operators and the
agent drive through the same state store.

## Layout

```text
packages/tcip-web/
  src/tcip_web/
    app.py              # FastAPI app
    state.py            # the open project, its GuiState (written to the project's GUI snapshot record on change)
    paths.py            # safe_join + the always-on path guard (derived allow-set, identity containment)
    jobstore.py         # background job tracking (inference, the review queue, image overviews)
    terminal.py         # in-app agent terminal (spawns a provider table row's harness)
    routes/
      annotate.py       # the editor: label document load/save with gestures, proposals, review queue
      audit_gap.py      # the 409 a committed write whose audit line was lost answers
      canvas.py         # live canvas capture for the agent's own image-capable read tool
      dataset.py        # tree + select + nav position
      fs.py             # filesystem browsing for path pickers
      images.py         # EXIF-oriented serving: display reads, native regions, a view's reads
      inference.py      # SAHI-tiled background jobs + progress WS
      meta.py           # crop/project metadata
      projects.py       # project open/create/list
      results.py        # plant mapping + per-plant curves + onset dates + CSV
      sessions.py       # GUI session state
      subjects.py       # subject registry load/save
      terminal.py       # in-app agent terminal endpoints
      training.py       # run and sweep relaunch / list / cancel / TensorBoard / metrics WS stream
  frontend/             # Vite + React + TS + Tailwind + Zustand + Konva
```

## Run

The repository README's Setup and Running sections install the frontend, build the bundle the
backend serves from `/` (`app.STATIC_DIR`), and start the backend and Vite's dev server, which
proxies the API routes to the backend. On startup, the backend:

- Resolves the workspace `TCIP_WORKSPACE` names, once, and refuses to start when it is unset.
- Binds the loopback address at `TCIP_WEB_PORT`, or at its default port.
- Records the port it bound in the workspace root's store so MCP tools can discover it, the one
  location every process on the machine resolves the same way regardless of which project each has
  open.
- Opens no project: the GUI starts at the project picker, and opening a project there replays its
  GUI snapshot if present.
- Checks the bundle, when `frontend/` sits beside the package and `app.STATIC_DIR` holds a page: it
  runs the frontend's `bundleSource.mjs` under Node over the source (`app.bundle_record`),
  compares its record with the `build.json` the build wrote beside the bundle, logs one warning
  naming the build command when the record is missing or differs, and holds that sentence as the
  `bundle` of every state snapshot until it stops.

## Agent ↔ GUI integration

MCP tools HTTP POST to `POST /api/events/{panel}` on the backend, which broadcasts those events to
any browser subscribed to `/ws/panel/{panel}`.

Port discovery inside MCP tools is `tcip_mcp.web_client.backend_url`'s.

## Keyboard map

`?` opens the full help overlay in the browser.

## Design & deployment decisions

Calls made for this single-operator desktop GUI:

- Dark-only theme. No light mode or theme toggle: the palette is the SI dark tokens plus
  `color-scheme: dark`. No Tailwind `dark:` variants are used.
- Accessibility bar. The core flows are keyboard-navigable (shortcut map above; focusable native
  controls). Full ARIA/screen-reader support is not targeted for this local single-user tool.
- Touch / pen: not supported. The annotation canvas is mouse-only (wheel-zoom, middle-drag pan,
  click-to-draw). Field-tablet (touch/pen) annotation is a known limitation and future work.
- Trust boundary: the repository's `SECURITY.md`, "The trust boundary today, and where it stops
  holding".
