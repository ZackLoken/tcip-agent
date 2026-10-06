# packages/tcip-web

FastAPI backend + Vite/React/TS/Tailwind/Konva frontend: the human's UI. Loads on top of the root
`CLAUDE.md`; invariants and operating posture there apply here and aren't restated.

## Layout

```
src/tcip_web/
  routes/          # annotate, audit_gap, canvas, dataset, fs, images, inference, meta,
                    # projects, results, sessions, subjects, terminal, training
  app.py, state.py, jobstore.py, paths.py, __main__.py
  terminal.py + agent_terminal.settings.json
                    # the breeder-facing in-app agent terminal, its provider table and the
                    # Claude row's permission lists
frontend/src/
  api/  components/  hooks/  lib/  store/  tabs/  test/
```

## Frontend gate: run in CI order

A partial run misses `format:check`/`lint`:

```bash
cd packages/tcip-web/frontend
npm run format:check && npm run lint && npm run typecheck && npm test && npm run build
```

Use the `frontend-design` skill for IA/visual work.

## The in-app agent terminal launches a provider row

`terminal.py` holds the provider table (`PROVIDERS`); each row's definition there is its launch:
its executable, its arguments, and the preparation the launch runs first when the harness takes
its MCP server or its tool approvals only through its own configuration (the Antigravity row
registers the server through `agy mcp add` and allows every tcip tool in agy's settings file,
both writes to the breeder's own agy configuration, recorded on the launch). The MCP server a
launch hands its harness has one producer, `mcp_server`, written as the JSON configuration file
Claude's row passes and registered through `agy mcp add` for Antigravity's. The ritual is pasted
together with every request staged by the time the harness takes input, as one message, so a
harness that starts a turn on the ritual still reads them. Session create and restart name a row
by its exact id; the status route lists every row with the reason it cannot launch, if any, and
the rail lets the breeder pick one. A row is added only after its harness's real flags are read
from the installed CLI and one live smoke (`tools/smoke_terminal_e2e.py <id>`) passes.

Claude's row passes `CLAUDE_SETTINGS`: an explicit deny list over platform internals plus a narrow
allowlist, which Claude Code's own permission system enforces, its academic WebFetch grants
generated from the `cv-research` document by `tools/generate_harness_discovery.py`. Those
permission lists merge (union) with the repo root's own, gitignored, developer-local `.claude/settings.json` and
the user's own settings rather than replacing them: list-valued settings keys merge across sources
(code.claude.com/docs/en/configuration). A broad allow entry in the developer's own user settings
therefore widens the breeder lane too. Don't extend or edit that file without calling it out
explicitly.

Each launch records what it ran: create and restart answer a `TerminalLaunch` (the provider id,
the executable and the version it declares, never probed on a `TCIP_TERMINAL_CMD` override, the
argv run before the launch, and the session-start ritual `session_ritual` builds for the open
project), with one
`agent_terminal_started` line in the open project's audit log per launch (none when no project is
open). The rail prints the ritual; the session delivers it, then the requests the rail submits,
to the agent under the condition `Provider` states. Which agent harness
the program is comes from the harness's own MCP handshake, not from here. The child is spawned
with `TCIP_TERMINAL_SESSION` set to the session id; the MCP server the agent launches reads it and
stamps it on its own records as a declared correlation (`tcip_mcp.agent_identity`), beside the
harness name and version the handshake declared. Declarations, all of them; nothing refuses on
them.

## Conventions specific to this package

- Path access from routes goes through `allowed_path`, the one adapter over `assert_path_allowed`
  that answers a refusal as 403, which is always on: the allow-set is derived from the workspace the
  backend was started with, every workspace project and its registered dataset roots, plus the
  additive image roots read from `TCIP_IMAGE_ROOTS` at startup, and containment is by filesystem
  identity. Every route uses the
  resolved path the guard returns, never the client's string. Never route around a 403 on
  escape. Every route that acts on a project acts on the one the backend has open
  (`StateStore.open_root()`, set by `/api/projects/open` from a project's id, 409 while none is
  open) and takes no project from the request; the Results doors also refuse evidence that does not
  belong to it.
- A path a route reads out of the platform's own records (a selection directory a run's config
  names, say) is trusted for reading and never for writing; `assert_path_allowed` is for a
  client-supplied path, not this kind.
- A job registry (`jobstore.JobRegistry`) holds this process's live jobs only; what survives a
  restart is what each job's own run directory or published bucket records.
- The workspace and image roots are read once, in `python -m tcip_web`'s `main`, which writes the
  port record under that workspace and hands both to `StateStore.configure` before serving; an
  unset `TCIP_WORKSPACE` refuses there. A test or script configures the store itself, with a
  scratch workspace.
- The editor saves the one label shape, the per-image label document, through the one label save
  (see `packages/tcip-annotation/CLAUDE.md`); don't add a frontend format option.
- The GUI follows minimalist design without dropping functionality: prefer nesting related
  actions into one structure (a menu, a split button, a grouped control)
  over adding sibling buttons, and combine existing buttons into nested structures where the
  grouping is natural.
