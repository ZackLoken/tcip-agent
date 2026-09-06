# packages/tcip-web

FastAPI backend + Vite/React/TS/Tailwind/Konva frontend: the human's UI. Loads on top of the root
`CLAUDE.md`; invariants and operating posture there apply here and aren't restated.

## Layout

```
src/tcip_web/
  routes/          # annotate, canvas, classes, coverage, dataset, fs, images, inference, meta,
                    # projects, results, review, sessions, terminal, training, tuning, validation
  app.py, state.py, jobstore.py, paths.py, identity.py, __main__.py
  terminal.py + agent_fence_rules.py + agent_bash_guard.py + agent_powershell_guard.py
    + agent_session_start.py
                    # the breeder-facing in-app agent terminal and its permission fence
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

## The in-app agent terminal is a separate, already-hardened fence

`terminal.py` spawns a real `claude` process for the browser-facing "Agent" terminal, with
`--settings` pointing at a spawn-time materialized copy of the committed
`packages/tcip-web/src/tcip_web/agent_terminal.settings.json` (absolute hook paths; the
committed file itself only if materialization fails), a permission fence (an explicit deny list
over platform internals plus a narrow allowlist, `Bash`/`PowerShell` guards, a `SessionStart`
ritual-injection hook, `SessionEnd` learning capture) scoped to that breeder-facing session; a
write that is neither denied nor allow-listed reaches a human approval prompt rather than being
auto-denied. `--settings` merges its permission lists (union) with the repo root's own,
gitignored, developer-local `.claude/settings.json` and the user's own settings rather than
replacing them: list-valued settings keys merge across sources. A broad allow entry in the
developer's own user settings therefore widens the breeder-lane fence too. The fence is
deny-list first. A `claude` session launched without `--settings` is unaffected by the fence
file. Don't extend or edit that fence file without calling it out explicitly.

Each launch records what it ran: the session answers `launched` (the executable, `argv[0]`, and
the version it declares to `--version`, probed once per process on the resolved CLI only and never
on a `TCIP_TERMINAL_CMD` override) on the create and restart routes, with one
`agent_terminal_started` platform audit line per launch. Which agent harness that program is comes
from the harness's own MCP handshake, not from here. The child is spawned with
`TCIP_TERMINAL_SESSION` set to the session id; the MCP server the agent launches reads it and stamps
it on its own records as a declared correlation (`tcip_mcp.agent_identity`), beside the harness name
and version the handshake declared. Declarations, all of them; nothing refuses on them.

That file's `Edit(...)` deny rules are the one declaration of the platform-protected set: both shell
guards classify each write target through `agent_fence_rules.classify()`, which derives the protected
directories, single files, and project-data segments from those deny rules
(`_declared_targets()`), so a path added there fences both shells and neither guard carries a path
list to keep in step. A guard that cannot read the declaration denies rather than falling through.

The classifier normalizes a target (strips shell quotes, unifies separators, collapses `..`,
handles Windows drive-relative forms) and anchors the repo rules to the repo root, so a breeder's own
same-named project file (their `README.md`) is not caught by basename. The protected set is a
function of deployment mode (`fence_mode()`): dev protects the repo tree plus breeder data;
production, an installed package behind an OS sandbox, protects only breeder data. Mode defaults to
dev unless `TCIP_FENCE_MODE=prod`. The guard is airtight only on the one no-prompt path (a redirect
riding an allow-listed read prefix); its in-place-writer coverage is defense-in-depth behind the
human approval prompt, and a `cd`-then-relative write is an accepted residual of a cwd-blind guard.

## Conventions specific to this package

- Path access from routes goes through `assert_path_allowed`, which is always on: the allow-set is
  derived from the workspace, every workspace project and its registered dataset roots, plus the
  additive `TCIP_IMAGE_ROOTS` list, and containment is by filesystem identity. Every route uses the
  resolved path the guard returns, never the client's string. A 403 on escape is not routed
  around. The Results doors go further: they serve only the project the GUI has
  open (`StateStore.project_root`, set by the guarded `/dataset/select`) and refuse evidence that
  does not belong to it.
- A path a route reads out of the platform's own records (a manifest directory an experiment
  config names, say) is trusted for reading and never for writing; `assert_path_allowed` is for
  a client-supplied path, not this kind.
- Under pytest, with starlette's `TestClient` module loaded, or on a request arriving from an
  in-process test transport (starlette's `TestClient` or httpx's `ASGITransport`), the app
  refuses to start unless `TCIP_WORKSPACE` is set (`app.WorkspaceUnsetUnderTest`); set it and
  `TCIP_STATE_ROOT` to scratch directories before starting one.
- Review save formats mirror the annotation-engine's `{json, coco}` scope (see
  `packages/tcip-annotation/CLAUDE.md`); don't add a frontend format option the backend can't read.
- The GUI follows minimalist design without dropping functionality: prefer nesting related
  actions into one structure (a menu, a split button, a grouped control)
  over adding sibling buttons, and combine existing buttons into nested structures where the
  grouping is natural.
