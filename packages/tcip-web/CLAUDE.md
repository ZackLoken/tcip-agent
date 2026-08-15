# packages/tcip-web

FastAPI backend + Vite/React/TS/Tailwind/Konva frontend: the human's UI. Loads on top of the root
`CLAUDE.md`; invariants and operating posture there apply here and aren't restated.

## Layout

```
src/tcip_web/
  routes/          # annotate, canvas, classes, dataset, fs, images, inference, meta, projects,
                    # results, review, sessions, terminal, training, tuning
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
`--settings packages/tcip-web/src/tcip_web/agent_terminal.settings.json`, a committed permission
fence (deny-by-default edits outside a narrow allowlist, `Bash`/`PowerShell` guards, a `SessionStart`
ritual-injection hook, `SessionEnd` learning capture) scoped to that breeder-facing session. It is
not merged with the repo root's own `.claude/settings.json`: the developer's own `claude`
session (this one, with no `--settings` flag) stays unrestricted by it, by design. Don't extend or
edit that fence file without calling it out explicitly; it's a security boundary, not incidental
config.

That file's `Edit(...)` deny rules are the one declaration of the platform-protected set: both shell
guards build their matcher from it at hook time via `agent_fence_rules.protected_pattern()`, so a
path added there fences both shells and neither guard carries a path list to keep in step. A guard
that cannot read the declaration denies rather than falling through, so the terminal stops visibly
instead of silently unfencing.

## Conventions specific to this package

- Path access from routes goes through `assert_path_allowed` (`TCIP_IMAGE_ROOTS` allow-list): a
  403 on escape is the rail working, not a bug to route around.
- Review save formats mirror the annotation-engine's `{json, coco}` scope (see
  `packages/tcip-annotation/CLAUDE.md`); don't add a frontend format option the backend can't read.
- The GUI follows minimalist design without dropping functionality (Zack's standing preference):
  prefer nesting related actions into one structure (a menu, a split button, a grouped control)
  over adding sibling buttons, and combine existing buttons into nested structures where the
  grouping is natural. Density of controls is a cost; capability is not the thing to cut.
