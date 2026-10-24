# packages/tcip-web

FastAPI backend + Vite/React/TS/Tailwind/Konva frontend: the human's UI. Loads on top of the root
`CLAUDE.md`; invariants and operating posture there apply here and aren't restated.

## Layout

```text
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

`terminal.py` holds the provider table (`PROVIDERS`); each row's definition there is its launch: its
executable, its arguments, the output sequence one installed version of its harness was recorded
writing when its composer appeared (`composer_ready`; no harness promises it), what the harness's
own enforcement restricts and leaves open (`confinement`, recorded on every launch), and the
preparation the launch runs first
when the harness takes its MCP server or its tool approvals only through its own configuration (the
Antigravity row registers the server through `agy mcp add` and allows every tcip tool and, when the
launch spelled it, the ritual's own doctor line for the open project, as a command and as an
unsandboxed run, in agy's settings file, keeping
the file's other entries, both writes to the breeder's own agy configuration, recorded on the
launch). The ritual's doctor line has one producer, `doctor_command`, running `tcip_web.cli doctor`
under the backend's own interpreter, the one `mcp_server` runs, so the line is the same literal
under any harness's `PATH`; each launch spells it once (`project_launch`), and the ritual, agy's
allow entry and Claude's rendered settings each read that one outcome. The MCP server a launch hands
its harness has one producer, `mcp_server`, written as the JSON configuration file Claude's row
passes, registered through `agy mcp add` for Antigravity's and rendered as `-c` overrides for
Codex's. The session hook a launch registers has one producer too, `conversation_hook_command`,
running `tcip_web.cli report-conversation` under the same interpreter and naming nothing of any
one launch, so its text, which Codex records its hook trust against, stays the same. Claude's
rendered settings carry it as a `SessionStart` hook, and Claude Code documents both a settings
file's hooks (held back until the workspace is trusted) and a hook inheriting its environment.
Codex's `-c` overrides carry it as one, to run once the breeder trusts it through `/hooks`; Codex
documents inline `[hooks]` tables in its configuration, `session_id` in a hook's input and trust
through `/hooks`, but not that a `-c` override registers a hook nor that a hook inherits its
environment (`TCIP_TERMINAL_SESSION`, `TCIP_WORKSPACE`), which only Codex's source supports, so
the live smoke decides both. Antigravity's row registers none. The session's own
text (the ritual, which a resumed launch has none of, and a switch's hand-off request) is pasted
together with every request staged by the time the harness takes input, as one message, so a
harness that starts a turn on it still reads them; the paste
waits until the harness has bracketed paste on and has written its row's `composer_ready`, since a
harness can turn bracketed paste on while a loading screen still discards input. Each launch probes
the version its harness declares and records it, so a harness updated in place is seen at its next
launch. No timer-free signal tells a launch whose sequence never arrives from one still starting,
and a sequence that moved earlier still pastes early. The child's environment is this process's
own less every variable `WITHHELD_ENV_PREFIXES` names: every `CLAUDE`-prefixed one, so a backend
started from a Claude Code session does not launch its agent as a nested session, and every
prompt-cache control Claude Code ranks above the row's `promptCacheTtl` or that turns its caching
off, so the lifetime the row requests is the one its agent runs with. Session create names a row
by its exact id; the status route lists every row with the reason it cannot launch, if any, and
the rail lets the breeder pick one. A row is added only after its harness's real flags are read
from the installed CLI and one live smoke (`tools/smoke_terminal_e2e.py <id>`) passes: the smoke
serves the backend on a loopback port recorded as `python -m tcip_web` records it, opens a scratch
project, then asks for one `report_friction` call carrying a token only that request names, and
passes when the project's friction reports hold exactly one report carrying it, of the requested
category, at the poll that first sees one; for a row whose conversations are resumed it then ends
the agent by
the backend's shutdown, launches the row again, and passes only when that launch resumed the
conversation and its agent reports the token continued, which no request of the resumed launch
names.

Claude's row passes a settings file each launch writes from `CLAUDE_SETTINGS`,
`agent_terminal.settings.json`, with the read-only tcip console commands
(`READ_ONLY_CONSOLE_COMMANDS`, under both invocations) and, when the launch spelled it, the ritual's
own doctor line allowed under both shells; its `confinement` sentence states what that file
enforces, its `promptCacheTtl` is the one place the row's one-hour prompt cache is requested and
read back as the row's cache lifetime, its `env` sets `CLAUDE_CODE_NO_FLICKER=1` so every
launch, fresh or resumed, starts in the fullscreen renderer whose alternate screen is the row's
`composer_ready` (Claude Code documents that variable as overriding the `tui` setting and a
failed fullscreen start, and a `--settings` file's `env` as applied at startup, but not that it
is applied before the renderer is chosen; the live smoke is what shows the pin holds), and the
shipped file's academic WebFetch grants are
generated from the `cv-research` document by `tools/generate_harness_discovery.py`. Don't extend or edit that file without calling
it out explicitly. Every row's agent runs in the backend's workspace, so no harness reads the
repository's instruction files as its own. Claude's row loads project settings only
(`--setting-sources project`), so neither the user's own settings, hooks and `CLAUDE.md` nor the
developer's auto-memory reach the breeder's agent, and adds the repository as a directory, from
which it reads the generated knowledge skills; the other rows reach the knowledge through
`serve_domain_knowledge`.

Each launch records what it ran: create answers a `TerminalLaunch` (the provider id, the
rendered argv and the version its executable declares, never probed on a `TCIP_TERMINAL_CMD`
override, the preparation run before the launch, the conversation it resumed or none, and the
session-start ritual `session_ritual` builds for the open project, none on a resumed launch), with
one `agent_terminal_started` line in the open project's audit log per launch (none when no project
is open). A session holds one launch for its life, and its one end is `TerminalSession.end` in
`routes/terminal.py`, whose `agent_terminal_ended` line (`TerminalEnded`) records whether the
conversation may be resumed (not after the person's Exit, unless that Exit ends a switch of
harness without waiting for its hand-off), the requests its agent was never given, which the end
route answers from that one record, the conversation the session hook last
reported, the time of the agent's last output and, for a switch of harness, whether the agent's
hand-off landed. A launch resumes its row's conversation when the newest end of that row in the
open project's log left it resumable, with an id, its last output within the row's
`Resume.cache_lifetime` (`_conversation_to_resume`); anything else, an override and Antigravity's
row, which documents no cache lifetime, are fresh. A switch of harness while an agent runs is
`hand_off_session`: the hand-off request goes through the session's delivery, the session ends once
the agent's `write_retrospective` line stamped with this session lands in the log, read when the
agent's output arrives, and the person's End meanwhile ends it at once, no hand-off landed; a
session for no project ends at once. The rail prints the ritual; the session delivers it, then
the requests the rail submits, to the agent under the condition `Provider` states. Which agent
harness the program is comes from the harness's own MCP handshake, not from here. The child is
spawned with `TCIP_TERMINAL_SESSION` set to the session id; the MCP server the agent launches reads
it and stamps it on its own records as a declared correlation (`tcip_mcp.agent_identity`), beside
the harness name and version the handshake declared. Declarations, all of them; nothing refuses on
them.

## Conventions specific to this package

- Path access from routes goes through `allowed_path`, the one adapter over `assert_path_allowed`
  that answers a refusal as 403, which is always on: the allow-set is derived from the workspace the
  backend was started with, every workspace project and its registered dataset roots, plus the
  additive image roots read from `TCIP_IMAGE_ROOTS` at startup, and containment is by filesystem
  identity. Every route uses the resolved path the guard returns, never the client's string. Never
  route around a 403 on escape. A route that acts on a project acts on the one the backend has open
  (`StateStore.held()`, set by `/api/projects/open` from a project's id, 409 while none is open),
  and the Results doors also refuse evidence that does not belong to it. The project routes (open,
  remove, rename) name a workspace project by id. A request a page or an agent builds for a project
  (a session end, a contribution naming no session, a canvas push, a panel event) carries that
  project's id, admitted against the open one (`StateStore.admit`, 409 naming it otherwise). A
  contribution naming a session is admitted by that session instead: it lands in the workspace
  project its id names while the session is the person's and unended. A job's cancel and status act
  on the job's own project, whichever is open.
- A path a route reads out of the platform's own records (a selection directory a run's config
  names, say) is trusted for reading and never for writing; `assert_path_allowed` is for a
  client-supplied path, not this kind.
- A job registry (`jobstore.JobRegistry`) holds this process's live jobs only; what survives a
  restart is what each job's own run directory or published bucket records.
- The workspace and image roots are read once, in `python -m tcip_web`'s `main`, which writes the
  port record under that workspace and hands both to `StateStore.configure` before serving; an unset
  `TCIP_WORKSPACE` refuses there. A test or script configures the store itself, with a scratch
  workspace.
- The editor saves the one label shape, the per-image label document, through the one label save
  (see `packages/tcip-annotation/CLAUDE.md`); don't add a frontend format option.
- The GUI follows minimalist design without dropping functionality: prefer nesting related actions
  into one structure (a menu, a split button, a grouped control) over adding sibling buttons, and
  combine existing buttons into nested structures where the grouping is natural.
