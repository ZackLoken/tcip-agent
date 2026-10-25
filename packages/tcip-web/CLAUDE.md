# packages/tcip-web

FastAPI backend + Vite/React/TS/Tailwind/Konva frontend: the human's UI. Loads on top of the root
`CLAUDE.md`; invariants and operating posture there apply here and aren't restated.

## Layout

```text
src/tcip_web/
  routes/          # annotate, audit_gap, canvas, dataset, fs, images, inference, meta,
                    # projects, results, sessions, subjects, terminal, training
  app.py, state.py, jobstore.py, paths.py, __main__.py
  cli/             # the tcip dispatcher: each operator-only command's module (COMMANDS, one of
                    # them session_hook here) and every registered tool by its own name
                    # through tcip_mcp.cli.run_tool
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
when the harness takes its MCP server, its tool approvals or its hooks only through its own
configuration (the Antigravity row registers the server through `agy mcp add`, allows every tcip
tool in agy's settings file, keeping the file's other entries, and writes the session hook into the
workspace's `.agents/hooks.json`, keeping that file's other entries, each write recorded on the
launch). The MCP server a launch hands its harness has one producer, `mcp_server`, written as the
JSON configuration file Claude's row passes, registered through `agy mcp add` for Antigravity's and
rendered as `-c` overrides for Codex's, which also make the server `required`, so Codex's startup
and resume wait for it before a turn's first tool catalog. The session hook every launch registers
has one producer, `session_hook_argv`, running `tcip_web.cli session-hook` under the backend's own
interpreter, the one `mcp_server` runs, and naming nothing of any one launch: Claude's rendered
settings carry it in Claude Code's exec form (`command` and `args`, no shell), Codex's `-c`
overrides as one shell line (`shell_line`, refusing a word a shell would not read as itself) whose
text, which Codex records its hook trust against, stays the same across launches, and agy's hooks
file as that line followed by the session's start route, since Antigravity documents neither a
hook's environment nor its working directory. Claude Code documents a settings file's hooks and a
hook inheriting its environment; Codex documents inline `[hooks]` tables, `session_id` and
`source` in a hook's input and trust through `/hooks`, but not that a `-c` override registers a hook
nor that a hook inherits its environment (`TCIP_TERMINAL_SESSION`, `TCIP_WORKSPACE`), which only
Codex's source supports, so the live smoke decides both. The hook posts its harness's conversation
and how its session started to the session start route, which answers what the session's agent is
told (`orientation`) for the project the session was launched for, never the GUI's; a session
launched for none stays one, whatever project its agent's MCP server creates and acts on. For a
new conversation it is the project as `inspect_project` answers for it, the findings
`diagnose_project` answers (beside how many there are) and each memory corpus's newest entries as
`load_project_memory` pages them; after a compaction the memory alone; on a resume nothing; with
where the knowledge documents are served beside each. The parts are fit to `ORIENTATION_CAP`,
Claude Code's documented cap: whole while all fit together, otherwise sharing the room so a part
needing less than an equal share keeps all of it, each cut part stating how many it was cut from
and a part that does not fit standing as a short fixed statement of it; Codex's hook passes its
whole context (`additionalContextLimit = 0`). The hook prints it in the shape its harness reads
from a hook's stdout, so it reaches the agent as context, never as a pasted message, and
`diagnose_project` runs in the backend, outside every harness's sandbox. The session's own text (a
switch's hand-off request) is pasted together with every request staged by the time the harness
takes input, as one message, so a harness that starts a turn on it still reads them; the paste
waits until the harness has bracketed paste on and has written its row's `composer_ready`, since a
harness can turn bracketed paste on while a loading screen still discards input. Each launch
probes the version its harness declares and records it, so a harness updated in place is seen at
its next launch. No timer-free signal tells a launch whose sequence
never arrives from one still starting, and a sequence that moved earlier still pastes early. The
child's environment is this process's own less every variable `WITHHELD_ENV_PREFIXES` names:
every `CLAUDE`-prefixed one, so a backend started from a Claude Code session does not launch its
agent as a nested session, and every prompt-cache control Claude Code ranks above the row's
`promptCacheTtl` or that turns its caching off, so the lifetime the row requests is the one its
agent runs with. Session create names a row by its exact id; the status route lists every row
with the reason it cannot launch, if any, and the rail lets the breeder pick one. A row is added
only after its harness's real flags are read from the installed CLI and one live smoke
(`tools/smoke_terminal_e2e.py <id>`) passes: the smoke serves the backend on a loopback port
recorded as `python -m tcip_web` records it, opens a scratch project under a display name made for
the run, then asks for one `report_friction` call, no other tool called, carrying a token only
that request names followed by the project's display name, which only the agent's orientation
states, and passes when the project's friction reports hold exactly one report carrying both, of
the requested category, at the poll that first sees one; for a row whose conversations are
resumed it then ends the agent by the backend's shutdown, launches the row again, and passes only
when that launch resumed the conversation and its agent reports the token continued, which no
request of the resumed launch names.

Claude's row passes a settings file each launch writes from `CLAUDE_SETTINGS`,
`agent_terminal.settings.json`, with the session hook added; its `confinement` sentence states what
that file enforces, its `promptCacheTtl` is the one place the row's one-hour prompt cache is
requested and read back as the row's cache lifetime, its `env` sets `CLAUDE_CODE_NO_FLICKER=1`
so every launch, fresh or resumed, starts in the fullscreen renderer whose alternate screen is the row's
`composer_ready` (Claude Code documents that variable as overriding the `tui` setting and a
failed fullscreen start, and a `--settings` file's `env` as applied at startup, but not that it
is applied before the renderer is chosen; the live smoke is what shows the pin holds), and the
shipped file's academic WebFetch grants are generated from the `cv-research` document by
`tools/generate_harness_discovery.py`. Don't extend or edit that file without calling it out
explicitly. Every row's agent runs in the backend's workspace, so no harness reads the
repository's instruction files as its own. Claude's row loads project settings only
(`--setting-sources project`), so neither the user's own settings, hooks and `CLAUDE.md` nor the
developer's auto-memory reach the breeder's agent, and adds the repository as a directory, from
which it reads the generated knowledge skills; the other rows reach the knowledge through
`serve_domain_knowledge`.

Each launch records what it ran: create answers a `TerminalLaunch` (the provider id, the
rendered argv and the version its executable declares, never probed on a `TCIP_TERMINAL_CMD`
override, the preparation run before the launch and the conversation it resumed or none), with
one `agent_terminal_started` line in the open project's audit log per launch (none when no project
is open). A session holds one launch for its life, and its one end is `TerminalSession.end` in
`routes/terminal.py`, whose `agent_terminal_ended` line (`TerminalEnded`; none for a session
launched with no project open) records whether the
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
session for no project ends at once. The session delivers the requests the rail submits to the
agent under the condition `Provider` states. Which agent
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
