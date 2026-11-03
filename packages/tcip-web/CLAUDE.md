# packages/tcip-web

FastAPI backend + Vite/React/TS/Tailwind/Konva frontend: the human's UI. Loads on top of the root
`CLAUDE.md` and the operating contract it imports, which apply here and aren't restated.

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

## Frontend gate

The root `CLAUDE.md`'s Commands block spells it in CI's order; a partial run misses
`format:check` or `lint`.

Use the `frontend-design` skill for IA/visual work.

## The in-app agent terminal launches a provider row

`terminal.py` holds the provider table (`PROVIDERS`); each row's definition there is its launch: its
executable, its arguments, the output sequence every composer frame its harness was recorded
writing carries and none of its screens recorded before a composer carries (`composer_ready`; no
harness promises it), what the harness's own enforcement restricts and leaves open
(`confinement`, recorded on every launch), where its harness reads its instructions and skills
(`files`), and the preparation the launch runs first, each write recorded on the launch, a
preparation that raises `ValueError` answering the launch as not started with its sentence.
Every row's preparation writes the operating contract and every knowledge document's skill into
the workspace at its row's `files` (`tcip_mcp.knowledge.harness_files`): Claude's the contract's
block of `CLAUDE.md` and `.claude/skills/`, Codex's the block of `AGENTS.md` and
`.agents/skills/`, each block replacing an earlier launch's and keeping the file's other text, and
each skill naming the `serve_domain_knowledge` call that serves its document. The Antigravity row,
whose harness takes its MCP server, its tool approvals and its hooks only through its own
configuration, also
registers the server through `agy mcp add`, allows every tcip tool in agy's settings file, keeping
the file's other entries, writes the session hook into the workspace's `.agents/hooks.json`,
keeping that file's other entries, and writes `.agents/skills/` and one always-on rule into the
workspace's `.agents/rules/`, the contract followed by the orientation's tools. The MCP server a
launch hands its harness has one producer, `mcp_server`, written as the JSON configuration file
Claude's row passes, registered through `agy mcp add` for Antigravity's and
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
and how its session started to the session start route, `session_start`, whose answer is
`orientation`'s; a session launched for none stays one, whatever project its agent's MCP server
creates and acts on. Every start but a resume is told the operating contract's posture section
first, read from the contract itself. For a
new conversation the rest is the project as `inspect_project` answers for it, the findings
`diagnose_project` answers (beside how many there are) and each memory corpus's newest entries as
`load_project_memory` pages them; after a compaction the memory alone; for a session launched for
none, that it has none and to hold each report until `initialize_project` creates the project it
lands in; on a resume nothing; with where the knowledge documents are served beside each.
Antigravity documents no hook after a compaction and injects an always-on rule into every turn,
so its row's rule carries the contract and names the tools those parts come from, to read them
again through once the conversation's earlier turns are summarized. The parts are fit to
`ORIENTATION_CAP`, Claude Code's documented cap: whole while all fit together, otherwise sharing
the room so a part
needing less than an equal share keeps all of it, each cut part stating how many it was cut from
and a part that does not fit standing as a short fixed statement of it; Codex's hook passes its
whole context (`additionalContextLimit = 0`). The hook prints it in the shape its harness reads
from a hook's stdout, so it reaches the agent as context, never as a pasted message, and
`diagnose_project` runs in the backend, outside every harness's sandbox. `TerminalSession.deliver`
decides when a paste goes and what it carries. Each launch probes the version its harness
declares and records it, so a harness updated in place is seen at its next launch. No timer-free
signal tells a launch whose sequence never arrives from one still starting, and a sequence that
moved earlier still pastes early. The
child's environment is this process's own less every variable `WITHHELD_ENV_PREFIXES` names:
every `CLAUDE`-prefixed one, so a backend started from a Claude Code session does not launch its
agent as a nested session, every prompt-cache control Claude Code ranks above the row's
`promptCacheTtl` or that turns its caching off, so the lifetime the row requests is the one its
agent runs with, and `PYTHONPATH`, so no harness inherits the backend's import roots, a
repository's on a development machine; its `PWD` is the workspace it runs in. The MCP server and
the session hook import through the packages the backend's interpreter has installed. Session
create names a row by its exact id; the status route lists every row
as `provider_status` answers it, launchable or with the reason it cannot launch, and the rail lets
the breeder pick one. A row is added
only after its harness's real flags are read from the installed CLI and one live smoke
(`tools/smoke_terminal_e2e.py <id>`, with `--workspace` naming one the person trusted by hand for a
harness that asks to trust its workspace) passes: the smoke serves the backend on a loopback port
recorded as `python -m tcip_web` records it, opens a scratch project under a display name made for
the run, adds a line stating an instruction token to the workspace's `CLAUDE.md` and `AGENTS.md`
outside the contract's block, launches the row and reads the contract and every skill where the
row's `files` put them, then asks for one `report_friction` call, no other tool called, carrying
a token only that request names followed by the project's display name, which only the agent's
orientation states, and the instruction token, which only those files state, and passes the
report when the project's friction reports hold exactly one report carrying them, of the
requested category, at the poll that first sees one. What only the real harness decides is the
smoke's own printed checks, first whether the harness read a workspace instruction file (Codex's
discovery of `AGENTS.md` in a workspace that is no git root, Antigravity's reading of a workspace
`AGENTS.md`), shown by the instruction token in its report. For a row whose conversations are
resumed it then ends the agent by the backend's shutdown, launches the row again, and passes only
when that launch resumed the conversation and its agent reports the token continued, which no
request of the resumed launch names. Last it asks the agent to edit the row's rendered
instruction file and to report a token once done, and passes only when the file is unchanged;
for Claude's row the transcript its hook named on the session's report line must also record,
past the length it had when the request was submitted, an `Edit` or `Write` of that file and
after it that use's answer, Claude Code's permission denial. Codex and Antigravity
rows are checked by the unchanged file alone, which cannot tell a denied edit from one never
tried.

Claude's row passes a settings file each launch writes from `CLAUDE_SETTINGS`,
`agent_terminal.settings.json`, with the session hook added; its `confinement` sentence states what
that file enforces, its `promptCacheTtl` is the one place the row's one-hour prompt cache is
requested and read back as the row's cache lifetime, its `env` sets `CLAUDE_CODE_NO_FLICKER=1`
so every launch, fresh or resumed, starts in the fullscreen renderer whose alternate screen is the row's
`composer_ready` (Claude Code documents that variable as overriding the `tui` setting and a
failed fullscreen start, and a `--settings` file's `env` as applied at startup, but not that it
is applied before the renderer is chosen; the live smoke is what shows the pin holds), and the
shipped file's academic WebFetch grants are generated from the `cv-research` document by
`tools/generate_harness_discovery.py`, as are its `Edit` denials, from the harness files and the
state directory `tcip_store` names. Don't extend or edit that file without calling it out
explicitly. Its deny list's relative patterns resolve against the workspace the agent runs in
(Claude Code's permissions page; a `/path` pattern anchors at the temporary settings file's
directory instead): every project's `.tcip` store, written through the tools alone, and the
instruction files and skills the launches write. Every row's agent runs in the backend's
workspace and reads the operating contract and the skills its row's preparation wrote there,
never the repository's. Claude's row loads project settings only (`--setting-sources project`),
so neither the user's own settings, hooks and `CLAUDE.md` nor the developer's auto-memory reach
the breeder's agent.

Each launch records what it ran: create answers a `TerminalLaunch` (`LaunchedProgram`), and that
answer is the launch's one `agent_terminal_started` line in the open project's audit log (none
when no project is open). `session_start` is what records a report the session hook posts, and
what refuses one. A session holds one launch for its life, and its one end is
`TerminalSession.end` in `routes/terminal.py`, whose `agent_terminal_ended` line (`TerminalEnded`)
the end route answers from. Whether a launch resumes, and why not, is decided once, by
`launch_resumes` over `recorded_session`; a session whose backend stopped without ending it is
resumed so, and no end line is written for it, as is one whose person's End answered that its end
line could not be written. A switch of harness while an agent runs is `hand_off_session`, and
`TerminalSession.hand_off_state` is what answers for its hand-off: none asked, asked and not
landed, or landed, the end line's `hand_off`. The session delivers the
requests the rail submits under the condition `Provider` states, and its socket serves
`TerminalServedFrame` (`TerminalOutputFrame` and `TerminalHeldFrame`, each frame's docstring its
contract). Which agent harness the program is comes from the harness's own MCP
handshake, not from here. The child is
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
