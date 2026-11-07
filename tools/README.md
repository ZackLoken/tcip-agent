# tools/

Not a Python package (no `__init__.py`); each file is an entry point run with
`python tools/<name>.py`, or from the repository root as `python -m tools.<name>` when it imports
the test suite's helpers or another tool. Read a script's own module docstring and `--help` for
exact usage; this index only says what each does. CI and development tooling only; the operator
commands a breeder or an operator runs against a project are documented in `README.md`'s Running
section instead.

- `gate_baseline.py` - runs CI's own jobs locally; its module docstring says which jobs and steps
  it leaves out.
- `prove_test_fails_before.py` - extracts a baseline revision with `git archive`, overlays the
  current test tree so conftest and helpers travel with the test, proves the baseline's own source
  is what gets imported, and reads pytest's per-test outcome to say whether a test actually fails
  without its fix. Four verdicts, `GUARDS`, `VACUOUS`, `INDETERMINATE` and `REFUSED`, each its own
  exit code; its module docstring states what each means and how the baseline is chosen, and only
  `GUARDS` is evidence. `--test-rev` checks a guard claim already in the history.
- `cross_family_ask.py` - poses one question to several agent harnesses and records comparable
  answers; its module docstring and `--help` say which families it runs and what each run records.
- `list_tools.py` - prints the live MCP tool registry (count + names); the single source of truth
  for "how many domain tools exist," since the count drifts as tools are added/renamed.
- `line_delta.py` - sums a change's insertions and deletions by area (`packages/`, `tests/`, other)
  for a commit, a `rev1..rev2` range, `<rev>..` for the working tree against `rev`, or `--cached`.
- `worktree_gate.py` - runs ruff, mypy and a pytest file list inside a worktree, its own `tcip_mcp`
  resolution proved first (the editable installs point at the main checkout, so a worktree needs its
  own `PYTHONPATH`) and refused before any gate runs if `tcip_mcp` resolves outside it; each
  requested gate runs in the foreground, stopping at the first failure with its exit code.
- `check_premise_cites.py` - reads an issue's work order with `gh` and checks every `at <path>:<a>[-<b>]`
  cite in its premise half against the premise half's commit: the path exists there, the range is
  inside the file, and a premise that backticks identifiers has one of them in the cited range; a
  premise with no cite in that form fails. Prints PASS or each failing premise and FAIL; a repository
  the header labels needs `--checkout <label>=<path>`.
- `generate_frontend_routes.py` - generates the browser's route-path module, over FastAPI's
  effective route contexts whose names the frontend's non-test sources read as `ROUTES.<name>` (the
  reads `tests.READS_SCRIPT` finds through the TypeScript compiler, run under `node`
  over the frontend's installed packages; a read naming no registered route is refused), and the
  dev server's proxy config over every registered route, both formatted by the frontend's
  installed Prettier, so the frontend references a path by name instead of restating the string a
  second time.
- `generate_frontend_types.py` - renders the browser's types module at its `GENERATED_PATH` from
  the pydantic models its `declared_models` lists, so the browser's types are a projection of the
  backend's rather than hand-transcribed. Run after changing a declared model;
  `tests/test_generated_frontend_types.py` fails when the checked-in module is stale.
- `generate_*_fixture.py` - each writes to its `GENERATED_PATH` what the platform's own routes
  serve for scenes it makes in a scratch session (`fixture_session.py`), the fixtures the frontend
  tests read; its module docstring names the scenes. Run after changing what those routes serve;
  `tests/test_generated_frontend_fixtures.py` holds each file to what a regeneration serves.
- `generate_harness_discovery.py` - writes the repository's `.claude/skills/<name>/SKILL.md` and
  `.agents/skills/<name>/SKILL.md` files and the operating contract's block in `AGENTS.md` through
  `tcip_mcp.knowledge.harness_files`, the writer every terminal launch runs in the workspace, and
  the Claude row's `Edit` denials, from each provider row's harness files and the store's state
  directory, and its `WebFetch` grants. Run after adding, renaming, or re-describing a document,
  changing the operating contract or a row's harness files;
  `tests/test_harness_discovery_generated.py` fails when a checked-in generated file is stale.
- `verify_skill_tools.py` - guardrail holding every tool name in agent-facing prose to the MCP tool
  registry: a fabrication check over every table whose header's first column is "Tool", and an
  orphan check for a registered tool no surface names anywhere.
- `verify_skill_traits.py` - flags any trait-like token in a crop/domain `SKILL.md` that isn't a
  real name in `crops.yml`, to catch fabricated trait names in skill prose.
- `serve_capture_app.py` - starts and stops `python -m tcip_web` under a scratch environment for a
  GUI capture harness: `start` sets `TCIP_WORKSPACE` beneath a caller-named harness root (refusing
  one under the repository; refusing one that aliases the caller's own active `TCIP_WORKSPACE` in
  either direction; refusing, when no `TCIP_WORKSPACE` is bound, a root that already looks like a
  workspace itself), waits for the projects route to answer, and records pid and port; `stop` kills
  the recorded process tree, Windows-only (`taskkill /T`), refusing outright on any other host.
  Carries no project fixture, seed, or crop name; seeding a project is the capture script's own job.
- `generate_favicon.ps1` - renders the browser-tab favicon from the source logo: crops its
  transparent margins, resizes the result to 512x512, and writes it plus a 32x32 copy to the
  frontend's public assets.
- `smoke_terminal_e2e.py` - live smoke exercising the in-app agent terminal against one provider
  row's real harness end to end (spawn, websocket attach, prompt, response), named by its id. Costs
  one model turn.
