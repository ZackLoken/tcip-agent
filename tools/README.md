# tools/

Not a Python package (no `__init__.py`); each file is a standalone entry point run with
`python tools/<name>.py`. Read a script's own module docstring and `--help` for exact usage;
this index only says what each does. CI and development tooling only; the operator commands a
breeder or an operator runs against a project are documented in `README.md` and
`CONTRIBUTING.md` instead.

- `build_module_inventory.py` - builds a module inventory and real import graph for the repo's
  Python and TypeScript source trees, so `check_architecture_doc.py` can cross-check
  ARCHITECTURE.md's module-ownership tables against the tree it actually describes.
- `census_double_published_buckets.py` - read-only census over one or more project roots of
  prediction buckets published more than once before the live-bucket refusal landed: a stamp
  whose `image_filenames` names fewer stems than the bucket holds documents for, a bucket whose
  stamp names no `image_filenames` map at all (unjudgeable rather than clean), and any
  validation record sealed over such a bucket's mixed-run digest. Prints findings, repairs
  nothing; the exit code says whether any was found. A one-off for this remediation: run once
  over the workspace projects and deleted when the census closes.
- `check_architecture_doc.py` - verifies ARCHITECTURE.md's module-ownership tables against the
  tree, for CI: every named path exists, and, given a module-inventory JSON, the in-repo-import
  and imported-by counts per row are cross-checked for drift.
- `check_architecture_citations.py` - verifies ARCHITECTURE.md's `file:line` citations against
  the code they quote, for CI; `--fix` rewrites a re-anchorable citation's stale line number in
  place, leaving a genuinely failed citation for a human.
- `gate_baseline.py` - runs the quality gate CI actually declares, parsed from
  `.github/workflows/ci.yml` rather than restated by hand, so a local pass predicts a CI pass.
- `prove_test_fails_before.py` - extracts a baseline revision with `git archive`, overlays the
  current test tree so conftest and helpers travel with the test, proves the baseline's own
  source is what gets imported, and reads pytest's per-test outcome to say whether a test
  actually fails without its fix. Four verdicts on four exit codes: `GUARDS` (0), `VACUOUS` (1),
  `INDETERMINATE` (2) when the baseline is not shown to precede the change, and `REFUSED` (3)
  when nothing was selected, everything was skipped, collection failed, or every failure was a
  missing import. Only `GUARDS` is evidence. The default baseline is `HEAD` for uncommitted work
  and the merge-base against the integration branch otherwise; the previous commit is not a
  baseline in a one-file-per-commit history, so pass `--baseline` when neither default applies.
  `--test-rev` checks a guard claim already in the history.
- `cross_family_ask.py` - poses one identical question to several agent harnesses (claude,
  codex, antigravity) and records comparable answers: the exact prompt, argv, stdout/stderr, the
  extracted response, and run metadata describing what the harness was and how long it took.
- `list_tools.py` - prints the live MCP tool registry (count + names); the single source of
  truth for "how many domain tools exist," since the count drifts as tools are added/renamed.
- `generate_frontend_routes.py` - generates the browser's route-path module and the dev
  server's proxy config from the backend's registered FastAPI routes, so the frontend
  references a path by name instead of restating the string a second time.
- `generate_frontend_types.py` - renders `frontend/src/api/types.generated.ts` from the pydantic
  models that declare the view-coverage record's shape and the GUI's tab/mode vocabulary
  (`tcip_web.state.GuiVocabulary`), so the browser's types are a projection of the backend's
  rather than hand-transcribed. Run after changing a declared model;
  `tests/test_generated_frontend_types.py` fails when the checked-in module is stale.
- `generate_frozen_manifest.py` - generates `frozen-formats.json`, the shipped freeze
  commitment, from the store registry; `tests/test_frozen_manifest.py` regenerates it in
  process and refuses any drift from the committed file.
- `generate_harness_discovery.py` - renders the thin `.claude/skills/<name>/SKILL.md` and
  `.agents/skills/<name>/SKILL.md` files, plus the generated block in `AGENTS.md`, from the
  canonical knowledge documents under `packages/tcip-mcp/src/tcip_mcp/knowledge/`. Run after
  adding, renaming, or re-describing a document; `tests/test_harness_discovery_generated.py`
  fails when a checked-in generated file is stale.
- `verify_skill_tools.py` - guardrail holding every tool name in agent-facing prose to the MCP
  tool registry: a fabrication check over every table whose header's first column is "Tool", and
  an orphan check for a registered tool no surface names anywhere.
- `verify_skill_traits.py` - flags any trait-like token in a crop/domain `SKILL.md` that
  isn't a real name in `crops.yml`, to catch fabricated trait names in skill prose.
- `generate_favicon.ps1` - renders the browser-tab favicon from the source logo: crops its
  transparent margins, resizes the result to 512x512, and writes it plus a 32x32 copy to the
  frontend's public assets.
- `smoke_phenology_e2e.py` - offline end-to-end smoke: builds synthetic geolocated imagery,
  runs `build_plant_mapping` + `deliver_phenology_milestones`, and asserts the delivered CSV and the
  measurement-integrity refusal both behave correctly.
- `smoke_terminal_e2e.py` - live smoke exercising the in-app agent terminal against the real
  `claude` CLI end to end (spawn, websocket attach, prompt, response). Costs one model turn.
- `smoke_fence_e2e.py` - live smoke confirming the fenced `claude` CLI actually refuses to
  edit platform internals when run through the in-app terminal's own settings. Costs one
  model turn.
