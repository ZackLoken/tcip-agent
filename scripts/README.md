# scripts/

Not a Python package (no `__init__.py`); each file is a standalone entry point run with
`python scripts/<name>.py`. Read a script's own module docstring and `--help` for exact
usage; this index only says what each does and whether it's a general capability worth
reaching for again versus one built narrowly for a specific past investigation.

## General capabilities

- `calibrate_operating_point.py` - runs one model pass over a disjoint calibration/holdout
  split of a labeled dir, derives a count-unbiased detection operating point, checks its
  held-out count bias, and prints the full provenance + gate evidence for inspection. It writes
  nothing; validated claims are minted only by the audited doors. `--split-manifest-dir`
  (with `--subject` required beside it) restricts the calibration universe to one capture
  date's held-out side of a split manifest instead of every labeled stem. `--project-root`
  is required: the checkpoint must be registered under it (`register_model`, explicit mode),
  and the script refuses one that is not.
- `adopt_store.py` - moves a root's existing record and log files into a store database, so
  the database backend can own state that predates it. Decodes every file it will adopt and
  refuses the whole root on the first that will not, publishes the database only after
  re-checking that nothing changed under it, and leaves blob files exactly where they are. A
  root that already holds a database is planned too, and takes in exactly the stores that
  database has never held. `--project` conforms a project's own roots and its registered
  datasets' at once.
- `_store_bootstrap.py` - imports every store-owning module, which is what registers every
  store, and names the roots a whole project's records live in. Not a standalone script;
  `adopt_store.py` and `export_store.py` both read it.
- `check_dataset_identity.py` - recomputes a dataset's on-disk fingerprint and compares it
  against the fingerprint recorded in `dataset.json` and the project's `.tcip/datasets.json`,
  to catch data that changed or moved since it was registered.
- `generate_frontend_types.py` - renders `frontend/src/api/types.generated.ts` from the pydantic
  models that declare the view-coverage record's shape and the GUI's tab/mode vocabulary
  (`tcip_web.state.GuiVocabulary`), so the browser's types are a projection of the backend's
  rather than hand-transcribed. Run after changing a declared model;
  `tests/test_generated_frontend_types.py` fails when the checked-in module is stale.
- `generate_harness_discovery.py` - renders the thin `.claude/skills/<name>/SKILL.md` and
  `.agents/skills/<name>/SKILL.md` files, plus the generated block in `AGENTS.md`, from the
  canonical knowledge documents under `packages/tcip-mcp/src/tcip_mcp/knowledge/`. Run after
  adding, renaming, or re-describing a document; `tests/test_harness_discovery_generated.py`
  fails when a checked-in generated file is stale.
- `distill_learnings.py` - gathers one project's (or, with `--workspace`, every project's)
  `report_friction` and `write_retrospective` records into one Markdown worksheet of
  recurring themes, for human review before calling `record_distillation_pass`.
- `doctor.py` - read-only scan of a live project for state inconsistencies code audits can't
  see: status-store vs disk disagreements on negatives, registry entries pointing at
  missing/test-fixture checkpoints, provenance smells, orphaned labels, and (`check_data_quality`,
  the retired per-file quality tool folded in there) per-file annotation quality across any
  supported format. Run at session start.
- `export_store.py` - writes a root's database-held records and logs back out as files, in
  the layout each store's locator names, so the doctor, an archive and anything else that
  reads state off disk reads what the database currently holds. Refuses rather than writing
  when two keys would land on one file, and reports any store that was written again while
  its files were being produced. `--project` covers a project's roots at once.
- `list_tools.py` - prints the live MCP tool registry (count + names); the single source of
  truth for "how many domain tools exist," since the count drifts as tools are added/renamed.
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
- `verify_skill_traits.py` - flags any trait-like token in a crop/domain `SKILL.md` that
  isn't a real name in `crops.yml`, to catch fabricated trait names in skill prose.
- `smoke_phenology_e2e.py` - offline end-to-end smoke: builds synthetic geolocated imagery,
  runs `build_plant_mapping` + `deliver_phenology_milestones`, and asserts the delivered CSV and the
  measurement-integrity refusal both behave correctly.
- `smoke_terminal_e2e.py` - live smoke exercising the in-app agent terminal against the real
  `claude` CLI end to end (spawn, websocket attach, prompt, response). Costs one model turn.
- `smoke_fence_e2e.py` - live smoke confirming the fenced `claude` CLI actually refuses to
  edit platform internals when run through the in-app terminal's own settings. Costs one
  model turn.
- `_script_root.py` - shared platform-state-root pinning for an operator script that calls an
  `@audited` tool function directly, outside the MCP server or web backend, so its resolution
  and its audit line land under the intended project root rather than the process cwd. Not a
  standalone script.
- `archive_project.py` - exports an annotation project (images, ground truth, class registry,
  `.tcip` state, experiments and their claimed manifests, every recognized blob home) as a
  portable bundle an `import_project.py` run can restore from elsewhere: a ZIP (`--output-path`)
  or, with `--output-dir`, the identical bundle written as a directory tree; exactly one of the
  two is required. Wraps `tcip_mcp.tools.project_tools.archive_project` with no MCP tool
  registration.
- `import_project.py` - imports an annotation project from a bundle `archive_project.py` wrote,
  a ZIP or a directory tree alike: stages it into private staging, refuses on any bookkeeping,
  collided, undecodable, or unaccounted member, adopts into a database when this process is
  bound to that backend, then moves the staged tree onto the destination. Wraps
  `tcip_mcp.tools.project_tools.import_project` with no MCP tool registration.
- `build_module_inventory.py` - builds a module inventory and real import graph for the repo's
  Python and TypeScript source trees, so `check_architecture_doc.py` can cross-check
  ARCHITECTURE.md's module-ownership tables against the tree it actually describes.
- `check_architecture_doc.py` - verifies ARCHITECTURE.md's module-ownership tables against the
  tree, for CI: every named path exists, and, given a module-inventory JSON, the in-repo-import
  and imported-by counts per row are cross-checked for drift.
- `check_architecture_citations.py` - verifies ARCHITECTURE.md's `file:line` citations against
  the code they quote, for CI; `--fix` rewrites a re-anchorable citation's stale line number in
  place, leaving a genuinely failed citation for a human.
- `verify_skill_tools.py` - guardrail holding every tool name in agent-facing prose to the MCP
  tool registry: a fabrication check over every table whose header's first column is "Tool", and
  an orphan check for a registered tool no surface names anywhere.
- `gate_baseline.py` - runs the quality gate CI actually declares, parsed from
  `.github/workflows/ci.yml` rather than restated by hand, so a local pass predicts a CI pass.
- `generate_frozen_manifest.py` - generates `frozen-formats.json`, the shipped freeze
  commitment, from the store registry; `tests/test_frozen_manifest.py` regenerates it in
  process and refuses any drift from the committed file.
- `generate_frontend_routes.py` - generates the browser's route-path module and the dev
  server's proxy config from the backend's registered FastAPI routes, so the frontend
  references a path by name instead of restating the string a second time.
- `generate_favicon.ps1` - renders the browser-tab favicon from the source logo: crops its
  transparent margins, resizes the result to 512x512, and writes it plus a 32x32 copy to the
  frontend's public assets.
- `conform_project_site.py` - writes or corrects one project's authored site record: the same
  create-only write `record_site` always is, or, with `--replace`, an unconditional overwrite
  for a site typed wrong once or a record damaged by hand.
- `scan_dataset.py` - read-only census of a dataset folder before splitting, validating, or
  training on it: image/label/prediction counts, the detected label format, and which files are
  excluded from every bucket walk because they collide with a prediction bucket's provenance
  stamp. Wraps `tcip_mcp.tools.data_tools.scan_dataset` with no MCP tool registration.
- `score_predictions.py` - scores on-disk predictions against on-disk ground truth (COCOeval):
  an image file returns per-box matches, a dataset dir returns aggregate metrics plus per-image
  TP/FP/FN. Wraps `tcip_mcp.tools.annotation_tools.score_predictions` with no MCP tool
  registration; `--trait` requires `--project` (or `$TCIP_STATE_ROOT`), since a trait's derived
  localization criterion reads the project's own trait registry.
- `triage_predictions.py` - sorts a checkpoint's own predictions by confidence into auto-accept,
  needs-review and unscoreable queues; writes nothing itself. `--auto-threshold` keeps the door's
  own refusal to auto-accept anything until derived from the model's validated confidence
  distribution and a breeder spot-check. Wraps `tcip_mcp.tools.feedback_tools.triage_predictions`
  with no MCP tool registration; `--project` (or `$TCIP_STATE_ROOT`) is required unconditionally,
  since checkpoint verification always reads the registry under it.
- `overlay_reference_grid.py` - renders an image with a labeled reference-grid overlay for
  spatial referencing, and echoes the full grid geometry (`tile_size`, `overlap`, `cols`, `rows`,
  width, height) so the caller can hand it straight to `segment_prompt(grid_cells=...)`. Wraps
  `tcip_mcp.tools.vision_tools.overlay_reference_grid` with no MCP tool registration, still
  `@audited` since it writes an artifact and a platform audit line; `--project` (or
  `$TCIP_STATE_ROOT`) is required unconditionally.
- `visualize.py` - one entry point for the common renders (annotations, predictions, a
  GT-vs-prediction comparison, or a dataset sample grid), saved to `.tcip/artifacts/viz/`. Wraps
  `tcip_mcp.tools.vision_tools.visualize` with no MCP tool registration, still `@audited` since
  it writes an artifact and a platform audit line; `--project` (or `$TCIP_STATE_ROOT`) is
  required unconditionally.
- `render_failure_cases.py` - finds and renders the worst predictions for failure analysis,
  ranked by a count-mismatch-plus-low-confidence heuristic; not a substitute for
  `score_predictions(detail=True)`'s IoU-matched TP/FP/FN when mislocalization is the question.
- `plant_aware_group_splits.py` - derives a plant-aware group key for `draw_splits` over
  per-stem georeferenced raster datasets, so every capture of one physical plant across every
  date lands in the same split side instead of only grouping by tile prefix.
- `preflight_config.py` - validates a training configuration before launching: structural checks
  and a builder import always run; `--smoke` also builds the model and runs
  `check_model_contract`; `--overfit` (with `--smoke`) additionally runs the voluntary
  `overfit_check` diagnostic. Wraps `tcip_mcp.tools.training_tools.preflight_config` with no MCP
  tool registration; `--project` (or `$TCIP_STATE_ROOT`) is required unconditionally.
- `shp_to_plant_csv.py` - converts a plant-locations shapefile into `read_plant_csvs`' CSV
  schema, handling both point and polygon source geometry and validating its own output by
  reading it back through the same reader before reporting success.
- `inspect_compute_resources.py` - reports the host's current compute headroom (CPU, memory,
  GPU free bytes, active training run count) to reason with before launching another concurrent
  training/HPO run; not an enforced cap. Wraps
  `tcip_mcp.tools.training_tools.inspect_compute_resources` with no MCP tool registration.
- `cross_family_ask.py` - poses one identical question to several agent harnesses (claude,
  codex, antigravity) and records comparable answers: the exact prompt, argv, stdout/stderr, the
  extracted response, and run metadata describing what the harness was and how long it took.

## One-off conforms

Each script here carries existing on-disk records onto a shape one specific past change now
requires, and says so in its own docstring: a one-off operator fix, never a runtime migration.
Run once against a root whose state predates the change it names; reaching for one again means a
fresh root predates another change, not a capability to build on.

- `conform_classified_predictions.py` - conforms a project's classified prediction buckets to the
  writer rail's `(subject, attribute)` stamp and to the shape a classified prediction now writes
  (the object class in `subject`, the decoded value under `attributes[attribute]`): stamps a
  bucket whose scope is sourced from its own training run, `--like` another bucket, or the
  operator, and rewrites a bucket's value-in-subject documents once every record in it conforms.
  Reports, never rewrites, a reviewed bucket, an unconformable record and a ground-truth record
  that may itself be a mislabeled value. `--plan` previews and writes nothing.

## Pilot/incident-bound

Written for one past investigation against one sample project (`Valley_Farm`, via the shared
`_paths.py` helper); not written to generalize to another project or dataset.

- `_paths.py` - shared path resolution (`vf_root()`, `repo_root()`); not a standalone script.
  The investigation scripts it served are gone.
