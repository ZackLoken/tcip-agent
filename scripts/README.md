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
- `drop_annotation_stats_image_status.py` - drops the dead `image_status` key from a project's
  `annotation_stats` record, which every writer put there empty and nothing read. `--plan`
  previews; a record whose `image_status` is not empty is refused, since that would mean a writer
  this script does not know about.
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
- `verify_claims.py` - lists every claim-shaped sentence (fallback/never/always/cannot/etc.)
  that a diff adds to comments and docstrings, so each gets verified deliberately instead of
  shipping as unverified reasoning.
- `verify_doc_examples.py` - checks that every Python code example in a skill or source
  docstring parses, imports what it claims, and would bind against the real signature it
  documents.
- `verify_skill_traits.py` - flags any trait-like token in a crop/domain `SKILL.md` that
  isn't a real name in `crops.yml`, to catch fabricated trait names in skill prose.
- `watch_agent_chat.py` - reads the in-app (breeder-lane) agent terminal's own
  `~/.claude/projects/.../<session>.jsonl` transcript so an orchestrating session can see
  that chat without a copy-paste round trip.
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
- `verify_citations.py` - checks that literature citations point at real code, real papers, and
  real sentences: each reference is retrievable, each anchor resolves to a real symbol, every
  `# cite:` marker resolves to a reference and vice versa, and every quote appears verbatim in
  the fetched PDF's text.
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
- `conform_model_registry_paths.py` - conforms a project's model registry index into the
  entries-mapping shape and respells every entry's `checkpoint_path` relative to the project
  root, per the storage convention.
- `conform_project_site.py` - writes or corrects one project's authored site record: the same
  create-only write `record_site` always is, or, with `--replace`, an unconditional overwrite
  for a site typed wrong once or a record damaged by hand.
- `drop_trait_spec_provenance.py` - conforms a project's trait-spec records to drop the retired
  free-text `provenance` field, and removes stale YAML copies an earlier YAML-to-record conform
  step left behind.
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

## One-off deletions

- `clear_dev_history.py` - removes a root's development-era `friction_reports` records,
  `retrospectives` records, and the `audit_log` and `learning_capture` logs' entries, before the
  root reaches an alpha tester: the production audit trail a tester reads serves a different
  purpose than the remediation program's own records. Leaves a stale exported loose copy of any
  of the four under a database-backed root as it is; the seam's own deletes tombstone what they
  remove, so the next `export_store.py` run is what reconciles the file, from that tombstone.
  Writes one closing audit line naming the operator (`--by`) and the reason (`--reason`), both
  required and never defaulted, carrying the counts removed, so the production log's first line
  states that development history was cleared and by whom. Touches nothing else: not
  annotations, images, predictions, models, experiments, caches, or any other store. `--plan` is
  the default and mutually exclusive with `--apply`.

## One-off conforms

Each script here carries existing on-disk records onto a shape one specific past change now
requires, and says so in its own docstring: a one-off operator fix, never a runtime migration.
Run once against a root whose state predates the change it names; reaching for one again means a
fresh root predates another change, not a capability to build on.

- `conform_cal_holdout_locks.py` - conforms every pre-existing `cal_holdout_split_lock` record
  under a root to carry `split_manifest_dir`, the key every lock this family writes now
  declares. A one-off operator fix for a record written before this family, never a runtime
  migration.
- `conform_classified_predictions.py` - conforms a project's classified prediction buckets to the
  writer rail's `(subject, attribute)` stamp and to the shape a classified prediction now writes
  (the object class in `subject`, the decoded value under `attributes[attribute]`): stamps a
  bucket whose scope is sourced from its own training run, `--like` another bucket, or the
  operator, and rewrites a bucket's value-in-subject documents once every record in it conforms.
  Reports, never rewrites, a reviewed bucket, an unconformable record and a ground-truth record
  that may itself be a mislabeled value. `--plan` previews and writes nothing.
- `conform_dataset_registry_paths.py` - conforms a project's dataset registry entries onto the
  relative-path row `register_dataset` now writes, for a project registered before that change.
- `conform_delivery_events.py` - checks a project's stored `delivery_events` records against the
  current `DeliveryEventRecord` shape and names, by `event_id`, any that no longer validate. The
  one write it makes is forwarding `acknowledged_by` and `acknowledgement_reason` to null on a
  record lacking exactly those two keys, since a delivery predating them was made under no
  acknowledgement; every other missing key is named, never rewritten, since its value was never
  computed for that delivery. `--plan` previews and writes nothing.
- `conform_image_stem_collisions.py` - censuses a project's registered dataset roots for a
  bucket already holding two logical identities under one case-folded stem, printing each with
  its files, the one the un-corrected reader still serves and the records made against it.
  `--plan` is the default; `--apply` takes one `--keep <path>` per collision and parks every
  other file under `.tcip/collisions/<bucket>/`, recording the move. `--plan` writes nothing.
- `conform_plant_mapping_records.py` - rewrites a project's stored plant-mapping records from
  the retired `plant_csvs` list to a `plant_registry` reference, registering the CSVs those
  records named under an operator-stated registry name (`--crop` and `--site` are the expert's
  facts), adds `supersedes: null`, and appends a fresh `plant_mapping_built` receipt naming each
  rewritten record's new digest; prints the delivery events whose cited mapping digest the
  rewrite strands. `--plan` previews and writes nothing.
- `conform_job_registry_roots.py` - stamps a job-registry document's own root (inference, review
  priority queue, HPO) onto every summary it holds that predates the `platform_root` field,
  since a document lives at the key that root is stored under, not a value to guess at
  rehydrate time. `--plan` previews and writes nothing.
- `conform_metrics_marker.py` - stamps the `metrics_logged` marker onto every experiment a
  root's status record predates, so `is_pristine` reads the marker instead of scanning the
  metrics log.
- `conform_region_completeness_attested_view.py` - write-forwards an empty `cells_attested_view`
  map onto a dataset's stored `region_completeness` records written before the key existed, since
  a record from before it predates any scale provenance to carry. `--plan` previews and writes
  nothing; the write happens inside the same lock the completeness route itself takes.
- `conform_registry_experiment_id.py` - conforms a project's registry entries to carry
  `experiment_id` (the run whose completion bound an entry, `null` for one no run bound), for an
  entry registered before the producer-binding field existed. `--plan` previews; an entry
  carrying a leftover `experiment:<id>` tag is refused, since the tag was never verified and no
  run record exists to check it against; re-register it through `register_model_from_experiment`
  instead.
- `conform_schema_version_reset.py` - rewrites a root's frozen-store documents that still carry
  an explicit `schema_version: 2` (a model registry index, a prediction bucket's sidecar
  documents, a `confidence_sweep` record) to carry no field, per the version-1 reset every frozen
  store now ships at. Names, and does not touch, any `audit_log`/`experiment_validations` line
  still carrying `2` (both append-only; `clear_dev_history.py` clears dev-era `audit_log` lines,
  `experiment_validations` is cleared or rewritten by nothing, and a stamp naming a refused row
  there is re-earned through that document's own calibration door). For a `confidence_sweep`
  record whose filename already equalled its own body's digest before the strip, names the
  filename the rewrite orphans (nothing in the platform reads a stored curve back by that
  identity today). `--plan` previews and writes nothing.
- `conform_view_coverage_viewing.py` - conforms a dataset's stored `view_coverage` records'
  `viewing` sub-object to the current `CoverageViewing` shape, mapping the old string forms of
  `stats_source` and `display_bounds` to the new structured ones. `--plan` previews; a `viewing`
  it cannot parse is refused by image name and the dataset left untouched.
- `conform_working_scale_at_write.py` - renames a dataset's stored `region_completeness`
  attestations' `working_scale_bar_at_write` key to `working_scale_at_write` and nulls the
  value, since the annotation-derived bar it once held cannot be reconstructed as the
  breeder-set zoom the new key carries. `--plan` previews; refuses a root with no `.tcip`
  directory by name.
- `restamp_dataset_fingerprint.py` - restamps a bare legacy dataset fingerprint onto the
  formula-version-prefixed form `dataset_fingerprint` now returns, refusing rather than papering
  over a fingerprint mismatch it cannot explain.

## Pilot/incident-bound

Written for one past investigation against one sample project (`Valley_Farm`, via the shared
`_paths.py` helper); not written to generalize to another project or dataset.

- `_paths.py` - shared path resolution (`vf_root()`, `repo_root()`) for the scripts below;
  not a standalone script.
- `compute_disagreements.py` - summarizes GT-vs-prediction disagreement candidates per image
  at several confidence thresholds, tuned to that project's own catkin bbox size/aspect ratio.
- `foreground_fn_candidates.py` - computes foreground-only high-confidence false-negative
  candidates per image and writes a JSON sidecar for manual review.
- `render_candidates_tile.py` - renders a tile with GT and numbered FN candidates (reads the
  sidecar `foreground_fn_candidates.py` writes) for visual review.
- `inspect_baseline_weights.py` - prints framework/model metadata from that project's baseline
  `weights.pt` checkpoint.
- `inspect_gps_exif.py` - prints GPS EXIF for a sample of that project's images, to check
  whether capture used RTK or phone-internal GPS.
