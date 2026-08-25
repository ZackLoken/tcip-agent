# scripts/

Not a Python package (no `__init__.py`); each file is a standalone entry point run with
`python scripts/<name>.py`. Read a script's own module docstring and `--help` for exact
usage; this index only says what each does and whether it's a general capability worth
reaching for again versus one built narrowly for a specific past investigation.

## General capabilities

- `calibrate_operating_point.py` - runs one model pass over a disjoint calibration/holdout
  split of a labeled dir, derives a count-unbiased detection operating point, checks its
  held-out count bias, and prints the full provenance + sweep for inspection. It writes
  nothing; validated claims are minted only by the audited doors.
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
- `conform_registry_metrics_source.py` - conforms a project's registry entries to carry
  `metrics_source` (which path produced an entry's metrics: `trainer`/`training_source`/`caller`/
  `null`), for an entry registered before the field existed. `--plan` previews; an
  experiment-tagged entry is refused until the operator states its source with
  `--source NAME=VALUE`, since the tag alone can't be trusted to say which path produced it.
- `distill_learnings.py` - gathers one project's (or, with `--workspace`, every project's)
  `claude_reports` and `project_retrospective` records into one Markdown worksheet of
  recurring themes, for human review before calling `record_distillation_pass`.
- `doctor.py` - read-only scan of a live project for state inconsistencies code audits can't
  see: status-store vs disk disagreements on negatives, registry entries pointing at
  missing/test-fixture checkpoints, provenance smells, orphaned labels. Run at session start.
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
  runs `build_plant_mapping` + `compute_phenology`, and asserts the delivered CSV and the
  measurement-integrity refusal both behave correctly.
- `smoke_terminal_e2e.py` - live smoke exercising the in-app agent terminal against the real
  `claude` CLI end to end (spawn, websocket attach, prompt, response). Costs one model turn.
- `smoke_fence_e2e.py` - live smoke confirming the fenced `claude` CLI actually refuses to
  edit platform internals when run through the in-app terminal's own settings. Costs one
  model turn.

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

## Dead

- `review_evidence.py` - evidence loader for a `.review-corpus/` review journal that is
  gitignored and does not exist on this (or any fresh) clone; every code path in it is
  unreachable without that corpus.
