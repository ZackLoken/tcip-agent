# scripts/

Not a Python package (no `__init__.py`); each file is a standalone entry point run with
`python scripts/<name>.py`. Read a script's own module docstring and `--help` for exact
usage; this index only says what each does and whether it's a general capability worth
reaching for again versus one built narrowly for a specific past investigation.

## General capabilities

- `calibrate_operating_point.py` - runs one model pass over a disjoint calibration/holdout
  split of a labeled dir, derives a count-unbiased detection operating point, checks its
  held-out count bias, and persists the full provenance + sweep to
  `.tcip/experiments/<id>/operating_point.json`.
- `check_dataset_identity.py` - recomputes a dataset's on-disk fingerprint and compares it
  against the fingerprint recorded in `dataset.json` and the project's `.tcip/datasets.json`,
  to catch data that changed or moved since it was registered.
- `distill_learnings.py` - gathers one project's (or, with `--workspace`, every project's)
  `claude_reports` and `project_retrospective` records into one Markdown worksheet of
  recurring themes, for human review before calling `record_distillation_pass`.
- `doctor.py` - read-only scan of a live project for state inconsistencies code audits can't
  see: status-store vs disk disagreements on negatives, registry entries pointing at
  missing/test-fixture checkpoints, provenance smells, orphaned labels. Run at session start.
- `list_tools.py` - prints the live MCP tool registry (count + names); the single source of
  truth for "how many domain tools exist," since the count drifts as tools are added/renamed.
- `prove_test_fails_before.py` - extracts a baseline revision with `git archive` and runs a
  given test file against it, to prove a regression test actually fails without its fix
  rather than passing vacuously everywhere.
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
