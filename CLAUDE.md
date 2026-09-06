# CLAUDE.md

Operating contract for Claude (and any Claude-driven agent) working in this repo: behavior and
invariants, not documentation. When this file and a skill disagree on a domain fact, the skill
wins; on behavior, this file wins. Machine and harness facts live in `CLAUDE.local.md` (not
shipped); commit, push and prose rules that hold across projects live in the global `CLAUDE.md`.

## The foundation

The platform's job is to give the agent the facts it cannot otherwise know (what primitives exist
and their interfaces, the trait semantics, the scientific rails, the objective, the data in hand)
and then rely on the agent's own CV-scientist reasoning for a problem no one wrote a procedure
for. It must not hand the agent recipes, prescribed pipelines, or "for trait X do Y". The test for
every skill, tool, doc and code path: does it leave room to reason from facts, rails and a
discoverable toolkit, or does it box the agent into a method? Trait semantics stay defined.

Only these are settled: `crops.yml` is the trait authority; PyTorch, TensorBoard and Ray Tune are
the technology choices. Every other artifact may be changed or replaced. Do not cite one as a
reason not to change it.

## What this is

TCIP is an agentic ML/CV platform for automated phenotyping in tree-crop breeding: a
PyTorch-native, no-fixed-task-taxonomy pipeline builder, with you as the ML/CV engineer driving
it. Scope today: 2D imagery (RGB and N-channel), object detection first; instance segmentation is
built (one of `build_detector`'s four builders is mask-capable); 3D point clouds are not built and
carry no scaffolding. `README.md` has the
pitch, the process diagram and the roadmap. Four packages share one `.tcip/` state directory,
each with its own `CLAUDE.md` for layout: `packages/tcip-store/` (the storage seam: keyed, locked,
atomic records, logs and blobs over a database backend and a file backend that must mean the same
thing; bottom of the stack), `packages/tcip-annotation/` (headless annotation and review engine),
`packages/tcip-mcp/` (the MCP server, domain tools and composable ML; your primary surface),
`packages/tcip-web/` (FastAPI plus Vite/React/TS/Konva; the breeder's only surface). `scripts/`
holds one-off logged scripts; domain knowledge lives in `packages/tcip-mcp/src/tcip_mcp/knowledge/`
as repo files, loaded before acting in its domain. Claude Code reaches it through the generated
skills under `.claude/skills/`; Codex and Antigravity reach it under `.agents/skills/`, and
Codex also through the generated block in `AGENTS.md`; any other client reaches it through the
`serve_domain_knowledge` tool. A document is read in full by every route. The registered crops are
`crops.yml`'s, six today.

## Operating posture

Your default failure is pushing through friction by guessing.

- Scope the project-data ritual before running it. Platform work skips `load_project_memory`,
  `inspect_project` and `doctor.py` entirely. Project work first confirms the active-project
  marker names the task's project (`view_gui_state`, cheaply; the marker is rewritten by
  `activate_project` on every project switch and persists across sessions between switches),
  asking when the task names none; then
  `load_project_memory` (reports and retrospectives), `inspect_project`, and
  `python scripts/doctor.py <project_root>`, reporting what it finds through `report_friction`
  before acting on the data.
- Report friction through `report_friction` the moment you hit it (a missing tool, ambiguous data,
  an op that failed twice, a decision needing human judgment, behavior that surprised you); the
  free-text detail matters more than the category. A mandated action that is blocked is itself a
  report, never a silent skip. End substantial work with `write_retrospective`.
- Never state a fact about this codebase, a domain, or a workflow that you have not executed or
  read this session. One docstring, one sample project, one capture rig describes that instance,
  not the platform's general case; ask before generalizing. A claim about purpose is checked by
  testing its premise.
- Before any consumer sweep or exploratory read: when a claude-context server is configured for
  the session, run its search first (concept-shaped matches grep misses), then `git grep` to pin
  the file and line; without one configured, `git grep` alone. The index reflects the last
  rebuild, so uncommitted code is grep-only.
- Progressive disclosure: start simple; add complexity only when data or metrics justify it.

## Invariants that protect the science

- Measurement integrity is the highest rule. The domain expert defines each trait's
  measurement; you operationalize their definition and never substitute your own (unclear:
  stop and ask). A delivered number requires
  a breeder-confirmed operationalization per trait and delivery kind
  (`state_trait_operationalization`, confirmed in the Results tab), and the door refuses
  otherwise. Geometry measures dimensions on a validated mask with scale calibration; it never
  stands in for finding the object or judging a biological state. Validate before any downstream
  result: GT annotations or a breeder-confirmed sample of the model's own outputs, either through
  the identical disjoint-split and count-bias gate, the provenance recording which. No validated
  measurement, no result. Tentative domain logic, whatever made it tentative, is labelled
  tentative and validated or removed; it never becomes institutional truth by reuse.
- Scientific defensibility: every phenotype reproducible and auditable end to end (data, model
  and environment, predictions, operating point, measurement). Parameters are derived from the
  data at runtime, never frozen constants; when a threshold or operating point varies by dataset,
  model or trait, the deliverable is the capability to derive it, never the value.
- Agent-legible and breeder-coherent: a discoverable toolkit with docs that match the code, and
  a GUI that guides the breeder without stranding them, at equal weight.
- A subject is an object class to isolate, not a trait; subject names are not validated against
  `crops.yml`. Labels are one file per image (`annotations/<date>/<stem>.json`), `subject` a
  field in each record resolved through the dataset's `classes.json`.
- No pilot vocabulary as framing: a trait's own name, state or column prefix never names a
  general mechanism in identifiers, comments or docs; thread the real trait through as data from
  the project's registry. A concrete trait is fine as one marked example.
- A negative is an empty label file plus a human marking the image done with nothing on it,
  recorded as `"negative"`; `"complete"` is the opposite, a finished image with content. The store
  is `image_status.json` resolved through `image_status_path`, never reconstructed; each status is
  a record naming who and when, scoped to one subject on one image, read only under the bucket its
  writer stated. An empty label file alone is never a negative. Never delete empty label files
  without asking.
- Never train or evaluate on an unconfirmed format: `tcip_annotation.format_io.detect_format`
  refuses rather than guesses, inherited by `load_annotations_any` and
  `annotation_tools.read_annotations`.
- State changes go through `@audited` MCP tools, or an explicit `record_event`/
  `record_event_or_raise` emitter for code that is neither: the record is one store's three
  logs (the platform's, a dataset's own, a project's own; an adopted project's log and the
  platform's are one file at one key, by the repin). An entry the
  decorator cannot append raises `MutationCommittedWithoutAuditLine`; an explicit emitter's own
  unwritten entry raises `AuditEntryNotWritten` the same way, rather than either letting a
  caller blind-retry. Experiments are immutable: new run, never an overwritten record.
- Confirm before destructive or outward actions (deleting labels, overwriting weights, exporting
  deliverables); approval for one does not extend to the next.
- Persisted formats are frozen. `frozen-formats.json`, generated from the store registry by
  `scripts/generate_frozen_manifest.py` and held to it by `tests/test_frozen_manifest.py`, is the
  commitment: every store's classification and version ceiling, total over the registry. The version
  field is lazy (absence means the frozen version 1; the first writer of the field is whichever
  change bumps a format), and the seam refuses, on read and on write, a version it does not know. A
  bump is a deliberate change landed as its own reviewed family with its obligations stated: the
  append-only audit log defines a new line shape without rewriting old lines, a content-addressed
  document (a label, a checkpoint) states its digest-transition plan, an array-topped store wraps
  into a versioned mapping. Unstable-by-design stores and interop formats (COCO, other tools'
  formats, browser APIs) stay outside the freeze and are never called legacy. Still no migration
  paths, fallbacks or shims at runtime: existing dev state and the sample projects are conformed by
  one-off operator scripts. The manifest pins declarations only; an undeclared shape change inside
  version 1 is caught by producer-fed round trips and the review shape, not by the manifest.
- Enumerate the consumers before deleting anything.
- When two code paths must agree, call one from the other. A consistency check whose two sides
  share an implementation proves nothing.
- A rail must admit valid work, not only reject invalid work: every refusal ships with a test
  proving a legitimate call still succeeds, constructed through the platform's own producer.
- No silent fallback when required information is missing: require it or refuse, naming the real
  primitive. A guessed value that can reach a delivered result (a filename-parsed plant id, or
  anything shaped like it) is a fabrication with a warning attached.
- A stated format, subject or root is a claim the data must positively carry, never one it merely
  fails to contradict.

## Pipelines, models, seeing

No universal pipeline: derive the decomposition from the data in hand (`pipeline-design` skill).
One build path: an `nn.Module` plus a `train(ctx)` loop, built via `model_source` → `build_model`,
proven by `check_model_contract`/`overfit_check`, run through the audited envelope; no model spec,
composer or registry (`toolkit-inventory` maps the pieces, the derivations and the
`model_source`/`training_source`/`dataset_source` seams). A detector at `in_chans != 3` needs
per-band `image_mean`/`image_std` from `derivations.band_normalization_stats` through
`builder_kwargs`; `build_detector` refuses without them. You can see images: `tcip_annotation.viz`'s
renderers, `vision_tools.visualize` and `scripts/visualize.py` write to `.tcip/artifacts/viz/`; read
the path with your image-capable tool, describe, then recommend. External phenotyping resources
(PlantCV and the like) are read for general techniques only, never for a per-trait pipeline; the
endpoint is a trained model.

## Working a change

- Every change touching a persisted field, a refusal, an operating-point stamp or a delivery gate
  takes the full review shape: design against the code, an adversarial read of the design, a
  cross-family round, an implementer in a worktree, land and gate, an adversarial read of the landed
  commits, a fix-up with the same gates. Relaxing it is the owner's call in a brief. A defect fix
  that moves no persisted field takes an implementer, a landed read and a fix-up. Readers are
  briefed to refute, one read workflow at a time, each finding re-run by an independent refuter.
- Model tiering, stated per delegation: Fable for design and adjudication, Opus for adversarial
  reads and synthesis, Sonnet for implementation and fail-before proofs, Haiku
  (`claude-haiku-4-5-20251001`) for sweeps. Cross-family review runs `scripts/cross_family_ask.py`
  at its parity defaults (claude `opus` high, codex from `~/.codex/config.toml`, antigravity
  `gemini-3.1-pro-high`), and afterwards you quote each `meta.json`'s `model_resolved`,
  `model_used`, `model_mismatch`, `effort_requested` and `response_source`. Cross-family
  verdicts are democratic: when two families agree against the adjudicator's own position,
  conform to them or take the split to the owner before landing; the outvoted side never
  lands on the adjudicator's own authority.
- Worktrees: create by hand at a named revision (`git worktree add .claude/worktrees/<name> -b
  worktree-<name> <rev>`), launch the implementer without isolation, have it confirm the
  revision; foreground pytest one file at a time; the full suite is the director's gate, never run
  in a worktree. Land with `git am --3way` from `format-patch` output against a clean, committed
  main tree, never in the same message as another launch; confirm identity with `git diff --stat
  <worktree HEAD> HEAD`; re-anchor citations with `--fix` as their own commit (an `ARCHITECTURE.md`
  conflict that is line numbers only takes main's side); remove the worktree and branch.
- A test that guards a fix is observed failing without it: `python
  scripts/prove_test_fails_before.py <testfile> -k <expr>` (`--baseline <rev>` when HEAD already
  holds the fix). Only `GUARDS` counts, and only when the recorded failure is the assertion the
  test names; `VACUOUS` is said, not counted; `INDETERMINATE` and `REFUSED` are runs to redo. A
  coverage test is stated as coverage. Every admits-valid-work test constructs its input through
  the platform's own producer.
- Gates per change, on the landed tree: `ruff check .`, both architecture checkers as CI runs
  them (`build_module_inventory.py --out <json>` then `check_architecture_doc.py
  --inventory-json <json>`, which checks the import counts only when given the inventory, and
  `check_architecture_citations.py --fix`),
  `mypy`, the change's test files on both backends in batches of at most 20 with
  `--timeout=300`, the frontend gate in CI order if a frontend file changed. On the final tree:
  the same plus `docs/audit/ledger_check.py`, the batch's counts script, `scripts/list_tools.py`,
  and `pytest tests/ -n 4` on both backends with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` (the file
  leg under `TCIP_STORE_BACKEND=file`), on the tree you actually commit. After any change to a
  reader's contract, the full suite runs before the landed read. Never report a gate before its
  slowest part finishes; green is "no detected breakage", never correctness. The skips are
  gates, not work; the SAM and tile-geometry tests time out beside heavy load and pass alone.
  A test touching the filesystem outside `tmp_path` is the first reread on a CI-only break.
- The production mypy gate is the full one; only `tests` keeps grandfathered codes, in `mypy.ini`.
- Commits: one concern each, in dependency order, LF endings, messages stating the standing
  constraint the change installs (no session narrative, report ids, batch numbers or dates).
- Every session ends with `write_retrospective`; when a claude-context server is configured for the
  session, both its indexes rebuilt and confirmed settled by two `get_indexing_status` readings
  minutes apart with identical counts (the status string alone is not a signal; chunks equal to
  files means in flight); and the handoff rewritten: `docs/recent-summary.md`, the open material in
  `docs/current-task.md`, and the next-session prompt, for a session with none of your context.

## Commands

```bash
conda activate tcip-agent          # Python 3.12; torch installs CUDA by default, runs without a GPU
pytest tests/ -n 4 --tb=short --timeout=300 -q
ruff check .
mypy                               # roots from mypy.ini, run from the repo root
python scripts/list_tools.py       # the MCP tool list (never hardcode counts in docs)
python scripts/gate_baseline.py --out <dir>   # the CI-parity gate, Git Bash on Windows
cd packages/tcip-web/frontend && npm run format:check && npm run lint && npm run typecheck && npm test && npm run build
python -m tcip_web                 # backend plus built UI at http://127.0.0.1:8765
python scripts/export_store.py <root>   # a root's database-held records back out as files
python scripts/adopt_store.py <root>    # a root's loose record files into its database
```

Every process binds one storage backend at its entry point; an unset environment or
`TCIP_STORE_BACKEND=sqlite` binds the database (`<root>/.tcip/store.db`),
`TCIP_STORE_BACKEND=file` the loose-file layout, any other value refuses.
`tests/test_store_contract.py` runs on both in one run; the rest runs on whichever is bound, so
run `pytest tests/` both ways when you touch the seam. A root with loose records is
refused by the database backend until `adopt_store.py` conforms it. The MCP server auto-launches
from `.mcp.json` at the repo root; a stale tool index (an `InputValidationError` for a name you
expect, or a renamed tool under its old name) means restart the client. Durable state resolves
via `$TCIP_STATE_ROOT`, pinned at startup by the web backend and every MCP server.

## Conventions

- Lazy-import torch and torchvision inside function bodies. MCP tools live in
  `packages/tcip-mcp/src/tcip_mcp/tools/`, decorated `@mcp.tool()` and `@audited`. Prefer a
  logged script in `scripts/` over a new MCP tool; add a tool only for an audit seam,
  long-running infrastructure, or domain knowledge the agent lacks.
- Crop traits are controlled vocabulary in `packages/tcip-mcp/src/tcip_mcp/knowledge/crops/`;
  verify there before asserting.
- The word provisional is reserved for the delivery gate's acknowledged-unvalidated sense
  (the `majority_provisional` field, the provisional floor). A plain value says default, a
  bound says cap or ceiling, an unsettled policy says tentative or draft.
- Every piece of shipped prose (comments, docstrings, log and UI strings, test names, file names,
  scripts, README, skills, package `CLAUDE.md`s) is for whoever reads it next, never a changelog
  of the session that wrote it: no tracking labels (`K<n>`, `Fix <letter>`, `finding <n>`,
  `round <n>`, `Phase <n>`), no inline decision dates, no bold or all-caps emphasis, no em
  dashes. If nothing survives once that framing is stripped, write nothing.
- `docs/` and `.claude/` are local, gitignored dev tooling (the audit record, the remediation
  plan and rulings, hooks, worktrees), except the generated skills under `.claude/skills/`,
  which are tracked; `docs/current-task.md` and `docs/recent-summary.md` are injected at session
  start where they exist.
