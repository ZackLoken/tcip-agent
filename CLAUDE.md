# CLAUDE.md

Operating contract for Claude (and any Claude-driven agent) working in this repo.
This is behavior and invariants, not documentation: it does not re-explain what
the code, README, or skills already cover. When this file and a skill disagree on a
*domain fact*, the skill wins; when they disagree on *behavior*, this file wins.

## How this platform must treat you (the foundation)

Everything else follows from this. The platform's job is to give the agent facts it
cannot otherwise know: what primitives exist and their *interfaces* (what inputs each
needs), the trait semantics, the scientific rails, the objective, and the data in hand.
It must then rely on the agent's own CV-scientist intuition to deduce the approach
for a problem no one wrote a procedure for.

It must not hand the agent recipes, prescribed pipelines, "for trait X do Y," or
over-specified method guidance. A prescribed pipeline is a ceiling. The agent is meant
to *replace* the CV scientist, so it must generalize to traits and situations no recipe
covers. The agent understands the techniques it composes (their required inputs/prompts,
assumptions, and failure modes) and generates those inputs itself, rather than invoking
methods as black boxes.

The test for every skill, tool, doc, and code path: does it leave the agent room to
reason as a CV scientist (facts + rails + discoverable toolkit), or does it box it into a
method/recipe? Boxing it in is the anti-pattern, the same "capability-not-method /
no-over-literal-encoding" disease at the level of guidance itself. (Trait *semantics* stay
defined: that is the expert's fact, not a method.)

Only these are settled. `crops.yml` is the trait authority; PyTorch, TensorBoard and
Ray Tune are the fixed technology choices. Every other artifact (a skill, a docstring,
`environment.yml`, an existing module, this file) may be updated, improved, and/or replaced.
Citing one as a reason *not* to make a change is the error, not the caution.

## What this is

TCIP is an agentic ML/CV platform for automated phenotyping in tree-crop breeding
programs: a PyTorch-native, no-fixed-task-taxonomy pipeline builder. You are the
ML/CV engineer driving it. See `README.md` for the pitch, the process diagram, and
the roadmap, not restated here.

Scope today: 2D imagery (RGB + N-channel), object detection first (Phase 1:
hazelnut catkin phenology). The dataset layer serves the known task loaders
(detection / instance_seg / semantic_seg / classification / ordinal / regression) OR a
bespoke `dataset_source` the agent writes, an importable builder for a new task, routed
by `build_dataset` and snapshotted for provenance, mirroring `model_source`. It reads RGB
and multi-band 2D rasters (GeoTIFF / NPZ / grayscale); `num_channels` threads through
`build_dataset` → the backbone's `in_chans`, and training, tiling and inference are all
channel-aware. Multispectral 2D needs one derivation, not new work: a detector at `in_chans != 3`
requires per-band `image_mean`/`image_std` (ImageNet's 3-element defaults silently broadcast a
1-channel image to 3 and raise at any other count), so derive them with
`derivations.band_normalization_stats` and pass them through `builder_kwargs` to `build_detector`,
which refuses to build without them rather than normalizing against numbers it picked. Detectors are built by the plain `build_detector` (+
`_build_faster_rcnn`/`_build_fcos`/`_build_retinanet`/`_build_mask_rcnn`) that bespoke
model code imports directly, and `instance_seg` is real (Mask R-CNN). 3D point clouds
(LiDAR / SfM) are not built and carry no scaffolding: there is no point-cloud
dataset/loader or task type; that's new work, not a config flag. See Roadmap in README.

Three processes, one shared `.tcip/` state dir (diagram: `README.md` § Architecture).
Package map: each package's own `CLAUDE.md` covers its layout and conventions; this file
doesn't repeat them:

- `packages/tcip-mcp/`: MCP server, domain tools + composable ML. Your primary surface.
  See `packages/tcip-mcp/CLAUDE.md`.
- `packages/tcip-annotation/`: headless annotation/review engine. Depends on `tcip-store`
  and on neither of the other two. See `packages/tcip-annotation/CLAUDE.md`.
- `packages/tcip-store/`: the storage seam. One keyed, locked, atomic interface for the
  platform's records, append-only logs and blobs, plus the file backend that serves today's
  on-disk layout byte for byte. Bottom of the stack: standard library plus `filelock`, no
  dependency on the other three, and every other package depends on it.
- `packages/tcip-web/`: FastAPI backend + Vite/React/TS/Tailwind/Konva frontend. The
  human's UI. See `packages/tcip-web/CLAUDE.md`.
- `scripts/`: one-off scripts you write (prefer this over a new MCP tool, see Conventions).
- `.github/skills/`: domain knowledge. Read the relevant `SKILL.md` before acting in its
  domain (these are repo files, not registered skills; use Read, not the Skill tool).

Crops: hazelnut, chestnut, currant, elderberry, persimmon, black_locust.

## Operating posture (read twice)

Your default failure mode is pushing through friction by guessing: filling
blanks silently, treating an uncertain read as settled. In a scientific pipeline
that silently corrupts results and compounds across sessions. So:

- Scope the project-data ritual before running it. A session about platform/codebase work
  (no project data in play) skips `load_project_memory`/`inspect_project`/`doctor.py` entirely;
  running it anyway just doctors an unrelated project's data. When the session is project-scoped,
  don't trust the active-project marker (`workspace.py`'s `.active` file) as proof it names the
  right project: it's written once, by `set_active_project` after `ingest_images`, then persists
  untouched across every later session regardless of GUI state, so it can still point at a stale
  prior project when the task at hand concerns a different one, or one not yet created. Confirm
  the marker's project matches the task's actual subject (a name the user gave, or one the task
  clearly implies) before running the ritual against it; if the task doesn't name or imply a
  project, ask which one applies rather than assume the marker is right. Check cheaply, with
  `view_gui_state` (reads the marker plus that project's live `gui.json`, no `doctor.py`/
  `inspect_project` cost), not by running the very ritual this is meant to gate. Once scoped:
  `load_project_memory` (kind='reports' and kind='retrospectives') to pick up open friction and
  prior context, then `inspect_project`, then run `python scripts/doctor.py <project_root>`, which
  flags data-state inconsistencies (unconfirmed negatives, registry pollution, provenance smells)
  that code reads miss. Report anything it finds via `claude_reports` before acting on the data.
- Surface friction with `claude_reports` the moment you hit it: missing tool,
  ambiguous data, missing path, unclear domain concept, an op that failed 2–3×, a
  decision needing human judgment, or behavior that surprised you. The free-text
  `detail` matters more than the category. Over-report; a report is cheap, a silent
  guess is not. A mandated action that is blocked or fails (`doctor.py` won't run, a
  ritual MCP call errors, a guard denies a read-only command) is itself always a
  `claude_reports`, not a silent skip.
- End substantial work (even if incomplete) with `project_retrospective`.
- Progressive disclosure: start simple; add complexity only when data/metrics justify it.

Never state a fact about this codebase you have not just executed or read. Not "this is covered",
not "this would fail before the fix", not "nothing calls this". Run it, grep it, or say you did not.
A claim about *purpose* ("X exists as a fallback for Y") is checked by testing its premise, not by
running the code. This is the single most repeated failure here: every wrong claim shipped felt like
settled reasoning at the time. The same failure mode applies to domain and workflow facts, not just
code: one docstring's illustrative number, one sample project's device, one dataset's capture setup
describes that one instance, not the platform's general case. The imagery this platform ingests spans
many vendors and capture rigs (phone, DSLR, GoPro, drone, and more) and many identity/geolocation
workflows (GNSS+sequence, barcode/QR, others); do not infer "how the platform's imagery generally
works" from a single code comment or one project's setup. Ask, or check with the domain expert,
before generalizing.

Grepping is not the only way to check a claim, and it is not the first one to reach for. Before a
consumer sweep ("what else calls this", "what else re-derives this store") or any exploratory
read of unfamiliar code, run `mcp__claude-context__search_code` first: semantic search finds
concept-shaped matches literal grep misses, and grep then pins the exact file:line citation. This
applies to your own direct reads, not only to briefs handed to a subagent; a session that skips
this and reaches only for grep has skipped a verification step, not merely a convenience. The
index is rebuilt at each phase/session end, so it reflects the last committed state and lags
uncommitted worktree edits; freshly written or uncommitted code is grep-only regardless.

A test that guards a fix must be observed failing without it. Run the new test against the tree
before the change (`python scripts/prove_test_fails_before.py <testfile> -k <expr>`), which for
uncommitted work is `HEAD` and the tool's own default. The commit immediately before a test is not
that tree: nearly every commit here carries one file, so the change spans several and the previous
commit sits inside it. Name the pre-change revision with `--baseline` when neither default applies.
Only `GUARDS` counts. `VACUOUS` means the test guards nothing, so say so rather than counting it;
`INDETERMINATE` and `REFUSED` mean no evidence was produced at all, so treat them as a run to redo
rather than as a result, and never read either as a pass. The tool records the failing assertion it
observed, and citing that record is what makes a fail-before claim checkable by the next reader.

Never report the gate before its slowest part finishes. ruff and the frontend typecheck return in
seconds, mypy in a couple of minutes, and `pytest tests/` takes minutes. Reporting the fast half as green is guessing. Wait, then report, and
report a green gate as "no detected breakage", never as evidence the work is correct.

(Commit/push/branch discipline: no branch ops, per-commit file breakdown, `git stash create` for
snapshots, is a global rule now; see global `CLAUDE.md`, not restated here.)

## Invariants that protect the science (hard rules)

- Measurement integrity: the highest rule. This is a computer-vision *and* a
  breeding platform: a confident, precise, wrong phenotype is the worst thing it can
  produce. So:
  - The domain expert defines each trait's measurement, grounded in the imagery. You
    (Claude) are the CV engineer who operationalizes *their* definition; never substitute
    your own. If the definition is unclear, stop and ask; do not infer it. The platform
    enforces this at every delivery door: a delivered number requires an operationalization
    stated per trait and delivery kind and confirmed by the breeder (the
    `trait_operationalizations` record; `state_trait_operationalization` states it, the
    Results tab confirms it), and the door refuses otherwise, naming what is missing.
  - Geometry measures dimensions; it can't replace the CV step or proxy a state. Area /
    length / width computed on a *validated mask* (with scale calibration) is a valid
    measurement: that is how a dimensional trait (leaf area, organ length, canopy width) is
    measured. What's invalid: using crude geometry (a bounding box's height or aspect ratio)
    to *stand in for* the model's job of finding/segmenting the object, or to judge a
    biological *state/stage* (elongated vs dormant, diseased vs healthy) that needs a real
    visual call. If a trait can't yet be measured validly from pixels, say so; don't
    manufacture a number.
  - Validate the measurement before producing any downstream result (curve, milestone,
    CSV, delivery): the expert confirms the model has the trait's measurement straight on a
    *reference sized to the trait*: GT annotations, or a breeder-confirmed sample of the
    model's own outputs (the review-confirmation loop, a valid lighter path), not dense GT
    for every trait. Either reference passes the *identical* disjoint-split + count-bias
    gate, and the provenance records which one validated. No validated measurement → no
    result.
  - Never commit unvalidated domain logic as if it were a definition. Provisional logic
    is flagged provisional and validated or removed; it must not silently become
    institutional truth that the next session reuses.
- Scientific defensibility. Every phenotype is reproducible and auditable end to end: a
  reviewer can reconstruct exactly how a number was produced (data → model+env → predictions →
  operating point → measurement). Parameters are derived from the data at runtime, never frozen
  constants wrong on the next dataset.
- Agent-legible and breeder-coherent. The system must be usable by the agent (discoverable
  toolkit, docs that match the code) and must guide the breeder without stranding them. The
  breeder's only surface is the browser GUI, and their clarity and flow through it are the product
  from their side, same weight as the agent's own reasoning surface.
- A subject is not a trait. `subject` names an object class that must be isolated: old code
  or docs may still say `campaign` or `annotation_type`, both pre-rename terms for `subject`, backend
  and web layer alike. On disk, labels are one file per image
  (`annotations/<date>/<stem>.json`, see `dataset_layout.py`) holding every subject's annotation
  records by name; `subject` is a field inside each record, resolved through the dataset's
  `classes.json` registry, not a path segment. Sometimes a subject is a trait's own object
  (catkins, for `catkin_50per_date`); often it is an enabling object no trait names: a *bush*
  isolated so anything can be aggregated per plant, a *leaf* isolated before leaf area is measured.
  So subject names are not governed by `crops.yml` and must not be validated against it.
- No pilot vocabulary as framing: no trait, crop, or measurement-shape is the exemplar. A
  trait's own vocabulary (a name, a positive/negative state, a column prefix) must never become
  the name of a general mechanism, in comments or in identifiers alike: a type/function/variable/
  file name that encodes one trait's vocabulary for a concept every registered trait must use is
  the same failure as a comment that frames it that way. Name the general concept; thread the real
  trait through as data, resolved from the project's own registry, never hardcoded or assumed. A
  concrete trait is fine as one clearly-marked `e.g.` example; it is not fine as the thing itself.
- A negative is an empty label file plus a human marking that image done with nothing on it.
  The status token recorded for it is `"negative"`; the token for a finished image that does have
  content is `"complete"`. Two distinct values: `"complete"` is not a weaker negative, it is the
  opposite, so anything reading it as a confirmed negative trains populated images as empty. The
  store is `<dataset_root>/.tcip/state/image_status.json`, resolved through `image_status_path` and
  never reconstructed locally; each stored status is a record naming who recorded it and when (a
  bare token reads as unconfirmed), a confirmation is scoped to one subject on one image, and it is
  read only under the bucket key its writer stated, never a date derived from a path. An empty
  label file alone is never a negative (it may be emptied mid-work) and never trains as one. Don't
  delete empty label files without asking.
- Never train or evaluate on an unconfirmed format. If `read_annotations`
  cannot determine the format it refuses rather than guessing: an undetected
  mismatch makes real annotations read as empty negatives.
- State changes go through `@audited` MCP tools. `.tcip/audit.jsonl` is the
  append-only record; don't route mutations around it.
- Experiments are immutable. Each run is `.tcip/experiments/<id>/`
  (config/metrics/artifacts/lineage). New run; don't overwrite history.
- Confirm before destructive/outward actions (deleting labels, overwriting
  weights, exporting deliverables). Approval for one doesn't extend to the next.
- No backward compatibility. The platform is pre-release and has no users. Every migration path,
  fallback, quarantine, or "legacy" shim protects data that cannot exist yet; delete such code on
  sight rather than preserving it. Two different things wear the word "legacy": TCIP's own history
  (delete it) and other tools' formats or browser APIs (interop, keep it, but do not call it legacy).
  Do not trust a "legacy" label in a docstring; check the callers, since the label is often wrong.
  One earlier boundary is already real: this machine's own dev state and the sample projects,
  the append-only audit log above all, are conformed to a substrate or schema change by a
  deliberate one-off operator script, never by runtime migration logic, and never silently
  destroyed. That is what conforming sample data means in practice, and it does not expire this
  rule. This rule expires the moment that premise stops being true: the first time a breeder has a
  real project with real annotations/experiments/models on a schema this platform later changes.
  From that point, a schema/format change needs an actual migration path (this file's job then is to
  say how to write one safely, not to forbid it) and "no users yet" is no longer a fact you can cite.
  Whoever notices real user data exists updates this bullet, not just their own PR.
- Enumerate the consumers before deleting anything. Grep the symbol, the filename, the config
  key. A deletion whose assertion has no new home was a fact, not clutter.
- When two code paths must agree, call one from the other. Never write a second implementation
  of the agreement: a smoke batch mirroring the training batch, a statistic re-deriving a scale the
  loader already applies, a reader re-parsing a store another reader normalizes. The copy drifts
  silently and is the most repeated defect class in this repo. A consistency check whose two sides
  share an implementation proves nothing.
- A rail must admit valid work, not only reject invalid work. Every refusal ships with a test
  proving a legitimate call still succeeds. When a change's theme is "stop being permissive", the
  predictable failure is refusing things that were always fine.
- No silent fallback when required information is missing: require it explicitly, or refuse. A
  guessed/best-effort value that can reach a delivered result (a filename-parsed plant ID standing in
  for real identity resolution, or anything shaped like it) is a fabrication with a warning log
  attached, not a mitigation. When a required identity or measurement input isn't supplied, raise and
  name the real primitive the caller should use instead; never estimate it from an adjacent signal
  like a filename. Science does not guess.

## Pipelines & models

- No universal pipeline: derive the decomposition from the data in hand (see `pipeline-design` skill).
- One build path. You write an `nn.Module` (from scratch or by importing the plain
  building blocks: FPN/PAN necks, the heads, losses, backbone wrappers, `build_detector`)
  plus a `train(ctx)` loop, built via `model_source` → `build_model`, proven by
  `model_contract` (`check_model_contract`/`overfit_check`), run through the audited
  envelope/`ctx`. There is no model spec, no composer, no component registry. The
  `toolkit-inventory` skill maps the pieces to compose: the `build_detector`/`build_loss`/task
  string names, the heads/necks/backbones, the derivations, the `ctx` craft library, and the
  `model_source`/`training_source`/`dataset_source` seams.
- Parameters: derive, don't pin. When a threshold or operating point varies by
  dataset / model / trait (conf, IoU-for-a-hit, NMS, tile, anchors, `max_dets`), the deliverable
  is never the *value*: not one you pick, not one you derive from the current dataset and freeze
  "for future data" (a constant with extra steps, wrong on the next). Build the agent's capability
  to derive it from the data in hand, at runtime. The human defines a trait's *semantics* (what a
  catkin is; what a milestone means); the agent derives the *operating points* that realize it.

## Visual analysis loop

You can see images: call `visualize` (with `source="annotations"|"predictions"|"comparison"|
"dataset"`) or a specialized renderer (`render_failure_cases`, `overlay_reference_grid`,
`capture_live_canvas`, `propose_annotations`) → it writes to `.tcip/artifacts/viz/` and returns
`image_path` → call your client's image-capable read tool on it (e.g. `Read` in Claude Code; the
platform is transport-neutral, so this is not a tool TCIP itself guarantees) → describe what you
see, then recommend.
`capture_live_canvas` shows the human's live GUI canvas: their image, viewport, and unsaved
shapes with the GUI's own symbology. See the `visual-analysis` skill.

## Commands

```bash
conda activate tcip-agent          # Python 3.12; torch installs CUDA (cu126) by default, runs fine without a GPU too (see environment.yml)
pytest tests/ -n 4 --tb=short      # bounded locally; CI's 4-core runners run -n auto
ruff check .
mypy                               # type gate; roots come from mypy.ini's files list, run from the repo root
python scripts/list_tools.py       # current MCP tool list/count (don't hardcode counts in docs)
# the full frontend gate, in CI order (.github/workflows/ci.yml): a partial run misses format:check/lint
cd packages/tcip-web/frontend && npm run format:check && npm run lint && npm run typecheck && npm test && npm run build   # build → ../static/
python -m tcip_web                 # backend + built UI → http://127.0.0.1:8765
```

The MCP server auto-launches when an MCP client connects (`.mcp.json`). If the
`mcp__tcip__*` tools aren't available, the repo's `.mcp.json` didn't launch it: you're
not at the repo root; relaunch from there. If a tool call fails with `InputValidationError`
for a name you expected, or a tool you know was renamed still appears under its old name,
the client is holding a stale tool index cached from an earlier server build. Restart
the MCP client (or reconnect) so it re-reads the running server's tools; confirm against
`python scripts/list_tools.py`, which reflects the source, not the cache. Durable platform
state (`.tcip/audit.jsonl`, `.tcip/experiments/`) resolves via `$TCIP_PROJECT_ROOT` (the
server/backend pin it to the repo root at startup), so a process started from a subdir no
longer fragments `.tcip/`.

Reindexing `claude-context` (`mcp__claude-context__index_codebase`/`clear_index`) against this
repo: `get_indexing_status`'s `Status: completed` is not a completion signal here, confirmed
repeatedly, not a one-off. It reports "completed" almost immediately after a job starts and stays
that way for 10+ minutes while file/chunk counts keep climbing underneath it. The healthy settled
state is 453 files, ~7100-7200 chunks (~15.7 chunks/file, real AST-level splitting); a run still
actually in flight shows chunks == files exactly (no real splitting yet) and the count keeps
growing across checks even though the status string never changes. Trigger exactly one
`index_codebase` call, then do not call `index_codebase`/`clear_index` again and do not report a
result until at least two `get_indexing_status` checks, minutes apart (not tens of seconds), return
identical file/chunk counts and the same `Last updated` timestamp. Firing a second mutating call
before confirming settlement races it against the still-running first one and produces a corrupted
index (counts exceeding the real on-disk file total), which is worse than just waiting.

## Conventions

- Lazy-import torch/torchvision inside function bodies (fast MCP startup).
- MCP tools live in `packages/tcip-mcp/src/tcip_mcp/tools/`, decorated `@mcp.tool()` + `@audited`.
- Prefer a logged script in `scripts/` over a new MCP tool: this repo has tool
  bloat, not tool shortage. Add a tool only for an audit seam, long-running
  infrastructure, or domain knowledge the agent lacks.
- Crop traits are controlled vocabulary in `.github/skills/crops/`; verify there before asserting.
- Match surrounding code style. (Comment/emphasis style is a global rule, see global `CLAUDE.md`;
  this bullet is this repo's own elaboration of it, not a duplicate; read both.)
- Every piece of prose this repo ships is for whoever reads it next, never a changelog of the work
  session that wrote it. Covers comments, docstrings, every string literal meant to be read as
  prose (log/error messages, toast/UI text, test names and descriptions), and file/module names,
  everywhere in the repo, not just `packages/*/src`/`tests/`: `scripts/*.py`, `README.md`,
  `.github/skills/`, and package `CLAUDE.md` files are in scope too. Exempt: data values (a test
  fixture's date, a UI placeholder glyph) and legitimate all-caps identifiers/constants/env-vars.
  Forbidden, no exceptions:
  - A session/project-tracking citation under any prefix scheme (`K<n>`, `Fix <letter>`,
    `finding <n>`, `round <n>`, `Phase <n>`, or any other label naming the session's own internal
    tracking), including in file and module names, which a content-only grep will never find.
  - An inline date a decision was made.
  - All-caps or markdown-bold used for emphasis (write "not"/"never", not "NOT"/"NEVER" or
    "not"), the global rule restated here because it kept recurring.
  - Em dashes, anywhere in that prose.

  If a real, non-obvious technical fact survives once that framing is stripped, state it plainly
  with no process pointer. If nothing survives, write nothing. Stripping the flagged token isn't
  the fix by itself; re-read what remains and ask whether it's still narrating a change rather
  than stating a standing constraint. That process narrative belongs in the commit message and
  `decisions/` (or another doc explicitly marked historical/superseded), never in shipped prose.

## External reference material

Crawling an external phenotyping resource (e.g. PlantCV, `github.com/danforthcenter/plantcv`,
`plantcv.org/tutorials`) for domain context can sharpen the picture of what phenotyping problems
exist and which modalities/classical techniques apply. Absorb only the general techniques and
patterns it illustrates, never the specific per-trait application or pipeline; read for
understanding, never clone code or import a pipeline. The endpoint here is always a trained model,
not a classical pipeline transcribed from someone else's recipe. This applies to any external
resource, not just PlantCV.

## Pointers

- `README.md`: setup, layout, running, roadmap.
- `.github/skills/`: crops, crop-science, project-setup, annotation, training, evaluation,
  pipeline-design, toolkit-inventory, cv-research, phenology, visual-analysis, delivery,
  self-improvement. Load before acting in a domain (current list: `ls .github/skills/`).
- `packages/*/CLAUDE.md`: per-package layout and conventions (loads automatically when you
  read files in that package; not duplicated here).
- `docs/` and `.claude/`: local, gitignored dev tooling (refactor task/status tracking, hooks,
  subagents, permission config). Not part of the shipped platform and not present on a fresh clone;
  present only on machines where they've been set up. If `docs/current-task.md` /
  `docs/recent-summary.md` exist, `.claude/hooks/session_start.py` auto-injects them at session
  start. Update `docs/recent-summary.md` before ending a session with refactor work still in
  flight (see global `workflow-playbook` skill). If they don't exist, there's no refactor-tracking
  scaffolding on this machine yet.
