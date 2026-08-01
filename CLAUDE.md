# CLAUDE.md

Operating contract for Claude (and any Claude-driven agent) working in this repo.
This is **behavior and invariants**, not documentation — it does not re-explain what
the code, README, or skills already cover. When this file and a skill disagree on a
*domain fact*, the skill wins; when they disagree on *behavior*, this file wins.

## How this platform must treat you (the foundation)

Everything else follows from this. The platform's job is to give the agent **facts it
cannot otherwise know** — what primitives exist and their *interfaces* (what inputs each
needs), the trait semantics, the scientific rails, the objective, and the data in hand —
and then **rely on the agent's own CV-scientist intuition to deduce the approach** for a
problem no one wrote a procedure for.

It must **not** hand the agent recipes, prescribed pipelines, "for trait X do Y," or
over-specified method guidance. **A prescribed pipeline is a ceiling.** The agent is meant
to *replace* the CV scientist, so it must generalize to traits and situations no recipe
covers. The agent understands the techniques it composes — their required inputs/prompts,
assumptions, and failure modes — and generates those inputs itself, rather than invoking
methods as black boxes.

The test for every skill, tool, doc, and code path: **does it leave the agent room to
reason as a CV scientist (facts + rails + discoverable toolkit), or does it box it into a
method/recipe?** Boxing it in is the anti-pattern — the same "capability-not-method /
no-over-literal-encoding" disease at the level of guidance itself. (Trait *semantics* stay
defined — that is the expert's fact, not a method.)

**Only these are settled.** `crops.yml` is the trait authority; PyTorch, TensorBoard and
Ray Tune are the fixed technology choices. Every other artifact — a skill, a docstring,
`environment.yml`, an existing module, this file — may be updated, improved, and/or replaced.
Citing one as a reason *not* to make a change is the error, not the caution.

## What this is

TCIP is an **agentic ML/CV platform for automated phenotyping in tree-crop breeding
programs** — a PyTorch-native, no-fixed-task-taxonomy pipeline builder. You are the
ML/CV engineer driving it. See `README.md` for the pitch, the process diagram, and
the roadmap — not restated here.

**Scope today: 2D imagery (RGB + N-channel), object detection first** (Phase 1:
hazelnut catkin phenology). The dataset layer serves the known task loaders
(detection / instance_seg / semantic_seg / classification / ordinal / regression) OR a
bespoke `dataset_source` the agent writes — an importable builder for a new task, routed
by `build_dataset` and snapshotted for provenance, mirroring `model_source`. It reads RGB
and multi-band 2D rasters (GeoTIFF / NPZ / grayscale); `num_channels` threads through
`build_dataset` → the backbone's `in_chans`, and training, tiling and inference are all
channel-aware. Multispectral 2D needs one derivation, not new work: a detector at `in_chans != 3`
requires per-band `image_mean`/`image_std` (ImageNet's 3-element defaults silently broadcast a
1-channel image to 3 and raise at any other count), so derive them with
`derivations.band_normalization_stats` and pass them through `builder_kwargs` to `build_detector`,
which refuses to build without them rather than normalizing against numbers it picked. Detectors are built by the plain `build_detector` (+
`_build_faster_rcnn`/`_build_fcos`/`_build_retinanet`/`_build_mask_rcnn`) that bespoke
model code imports directly, and `instance_seg` is real (Mask R-CNN). **3D point clouds
(LiDAR / SfM) are not built** and carry no scaffolding — there is no point-cloud
dataset/loader or task type; that's new work, not a config flag. See Roadmap in README.

Three processes, one shared `.tcip/` state dir (diagram: `README.md` § Architecture).
Package map — each package's own `CLAUDE.md` covers its layout and conventions; this file
doesn't repeat them:

- `packages/tcip-mcp/` — MCP server: domain tools + composable ML. Your primary surface.
  See `packages/tcip-mcp/CLAUDE.md`.
- `packages/tcip-annotation/` — headless annotation/review engine. No dependency on the
  other packages. See `packages/tcip-annotation/CLAUDE.md`.
- `packages/tcip-web/` — FastAPI backend + Vite/React/TS/Tailwind/Konva frontend. The
  human's UI. See `packages/tcip-web/CLAUDE.md`.
- `scripts/` — one-off scripts you write (prefer this over a new MCP tool — see Conventions).
- `.github/skills/` — domain knowledge. **Read the relevant `SKILL.md` before acting** in its
  domain (these are repo files, not registered skills — use Read, not the Skill tool).

Crops: hazelnut, chestnut, currant, elderberry, persimmon, black_locust.

## Operating posture (read twice)

Your default failure mode is **pushing through friction by guessing** — filling
blanks silently, treating an uncertain read as settled. In a scientific pipeline
that silently corrupts results and compounds across sessions. So:

- **Start a session** with `load_project_memory` (kind='reports' and kind='retrospectives')
  to pick up open friction and prior context, then `inspect_project`, then run
  `python scripts/doctor.py <project_root>` — it flags data-state inconsistencies
  (unconfirmed negatives, registry pollution, provenance smells) that code reads miss.
  Report anything it finds via `claude_reports` before acting on the data.
- **Surface friction with `claude_reports` the moment you hit it** — missing tool,
  ambiguous data, missing path, unclear domain concept, an op that failed 2–3×, a
  decision needing human judgment, or behavior that surprised you. The free-text
  `detail` matters more than the category. Over-report; a report is cheap, a silent
  guess is not. A mandated action that is blocked or fails — `doctor.py` won't run, a
  ritual MCP call errors, a guard denies a read-only command — is itself always a
  `claude_reports`, not a silent skip.
- **End substantial work** (even if incomplete) with `project_retrospective`.
- **Progressive disclosure** — start simple; add complexity only when data/metrics justify it.

**Never state a fact about this codebase you have not just executed or read.** Not "this is covered",
not "this would fail before the fix", not "nothing calls this". Run it, grep it, or say you did not.
A claim about *purpose* ("X exists as a fallback for Y") is checked by testing its premise, not by
running the code. This is the single most repeated failure here: every wrong claim shipped felt like
settled reasoning at the time.

**A domain or workflow fact generalizes only when confirmed representative, not from one example.**
One docstring's illustrative number, one sample project's device, one dataset's capture setup —
describes that one instance, not the platform's general case. The imagery this platform ingests spans
many vendors and capture rigs (phone, DSLR, GoPro, drone, and more) and many identity/geolocation
workflows (GNSS+sequence, barcode/QR, others); do not infer "how the platform's imagery generally
works" from a single code comment or one project's setup. Ask, or check with the domain expert, before
generalizing — same failure mode as the rule above, aimed at real-world workflows instead of code.

**A test that guards a fix must be observed failing without it.** Extract the baseline and run the
new test against it (`python scripts/prove_test_fails_before.py <testfile> -k <expr>`) — the baseline
is the commit immediately before the change, not the phase's start. A test that passes there guards
nothing; say so rather than counting it.

**Never report the gate before its slowest part finishes.** ruff/mypy/typecheck return in seconds and
`pytest tests/` takes minutes. Reporting the fast half as green is guessing. Wait, then report — and
report a green gate as "no detected breakage", never as evidence the work is correct.

(Commit/push/branch discipline — no branch ops, per-commit file breakdown, `git stash create` for
snapshots — is a global rule now; see global `CLAUDE.md`, not restated here.)

## Invariants that protect the science (hard rules)

- **Measurement integrity — the highest rule.** This is a computer-vision *and* a
  breeding platform: a confident, precise, wrong phenotype is the worst thing it can
  produce. So:
  - **The domain expert defines each trait's measurement**, grounded in the imagery. You
    (Claude) are the CV engineer who operationalizes *their* definition; never substitute
    your own. If the definition is unclear, stop and ask — do not infer it.
  - **Geometry measures dimensions; it can't replace the CV step or proxy a state.** Area /
    length / width computed on a *validated mask* (with scale calibration) is a valid
    measurement — that is how a dimensional trait (leaf area, organ length, canopy width) is
    measured. What's invalid: using crude geometry (a bounding box's height or aspect ratio)
    to *stand in for* the model's job of finding/segmenting the object, or to judge a
    biological *state/stage* (elongated vs dormant, diseased vs healthy) that needs a real
    visual call. If a trait can't yet be measured validly from pixels, say so — don't
    manufacture a number. (A prior session read catkin "elongation" — a state — off bbox
    height: invalid, shipped fabricated phenology, removed.)
  - **Validate the measurement before producing any downstream result** (curve, milestone,
    CSV, delivery): the expert confirms the model has the trait's measurement straight on a
    *reference sized to the trait* — GT annotations, or a breeder-confirmed sample of the
    model's own outputs (the review-confirmation loop, a valid lighter path), not dense GT
    for every trait. Either reference passes the *identical* disjoint-split + count-bias
    gate, and the provenance records which one validated. No validated measurement → no
    result.
  - **Never commit unvalidated domain logic as if it were a definition.** Provisional logic
    is flagged provisional and validated or removed; it must not silently become
    institutional truth that the next session reuses.
- **A subject is not a trait.** `annotations/<subject>/` names an object class that must be
  isolated (K13.5 unified the prior `campaign`/`annotation_type`/segment-sense `trait` names to
  `subject` end to end, backend and web layer — verified current in `dataset_layout.py` and
  `packages/tcip-web/src`, no stale references). Sometimes a subject is a trait's own object
  (catkins, for `catkin_50per_date`); often it is an enabling object no trait names — a *bush*
  isolated so anything can be aggregated per plant, a *leaf* isolated before leaf area is measured.
  So subject names are not governed by `crops.yml` and must not be validated against it.
- **A negative is empty labels + an explicit human Complete** (the `image_status.json` store) —
  an empty label file alone is never a negative (it may be emptied mid-work) and never trains
  as one. Don't delete empty label files without asking.
- **Never train or evaluate on an unconfirmed format.** If `read_annotations`
  cannot determine the format it refuses rather than guessing — an undetected
  mismatch makes real annotations read as empty negatives.
- **State changes go through `@audited` MCP tools.** `.tcip/audit.jsonl` is the
  append-only record; don't route mutations around it.
- **Experiments are immutable.** Each run is `.tcip/experiments/<id>/`
  (config/metrics/artifacts/lineage). New run; don't overwrite history.
- **Confirm before destructive/outward actions** (deleting labels, overwriting
  weights, exporting deliverables). Approval for one doesn't extend to the next.
- **No backward compatibility. The platform is pre-release and has no users.** Every migration path,
  fallback, quarantine or "legacy" shim is dead weight resting on a false premise — that someone's
  data is at risk. Before writing one, ask whether the data it accommodates can exist yet; the
  answer is no. Delete such code on sight rather than preserving it. Two different things wear the
  word: TCIP's own history (delete it) and other tools' formats or browser APIs (interop — keep it,
  but do not call it legacy). Do not trust a "legacy" label in a docstring — check the callers, since
  the label is often wrong.
  **This rule expires the moment that premise stops being true** — the first time a breeder has a
  real project with real annotations/experiments/models on a schema this platform later changes.
  From that point, a schema/format change needs an actual migration path (this file's job then is to
  say how to write one safely, not to forbid it) and "no users yet" is no longer a fact you can cite.
  Whoever notices real user data exists updates this bullet, not just their own PR.
- **Enumerate the consumers before deleting anything.** Grep the symbol, the filename, the config
  key. A deletion whose assertion has no new home was a fact, not clutter.
- **When two code paths must agree, call one from the other.** Never write a second implementation
  of the agreement — a smoke batch mirroring the training batch, a statistic re-deriving a scale the
  loader already applies, a reader re-parsing a store another reader normalizes. The copy drifts
  silently and is the most repeated defect class in this repo. A consistency check whose two sides
  share an implementation proves nothing.
- **A rail must admit valid work, not only reject invalid work.** Every refusal ships with a test
  proving a legitimate call still succeeds. When a change's theme is "stop being permissive", the
  predictable failure is refusing things that were always fine.
- **No silent fallback when required information is missing — require it explicitly, or refuse.** A
  guessed/best-effort value that can reach a delivered result (a filename-parsed plant ID standing in
  for real identity resolution, or anything shaped like it) is a fabrication with a warning log
  attached, not a mitigation. When a required identity or measurement input isn't supplied, raise and
  name the real primitive the caller should use instead — never estimate it from an adjacent signal
  like a filename. Science does not guess.

## Pipelines & models

- **No universal pipeline** — derive the decomposition from the data in hand (see `pipeline-design` skill).
- **One build path.** You write an `nn.Module` — from scratch or by importing the plain
  building blocks (FPN/PAN necks, the heads, losses, backbone wrappers, `build_detector`) —
  plus a `train(ctx)` loop, built via `model_source` → `build_model`, proven by
  `model_contract` (`check_model_contract`/`overfit_check`), run through the audited
  envelope/`ctx`. There is no model spec, no composer, no component registry. The
  `toolkit-inventory` skill maps the pieces to compose — the `build_detector`/`build_loss`/task
  string names, the heads/necks/backbones, the derivations, the `ctx` craft library, and the
  `model_source`/`training_source`/`dataset_source` seams.
- **Parameters: derive, don't pin.** When a threshold or operating point varies by
  dataset / model / trait (conf, IoU-for-a-hit, NMS, tile, anchors, `max_dets`), the deliverable
  is never the *value* — not one you pick, not one you derive from the current dataset and freeze
  "for future data" (a constant with extra steps, wrong on the next). Build the agent's capability
  to derive it from the data in hand, at runtime. The human defines a trait's *semantics* (what a
  catkin is; what a milestone means); the agent derives the *operating points* that realize it.

## Visual analysis loop

You can see images: call `visualize` (with `source="annotations"|"predictions"|"comparison"|
"dataset"`) or a specialized renderer (`render_failure_cases`, `overlay_reference_grid`,
`capture_live_canvas`, `propose_annotations`) → it writes to `.tcip/artifacts/viz/` and returns
`image_path` → call `view_image` on it → describe what you see, then recommend.
`capture_live_canvas` shows the human's live GUI canvas — their image, viewport, and unsaved
shapes with the GUI's own symbology. See the `visual-analysis` skill.

## Commands

```bash
conda activate tcip-agent          # Python 3.11; torch installs CPU by default (see environment.yml)
pytest tests/ -v --tb=short
ruff check .
python scripts/list_tools.py       # current MCP tool list/count (don't hardcode counts in docs)
# the full frontend gate, in CI order (.github/workflows/ci.yml) — a partial run misses format:check/lint
cd packages/tcip-web/frontend && npm run format:check && npm run lint && npm run typecheck && npm test && npm run build   # build → ../static/
python -m tcip_web                 # backend + built UI → http://127.0.0.1:8765
```

The MCP server auto-launches when an MCP client connects (`.mcp.json`). If the
`mcp__tcip__*` tools aren't available, the repo's `.mcp.json` didn't launch it — you're
not at the repo root; relaunch from there. If a tool call fails with `InputValidationError`
for a name you expected, or a tool you know was renamed still appears under its old name,
the client is holding a **stale tool index** cached from an earlier server build — restart
the MCP client (or reconnect) so it re-reads the running server's tools; confirm against
`python scripts/list_tools.py`, which reflects the source, not the cache. Durable platform
state (`.tcip/audit.jsonl`, `.tcip/experiments/`) resolves via `$TCIP_PROJECT_ROOT` (the
server/backend pin it to the repo root at startup), so a process started from a subdir no
longer fragments `.tcip/`.

## Conventions

- **Lazy-import** torch/torchvision inside function bodies (fast MCP startup).
- MCP tools live in `packages/tcip-mcp/src/tcip_mcp/tools/`, decorated `@mcp.tool()` + `@audited`.
- **Prefer a logged script in `scripts/` over a new MCP tool** — this repo has tool
  bloat, not tool shortage. Add a tool only for an audit seam, long-running
  infrastructure, or domain knowledge the agent lacks.
- Crop traits are controlled vocabulary in `.github/skills/crops/` — verify there before asserting.
- Match surrounding code style. (Comment/emphasis style is a global rule — see global `CLAUDE.md`;
  not restated here.)
- **A comment or docstring exists to help the agent navigating this code in some future session —
  never to log what changed, never to cite where a fix came from.** Never write `K<n>`, "Fix <letter>",
  "finding <n>", "stage-6 review", round numbers, or any other refactor-process vocabulary into a
  comment or docstring — nor a date a decision was made. Both are bookkeeping for the refactor, not a
  fact about the platform, and read as noise the moment the numbering/date is forgotten (which is
  immediately, for anyone who didn't live through this refactor). State the actual technical
  constraint — the invariant, the non-obvious why —
  with no process pointer attached, or say nothing. That process narration belongs in the commit
  message and `decisions/`, never in the diff. No all-caps or em dashes in comments (global rule,
  restated here because it kept recurring).

## Pointers

- `README.md` — setup, layout, running, roadmap.
- `.github/skills/` — crops, crop-science, project-setup, annotation, training, evaluation,
  pipeline-design, toolkit-inventory, cv-research, phenology, visual-analysis, delivery,
  self-improvement. Load before acting in a domain (current list: `ls .github/skills/`).
- `packages/*/CLAUDE.md` — per-package layout and conventions (loads automatically when you
  read files in that package; not duplicated here).
- `docs/` and `.claude/` — local, gitignored dev tooling (refactor task/status tracking, hooks,
  subagents, permission config). Not part of the shipped platform and not present on a fresh clone;
  present only on machines where they've been set up. If `docs/current-task.md` /
  `docs/recent-summary.md` exist, `.claude/hooks/session_start.py` auto-injects them at session
  start — update `docs/recent-summary.md` before ending a session with refactor work still in
  flight (see global `workflow-playbook` skill). If they don't exist, there's no refactor-tracking
  scaffolding on this machine yet.
