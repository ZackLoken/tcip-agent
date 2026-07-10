# Learning journal

Append-only. Friction → proposed repo improvements (see the `self-improvement` skill). Each
entry: **observation** (what happened) → **proposal** (the exact artifact) → **status**.
Distilled entries become skills / CLAUDE.md diffs / tool proposals for the owner to approve.

---

## 2026-07-09 — Session 1: hazelnut catkin bloom phenology (first e2e; plan rejected)

Source: Zack's transcript of the in-app agent attempting the 5/50/95% catkin bloom task on
`hazelnut_catkin-05-50-95-per-date_valley-farm`. The agent oriented and reasoned well but its
plan was rejected (reason TBD — awaiting Zack). Distilled friction, highest-leverage first:

### L1 — The phenology capability is invisible to the agent (biggest gap)
**Observation.** The 5/50/95-per-date trait is a *core, repeated* pattern, but its machinery is
scattered and none of it is agent-facing: `per_plant_curves` + `_crossing_date` are **tcip-web
routes**, `build_mapping`/`assign_plants` (GPS→plant) is a **tcip-web module**,
`export_phenology_csv` is an untooled `pipelines/postprocessing/export.py` fn. No MCP tool and
no skill expose any of it. The agent burned three Explore agents (~200k tokens) rediscovering
this, then planned to `import tcip_web...` from bespoke scripts.
**Proposal.** (a) A **`phenology` skill** consolidating the pattern (isolate → detect →
classify elongation → per-plant ratio → curve fit → 5/50/95 crossings → genotype rollup) and an
**inventory of the existing pieces** so no one re-discovers them. (b) **Architecture (owner's
call):** lift plant-mapping + per-plant-curve + crossing-date out of `tcip-web` into a shared
`pipelines/` module and expose thin MCP tools, so the agent *composes tools* instead of
scripting into the web backend. **Status:** proposed — skill blocked on Zack's bloom-definition
+ why-rejected; architecture is a scoped build to confirm.

### L2 — `run_inference` already does SAHI tiling; the agent nearly rewrote it
**Observation.** The plan said the baseline "must run via SAHI, not the tcip MCP run_inference
path" and scoped a new `infer_catkins_all_dates.py`. But `run_inference(tile=True,
tile_size=…, overlap=…)` *is* SAHI-style tiled inference. Discoverability miss (compounded by
the MCP server being disconnected — see L3).
**Proposal.** One line in the `training`/`evaluation` (or new `phenology`) skill + CLAUDE.md:
"`run_inference(tile=True,…)` does SAHI tiling — don't script it." **Status:** proposed
(high-confidence, factual). Ties to Zack's standing "don't rewrite tiling each project" ask.

### L3 — The MCP toolkit silently wasn't connected (cwd ≠ repo)
**Observation.** The session ran from `C:\Users\zack`, not the repo, so the repo's `.mcp.json`
never launched → **none of the tcip tools available** (see `scripts/list_tools.py`), and the CLAUDE.md startup ritual
(`load_reports`/`get_project_status`) couldn't run. The agent worked around it via scripts. The
in-app rail avoids this (spawns at repo root), but a direct `claude` launch doesn't.
**Proposal.** A preflight/skill note: "if the tcip tools aren't listed, you're not in the repo —
relaunch from the repo root." Onboarding doc for direct-launch. **Status:** proposed.

### L4 — INVENTED SCIENCE: bloom defined from a bbox-height proxy (RESOLVED — purged)
**Observation.** `results.py` classified catkin "elongation" (a morphological bloom stage) from a
normalized bbox-height threshold (`elongation_height=0.020`) — a number a prior session made up,
committed, and shipped into the delivered `catkin_phenology.csv`. Bbox height is scale/zoom/pose
dependent and is *not* a measure of elongation. My first instinct (the original wording of this
entry) called it a "proxy to upgrade from" — **that was the same error one level up.** You do not
upgrade from invented science; you remove it.
**Resolution (2026-07-09).** Removed the bbox-height logic. `per_plant_curves` now counts
elongated by a **class** from the validated classifier and reports `elongation_classified` —
false when predictions carry no elongation class, so it can't be passed off as a bloom
measurement. Codified a **measurement-integrity invariant** in CLAUDE.md (never invent a proxy
for a biological quantity; the expert defines the trait; validate before delivering).
**Zack's authoritative definition:** bloom = the fraction of a plant's detected catkins that are
elongated (elongation = an expert-scored *visible morphological stage*); 5/50/95% = the dates
that fraction crosses those levels. Plant-level.

### Audit — invented science across the codebase (2026-07-09)
A dedicated audit (distinguishing invented domain science from standard ML tunables like IoU/conf
/seeds) found, ranked:
- **#1 CRITICAL (resolved above).** The bbox-height elongation → `catkin_phenology.csv`.
- **#2 CONFLICT + weak method.** `pipelines/postprocessing/aggregation.py::_agg_sigmoid` computes
  the *same* `catkin_05/50/95per_date` a **different** way — normalize catkin COUNT by the
  observed max + linear interpolation, *labeled* "sigmoid fit" (with a `fit_steepness` column it
  never computes). Two problems: (a) it's not a fit and observed-max normalization is
  sample-dependent; (b) **it defines the trait as count-of-peak, which contradicts Zack's
  elongated-fraction definition.** → the `crops/hazelnut` skill's stated definition also says
  count-of-peak. **The platform holds ≥2 conflicting definitions of one trait.** BLOCKS the
  phenology skill: need Zack to set the single authoritative definition (and say whether
  count-of-peak is a *separate* trait or just wrong).
- **#3 LOW-MED.** `plant_mapping.py` emits a `confidence = max(0, 1 − d/(nn_tol*2))` (arbitrary
  linear form + arbitrary ×2/×3) presented as a probability. Uncalibrated fabricated score; not
  in the delivered CSV. → calibrate against hand-checked assignments or label it a raw distance
  score.
- **#4 LOW.** `aggregation._extract_plant_id` guesses plant id by filename token-stripping
  (mis-handles `YYYY_MM_DD`); already warns loudly (prior M6). → configured `plant_id_fn`.
- Cleared as standard ML tunables: AL entropy/diversity, class-balanced samplers, NMS IoU, conf,
  mask-binarize.

**Held pending Zack:** the `phenology` and `crops/hazelnut` skills, and the aggregation
reconciliation — all depend on the single authoritative trait definition (#2 conflict).

### Audit v2 — comprehensive + adversarially verified (2026-07-09)
Ran a platform-wide workflow (all 6 crop skills + crop-science, all domain-quantity code,
active-learning/eval, and a systematic code-vs-skill pass) with an adversarial skeptic per
finding (default-REJECT to kill standard-ML-tunable false positives). **44 candidates → 1
CONFIRMED.** Reassuring: after verification the platform has exactly one invented/mis-defined-
science defect, not a swamp.
- **The one confirmed defect (CRITICAL):** `catkin_05/50/95per_date` is operationalized two
  ways. Web `results.py` = elongated-fraction crossing (correct — Zack's definition, and the
  path that produces the *delivered* `catkin_phenology.csv`). Pipeline `aggregation._agg_sigmoid`
  (count normalized to **peak** count) → `export.export_phenology_csv` writes the SAME columns —
  an *abundance* signal, and a peak-normalized instantaneous count isn't even a valid cumulative
  progress curve. Not agent-facing; degrades to `insufficient_data` in the orchestrator (no time
  axis), so latent, not live-corrupting today. **Resolution:** canonicalize phenology on ONE
  shared elongated-fraction implementation used by web + pipeline + an MCP tool (also closes L1:
  phenology is web-only, agent can't do it); remove the count-of-peak twin (no abundance trait is
  wanted per the skills). Architecture go = Zack.
- **Cleared but judgment calls** (verifier default-REJECTED as engineering heuristics, not
  fabricated biology — flagged for Zack's eye): `plant_mapping.py` `confidence = 1 − d/(nn_tol·2)`
  (uncalibrated 0–1 presented as a probability; internal only, not delivered); `_extract_plant_id`
  filename token-guessing (already warns loudly).
- Definitions locked: `catkin_05/50/95per_date` = elongated-fraction crossing;
  `catkin_elongation_date` = when ≥95% elongated (≡ `catkin_95per_date` under these defs — asked
  Zack to disambiguate).

### L5 — Domain facts the user had to supply live
**Observation.** Zack had to tell the agent: elongated vs non-elongated is *visually* distinct;
early dates ≈ non-elongated, late ≈ elongated; the plant grid is **RTK-collected + GIS-rectified
onto canopy** (accurate) while image GPS carries ~4.7 m error; include **genotype** in the
per-plant CSV. The agent initially mis-assumed GPS made per-plant infeasible.
**Proposal.** Put these in `crops/hazelnut` + `crop-science`: the hazelnut catkin phenophases,
the RTK-grid-vs-fuzzy-image-GPS linkage framing (why `plant_mapping.py`'s sequence-anchored
matcher is the right tool), and "carry genotype through to deliverables." **Status:** proposed.

### L6 — Environment gaps surfaced late
**Observation.** `scipy` is missing (needed for `curve_fit` — a core phenology need) and torch
is CPU-only though an RTX 4060 is present (~49 h vs minutes for 5,363-image inference).
**Proposal.** Add `scipy` to `environment.yml`; document the CUDA-torch option; a preflight that
warns "CUDA not available — inference will be slow." **Status:** proposed (env change = owner).

### L7 — Stale machine-specific path in project config
**Observation.** The project config's `data/` root pointed at another machine
(`C:\Users\exx\...`); the agent had to locate the real data by hand.
**Proposal.** `init_project`/config should store workspace-relative paths, never absolute
machine paths. **Status:** proposed (small platform fix, dev-lane).

### L8 — Fable 5 safeguards false-flagged the biology work
**Observation.** Fable 5's safeguards flagged the phenology request as biology-adjacent and
auto-switched to Opus 4.8. Not a TCIP defect (Anthropic model behaviour), but real friction a
breeder would hit and not understand.
**Proposal.** Note in onboarding: breeding/biology tasks may trip Fable 5 safeguards; use Opus/
Sonnet (or accept the auto-switch). Not fixable in-repo. **Status:** noted.

### L9 — The plan was script-heavy (4 new bespoke scripts)
**Observation.** Partly forced by L3 (no MCP), but the plan created four new one-off scripts. For
a hands-off, repeated trait, that's fragility — each project re-scripts the same pattern.
**Proposal.** Subsumed by L1: make phenology first-class (skill + tools) so it's *composed*, not
re-scripted. **Status:** proposed.

### Meta — process
The owner had to run this distillation manually because the capture wasn't automatic and the
retrospective ritual was skipped (again). Harden per AGENT_GOVERNANCE_PLAN §2.1 (enforced
capture hook) once the soft loop proves out.

---

## 2026-07-09 — Phenology canonicalization SHIPPED (closes L1, audit #1/#2, #3)

Zack gave the go and locked the definitions; implemented end-to-end. What landed:

- **One canonical implementation.** New `pipelines/postprocessing/phenology.py` is the single
  source of the elongated-fraction milestones (`crossing_date`, `elongation_onset_date`,
  `plant_milestones`, `count_by_class`, `per_plant_phenology`, `write_phenology_csv` +
  `PHENOLOGY_CSV_COLUMNS`). Web `results.py` (`/per_plant_curves`, `/onset_dates`) now
  **delegates** to it instead of carrying its own `_crossing_date`/`_date_key`. Tested in
  `tests/test_phenology.py`.
- **Count-of-peak twin removed (audit #2 / v2 confirmed defect).** Deleted
  `aggregation._agg_sigmoid` + its `"sigmoid"` strategy and `export.export_phenology_csv`
  (both dead), and the 4 sigmoid tests. No abundance-phenology trait exists; the platform now
  holds exactly **one** definition of `catkin_05/50/95per_date`.
- **L1 closed — phenology is agent-facing.** New `compute_phenology` MCP tool
  (`tools/phenology_tools.py`): persisted `plant_mapping.json` + classified preds →
  delivered `catkin_phenology.csv`. **Measurement-integrity guard:** refuses to write the CSV
  when predictions carry no elongation class (`elongation_classified` false). Tested in
  `tests/test_phenology_tools.py`.
- **`catkin_elongation_date` = first date any elongation appears (fraction > 0).** Zack's
  final call (supersedes the audit-v2 "≡ 95per" guess). Threaded through the module, web
  route, `OnsetRow` + the Results tab table.
- **Fabricated `confidence` purged (audit #3).** Removed `confidence = 1 − d/(nn_tol·k)` and
  the `Assignment.confidence` field from `plant_mapping.py`; kept `distance_m` + `source` as
  the honest signals. Consumers/tests updated.
- **Skills written/fixed.** New `phenology` skill (authoritative definition + the pieces to
  compose + the RTK-grid-vs-fuzzy-GPS framing from L5 + SAHI-tiling note from L2). Fixed
  `crops/hazelnut`, `crop-science`, and `delivery` skills, which had encoded the *wrong*
  (sigmoid/peak-count) definition — the conflicting-definitions problem is resolved.
- **CLAUDE.md** measurement-integrity invariant added as the first hard rule.

Still open (not this pass): `_extract_plant_id` filename guessing (warns loudly); scipy/CUDA
env gaps (L6) — though the canonical phenology no longer needs scipy (linear interpolation,
not `curve_fit`).

**Update (same day): plant-mapping lifted into `pipelines/`.** Moved `plant_mapping.py` from
`tcip-web` to `tcip-mcp/pipelines/postprocessing/` (next to `phenology.py`) — one shared
module now, imported by both the web route and MCP. Added the `build_plant_mapping` MCP tool,
so the agent composes the **whole** pipeline via tools (`build_plant_mapping` →
`run_inference(tile=True)` → elongation classifier → `compute_phenology`) with no scripting
into the web backend. Closes L1/L9 fully.

### Adversarial verification of the canonicalization (2026-07-09)
A skeptic pass over the new module + tool + routes confirmed the math is genuinely unified
(one implementation; old bbox-height/peak-count definitions gone) and surfaced two REAL gaps,
both fixed:
- **Web delivery bypassed the integrity guard.** The `compute_phenology` MCP tool hard-refuses
  to write a CSV when predictions carry no elongation class, but the web Results tab only
  *showed a banner* — the Onset/Curves CSV export buttons stayed enabled. Fixed: the GUI now
  disables both exports **and** skips deriving onset rows when `elongation_classified` is
  false, so the human surface matches the agent surface (the top CLAUDE.md invariant).
- **Malformed date folder crashed interpolation.** `date_key` only rejected non-3-integer
  values, so `2026-13-01` survived and then raised `date(2026,13,1)` mid-`crossing_date`,
  taking down the whole per-plant computation. Fixed: `date_key` now validates calendar
  legality (constructs `datetime.date`), collapsing bad folders to the `(0,0,0)` sentinel so
  onset and crossings agree on what's a real point.
- Confirmed intentional (not bugs): `catkin_elongation_date` (discrete first observation) can
  trail the interpolated `catkin_05per_date` on a sparse 0→1 jump — different quantities, per
  Zack's locked defs; now documented + test-pinned. `count_by_class` 6-field filter matches
  the canonical pred format (not the 5-field-label bug it resembles). Confidence purge is
  round-trip consistent.

### Side fix — flaky orchestrator timing (unrelated, opportunistic)
The full-suite run surfaced `test_exception_in_phase_body_fails_and_logs` failing (~4/5 at
clean HEAD too, so pre-existing). Root cause: the orchestrator measured phase
`elapsed_seconds` with wall-clock `time.time()` (coarse ~16 ms on this Windows box), so a fast
phase rounded to exactly 0.0 and tripped `assert elapsed_seconds > 0.0`. Switched the 5
phase-duration measurements to monotonic `time.perf_counter()` (left the genuine wall-clock
timestamps as `time.time()`). Now 5/5 green — a latent CI-flake that would have bitten a
hands-off build.
