# TCIP Vision, Critique, and Roadmap

Plan-of-record for the Tree Crop Imaging Pipeline (TCIP). Revised across sessions.

- **Last updated:** 2026-04-20
- **Revised by:** Zack Loken (PD) + Claude Opus 4.7, revising the April 2026 draft by Claude Opus 4.7
- **Format:** living engineering plan. Dated revision history at the bottom.

---

## How to use this document

You are working on TCIP, an agentic computer vision system for specialty tree crop breeding programs at Savanna Institute. The PD (Zack Loken) is the Imagery Analyst at Savanna Institute, a capable ML/CV practitioner — a phenotyping specialist who learned ML to automate their own work, not a computer scientist. They built the prototype you are now looking at.

Read this document in full before you touch the codebase. It contains:

1. The vision the PD is building toward.
2. An honest assessment of what exists today.
3. A critique of the current design and why we are changing it.
4. The target architecture.
5. The eyes-and-hands annotation loop.
6. The self-improving meta-loop.
7. A phased roadmap.
8. Context on the USDA SCRI grant tied to this project.
9. Open questions.
10. Operational guidance for working in this codebase.
11. Revision history.

When the PD asks you to do something, your job is usually not to execute blindly. This is a research project in flux. Much of the code is prototype that will be rewritten. Some of it is stable and useful. Know the difference before you change things.

When you are confused, missing context, or think the PD is asking you to do something that conflicts with the direction in this document, say so. Do not silently guess or work around gaps. The system improves only when confusions are surfaced.

---

## 1. The vision

### North star

A fully autonomous computer vision system for plant breeding that reasons about novel phenotyping problems from first principles, handles any sensor modality, bootstraps from minimal labels via active learning, and generalizes across crops.

A breeder opens a project, says "I want to measure Eastern Filbert Blight severity across my hazelnut trials," and the agent handles everything: understands the trait biologically, locates or requests imagery, asks a human to label a handful of examples, trains a model, validates it, and delivers per-plant CSV results the breeder can load into their selection pipeline.

The agent is Claude (you, or a future you). Our contribution is the domain knowledge, the ML tooling the agent connects to, the visual reasoning loop, and the breeding-program infrastructure. We are not building a new foundation model.

### The near-term proof point

The north star is a multi-year effort. What the grant will fund is narrower:

> An agent that can take a breeder from "I want to measure trait X on crop Y" to "here's a validated per-plant CSV for your trial" in one interactive session per novel trait, for a defined subset of traits covering multiple pipeline patterns.

If we can do this across 15–20 traits spanning detection, classification, regression, phenology, and 3D point cloud analysis, on data the breeder already collected, that is a full CAP's worth of work and genuinely useful. The rest is follow-on.

Do not pitch the north star as if it were the deliverable. The north star is where we aim. The proof point is what we commit to.

---

## 2. System state: what exists today

### Repository structure

```
packages/
  tcip-mcp/          # MCP server + 54 tools + pipeline components
  tcip-annotation/   # Headless annotation engine, SAM wrapper, format I/O
  tcip-web/          # FastAPI backend + React/Konva GUI (yolo-annotator parity)
                     # — replaces the retired tcip-vscode extension
tests/               # pytest suite
.github/
  copilot-instructions.md   # Main agent prompt
  agents/tcip.agent.md      # Agent definition
  skills/                    # Domain knowledge as markdown
    crop-science/
    crops/{hazelnut,chestnut,currant,elderberry,persimmon,black-locust}/
    annotation/
    evaluation/
    pipeline-design/
    training/
    visual-analysis/
    delivery/
  prompts/                   # Task templates
    new-project.prompt.md
    train-model.prompt.md
    review-annotations.prompt.md
    run-inference.prompt.md
    export-results.prompt.md
```

### Architecture summary

Claude Code (via VS Code Copilot Chat or Claude extension) is the agent. It connects to a stdio MCP server defined at [packages/tcip-mcp/src/tcip_mcp/server.py](packages/tcip-mcp/src/tcip_mcp/server.py) that exposes domain-specific tools. The tools call into a composable ML system ([pipelines/composer.py](packages/tcip-mcp/src/tcip_mcp/pipelines/composer.py), [pipelines/registry.py](packages/tcip-mcp/src/tcip_mcp/pipelines/registry.py), [pipelines/components/](packages/tcip-mcp/src/tcip_mcp/pipelines/components/)) built on PyTorch and torchvision. SAM1 provides geometry for assisted annotation. An event-based bridge writes JSON to `.tcip/events/` which the VS Code extension tails to drive webview panels.

This is a significant improvement over earlier prototypes. A previous version included a custom Rust agentic runtime (conversation loop, compaction, subagent spawning). That was deleted in favor of letting Claude Code handle orchestration. Correct call. Our job is now domain tooling, not agent plumbing.

### Current tool inventory (as of 2026-04-20)

54 `@mcp.tool` registrations across ten files in [packages/tcip-mcp/src/tcip_mcp/tools/](packages/tcip-mcp/src/tcip_mcp/tools/):

| File | Count | Tools |
|------|-------|-------|
| active_learning_tools.py | 2 | score_unlabeled, get_review_queue |
| annotation_tools.py | 7 | load_annotations, save_annotations, evaluate_detections, evaluate_dataset, sam_predict, run_matching, push_panel_data |
| data_tools.py | 3 | load_dataset, validate_data_quality, split_dataset |
| experiment_tools.py | 6 | create_experiment, log_metrics, record_artifact, get_experiment, compare_experiments, get_experiment_lineage |
| inference_tools.py | 3 | run_inference, export_predictions_yolo, export_results_csv |
| model_tools.py | 4 | list_available_models, register_model, list_registered_models, get_best_model |
| pipeline_tools.py | 6 | list_components, recommend_model, validate_model_spec, validate_pipeline_spec, run_pipeline, compose_and_summarize |
| project_tools.py | 8 | init_project, create_session, append_session_event, list_sessions, get_session, get_project_status, export_project, import_project |
| training_tools.py | 7 | validate_config, launch_training, check_training_status, list_training_runs, run_hpo, get_training_metrics_path, get_worst_predictions |
| vision_tools.py | 8 | visualize_annotations, visualize_predictions, visualize_comparison, visualize_worst_predictions, visualize_dataset_sample, sam_auto_label, accept_candidates, visualize_grid_overlay |

The earlier version of this document said "~40 tools." That was an undercount. The real number is 54.

### What works

- **MCP server loads cleanly** with torch-gated tool registration. Tools that need torch fail gracefully if it is missing.
- **Multi-format annotation I/O** (YOLO, COCO, VOC, LabelMe) via `tcip_annotation.format_io.detect_format()`.
- **Visual reasoning loop** for annotation is functional: `sam_auto_label` generates SAM candidates, renders a numbered overlay, Claude views it, classifies each candidate, and `accept_candidates` saves results as annotations.
- **Composable ML** (backbone + neck + heads) works as a library. No forced architecture.
- **Experiment tracking** in `.tcip/experiments/` captures config, metrics, artifacts, lineage.
- **Audit logging** via `@audited` decorator writes every tool call to `.tcip/audit.jsonl`. This is load-bearing — see §3.1 on why that affects tool-vs-script decisions.
- **Per-crop skills** in [.github/skills/crops/](.github/skills/crops/) contain real breeder knowledge (phenology calendars, trait inventories, annotation gotchas).
- **Three one-off pipelines that actually work** in production outside this codebase: chestnut bur detection from drone, hazelnut biomass/carbon estimation from LiDAR, hazelnut catkin elongation phenology (in progress).

### What does not work

- **The system is not end-to-end deployed.** No breeder has used it. No trait has been phenotyped through the agentic loop from scratch to CSV.
- **Only two crops have meaningful labeled data.** ~40K chestnut bur annotations, a few thousand hazelnut catkin annotations, some NIR for kernel oil, point clouds with ground classification. The other four crops have essentially nothing.
- **The two-layer pipeline paradigm is still baked into code and skills** despite being too rigid for the full trait set. See §3.2.
- **SAM1 not SAM2.** SAM2 is materially better for small objects and offers video/temporal support we will need.
- **Tool inventory is bloated at 54 tools.** Many wrap operations Claude could do inline. See §3.5.
- **No meta-loop.** No `claude_reports` tool for surfacing confusions. No retrospective mechanism. When Claude gets stuck or confused, it disappears into conversation history instead of accumulating into system improvements. `.tcip/retrospectives/` and `.tcip/reports/` do not yet exist.
- **Skills are still schema-heavy.** Per-crop skills have trait tables with `(sensor, ML task, format)` tuples. Useful as reference, but missing the messy breeder knowledge that actually matters for pipeline design.

---

## 3. Critique of the current design

This section is the most important part of the document. Read it twice.

### 3.1 We are building the wrong layer — mostly

The MCP toolkit is currently a collection of thin wrappers around ML operations: `load_dataset`, `validate_data_quality`, `launch_training`, `evaluate_detections`, `run_inference`, `export_predictions_yolo`. This teaches Claude to be an ML engineer. Claude is already an ML engineer. It can read PyTorch docs, write training loops, compute mAP, run inference.

What Claude cannot do without us is reason about tree crop phenotyping: what "Eastern Filbert Blight severity" means, what sensor data answers that question, what a valid phenotype looks like from a breeder's perspective, how to score results against manual scoring, what tolerances are acceptable, which genotype comparisons actually inform selection.

We have this backwards. The ML operations should mostly be Claude writing code. The domain knowledge should be the tools.

**The audit-trail caveat.** "Tool vs. script" is not the only axis. Some of the existing ML wrappers exist for a reason: `@audited` tool calls produce an auditable lineage in `.tcip/audit.jsonl`, and `create_experiment` / `log_metrics` / `record_artifact` enforce reproducible experiment tracking. A breeder coming back in two years needs to know what data produced what model. If we replace audit-backed tool calls with ad-hoc Python scripts, that lineage evaporates unless we explicitly preserve it.

So the real question for each tool is **"which boundary crossings need to be audited?"** Rewrite candidates:

- **Keep** (audit seam / long-running / domain knowledge): training launch, SAM inference, experiment tracking, trait/dataset knowledge retrieval, visual rendering.
- **Consider replacing with Claude-written script** (ML wrapper that adds no audit value): config validation, format conversion, thin pipeline orchestration.
- **When replacing**: keep the script under a repo-visible path (e.g. `scripts/` or inside `.tcip/session-scripts/`) so lineage survives the session.

**Example of what we want:** `get_trait_profile("efb_damage", "hazelnut")` returns the trait definition, scoring scale, known image characteristics, existing annotated data, prior successful pipelines on similar traits, known failure modes, and links to stakeholder discussions about this trait. That is a tool Claude cannot replicate by writing Python.

**Example of what we have too much of:** `validate_config(config)` checks whether a training config has the right keys. Claude can write five lines of Python to do this.

**Open question (§9):** what does `get_trait_profile` return that isn't just "load this skill file into context"? If the answer is "nothing," then the tool is a file read with extra steps. The schema needs real design.

### 3.2 The two-layer paradigm is a cage

"Isolation → Task → Post-processing → per-plant CSV" works for counting catkins on pre-segmented bushes. It breaks for:

- Traits that are not localizable (overall plant vigor from a canopy photo)
- Traits requiring temporal reasoning across many images (phenology onset dates)
- Traits requiring 3D reasoning (crown volume, biomass)
- Traits that are relational (how does this plant look relative to its neighbors on the same day)
- Traits from 1D data (NIR spectra for kernel oil percentage)

We have all of those in the trait inventory. The two-layer paradigm is one pattern among many, not a universal scaffold. It should live in skills as one option the agent can choose, not in orchestrator code as a mandatory structure.

**Where this still lurks in the current codebase:**
- [packages/tcip-mcp/src/tcip_mcp/pipelines/orchestrator.py](packages/tcip-mcp/src/tcip_mcp/pipelines/orchestrator.py): hardcodes `VALID_PHASE_TYPES = {"training", "inference", "cropping", "aggregation", "export"}` and enforces phase structure.
- [.github/copilot-instructions.md:102-107](.github/copilot-instructions.md#L102-L107): states "Every crop analysis follows: Isolation → Task → Post-processing."
- [.github/skills/pipeline-design/SKILL.md](.github/skills/pipeline-design/SKILL.md): leads with "Two-Layer Paradigm" as if universal.

Kill these. Replace with a library of pipeline patterns the agent selects from based on trait characteristics.

### 3.3 Annotation format handling is solved at the wrong level

Format-agnostic I/O for YOLO, COCO, VOC, LabelMe is real work. But the agent does not need to care about annotation formats. Format is an ingestion concern, not a reasoning concern.

The agent's job is phenotyping, not dataset engineering. We should pick one internal representation (simple JSON schema with pixel coordinates), convert from the user's format at ingestion, and make the agent blind to what format the user originally had.

**This is a migration, not a deletion.** Real breeder data arrives in all four formats; `tcip_annotation.format_io` handles that today. A naive "collapse the format surface" that breaks ingestion is worse than the current state. The right sequence:

1. Define the internal representation.
2. Build a one-way conversion pass at ingestion (user format → internal).
3. Verify ingestion against a representative sample from each format.
4. Stop exposing format-specific tools to the agent.
5. Only then delete multi-format save paths (output, if we decide all exports go through one internal format and optional one-way writers).

Current auto-detection and multi-format save paths are code surface area that adds bugs and cognitive load without adding agent capability. Collapse them, but do it as a migration.

### 3.4 Skills are still schema-heavy, not knowledge-heavy

The per-crop skills are better than nothing but mostly tabular: trait name, sensor, ML task, format. The knowledge that actually matters for pipeline design is prose:

- "Breeders rank genotypes rather than measuring absolute values. If the model is off by 10% but preserves ordering, that is fine."
- "EFB severity scored mid-season on overcast days. Direct sun makes dark lesions invisible in shadow."
- "Catkin counts are unreliable in windy conditions because catkins are pendulous."
- "We have tried X architecture on Y trait three times. It failed because of Z."
- "Trait A correlates strongly with trait B that we measure well. A regression from B's model output is a reasonable prior."

This knowledge lives in the PD's head and in Slack threads with breeders. It needs to be extracted into skills, one trait at a time, as we work on that trait. Do not try to write all skills upfront. Write them as they are earned through real project work.

### 3.5 Tool count is too high

Current inventory: 54 tools. Rough categorization:

- **Genuinely necessary** (Claude cannot do this by writing a script): SAM inference, long-running training launch and monitoring, experiment tracking, trait/dataset knowledge retrieval, visual rendering for vision-enabled review. Maybe 12–15 tools.
- **Marginal** (Claude could do it with a script but a tool is more reliable): config validation, format conversion, pipeline spec validation, project status aggregation. Keep if audit trail matters, drop otherwise.
- **Probably bloat** (thin wrapper around something Claude writes better inline): list/recommend/compose operations, most file I/O wrappers, many pipeline orchestration tools.

We do not yet know which are which for certain. The audit comes from running the Phase 0 exercise and tagging which tools we actually called, which we wanted and couldn't find, and which we ignored. Flag candidates now; delete after evidence.

### 3.6 There is no mechanism for the system to improve itself

Every session starts fresh. Claude's confusions, dead ends, and insights from session N are not available to session N+1. This means the system cannot get smarter unless the PD manually notices patterns and writes them down. That does not scale.

We need first-class machinery for:

- Structured self-reporting when Claude gets stuck (`claude_reports`).
- Per-project retrospectives at session end.
- An artifact pile that future sessions read as context.
- Eventually: automated skill drafting from accumulated retrospectives.

None of this exists yet.

---

## 4. Target architecture

### 4.1 Principles

1. **Domain knowledge in tools and skills. ML operations in code Claude writes — *unless* the operation is an audit seam.** If you find yourself writing a thin wrapper, stop and ask: what can Claude not do without this tool? If the answer is "produce a tamper-evident audit record," that's a reason to keep the wrapper. If it's "save typing," rewrite as a script.
2. **No mandatory pipeline structure.** Pipelines are patterns the agent selects, not schemas the orchestrator enforces.
3. **One internal data representation.** Convert at ingestion. Agent never sees format variations. Migrate, don't delete.
4. **Minimize tool count. Every tool must justify its existence.**
5. **Surface confusions.** The system improves only when failure modes become artifacts.
6. **Visual reasoning as a first-class workflow.** Claude sees images. Every step that involves judgment about image content should have a render-then-view-then-decide loop.

### 4.2 Tool inventory (target)

The following is what we think we need. Expect this list to change after the Phase 0 exercise (§7).

**Domain knowledge:**
- `get_trait_profile(trait, crop)` — trait definition, prior art, known failure modes, scoring conventions, stakeholder notes. (Schema design is an open question — see §9.)
- `get_dataset_catalog(crop?, sensor?)` — what data we have, where it lives, what is labeled, data quality notes.
- `get_prior_experiments(trait, crop)` — what has been tried, what worked, what did not, with links to experiments.

**Annotation loop:**
- `propose_annotations(image_path, trait)` — SAM2 + classifier candidate generation with confidence scores.
- `review_batch(trait, n_samples, strategy)` — active learning selection of most informative unlabeled examples.
- `commit_labels(image, labels, reviewer, confidence)` — save labels with provenance, trigger classifier retraining when threshold met.
- `model_readiness(trait)` — is the current model good enough to deploy? Returns decision and evidence.

**Visual reasoning:**
- `render(what, how)` — unified render tool. `what` is data (annotations, predictions, comparison, dataset grid, failure cases). `how` is rendering parameters.
- `view_image` — native Claude capability, not a TCIP tool.

**Training and inference (audit seams):**
- `launch_training(spec)` — kick off a long-running training job, returns run_id.
- `monitor_training(run_id)` — status, metrics, TensorBoard URL.
- `run_inference(model, images)` — batch inference returning predictions.

**Experiment tracking (audit seams, already good):**
- `create_experiment`, `log_metrics`, `record_artifact`, `get_experiment`, `compare_experiments`.

**Meta-loop:**
- `claude_reports(category, detail, context)` — structured confusion logging.
- `project_retrospective(project_id)` — end-of-project reflection.
- `load_retrospectives(filter)` — read past retrospectives at session start.

**Delivery:**
- `export_phenotype_csv(experiment, format)` — produce the per-plant CSV breeders actually use.

This is roughly 15–18 tools, not 54. Everything else Claude writes as Python scripts checked into `scripts/` or session artifacts.

### 4.3 Workspace conventions

Maintain the existing `.tcip/` convention for project state. Extend it with:

```
.tcip/
  project.json         # current state, phase, active experiments
  audit.jsonl          # tool call log (existing)
  experiments/         # experiment tracking (existing)
  artifacts/           # rendered images, intermediate files (existing)
  events/              # VS Code panel events (existing)
  sessions/            # session event logs (existing)
  state/               # project state (existing)
  reports/             # claude_reports output (NEW)
  retrospectives/      # end-of-project reflections (NEW)
  learned/             # auto-drafted skills from retrospectives, pre-review (NEW, later)
```

Skills and prompts stay in `.github/` where Copilot and Claude extensions expect them.

---

## 5. The eyes-and-hands architecture (annotation loop)

This is the most important system to get right. Label scarcity is the binding constraint across 180 traits. Every hour of human labeling time has to produce maximum information.

### 5.1 Three models working together

- **SAM2 as the eyes (localization).** Upgrade from SAM1. Faster, handles video/multi-frame, better on small objects. Use ViT-B variant for throughput; ViT-L when quality matters for a specific case.
- **Claude as the brain (semantics + orchestration).** Does not look at every pixel. Looks at rendered overlays: SAM2 candidate masks numbered on the image. Reasons about what they are. Handles edge cases and triage.
- **Lightweight classifier as the multiplier.** DINOv2 or CLIP embeddings feed a small classifier trained on accepted labels. Scores new candidates. Auto-accepts high confidence, routes uncertain to human or Claude review.

Claude is not in the per-image loop once the model is bootstrapped. The classifier handles volume. Claude handles setup, edge cases, uncertainty triage, and deployment decisions. This keeps API costs sane.

### 5.2 Bootstrap workflow (zero labels on a new trait)

```
1. Human uploads 5–10 example images.
2. Claude reads trait profile for context.
3. Claude + SAM2 render candidate masks on each image.
4. Claude proposes labels based on visual reasoning and trait knowledge.
5. Human confirms or corrects in ~30 seconds per image.
6. System has 50–100 labeled instances.
```

### 5.3 Active learning loop (growing to a working model)

```
1. SAM2 generates candidates on unlabeled images.
2. Embedding classifier scores each candidate.
3. High confidence → auto-label.
4. Low confidence → batched for review.
5. Claude pre-triages the review batch, filtering obvious mistakes and surfacing genuine uncertainty.
6. Human reviews what remains.
7. Retrain classifier on accepted labels.
8. Repeat until model_readiness() says we're done.
```

### 5.4 Modality-specific backends

The interface (`propose_annotations`, `review_batch`, etc.) stays the same. The backend changes:

- **2D RGB imagery:** SAM2 + DINOv2/CLIP classifier.
- **Point clouds (LiDAR, SfM):** Tree segmentation via CHM + watershed, or point cloud foundation models when they mature. Annotation is on tree instances, not pixels.
- **NIR / hyperspectral:** No spatial structure. Annotation is a whole-sample value or class. Active learning still applies. Different tool backend entirely, same interface.
- **Temporal traits (phenology):** SAM2 video mode. Annotation unit is a time series per plant. Agent sees the sequence and identifies onset, midpoint, completion.

Claude never needs to know the backend. It calls `propose_annotations(image_path, trait)` and receives candidates. The trait profile tells the tool which backend to use.

### 5.5 What to build first

In priority order:

1. Upgrade SAM1 to SAM2 in [tcip_annotation/sam_wrapper.py](packages/tcip-annotation/src/tcip_annotation/sam_wrapper.py).
2. Wire up DINOv2 embedding extraction as a new module.
3. Train a tiny classifier head on accepted labels (scikit-learn is fine, does not need to be deep).
4. Replace `sam_auto_label` and `accept_candidates` with the new `propose_annotations` + `commit_labels` interface.
5. Add `model_readiness` with explicit criteria (val metrics, uncertainty distribution, labeling velocity).
6. Exercise on hazelnut catkin phenology detection as a regression test.

Do not build modality-specific backends until 2D RGB is solid. Do not add active learning selection strategies beyond uncertainty + diversity until we have real data on what works.

---

## 6. The self-improving meta-loop

The goal: every Claude session leaves the system smarter than it started.

### 6.1 Three layers of observability

**Layer 1: Full interaction logging.** Every message, every tool call, every result. Extend the existing `audit.jsonl` to capture conversation turns in addition to tool calls. Already mostly there.

**Layer 2: Structured failure tags.** `claude_reports(category, detail, context)` is the most important new tool in the system. Claude calls it whenever it hits friction. Categories:

- `missing_tool` — needed capability does not exist.
- `ambiguous_data` — data is unclear, could be interpreted multiple ways.
- `cant_find_file` — referenced path does not exist or is ambiguous.
- `confused_about_domain` — trait, crop, or breeding concept is unclear.
- `failed_repeatedly` — same operation failing more than 2–3 times.
- `needs_human_judgment` — decision beyond Claude's purview (breeder priorities, biological interpretation).
- `unexpected_behavior` — something worked but not how Claude expected.

The agent prompt must actively encourage calling this tool. LLMs are biased toward pushing through problems by making assumptions. We need to reward stopping and reporting instead.

**Caveat on self-reporting accuracy.** LLMs are not reliable self-reporters on a structured taxonomy. When asked to pick from seven categories, I will systematically over-pick `unexpected_behavior` (cheapest label) and under-pick `missing_tool` (requires noticing a counterfactual capability I wasn't told about). Mitigations:

- The PD reviews reports and relabels as part of the retrospective loop (§6.2).
- Include free-text `detail` alongside the categorical tag so the raw signal is preserved even when the label is wrong.
- Track over time whether category distribution drifts; if one category dominates, suspect reporting bias before concluding the system is broken in that way.

**Layer 3: Project retrospectives.** At the end of each project, call `project_retrospective()` which prompts structured reflection:

- What was the task?
- What worked?
- What did not work?
- What assumptions turned out to be wrong?
- What knowledge would future sessions benefit from?
- What tools were missing or hard to use?
- What would we do differently?

Output goes to `.tcip/retrospectives/<project_id>.md`. New sessions read recent retrospectives at startup.

### 6.2 The review process

The logs and retrospectives are input for the PD, not for Claude directly (yet). The PD reviews them periodically and asks:

- Which failure tags appear most often? These are tool gaps.
- Which assumptions were wrong? These are knowledge gaps.
- Which files could Claude not find? Data organization problems.
- Which tools were called unsuccessfully? Bad interfaces.
- Which tools were never called? Unnecessary or poorly described tools.
- **Which report labels were wrong?** Feedback for the agent prompt on reporting accuracy.

This is the loop that converts real project work into system improvements.

### 6.3 Skill auto-generation (later, not yet)

Once retrospectives accumulate (10+ projects minimum), Claude can review them and propose new skill documents or tool specifications. PD reviews, edits, and accepts into `.github/skills/`. Premature automation will generate noise. Do not build this until we have retrospective density.

### 6.4 What to build first

1. `claude_reports` tool. Append to `.tcip/reports/YYYY-MM-DD.jsonl`.
2. `project_retrospective` tool. Prompts structured reflection, writes markdown to `.tcip/retrospectives/`.
3. `load_retrospectives` tool. Reads recent retrospectives so new sessions have context.
4. Update agent instructions to actively encourage `claude_reports` on friction.
5. Nothing else until we have a month of real usage data.

---

## 7. Roadmap

Sequenced work. Earlier items unblock later items. Do not skip ahead.

### Phase 0: Pre-exercise refinement (current phase, ~1–2 weeks)

**Revised 2026-04-20.** The original plan was "no code changes before the exercise." We are revising that: the exercise should test what we *don't* already know, not re-surface deficiencies that are diagnosable without it. The distinction is between *speculative* cleanup (pruning tools based on guesses) and *known-wrong cleanup* (fixing things whose wrongness is independent of exercise outcome).

Do first:

1. **Create `.tcip/retrospectives/` and `.tcip/reports/`** so friction has a place to land from the start.
2. **Build the meta-loop tools:** `claude_reports`, `project_retrospective`, `load_retrospectives`. We cannot run a meaningful exercise without these; friction vanishes into chat history otherwise.
3. **Remove the two-layer paradigm from the agent prompt and `pipeline-design` skill.** Text change. Diagnosable from the trait inventory, not from exercise data.
4. **Delete `VALID_PHASE_TYPES` and phase enforcement** in [orchestrator.py](packages/tcip-mcp/src/tcip_mcp/pipelines/orchestrator.py). Also a text/code change with no exercise dependency.
5. **Upgrade SAM1 → SAM2** in [tcip_annotation/sam_wrapper.py](packages/tcip-annotation/src/tcip_annotation/sam_wrapper.py). Justified by published benchmarks on small objects (chestnut burs, catkins), not by needing exercise data.
6. **First-pass tool audit:** go through all 54 tools and flag candidates for deletion. Do NOT delete yet — evidence comes from the exercise.
7. **Update agent prompt** to encourage friction reporting via `claude_reports`.

Do NOT do before the exercise:

- DINOv2 + classifier architecture. The shape of the bootstrap loop is unknown until we try it.
- `get_trait_profile` tool. Schema is an open question; designing it in advance is guessing.
- Full tool deletion pass. Need friction data to separate bloat from "not-called-yet-but-needed."
- Annotation format collapse. Migration, not a fast change — defer until we know the ingestion boundary conditions from real data.

### Phase 1: The manual exercise (1 week, PD-driven)

Run hazelnut catkin phenology detection end-to-end with Claude Code as the only assistant. Data is already in `/data` (ground imagery). PD drives; Claude assists. Start in a fresh workspace. Use file access, bash, and the refined MCP tool surface. Narrate everything in chat. Let Claude try things before reaching for custom code, but timebox attempts so one stuck experiment doesn't consume the week.

Chestnut bur detection was the original candidate but requires Agisoft Metashape photogrammetric reconstruction → canopy instance segmentation → easyIDP image selection → detection. Too many external pipeline dependencies to isolate agent friction from pipeline friction.

At every point of confusion (PD's or Claude's), call `claude_reports` or have the PD note it. Every time a tool is wanted that doesn't exist, file `missing_tool`. Every surprising tool behavior, file `unexpected_behavior`.

At the end, call `project_retrospective`. Wait three days. Read it again. Then do a second pass.

The output is a list of what tools are actually needed, what knowledge needs to be captured in skills, what interfaces need to change. That list drives every subsequent phase.

### Phase 2: Foundation consolidation (2–3 weeks, evidence-driven)

Driven by Phase 1 retrospectives.

- Collapse annotation format handling: pick one internal representation, convert at ingestion, remove multi-format code from the agent-facing surface. Migration sequence in §3.3.
- Execute the tool audit. Delete what the exercise confirmed as bloat; keep what proved useful or necessary.
- Tighten the agent prompt based on observed friction patterns.

### Phase 3: Eyes and hands upgrade (3–4 weeks)

- DINOv2 embedding extractor.
- Lightweight classifier on top of embeddings (start with scikit-learn).
- Replace `sam_auto_label` / `accept_candidates` with `propose_annotations` / `commit_labels`.
- Add `model_readiness`.
- Regression test on hazelnut catkin detection: bootstrap from scratch, match or exceed the existing pipeline.

### Phase 4: Domain knowledge reorganization (ongoing)

- Rewrite per-crop skills as prose knowledge documents, not trait tables.
- Design and implement `get_trait_profile` and `get_dataset_catalog` (now with §9 open question resolved by Phase 1 data).
- Start with hazelnut and chestnut, expand as we work on other crops.
- Trait-level knowledge docs get written as we work on each trait, not upfront.

### Phase 5: Second real trait (4–6 weeks)

Pick something structurally different from hazelnut catkins. Chestnut bur detection is the natural candidate once the Metashape/easyIDP pipeline is in scope. Hazelnut EFB severity is another option: ordinal classification, different modality, different scoring conventions. Drive it through the agentic loop. Generate retrospectives. Find what breaks.

### Phase 6: Expand and harden

Once two real traits work end-to-end through the agent, tackle remaining traits in priority order based on breeder demand. Each trait is a full cycle: data acquisition, annotation, training, validation, delivery, retrospective.

Target for grant period end (Year 5): 15–20 traits across multiple pipeline patterns, deployable by breeders with no ML person in the loop.

---

## 8. Grant context

### The grant

USDA NIFA Specialty Crop Research Initiative (SCRI), FY2026.
Funding opportunity number: USDA-NIFA-SCRI-011676.
Pre-application deadline: June 15, 2026.
Target: Coordinated Agricultural Project (CAP), five years.
AAIMS special emphasis (Automation, AI, Mechanization, Sensors).
Target federal request: ~$10M over 5 years.

### What the grant funds

Not the idea. The team and infrastructure needed to take the prototype to production across 6 crops. Personnel are the primary line item: a research software engineer, a data architect, a data annotation technician, partial breeder and PD time, an extension specialist, and a subaward to a university ML/CV collaborator.

### How this document relates to the pre-application

The pre-application is a Stakeholder Relevance Statement. It is evaluated by industry reviewers, not ML researchers. The document they read emphasizes:

- Why automated phenotyping matters to specialty crop breeders (55 of 100 review points come from stakeholder engagement).
- How the platform delivers value to growers and industry (not how the ML works).
- Support letters from growers and industry (Appendix C, weighted heavily).
- Transdisciplinary approach integrating breeders, ML, extension from project inception.

The technical sophistication of the platform is not the selling point. The selling point is that breeders defined the problem and will use the result.

The full submission roadmap and internal planning document lives separately from this repository, in the PD's grant files.

### What to do when grant work intersects with technical work

During April–June 2026, the PD's attention is split between the technical prototype and the grant. If the PD asks you to work on grant-related things (drafting text, reviewing NOFO compliance, preparing figures), do that in a focused session. Do not mix grant work into code sessions.

---

## 9. Open questions

Things we have not figured out. Do not invent answers. Surface these to the PD when they become relevant.

1. **How much API cost is acceptable?** The agent makes many LLM calls. At 180 traits and iterative active learning, budget matters. We have not priced this out.

2. **Where does the system run?** Currently the PD's desktop. Eventually a remote server colleagues can access. We do not know what that server looks like, who administers it, or how multi-user sessions coordinate on shared state.

3. **How does the S3/Globus integration work?** Imagery data lives (or will live) in S3 via Globus endpoint. We have not designed the data ingestion path from that source into TCIP projects.

4. **Who hosts SAM2 and DINOv2?** Local inference works fine on the PD's desktop for one user. Multi-user remote deployment needs an inference server. Undesigned.

5. **What is the minimum viable `get_trait_profile` schema?** We have skill files today. We have not specified what a tool-returned trait profile looks like, what fields are required, what is optional. **Related sub-question:** does it return content beyond "load this skill file into context"? If not, the tool is ceremony around a file read and may not justify itself.

6. **How do we validate against breeder manual scoring?** The gold standard for phenotype quality is often "does the ranking of genotypes match what the breeder would rank." We have not designed the validation methodology.

7. **What is the annotation GUI story long-term?** The VS Code extension is one path. The FastAPI web app is another. Breeders may want neither. Possibly a tablet-optimized PWA. Undecided.

8. **How do we preserve audit lineage when Claude writes scripts instead of calling tools?** If script-based ML replaces tool-based ML per §3.1/§4.1, we need a convention: where the script lives, how it's logged, how `.tcip/audit.jsonl` references it. Undesigned.

9. **How do we timebox "let Claude try things" in Phase 1?** Left unbounded, one stuck attempt can eat the week. Need a rule of thumb — e.g., "20 minutes of unproductive attempts → escalate to PD intervention." Not yet decided.

---

## 10. Operational guidance for Claude sessions

### Before you touch anything

1. Read this document in full.
2. Read [.github/copilot-instructions.md](.github/copilot-instructions.md) for repository conventions.
3. Check `.tcip/retrospectives/` for recent session notes. If any exist, skim them.
4. Check `.tcip/reports/` for recent friction reports.
5. Ask the PD what they are trying to accomplish in this session.

### When the PD asks you to do something

1. If the request conflicts with the direction in this document, say so. Do not silently work around it.
2. If the request requires a tool that does not exist, call `claude_reports("missing_tool", ...)` (once that tool exists) or note it in chat.
3. If you can accomplish the task by writing a Python script instead of adding an MCP tool, prefer the script — *unless* the operation needs to be auditable, in which case prefer a tool. See §3.1.
4. If the task involves images, use the visual reasoning loop: render, view, decide.
5. If you are confused, say so. Do not guess.

### When you are tempted to add a new MCP tool

Ask: what can Claude not do without this tool?

- If the answer is "produce a tamper-evident audit record of a state-changing operation," add the tool.
- If the answer is "access long-running infrastructure" (training, SAM, etc.), add the tool.
- If the answer is "access domain knowledge it does not have," add the tool.
- If the answer is "write a reliable script quickly," maybe add the tool — but check whether a script in `scripts/` with experiment-logging is the better call.
- If the answer is "save typing," no, do not add the tool.

### When a session ends

Call `project_retrospective()` if you accomplished something substantial, even if incomplete. The retrospective is how we learn.

### When you notice something is wrong with this document

Say so. This document is a living artifact. It is wrong about some things. Those are the parts the PD most needs to know about.

---

## 11. Revision history

- **2026-04-20** — Revised by Opus 4.7 + Zack Loken. Changes: (1) Reversed Phase 0 from "no code changes" to "refine known-wrong tooling first, then exercise," redefined phases accordingly. (2) Fixed tool count from "~40" to "54" with per-file breakdown. (3) Added audit-trail caveat to §3.1/§4.1 principle on tool-vs-script. (4) Added migration-not-deletion caveat to §3.3 on annotation formats. (5) Added LLM self-reporting bias caveat to §6.1. (6) Removed former §9.8 (Anthropic collaboration) — PD has not reached out yet, so it's not an open question, it's a not-started item. (7) Added new open questions: audit lineage for scripts (§9.8), timeboxing Phase 1 attempts (§9.9). (8) Reframed document from handoff-letter to plan-of-record format.
- **April 2026 (original)** — Initial draft by Opus 4.7 after multi-session design review with the PD.

---

## Appendix: contact and references

- **PD:** Zack Loken, Imagery Analyst at Savanna Institute (MS Renewable Natural Resources). zack@savannainstitute.org.
- **Team:** 3 tree crop breeders + 1 breeding operations manager, all PhD plant breeders.
- **Institution:** Savanna Institute, agroforestry nonprofit, previously funded by SCRI and NIFA.
- **Existing production pipelines (outside TCIP):**
  - https://github.com/savannainstitute/chestnut-burr-detection-from-drone
  - https://github.com/savannainstitute/hazelnut-biomass-from-drone
- **Grant NOFO:** FY26-SCRI-PA-NOFO (in grant application materials).
- **Allen Institute / HHMI precedent for agentic AI in science:** https://www.anthropic.com/news/anthropic-partners-with-allen-institute-and-howard-hughes-medical-institute
