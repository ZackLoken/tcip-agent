# Handoff Prompt for Implementation Agent

Copy everything below the line and paste it as your first message to the new Claude Opus agent in VS Code Copilot chat.

---

## Your Role

You are building a new application from scratch in this empty repo. You are an expert Rust and Python engineer. The complete specification is in `docs/system_redesign/`. Read ALL of it before writing any code.

## Reading Order

Start here — these two index files link to everything else:

1. `docs/system_redesign/10-build-plan-index.md` — monorepo structure, 5-phase sequence, dependency graph, prerequisites
2. `docs/system_redesign/09-gap-analysis-index.md` — 14 architectural decisions (answers to consultation questions) + links to all adaptation docs

Then read the reference docs (claw-code architecture you're adapting from):
- `00-reference-index.md` through `08-recovery-plugins.md`

Then the adaptation docs (what changes from claw-code):
- `adaptation/01` through `adaptation/10`

Then the phase specs (what to build, in order):
- `phases/01-mcp-server.md` through `phases/05-full-loop.md`

## What This System Is

An AI agent that automates tree crop phenotyping. Given a trait to measure (e.g., "hazelnut catkin phenology"), the agent designs a computer vision pipeline, generates PyTorch training code, orchestrates training/HPO, and delivers per-plant CSV outputs. The user interacts via a desktop GUI with chat panel, annotation canvas, and results viewer.

## Architecture (Non-Negotiable)

- **Monorepo**: `packages/tcip-annotation/`, `packages/tcip-mcp/`, `packages/tcip-gui/` (Python), `tcip-agent/` (Rust)
- **Rust agent core**: Adapted from claw-code (a Rust rewrite of Claude Code). See reference docs 01-08 for the source architecture. The agent is NOT a wrapper — it's a proper conversation runtime with tool dispatch, permission enforcement, and session management.
- **Python MCP server**: Domain tools (registry queries, data loading, training, inference, evaluation) exposed via MCP over stdio. Uses FastMCP.
- **Sibling process model**: GUI spawns BOTH the Rust agent AND the Python MCP server as child processes. Agent connects to MCP server via stdio. GUI connects to agent via stdio.
- **Subagent modes** (not separate processes): PipelineDesigner (Opus), CodeGenerator (Sonnet), TrainingOrchestrator (Sonnet), ResultsAnalyzer (Sonnet). Mode switch = different system prompt skills + tool filter + model.
- **Skills**: 7 plain markdown files in `skills/` directory, injected into system prompt by a `SkillInjector` component using claw-code's `append_section()` API. See `adaptation/10-skills-architecture.md`.
- **Permission levels**: ReadOnly, WorkspaceWrite, FullAccess. NOT claw-code's 5-level system.
- **Config**: 4-level cascade: defaults.toml → user.toml → project .tcip/config.toml → CLI flags.
- **Sessions**: JSONL conversation logs in `<project>/.tcip/sessions/`.
- **Project state**: `<project>/.tcip/state.toml` tracks active pipeline, models, datasets.

## Hard Constraints

- **Pure PyTorch**: All models are `nn.Module`. No Ultralytics, no MMDetection, no HuggingFace model wrappers.
- **torchvision.transforms** for augmentation. Not albumentations.
- **No schema.py**: The registry (`registry/crops.yml`) is queried directly as YAML → dict → JSON. No Pydantic validation layer, no Python enums. See `adaptation/09-schema-registry-redesign.md`.
- **Generic abstractions**: `ImageTiler`, not `CanopyTiler`. `run_training`, not `train_catkin_model`.
- **Two-layer pipeline**: Every trait has (1) isolation task (find the plant/object) and (2) ML task (extract the trait). Plus optional temporal aggregation for time-series traits.
- **HITL checkpoints**: 5 mandatory human-in-the-loop gates. Agent MUST pause and get user approval at: pipeline design, config review, training launch, results review, model deployment.
- **6 crops only**: chestnut, hazelnut, persimmon, black_locust, currant, elderberry.

## MVP Target (Phase 5)

Hazelnut catkin phenology: 4 date traits (catkin_05per_date, catkin_50per_date, catkin_95per_date, catkin_elongation_date). Two-layer pipeline: bush isolation (detection) → catkin detection + classification (by phenological stage) → temporal aggregation (fit sigmoid curves per plant to derive date thresholds) → per-plant CSV.

## What I Provide

- Hazelnut catkin images and YOLO-format annotations (I'll give you the path)
- A pre-trained hazelnut catkin detection model (I'll give you the checkpoint path)
- `registry/crops.yml` with 180 traits × 6 crops (already in this repo)
- `docs/system_redesign/` (already in this repo)
- Access to claw-code source (I'll give you the path — it's a local Rust repo)

## About Me

I'm an imagery analyst, not a software engineer. I can read and modify Rust but I'm not fluent. I'm very comfortable with Python and ML/CV concepts. I can review code, test things, and provide domain feedback. Don't ask me to write Rust from scratch.

## How to Work

1. **Start with Phase 1** (Python MCP server). It has zero Rust dependencies and I can test it immediately with any MCP client.
2. Each phase is independently testable. Don't move to the next phase until the current one works.
3. When you're unsure about a decision, check the adaptation docs — most decisions are already made and documented.
4. For the Rust agent (Phase 2), you'll need me to provide the claw-code source path so you can reference it directly. Ask me for it when you get there.
5. Write tests as you go — each phase doc has specific test criteria.
6. The skill files (`skills/*.md`) should be created during Phase 1 or 2 — see `adaptation/10-skills-architecture.md` for the consolidation plan (13 existing files → 7 new files). The source material is in `skills/` in this repo.

## First Task

Read all docs in `docs/system_redesign/` following the reading order above. Then present me a summary of your understanding and your plan for Phase 1. Do NOT start coding until I confirm your understanding is correct.
