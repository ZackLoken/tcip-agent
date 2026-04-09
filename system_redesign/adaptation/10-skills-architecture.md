# 10 — Skills Architecture

## Problem

The current `skills/` directory has 13 skill files that are comprehensive domain
knowledge documents but have no formal integration mechanism. They are markdown
files with YAML frontmatter that nothing reads. The system_redesign docs reference
a `SkillInjector` component and subagent modes, but never specify:
- How skills are discovered and loaded
- How the system prompt is assembled from skills
- Size constraints (context window budget)
- The exact skill inventory needed
- How skills relate to claw-code's instruction file system

## Claw-code reference

Claw-code has NO skills system. It has:

1. **Instruction file discovery**: walks up from cwd, reads `CLAUDE.md`,
   `CLAUDE.local.md`, `.claw/CLAUDE.md`, `.claw/instructions.md`
2. **SystemPromptBuilder**: assembles prompt in fixed order:
   ```
   intro → output_style → system_rules → doing_tasks → actions_with_care
   → [DYNAMIC_BOUNDARY] → environment → project_context → claude_instructions
   → runtime_config → appended_sections
   ```
3. **Truncation**: 4,000 chars per instruction file, 12,000 chars total
4. **Deduplication**: FNV content hash prevents duplicate instructions
5. **`append_section()`**: public API for programmatic injection (unused in claw-code itself)

The key pattern: **static scaffolding + discovered context + programmatic extension**.

## TCIP skills architecture

We adapt claw-code's pattern and ADD skill injection as a first-class concept.

### System prompt assembly order

```
1. Base system prompt           (static, ~800 tokens)
   "You are an ML/CV agent for tree crop breeding programs.
    You design pipelines, generate code, orchestrate training,
    and deliver per-plant CSV outputs."

2. System rules                 (static, from claw-code's "System" + "Doing tasks"
                                 + "Actions with care" sections, adapted for ML domain)

3. [DYNAMIC_BOUNDARY]

4. Active skills                (0-3 skill files, selected by SkillInjector
                                 based on subagent mode)

5. Project context              (from <project>/.tcip/state.toml — crop, traits,
                                 pipeline stage, model versions, data status)

6. Registry excerpt             (relevant portion of crops.yml for active crop/traits,
                                 injected by SkillInjector only when relevant)

7. Environment                  (model name, cwd, date, platform, GPU info)

8. Tool descriptions            (auto-generated from native + MCP tool specs)
```

Sections 1-3 are static (~2K tokens). Section 4 is the skill injection layer
(budget: ~6K tokens for skills). Sections 5-8 are dynamic context (~4K tokens).
Total budget: ~12K tokens, well within Opus/Sonnet context limits.

### SkillInjector component

Lives in the Rust agent. Responsibilities:
- Map active subagent mode → skill file set
- Load skill files from `skills/` directory
- Truncate per claw-code limits (4K per file)
- Deduplicate by content hash
- Inject into system prompt via `append_section()`

```rust
pub struct SkillInjector {
    skills_dir: PathBuf,
    cache: HashMap<String, String>,  // name → content, loaded once
}

impl SkillInjector {
    pub fn load_skills(&mut self, skills_dir: &Path) -> Result<()>;
    pub fn get_skills_for_mode(&self, mode: SubagentMode) -> Vec<&str>;
    pub fn inject(&self, builder: &mut SystemPromptBuilder, mode: SubagentMode);
}
```

### Subagent mode → skill mapping

| Mode | Model | Skills injected | Tool filter |
|------|-------|-----------------|-------------|
| PipelineDesigner | Opus | pipeline-design, model-selection, crop-science | Registry queries, data tools (ReadOnly) |
| CodeGenerator | Sonnet | training-config, image-processing | File I/O, registry, data tools (WorkspaceWrite) |
| TrainingOrchestrator | Sonnet | training-config | Training, inference, model tools (FullAccess) |
| ResultsAnalyzer | Sonnet | crop-science, evaluation-method | Evaluation, model, inference tools (WorkspaceWrite) |

Note: the agent itself decides when to switch modes based on conversation phase.
The mode switch changes: (a) which skills are injected, (b) which model is used,
(c) which tools are available.

## Skill file format

Each skill is a single markdown file in `skills/`. No subdirectories, no references.md,
no YAML frontmatter. Just markdown with a consistent structure:

```markdown
# Skill: Pipeline Design

> Activated for: PipelineDesigner mode
> Budget: ~2000 tokens

## Purpose
[1-2 sentences: when this skill is relevant]

## Decision framework
[The actual domain knowledge, structured as decision trees/tables/rules]

## Key constraints
[Hard rules the agent must follow]

## References
[Inline citations — author, year, key finding. No separate file.]
```

**Why no YAML frontmatter**: The Rust SkillInjector reads markdown files. It doesn't
need structured metadata — the filename IS the identifier, and the mode mapping is
in Rust code (not in the skill file). This follows claw-code's pattern: CLAUDE.md
files are plain markdown with no metadata.

**Why no references.md**: Splitting references into a separate file adds complexity
without value. The agent doesn't query references separately — they're part of the
knowledge context. Inline them.

**Size constraint**: Each skill file should be under 4,000 characters (~1,500 tokens).
This matches claw-code's `MAX_INSTRUCTION_FILE_CHARS`. The SkillInjector truncates
files that exceed this limit.

## Skill inventory (7 skills, consolidated from 13)

The current 13 skills have significant overlap and are too granular. Consolidate to 7:

### 1. `pipeline-design.md`
**Replaces**: pipeline-design, registry-queries
**Content**:
- Two-layer pipeline paradigm (isolation → ML task)
- Pipeline patterns per ML task (detection, segmentation, classification, regression,
  change detection, color analysis, point cloud, motion tracking)
- Temporal aggregation for phenology traits (sigmoid fitting)
- How to query the registry (which MCP tools, query patterns)
- Architecture recommendations table (task × sensor → model family)
- Cross-crop isolation model reuse guidance

### 2. `model-selection.md`
**Replaces**: cv-model-selection, sensor-perspective-guide
**Content**:
- Sensor → dimensionality mapping (8 sensors)
- Perspective → sensor compatibility (4 perspectives)
- Task × dimensionality → recommended architecture
- Backbone selection by dataset size (small/medium/large)
- Loss function selection (especially ordinal: CORN/CORAL for ~41 traits)
- Pure PyTorch constraint (no Ultralytics, no MMDet)
- NIRS/hyperspectral: NOT standard 2D CNNs — use 1D-CNN or PLS regression

### 3. `training-config.md`
**Replaces**: training-strategy, hpo-strategy
**Content**:
- Multi-stage progressive unfreezing (4-6 stages)
- Per-stage LR scaling, optimizer state handoff
- FrozenBatchNorm transitions, mixed precision, EMA
- Early stopping, warmup, checkpoint strategy
- GPU memory budgeting
- HPO: multi-round narrowing (broad → targeted → edge-check)
- ASHA scheduler + Optuna TPE sampler settings
- Composite objective functions per task type

### 4. `image-processing.md`
**Replaces**: traditional-image-analysis, remote-sensing (partially)
**Content**:
- When to use traditional CV vs ML (decision table)
- Color spaces, thresholding, morphology, contour analysis
- Integration with ML (preprocessing, postprocessing, feature engineering)
- Point cloud preprocessing (CHM, ground classification, height normalization)
- NIRS/spectral preprocessing (SNV, derivatives, smoothing)
- Data format handling per sensor type

### 5. `annotation-workflow.md`
**Replaces**: annotation-automation
**Content**:
- Iterative refinement loop (seed → train → predict → review → retrain)
- Model-assisted pre-labeling workflow
- Active learning: confidence-based image prioritization
- SAM-assisted polygon generation
- Annotation quality validation
- Integration with HITL checkpoints #2 and #4

### 6. `crop-science.md`
**Replaces**: plant-science
**Content**:
- 6 crops: hazelnut, chestnut, currant, elderberry, persimmon, black_locust
- Growth forms (tree vs bush vs shrub), phenological stages
- Key trait targets per crop, imaging context
- Disease identification guidance
- Breeding program context (what breeders care about and why)

### 7. `evaluation-method.md`
**Replaces**: scientific-method (partially)
**Content**:
- Metric selection per task type (mAP, F1, R², MAE, Spearman-ρ)
- Per-class analysis, confusion matrix interpretation
- Failure case triage (worst predictions → annotation errors vs model errors)
- When to retrain vs HPO vs more data
- Ablation study design
- Statistical significance for breeding decisions

### Skills removed (no longer separate files)

| Old skill | Where it goes |
|-----------|---------------|
| data-mining | Agent's base system prompt (general capability, not domain skill) |
| self-learning | Agent's base system prompt (meta-reasoning, not injectable knowledge) |
| remote-sensing | Split between model-selection.md (sensor mapping) and image-processing.md (preprocessing) |
| sensor-perspective-guide | Merged into model-selection.md |
| registry-queries | Merged into pipeline-design.md |
| scientific-method | Split between evaluation-method.md (metrics) and base prompt (reasoning discipline) |

## How skills flow through the system

```
User message arrives
        ↓
Agent determines subagent mode (or stays in current mode)
        ↓
SkillInjector maps mode → skill set:
  PipelineDesigner  → [pipeline-design, model-selection, crop-science]
  CodeGenerator     → [training-config, image-processing]
  TrainingOrch.     → [training-config]
  ResultsAnalyzer   → [crop-science, evaluation-method]
        ↓
SkillInjector loads files, truncates to 4K each, deduplicates
        ↓
SystemPromptBuilder.append_section() for each skill
        ↓
Full prompt sent to API with tool definitions
```

When the agent switches mode mid-conversation, the system prompt is reassembled
for the next API call. Previous messages in the conversation history remain
unchanged — only the system prompt updates.

## annotation-workflow.md: special case

This skill is NOT mapped to a specific subagent mode. Instead, it's injected
when the conversation enters an annotation phase (detected by the agent calling
annotation-related tools). The SkillInjector can also inject skills based on
recent tool calls, not just mode:

```rust
if recent_tools.contains("load_annotations") || recent_tools.contains("canvas_control") {
    inject_skill("annotation-workflow");
}
```

This keeps the annotation workflow knowledge available across modes (both
CodeGenerator preparing data and TrainingOrchestrator reviewing predictions
may need it).

## Impact on phase docs

- **Phase 1**: Skills directory ships with the MCP server package (read-only data).
  The MCP server does NOT use skills — they're for the Rust agent's system prompt.
- **Phase 2**: SkillInjector implemented in Rust. Skill files loaded from `skills/`.
  Mode switching mechanism. System prompt assembly with skill injection.
- **Phase 5**: Full subagent mode flow demonstrated — PipelineDesigner → CodeGenerator
  → TrainingOrchestrator → ResultsAnalyzer with different skills at each stage.

## Migration from current skills

The 13 current skill files are the source material. The implementing agent should:
1. Read all 13 current SKILL.md files
2. Consolidate into 7 files per the inventory above
3. Strip YAML frontmatter, inline references, enforce 4K char limit
4. Place in `skills/` directory (flat, no subdirectories)
5. Implement SkillInjector in Rust per the spec above
