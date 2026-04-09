# 08 — ML Domain Additions

Things that don't exist in claw-code at all. Entirely new to our system.

## 1. Two-Layer Pipeline Paradigm

Every automatable trait has:
1. **Isolation task** — find the plant/object in the scene (detection or segmentation)
2. **ML task** — extract the trait value (detection, classification, regression, etc.)

The agent must understand this paradigm and design pipelines accordingly.
This is domain knowledge injected via skills/system prompt, not code architecture.

## 2. Registry Data (YAML-direct, no Python types)

180 traits across 6 crops in `crops.yml`. Queried directly by MCP tools.
No intermediate Python types (no enums, no Pydantic models, no registry loader).

The YAML structure serves as the schema:
- Pipeline group keys encode sensor, perspective, dimensionality, ML task
- Each trait entry has: name, trait_category, trait_format, crops list
- Auto-derivation (sensor → dimensionality, task+dim → model family) is done
  by MCP tool logic, not by enum mappings in Python

Adding a new crop or trait = editing YAML. Zero code changes.
See [09-schema-registry-redesign.md](09-schema-registry-redesign.md) for full rationale.

## 3. Multi-Stage Progressive Unfreezing Training

Standard training pattern for all trait pipelines:
- Stage 0: head only (high LR)
- Stage 1: head + last backbone block
- Stage 2: head + last 2 blocks
- Stage 3: full fine-tune (low LR)

4-6 stages with per-stage LR scaling, early stopping, EMA.
Implemented in Python. Agent configures it, doesn't implement it.

## 4. Semi-Automated Annotation

The user's top pain point. Techniques to integrate:

**Model-assisted pre-labeling**: Run existing model (or foundation model)
on unannotated images, output predictions, human reviews in GUI.

**Active learning**: After initial model trained on small set, model identifies
images where it's least confident → prioritize those for annotation.

**SAM-assisted polygon generation**: Click a point → Segment Anything generates
polygon. Human refines. Much faster than drawing from scratch.

**Iterative refinement loop**:
1. Annotate small seed set (50-100 images)
2. Train quick model
3. Model predicts rest of dataset
4. Human reviews predictions (accept/edit/reject in review panel)
5. Retrain with expanded dataset
6. Repeat until quality threshold met

The agent orchestrates this loop. The GUI provides the review interface.

## 5. Model/Dataset Versioning

Currently ad-hoc. Need structured tracking:

**Dataset versions**: Each annotation session creates a snapshot.
- `datasets/hazelnut_catkins/v1/` — initial 100 annotated images
- `datasets/hazelnut_catkins/v2/` — +200 images after review round 1
- Metadata: image count, class distribution, annotator, date, parent version

**Model versions**: Each training run produces a versioned artifact.
- `models/hazelnut_catkin_det/v1/` — trained on dataset v1
- `models/hazelnut_catkin_det/v2/` — trained on dataset v2, HPO round 1
- Metadata: dataset version, config, metrics, checkpoint path, parent model

The agent should know what exists and exercise judgment:
- "Dataset v2 has 300 images with mAP 0.72. We could annotate more or try HPO."
- "There's already a tree crown detector for chestnut — reuse it for isolation."

## 6. Model Reuse at Isolation Stage

Key user requirement. Cross-crop isolation models:
- Tree crown detector (works across chestnut, black locust, persimmon, etc.)
- Shrub instance segmenter (works across hazelnut, elderberry, currant)
- Nut/fruit segmenter on flatbed scanner (works across species)

The registry should track which isolation models are generic vs species-specific.
When designing a new pipeline, the agent checks for existing isolation models first.

## 7. Per-Plant CSV Output

Every pipeline ultimately produces per-plant CSV:
```csv
plant_id,trait_name,value,confidence,date,model_version
HAZ-001,catkin_05per_date,2026-02-28,0.95,NA,hazelnut_catkin_det_v1
HAZ-001,catkin_50per_date,2026-03-12,0.93,NA,hazelnut_catkin_det_v1
HAZ-001,catkin_95per_date,2026-03-24,0.95,NA,hazelnut_catkin_det_v1
HAZ-001,catkin_elongation_date,2026-03-15,0.92,NA,hazelnut_catkin_det_v1
```

The postprocessing step aggregates model predictions → plant-level summaries.
Temporal traits (phenology) require multi-date aggregation:
- Per plant and date: count elongated vs total catkins
- Compute elongation percentage time series
- Fit sigmoid curve and extract threshold dates (5%, 50%, 95%)

## 8. Annotation Acceleration Research

Techniques I need to research further for the user (they expressed interest):

- **DINO / Grounding DINO**: Zero-shot detection with text prompts.
  "Detect catkins" → bounding boxes without training. Quality varies.
- **SAM 2**: Improved segment-anything with video/temporal support.
- **Interactive segmentation**: Click positive/negative points → refine mask.
- **Embedding-based similarity**: Annotate one example → find similar objects
  across dataset automatically.
- **Weak supervision**: Use noisy auto-labels + clean manual labels together,
  with noise-aware training losses.

These are future enhancements, not MVP requirements. But the architecture should
not preclude them — the annotation engine needs to support "prediction overlay"
and "review workflow" patterns from the start.
