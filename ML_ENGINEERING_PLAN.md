# ML Engineering Plan: Composable ML Primitives for Agentic Model Engineering

**Goal**: Rewrite the ML/CV pipeline layer so the AI agent acts as an ML engineer — reasoning about and composing backbone + neck + head(s) + loss + optimizer + sampler + augmentation for multi-phase crop phenotyping pipelines.

**Decisions**: PyTorch + timm + ONNX export | Ultralytics optional backend | Multi-head models | Multi-phase pipelines (isolation → extraction → classification → temporal) | Point cloud (LiDAR) | Clean rewrite | Both post-hoc and learned temporal | Automated active learning scoring | Local GPU (3D) + ONNX edge (RGB)

**Location**: `packages/tcip-mcp/src/tcip_mcp/pipelines/` — full rewrite of this subtree

**What stays**: MCP tools layer (`tools/*.py`), postprocessing/aggregation.py (temporal logic), data/classes.py (ClassMap), data/tiling.py. These get updated interfaces but not rewritten.

---

## Phase 0: Component Registry Foundation ✅

**File**: `pipelines/registry.py`

**ComponentRegistry class**:
- `register(name, category, factory_fn, metadata)` — register a component
- `get(name)` → factory function
- `list(category=None, filter_fn=None)` → list of `{name, category, metadata}`
- `build(name, **kwargs)` → instantiated component
- `describe(name)` → metadata dict for agent reasoning
- metadata includes: `description`, `valid_tasks`, `input_format`, `output_format`, `constraints`, `default_params`

**Global singletons**: `BACKBONES`, `NECKS`, `HEADS`, `LOSSES`, `OPTIMIZERS`, `SCHEDULERS`, `SAMPLERS`, `AUGMENTATIONS`

---

## Phase 1: Backbone + Neck Layer ✅

### `pipelines/components/backbones.py`

**BackboneWrapper class** (nn.Module):
- `forward(x) → dict[str, Tensor]` — returns multi-scale feature dict
- `out_channels: list[int]` — channel counts per scale level
- `freeze_to(stage: int)` — freeze layers up to stage N

**Registered backbones** (via timm): resnet18/34/50/101, efficientnet_b0–b4, mobilenet_v2/v3_small/v3_large, convnext_tiny/small, vit_small/base_patch16_224

### `pipelines/components/necks.py`

**Registered necks**: FPN, PAN, IdentityNeck, GlobalAvgPoolNeck

---

## Phase 2: Task Heads ✅

### `pipelines/components/heads.py`

**BaseHead(nn.Module)** ABC with `forward()`, `compute_loss()`, `decode()`, `task_type`, `default_loss`

| Head | Task | Default Loss |
|------|------|-------------|
| `ClassificationHead` | classification | cross_entropy |
| `OrdinalHead` (CORN) | ordinal | corn |
| `RegressionHead` | regression | smooth_l1 |
| `AnchorDetectionHead` | detection | focal + smooth_l1 |
| `SemanticSegHead` | semantic_seg | ce + dice |

---

## Phase 3: Loss Registry ✅

### `pipelines/components/losses.py`

10 losses: CrossEntropy, Focal, SmoothL1, Huber, BCE, Dice, GIoU, CORN, CORAL, CombinedLoss

`build_loss(name)` parses "bce+dice" notation into CombinedLoss.

---

## Phase 4: Model Composer ✅

### `pipelines/composer.py`

- `ComposedModel(nn.Module)`: backbone → neck → heads, multi-head, `freeze_backbone()`, `get_param_groups()`, `total_loss()`
- `compose_model(spec) → ComposedModel`
- `validate_model_spec(spec) → list[str]`
- `recommend_model_spec(task, dataset_size, sensor, num_classes, num_ranks) → dict`

---

## Phase 5: Data Layer ✅

### `pipelines/data/datasets.py` (rewrite)

**BaseDataset(Dataset)** ABC with `task_type`, `num_classes`, `class_distribution`, `num_samples`

| Dataset | task_type | Label Format | Target Dict |
|---------|-----------|-------------|-------------|
| `DetectionDataset` | detection | YOLO boxes `cls cx cy w h` | `{boxes, labels, image_id}` |
| `InstanceSegDataset` | instance_seg | YOLO polygons `cls x1 y1 x2 y2...` | `{boxes, labels, masks, image_id}` |
| `SemanticSegDataset` | semantic_seg | PNG mask images | `{mask: [H,W]}` |
| `ClassificationDataset` | classification | Folder structure or CSV | `{label: int}` |
| `OrdinalDataset` | ordinal | CSV (image, rank) | `{rank: int, num_ranks: int}` |
| `RegressionDataset` | regression | CSV (image, value) | `{value: float}` |

`build_dataset(task, root, split, transforms, class_map) → BaseDataset`

### `pipelines/data/augmentation.py` (rewrite)

Task-aware augmentation pipelines:
- Detection: RandomHFlip, ColorJitter, RandomCrop(box-aware), Mosaic, MixUp, GaussianBlur, CLAHE
- Classification: RandomHFlip, ColorJitter, RandomResizedCrop, RandAugment, CutMix
- Segmentation: RandomHFlip, ColorJitter, RandomCrop(mask-aware), ElasticTransform

`build_augmentation(task, config) → Compose`

### `pipelines/data/samplers.py` (new)

| Sampler | When to Use |
|---------|------------|
| `RandomSampler` | Balanced datasets |
| `ClassBalancedSampler` | Imbalanced classes (disease scoring) |
| `OverSampler` | Severe minority class (<10 samples) |
| `WeightedRandomSampler` | Custom per-sample weights |

`build_sampler(name, dataset) → Sampler`

---

## Phase 6: Generic Trainer ✅

### `pipelines/training/trainer.py` (rewrite)

Task-agnostic training loop using ComposedModel interface.

**TrainConfig** dataclass:
- `model_spec`, `dataset`, `augmentation`, `sampler`, `optimizer`, `stages` (progressive unfreezing), `early_stopping`, `mixed_precision`, `gradient_accumulation_steps`, `batch_size`, `num_workers`

**`train(run, config) → TrainRun`**: compose_model → build_dataset → build_sampler → DataLoader → stage loop → train loop → validate → checkpoint → early stopping → logging

**`task_collate(task) → callable`** — detection collate (list of dicts) vs classification collate (stacked tensors)

### `pipelines/training/optimizer_factory.py` (new)

Registered optimizers: SGD, Adam, AdamW, LAMB. Supports differential LR via `model.get_param_groups()`.

### `pipelines/training/stages.py` (rewrite)

Backbone-agnostic progressive unfreezing: `freeze_to: int` stage number, `apply_stage(model, stage_config)`

### `pipelines/training/hpo.py` (update)

Keep random search + Optuna. Task-aware search spaces: backbone, head, loss, optimizer, lr, batch_size, weight_decay, augmentation_strength, sampler.

---

## Phase 7: Multi-Phase Pipeline Orchestrator ✅

### `pipelines/orchestrator.py` (new)

**PipelineSpec** example (hazelnut catkin phenology):
```json
{
  "name": "hazelnut_catkin_phenology",
  "phases": [
    {"name": "isolate_bushes", "task": "instance_seg", "model_spec": {}, "output": "bush_crops"},
    {"name": "detect_catkins", "task": "detection", "model_spec": {}, "input": "bush_crops", "output": "catkin_detections"},
    {"name": "classify_catkins", "task": "classification", "model_spec": {}, "input": "catkin_detections", "output": "catkin_classes"},
    {"name": "temporal_aggregation", "task": "aggregation", "strategy": "sigmoid", "input": "catkin_classes", "output": "phenology_csv"}
  ]
}
```

**PipelineOrchestrator**: `validate_pipeline()`, `run_phase()`, `run_pipeline()`, `get_phase_status()`

**Phase types**: training, inference, cropping, aggregation, export

---

## Phase 8: Inference + ONNX Export ✅

### `pipelines/inference/predictor.py` (rewrite)

- `GenericPredictor(checkpoint_path, device, score_threshold)` — loads any ComposedModel checkpoint, auto-detects task
- `predict(image_path) → dict`, `predict_batch(paths) → list[dict]`
- `export_onnx(output_path, opset=17)` — dynamic axes
- `export_yolo(result, path)` — detection/seg results to YOLO format

### `pipelines/inference/onnx_runtime.py` (new)

- `OnnxPredictor(onnx_path)` — ONNX Runtime inference for edge deployment, same `predict()` interface

---

## Phase 9: Active Learning Pipeline ✅

### `pipelines/active_learning/scorer.py` (new)

- `UncertaintyScorer` — rank by prediction entropy/confidence spread
- `DiversityScorer` — rank by embedding distance from labeled set
- `CombinedScorer(uncertainty_weight, diversity_weight)`

### `pipelines/active_learning/selector.py` (new)

- `select_batch(scorer, unlabeled_paths, model, budget) → list[str]`
- `auto_accept(predictions, threshold=0.8) → list[dict]`
- `review_queue(predictions, low=0.3, high=0.8) → list[dict]`

---

## Phase 10: Point Cloud Primitives ✅

### `pipelines/components/backbones_3d.py` (new)

- `PointNetPPBackbone` — PointNet++ set abstraction layers

### `pipelines/data/datasets_3d.py` (new)

- `PointCloudDataset` — loads .las files via laspy, returns (points, features, targets)
- Preprocessing: ground classification (CSF), height normalization, voxel downsampling

### `pipelines/data/preprocessing_3d.py` (new)

- `ground_classify()`, `height_normalize()`, `voxel_downsample()`, `compute_chm()`

---

## Phase 11: Temporal Modeling ✅

### `pipelines/components/temporal.py` (new)

- `TemporalHead` — LSTM or Transformer over per-date embeddings → phenology date predictions

### Keep existing `postprocessing/aggregation.py`

- Add: `fit_gompertz()`, `fit_logistic_growth()` for growth curve traits

---

## Phase 12: Skill + MCP Tool Updates ✅

### Skills to update
- `skills/pipeline-design.md` — reference composable spec format
- `skills/training-config.md` — reference optimizer factory, loss registry, sampler options
- `skills/model-selection.md` — update backbone selection matrix with timm options

### MCP tools to update
- `tools/training_tools.py` — `launch_training()` accepts TrainConfig with model_spec
- `tools/model_tools.py` — `list_available_models()` queries all registries
- `tools/inference_tools.py` — use GenericPredictor, add ONNX export tool
- New: `tools/pipeline_tools.py` — `design_pipeline()`, `run_pipeline()`, `get_pipeline_status()`
- New: `tools/active_learning_tools.py` — `score_unlabeled()`, `get_review_queue()`

---

## File Map

### New files (created)
- ✅ `pipelines/registry.py`
- ✅ `pipelines/composer.py`
- ✅ `pipelines/components/__init__.py`
- ✅ `pipelines/components/backbones.py`
- ✅ `pipelines/components/necks.py`
- ✅ `pipelines/components/heads.py`
- ✅ `pipelines/components/losses.py`
- ✅ `pipelines/orchestrator.py`
- ✅ `pipelines/components/temporal.py`
- ✅ `pipelines/components/backbones_3d.py`
- ✅ `pipelines/data/datasets_3d.py`
- ✅ `pipelines/data/preprocessing_3d.py`
- ✅ `pipelines/data/samplers.py`
- ✅ `pipelines/training/optimizer_factory.py`
- ✅ `pipelines/training/generic_trainer.py`
- ✅ `pipelines/active_learning/scorer.py`
- ✅ `pipelines/active_learning/selector.py`
- ✅ `pipelines/inference/generic_predictor.py`
- ✅ `pipelines/inference/onnx_runtime.py`
- ✅ `tools/pipeline_tools.py`
- ✅ `tools/active_learning_tools.py`

### Rewritten files
- `pipelines/data/datasets.py` — adds 5 dataset types
- `pipelines/data/augmentation.py` — task-aware pipelines
- `pipelines/training/trainer.py` — task-agnostic loop
- `pipelines/training/stages.py` — backbone-agnostic freeze
- `pipelines/training/hpo.py` — task-aware search spaces
- `pipelines/inference/predictor.py` — generic, any task

### To delete (after all phases done)
- `pipelines/models/builder.py` → replaced by composer.py
- `pipelines/models/backbones.py` → replaced by components/backbones.py
- `pipelines/models/heads.py` → replaced by components/heads.py
- `pipelines/models/losses.py` → replaced by components/losses.py
- `pipelines/models/__init__.py`

### Kept (minor updates)
- `pipelines/data/classes.py` — ClassMap, no changes
- `pipelines/data/tiling.py` — minor interface update
- `pipelines/postprocessing/aggregation.py` — add gompertz/logistic
- `pipelines/postprocessing/export.py` — keep

---

## Execution Order

```
Phase 0 (registry) → Phase 1 (backbones+necks) → Phase 2 (heads) → Phase 3 (losses)
→ Phase 4 (composer) — depends on 0-3
→ Phase 5 (data layer) — parallel with 4
→ Phase 6 (trainer) — depends on 4+5
→ Phase 7 (orchestrator) — depends on 6
→ Phase 8 (inference+ONNX) — depends on 4
→ Phase 9 (active learning) — depends on 8
→ Phase 10 (point cloud) — depends on 0-2
→ Phase 11 (temporal) — depends on 2
→ Phase 12 (skills+tools) — depends on all
```

## Verification

1. Per-phase unit tests in `tests/test_composable_ml.py`
2. Integration test: `test_full_pipeline.py` — registry query → compose → train 2 epochs → infer → aggregate → CSV
3. Regression: Existing MCP tool tests must pass
4. GUI compatibility: TrainingDashboard reads metrics.jsonl — format preserved
5. Smoke test: MCP server `list_available_models()` returns all registries
