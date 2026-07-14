---
name: pipeline-design
description: "Pipeline patterns for phenotyping. The composable ML component registry is a library, not a scaffold. Choose a pattern that fits the trait — do not force every trait through the same structure. Load when deciding how to measure a new trait, choosing or composing an ML pipeline, or picking a model architecture for a trait."
---

# Pipeline Design

## There is no universal pipeline structure

Different traits need different pipeline shapes. An earlier version of this skill led with a "Two-Layer Paradigm" (Isolation → Task → Post-processing) and treated it as universal. That paradigm is **one pattern among many**, not a scaffold every trait must fit into. It works well for traits that localize to plant-parts inside pre-segmented plants (catkin counts on isolated bushes, burs on isolated trees). It breaks for:

- Traits that are not localizable (overall plant vigor from a canopy photo).
- Traits requiring temporal reasoning across many images (phenology onset dates).
- Traits requiring 3D reasoning (crown volume, biomass from point clouds).
- Traits that are relational (how does this plant compare to its neighbors on the same day).
- Traits from non-spatial data (NIR spectra for kernel oil percentage).

Pick a pattern from the library below based on the trait's characteristics. If none fit, design a new one — and capture it as a retrospective so it becomes part of this library next time.

## Pattern library

### Pattern A: Isolate → Detect/Segment → Aggregate

```
aerial imagery → plant instance seg → per-plant crops → part detection → per-plant count/CSV
```

Good for: counting plant parts on discrete plants (chestnut burs, hazelnut catkins, elderberry fruit clusters) when plants are individually separable in imagery.

Failure modes: plants that overlap or merge; very small parts where SAM-style candidates are unreliable.

### Pattern B: Whole-plant classification/regression

```
canopy photo → classifier or regressor → per-plant score → CSV
```

Good for: ordinal severity scoring (EFB severity), overall vigor scoring, traits where the phenotype is gestalt rather than localizable.

Failure modes: training label scarcity; strong lighting/weather covariates.

### Pattern C: Point-cloud tree segmentation → tree-level geometric measurement

```
LiDAR or SfM point cloud → ground filter → CHM → watershed tree seg → per-tree geometry → CSV
```

Good for: biomass, crown volume, height, DBH-proxy, canopy architecture.

Failure modes: dense canopies where watershed under-segments; low point density; ground-plane noise.

### Pattern D: Temporal / phenology sequence

```
image time series → per-plant isolation → per-time-step trait estimate → curve fit → onset/midpoint/completion date → CSV
```

Good for: phenology traits (flowering onset, catkin elongation, leaf-out date, senescence).

Failure modes: irregular sampling intervals; missing time points; wind/lighting confounding trait signal.

### Pattern E: Non-spatial spectral

```
spectral reading (NIR, hyperspectral) → preprocessor → regression/classification → per-sample value → CSV
```

Good for: kernel oil percentage, moisture, sugar content, disease screening from reflectance.

Failure modes: instrument drift; sample preparation variability; calibration transfer across instruments.

### Pattern F: Relational / contextual

```
plot-level imagery → per-plant detection → contextual ranking within plot/block/day → rank-based phenotype → CSV
```

Good for: traits where breeders rank rather than measure absolutes (relative vigor, relative bloom advancement).

Failure modes: requires dense plot layout metadata; needs enough plants per plot for ranking to be meaningful.

## Composable ML component registry

Models are built from specs when you need a standard CNN/transformer architecture:

```python
model_spec = {
    "backbone": {"name": "resnet50", "pretrained": True},
    "neck": {"name": "fpn"},
    "heads": [
        {"name": "detection_head", "task": "detection", "num_classes": 3}
    ],
    "loss": {"name": "focal_loss"}
}
```

The registry is a **library**, not a constraint. Available components:
- **Backbones**: resnet18/34/50/101, mobilenet_v3, efficientnet, vit, swin
- **Necks**: FPN, PAN, BiFPN
- **Heads**: detection, classification, segmentation, regression, ordinal
- **Losses**: focal, cross_entropy, smooth_l1, dice, coral

You can also ignore the registry entirely and write a model from scratch. No architecture is forced.

## Multi-phase pipelines

When a pattern requires chaining phases (Pattern A, C, D), use a pipeline spec:

```python
pipeline_spec = {
    "name": "hazelnut_catkin_phenology",
    "phases": [
        {"name": "isolate_bushes", "task": "instance_seg", "model_spec": {...}, "output": "bush_crops"},
        {"name": "detect_catkins", "task": "detection", "input": "bush_crops", "model_spec": {...}, "output": "catkin_detections"},
        {"name": "classify_stage", "task": "classification", "input": "catkin_detections", "model_spec": {...}, "output": "catkin_classes"},
        {"name": "aggregate", "task": "aggregation", "input": "catkin_classes", "output": "phenology_csv"}
    ]
}
```

Phase names are not restricted to a fixed set. Use names that describe what the phase does for this specific trait.

## Tools

| Tool | Purpose |
|------|---------|
| `list_components` | List available registry components |
| `recommend_model` | Get model architecture recommendation for a task |
| `validate_model_spec` | Validate a model spec against the registry |
| `validate_pipeline_spec` | Validate a multi-phase pipeline spec |
| `run_pipeline` | Execute a full pipeline |
| `compose_and_summarize` | Build a model from spec and show architecture summary |

## Design principles

- **Match the pattern to the trait, not the trait to the pattern.** If the pattern list doesn't fit, design a new one.
- Start with the simplest viable architecture for the chosen pattern.
- Use pretrained backbones unless data is very different from ImageNet.
- Use progressive unfreezing for transfer learning.
- Validate pipeline specs before executing.
- **Write a retrospective** (`project_retrospective`) when you finish, noting whether the pattern you chose was right. That feedback grows the library.
