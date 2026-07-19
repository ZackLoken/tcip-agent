---
name: pipeline-design
description: "Pipeline patterns for phenotyping. You build every model one way — an agent-written nn.Module + train(ctx) loop, via model_source. Choose a pattern that fits the trait — do not force every trait through the same structure. Load when deciding how to measure a new trait, designing an ML pipeline, or building a model architecture for a trait."
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

> **Out of current build scope.** 3D point clouds (LiDAR / SfM) are not built — see CLAUDE.md's
> Scope section and README's Roadmap. A `pointnet++` backbone exists but is intentionally
> unregistered; there is no point-cloud dataset/loader or task type. This pattern describes
> future capability, not something you can compose today.

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

> **Out of current build scope.** The dataset layer reads 2D imagery (RGB + N-channel
> rasters) — a bare non-imagery spectral reading has no dataset/loader today. This pattern
> describes future capability, not something you can compose today.

```
spectral reading (NIR, hyperspectral) → preprocessor → regression/classification → per-sample value → CSV
```

Good for: kernel oil percentage, moisture, sugar content, disease screening from reflectance.

Failure modes: instrument drift; sample preparation variability; calibration transfer across instruments.

### Pattern F: Relational / contextual

> **Out of current build scope.** README's Roadmap lists relational pipeline patterns beyond
> per-image traits as not built yet; there is no contextual-ranking task type today. This
> pattern describes future capability, not something you can compose today.

```
plot-level imagery → per-plant detection → contextual ranking within plot/block/day → rank-based phenotype → CSV
```

Good for: traits where breeders rank rather than measure absolutes (relative vigor, relative bloom advancement).

Failure modes: requires dense plot layout metadata; needs enough plants per plot for ranking to be meaningful.

## You own the model AND the training loop

The platform is a **toolkit you build with, not a mold you fill**. You are the CV scientist:
for every trait you write a bespoke `nn.Module` — from scratch or by importing the plain
building blocks (FPN/PAN necks, the classification/ordinal/regression/semantic-seg heads, the
losses, backbone wrappers, and `build_detector` + the `_build_*` detector functions) — and,
when the technique is novel, a custom training loop. There is no model spec, no composer, and
no component registry: nothing forces a model to a fixed shape or the default trainer.

The building blocks are plain importable symbols, e.g.:

```python
from tcip_mcp.pipelines.components.backbones import BackboneWrapper, _build_timm_backbone
from tcip_mcp.pipelines.components.necks import FPN, PAN
from tcip_mcp.pipelines.components.heads import ClassificationHead, SemanticSegHead
from tcip_mcp.pipelines.components.losses import build_loss, compute_class_weights
from tcip_mcp.pipelines.components.detectors import build_detector, BackboneNeckAdapter
```

Compose them inside your own `nn.Module`, or ignore them and write the network from scratch.
No architecture is imposed.

**Tailor the architecture to the data in hand** (CLAUDE.md: derive, don't pin):

- **Anchors from the GT box shapes**, not a fixed `(0.5, 1, 2)`. Feed the dataset's GT `(w, h)`
  through `pipelines.derivations.gt_aspect_ratios` and set anchor *sizes* from the GT object-size
  distribution, so anchors cover the objects that actually occur (e.g. elongated organs a default
  ratio can't match).
- **Strides / feature levels to the object scale** — add a finer pyramid level for tiny objects,
  drop levels you don't need.
- **Normalization to the batch size** — with the tiny batches large detectors force, BatchNorm
  statistics are unreliable; prefer `GroupNorm` (or another batch-independent norm).
- **Activations / layers where the data warrants** it — this is engineering judgment, not a menu.

**Two seams make bespoke work first-class, and the platform guarantees integrity around it:**

- `pipelines.model_build.build_model(config)` builds from a `model_source` — an *importable*
  builder you wrote (`{"builder": "my_module:build_net", "builder_kwargs": {...},
  "source_files": [...], "task": "detection", "in_chans": 3}`). It is imported, never `exec`'d,
  so the run is reproducible from source. `pipelines.model_contract`
  states the *only* model-side contract — the measurement boundary: your model must train (finite
  gradient loss) and emit inference output the library scorers consume. `check_model_contract` and
  `overfit_check` are the cheap pre-flight proofs it learns.
- `training_source` points the envelope at your custom `train(ctx)`. The `TrainContext` (`ctx`,
  `pipelines.training.envelope`) hands you the craft library — prebuilt leakage-free loaders,
  `ctx.build_optimizer` / `ctx.build_scheduler` / `ctx.evaluate` / `ctx.set_seed` — plus the
  envelope-owned sinks `ctx.log_metrics`, `ctx.save_checkpoint`, `ctx.record_artifact`,
  `ctx.should_cancel`. Route your loop's metrics and checkpoints through those sinks and the run
  stays audited, immutably versioned, and provenance-snapshotted no matter what your loop does.
  `ctx.default_train()` is **one convenience**, not a requirement — call it, extend it, or replace
  it entirely.

When the plain blocks and your own primitives both plateau on a trait, the next move is to research the
literature for a technique that fits — see the `cv-research` skill for the research→implement→validate
loop (and the rule that a new method must beat the baseline on the *measured phenotype* before you
trust it).

**Dimensional traits — mask geometry is a supported measurement.** For area / length / width of a
segmented organ, `pipelines.measurement.mask_geometry` (also `ctx.mask_geometry`) computes those on
a *validated* mask in pixels, and in mm when given a scale — geometry on a validated mask is a valid
measurement, subject to the same validate-before-you-trust rule as any other.

## Multi-phase pipelines

When a pattern requires chaining phases (Pattern A, C, D), use a pipeline spec whose
training phases each point at a `model_source` builder:

```python
pipeline_spec = {
    "name": "hazelnut_catkin_phenology",
    "phases": [
        {"name": "isolate_bushes", "task": "instance_seg", "model_source": {...}, "output": "bush_crops"},
        {"name": "detect_catkins", "task": "detection", "input": "bush_crops", "model_source": {...}, "output": "catkin_detections"},
        {"name": "classify_stage", "task": "classification", "input": "catkin_detections", "model_source": {...}, "output": "catkin_classes"},
        {"name": "aggregate", "task": "aggregation", "input": "catkin_classes", "output": "phenology_csv"}
    ]
}
```

Phase names are not restricted to a fixed set. Use names that describe what the phase does for this specific trait.

## Tools

| Tool | Purpose |
|------|---------|
| `run_pipeline` | Execute a full multi-phase pipeline |

## Design principles

- **Match the pattern to the trait, not the trait to the pattern.** If the pattern list doesn't fit, design a new one.
- Start with the simplest viable architecture for the chosen pattern.
- Use pretrained backbones unless data is very different from ImageNet.
- Use progressive unfreezing for transfer learning.
- **Write a retrospective** (`project_retrospective`) when you finish, noting whether the pattern you chose was right. That feedback grows the library.
