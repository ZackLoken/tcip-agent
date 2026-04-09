---
name: pipeline-design
description: "End-to-end ML pipeline design for trait phenotyping. Two-layer paradigm, pipeline patterns, architecture/backbone/loss selection, registry query workflow, and sensor-aware preprocessing."
triggers:
  - pipeline
  - design
  - two-layer
  - isolation
  - temporal
  - sigmoid
  - registry
  - phenotype
  - CSV
  - workflow
  - model
  - architecture
  - backbone
  - loss function
  - CORN
  - CORAL
  - detection
  - segmentation
  - classification
  - regression
modes: [PipelineDesigner, CodeGenerator]
priority: high
max_chars: 6000
---

# Pipeline Design

## Purpose

Design end-to-end ML pipelines for trait phenotyping. Every trait follows a two-layer paradigm: (1) isolation → (2) ML task, with architecture, backbone, and loss selected by sensor modality and dataset size.

## Two-Layer Pipeline Paradigm

Layer 1 — **Isolation**: Locate the plant/organ in the image.
Layer 2 — **ML task**: Measure/classify the trait on the isolated region.

Cross-crop reuse: isolation models (e.g. bush detector) can serve multiple traits on the same crop.

## Pipeline Patterns by ML Task

| ML Task | Stages | Example Trait |
|---------|--------|---------------|
| object_detection | data → train detector → predict → count/measure → aggregate → CSV | catkin dates |
| instance_segmentation | data → train seg → predict masks → measure area → CSV | canopy area |
| classification | data → train classifier → predict classes → aggregate → CSV | vigor class |
| regression | data → train regressor → predict → aggregate → CSV | trunk diameter |
| color_analysis | data → color extraction → stats → CSV | fruit color stage |
| point_cloud_analysis | data → ground filter → normalize → segment → measure → CSV | tree height |

## Temporal Aggregation for Phenology Traits

For date traits (e.g. catkin_05per_date), fit a sigmoid to per-date counts:
- 5% threshold → onset date, 50% → midpoint, 95% → completion

## Architecture Selection

### Task × Sensor → Architecture

| Task | 2D RGB | 3D Point Cloud | 1D Spectral |
|------|--------|----------------|-------------|
| detection | Faster R-CNN, FCOS | — | — |
| instance_seg | Mask R-CNN | — | — |
| classification | ResNet/EfficientNet | PointNet++ | 1D-CNN |
| regression | CNN + regression head | PointNet++ | PLS / 1D-CNN |
| semantic_seg | DeepLabV3+, U-Net | RandLA-Net | — |

### Backbone by Dataset Size

| Images | Backbone | Rationale |
|--------|----------|-----------|
| <500 | MobileNetV2 | Small model, less overfitting |
| 500-5K | ResNet-50 | Good capacity/data balance |
| >5K | ResNet-101 / EfficientNet-B4 | Enough data for larger model |

### Loss Function Selection

| Scenario | Loss |
|----------|------|
| Detection | Focal + Smooth-L1 (built into torchvision) |
| Classification (nominal) | CrossEntropy |
| Classification (ordinal, ~41 traits) | CORN or CORAL — NOT standard CE |
| Regression | Smooth-L1 or Huber |
| Segmentation | BCE + Dice |

## Sensor-Specific Preprocessing

**Point cloud** (LiDAR/SfM): ground classification (CSF/SMRF) → height normalize → noise removal → voxel subsample → CHM → tree segmentation.

**Spectral** (NIRS/hyperspectral): Savitzky-Golay smooth → SNV normalize → derivative transform → wavelength selection → outlier detection. NIRS is 1D — NEVER reshape to 2D for CNN.

**RGB preprocessing**: white balance, EXIF auto-orient, lens distortion correction, CLAHE contrast enhancement.

## Registry Query Workflow

1. `list_crops` → pick crop
2. `get_crop_traits(crop)` → identify target traits
3. `get_trait_info(crop, trait)` → get pipeline group, ML task, sensor
4. `find_traits_by_task(crop, task)` → find similar traits to batch
5. Design pipeline using pattern table above

### Query Strategy

1. **Scope**: `get_crop_traits(crop)` → how many traits total, what pipeline groups?
2. **Target**: `get_trait_info(crop, trait)` for each target → extract ML task + sensor
3. **Batch**: `find_traits_by_task(crop, task)` → group traits sharing the same isolation model
4. **Reuse**: Check if isolation model (layer 1) from another trait can be shared
5. **Validate**: Confirm sensor/perspective match between trait requirements and available data

## Key Constraints

- Pure PyTorch / torchvision only — no Ultralytics, MMDetection, or HuggingFace wrappers
- Ordinal traits MUST use ordinal-aware losses (CORN/CORAL), never standard CE
- NIRS/hyperspectral: NEVER use 2D CNNs — use 1D-CNN or PLS regression
- Every pipeline must produce a per-plant CSV as final output
- HITL checkpoint #1: present pipeline design for human approval before coding
- 6 crops only: hazelnut, chestnut, currant, elderberry, persimmon, black_locust
