---
name: pipeline-design
description: "Composable ML pipeline design with the component registry. Multi-phase pipelines for isolation, task execution, and post-processing."
---

# Pipeline Design

## Two-Layer Paradigm

Every crop analysis follows: **Isolation → Task → Post-processing**

1. **Isolation**: Segment or detect individual plants from aerial/ground/lab imagery
2. **Task**: Per-plant classification, detection, segmentation, regression, change detection, or tracking
3. **Post-processing**: Aggregate per-image results to per-plant CSV deliverables

## Composable Model System

Models are built from specs using a component registry:

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

The registry is a **library** — available components include:
- **Backbones**: resnet18/34/50/101, mobilenet_v3, efficientnet, vit, swin
- **Necks**: FPN, PAN, BiFPN
- **Heads**: detection, classification, segmentation, regression, ordinal
- **Losses**: focal, cross_entropy, smooth_l1, dice, coral

You can also build models from scratch — no constraints.

## Multi-Phase Pipelines

```python
pipeline_spec = {
    "name": "hazelnut_catkin_phenology",
    "phases": [
        {
            "name": "isolate_bushes",
            "task": "instance_seg",
            "model_spec": {...},
            "output": "bush_crops"
        },
        {
            "name": "detect_catkins",
            "task": "detection",
            "input": "bush_crops",
            "model_spec": {...},
            "output": "catkin_detections"
        },
        {
            "name": "classify_stage",
            "task": "classification",
            "input": "catkin_detections",
            "model_spec": {...},
            "output": "catkin_classes"
        },
        {
            "name": "aggregate",
            "task": "aggregation",
            "input": "catkin_classes",
            "output": "phenology_csv"
        }
    ]
}
```

## Tools

| Tool | Purpose |
|------|---------|
| `list_components` | List available registry components |
| `recommend_model` | Get model architecture recommendation for a task |
| `validate_model_spec` | Validate a model spec against the registry |
| `validate_pipeline_spec` | Validate a multi-phase pipeline spec |
| `run_pipeline` | Execute a full pipeline |
| `compose_and_summarize` | Build a model from spec and show architecture summary |

## Design Principles

- Start with the simplest viable architecture
- Use pretrained backbones unless data is very different from ImageNet
- Match head to task: detection_head for boxes, classification_head for classes
- Use progressive unfreezing for transfer learning
- Validate before executing: `validate_pipeline_spec` catches issues early
