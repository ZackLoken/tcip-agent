# 09 — Schema & Registry Redesign

## Problem

The previous design encoded domain knowledge in Python types:
- 16 string enums (`SensorType`, `MLTask`, `ModelFamily`, `Optimizer`, `LossFunction`, etc.)
- 9 Pydantic models (`Trait`, `Crop`, `PipelineStage`, `PipelineTemplate`, `ProjectConfig`, etc.)
- 2 auto-derivation maps (`SENSOR_DIMENSIONALITY`, `TASK_DEFAULT_MODEL`)
- 1 registry loader (`registry.py`) that validates YAML against Pydantic at load time

This approach has structural problems:
1. **Adding a crop or sensor type requires code changes** — enum edits, not just data edits
2. **Enums constrain agent reasoning** — `ModelFamily` lists 20 architectures, but the agent
   knows hundreds. `LossFunction` lists 8, but the agent can reason about any loss.
3. **Validates at the wrong boundary** — Pydantic validation fires at YAML load time, not at
   action time. A partially-defined trait (work in progress) can't be loaded.
4. **Pydantic models duplicate YAML structure** — the consumer is an LLM that parses JSON natively.
   Typed Python models add a translation layer without adding value.
5. **Auto-derivation maps are agent knowledge, not code** — "2D detection → YOLO" is a
   recommendation the agent should make contextually, not a hardcoded Python dict.

## claw-code reference pattern

claw-code's domain data model:
- **Tool specs** defined as simple Rust structs (name + description + JSON Schema + permission)
- **Config** from TOML/JSON files, deep-merged, accessed via typed but flat structs
- **No intermediate schema validation layer** between config files and usage
- **Adding an MCP server** = add a config entry, not a code change
- **Adding a tool** = add a struct, not modify an enum

The key principle: **data files define capabilities; code provides access; the agent reasons.**

## New approach

```
crops.yml  ──→  MCP tools (yaml.safe_load → query → return JSON)  ──→  agent reasons
                                                                          ↓
                                          skill files provide domain knowledge (not code)
```

### What stays
- **`crops.yml`** — source of truth for crops, traits, pipeline skeletons. Its structure
  IS the schema, documented by convention.
- **Pipeline group keys** — `ground_rgb_object_detection`, `aerial_multispectral_regression`, etc.
  These encode the two-layer pipeline structure elegantly.
- **Trait entries** — name, definition, format, category, crops list.

### What's removed
- **`schema.py`** — all 16 enums and 9 Pydantic models. Gone.
- **`registry.py`** — the Pydantic validation loader. Gone.
- **Auto-derivation maps in Python code** — moved to skill files.

### What replaces them

**MCP tools query YAML directly:**
```python
import yaml
from pathlib import Path

_REGISTRY = None

def _load():
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = yaml.safe_load(open(Path(__file__).parent / "crops.yml"))
    return _REGISTRY

def list_crops() -> dict:
    """Returns {crop_name: {trait_count, automatable_count, categories}}"""
    reg = _load()
    crops = {}
    for group_key, group in reg.items():
        has_pipeline = all(f in group for f in
            ("image_perspective", "sensor_type", "isolation_task", "ml_task"))
        for trait in group.get("traits", []):
            for crop in trait.get("crops", []):
                entry = crops.setdefault(crop, {"traits": [], "automatable": []})
                entry["traits"].append(trait["name"])
                if has_pipeline:
                    entry["automatable"].append(trait["name"])
    return {name: {"total": len(d["traits"]), "automatable": len(d["automatable"])}
            for name, d in sorted(crops.items())}
```

No Pydantic, no enums, no validation. Just YAML → dict → JSON over MCP.

**Domain knowledge moves to skill files:**

The auto-derivation logic (`sensor → dimensionality`, `task × dim → model family`)
becomes part of `skills/model-selection.md`:

```markdown
## Default Model Recommendations

| Sensor | Dimensionality |
|--------|---------------|
| rgb, rgbd, multispectral, hyperspectral, nirs | 2D |
| lidar, sfm, radar | 3D |

| Task + Dim | Recommended Family |
|------------|-------------------|
| 2D detection | Faster R-CNN (accuracy) or YOLO (speed) |
| 2D instance segmentation | Mask R-CNN |
| 2D semantic segmentation | U-Net / DeepLab |
| 2D classification | ResNet / EfficientNet / ViT |
| 3D point cloud | PointNet++ |

These are starting points. Agent should consider dataset size, compute budget,
and crop-specific characteristics when making final architecture decisions.
```

The agent references this knowledge from its system prompt, not from Python code.
It can override recommendations when its reasoning justifies it.

**Validation happens at action boundaries:**

```python
def launch_training(config: dict) -> dict:
    """MCP tool: validate and launch training."""
    # Validate HERE, not at YAML load time
    errors = []
    if not Path(config["data_dir"]).exists():
        errors.append(f"data_dir not found: {config['data_dir']}")
    if not Path(config["images_dir"]).exists():
        errors.append(f"images_dir not found: {config['images_dir']}")
    if config.get("device", "cpu").startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            errors.append("CUDA requested but not available")
    if errors:
        return {"success": False, "errors": errors}
    # ... proceed with training
```

### YAML schema (documented by structure)

The YAML format is self-documenting. No separate schema file needed.
The structure convention:

```yaml
# Top-level keys are pipeline group identifiers
# Format: {perspective}_{sensor}_{ml_task}
#
# Each group defines shared pipeline fields:
#   image_perspective: ground | aerial | satellite | lab
#   sensor_type: rgb | rgbd | multispectral | hyperspectral | lidar | sfm | ...
#   isolation_task: object_detection | instance_segmentation | semantic_segmentation
#   ml_task: object_detection | classification | regression | ...
#
# Each group contains a traits list:
#   traits:
#     - name: snake_case identifier
#       definition: human-readable description
#       format: numeric | binary | ordinal | nominal | date | string
#       category: morphology | phenology | yield | disease | quality | color
#       crops: [crop1, crop2, ...]
#
# Groups without pipeline fields contain non-automatable traits.
```

### Adding a new crop or trait

**Before (old approach):**
1. Edit crops.yml — add trait entry
2. If new sensor type: edit schema.py `SensorType` enum
3. If new ML task: edit schema.py `MLTask` enum  
4. If new model family: edit schema.py `ModelFamily` enum
5. If new category: edit schema.py `TraitCategory` enum
6. Run tests to validate Pydantic parsing

**After (new approach):**
1. Edit crops.yml — add trait entry
2. Done.

### What the training code uses internally

The MCP server's training pipeline code still needs typed configs for its own
internal use (model builder needs to know architecture, optimizer config, etc.).
These are simple dataclasses internal to the training module:

```python
@dataclass
class TrainingConfig:
    """Internal to training module. Not part of the registry schema."""
    model_arch: str          # "fasterrcnn_resnet50_fpn"
    num_classes: int
    learning_rate: float
    batch_size: int
    epochs_per_stage: list[int]
    optimizer: str           # "adamw"
    scheduler: str           # "cosine"
    device: str              # "cuda:0"
    # ... etc
```

These are populated from the agent's pipeline config dict, validated at launch time,
and used internally. They're NOT part of the public schema — they're implementation
details of the training module.

## Impact on phase docs

- **Phase 1**: MCP tools load YAML directly. No schema.py/registry.py porting.
  Package structure simplifies significantly.
- **Phase 2**: Agent reasons about trait data as JSON dicts, not typed objects.
  SkillInjector injects domain knowledge (model recommendations, etc.) via skill files.
- **Phase 5**: Agent can work with partially-defined traits during iterative development.
