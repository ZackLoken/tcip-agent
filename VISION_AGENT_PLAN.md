# Vision Agent Plan

> **Goal**: Give the TCIP agent the ability to "look at" images, annotations, predictions, and training artifacts — enabling visual quality assessment, annotation review, and failure analysis.

## Current State

- **MCP tools return dicts only** — no image data flows through the pipeline
- **PIL/Pillow** present in both `tcip-annotation` and `tcip-mcp` (image I/O only)
- **OpenCV** present in `tcip-annotation` (SAM segmentation only — no drawing)
- **No rendering code exists** — all visualization happens client-side in webview JS
- **VS Code agent has `view_image`** — built-in tool that sends image files to multimodal models
- **FastMCP supports `Image` returns** — `from fastmcp.utilities.types import Image`

## Architecture Decision

### Hybrid approach (recommended)

1. **MCP tools render annotated images to disk** → save to `.tcip/artifacts/viz/`
2. **Return the file path in the text result** so the agent knows where to look
3. **Agent uses `view_image` on the saved file** when it needs to visually analyze
4. **Optionally also return `Image(path=...)`** for direct inline display in chat

This is the most reliable path because:
- `view_image` is the proven mechanism for sending images to multimodal models in VS Code
- Avoids large base64 payloads bloating MCP tool results
- Saved artifacts create an audit trail
- Works regardless of model's multimodal support

## Implementation Phases

### Phase 1: Rendering Engine (Python library)

Create `packages/tcip-annotation/src/tcip_annotation/viz.py` — a rendering module that draws annotations on images.

**Functions:**

```python
def render_detections(
    image_path: str,
    boxes: list[dict],           # [{"x1","y1","x2","y2","class_id"}]
    class_names: dict[int, str] | None = None,
    output_path: str | None = None,
    line_width: int = 2,
) -> str:
    """Draw bounding boxes on image. Returns output path."""

def render_segmentations(
    image_path: str,
    polygons: list[dict],        # [{"points": [(x,y),...], "class_id": int}]
    class_names: dict[int, str] | None = None,
    output_path: str | None = None,
    alpha: float = 0.3,
) -> str:
    """Draw filled polygons on image. Returns output path."""

def render_comparison(
    image_path: str,
    gt_boxes: list[dict],
    pred_boxes: list[dict],
    matches: list[dict] | None = None,  # from run_matching
    class_names: dict[int, str] | None = None,
    output_path: str | None = None,
) -> str:
    """Side-by-side or overlay of GT vs predictions with match lines."""

def render_grid(
    image_paths: list[str],
    titles: list[str] | None = None,
    cols: int = 4,
    output_path: str | None = None,
) -> str:
    """Tile multiple images into a grid for overview."""

def render_confusion_examples(
    worst_predictions: list[dict],  # from get_worst_predictions
    output_dir: str | None = None,
) -> list[str]:
    """Render the worst prediction cases for visual failure analysis."""
```

**Dependencies:** PIL/Pillow only (already in deps). No OpenCV or matplotlib needed.

**Design principles:**
- All functions return the output file path (for `view_image` consumption)
- Default output to `.tcip/artifacts/viz/{timestamp}_{function_name}.png`
- Normalized YOLO coordinates → pixel coordinates conversion built in
- Consistent color palette (reuse the 20-class COLOR_PALETTE from webview)
- Clean, legible renders: class labels with background boxes, confidence scores

### Phase 2: Vision MCP Tools

Add new tools in `packages/tcip-mcp/src/tcip_mcp/tools/vision_tools.py`:

```python
@mcp.tool()
@audited
def visualize_annotations(
    image_path: str,
    task: str = "detect",
) -> dict:
    """Render annotations on an image for visual inspection.
    
    Returns path to rendered image suitable for view_image analysis.
    """

@mcp.tool()
@audited  
def visualize_predictions(
    image_path: str,
    task: str = "detect",
) -> dict:
    """Render model predictions on an image."""

@mcp.tool()
@audited
def visualize_comparison(
    image_path: str,
    task: str = "detect",
    iou_threshold: float = 0.5,
) -> dict:
    """Render GT vs prediction comparison with match indicators."""

@mcp.tool()
@audited
def visualize_worst_predictions(
    predictions_dir: str,
    labels_dir: str,
    task: str = "detect",
    top_k: int = 10,
) -> dict:
    """Find and render the worst predictions for failure analysis.
    Returns grid image and individual case images."""

@mcp.tool()
@audited
def visualize_dataset_sample(
    folder_path: str,
    n: int = 16,
    task: str = "detect",
) -> dict:
    """Render a grid of sample annotated images from a dataset."""
```

**Return format:**
```python
{
    "image_path": "/abs/path/to/.tcip/artifacts/viz/...",
    "summary": "Rendered 5 detections (2 catkin, 3 nut) on IMG_0133.jpg",
    "details": { ... }
}
```

The agent then uses `view_image` on `image_path` to visually analyze the render.

### Phase 3: Agent Skill — Visual Analysis Workflows

Create `.github/skills/visual-analysis/SKILL.md` with workflows:

**Annotation QA Workflow:**
1. `load_dataset` → get image inventory
2. `visualize_dataset_sample` → grid of annotated samples → `view_image`
3. Agent describes what it sees, flags issues (missed objects, wrong classes, sloppy boxes)
4. For flagged images: `visualize_annotations` → `view_image` → detailed assessment

**Prediction Review Workflow:**
1. `run_inference` on dataset
2. `visualize_worst_predictions` → failure cases → `view_image`
3. Agent categorizes failures: false positives, false negatives, localization errors, class confusion
4. Agent recommends corrective actions (more training data, augmentation, model changes)

**Training Artifact Review:**
1. After training: `get_worst_predictions` → `visualize_worst_predictions`
2. Agent visually inspects failure modes
3. Cross-reference with metrics from `check_training_status`
4. Recommend next steps (more epochs, data augmentation, architecture change)

**Comparison Workflow:**
1. `visualize_comparison` for selected images → `view_image`
2. Agent assesses: IoU quality, missed detections, false positives
3. Provides actionable feedback

### Phase 4: Copilot Instructions Update

Update `.github/copilot-instructions.md` and `.github/agents/tcip.agent.md`:

```markdown
## Visual Analysis

The agent can visually inspect images using the `view_image` tool. 
Use visualization tools to render annotations/predictions on images, 
then view_image to analyze them.

Pattern:
1. Call a `visualize_*` MCP tool → returns image path
2. Call `view_image` on that path → model sees the rendered image
3. Describe findings and recommend actions
```

## File Layout After Implementation

```
packages/tcip-annotation/src/tcip_annotation/
    viz.py                    # NEW — rendering engine
    
packages/tcip-mcp/src/tcip_mcp/tools/
    vision_tools.py           # NEW — visualization MCP tools
    
.github/skills/visual-analysis/
    SKILL.md                  # NEW — visual analysis workflows
    
.tcip/artifacts/viz/          # NEW — rendered images (auto-created)
```

## Estimated Scope

| Phase | Files | Effort |
|-------|-------|--------|
| 1. Rendering Engine | 1 new file (~200 lines) | Core implementation |
| 2. Vision MCP Tools | 1 new file (~150 lines) | Tool wrappers |
| 3. Agent Skill | 1 new file (~80 lines) | Workflow documentation |
| 4. Instructions Update | 2 edits | Minor additions |
| **Total** | **3 new, 2 edited** | — |

## Key Design Decisions

1. **PIL only, no matplotlib** — keeps dependencies minimal, renders are simple box/polygon overlays
2. **Save to disk, use `view_image`** — most reliable path for model vision in VS Code
3. **Normalized → pixel conversion built into viz.py** — tools don't need to worry about coordinate systems
4. **Consistent color palette** — same 20 colors as webview annotation canvas
5. **Artifact directory** — `.tcip/artifacts/viz/` creates audit trail of what agent saw
6. **No base64 in MCP returns by default** — avoids bloating tool results; agent uses `view_image` instead

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Model can't see images (model doesn't support vision) | `view_image` is the official VS Code mechanism; falls back to text summary |
| Large images exceed context window | Resize renders to max 1024px longest edge |
| Many visualization calls slow workflow | Grid rendering batches multiple images into one `view_image` call |
| PIL drawing quality insufficient | PIL.ImageDraw handles boxes/polygons/text well for QA purposes |

## Not In Scope (Future)

- Real-time annotation corrections (agent editing labels — that's the annotation workflow)
- Video frame analysis
- 3D point cloud visualization
- Automated annotation from visual inspection (SAM already covers assisted labeling)
