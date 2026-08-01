# packages/tcip-mcp

MCP server package (`python -m tcip_mcp`). Loads on top of the root `CLAUDE.md`: invariants,
operating posture, and pipeline/model rules there apply here and aren't restated.

## Layout

```
src/tcip_mcp/
  tools/        # domain tools, one module per area: annotation, data, experiment, feedback,
                # inference, ingest, meta, model, phenology, project, training, vision
  pipelines/    # composable ML: active_learning, components, data, feedback, inference,
                # measurement, postprocessing, training
  utils/
```

Every tool in `tools/` is decorated `@mcp.tool()` + `@audited`. Run `python scripts/list_tools.py`
for the current tool list/count; never hardcode a count in a doc or comment, since it drifts.

## Conventions specific to this package

- **Lazy-import** torch/torchvision inside function bodies: the MCP server must start fast, and
  most tool calls never touch the model layer.
- Detectors are built via the plain `build_detector` (+ `_build_faster_rcnn` / `_build_fcos` /
  `_build_retinanet` / `_build_mask_rcnn`); bespoke model code imports these directly. There is no
  model spec or component registry; see the `toolkit-inventory` skill for the full composition
  surface (`build_detector`/`build_loss` task strings, heads/necks/backbones, derivations, the `ctx`
  craft library, and the `model_source`/`training_source`/`dataset_source` seams).
- **Prefer a logged script in `scripts/` over a new tool here.** This package already has tool
  bloat, not tool shortage; add a tool only for an audit seam, long-running infrastructure, or
  domain knowledge the agent lacks that a script can't carry.
- State mutations route through `@audited` tools only; `.tcip/audit.jsonl` is the append-only
  record other code (including scripts) must not write around.
