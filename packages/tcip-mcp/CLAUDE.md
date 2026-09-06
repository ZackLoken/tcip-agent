# packages/tcip-mcp

MCP server package (`python -m tcip_mcp`). Loads on top of the root `CLAUDE.md`: invariants,
operating posture, and pipeline/model rules there apply here and aren't restated.

## Layout

```
src/tcip_mcp/
  server.py, __main__.py   # MCP entry point; registers all tool modules
  knowledge/      # the canonical domain-knowledge directory (the domain documents plus
                  # crops/<crop>.md, crops/crops.yml), read through __init__.py; source
                  # for the generated Claude Code, Codex and Antigravity skills, AGENTS.md's
                  # generated block, and the serve_domain_knowledge tool
  tools/          # domain tools, one module per area: annotation, data, experiment, feedback,
                  # gui, inference, ingest, knowledge, meta, model, operationalization,
                  # orthomosaic, phenology, project, proposal, training, vision
  pipelines/      # composable ML: active_learning, components, data, feedback, inference,
                  # measurement, postprocessing, training (submodules), plus:
    derivations.py        # Tier-A derivations: compute a parameter (channels, num_classes,
                           # anchor ratios) from the artifact in hand instead of pinning it
    pixel_size.py          # the one raster-georeferencing-to-metres-per-pixel resolver, shared
                            # by the completeness bar and the block-scale derivation
    model_build.py          # build_model: the one seam from a model_source config to an nn.Module
    model_contract.py        # the measurement boundary a bespoke model must pass
                              # (check_model_contract, overfit_check)
    proposal.py               # auto-labeling engine seam: built-in SAM, or a bespoke Proposer
    resolution.py              # ResolvedParam: the derive-don't-pin currency and the
                                # unvalidated-value firewall
    operating_point.py          # five resolvers, resolve_operating_point (the calibrated
                                 # conf/NMS/max_dets/tile path) among them, plus resolve_classifier_operating_point, _resolve_scalar_operating_point, resolve_ordinal_operating_point and resolve_regression_operating_point; resolution.py's raw and block-calibrated-export paths are the other two regimes, the latter carrying conf and cross_tile_nms straight from the calibrated bundle and sharing resolve_tile_size_param with the others for tile scale
    schemas.py, image_utils.py
  dataset_layout.py      # the single path resolver on the backend: where an image's
                          # labels/predictions live on disk. The frontend cannot import it, so paths.ts's RECORD_EXT and classes.ts's ImageStatus union each restate a fact of it, held equal by tests/test_frontend_dataset_vocabulary.py
  class_registry.py      # classes.json: subjects, attributes, the name<->id assignment
  traits.py               # the trait registry: human-defined measurement semantics per trait
  operationalization.py    # per-project records of what a trait's delivered number means, who
                            # confirmed it, and the precondition every delivery door checks
  prediction_buckets.py    # prediction-bucket immutability: never overwrite predictions a human reviewed
  project_paths.py, workspace.py     # platform-state-root and workspace-root resolvers
  experiments.py, model_registry.py   # experiment tracking (.tcip/experiments/) and the trained-model registry
  identity.py             # the user:<name> identity convention, spelled once
  agent_identity.py       # the harness the MCP handshake declared, this run's minted session, and what
                            # the harness exported about itself, stamped on every audit line, statement
                            # record and HTTP push; declarations, never verified
  audit.py, project_status.py, web_client.py
```

Every MCP tool in `tools/` is decorated `@mcp.tool()` + `@audited`, except `serve_domain_knowledge`,
whose `@mcp.tool(description=...)` composes its client-visible description from the knowledge
corpus at import time rather than leaving it as the bare docstring. A door demoted from tool
status (run only through its own `tcip` subcommand) keeps `@audited` without registering.
Run `python tools/list_tools.py` for the current tool list/count; never hardcode a count in a
doc or comment.

## Conventions specific to this package

- `tools/` lazy-imports torch/torchvision and the `pipelines/` modules that carry them, inside
  function bodies. `pipelines/` itself imports torch at module level; a pipeline module only
  loads when a tool reaches into it.
- Detectors are built via the plain `build_detector` (+ `_build_faster_rcnn` / `_build_fcos` /
  `_build_retinanet` / `_build_mask_rcnn`); bespoke model code imports these directly. There is no
  model spec or component registry; see the `toolkit-inventory` skill for the full composition
  surface (`build_detector`/`build_loss` task strings, heads/necks/backbones, derivations, the `ctx`
  craft library, and the `model_source`/`training_source`/`dataset_source` seams).
- A one-off script the agent writes for one project lives with that project, in the project's own
  directory, never in this repository. A standing operator capability is a console-command door in
  `cli/`. Add a tool only for an audit seam, long-running infrastructure, or domain knowledge the
  agent lacks that a console command can't carry.
- State mutations route through `@audited` doors only: the MCP tools and the console-command doors
  demoted from them; the record is `audit_log`, one store addressed by `audit.audit_log_key` under
  three kinds of root (the platform's own, a dataset's own, a project's own), held by whichever
  backend the process bound, that other code (including scripts) must not write around. `audit.py`
  decides where an entry goes and what a failed append means: the decorator raises
  `MutationCommittedWithoutAuditLine`; a caller that is neither an MCP tool nor a demoted door
  emits through `record_event` (best-effort) or `record_event_or_raise` (raises
  `AuditEntryNotWritten` on a failed append) rather than composing an entry of its own.
