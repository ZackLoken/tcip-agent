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
  tools/          # domain tools, one module per area: annotation, calibration (the assessment
                  # doors), data, experiment, feedback, gui, inference, ingest, knowledge, meta,
                  # model, orthomosaic, phenology, project, proposal, trait, training, vision
  pipelines/      # composable ML: active_learning, components, data, feedback, inference,
                  # measurement, postprocessing, training (submodules), plus:
    derivations.py        # Tier-A derivations: compute a parameter (channels, num_classes,
                           # anchor ratios) from the artifact in hand instead of pinning it
    pixel_size.py          # the one raster-georeferencing-to-meters-per-pixel resolver, shared
                            # by the completeness bar and the block-scale derivation
    model_build.py          # build_model: the one seam from a model_source config to an nn.Module
    model_contract.py        # the measurement boundary a bespoke model must pass
                              # (check_model_contract, overfit_check)
    proposal.py               # auto-labeling engine seam: a registered or dotted Proposer
    execution.py               # one pass's execution record (conf, cap, tile geometry, merge)
                                # and prepare_pass, the one place a pass is built
    operating_point.py          # the criteria an assessment judges against: count, classifier,
                                 # scalar, and the spatial held-out check
    schemas.py, image_utils.py
  dataset_layout.py      # the single path resolver on the backend: where an image's
                          # labels/predictions live on disk, and the one label save. The frontend cannot import it: the label suffix and the completion-state vocabulary reach it through the generated types
  subject_registry.py    # subjects.json: subjects and the attributes each declares, in declared order
  traits.py               # one record per trait: its entry (spec fields and what each delivered
                            # number means) as appended revisions the breeder confirms
  operationalization.py    # the check every delivery door runs on the latest confirmed revision
  buckets.py               # a published bucket and its bucket.json, written once
  assessment.py            # the assessment a delivered number rests on, written once
  delivery.py              # the one delivery gate and the delivery event
  project_record.py       # a project's own record: its id, display name and site
  workspace.py            # the workspace root, its project directories and last-opened pointer
  project_paths.py        # paths under one project's .tcip/ (state, viz) and the repo root
  experiments.py, model_registry.py   # run and sweep directories (.tcip/experiments/, .tcip/hpo/), each
                                        # written once as it goes; the registry is completed runs plus
                                        # the foreign checkpoints registered beside them
  identity.py             # the user:<name> identity convention, spelled once
  agent_identity.py       # the client the MCP handshake declared and this run's minted session,
                            # stamped on every audit line and HTTP push; declarations, never verified
  audit.py, project_status.py, web_client.py
```

Every MCP tool in `tools/` is decorated `@tool()` (`server.tool`); a tool that acts on a project
takes the project the server was started for as its first parameter (the knowledge door and the
project-creation door act on none), and every mutating door leaves exactly one
audit line per act, the decorator's or the library's: a tool that changes state is `@audited`
unless the library function it calls records its own event with facts the decorator cannot carry
(a digest, what was written), and then it is not decorated. The decorator writes no line for a
call returning its error dict and an exception line for a call that raises. A read-only tool (a status poll, a listing, a
document served back) leaves none, so the audit log records mutations and nothing else. `serve_domain_knowledge`'s
`@tool(description=...)` composes its client-visible description from the knowledge corpus
at import time rather than leaving it as the bare docstring. A mutating door demoted from tool
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
- State mutations route through audited doors only, each leaving one line: an `@audited` tool or
  console-command door, or a library function recording its own event through
  `record_event_or_raise`; the record is `audit_log`, one store addressed by `audit.audit_log_key` under
  two kinds of root (a dataset's own, a project's own), held by whichever
  backend the process bound, that other code (including scripts) must not write around. `audit.py`
  decides where an entry goes and what a failed append means: the decorator raises
  `MutationCommittedWithoutAuditLine`; a caller that is neither an MCP tool nor a demoted door
  emits through `record_event` (best-effort) or `record_event_or_raise` (raises
  `AuditEntryNotWritten` on a failed append) rather than composing an entry of its own.
