# packages/tcip-mcp

MCP server package (`python -m tcip_mcp`). Loads on top of the root `CLAUDE.md` and the operating
contract it imports, which apply here and aren't restated.

## Layout

```text
src/tcip_mcp/
  server.py, __main__.py   # MCP entry point; registers all tool modules
  knowledge/      # the canonical domain-knowledge directory (the domain documents plus
                  # crops/<crop>.md, crops/crops.yml), read through __init__.py, served by
                  # serve_domain_knowledge and written for each harness by harness_files.py
  cli/            # the operator-only command modules, and run_tool, the one adapter running any
                  # registered tool as the tcip command of its own name
  tools/          # domain tools, one module per area: annotation, calibration (the assessment
                  # doors), data, delivery, doctor, experiment, feedback, gui, inference, ingest, knowledge,
                  # meta, model, orthomosaic, phenology, project, proposal, trait, training, vision
  pipelines/      # composable ML: active_learning, components, data, inference, measurement,
                  # postprocessing, training (submodules), plus:
    derivations.py        # Tier-A derivations: compute a parameter (channels, num_classes,
                           # anchor ratios) from the artifact in hand instead of pinning it
    pixel_size.py          # the one raster-georeferencing-to-meters-per-pixel resolver, read
                            # by the block-scale derivation
    model_build.py          # build_from_model_source: the one seam from a validated
                             # model_source to an nn.Module
    model_contract.py        # the measurement boundary a bespoke model must pass
                              # (check_model_contract, overfit_check)
    proposal.py               # auto-labeling engine seam: a registered or dotted Proposer
    execution.py               # one pass's execution record (conf, density, tile geometry, merge)
                                # and prepare, which readies a Preparation whose runnable
                                # builds the one Pass from a complete record
    operating_point.py          # the criteria an assessment judges against: count, classifier,
                                 # scalar, and the spatial held-out check
    slicing.py             # tiled inference over sahi: tile geometry, slice lattice, a checkpoint
                           # as a SAHI detection model, and the cross-tile merge
    proposal_engines.py    # the proposal engines that ship registered
    raster_source.py       # the open-and-read surface over every image source the platform decodes
    band_stats.py          # display band statistics, the 8-bit band stretch, the RGB composite
    overviews.py           # external .ovr overview pyramids for large rasters
    display_bounds.py      # pixel bounds for what is served to a screen or written for the agent
    reference_grid.py      # the named reference grid over a raster's native pixel frame
    block_calibration.py   # the reserved-region reference of a within-image split
    delivery_events_schema.py  # the declared shapes of a delivery event and its acknowledgment
    schemas.py, image_utils.py
  dataset_layout.py      # the one resolver: an image's place, its label document's key, a bucket's
                          # keys, the one label save and the one writer of stored label records;
                          # the frontend names images only
  subject_registry.py    # subjects.json: subjects and the attributes each declares, in declared order
  traits.py               # one record per trait: its entry (spec fields and what each delivered
                            # number means) as appended revisions the breeder confirms
  operationalization.py    # the check every delivery door runs on the latest confirmed revision
  buckets.py               # a published bucket: its record and its documents, one commit, once
  assessment.py            # the assessment a delivered number rests on, written once
  assessment_evidence.py   # the evidence records an assessment holds and its failure names
  delivery.py              # the one delivery gate and the delivery event
  project_record.py       # a project's own record: its id, display name, site and invariances
  workspace.py            # the workspace root and its project directories
  project_paths.py        # paths under one project's .tcip/ (state, viz) and the repo root
  registry_paths.py       # how a record stores a path and where a stored path resolves
  experiments.py, model_registry.py   # runs and sweeps with their trial runs, side by side in
                                        # .tcip/experiments/; the registry: completed runs plus foreign ones
  identity.py             # the user:<name> identity convention, spelled once
  agent_identity.py       # the client the MCP handshake declared and this run's minted session,
                            # stamped on every audit line and HTTP push; declarations, never verified
  audit.py, web_client.py
```

Every MCP tool is declared by `server.tool` in its area's `tools/` module: decorated `@tool()`, or,
for a function another module defines, `tool()(fn)` (`compare_experiments`, `available_search_algs`,
`available_schedulers`, `scorer_catalog`); a tool that acts on a project takes the project the
server was started for as its first parameter: `project`, its directory, or `bound`, the whole
`workspace.BoundProject` (directory, record id, workspace) when it also names the project to the
backend. The server supplies only those two and `actor` (`server._CONTEXT`), the person the server's
calls are made by, each read from the server's `server.Session` at the call (`Session.bind` is the
one way a session gains them; `None` is an MCP client's person), so any other parameter, whatever
its name, stays the client's (the knowledge door acts on none, the project-creation door on `actor`
alone), and a call naming an argument outside the tool's listed schema, a supplied one included,
is refused before the tool runs; a tool that changes state is `@audited`, by that `actor`, unless
the library function it calls records its own event with facts the decorator cannot carry (a
digest, what was written), and then it is not decorated. The decorator writes no line for a call
returning its error dict and an exception line for a call that raises. A read-only tool (a status
poll, a listing, a document served back) leaves none, so the audit log records mutations and
nothing else; a render under
`.tcip/artifacts/viz/` is a derived view that can be regenerated, not a state change, and leaves
none. `serve_domain_knowledge`'s `@tool(description=...)` composes its client-visible description
from the knowledge corpus at import time rather than leaving it as the bare docstring. A mutating
door only an operator runs (an operator-only command module in `cli/` calls it, as
`archive_project`) keeps `@audited` without registering.

## Conventions specific to this package

- `pipelines/` imports torch at module level, so `tools/` lazy-imports the `pipelines/` modules
  too; a pipeline module loads only when a tool reaches into it.
- The platform builds no model: a detection model is the agent's own module over a library
  detector, held to the model contract (`model_contract.check_model_contract`). There is no model
  spec or component registry; see the `toolkit-inventory` skill for the full composition surface
  (the model contract, `build_loss` names and task strings, heads/necks/backbones, derivations,
  the `ctx` craft library, and the `model_source`/`training_source`/`dataset_source` seams).
- A standing capability lives in `tools/`, and every registered tool is also the `tcip` command of
  its own name (`cli.hyphenated`): the one adapter `cli.run_tool` parses its arguments into the
  tool's advertised input schema and calls it through the server `server.server_for` returns, its
  session bound to the parsed `--project` and `--user`. An operator-only command is a module in
  `cli/` exposing `main(argv)`, named in the `tcip_web.cli` dispatcher.
- State mutations route through audited doors only, each leaving one line: an `@audited` tool or
  operator-only door, or a library function recording its own event through
  `record_event_or_raise`; the record is `audit_log`, one log addressed by `audit.audit_log_key`
  under two kinds of root (a dataset's own, a project's own), held in that root's store database,
  that other code (including scripts) must not write around. `audit.py` decides where an entry goes
  and what a failed append means: the decorator raises `MutationCommittedWithoutAuditLineError`. A
  library write holding its records and its line in one transaction appends the line inside it
  as `audit_line(...).stored()`, so neither lands without the other; a door whose line follows a committed
  write emits through `record_event_or_raise`, which raises `AuditEntryNotWrittenError` on a failed
  append. A line written before its change (the training envelope's opening `training_run` line)
  is refused before anything is committed, and its caller says so rather than repeating the
  error's committed reading.
