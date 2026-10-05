# TCIP architecture and API

This document maps the public surface an adopter may rely on, the on-disk formats, and the
cross-layer seam inventory. Each package's `CLAUDE.md` states where its modules live. A citation
names a file and the symbol or fragment it means, never a line number; treat any sentence that
disagrees with the code as a defect in this document and correct the document.

Sections:

1. Public surface
2. On-disk formats
3. Seam inventory


## Public surface

## 1. MCP tools

`packages/tcip-mcp/src/tcip_mcp/server.py` builds the server (`build_server`) for the project it
was started for. `python tools/list_tools.py` is the authority for the registered tool list and
count; a count written here drifts, so none is. Every registered tool's name matches a `@tool()`
decorator site in `packages/tcip-mcp/src/tcip_mcp/tools/*.py`. A mutating tool's site is
followed by an `@audited` decorator (some spelled `@audited(scope_arg=...)`) unless its library
records each of its acts itself, and a tool that only reads carries none. The "audited" column
says which: `yes` for the decorator, `library` for a line its library operation writes, `no` for
a read.

Tables below group by defining module. Docstring is the function's docstring first line,
verbatim.

### annotation_tools.py

| tool | audited | docstring first line |
|---|---|---|
| `save_annotations` | library | Write an image's annotations to its single per-image label document (all subjects, one document). |
| `write_subject_registry` | library | Author the dataset's nested subject registry, a thin wrapper over ``subject_registry``. |

### data_tools.py

| tool | audited | docstring first line |
|---|---|---|
| `freeze_selection` | yes | Freeze a finished run's own drawn train/val partition into a selection, so a later run can |
| `draw_splits` | yes | Compute a leakage-free, annotation-stratified train/val/calibration split. |

### experiment_tools.py

| tool | audited | docstring first line |
|---|---|---|
| `get_experiment` | no | Read one run's directory. |
| `list_experiments` | no | Every run and sweep of the project (`experiments.training_listing`): |

### feedback_tools.py

| tool | audited | docstring first line |
|---|---|---|
| `prioritize_review_queue` | no | Rank unfinished images by active-learning informativeness for the next review batch. |

### gui_tools.py

| tool | audited | docstring first line |
|---|---|---|
| `push_panel_event` | no | Push structured data to a TCIP GUI panel via the tcip-web backend. |
| `focus_human_attention` | no | Drive the live Annotate tab to a (subject, date) frame in the right mode, showing the |

### inference_tools.py

| tool | audited | docstring first line |
|---|---|---|
| `run_inference` | no, its library event | Run a trained model over images or a raster, and publish the predictions as a bucket. |
| `deliver_per_image_counts` | no, its library event | Deliver a published bucket's per-image detection counts as a CSV. |

### calibration_tools.py

| tool | audited | docstring first line |
|---|---|---|
| `assess_checkpoint` | no, its library event | Assess a checkpoint against the calibration and holdout sides of a selection, for one |
| `assess_reserved_regions` | no, its library event | Assess a checkpoint trained on one mosaic against that mosaic's own reserved calibration and |
| `calibrate_physical_scale` | no, its library event | Derive a per-pixel physical scale on a selection's calibration side and check it against |

### ingest_tools.py

| tool | audited | docstring first line |
|---|---|---|
| `ingest_images` | yes | Copy raw images into a structured project, bucketed by the capture date each file states. |
| `import_coco` | no, its library event | Convert an external dataset-level COCO document into the dataset's per-image label documents. |

### meta_tools.py

| tool | audited | docstring first line |
|---|---|---|
| `report_friction` | yes | Log structured friction when you get stuck, confused, or surprised. |
| `load_project_memory` | yes | Read one project-memory corpus into context so context isn't lost between sessions. |
| `read_audit_log` | yes | Read one audit log's own entries: which door touched a dataset or project, when, with |
| `write_retrospective` | yes | Write an end-of-project retrospective to markdown. |
| `record_distillation_pass` | yes | Record that this project's friction reports and retrospectives were reviewed, resetting |

### knowledge_tools.py

| tool | audited | docstring first line |
|---|---|---|
| `serve_domain_knowledge` | yes | Read the platform's domain knowledge: trait semantics, workflow patterns, and per-crop |

### model_tools.py

| tool | audited | docstring first line |
|---|---|---|
| `register_model` | no | Register a trained model in the project model registry. |
| `rank_registered_models` | yes | List the project's registered models, or rank them by an explicit metric. |

### trait_tools.py

| tool | audited | docstring first line |
|---|---|---|
| `propose_trait` | no, its library event | Propose a trait's complete entry, its spec fields and what its delivered number means per |

### orthomosaic_tools.py

| tool | audited | docstring first line |
|---|---|---|
| `deliver_orthomosaic_plant_counts` | no, its library event | Per-plant detection counts from a persisted orthomosaic prediction bucket plus plant CSV(s). |

### delivery_tools.py

| tool | audited | docstring first line |
|---|---|---|
| `deliver_per_plant_csv` | no, its library event | The general per-plant CSV door: ``aggregate_per_plant``'s own output delivered over the |

### phenology_tools.py

| tool | audited | docstring first line |
|---|---|---|
| `register_plant_registry` | yes | Register a plant-locations CSV set under a name, so `build_plant_mapping` and |
| `build_plant_mapping` | no, its library event | Assign each geolocated image to a plant, then persist the mapping under this project. |
| `deliver_phenology_milestones` | no, its library event | Per-plant phenology milestones from prediction buckets whose scope declares the positive state's attribute, and a plant mapping. |

### project_tools.py

| tool | audited | docstring first line |
|---|---|---|
| `register_dataset` | yes | Record a dataset's identity so a delivered number can be traced to the exact data behind it. |
| `initialize_project` | yes | Create a TCIP project: ``.tcip/`` with its artifacts and models directories, and its record |
| `view_gui_state` | yes | The live GUI session the human is looking at in this project: dataset, date, trait, tab, |
| `inspect_project` | yes | Get an overview of the project. |

`archive_project` bundles every file under the project except the store's bookkeeping
(`tcip_store.file_backend.is_bookkeeping`: locks, temp files, the database's WAL sidecars), each
`.tcip/store.db` as the consistent copy `tcip_store.sqlite_backend.copy_database` takes, and
without the checkpoints `model_registry.checkpoint_files` names unless `include_models`.
`import_project` stages the bundle, refuses any bookkeeping member or one escaping the staging
directory, and moves the staged tree onto an empty destination.

### proposal_tools.py

| tool | audited | docstring first line |
|---|---|---|
| `propose_annotations` | yes | Propose candidate annotations on an image for review, using a chosen auto-labeling engine. |
| `stage_proposals` | yes | Stage model-/agent-proposed shapes as predictions for canvas review, the "show on canvas |

### training_tools.py

| tool | audited | docstring first line |
|---|---|---|
| `launch_training` | library | Launch a training run in an isolated subprocess from a bespoke ``model_source`` builder. |
| `monitor_training` | yes | Check a training run, a sweep's trial included, or a hyperparameter sweep, by the id that |
| `cancel_training` | yes | Request graceful cancellation of a running training run, a sweep's trial or a whole |
| `run_hyperparameter_search` | library | Run hyperparameter optimization on Ray Tune, training each trial for real. |
| `evaluate_model` | no | Evaluate a trained checkpoint on a (held-out) dataset and return the result. |

### vision_tools.py

| tool | audited | docstring first line |
|---|---|---|
| `capture_live_canvas` | no | Render exactly what the human's GUI canvas shows right now: image, shapes, viewport. |

## 2. HTTP routes and WebSocket endpoints

`packages/tcip-web/src/tcip_web/app.py` builds the FastAPI app, registers its own routes and
sockets directly, then calls `register_all(app)` from
`packages/tcip-web/src/tcip_web/routes/__init__.py`, which `include_router`s each route module
under `routes/` with a fixed prefix. `routes/__init__.py` and `routes/_body_common.py` define no
routes of their own; `_body_common.py` holds `EmptyBodyPayload`, the empty body model a
path-parameter-only route declares so the browser must send a preflighted request rather than
reaching the handler as a simple one.

WebSocket routes: `/ws/state` and `/ws/panel/{panel}` on `app.py`;
`/api/terminal/ws/{session_id}` on `routes/terminal.py`; `/api/inference/jobs/{job_id}/stream`
on `routes/inference.py`; `/api/training/runs/{experiment_id}/stream` on `routes/training.py`.

Tables below group by defining module. Every method, path, and handler name below is the one
registered at HEAD.

### app.py (not under a router prefix)

| method | path | handler |
|---|---|---|
| GET | `/api/state` | `get_state` |
| POST | `/api/state/tab` | `set_active_tab` |
| WS | `/ws/state` | `state_ws` |
| GET | `/health` | `health` |
| GET | `/` | `index` |
| POST | `/api/events/{panel}` | `post_panel_event` |
| WS | `/ws/panel/{panel}` | `panel_ws` |

### routes/annotate.py, prefix `/api/annotate`

| method | path | handler |
|---|---|---|
| GET | `/labels` | `load_labels` |
| POST | `/labels` | `save_labels` |
| GET | `/proposals` | `load_proposals` |
| POST | `/queue/launch` | `launch_priority_queue` |
| GET | `/queue/{job_id}` | `get_priority_queue_job` |

### routes/canvas.py, prefix `/api/canvas`

| method | path | handler |
|---|---|---|
| POST | `/state` | `push_canvas_state` |

### routes/dataset.py, prefix `/api/dataset`

| method | path | handler |
|---|---|---|
| GET | `/tree` | `get_dataset_tree` |
| POST | `/select` | `select_dataset` |
| POST | `/nav` | `set_current_image` |

### routes/fs.py, prefix `/api/fs`

| method | path | handler |
|---|---|---|
| GET | `/list` | `list_dir` |

### routes/images.py, prefix `/api/images`

| method | path | handler |
|---|---|---|
| GET | `/serving_grid` | `get_serving_grid` |
| GET | `` (root) | `serve_image` |
| GET | `/bands` | `get_bands` |
| POST | `/overviews` | `build_image_overviews` |
| GET | `/overviews/status` | `get_overview_job` |

### routes/inference.py, prefix `/api/inference`

| method | path | handler |
|---|---|---|
| POST | `/launch` | `launch_inference` |
| GET | `/jobs` | `list_jobs` |
| POST | `/jobs/{job_id}/cancel` | `cancel_job` |
| WS | `/jobs/{job_id}/stream` | `stream_job` |

### routes/meta.py, prefix `/api/meta`

| method | path | handler |
|---|---|---|
| GET | `/reports` | `get_reports` |
| GET | `/retrospectives` | `get_retrospectives` |

### routes/projects.py, prefix `/api/projects`

| method | path | handler |
|---|---|---|
| GET | `` (root) | `list_projects` |
| POST | `/open` | `open_project` |
| POST | `/remove` | `remove_project` |
| POST | `/rename` | `rename_project_route` |

### routes/results.py, prefix `/api/results`

| method | path | handler |
|---|---|---|
| POST | `/plant_mapping/build` | `build_plant_mapping` |
| POST | `/plant_mapping/load` | `load_plant_mapping` |
| GET | `/plant_mapping/list` | `list_plant_mappings` |
| POST | `/phenology_measurement` | `phenology_measurement` |
| POST | `/export_csv` | `export_csv` |
| POST | `/export_count_csv` | `export_count_csv` |
| GET | `/traits` | `list_traits` |
| POST | `/traits/confirm` | `confirm_trait_revision` |
| GET | `/delivery-events` | `list_delivery_events` |
| GET | `/models/registered` | `registered_models` |

### routes/sessions.py, prefix `/api/sessions`

| method | path | handler |
|---|---|---|
| POST | `/image_event` | `image_event` |
| POST | `/end` | `end_session` |
| GET | `/load` | `load_sessions` |

### routes/subjects.py, prefix `/api/subjects`

| method | path | handler |
|---|---|---|
| GET | `/load` | `load_subjects` |
| POST | `/save` | `save_subjects` |

### routes/terminal.py, prefix `/api/terminal`

| method | path | handler |
|---|---|---|
| GET | `/status` | `get_status` |
| POST | `/sessions` | `create_session` |
| POST | `/sessions/{session_id}/restart` | `restart_session` |
| POST | `/sessions/{session_id}/submit` | `submit_to_session` |
| WS | `/ws/{session_id}` (full path `/api/terminal/ws/{session_id}`) | `terminal_ws` |

### routes/training.py, prefix `/api/training`

| method | path | handler |
|---|---|---|
| GET | `/configs/{experiment_id}/splits` | `list_split_choices_route` |
| POST | `/runs` | `relaunch_route` |
| GET | `/runs` | `list_runs_route` |
| GET | `/runs/{experiment_id}` | `get_run` |
| POST | `/runs/{experiment_id}/tensorboard` | `launch_run_tensorboard` |
| POST | `/runs/{experiment_id}/cancel` | `cancel_run_route` |
| POST | `/compare` | `compare_runs_route` |
| POST | `/compare/best` | `compare_best_route` |
| GET | `/metric-directions` | `metric_directions_route` |
| WS | `/runs/{experiment_id}/stream` (full path `/api/training/runs/{experiment_id}/stream`) | `training_stream_ws` |

### Routes with no frontend caller

These registered backend routes have no caller under `packages/tcip-web/frontend/src/`.

| method | path | note |
|---|---|---|
| GET | `/health` | not fetched from `frontend/src`; a liveness endpoint |
| GET | `/` | loaded by the browser's own navigation, not via fetch/XHR from app code |
| POST | `/api/events/{panel}` | posted by MCP tools (`tcip_mcp.web_client`), not by the browser |
| GET | `/api/state` | no caller found |

## 3. tcip-annotation importable public symbols

`packages/tcip-annotation/src/tcip_annotation/__init__.py` defines `__all__`: 9 entries.

| name | re-exported from |
|---|---|
| `Annotation` | `state` |
| `BBox` | `state` |
| `Point` | `state` |
| `Polygon` | `state` |
| `bbox_of` | `state` |
| `parse_coco_annotations` | `format_io` |
| `point_in_polygon` | `matching` |
| `mask_to_polygon_rings` | `mask_contours` |
| `cell_fields` | `grid` |

No `import tcip_mcp` / `import tcip_web` statement exists anywhere in
`packages/tcip-annotation/src/tcip_annotation/`; `packages/tcip-annotation/CLAUDE.md`
states the same rule.

Names not re-exported in `__all__` but importable directly from their defining submodule:
`json_io.ANNOTATIONS_KEY`,
`json_io.UNASSESSED`, `json_io.attribute_ids`, `json_io.attribute_values`,
`json_io.LabelDocument`, `json_io.read_label_document`, `json_io.write_label_document`,
`format_io.coco_categories`, `mask_contours.DEFAULT_EPSILON_FRAC`,
`verdicts.Verdict`, `verdicts.verdict_key`, `verdicts.read_verdicts`,
`matching.pair_detections`, `matching.pair_proposals`, `grid.grid_to_rect`,
`grid.column_label`, `grid.column_index`, `utils.auto_orient_image`,
`utils.get_image_dimensions`, and every public name in `viz.py`
(`COLOR_PALETTE`, `render_detections`, `render_segmentations`, `render_comparison`,
`render_grid`, `render_candidates`, `render_grid_overlay`, `render_canvas_state`).

## 4. Entry points

`python -m tcip_mcp`: `packages/tcip-mcp/src/tcip_mcp/__main__.py` imports `main`
from `tcip_mcp.server` and calls it: `packages/tcip-mcp/src/tcip_mcp/server.py`
(`def main(argv: list[str] | None = None) -> None:`), which takes `--project <path>`, the one
project the server acts on, and runs the server `build_server` (`server.py`) builds for it:
every function a `@tool()` decorator (`server.py`) in
`packages/tcip-mcp/src/tcip_mcp/tools/*.py` declared, each bound to that project
(`python tools/list_tools.py` lists them; the count is never written down, since it drifts).

`python -m tcip_web`: `packages/tcip-web/src/tcip_web/__main__.py` defines `main()`, which
reads the workspace once
(`tcip_mcp.workspace.workspace_from_environment`, `workspace.py`, refusing an unset
`TCIP_WORKSPACE`), reads `TCIP_WEB_PORT` (default `8765`) and binds the loopback address
`LOOPBACK_HOST` (`packages/tcip-mcp/src/tcip_mcp/web_client.py`, `"127.0.0.1"`, the one host the
MCP tools reach too), writes the bound port under that
workspace (`replace(backend_port_key(workspace), str(port))`), configures the app's `StateStore`
with the same workspace (`store.configure(workspace, image_roots_from_environment())`), and
serves the app via `uvicorn.run(app, host=LOOPBACK_HOST, port=port)`.
The app's lifespan (`app.py`) opens the project the workspace's last-opened pointer names
(`open_last_opened`, `packages/tcip-web/src/tcip_web/routes/projects.py`); with no pointer
the backend starts with no project open and every project-scoped route answers 409 until one is
opened from the picker. Every process reads the workspace once at its entry and hands it on as a
value; a test configures its own `StateStore`. Locality is a property
of the accepted connection rather than the bind host: every request is checked by
`tcip_web.trust_boundary.TrustBoundaryMiddleware` (`trust_boundary.py`), which refuses an
arrival that is not through this machine (`local_arrival`, `trust_boundary.py`). The same
middleware applies one Origin policy (`origin_allowed`, `trust_boundary.py`) to every
WebSocket scope and to every HTTP scope whose method is state-changing
(`STATE_CHANGING_METHODS`, `trust_boundary.py`), rather than leaving each handler to call it
for itself.

`.mcp.json` (repo root): declares one MCP server, `tcip`, which launches
`conda run -n tcip-agent --no-capture-output python -m tcip_mcp`. A semantic code-search server
(claude-context, backed by an embedding model and a vector store) is optional developer tooling
some maintainers configure locally; it is not part of the platform and is not declared in this
tracked file, so its presence or configuration varies per machine.

`tools/` (repo root): a non-API surface, not imported by `tcip_mcp`, `tcip_web`, or
`tcip_annotation` package code; each file is a standalone CI/development script invoked
directly (`python tools/<name>.py`), and `tools/README.md` says what each does. An
operator command instead lives as a `cli/` module inside the package whose imports it needs,
registered under a name in `tcip_web.cli.COMMANDS` and run as `tcip <name>` (or, without the
console script installed, `python -m tcip_web.cli <name>`).


## On-disk formats

Every writer and reader below is cited by its file and symbol.

## 1. Annotation document, canonical per-image label record

Key: a record in the dataset root's database, `label_documents` keyed `(<capture>, <stem>)` for
ground truth (`tcip_mcp.dataset_layout.label_key`,
`packages/tcip-mcp/src/tcip_mcp/dataset_layout.py`) and `prediction_documents` keyed
`(<bucket>, <stem>)` for predictions, identical schema (`dataset_layout.prediction_key`,
`packages/tcip-mcp/src/tcip_mcp/dataset_layout.py`). A person reads them as files through
`tcip dump-store`.

Writers: `tcip_annotation.json_io.write_label_document`,
`packages/tcip-annotation/src/tcip_annotation/json_io.py`;
`tcip_mcp.pipelines.data.coco_import.import_coco_document`,
`packages/tcip-mcp/src/tcip_mcp/pipelines/data/coco_import.py`, every document of one
import in one commit;
`tcip_mcp.dataset_layout.save_label_document`,
`packages/tcip-mcp/src/tcip_mcp/dataset_layout.py`, the one label save every door calls, its
document, verdicts and audit line in one commit;
`tcip_mcp.buckets.publish`, `packages/tcip-mcp/src/tcip_mcp/buckets.py`;
`tcip_mcp.tools.proposal_tools._stage_document`,
`packages/tcip-mcp/src/tcip_mcp/tools/proposal_tools.py`.

Readers: `tcip_annotation.json_io.read_document_versioned`,
`packages/tcip-annotation/src/tcip_annotation/json_io.py`, which decodes through
`json_io.label_document` (`json_io.py`);
`tcip_mcp.dataset_layout.capture_subjects`,
`packages/tcip-mcp/src/tcip_mcp/dataset_layout.py`.

A prediction record's `created_by` is one spelling, `buckets.prediction_producer`,
`packages/tcip-mcp/src/tcip_mcp/buckets.py`, so every checkpoint-backed writer
stamps the same `model:<checkpoint-stem>@<sha256-prefix>` producer identity rather than each door
composing its own string.

Seam S17 ("Canonical per-image annotation JSON schema"), verdict `both-sides-one-implementation`:
every writer and the shared `json_io.read_document_versioned`
reader are exercised in round-trip tests
(`tests/test_tcip_web_routes.py`, `tests/test_label_document_gestures.py`,
`tests/test_review_channel.py`, `tests/test_name_based_annotation_schema.py`). Gap
recorded by S17: no test cross-writes through one production writer and cross-reads through a
different consumer path in the same test.

## 2. External COCO dataset JSON, import only

Path: caller-supplied, single dataset-level `.json` file, not per-image. The platform writes none.

Reader: `tcip_annotation.format_io.parse_coco_annotations`, `format_io.py`, which names each
record's subject and decodes it through format 1's own decoder, called only by
`tcip_mcp.pipelines.data.coco_import.import_coco_document`, `coco_import.py`, which writes the
per-image documents of format 1 from it. Format 1's reader refuses this shape wherever it sits.

## 3. `subjects.json`, subject registry

Path: `<dataset_root>/subjects.json`. A dataset root carrying the retired `classes.json` is
refused by every registry writer until the file is renamed to `subjects.json` by hand; no
platform door conforms it. See S20 below.

Writer: `tcip_mcp.subject_registry.replace_registry`,
`packages/tcip-mcp/src/tcip_mcp/subject_registry.py`, the one write both registry doors call
(the GUI's `save_subjects` and the tool's `write_subject_registry`,
`packages/tcip-mcp/src/tcip_mcp/tools/annotation_tools.py`).

Readers: `tcip_mcp.subject_registry.read_registry`, `subject_registry.py`;
`tcip_mcp.dataset_layout.list_subjects` (delegates to `subject_registry`),
`packages/tcip-mcp/src/tcip_mcp/dataset_layout.py`.

`registry_scope`, `packages/tcip-mcp/src/tcip_mcp/pipelines/data/label_queries.py`, reads every
attribute this file declares for a subject into the admission's scope; a value's id is its position
in its attribute's declared order, and no integer id is stored in the file itself.

Seam S20 ("subjects.json subject registry"), verdict `both-sides-one-implementation`:
`tests/test_name_based_annotation_schema.py` writes the
registry through the real `replace_registry` and reads its attributes back through the real
training loader's scope, `packages/tcip-mcp/src/tcip_mcp/pipelines/data/label_queries.py`
(`return ClassScope(subject=subject, attributes=declared.attributes)`).
Gap: no test drives the actual `/api/subjects/save` HTTP route in the same test as the
training-side read.

## 4. `dataset.json`, dataset identity

Path: `<dataset_root>/dataset.json`.

Writer: `tcip_mcp.tools.project_tools.register_dataset`,
`packages/tcip-mcp/src/tcip_mcp/tools/project_tools.py`.

Reader: `tcip_mcp.pipelines.data.dataset_fingerprint.dataset_fingerprint` (recompute-on-read is
the stated authority; the stored value is a cache),
`packages/tcip-mcp/src/tcip_mcp/pipelines/data/dataset_fingerprint.py`
(`def dataset_fingerprint(dataset_root: str | Path, admitted: Iterable[Sample] = ()) -> str | None:`).

Seam S26 ("dataset.json identity and fingerprint"), verdict `both-sides-one-implementation`:
`tests/test_dataset_identity_recording.py` calls the
real `register_dataset` writer, then the real `split_construction.dataset_identity` reader, both of
which call the identical `dataset_fingerprint` function, and asserts they agree. Gap: the seam's
named third consumer, `tcip check-dataset-identity`, is never executed by any test.

## 5. Completion marks, inside format 1's label document

Path: format 1's own document, under its `complete` key: per subject, a list of marks, each
`{rect: [x, y, w, h], by, at, digest, proposals_hidden}`.

Writer: `tcip_mcp.dataset_layout.save_label_document`,
`packages/tcip-mcp/src/tcip_mcp/dataset_layout.py`, through the one encoder
`json_io.document_payload`, which keeps a subject's marks only while their digest
(`json_io.subject_digest`, over `json_io.canonical_digest`) still names that subject's
annotations, so an edit and the loss of its subject's marks are one write of one document.

Reader: `tcip_annotation.json_io.completion_marks`, decoded by `label_document`, and
`LabelDocument.state(subject)`, the one reading of `complete`, `negative`, `partial` and
`unannotated`, which the training admission (`label_queries.admitted_records`), the review queue
(`feedback_tools._prepare_queue_sources`), the editor's listing (`routes/annotate.py`'s
`_completion`), the session telemetry (`routes/sessions.py`'s `_marked_negative`) and the doctor
read; block calibration reads each mark's rect through `json_io.covers`
(`pipelines/block_calibration.py`'s `check_completeness`). A selection's sample names the
document its marks live in through its own `ground_truth` key.

Guards: `tests/test_label_document_gestures.py` (a mark on one subject survives another
subject's edit, a negative is read alike at the admission, the queue and the listing, a mark
carries `proposals_hidden`), `tests/test_block_calibration.py` (a mark's rect attests that region
alone).

## 9. `audit_log`, one append-only store under two kinds of root

Key: `audit_log_key`, `packages/tcip-mcp/src/tcip_mcp/audit.py`, a log in the database of the
root an entry's scope names: a dataset root for an event that changed a record traveling with
the data; the project root otherwise. Every writer names its scope; there is no process-wide
default log. `tcip dump-store` writes the log out as a file for reading.

Writers: two write paths, `audited` and `record_event_or_raise` (`audit.py`), each appending at
`audit_log_key(scope)`; a line records no `scope` field, since the log it sits in is its scope.

Every mutating door leaves exactly one line per act, the decorator's or the library's. The
decorator writes none for a call that returns its error dict and an `exception` line for a call
that raises; whatever a refusing call committed first is recorded by the library that committed
it. The library that mutates records the act, and the decorator stays only on a door whose own
act has no library owner: `audited` covers the platform's mutating doors (the MCP tools in
`tools/`, plus the script-invoked doors demoted from them) except a door whose libraries record
each of its acts (`import_coco`, `build_plant_mapping`, `deliver_per_plant_csv`,
`deliver_orthomosaic_plant_counts`, `deliver_phenology_milestones`, `register_model`,
`run_inference`, `deliver_per_image_counts`, `propose_trait`, `save_annotations`,
`write_subject_registry`, `run_hyperparameter_search`, and the three assessment doors
`assess_checkpoint`, `assess_reserved_regions` and `calibrate_physical_scale`, whose assessment
leaves its library's line): bare, the project's own log;
`@audited(scope_arg=...)` names the argument carrying
a dataset or project location, resolved via `dataset_scope_of` (`audit.py`) (through the tool's own
canonicalizer when the declaration passes one as `scope_via`). Four doors declare one: three
dataset-scoped (`register_dataset`, `tools/project_tools.py`; `propose_annotations`,
`tools/proposal_tools.py`; `stage_proposals`, `tools/proposal_tools.py`). A resolution that answers "no dataset" leaves
the entry in the project's own log; a resolver that raises refuses the call rather than filing it
there.

`record_event_or_raise` covers code that is neither an MCP tool nor a demoted door.
Project-scoped: the training envelope's open/close events
(`pipelines/training/envelope.py`), the model registry's write event (`model_registered`,
`model_registry.py`), an assessment's one line once its record is written
(`assessment_recorded`, `assessment.py`'s `_finish`),
and `routes/terminal.py`'s one line per agent-terminal launch (`agent_terminal_started`, `routes/terminal.py`). The `@audited(scope_arg=...)`
doors span every category by whatever root their declared argument resolves; this paragraph
names the explicit-emitter files, not a closed census of the decorator's doors.
Dataset-scoped: the label save every door calls (`save_label_document`,
`dataset_layout.py`), its line committed with the document it records, the registry write both
doors call (`replace_registry`, `subject_registry.py`),
`delivery.py`'s `delivery_event` line
(`delivery.py`'s `deliver_csv`, dataset-scoped when a
delivery's buckets share one dataset root, project-scoped otherwise),
the COCO import's `coco_document_imported` (`pipelines/data/coco_import.py`), committed with
the documents it imported, and a prediction bucket's `prediction_bucket_published`
(`buckets.py`'s `publish`), committed with the bucket's record and its documents, for an image
bucket and a raster bucket alike, whichever door published it (the GUI's inference worker
included); a pass that raises publishes nothing and leaves no line.
Project-scoped: `pipelines/postprocessing/plant_mapping.py`'s `persist_mapping`, reached from
both doors through the one build (`build_plant_mapping`, `pipelines/postprocessing/plant_mapping.py`),
whose receipt is the build's one line in the project's own log: the MCP tool passes the project
its server was started for and the web build route passes the backend's open project; and a
sweep's opening (`open_sweep`, `tools/training_tools.py`), which both the tool and the GUI's
relaunch reach.

A failed append is raised on both paths: `record_event_or_raise` raises `AuditEntryNotWritten`;
the decorator raises `MutationCommittedWithoutAuditLine`, because its append runs after the tool
body and a warning there invites a blind retry of a mutation already on disk.

A log's own root is its scope, so a line names no root and a moved or imported project's log
carries no machine path of its own; arguments a caller passed as absolute paths travel in an
archive unredacted, since a project archive is provenance-preserving, not path-sanitized.

Readers: every reader goes through the storage seam's `read_log` rather than decoding lines by
hand. The receipt gate refuses (never scans past) a page reporting corruption:
`plant_mapping._scan_receipts`
(`pipelines/postprocessing/plant_mapping.py`) and `_require_receipt`
(`pipelines/postprocessing/plant_mapping.py`), the hard receipt gate `load_mapping` runs
before trusting a persisted mapping record:
every `plant_mapping_built` entry in the record's own project log is scanned for a receipt naming
the record's digest, and a page reporting `page.corrupt` raises rather than reading past it,
since an entry could be hiding behind an unreadable line unread. `read_audit_log`
(`tools/meta_tools.py`) answers an error naming the corrupt count, and `inspect_project`'s recent
activity (`tools/project_tools.py`) answers `status_unavailable` naming it.

## 10-15. `.tcip/experiments/<experiment_id>/`, a run directory

Path root: `.tcip/experiments/<experiment_id>/`, resolved via `experiments_dir`,
`packages/tcip-mcp/src/tcip_mcp/experiments.py`, under the project the caller names. A run
is a directory of plain files, not store records: each file is written by one process, the launch
record once by the parent and the final status once by the child, and nothing in it is rewritten.
Immutability is the files' own: `publish_once`, `experiments.py`, writes a file's whole bytes
to a staging name and publishes them under the final name without replacing anything, so a file
under its final name is always whole. A relaunch or a resume is a new directory naming its source
(`relaunched_from`, `resume_from`); `create_run_directory`, `experiments.py`, refuses a
directory that already exists (`RunDirectoryExists`). A sweep is the same shape one level up,
`.tcip/experiments/<sweep_id>/` beside the runs, so the one directory creation reserves a name for
runs and sweeps alike, and a sweep is told from a run by its `sweep.json` input written
once before its body starts, a heartbeat, its final status, and one `<sweep_id>_<trial id>/` run
directory per trial, which `run_dirs` lists beside the project's own runs, so every run route and
tool reaches a trial by its id; the sweep's TensorBoard board is its own directory (`board_of`).

- `run.json` (`RUN_FILE`, `experiments.py`): written once by `open_run`,
  `packages/tcip-mcp/src/tcip_mcp/tools/training_tools.py` (`def open_run(`), before the child
  starts: the config as launched with its seed drawn onto it, what the launcher resolved it to
  (`resolved`: the data section, the partition and the objective with its direction, from
  `split_construction.resolve_run`; `null` for an HPO trial whose point failed to resolve, whose
  final status is written `failed` naming why), the environment, the dataset identity, the
  run it relaunched and the checkpoint it resumes from, its wall clock, the model contract
  preflight proved, the source snapshot, and an HPO trial's sampled point. The child builds its
  loaders from `resolved` and resolves nothing again. Read through `observe`,
  `experiments.py`, and `run_resolution`, `experiments.py`, which `assessment.py` takes the
  partition from.
- `metrics.jsonl` (`METRICS_FILE`, `experiments.py`, append-only): created empty with the
  directory, appended through `append_row`, `experiments.py`, by the training envelope's one
  sink, `packages/tcip-mcp/src/tcip_mcp/pipelines/training/envelope.py`
  (`def _epoch_sink(`). Read by `read_rows`, `experiments.py`. A row landing after the final
  status is reported as late (`rows_after_end`), never read as reopening the run.
- `heartbeat` (`HEARTBEAT_FILE`, `experiments.py`): touched by `keep_heartbeat`,
  `experiments.py`, while the run's process lives; `last_alive`, `experiments.py`, falls
  back to the launch record before the first touch.
- `cancel_requested.json` (`CANCEL_FILE`, `experiments.py`): written once by `request_cancel`,
  `experiments.py`, which keeps the first request's time and refuses a directory whose final
  status is written.
- `final_status.json` (`FINAL_STATUS_FILE`, `experiments.py`): written once through
  `write_final_status`, `experiments.py`, by the envelope or by the child's pre-envelope
  failure path: the state, when it ended, its error, and for a completed run the checkpoint
  (path inside the run directory and sha256) the verified checkpoint reader admitted. `observe`
  answers that state once written, else `running` or `interrupted` by the heartbeat window; a
  run's summary, completed or live, is the best selection over its own metrics rows under its
  recorded objective (`run_summary`, `experiments.py`).
- The checkpoints: the files the run's body saves (`model_best.pt`, held in memory until the run
  ends, `model_final.pt`, `checkpoint_epoch_*.pt`, a bespoke loop's own tags), each written once
  under a name no other write takes. An evaluation of the run writes nothing here.

Readers over the whole directory: `get_experiment`, `experiments.py`; `compare_experiments`,
`experiments.py`; `get_experiment_lineage`, `experiments.py`.

The run directory's lifecycle, each case through the real launcher and a real child process
(launch to completion, cancel by id, resume into a new directory, a child killed before its
final status, a row logged after the final status, a launch into an existing directory, and an
HPO trial as a run directory reporting its sweep's one objective), is held by
`tests/test_run_directory_lifecycle.py`. The metrics log's writer and the training stream's reader
are held against each other by `tests/test_metrics_row_writer_reader_agreement.py`, which writes
rows through the real run body and reads them back through the real websocket route. The
partition's writer (the
launcher's `resolve_run`, recorded by `open_run`) and its readers (`assessment.py`'s disjointness
check and its reserved-regions assessment) are driven against the same
resolved record by `tests/test_spatial_region_containment.py`,
`tests/test_run_partition_membership_fidelity.py` and `tests/test_selection_binding.py`.

## 16. The trained-model registry index

Key: `registry_index_key`, `packages/tcip-mcp/src/tcip_mcp/model_registry.py`, store
`model_registry`, in the project's database.

The registry is one relation from sha256 to its one owner, `registered_entries`,
`packages/tcip-mcp/src/tcip_mcp/model_registry.py`: the completed run whose final status names
those bytes (`run_entry`, `model_registry.py`, named by the run's id and carrying it as
`experiment_id`; the earliest to complete when two do), else the index's foreign entry with
`experiment_id` null. A run's completion is its registration: nothing writes a run's entry into
the index. The index holds foreign registrations, one entry per sha256, written by
`ModelRegistry.register_model` (the `register_model` tool's door, `tools/model_tools.py`),
which digests the bytes the verified checkpoint reader admits (`admitted_digest`,
`model_registry.py`), replaces an earlier entry of the same sha256 and returns the owner the
relation answers, so a registration before or after the producing run completes leaves one entry;
a name is presentation only. What a ranking reads (`metrics`, `metrics_source`) comes from
the checkpoint's payload through that reader (`entry_facts`, `model_registry.py`): a run's
metrics are the ones its payload carries, sourced `"trainer"` or `"training_source"`, a foreign
entry's the ones its registration stated, sourced `"caller"`.

The index record is `{entries: [...]}` at `registry_index_key`; an absent record answers no
entries. `ModelRegistry.register_model` spells `checkpoint_path` through
`registry_paths.checkpoint_registry_path_for` against the registry's own scope root: relative
POSIX when the checkpoint resolves under it, absolute when it does not (the dataset registry's
own `entry_is_external`/`registry_path_for` share the same containment core and grammar-aware
`is_external_form` test, `registry_paths.py`). An entry's path is read as written; nothing
searches for a moved checkpoint by suffix, basename or digest.

Readers: `read_registry_index`, `model_registry.py`, the read path for anything outside the
module (`packages/tcip-mcp/src/tcip_mcp/cli/doctor.py`, `"metrics_source"`), and the accessors built on
it: `ModelRegistry.list_models`, `model_registry.py`, and the module's `best_model`,
`model_registry.py`, over its entries. `best_model` takes `metric_key` and `higher_is_better`
as required keywords, no default and no name heuristic, and by default ranks only entries whose
`metrics_source` is `"trainer"` (`include_unverified=True` also ranks the rest). The
`rank_registered_models` tool (`tools/model_tools.py`) resolves `higher_is_better` from
`evaluation.HIGHER_IS_BETTER_BY_METRIC` (`pipelines/training/evaluation.py` (`HIGHER_IS_BETTER_BY_METRIC: dict[str, bool] = {`)) when the caller
states none, the single declared-direction mapping `resolve_selection_metric`
(`pipelines/training/generic_trainer.py`) also reads for the trainer's own checkpoint selection.
Every one of these accessors, plus `register_model`'s own return, answers `checkpoint_path`
resolved to an absolute path (`registry_paths.resolved_registry_path`) on a copy, never the
registry's own internal relative-or-absolute storage spelling; `checkpoint_files` (the
checkpoints an archive leaves out) and `doctor.py`'s registry check resolve through the same
function. `unresolved_registered_checkpoints`
and `import_project`'s own disclosure split on `is_external_form`: a designed-external entry is
`external_checkpoints` (its own existence stated per entry), never counted toward
`checkpoint_paths_unresolved` (an entry expected to resolve under the tree that does not).

Seam S27 ("Trained-model registry index"), verdict `one-side-only`: `tests/test_lifecycle_wiring.py`,
`tests/test_model_registry_metrics.py`, `tests/test_provenance_spine.py`,
`tests/test_tcip_web_results_routes.py`. Gap: no test registers a real model and then
calls `GET /api/results/models/registered`, or runs an inference launch end to end through the web
route's identity-stamp block, to confirm the GUI-visible entry matches the MCP-registered one; the
only two web-route tests check a 403-confinement case and an empty-registry case.

## 17. Prediction buckets, write-once named publications

A prediction bucket is not a score bin or a quota allocation: it is one named publication of a
single model run's per-image prediction documents (format 1's `prediction_documents`) and one
bucket record (format 18), both in the database of the dataset root its source images belong to,
its identity that root and its name. A bucket is published once: `publish`,
`packages/tcip-mcp/src/tcip_mcp/buckets.py`, commits the record, every document and its
`prediction_bucket_published` line in one transaction and refuses a name already published
(`BucketExists`), so a new run names a new bucket and nothing rewrites a published one. A
publication with no document refuses. A staged proposal is a bucket of its own, one per staged
image, its record naming the engine or agent that proposed it and no execution.

Name: caller-named, on every door alike; nothing composes it from a model name and a date.

Writer: `publish`, `buckets.py`, encoding each document with format 1's schema.

Readers: `read_bucket`, `buckets.py`, the one decoder; `buckets_by_date`, `buckets.py`,
enumerates a dataset's buckets by reading each record; `Bucket.document_key`, the one document
lookup, answers only the documents the record names.

A second run into a published bucket refuses and leaves it
(`tests/test_run_inference_bucket_handling.py`); publishing the same bucket twice refuses the
second publish (`tests/test_end_to_end_measurement_chain.py`).

## 18. The bucket record, a prediction bucket's statement

Key: `bucket_key`, `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py`, store
`prediction_buckets` keyed by the bucket's name, in the dataset root's database (format 17).

Writer: `publish`, `packages/tcip-mcp/src/tcip_mcp/buckets.py`, once, in the transaction
that writes the documents it states: the producer (the checkpoint's sha256 and producing
run), the scope (subject, attribute, id map), the execution record the pass ran under, the
capture's dataset id and date, the raster path with the raster's identity for a raster pass, the
documents by stem with the source file each was predicted from, the count of dropped zero-extent
boxes, and the id of the assessment it was published under, if any. Every door that predicts
publishes through it: `run_inference` for an image directory and a raster alike, and the GUI's
inference worker.

Reader: `read_bucket`, `buckets.py`, the one decoder, refusing a name with no record or a
record that does not decode (`NotABucket`). `tools/orthomosaic_tools.py`'s
`deliver_orthomosaic_plant_counts` reads `raster_identity` back and refuses a delivery whose
supplied raster does not match it; `delivery.gate` reads `assessment_id`, the producer and the
date for its checks (format 27).

## 19. The live GUI state snapshot

Key: `gui_snapshot_key`, `packages/tcip-mcp/src/tcip_mcp/web_client.py`, store
`gui_snapshot`, in the project's database.

Writer: `write_gui_snapshot`, `tcip_mcp/web_client.py`, called by `StateStore.mutate`
(`tcip_web/state.py`) for the project open when the change is made, before the change is
held; a write that fails raises and the change is not held.

Reader: `read_gui_snapshot`, `tcip_mcp/web_client.py`, run by `StateStore.open_project`
(`tcip_web/state.py`) each time a project is opened and by the MCP `view_gui_state` tool; a
snapshot that does not decode as its whole shape raises.

`StateStore.mutate` validates the merged mutation through `GuiState` before holding it, raising
`GuiMutationInvalid` (`tcip_web/state.py`) on a field that does not validate or a key
`GuiState` does not declare; `app.py`'s `_gui_mutation_invalid_handler` answers a route that
raises it with 400 and the validation message rather than the 500 an unhandled `ValueError` would
produce.

Seam S10 ("Live GUI state snapshot"): one writer and one reader, both in
`tcip_mcp/web_client.py`; `tests/test_active_context.py` persists through the backend's own
`StateStore` and reads the snapshot back through `view_gui_state` and a reopened store.

## 20. Recent project activity, derived from the project's audit log

No record holds it. `inspect_project`'s `recent_activity` is derived when it is read
(`_recent_activity`, `tools/project_tools.py`) from the project log's `report_friction`,
`write_retrospective` and `record_distillation_pass` lines, in the order they landed.

## 21. The verdict shard log

A log keyed by `verdicts.verdict_key` on the store `review_verdicts` in the dataset root's
database, parts `(bucket, stem)`: the prediction bucket's name and the stem the image's label key
carries (`dataset_layout.verdict_key_of`).

Each entry is one decision, `{proposal, action, by, at}`: the proposal's index in the bucket's
document for the image, `accepted` or `rejected`, the person in the label record's own spelling,
and the time. Writer: `verdicts.record_verdicts`, called only by
`dataset_layout.save_label_document` when a save accepts or rejects a proposal, each entry
through `verdicts.encode_verdict`, all of one save's entries in one commit. Reader: `verdicts.read_verdicts`, every entry through
`verdicts.decode_verdict`, which refuses a malformed entry by name; the editor's proposals route
reads each proposal's last decision through it, and `tcip doctor`'s `check_state` reports a
shard that will not read.

Guard: `tests/test_label_document_gestures.py` (each decision recorded once, a malformed entry
refused); `tests/test_store_contract.py` (a save whose verdict will not encode lands none of its
verdicts).

## 22. The project-level dataset identity registry

Key: `dataset_registry_key`, `packages/tcip-mcp/src/tcip_mcp/tools/project_tools.py`, store
`dataset_registry`, in the project's database.

Writer: `upsert_dataset`, `packages/tcip-mcp/src/tcip_mcp/tools/project_tools.py`.

Reader: `read_datasets`, `project_tools.py`.

Shape: each entry's `path` is relative to `<project_root>` whenever the dataset sits under it by
filesystem identity (`registry_path_for`, `project_tools.py`), the project's own tree becoming
`"."`; absolute for a dataset outside it. Resolved on read through the one accessor,
`dataset_entry_path`, `project_tools.py`, which every consumer of an entry's path calls rather
than reading `path` off the entry directly.

## 23. Workspace last-opened pointer

Key: `last_opened_key`, `packages/tcip-mcp/src/tcip_mcp/workspace.py`, store
`workspace_last_opened`, in the workspace root's database: one project id as a JSON string.

Writer: `write_last_opened`, `workspace.py`, called by the web backend's `open_project_by_id`
(`packages/tcip-web/src/tcip_web/routes/projects.py`) on every open.

Readers: `read_last_opened`, `workspace.py`, read by the backend's lifespan through
`open_last_opened` (`routes/projects.py`), which opens the project whose record holds that id
(`project_by_id`, `workspace.py`); when none does it opens nothing, and the project list
names the missing id (`last_opened_problem`). The
pointer is a preference, never the MCP server's project: each server is started for one project
by `--project`.

## 24. Formats named but not enumerated here

`classifier_operating_point.json`, `ordinal_operating_point.json`, `regression_operating_point.json`
(sibling sidecars to `operating_point.json`, named in `pipelines/operating_point.py`); any format
defined inside `pipelines/postprocessing/` (`selection.json` is covered in format 26).

## 25. The per-project record (id, display name, site)

Key: `project_record_key`, `packages/tcip-mcp/src/tcip_mcp/project_record.py`, on the store
`PROJECT_RECORD_STORE`, `project_record.py`, in the project's database. It holds `id` (minted once at creation by `mint_id`,
`project_record.py`), `display_name` and `site`; the id is the project's identity, never its
directory name.

Writers: `create_record`, `project_record.py`, a create-only write: an absent record is
written, a present record with the same display name and site is left as is, and one differing
raises. `rename_project`, `project_record.py`, replaces the display name alone; `replace_site`,
`project_record.py`, behind `tcip write-project-site --replace`, the site alone. Each keeps the
id.

Readers: `read_record`, `project_record.py`; `record_fields`, `project_record.py`, is the
one reader every surface (the picker, `inspect_project`, the doctor, `project_by_id`) calls, and
never raises.

## 26. `selection.json`, a partition `draw_splits` drew

Path: `<output_path>/selection.json`, addressed by `selection_key`,
`packages/tcip-mcp/src/tcip_mcp/pipelines/data/selection.py`, under whatever directory the
caller asked the partition to be written to; no dataset resolver owns this layout.

Writer: `draw_splits`, `data_tools.py`, when `output_path` is given. A finished run's own
drawn train/val partition freezes into the identical shape through `freeze_selection`
(`data_tools.py`), the second writer; both go through the one `write_selection`
(`selection.py`), which composes the document through `selection_document`
(`selection.py`) and refuses a crossing partition before anything lands, so the two writers
can never disagree on what a selection carries or write one a reader would reject. A frozen
selection also carries `origin` (`{"experiment_id", "frozen_at"}`), absent on a drawn one, the
field `read_selection` and its callers use to tell the two apart.

The record is its sample list. Each sample carries `source` (the image path, the `.bandgroup`
manifest standing in for a grouped capture, or the raster path when the sample is a region),
`ground_truth` (whatever answers for it, never derived from `source`: for a label document its
record's `{root, capture, stem}`, the document whose completion marks admitted it; for a mask or
a table its path), `group` (the key that keeps
related samples together), `side` (one of `selection.SIDES`: `train`, `val`, `calibration`), and
optionally `rect` (a half-open pixel rect
for a within-image draw), `row_key` (the row inside a tabular ground truth) and
`ground_truth_digest` (that ground truth's version token at draw time). `rect` and `row_key` are refused by
`selection.refuse_unreadable_samples` wherever a loader would otherwise read past them. Nothing in the record names a capture date or a directory scope:
every sample names its own paths, so one selection spans as many dates as the draw admitted, and
two dates holding a same-named image are two samples rather than one identity that has to be told
apart from itself. Beside the samples the record carries `scope` (the subject and every
attribute record the admission read from the registry for it), `seed`, `group_by` (the
resolved policy), `dataset_fingerprint`, `admission_counts` (the summed admission counts;
a frozen selection records `{}`, since freezing a training run's own drawn partition records no
admission draw), `realized_ratios` and `origin`.

A side whose ratio is zero is not drawn, written or not, and a negative ratio refuses. The draw
refuses, before any write, when the tree holds fewer foreground groups of `subject` (and
`attribute`, when scoped) than one per side it draws, counted through a subject-scoped `count_label_lines` on every draw,
independent of `stratify_foreground` (which only gates the balancing pass's own foreground
signal). `realized_ratios` records each side's delivered member share: at a floor-sized tree the
minimum pass can consume every foreground group before the balancing pass ever sees the caller's
fractions, so the delivered shares can diverge from the ratios asked for. The answer also carries
`calibration_foreground_groups`, how many of the calibration side's own groups carry foreground
at all, since the floor above is over the whole draw and the calibration slice can still land
short of two; the calibration door's own floor is where that absence bites. A stats-only call (no
`output_path`) is the same draw with the same answer, written nowhere.

Readers: `selection.read_selection` (`selection.py`), the one reader a training run's or sweep
trial's `data.split.selection_dir` resolves through (`split_construction.auto_train_val`) and a
selection-restricted calibration resolves through
(`splits.selection_calibration_universe`), which refuses by name when the record is
absent, undecodable, not a mapping, lists no samples, holds a sample missing any of
`source`/`label`/`group`/`side`, names a side outside `selection.SIDES`, carries a malformed
`rect`, or holds a partition whose sides cross. That last check is `refuse_crossing_sides`
(`selection.py`), the same one the writer runs: one source identity on two sides is the same
pixels trained on and selected on, and one group key on two sides splits the crops of one parent
across sides. `tcip plant-aware-group-splits` reads no selection back, it only writes one through
`draw_splits`.

`selection.read_selection_checked` (`selection.py`) is the checked variant a listing calls in
place of the raising reader: absence answers `(None, None)`, a record that exists but will not
decode or fails a shape check answers `(None, text)`, so a refused record never reads as an
ordinary absence.
`split_construction.selection_compatibility` is every objection a launch binding one config to
one selection would raise, checked ahead of that launch: the config-only checks (a drawn split's
own keys, and a stated `data.labels_dir`, since each sample names its own ground truth), computed
before any read so an unreadable selection never suppresses them, and the selection-dependent
checks (a stated `data.scope`, an empty train or val side). A bound run states neither: it reads
subject, attribute and class map off the selection, and each sample's ground truth off itself. `preflight_config` calls both halves directly, in the same order, over a selection it
read itself; `training_tools.list_split_choices`, the relaunch data picker's own reader wrapped by
`GET /api/training/configs/{experiment_id}/splits`, calls the composed function per candidate
selection it read through the checked variant above, and builds each candidate's launch config
through `training_tools.candidate_config_with_selection`, the same function the relaunch route's
own launch build calls.

`tests/test_selection_binding.py` calls the real writer and the real consumer
(`auto_train_val`'s selection branch) against the same files.

## 27. `assessment.json`, `.tcip/assessments/<assessment_id>/`

Path: one directory per assessment under the project's `.tcip/assessments/`, holding
`assessment.json` and, under `reference/`, the one read of each reference ground truth it
measured: a label document as its stored record, a mask or a table as its bytes
(`packages/tcip-mcp/src/tcip_mcp/assessment.py`).

Writer: `assessment.assess`, `assessment.assess_reserved_regions` and
`assessment.assess_physical_scale`, each finishing through `assessment._finish`, one write-once
record and one audit line; nothing rewrites an assessment. The record states the trait revision
(`TraitRevision.ref`), the delivery kind, the producer, the execution record, the reference (each
sample's source digest and ground-truth digest, the retained copies, the captures or raster it
covers), the disjointness evidence, the criterion's evidence, the scale (physical-scale only), the
failures and whether it passed.

Readers: `assessment.read_assessment`, the one decoder into a typed `Assessment`, by
`delivery.gate` for every delivered bucket naming one and for a delivery's scale assessment, by
`run_inference` restoring the execution record an assessment measured, and by the assessment
doors answering with it.

## Formats with a general path-resolution seam but no per-format seam entry above

Seam S14 ("dataset_layout.py as the on-disk path resolver"), verdict `both-sides-restated`,
and seam S15 ("Per-image label document key"), verdict `one-side-only`, both bear on how formats 1-5's
keys are derived rather than naming one format's own schema; they are cross-referenced here
rather than assigned to a single numbered format above. A label document's key is derived once,
from the image's own path, by `dataset_layout.parse_image_path` and `label_key`; the frontend
names the image and composes no label location.


## Seam inventory

Each seam below is a fact two sides must agree on, each side cited by its file and the fragment
it quotes.

## S03. Backend port discovery record

Must agree: the MCP process finds the port the web backend actually bound.
Side A: `packages/tcip-mcp/src/tcip_mcp/web_client.py` (`BACKEND_PORT_STORE`, declared beside the reader because the reader cannot import `tcip_web`; `backend_port_key`, `web_client.py`, is the one address, read in `web_client.py`).
Side B: `packages/tcip-web/src/tcip_web/__main__.py` (`replace(backend_port_key(workspace), str(port))`, publishing through that same key under the workspace `main` resolved once, and raising rather than swallowing a failure, since the fallback silently misses an OS-picked port).

## S04. Panel-event panel vocabulary (VALID_PANELS)

Must agree: sender and receiver accept the same set of panel names.
Side A: `packages/tcip-mcp/src/tcip_mcp/web_client.py` (`VALID_PANELS = frozenset(`).
Side B: `packages/tcip-web/src/tcip_web/app.py` (`VALID_PANELS,`).

## S05. Panel event_type vocabulary

Must agree: the Python poster, the FastAPI hub, and the browser handler use the same event_type strings.
Side A: `packages/tcip-mcp/src/tcip_mcp/tools/gui_tools.py` (`result = post_panel_event(project, workspace, "app", PANEL_EVENT_ANNOTATE_FOCUS, payload)`).
Side B: `packages/tcip-web/src/tcip_web/app.py` (`if event.event_type == PANEL_EVENT_ANNOTATE_FOCUS:`).
The posted payload carries `subject` beside `active_subject` (`packages/tcip-mcp/src/tcip_mcp/tools/gui_tools.py`), the key both readers take (`packages/tcip-web/src/tcip_web/app.py`, `frontend/src/lib/annotateFocus.ts`), held by `tests/test_event_integration.py`'s producer-driven test, which posts the focus_human_attention tool's own event and asserts the advisory state's `active_subject`.

## S06. `audit_log`, one append-only store under two kinds of root

Must agree: mutations from any process land in the log the scope names, a dataset's own for a
record traveling with the data and the project's own otherwise, all with the same entry shape;
and the project's own receipt gate (`plant_mapping.load_mapping`) trusts only what that project's
own log actually recorded.
Side A: `packages/tcip-mcp/src/tcip_mcp/audit.py` (`def audited(`, taking a declared
`scope_arg` naming which tool argument carries the dataset a scoped tool mutates a record of) and
`record_event_or_raise` (`audit.py`), the emitter for code that is neither an MCP tool nor a
script-invoked door demoted from one; both append at the one `audit_log_key`, `audit.py`, and
differ only in what a failed append raises: `record_event_or_raise` raises
`AuditEntryNotWritten`; the decorator refuses (`MutationCommittedWithoutAuditLine`), since its
append runs after the tool body.
Side B: the label save records through its library (`save_label_document`), its line committed
with the document, so a line that cannot be written leaves the document unwritten; the registry
write records through `replace_registry` whichever door calls it, a failed append raising
`AuditEntryNotWritten`, which the GUI route answers as `routes/audit_gap.py`'s 409; the GUI
inference worker writes no line of its own and publishes through the one publisher,
`routes/inference.py` (`result = infer(`), whose line commits with the bucket. Reader:
`pipelines/postprocessing/plant_mapping.py` (`_require_receipt`)
trusts only a `plant_mapping_built` entry it finds in the log under the project its caller names
(the MCP server's started project; the web backend's open project), scanned by `_scan_receipts`
(`pipelines/postprocessing/plant_mapping.py`), which refuses (never scans past) a page
reporting corruption.
Each writer is exercised through a real append and checked for its own
tool name landing in the log its own scope names: `tests/test_tcip_web_routes.py`'s
`test_annotate_save_audits_into_the_log_of_the_dataset_it_wrote` (a dataset-scoped GUI write,
checked against the same dataset's log, never the platform's);
`tests/test_tcip_web_results_routes.py` (a dataset-scoped delivery-binding event beside a
project-scoped export audit line, from the one route); `tests/test_tcip_web_subjects_routes.py`'s
`test_a_registry_save_leaves_one_library_line_through_either_door` (a dataset-scoped registry
write, checked against the project log for the same request); and
`tests/test_audit_row_core_field_agreement.py`, which runs a real GUI route and a real
platform write against one log and holds the two rows to the same core fields. The receipt
gate's own agreement is held by `tests/test_plant_mapping.py` (a real receipt admits) and
`tests/test_plant_mapping_binding.py` (a receipt that cannot be written fails the build
and the web route alike).

## S07. Experiment record .tcip/experiments/<id>/

Must agree: the launching process and the run's own child agree on the run directory's layout, and each file is written by one of them once.
Side A: `packages/tcip-mcp/src/tcip_mcp/experiments.py` (`def experiment_dir(` plus the file names beside it, the one declaration of the directory's path and members).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/training/subprocess_worker.py` (`def run_directory(`, the child reading the launch record `open_run` wrote and writing its metrics log and final status beside it).

## S08. metrics.jsonl row format

Must agree: the writer's row shape is what the reader and the stream consumer expect.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/training/envelope.py` (`def _epoch_sink(`, the one writer; the trainer and a bespoke loop hand rows to it).
Side B: `packages/tcip-web/src/tcip_web/routes/training.py` (`rows, cursor = await asyncio.to_thread(read_rows, observation.metrics_log, after=cursor)`, the training stream's incremental tail read off the event loop, pushed as a `TrainingMetricFrame` per row, for a sweep's trial as for any run).
An HPO trial is a run directory, so its log is the same file shape written by the same sink.

## S10. Live GUI state snapshot

Must agree: the MCP agent reading GUI context parses the snapshot the web backend wrote.
Side A: `packages/tcip-mcp/src/tcip_mcp/web_client.py` (`def write_gui_snapshot(`, the one writer, through the one address `gui_snapshot_key`, which `StateStore.mutate` in `tcip_web/state.py` calls).
Side B: `packages/tcip-mcp/src/tcip_mcp/tools/project_tools.py` (`state = read_gui_snapshot(project)`, the one reader, which the backend's own `open_project` also calls).

## S11. Live canvas state records, under the backend's open project

Must agree: the push route writes the canvas meta and geometry records under the project the backend has open, and capture_live_canvas reads them from the project its MCP server was started for, so a capture reads the GUI's live canvas only while the two are the same project. Both sides address the records through one key pair (`canvas_meta_key`, `packages/tcip-mcp/src/tcip_mcp/web_client.py`; `canvas_geometry_key`, `web_client.py`). The push carries the project id the browser drew for (`packages/tcip-web/src/tcip_web/routes/canvas.py`, `def push_canvas_state(`), compared with the open project's own id (`StateStore.project_id`, `packages/tcip-web/src/tcip_web/state.py`) before anything is written, and a panel event from the MCP side carries its project's id the same way (`web_client.py`), delivered only when that project is open.

## S12. Friction reports and retrospectives under .tcip/

Must agree: the GUI reader finds, decodes and orders what the MCP writer produced.
Side A: `packages/tcip-mcp/src/tcip_mcp/tools/meta_tools.py` (`def report_documents(`, the one enumeration, decode and ordering of the friction reports, with `retrospective_documents`, `meta_tools.py`, doing the same for the retrospectives). Both stores are records, enumerated through the seam and ordered by the timestamp each document states (a report's own `timestamp` field, a retrospective's own `## Retrospective:` section headers), never by when the bytes landed. A report is one whole JSON document, not a line of a stream.
Side B: `packages/tcip-web/src/tcip_web/routes/meta.py` (`get_reports`) and `routes/meta.py` (`get_retrospectives`), both routes importing those MCP-side enumerators directly and presenting the rows they return, rather than walking a directory of their own.

## S13. Session telemetry's negative-confirmation time

Must agree: nothing; a contribution's activity is stated once, when it is recorded, by its
producer: `packages/tcip-web/frontend/src/store/slices/registryStatus.ts` (`? "negative_confirmation"`)
into `packages/tcip-web/src/tcip_web/routes/sessions.py` (`Activity = Literal[`), and no
reader reclassifies it from the label document later.

## S14. dataset_layout.py as the image-tree and record-key resolver

Must agree: agent writes and GUI reads resolve to the same records.
Side A: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py` (`def label_key(`, with `prediction_key` and `bucket_key` beside it and `parse_image_path` deriving an image's root, capture and stem).
Side B: `packages/tcip-web/src/tcip_web/routes/dataset.py` (`select_dataset` resolves the image directory through the resolver; `tcip_mcp.cli.doctor`, `data_tools`, `project_tools` and `annotation_tools` no longer re-spell the tree).

## S15. Per-image label document key

Must agree: the browser and the Python resolver name the same label document.
Side A: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py` (`def parse_image_path(`, the one derivation of an image's document key from its own path).
Side B: the browser names the image only; every annotate route derives the key through Side A.

## S16. Verdict shard store

Must agree: the verdict writer and every verdict reader look at the same shard (a bucket
here is format 17's named publication, not a score bin).
Side A: `packages/tcip-annotation/src/tcip_annotation/verdicts.py` (`def verdict_key(`, a shard in the dataset root's database keyed by the bucket's name and the image).
Side B: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py` (`def verdict_key_of(`, the one address the label save writes and the editor's proposals route reads).

## S17. Canonical per-image annotation JSON schema

Must agree: every writer produces, and every reader accepts, the same name-based record shape.
Side A: `packages/tcip-annotation/src/tcip_annotation/json_io.py` (`def annotation_from_payload(`, the one conversion from a client payload to a record, with `json_io.py` (`_annotations_of`) the one parse back and `json_io.py` (`document_payload`) the one encoder every writer commits).
Side B: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py` (`contents.append(annotation_from_payload(payload))`, inside the one label save every door calls rather than assembling records of its own).

## S18. Bounding-box coordinate convention across the HTTP boundary

Must agree: the browser and the route use corner coordinates while the file uses xywh, with the conversion happening once.
Side A: `packages/tcip-web/src/tcip_web/routes/annotate.py` (`bbox: Optional[list[float]] = None          # [x1, y1, x2, y2], pixel`, the wire form).
Side B: `packages/tcip-annotation/src/tcip_annotation/json_io.py` (`def xywh(`, the one corner-to-xywh conversion and the 2-decimal grid the stored document lives on, applied on write and, via the import at `packages/tcip-mcp/src/tcip_mcp/pipelines/training/evaluation.py` (`from tcip_annotation.json_io import xywh`), to every box scored against a stored label so both sides of a match sit on one grid; a wire box becomes a `BBox` in `annotation_from_payload` (`json_io.py`), and `_annotations_of` (`json_io.py`) is the inverse read).

## S19. Annotation format detection scope (json, coco)

Must agree: a label document's shape is decided by its one reader, never guessed beside it.
Side A: `packages/tcip-annotation/src/tcip_annotation/json_io.py` (`def label_document(`), the one decoder of a stored document; a dataset-level COCO enters only through `coco_import.import_coco_document`, never at a document's key.
Side B: none; admission, the doctor and the read tool take that reader's answer.

## S20. subjects.json subject registry

Must agree: the GUI editor, the path resolver, and the training loader read one registry shape.
Side A: `packages/tcip-mcp/src/tcip_mcp/subject_registry.py` (`The on-disk registry (`` `<dataset_root>/subjects.json` ``) is self-describing and name-based::`).
Side B: `packages/tcip-web/src/tcip_web/routes/subjects.py` (`registry, version = read_versioned_registry(root)`, over the dataset root the path guard admitted).

## S21. Training attribute ids versus inference attribute values

Must agree: a prediction's attribute ids decode to the values the run trained them as.
Side A: `packages/tcip-annotation/src/tcip_annotation/json_io.py` (`def attribute_ids(`, the one decoder of a record's values into ids, each a position in its attribute's declared order under the admission's scope).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/execution.py` (`scope=ClassScope.of(checkpoint.data_config)`, the checkpoint's own recorded `scope`, every attribute record included, taken by the prepared pass every inference regime, the GUI worker and every assessment share, and decoded back to values by `json_io.py` (`def attribute_values(`); the reserved-regions assessment reads it off that pass in `assessment.py` (`scope = p.scope.admitted_for(DOCUMENT`); nothing re-reads the attributes from a live registry).

## S22. Completion marks in the label document

Must agree: a negative is an empty subject plus a person's mark, every consumer reads the same marks the same way, and each mark carries who made it and when.
Side A: `packages/tcip-annotation/src/tcip_annotation/json_io.py` (`def completion_marks(`, the one decoder of a document's marks, keeping only those whose digest still names their subject's annotations).
Side B: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py` (`marks = {s: held for s, held in stored.marks.items() if gestures.complete.get(s, True)}`, the one writer's carry-over through the label save); the browser receives each subject's derived state from the load route and restates no predicate, its `SubjectState` type generated from the Python `Literal`.

## S25. Region completeness on an orthomosaic

Must agree: a region a person marks complete is the region the calibration gate reads as attested.
Side A: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py` (`rect=gestures.rect or (0, 0, width, height), by=cast(str, author), at=now,`, the mark's rect as the save records it).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/block_calibration.py` (`uncovered = sorted(name for name, rect in rects.items() if not covers(marks, rect))`).

## S26. dataset.json identity and fingerprint

Must agree: the stored fingerprint and the recomputed one cover the same inputs.
Side A: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py` (`def dataset_identity_path(dataset_root: str | Path) -> Path:`).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/data/dataset_fingerprint.py` (`def dataset_fingerprint(dataset_root: str | Path, admitted: Iterable[Sample] = ()) -> str | None:`).

## S27. Trained-model registry index

Must agree: the MCP registrar and the GUI model pickers read one registry entry shape.
Side A: `packages/tcip-mcp/src/tcip_mcp/model_registry.py` (`def registered_entries(`, the one read every consumer goes through: one owner per sha256, the completed run whose final status names the bytes, else the index's foreign entry; `_write_registry_entry`, `model_registry.py`, replaces one foreign entry by sha256 inside one `tcip_store.transaction` on the key `registry_index_key`, `model_registry.py`, mints).
Side B: `packages/tcip-web/src/tcip_web/routes/results.py` (`@router.get("/models/registered")`, serving `model_tools.rank_registered_models`'s listing view) and the browser's one entry declaration, `packages/tcip-web/frontend/src/api/inference.ts` (`export interface RegisteredModel {`), held field by field against an entry the real registrar wrote by `tests/test_registry_entry_shape_agreement.py`.

## S28. The prediction-bucket record

Must agree: every publisher writes, and every consumer reads, the same record beside a bucket's predictions.
Side A: `packages/tcip-mcp/src/tcip_mcp/buckets.py` (`def publish(`, the one writer, called through `inference_tools.infer` by `run_inference` for an image directory and a raster alike and by the GUI's inference worker, and by `stage_proposals` for a staged proposal).
Side B: `packages/tcip-mcp/src/tcip_mcp/buckets.py` (`def read_bucket(`, the one decoder every reader calls).

## S29. Prediction-bucket immutability

Must agree: no writer writes into a bucket that already exists.
Side A: `packages/tcip-mcp/src/tcip_mcp/buckets.py` (`def publish(`, refusing a name already published inside the transaction that would publish it).
Side B: every writer of a bucket, the inference operation and the proposal staging alike, publishes through it; a raster pass keeps its resume progress under the project, apart from the bucket it will publish.

## S30. split.json train/val partition

Must agree: an assessment's reference is disjoint from the split the producing run actually
trained and selected on.
Side A: `packages/tcip-mcp/src/tcip_mcp/experiments.py` (`def run_resolution(`, the one reader of the partition the launcher resolved and wrote into the run's `run.json`).
Side B: `assessment.py`'s `_disjointness`, a set intersection over group keys and source digests between the reference sides and the run's train and val sides read through it, and the reserved-regions assessment, which takes its regions from the same reader.

## S31. Checkpoint payload structural markers

Must agree: a checkpoint written by the training envelope is rebuildable by the predictor that later loads it.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/model_build.py` (`MODEL_SOURCE_KEY` and `STATE_DICT_KEY`, the one key vocabulary).
Side B: the three checkpoint writers in `pipelines/training/generic_trainer.py` and the readers (`generic_predictor.py`, `inference/predictor.py`, `training/eval_runners.py`) all bind through the constants.

## S32. One execution record for every pass

Must agree: the same model and stated values yield the same conf, cap, tile edge, overlap and merge whichever entry point asks for them, and the record executed is the record stamped.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/execution.py` (`def prepare_pass(`, preparing a record from the one `Stated` mapping a caller hands it or restoring one exactly, refusing a stated value the restored record differs on).
Side B: every pass prepares through it: `tools/inference_tools.py`, `packages/tcip-web/src/tcip_web/routes/inference.py`, `assessment.py` (both assessment kinds, through `_prepared`), and `pipelines/training/eval_runners.py` (the full-frame regime); the bucket record and `assessment.json` both carry `Execution.record()`.

## S33. Shared inference defaults DEFAULT_CONF / DEFAULT_NMS_IOU / DEFAULT_MAX_DETS

Must agree: the MCP entry point and the GUI entry point start from the same unresolved defaults, and both read a caller's unstated parameter off the `None` sentinel rather than off equality with the default, so a caller who states the default value is honored as an override instead of being resolved as if they had stated nothing.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/execution.py` (`DEFAULT_CONF = 0.5`, with `DEFAULT_NMS_IOU`, `DEFAULT_OVERLAP` and `DEFAULT_MAX_DETS` declared beside it).
Side B: `packages/tcip-web/src/tcip_web/routes/inference.py` (`stated=payload.stated`: the GUI launch carries the one `Stated` mapping unresolved, `None` where omitted, and its worker resolves it through `prepare_pass`, which records each value's source, `explicit` or `default`, on the execution record) and `packages/tcip-mcp/src/tcip_mcp/pipelines/training/eval_runners.py` (the tile-level regime resolving its conf and cap through `untiled_execution`).

## S34. One delivery gate behind every delivery path

Must agree: no delivered result ships unvalidated without an explicit acknowledgment, and every delivery's validated column, revision and event id come from one clearance.
Side A: `packages/tcip-mcp/src/tcip_mcp/delivery.py` (`def gate(`, clearing every bucket against the assessment it names: passed, of this delivery's kind, under this revision, its reference unmoved, produced by the bucket's own checkpoint and execution record, covering its capture; refusing no bucket, differing producers, and a detector delivery whose buckets do not count the measured subject).
Side B: the three delivery functions, each calling it once and writing its rows and its one event through `delivery.py` (`def deliver_csv(`): `pipelines/postprocessing/export.py` (`def deliver_per_image_counts_csv(`), `pipelines/postprocessing/aggregation.py` (`def deliver_per_plant_aggregate(`, which the orthomosaic plant-count door also delivers through), and `pipelines/postprocessing/phenology.py` (`def deliver_phenology(`).

## S36. Count-objective vocabulary versus registered pickers

Must agree: every named count objective has a registered picker function.
Side A: `packages/tcip-mcp/src/tcip_mcp/traits.py` (`COUNT_UNBIASED = "count_unbiased"`, with `DETECTION_F1` and `PRESENCE` declared beside it).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/operating_point.py` (`COUNT_OBJECTIVE_PICKERS`, keyed by those three names; an objective with no picker refuses in the count criterion). A picker's provenance label is read off that registry by `pipelines/derivations.py`, so registering a picker registers its label.

## S37. Trait entries against crops.yml controlled vocabulary

Must agree: a trait entry's delivered phenotypes exist in the crops.yml vocabulary.
Side A: `packages/tcip-mcp/src/tcip_mcp/knowledge/__init__.py` (`def crops_yml_path(`, the one placement of `packages/tcip-mcp/src/tcip_mcp/knowledge/crops/crops.yml`), reached through `packages/tcip-mcp/src/tcip_mcp/traits.py` (`def crops_yml_path(`, delegating), read whole or raising by `_crops_traits`, `traits.py`.
Side B: `packages/tcip-mcp/src/tcip_mcp/traits.py` (`def check_proposed_entry(`, a proposal's one check of `delivers` against that read; a stored record decodes without it) and `tools/verify_skill_traits.py` (`load_vocab` checks a skill's trait tokens through that same read).

## S38. Per-project trait records

Must agree: the proposing tool, the confirmation door, the delivery doors and the GUI trait list read one record per trait and agree on which revision a delivery ships under.
Side A: `packages/tcip-mcp/src/tcip_mcp/traits.py` (`def trait_key(`, the one placement, with `TRAITS_STORE`, `traits.py`, the store every reader and writer addresses).
Side B: `packages/tcip-mcp/src/tcip_mcp/traits.py` (`def propose_trait(`, the one append) and `packages/tcip-mcp/src/tcip_mcp/operationalization.py` (`def confirmed_revision(`, the one read of the latest confirmed revision every delivery door makes). `packages/tcip-web/src/tcip_web/routes/results.py` (`def list_traits(`) and `packages/tcip-mcp/src/tcip_mcp/cli/doctor.py` (`def check_traits(`) read the same record through `read_trait`.

## S39. Phenology CSV column vocabulary

Must agree: the delivered CSV's column names derive from the trait spec on every path that writes them.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/phenology.py` (`def phenology_csv_columns(spec) -> list[str]:`, the one owner).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/phenology.py` (`phenology_csv_columns(spec)`, inside `deliver_phenology`, the one writer both `tools/phenology_tools.py`'s `deliver_phenology_milestones` and `packages/tcip-web/src/tcip_web/routes/results.py`'s `export_csv` call through instead of assembling the names themselves).

## S40. Per-band normalization stats for a non-3-channel detector

Must agree: the values passed as image_mean/image_std are per-band stats of the same length as in_chans.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/derivations.py` (`def band_normalization_stats(`).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/components/detectors.py` (`def _normalization(adapter: Any, in_chans: int | None, image_mean, image_std,`).

## S41. model_source bespoke build seam

Must agree: the dict an agent writes into the config carries the keys the builder, the snapshotter, and the predictor all read.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/model_build.py` (`MODEL_SOURCE_KEY = "model_source"`).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/inference/generic_predictor.py` (`self.model_source = ckpt.get("model_source")`).

## S42. training_source bespoke train(ctx) seam

Must agree: a bespoke train(ctx) callable is importable and accepts the TrainContext the envelope hands it.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/training/envelope.py` (`training_source = run.config.get("training_source")`).
Side B: `packages/tcip-mcp/src/tcip_mcp/tools/training_tools.py` (`training_source = normalized.get(TRAINING_SOURCE_KEY)`).

## S43. dataset_source bespoke dataset seam

Must agree: the builder the reader resolves off `data.dataset_source` returns a Dataset the trainer's loaders accept.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/model_build.py` (`dataset_source = (config.get("data") or {}).get(DATASET_SOURCE_KEY)`).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/data/datasets.py` (`def build_dataset(`).

## S44. Model-contract smoke batch versus the trainer's real batch

Must agree: the smoke batch has the same shape the trainer actually feeds model.forward for the task.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/model_contract.py` (`def _synth_batch(`, which synthesizes per-sample `(image, target)` items shaped like a dataset's `__getitem__` and hands them to the trainer's own collate).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/training/collation.py` (`def task_collate(task: str):`, the collate the DataLoader assembles the training batch with).

## S45. Verdict entry encoding

Must agree: the decision the label save records is the decision every reader decodes.
Side A: `packages/tcip-annotation/src/tcip_annotation/verdicts.py` (`def encode_verdict(verdict: Verdict) -> dict:`, the one encoder, called by `record_verdicts` inside the label save).
Side B: `packages/tcip-annotation/src/tcip_annotation/verdicts.py` (`def decode_verdict(entry: Mapping) -> Verdict:`, the one decoder, refusing a malformed entry by name), read by the editor's proposals route and the doctor.

## S46. Frontend api/ layer against backend route paths

Must agree: every URL the browser builds matches a registered FastAPI route path and method.
Side A: `packages/tcip-web/frontend/src/api/routes.ts` (generated: the browser's only copy of the paths, each named for its method).
Side B: `packages/tcip-web/src/tcip_web/routes/__init__.py` (`register_all` mounts each route module's router under its fixed prefix).
The api/ helpers keep their hand-written signatures and reference a generated name; `tools/generate_frontend_routes.py` projects the registered routes into that module, and `tests/test_frontend_route_paths.py` fails when the projection is stale or a call site writes a path of its own.

## S47. GuiState shape between state.py and store/types.ts

Must agree: the snapshot the backend serializes deserializes into the store's typed shape.
Side A: `packages/tcip-mcp/src/tcip_mcp/web_client.py` (`class GuiState(_GuiFields):`).
Side B: `packages/tcip-web/frontend/src/api/types.generated.ts` (`export interface GuiState {`), generated from Side A by `tools/generate_frontend_types.py` and re-exported by `store/types.ts`.

## S48. State WebSocket snapshot protocol

Must agree: the browser knows which slices of a broadcast snapshot are backend-authoritative and orders them by version.
Side A: `packages/tcip-web/src/tcip_web/app.py` (`@app.websocket("/ws/state")`).
Side B: `packages/tcip-web/src/tcip_web/state.py` (`def version(self) -> int:`, "Monotonic version, bumped on every state change.").

## S49. Terminal PTY WebSocket protocol

Must agree: control-message type names and field names match, and output frames are treated as raw text rather than JSON.
Side A: `packages/tcip-web/src/tcip_web/routes/terminal.py` (`@router.websocket("/ws/{session_id}")`).
Side B: `packages/tcip-web/frontend/src/components/TerminalRail.tsx` (`send({ type: "input", data });`).

## S50. Inference job stream WebSocket

Must agree: the browser recognizes the terminal frame and the status vocabulary the backend uses.
Side A: `packages/tcip-web/src/tcip_web/routes/inference.py` (`@router.websocket("/jobs/{job_id}/stream")`).
Side B: `packages/tcip-mcp/src/tcip_mcp/experiments.py` (`TERMINAL_STATES = frozenset({*FINAL_STATES, "interrupted"})`, the one declaration the job registry and the generated frontend vocabulary both read).

## S51. Training run stream WebSocket

Must agree: the epoch rows and the terminal row the stream sends are the ones the listing and the MCP tools read.
Side A: `packages/tcip-web/src/tcip_web/routes/training.py` (`@router.websocket("/runs/{experiment_id}/stream")`, each epoch sent whole and the terminal frame carrying the run's `RunRow`).
Side B: `packages/tcip-mcp/src/tcip_mcp/experiments.py` (`epoch_rows`, the one accumulation of an epoch's rows, and `run_summary`, the one `RunRow`).

## S52. Image-serve response headers

Must agree: header names and value encodings match.
Side A: `packages/tcip-web/src/tcip_web/routes/images.py` (`extra = {"X-TCIP-Served-Size": f"{out_w}x{out_h}"}`, the one render header besides the cache's own).
Side B: `packages/tcip-web/frontend/src/lib/imageLoader.ts` (`servedSize: parseServedSize(headers.get("X-TCIP-Served-Size")),`).

## S53. Optimistic-concurrency token for label saves

Must agree: the token the browser echoes is the same token the backend minted for that label document.
Side A: `packages/tcip-web/src/tcip_web/routes/annotate.py` (`"completion": _completion(doc), "base_mtime": version.token}`, the record version the load route answers; the save door compares the echoed one inside its commit in `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py` (`if expect is not None and expect != read.version:`)).
Side B: `packages/tcip-web/frontend/src/tabs/AnnotateTab.tsx` (`base_mtime: paths.mtime,`).

## S54. Built frontend bundle location

Must agree: the directory Vite writes is one of the directories the backend looks in.
Side A: `packages/tcip-web/frontend/vite.config.ts` (`outDir: "../static",`).
Side B: `packages/tcip-web/src/tcip_web/app.py` (`def _find_static_dir() -> Path:`).

## S55. Vite dev-server proxy prefixes

Must agree: every backend path the browser calls in dev falls under a proxied prefix.
Side A: `packages/tcip-web/frontend/vite.config.ts` (`proxy: {`).
Side B: `packages/tcip-web/src/tcip_web/app.py` (`@app.websocket("/ws/state")`, one of the endpoints not under the `/api` prefix).
The prefix literals stand on their own; `tests/test_frontend_route_paths.py` fails when a path the frontend references falls outside them, sockets under the API prefix included.

## S56. Tab-name vocabulary

Must agree: the tab a panel event targets, the tab the browser can restore, and the tab the backend persists are the same set of names.
Side A: `packages/tcip-mcp/src/tcip_mcp/web_client.py` (`ActiveTab = Literal["setup", "annotate", "training", "inference", "results", "meta"]`, with `TAB_NAMES = get_args(ActiveTab)` beside it, `tcip_web.state` importing both).
Side B: `packages/tcip-web/frontend/src/api/types.generated.ts` (`export const TAB_NAMES = [`, generated from the same declaration).

## S57. One matcher for the assessment and the editor

Must agree: the pairs the editor shows a person and the pairs the assessment counts come from one matcher under one crowd rule.
Side A: `packages/tcip-annotation/src/tcip_annotation/matching.py` (`def pair_detections(gt: list[dict], dt: list[dict],`, the one matcher, COCO's center-in-crowd rule included).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/training/evaluation.py` (`return [class_matchings(rec["gt"], rec["dt"], criterion, conf=conf, policy=policy, held=held)`) and `packages/tcip-annotation/src/tcip_annotation/matching.py` (`m = pair_detections([gt[i] for i in gi], [dt[i] for i in di], criterion, policy=policy)`, the editor's pairing through `dataset_layout.proposal_pairs` under the bucket's assessment's criterion, and the single-image scoring and its comparison render), held by `tests/test_label_document_gestures.py`'s agreement tests.

## S58. Reference-grid geometry

Must agree: the cell name the agent points at and the cell the GUI highlights are the same rectangle.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/reference_grid.py` (`def reference_cells(`, which builds the cells, with `grid_geometry`, `reference_grid.py`, the geometry handed over beside them).
Side B: `packages/tcip-annotation/src/tcip_annotation/grid.py` (`def grid_to_rect(cell: str, cells: "list[Any]") -> tuple[float, float, float, float]:`, the one cell-name lookup) and `packages/tcip-web/src/tcip_web/routes/images.py` (`@router.get("/serving_grid")`, whose cell list the browser's region serving consumes verbatim).

## S59. Path confinement (the derived allow-set)

Must agree: every route that accepts a client-supplied path confines it to the same allowed roots.
Side A: `packages/tcip-web/src/tcip_web/paths.py`
(`def allowed_roots() -> list[Path]:`).
Side B: `packages/tcip-web/src/tcip_web/paths.py` (`p = allowed_path(path)`, the one adapter every route shares).

## S64. MCP tool registry against documented tool names

Must agree: any document naming a tool names one the server actually registers.
Side A: `packages/tcip-mcp/src/tcip_mcp/server.py` (`def list_registered_tools() -> list[str]:`).
Side B: `tools/list_tools.py` (`from tcip_mcp.server import list_registered_tools`).

## S65. MCP client launch configuration

Must agree: the environment name in the client config, the docs, and the environment file match.
Side A: `.mcp.json` (launches `conda run -n tcip-agent python -m tcip_mcp`).
Side B: `environment.yml` (`name: tcip-agent`).

## S66. Skill and docstring examples against real signatures

Must agree: a documented call binds against the real function signature.
Side A: `packages/tcip-mcp/src/tcip_mcp/knowledge/` (python fenced examples in the knowledge documents).
Side B: none; no check verifies a knowledge document's fenced example against the real
signature it calls.

## S67. Local gate commands against the CI gate

Must agree: the checks a contributor runs locally are the checks CI runs.
Side A: `CLAUDE.md` (documents `pytest -n 4`, `ruff`, `mypy`, and the frontend command chain; the docker job is CI-only by design, with no local counterpart).
Side B: `.github/workflows/ci.yml` (mypy job, python job with `pytest -n auto` and `TCIP_MIN_TESTS`, typescript job with format:check/lint/typecheck/test/build, docker job building `packages/tcip-web/Dockerfile` and polling the served GUI).
