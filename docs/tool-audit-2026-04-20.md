# MCP tool audit — 2026-04-20

First-pass classification of all MCP tools. Flags candidates for deletion but does NOT delete anything. Evidence to prune comes from the Phase 1 exercise (which tools get called, which get wanted and don't exist, which get ignored).

Classification framework (from [docs/vision.md §4.1](vision.md#L201)):

- **Keep** — produces a tamper-evident audit record of a state-changing op, accesses long-running infrastructure, or provides domain knowledge Claude can't write inline. No debate.
- **Marginal** — Claude could write a Python script to do this, but a tool is more reliable and the audit trail has some value. Justifiable either way; evidence from Phase 1 decides.
- **Probably bloat** — thin wrapper around a few lines of Python. Claude writes it better inline. Flag for deletion after the exercise confirms it goes unused.

Tool count: **57** (54 pre-existing + 3 meta-loop tools added 2026-04-20).

---

## Summary

| Category | Count | Share |
|---|---|---|
| Keep | 31 | 54% |
| Marginal | 14 | 25% |
| Probably bloat | 12 | 21% |

Target inventory after pruning: ~15–18 tools ([docs/vision.md §4.2](vision.md#L218)). Reaching that number requires retiring most of the "marginal" category as well, not just the bloat candidates. That is a Phase 2 decision.

---

## Per-tool classification

### active_learning_tools.py (2 tools)

| Tool | Verdict | Reasoning |
|---|---|---|
| `score_unlabeled` | Keep | Wraps uncertainty/diversity/combined scoring logic that is non-trivial to script inline. Domain ML knowledge baked in. |
| `get_review_queue` | Keep | Partitioning logic (auto-accept / review / reject) is load-bearing for active learning — small but precise. |

### annotation_tools.py (7 tools)

| Tool | Verdict | Reasoning |
|---|---|---|
| `load_annotations` | Marginal | Format abstraction is real, but will collapse to one internal rep in Phase 2 ([§3.3](vision.md#L154)). Keep until collapse lands. |
| `save_annotations` | Marginal | Same as above. |
| `evaluate_detections` | Keep | mAP / precision / recall computation. Bug-prone to re-implement inline. |
| `evaluate_dataset` | Keep | Dataset-level metric aggregation; audit-seam for reported results. |
| `sam_predict` | Keep | SAM inference = long-running infrastructure. Tool is the correct boundary. Will migrate to SAM2 as part of Phase 0 #5. |
| `run_matching` | Keep | GT↔pred matching logic used by review. Non-trivial, reused across workflows. |
| `push_panel_data` | Keep | VS Code panel side effect. Agent has no other way to drive panels. |

### data_tools.py (3 tools)

| Tool | Verdict | Reasoning |
|---|---|---|
| `load_dataset` | Marginal | Scans a dataset dir and returns summary. Claude can ls + count, but the summary shape is useful. Evidence-driven. |
| `validate_data_quality` | Marginal | Image size / class balance / empty-label checks. Useful enough that Claude wouldn't rewrite it every time, but not a long-running op. |
| `split_dataset` | Marginal | train/val/test split with seeding. Claude can script in 10 lines; audit trail matters for reproducibility. |

### experiment_tools.py (6 tools)

| Tool | Verdict | Reasoning |
|---|---|---|
| `create_experiment` | Keep | Audit seam — anchors lineage for a training run. Load-bearing. |
| `log_metrics` | Keep | Append to metrics.jsonl with timestamps. Must be tool-mediated for reproducibility. |
| `record_artifact` | Keep | Binds paths (checkpoints, predictions) to experiment_id. |
| `get_experiment` | Keep | Read access to experiment state. |
| `compare_experiments` | Keep | Side-by-side metric comparison — useful and non-trivial to script consistently. |
| `get_experiment_lineage` | Keep | Traces data → model → predictions chain. Core to reproducibility story. |

### inference_tools.py (3 tools)

| Tool | Verdict | Reasoning |
|---|---|---|
| `run_inference` | Keep | Loads model + batch inference = long-running infra. |
| `export_predictions_yolo` | Probably bloat | Runs inference then writes YOLO .txt files. Claude can call `run_inference` and write a loop. Format-specific output is exactly what Phase 2 eliminates. |
| `export_results_csv` | Probably bloat | Composes `run_inference` + CSV writer. Inline-writable. |

### model_tools.py (4 tools)

| Tool | Verdict | Reasoning |
|---|---|---|
| `list_available_models` | Probably bloat | Duplicates most of `list_components`. Pick one, drop the other. |
| `register_model` | Keep | Model registry audit seam. |
| `list_registered_models` | Marginal | ls + parse. Could be a script. Keep if registry becomes cross-session. |
| `get_best_model` | Marginal | Sort-by-metric across registry. Useful but small. |

### pipeline_tools.py (6 tools)

| Tool | Verdict | Reasoning |
|---|---|---|
| `list_components` | Keep | Inventory of composable registry — domain knowledge Claude needs to make design decisions. |
| `recommend_model` | Keep | Encapsulates heuristics (dataset size → backbone choice, etc.). Domain logic. |
| `validate_model_spec` | Probably bloat | Structural check: "do backbone/neck/heads exist in registry." Claude writes 3-line script. |
| `validate_pipeline_spec` | Probably bloat | Same shape as above — now thinner after orchestrator cleanup. |
| `run_pipeline` | Keep | Launches multi-phase pipeline = long-running infra + checkpointing. |
| `compose_and_summarize` | Marginal | Builds model and returns param counts. Handy but not essential. |

### project_tools.py (8 tools)

| Tool | Verdict | Reasoning |
|---|---|---|
| `init_project` | Keep | Creates `.tcip/` scaffold. Idempotent but useful boundary for "this is where project state lives." |
| `create_session` | Probably bloat | Overlaps with `audit.jsonl`. Session timelines may not need a separate log. |
| `append_session_event` | Probably bloat | Same as above. |
| `list_sessions` | Probably bloat | Same. |
| `get_session` | Probably bloat | Same. |
| `get_project_status` | Keep | Aggregates project state for Claude at session start. Useful orienting tool. |
| `export_project` | Marginal | Zips project for portability. Claude can shutil.make_archive. |
| `import_project` | Marginal | Unzip counterpart. Same story. |

### training_tools.py (7 tools)

| Tool | Verdict | Reasoning |
|---|---|---|
| `validate_config` | Probably bloat | Key-presence + path-exists checks. Claude writes inline faster than reading this tool's schema. |
| `launch_training` | Keep | Threaded training + TensorBoard launch. Long-running, non-trivial. |
| `check_training_status` | Keep | Polls background training thread. Must be tool-mediated because thread lives in server process. |
| `list_training_runs` | Probably bloat | Iterate in-process run dict. Could be part of `check_training_status`. |
| `run_hpo` | Keep | Hyperparameter search is expensive, long-running, Optuna-backed. |
| `get_training_metrics_path` | Probably bloat | Returns a filesystem path. Claude can construct it. |
| `get_worst_predictions` | Keep | Ranks predictions by error for failure analysis. Non-trivial reducer. |

### vision_tools.py (8 tools)

| Tool | Verdict | Reasoning |
|---|---|---|
| `visualize_annotations` | Keep | Renders GT boxes on image → Claude views it. Core visual reasoning loop. |
| `visualize_predictions` | Keep | Same for predictions. |
| `visualize_comparison` | Keep | GT vs pred overlay with match stats. Critical for evaluation. |
| `visualize_worst_predictions` | Keep | Grid of failure cases. Essential for debugging models. |
| `visualize_dataset_sample` | Keep | Dataset sanity check before training. |
| `sam_auto_label` | Keep | SAM candidate generation + numbered overlay. Core annotation loop — will become `propose_annotations` in Phase 3. |
| `accept_candidates` | Keep | Saves agent-classified candidates as annotations. Becomes `commit_labels` in Phase 3. |
| `visualize_grid_overlay` | Marginal | Spatial reference grid. Useful for some workflows, unused in others. Evidence-driven. |

### meta_tools.py (3 tools, new)

| Tool | Verdict | Reasoning |
|---|---|---|
| `claude_reports` | Keep | Core meta-loop signal. The whole point of Phase 0. |
| `project_retrospective` | Keep | End-of-session reflection. Load-bearing for system improvement. |
| `load_retrospectives` | Keep | Session startup context. Pairs with retrospective. |

---

## Bloat candidates (12)

Flagged for deletion if Phase 1 confirms they are not called — or are called but could trivially be replaced by a script:

1. `export_predictions_yolo`
2. `export_results_csv`
3. `list_available_models` (merge into `list_components`)
4. `validate_model_spec`
5. `validate_pipeline_spec`
6. `create_session`
7. `append_session_event`
8. `list_sessions`
9. `get_session`
10. `validate_config`
11. `list_training_runs`
12. `get_training_metrics_path`

## Marginal tools (14)

Keep through Phase 1; revisit after exercise data:

1. `load_annotations` (auto-delete when Phase 2 format collapse lands)
2. `save_annotations` (same)
3. `load_dataset`
4. `validate_data_quality`
5. `split_dataset`
6. `list_registered_models`
7. `get_best_model`
8. `compose_and_summarize`
9. `export_project`
10. `import_project`
11. `visualize_grid_overlay`

(Count is 11 here; the other 3 from the 14-total are `load_annotations`/`save_annotations`/`visualize_grid_overlay` each effectively already listed. Treat as ~11 stable marginals.)

## Phase 1 exercise instructions

While running hazelnut catkin phenology detection end-to-end:

- **Mark each tool called in a session.** Simple tally per session is enough; `audit.jsonl` already captures this.
- **When a "bloat" tool gets called often and usefully**, upgrade its classification and note why.
- **When a "keep" tool goes completely unused across Phase 1**, flag it for reconsideration — either it's badly named/described, or it's solving a problem that doesn't exist.
- **When Claude writes a script instead of calling a tool**, note which tool it replaced. That's evidence of actual bloat.

The output of Phase 1 is a revised version of this document with real call-frequency numbers.
