# Phase 5 — Full Loop

## Goal
End-to-end two-layer pipeline on hazelnut catkin phenology:
annotate → train isolation model → train catkin detector+classifier →
temporal aggregation → per-plant phenology dates → export CSV.

This phase integrates everything from Phases 1–4 into a cohesive workflow, adds the
training dashboard, and validates that the entire system works on real data.

### Target traits (all hazelnut, all phenology)
- `catkin_05per_date` — date when 5% of the bush's catkins have elongated
- `catkin_50per_date` — date when 50% have elongated
- `catkin_95per_date` — date when 95% have elongated
- `catkin_elongation_date` — date when most catkins have elongated

### Pipeline structure (two-layer + temporal)
1. **Isolation**: detect/segment individual hazelnut bush in ground-level RGB image
2. **ML task**: detect catkins within bush region, classify as elongated vs non-elongated
3. **Temporal aggregation**: per plant across multi-date imagery, compute elongation
   percentage over time → fit sigmoid → extract threshold dates

## What's new in Phase 5

### Training dashboard (center panel mode)
Real-time visualization of training progress:

```
┌────────────────────────────────────────────────────┐
│  Training: hazelnut_catkin_det_v3       [⏸] [⏹]   │
├────────────────────────────────────────────────────┤
│                                                    │
│  Stage 3/5: Unfreeze layer3+            ████░ 72%  │
│  Epoch 42/60  |  LR: 0.0003                       │
│                                                    │
│  ┌──────────────────────────┐ ┌──────────────────┐ │
│  │ Loss                     │ │ mAP@50           │ │
│  │  ╲                       │ │        ╱──────   │ │
│  │   ╲                      │ │      ╱           │ │
│  │    ╲___                  │ │    ╱              │ │
│  │        ╲___╱─            │ │  ╱               │ │
│  │              ╲─          │ │╱                  │ │
│  │       train ── val ──    │ │ 0.72              │ │
│  └──────────────────────────┘ └──────────────────┘ │
│                                                    │
│  Best checkpoint: epoch 38  mAP@50: 0.74           │
│  ETA: ~14 min remaining                            │
│                                                    │
│  ┌─ HPO Trials (if running) ──────────────────┐   │
│  │ Trial 1: lr=0.001, wd=0.01    mAP=0.71  ✓ │   │
│  │ Trial 2: lr=0.003, wd=0.005   mAP=0.74  ✓ │   │
│  │ Trial 3: lr=0.0005, wd=0.01   running... ● │   │
│  │ Trial 4: lr=0.002, wd=0.02    pruned    ✗ │   │
│  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
```

### Training data source
- GUI reads metrics from a file the MCP server writes (TensorBoard log or JSON lines)
- Agent polls `check_training_status` → gets text summary for chat
- GUI reads the same underlying data → renders charts independently
- No coupling between agent's view and GUI's view — both read the same source

### Results review panel
After training completes, agent triggers evaluation and presents results:

```
┌────────────────────────────────────────────────────┐
│  Results: hazelnut_catkin_det_v3                    │
├────────────────────────────────────────────────────┤
│                                                    │
│  Overall:  mAP@50: 0.74  |  mAP@50-95: 0.52      │
│            Precision: 0.81  |  Recall: 0.78        │
│                                                    │
│  Per-class:                                        │
│  ┌────────────────┬────────┬──────┬────────┐      │
│  │ Class          │ AP@50  │ Prec │ Recall │      │
│  ├────────────────┼────────┼──────┼────────┤      │
│  │ elongated      │ 0.71   │ 0.78 │ 0.73   │      │
│  │ non-elongated  │ 0.68   │ 0.75 │ 0.70   │      │
│  └────────────────┴────────┴──────┴────────┘      │
│                                                    │
│  Worst predictions:                                │
│  [img1.jpg] [img2.jpg] [img3.jpg] ← clickable     │
│                                                    │
│  ┌─ ⚠ Checkpoint: Accept Results ──────────────┐  │
│  │ Accept this model?                           │  │
│  │ [Accept & Deploy] [Retrain] [Retrain + HPO]  │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
```

### CSV export
The final deliverable — per-plant phenology dates derived from temporal aggregation:

```csv
plant_id,catkin_05per_date,catkin_50per_date,catkin_95per_date,catkin_elongation_date,sigmoid_r2,n_obs_dates
HAZ-001,2026-02-28,2026-03-12,2026-03-24,2026-03-15,0.95,11
HAZ-002,2026-03-02,2026-03-14,2026-03-28,2026-03-17,0.93,10
HAZ-003,2026-02-25,2026-03-08,2026-03-19,2026-03-11,0.97,12
...
```

Agent calls `run_inference` on all images across all dates → counts elongated/total
catkins per plant per date → fits sigmoid → extracts threshold dates → writes CSV.

## The full workflow (hazelnut catkin phenology)

### Act 1: Pipeline Design (PipelineDesigner subagent, Opus)
```
User: "I want to automate catkin phenology for hazelnut: catkin_05per_date,
       catkin_50per_date, catkin_95per_date, catkin_elongation_date."

Agent: [switches to PipelineDesigner mode]
Agent: [calls get_trait_info("hazelnut", "catkin_05per_date")]
Agent: "These are phenology traits in the ground_rgb_object_detection pipeline
        group. I'll design a two-layer pipeline:

        Layer 1 (Isolation): Bush detection — detect individual hazelnut bushes
        in each ground-level RGB image using Faster R-CNN.

        Layer 2 (ML task): Catkin detection + classification — detect catkins
        within each bush region, classify each as 'elongated' or 'non-elongated'
        (two-class object detection).

        Temporal aggregation: Per plant across imaging dates, compute the
        percentage of elongated catkins. Fit a sigmoid curve to the time series.
        Extract the dates at 5%, 50%, and 95% thresholds.

        Let me check your data."

Agent: [calls load_dataset with user's image directory]
Agent: "You have 847 images across 12 imaging dates spanning February–April.
        42 plants, ~20 images per plant per date. Some images already have
        catkin annotations. No bush annotations yet."
       
→ HITL Checkpoint #1: Pipeline design approval
  [Shows proposed two-layer pipeline config]
  User: [Approve]
```

### Act 2: Isolation Model (TrainingOrchestrator subagent)
```
Agent: "We need a bush detection model first. Do you have bush annotations,
        or should I generate predictions using a general detector?"
User: "I have a few annotated, use those to train a quick model."

Agent: [calls load_dataset for bush annotations → 50 images annotated]
Agent: "50 images with bush bounding boxes. I'll train a quick Faster R-CNN
        to bootstrap predictions for the rest."

→ HITL Checkpoint #2: Pipeline config review
  [Shows bush detection training config: FasterRCNN ResNet50, 50 images,
   progressive unfreezing 3 stages, short run]
  User: [Approve]

Agent: [calls launch_training → run_id: bush_det_v1]

→ Training dashboard shows progress
→ Training completes: mAP@50=0.91 on bush detection (bushes are easy)

Agent: "Bush detector trained (mAP 0.91). Now I'll run inference on all
        847 images to crop bush regions for catkin annotation."
Agent: [calls run_inference with bush model on all images]
Agent: "847 bush crops extracted. Ready for catkin annotation."
```

### Act 3: Catkin Annotation + Model
```
Agent: [switches to CodeGenerator mode for data prep]
Agent: "You have some existing catkin annotations. Let me assess quality."
Agent: [calls validate_data_quality]
Agent: "312 images have catkin annotations (2841 catkin boxes). But they're
        not classified as elongated/non-elongated — they're all one class.
        We'll need to either re-annotate with two classes, or detect catkins
        first and then add classification."
User: "Let's detect catkins first, then I'll classify them."

→ Agent trains catkin detector (one class) on 312 annotated images
→ HITL Checkpoint #3: Training launch
→ Training completes: mAP@50=0.74

Agent: "Catkin detector trained. Now I'll run inference on all bush crops
        and push predictions to canvas for you to review and classify."
Agent: [calls run_inference on all bush crops]
Agent: [calls canvas_control({action: "show_predictions", ...})]

→ User reviews predictions in annotation canvas
→ User classifies each detected catkin as elongated/non-elongated
→ canvas.batch_complete sent to agent

Agent: "Classifications complete. Now I'll retrain with two-class labels."

→ HITL Checkpoint #3: Training launch (two-class detector)
→ Training completes: mAP@50=0.71 (elongated), 0.68 (non-elongated)

→ HITL Checkpoint #4: Results review
  [Shows per-class metrics, worst predictions for review]
  User: [Accept]
```

### Act 4: Temporal Aggregation
```
Agent: [switches to ResultsAnalyzer mode]
Agent: "Running two-class catkin detector on all 847 images across all dates."
Agent: [calls run_inference on full dataset]
Agent: "For each plant and date, I now have counts of elongated vs
        non-elongated catkins. Computing elongation percentage time series
        and fitting sigmoid curves."
Agent: [calls export_results with temporal_aggregation config]

Agent: "Sigmoid fitting results:
        - 38/42 plants have good fits (R² > 0.90)
        - 4 plants have sparse data (fewer than 5 imaging dates)
        
        Example for plant HAZ-001:
        catkin_05per_date: 2026-02-28
        catkin_50per_date: 2026-03-12
        catkin_95per_date: 2026-03-24
        catkin_elongation_date: 2026-03-15"
```

### Act 5: Export + Deploy
```
→ HITL Checkpoint #5: Model deployment
  [Shows CSV preview + model info + sigmoid fit quality]
  User: [Accept]

Agent: [calls register_model for bush detector + catkin detector]
Agent: [calls export_results → final CSV]

Agent: "Done. Models registered:
        - hazelnut_bush_det_v1 (mAP@50: 0.91)
        - hazelnut_catkin_det_v1 (mAP@50: 0.71/0.68)
        
        CSV exported to output/hazelnut_catkin_phenology.csv
        (42 plants × 4 traits = 168 data points)"
```

### Final CSV output
```csv
plant_id,catkin_05per_date,catkin_50per_date,catkin_95per_date,catkin_elongation_date,sigmoid_r2,n_obs_dates,model_version
HAZ-001,2026-02-28,2026-03-12,2026-03-24,2026-03-15,0.95,11,hazelnut_catkin_det_v1
HAZ-002,2026-03-02,2026-03-14,2026-03-28,2026-03-17,0.93,10,hazelnut_catkin_det_v1
HAZ-003,2026-02-25,2026-03-08,2026-03-19,2026-03-11,0.97,12,hazelnut_catkin_det_v1
...
```

## Components needed for Phase 5

### New panels
| Component | Purpose |
|-----------|---------|
| `training_dashboard.py` | Live training metrics + HPO trial table |
| `results_panel.py` | Post-training evaluation display |

### New widgets
| Component | Purpose |
|-----------|---------|
| `metric_chart.py` | Line chart for loss/mAP (matplotlib or pyqtgraph) |
| `trial_table.py` | HPO trial status table |
| `csv_preview.py` | Preview exported CSV in a table view |

### MCP server additions
| Tool | Purpose |
|------|---------|
| `get_training_metrics_path` | Return path to live metrics file for GUI to read |
| `get_worst_predictions` | Return N images with highest loss / lowest confidence |

### Bridge protocol additions
| Method | Direction | Purpose |
|--------|-----------|---------|
| `training.started` | Agent → GUI | Switch center panel to training dashboard |
| `training.metrics_update` | Agent → GUI | Trigger dashboard refresh |
| `training.complete` | Agent → GUI | Training finished, show results |
| `results.show` | Agent → GUI | Switch to results panel with evaluation data |

## Integration testing

Beyond the per-phase test criteria, Phase 5 adds end-to-end tests:

### Smoke test (manual, real data)
1. Launch app
2. Type "Automate catkin phenology for hazelnut"
3. Agent proposes pipeline → approve
4. Agent loads data → validates → suggests annotation review
5. Annotate/review 10 images in canvas
6. Agent splits data → proposes training config → approve
7. Training runs → dashboard shows progress
8. Training completes → results displayed → accept
9. Export CSV → verify correct per-plant counts
10. Total wall-clock time logged

### Automated integration test (mock API + test data)
1. Mock Anthropic service returns scripted tool calls
2. Small test dataset (20 images, pre-annotated)
3. Training runs for 2 epochs (fast)
4. CSV exported and validated against expected phenology dates
5. All 5 HITL checkpoints fire and resolve correctly

## Package additions (from Phase 4)

```
tcip-gui/src/tcip_gui/
├── panels/
│   ├── training_dashboard.py       # NEW
│   └── results_panel.py            # NEW
├── widgets/
│   ├── metric_chart.py             # NEW
│   ├── trial_table.py              # NEW
│   └── csv_preview.py              # NEW
```

## Success criteria (MVP complete when)

1. Two-layer pipeline: bush isolation → catkin detection+classification → temporal aggregation → phenology CSV
2. All 5 HITL checkpoints fire correctly with correct payloads
3. Agent uses subagents contextually (PipelineDesigner for Act 1, TrainingOrchestrator for Acts 2-3, ResultsAnalyzer for Act 4)
4. Training dashboard shows real-time progress for both isolation and catkin models
5. Annotation canvas supports prediction review and class assignment (elongated/non-elongated)
6. Temporal aggregation produces sigmoid fits with R² > 0.85 for ≥80% of plants
7. CSV output contains all 4 phenology traits per plant with quality metadata (sigmoid_r2, n_obs_dates)
8. System recovers from agent crash mid-conversation (session reload from project state)
9. System recovers from MCP server crash (degraded mode + restart)
10. Total API cost for full workflow is tracked and displayed
11. A colleague (non-developer) can follow the workflow with minimal guidance
