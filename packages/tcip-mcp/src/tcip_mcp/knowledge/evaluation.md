---
name: evaluation
description: "Model evaluation methods, metrics interpretation, failure triage, worst-case analysis, and experiment comparison for ML models. Load when interpreting evaluation metrics, triaging or diagnosing model failures, inspecting worst predictions, or comparing experiments or checkpoints."
---

# Model Evaluation

## Metrics by Task Type

| Task | Comparability metric (labeled) | Other metrics |
|------|---------------|-------------------|
| Detection | mAP@50 | mAP@50:95, precision, recall |
| Instance Segmentation | mask mAP@50 | box mAP, mask mAP@50:95, precision, recall |
| Classification | Accuracy | F1 (macro), per-class precision/recall |
| Regression | RMSE | R², MAE, concordance correlation coefficient (a selectable regression operating-point criterion, see `operating_point`) |
| Ordinal | Quadratic weighted κ | MAE, rank accuracy |

These are labeled comparability metrics: a fixed-convention number (mAP@50 = AP at IoU 0.5) that
lets runs be compared on the same ruler. They do not govern the phenotype. The criterion that
governs the delivered measurement (which detections are a hit, what the count is) is the *trait's*
localization criterion with a tolerance derived from the data in hand, e.g. a center-match with
`half_class_avg_size` tolerance for small, thin objects like catkins, not a frozen IoU@0.5. Choose the governing
criterion per trait/data; keep mAP@50 alongside only as the comparability label (see `operating_point`
/ the derive-don't-pin rail). There is no single mandated "primary" metric per task.

Detection/instance-seg metrics (`coco_detection_metrics`) are aggregate only; no per-class AP
today. Per-class precision/recall/F1 is real for classification (`evaluate_model`); ordinal and
regression get only the scalar metrics in the table above, no per-class breakdown. Change
detection is not a built task type; see README's Roadmap.

R² and CCC ask related but distinct questions of a regression trait, and are not directly
comparable numbers: R² is "how much better than trivially predicting this set's own mean" (overall
predictive skill), unbounded below; CCC is bounded in [-1, 1] and explicitly decomposes into
correlation (precision) and a separate bias/scale term (accuracy), the more standard lens in
measurement-agreement/method-comparison contexts specifically because of that decomposition.

## Tools

| Tool | Purpose |
|------|---------|
| `evaluate_model` | Evaluate a checkpoint on a held-out dataset, or a named split manifest's `calibration` side (`split_manifest_dir`); writes `test_results.json` |
| `annotation_tools.score_predictions` (library call) / `scripts/score_predictions.py` (logged script) | Score on-disk predictions vs GT: an image file returns per-box matches (`detail=True` adds a per-detection breakdown); a dataset dir returns aggregate metrics + per-image TP/FP/FN. On a classified bucket this scores the object's localization, never the classifier's own confirmed-state call |
| `scripts/render_failure_cases.py` (logged script, run with python) | Surface + render the N images with highest triage error |
| `experiment_tools.compare_experiments` (library call) | Side-by-side metrics across experiments |
| `get_experiment` (`view='lineage'`) | Trace data → model → predictions chain |
| `list_experiments` | Enumerate every experiment on record, including one no other tool can rediscover (a calibration experiment, a pre-created one never launched) |
| `rank_registered_models` | Rank registered models by a stated metric, direction and verification status |

`evaluate_model` accepts an optional `trait=`: when set, the trait's own governing criterion
(not the IoU@0.5 comparability convention) determines detection counts/F1, matching what governs
delivery (see Metrics by Task Type above); omit it and the IoU@0.5 convention governs instead. For
a tile-trained checkpoint, `evaluate_model` reports in one of two regimes: the default tile-level
run is a diagnostic only (matches training-time val mAP, not the shipped full-frame count);
`use_tiled_inference=True` reconstructs predictions to full frame and is the delivery-grade metric
to report for gating. An untiled checkpoint has no regime split; its one run already is the
delivery metric (see `evaluate_model`'s own docstring for the full precedence). Either a run id or
a bare checkpoint path resolves to a file that must be registered under the process's platform
state root (`register_model`, explicit mode for a foreign or bespoke checkpoint); `evaluate_model`
refuses
before loading an unregistered one.

`rank_registered_models` requires a `metric` (no default) and resolves its ranking direction from
`evaluation.HIGHER_IS_BETTER_BY_METRIC` (keyed by the metric with any `val_` prefix stripped);
`higher_is_better` overrides the declaration when a caller states one, required for a metric the
declaration does not name. It ranks only `metrics_source="trainer"` entries by default (the
platform's own `default_train` measured them); `include_unverified=True` also ranks
`"training_source"`/`"caller"` entries, whose numbers were never measured by the platform, and
`excluded_unverified` in the response names what a default call left out.

## Calibration/Holdout Split

A validated operating point calibrates against a locked calibration/holdout split
(`resolve_locked_cal_holdout_split`): the split draws once, on first use, and every later call
reuses it, so the delivery gate can't silently pass by drawing a different, weaker holdout after
the fact. The lock lives under the dataset root of the labels or records the split was drawn over,
so it travels with that data and survives adopting a project mid-session. Redrawing a locked split
is a real, audited decision, never automatic:
`redraw_calibration_holdout(dataset_root=..., labels_dir=..., reason=...)` is the tool for it,
and `dataset_root` is that same root, so the redraw replaces the lock the calibration reads. `reason` is
required and non-empty; every redraw is appended to the lock's `redraw_history` with its policy,
seed and the old/new split's content hashes (not the stems themselves); the old and new split
membership is recorded in a dataset's audit log alongside the reason, so a redraw-until-it-passes
pattern stays visible on review.

`run_inference` and `deliver_per_image_counts` (whose live regime forwards it to the shared
verified pass), `redraw_calibration_holdout` and `evaluate_model` all take
`split_manifest_dir`: draw the calibration universe from one capture date's `calibration` side of
a named `split_manifest` record instead of every labelled stem with an image, a side
`draw_splits` drew held out from both training and checkpoint selection (see the `training`
skill's Dataset Splits section). `evaluate_model` is the one whose purpose is a held-out score:
without `split_manifest_dir` it scores the whole directory, as today; with it, the loader's own
admitted count is recorded as `evaluated_stem_count`, refused by name when it falls short of the
universe the manifest drew. The manifest's own subject/attribute must match this call's
(`redraw_calibration_holdout` takes `subject`/`attribute` directly; `run_inference` and
`evaluate_model` resolve them from the run's own training scope), and `split_manifest_dir`
conflicts with an explicit `group_by`/`group_key_map`, whose default becomes `None` for this
reason (resolved to `tile_prefix` when neither was given). `redraw_calibration_holdout`
additionally requires `labels_dir`, `subject` and `images_dir` alongside `split_manifest_dir`: it
refuses by name without one, since a labels-only universe can include a stem whose image is gone.

A calibration under a named manifest also earns a `selection_disjointness` check: whether the
cal/holdout stems it drew also sit on the checkpoint being calibrated's own selection (`val`)
side, the leak this whole family of checks closes (a checkpoint chosen on a side, then
calibrated over that same side, would otherwise clear every other gate while measuring the
operating point on exactly the data the shipped weights were picked to fit). It rides beside
`train_disjointness` in the validation row and floors `verify_stamp_binding` when a manifest-scoped
reference carries none.

For a run bound to a split manifest, the same check also names a calibration label that moved
since the split was drawn: `labels_moved_draw_to_run` (a stem whose digest at the draw differs
from its digest when the run bound), `labels_moved_run_to_now` (differs again between the bind
and this calibration's own read of the labels directory, `null` when the calibration named none),
`calibration_labels_moved` (the calibration-side stems among those two lists) and
`manifest_redrawn` (the manifest directory was overwritten since the run bound). This is a
disclosure, not a floor: the row still validates with the moved stems named on it, and
`describe_review_validation` renders one sentence when `calibration_labels_moved` is non-empty. A
run bound before this check existed, or one calibrated with no bound run under a caller-named
manifest, seals all four keys `null`.

## Failure Triage

When metrics are poor, investigate systematically:

1. Data issues: `python scripts/doctor.py <root>`'s `check_data_quality`, missing labels, format errors, class imbalance
2. Worst cases: `python scripts/render_failure_cases.py`, surface and visually inspect the worst
   N images
3. Per-image breakdown: `annotation_tools.score_predictions` (library call, or
   `scripts/score_predictions.py`) on a dataset dir; find images with the highest FP/FN counts
   (no built-in per-class breakdown for detection; use
   `annotation_tools.score_predictions(<image>, detail=True)` per image and aggregate by
   `class_id` if class-level numbers are needed). On a classified bucket the breakdown is the
   object's localization, not the classifier's own call: use the classifier calibrator to triage
   the confirmed-state axis instead
4. Training dynamics: Check metrics.jsonl; is loss still decreasing? Overfitting?
5. Architecture: Is the model appropriate for the task and data scale?

## Comparison Protocol

When comparing models:
1. Same dataset split: draw one manifest with `draw_splits` and name it from every compared run
   with `data.split.manifest_dir`, so each binds to the identical membership rather than each
   redrawing its own from a shared seed
2. Same evaluation set: the manifest's own `calibration` side (`evaluate_model` with
   `split_manifest_dir`), never each run's own `val`, which is the side its checkpoint was chosen
   on
3. Compare using the metric that governs this trait/task's phenotype (see Metrics by Task Type
   above), not necessarily the labeled comparability metric
4. For classification, check per-class performance; overall accuracy can hide class-specific
   failures; for ordinal, check quadratic weighted kappa alongside rank accuracy, since exact-rank
   accuracy alone can hide a model that's frequently off by one rank; for detection, check
   per-image FP/FN patterns instead (see Failure Triage above; there's no per-class AP today)
5. Use `experiments.compare_experiments` (library call; the web comparison route calls it too)
   for side-by-side analysis
