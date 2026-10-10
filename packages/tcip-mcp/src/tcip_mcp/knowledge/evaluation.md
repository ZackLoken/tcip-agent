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
| Regression | RMSE | R², MAE, concordance correlation coefficient (the statistic a trait revision's `regression_criterion` names for its assessment) |
| Ordinal | Quadratic weighted κ | MAE, rank accuracy |

These are labeled comparability metrics: a fixed-convention number (mAP@50 = AP at IoU 0.5) that
lets runs be compared on the same ruler. They do not govern the phenotype. The criterion that
governs the delivered measurement (which detections are a hit, what the count is) is the *trait's*
localization criterion with a tolerance derived from the data in hand, e.g. a center-match with
`half_class_avg_size` tolerance for small, thin objects like catkins, not a frozen IoU@0.5. Choose the governing
criterion per trait/data; keep mAP@50 alongside only as the comparability label (see `operating_point`
/ the derive-don't-pin rail). There is no single mandated "primary" metric per task.

Detection/instance-seg metrics (`detection_metrics`) come from the platform's one matcher: the
counts at the operating conf under the governing criterion, and average precision (`map50`, `map`
over IoU 0.50 to 0.95) from the same matcher over the confidence sweep, matched by mask for
instance segmentation. They are aggregate only; no per-class AP is reported. Per-class precision/recall/F1 is real for classification (`evaluate_model`); ordinal and
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
| `evaluate_model` | Evaluate a checkpoint on a held-out dataset; returns the result and writes nothing |
| `assess_checkpoint` | Assess a checkpoint for one delivery kind of a confirmed trait against a drawn selection's calibration and holdout sides, recording the assessment a delivery rests on |
| `assess_reserved_regions` | The same assessment over a mosaic's reserved, attested-complete regions, for a checkpoint trained on a within-image split |
| `annotation_tools.score_predictions` (library call) / `tcip score-predictions` (command) | Score on-disk predictions vs GT: an image file returns per-box matches (`detail=True` adds a per-detection breakdown); a dataset dir returns aggregate metrics + per-image TP/FP/FN. It scores the object's localization, never an attribute head's call |
| `tcip render-failure-cases` (command) | Surface + render the N images with highest triage error |
| `experiment_tools.compare_experiments` (library call) | Side-by-side metrics across experiments |
| `get_experiment` (`view='lineage'`) | Trace data → model → predictions chain |
| `list_experiments` | List every run of the project, and every sweep with its trial runs under it |
| `rank_registered_models` | Rank registered models by a stated metric, direction and verification status |

`evaluate_model` accepts an optional `trait=`: when set, the trait's own governing criterion
(not the IoU@0.5 comparability convention) determines detection counts/F1, matching what governs
delivery (see Metrics by Task Type above); omit it and the IoU@0.5 convention governs instead. For
a tile-trained checkpoint, `evaluate_model` reports in one of two regimes: the default tile-level
run is a diagnostic only (matches training-time val mAP, not the shipped full-frame count);
`use_tiled_inference=True` reconstructs predictions to full frame and is the delivery-grade metric
to report for gating. An untiled checkpoint has no regime split; its one run already is the
delivery metric (see `evaluate_model`'s own docstring for the full precedence). Either a run id or
a bare checkpoint path resolves to a file that must be registered in the project this server was
started for (`register_model`, explicit mode for a foreign or bespoke checkpoint); `evaluate_model`
refuses before loading an unregistered one.

The full-frame result records the `execution` record the pass ran under, the one a published
bucket's record carries: each value beside its source in `sources`, `explicit` when the caller
stated it; `cross_tile_nms` holds the merge threshold the evaluation ran at, in the metric
`postprocess` compares over. `evaluate_model` takes what it states of that record as `stated`, as
`run_inference` does. A detector's `conf` and `max_dets` are stated, no default standing behind
either; the full-frame evaluation derives an unstated `cross_tile_nms` from the evaluated ground
truth for an IoU merge. A value with neither a statement nor a basis refuses naming it.
An evaluation answers for no delivery: only an assessment does.

`rank_registered_models` requires a `metric` (no default) and resolves its ranking direction from
`evaluation.HIGHER_IS_BETTER_BY_METRIC` (keyed by the metric with any `val_` prefix stripped);
`higher_is_better` overrides the declaration when a caller states one, required for a metric the
declaration does not name. It ranks only `metrics_source="trainer"` entries by default (the
platform's own `default_train` measured them); `include_unverified=True` also ranks
`"training_source"`/`"caller"` entries, whose numbers were never measured by the platform, and
`excluded_unverified` in the response names what a default call left out.

## Assessment

A delivered number rests on an assessment: one record of what a checkpoint measured, for one
delivery kind of a trait's confirmed revision, against a held-out reference. `assess_checkpoint`
fits every derived value (the conf the trait's count objective picks, the merge threshold) on a
drawn selection's `calibration` side alone, a detector's `max_dets` stated since the frames its
pass later publishes on are not known there, and judges it on the `holdout` side
against the revision's authored floors and tolerances, measuring the copy of the reference it
retains. Every field the kind's criterion reads (its localization, tolerances, floors and,
for a regression, the statistic) is required when the operationalization is proposed, so a
revision that leaves one unauthored never exists to assess. `assess_reserved_regions` does the same
over a mosaic's reserved calibration and holdout regions, refused until those regions are attested
complete and while the recorded mosaic no longer reads at its recorded size.

The record states the producer (the checkpoint's digest and producing run), the execution record
the pass ran under (each value beside its source), the reference (each sample's source digest and
ground-truth digest, the captures it covers), the disjointness checks (the holdout shares no
source digest with the calibration side, and neither shares a group or a source digest
with the producing run's training or selection sides), the criterion's evidence and the failures,
empty when it passed. It is written once. A mosaic reference region that does not lie inside a
held-out region, or overlaps a training region, refuses the assessment.

A bucket is published under an assessment with `run_inference(assessment_id=...)`: the pass
restores the execution record the assessment measured, and a value `stated` records differently
refuses naming each. Every delivery door clears its buckets through one gate: each bucket must
name an assessment that passed for this delivery kind under the revision delivered, was produced
by the checkpoint and execution record that assessment measured, and lies inside the captures its
reference covers; a reference whose ground truth or source images changed since refuses naming
the file. A delivery that fails any of these ships only under the breeder's acknowledgment,
recorded in the Results tab, of exactly the rows and disclosure it writes, stamped unvalidated,
with the reason on its delivery event.

## Failure Triage

When metrics are poor, investigate systematically:

1. Data issues: `tcip doctor <root>`'s `check_data_quality`, missing labels, format errors, class imbalance
2. Worst cases: `tcip render-failure-cases`, surface and visually inspect the worst
   N images
3. Per-image breakdown: `annotation_tools.score_predictions` (library call, or
   `tcip score-predictions`) on a dataset dir; find images with the highest FP/FN counts
   (no built-in per-class breakdown for detection; use
   `annotation_tools.score_predictions(<image>, detail=True)` per image and aggregate by
   `class_id` if class-level numbers are needed). The breakdown is the object's localization,
   never an attribute head's call: `evaluate`'s `attribute_agreement` (per attribute, over the
   matched pairs) and an assessment's classifier criterion triage the attribute axes instead
4. Training dynamics: Check metrics.jsonl; is loss still decreasing? Overfitting?
5. Architecture: Is the model appropriate for the task and data scale?

## Comparison Protocol

When comparing models:
1. Same dataset split: draw one selection with `draw_splits` and name it from every compared run
   with `data.split.selection_dir`, so each binds to the identical membership rather than each
   redrawing its own from a shared seed
2. Same evaluation set: the selection's own held-out side, never each run's own `val`, which is
   the side its checkpoint was chosen on
3. Compare using the metric that governs this trait/task's phenotype (see Metrics by Task Type
   above), not necessarily the labeled comparability metric
4. For classification, check per-class performance; overall accuracy can hide class-specific
   failures; for ordinal, check quadratic weighted kappa alongside rank accuracy, since exact-rank
   accuracy alone can hide a model that's frequently off by one rank; for detection, check
   per-image FP/FN patterns instead (see Failure Triage above; there's no per-class AP today)
5. Use `experiments.compare_experiments` (library call; the web comparison route calls it too)
   for side-by-side analysis
