---
name: delivery
description: "How to deliver phenotyping results: the per-plant CSV schema, per-image-to-per-plant aggregation rules by trait type, the export tools, and pre-delivery quality control. Load when exporting results, building a per-plant deliverable, aggregating per-image measurements to per-plant, or quality-checking a breeding-program CSV before hand-off."
---

# Results Delivery

## Per-Plant CSV Schema

The final deliverable is a CSV with one row per plant per trait:

Columns, in the order `export_aggregated_csv` writes them (its `fieldnames` is the authority; check
it in `pipelines/postprocessing/aggregation.py` if this table looks stale):

| Column | Type | Description |
|--------|------|-------------|
| plant_id | string | Unique plant identifier |
| crop | string | Crop species |
| trait_name | string | Measured trait (from crop skill) |
| value | float/string | Measurement value |
| units | string | Physical unit implied by the value's own `value_key`, blank for a count trait or a trait crops.yml declares no unit for. Under `operating_point`, a unit-declared trait's `value_key` must itself imply a unit; a px-space value refuses rather than shipping blank. Under a scalar head (`ordinal_operating_point`/`regression_operating_point`), a `value_key` implying no unit takes the trait's own declared unit from `crops.yml` instead of blank, since a calibrated head predicts in that unit by construction |
| value_key | string | Which aggregated field `value` came from (e.g. `count`, `area_mm2`), so a reader can detect a px/mm mismatch independently |
| measurement_document | string | Which sidecar document (`operating_point`, `ordinal_operating_point`, `regression_operating_point`) answers for the measurement that produced `value`, stated by the caller's records and checked against every named bucket |
| scale_document | string | `resolve_scale` when a per-pixel physical scale produced `value`, blank otherwise |
| confidence | float | Model confidence [0,1] |
| n_images | int | Number of source images |
| pipeline_version | string | Pipeline that produced this result |
| plant_id_source | string | How the plant identity was resolved for this plant's images (`"mixed"` when they disagree); blank when the records carried no identity provenance |
| plant_id_distance_m_max | float | Worst per-image plant-assignment distance, in metres, across this plant's images; the identity-confidence signal `build_plant_mapping` produces |
| producer_model_sha256 | string | Checkpoint hash of the model that produced the predictions; blank when the bucket names an experiment nothing outside it can corroborate |
| experiment_id | string | The run that produced the predictions, blank when there was none or nothing corroborates it; never the calibration a claim was earned under |
| produced_at | string | Timestamp stamped by the producing run |
| measurement_validated | string | The reconciled validity state stamped into every row by the delivery gate |
| validation_record | string | `experiment:digest` of the validation record the delivered claim was verified against; blank when the numbers rest on no record |

## Aggregation Rules

Aggregating per-image measurements into a per-plant value is an agent choice keyed to the trait's
read-semantics, not a frozen one-function-per-type map. The table below is a starting reference of
common choices; pick (or compose) the aggregation the trait's definition actually calls for (a
skewed count may want a median, a robust mean, or a yield-model estimate; a date wants a crossing),
and record which you used. When the trait carries read-semantics fields (`TraitSpec`), those govern.

Examples use real `crops.yml` trait names; verify any trait against `crops.yml` before use.

| Trait Type | Common choice | Example |
|-----------|-------------|---------|
| Count | Median across images | `stem_count` → median across the plant's images |
| Date (bloom) | Elongated-fraction crossing | `catkin_50per_date` → date the elongated fraction crosses 50% (see `phenology` skill) |
| Ordinal | Mode | `efb_damage` → most common rating |
| Continuous | Mean | `fruit_diameter` → average across images |
| Area | Sum / 3D model | `plant_surface_area` → planimetric crown area from a 3D canopy model, out of current build scope (3D LiDAR/SfM is not built today; see CLAUDE.md Scope). A validated 2D mask yields a calibrated pixel area, not this trait |

## Tools

| Tool | Purpose |
|------|---------|
| logged `scripts/` script | Produces the Per-Plant CSV Schema above by chaining `launch_training` + `run_inference` then the importable postprocessing libs `aggregate_per_plant` / `export_aggregated_csv`; see `pipeline-design` skill for the chaining mechanics |
| `export_predictions` | Export predictions as per-image JSON (COCO-shaped). A bucket (`output_dir`) with review verdicts is immutable; the export redirects to a fresh `<dir>@r2` bucket (see the response's `output_dir`); `overwrite=True` forces in-place but is refused when verdicts exist |
| `run_inference` | Run batch inference on images |
| `compute_phenology` | Per-plant bloom CSV (05/50/95-per-date) from classified preds + plant mapping; its own column schema; see `phenology` skill |
| `deliver_orthomosaic_plant_counts` | Per-plant detection counts from a persisted whole-raster prediction bucket plus plant-locations CSV(s); georeferences the boxes, assigns them to plants, and delivers through `export_aggregated_csv`, so it inherits the same gate and provenance columns; refuses a bucket that cannot vouch for the caller's raster |
| `materialize_review_dataset` | Turn human review verdicts into a curated training set for re-delivery after correction; see `annotation` skill |

`tabulate_counts` produces a different, per-image `image, detection_count,
avg_confidence` CSV, not the per-plant schema above. Don't reach for it when the
per-plant schema is what's wanted.

## The meaning door (what the number is)

Before the evidence gate below, every count and per-plant delivery answers a different question:
what the delivered number means, recorded per project and confirmed by the breeder. `crops.yml`
gives a field criterion, which is not something a model can realize on its own, so the record says
what the number means, what decides it in the imagery, and which subject it is about. No record, or
one nobody confirmed, or one whose spec fields moved since, and the door refuses and names the
primitive that fixes it. `acknowledge_unvalidated` does not reach this: it says a number's error is
uncharacterized, which is a claim about a quantity that has been defined.

- `export_detection_csv` and `tabulate_counts` take a required `trait` and rest on its
  `per_image_count` record. That record names no delivered phenotype, because the per-image CSV
  carries no phenotype column; what it names is the counted subject, checked against the recorded
  `id_map` of every bucket that recorded one.
- `export_aggregated_csv` and `deliver_orthomosaic_plant_counts` take `trait_name`, which stays a
  crop-vocabulary delivered-phenotype name (the CSV column and the unit cross-check read it), and
  resolve it to the registered trait whose spec `delivers` it: none or more than one refuses. Which
  record applies follows from the records' own `measurement_document` (`operating_point`,
  `ordinal_operating_point` or `regression_operating_point`), since a count, an ordinal and a
  regression aggregate rest on three different spec floors and are three separate confirmations.
  Every row carries a value key and every one has to be inside the confirmed set.
- A `state_crossing_dates` statement and every delivery under it are checked against the delivered
  dataset's own class registry, never a bare spec value: `state_trait_operationalization` resolves
  it from `dataset_root` (given explicitly, or the project root's own registry when that project is
  unambiguously one dataset), and a positive class the registry does not declare for the measured
  subject refuses at the statement, or supersedes an existing confirmation at delivery.

## The delivery gate (measurement integrity)

Every phenotype-delivery door refuses a bare write: an unvalidated measurement number with no
acknowledgement. The count/date/value is the phenotype, so each door runs one shared
`check_delivery_gate`: it ships only when each measurement dimension the deliverable rests on is
validated against a reference of that dimension's own kind (for the count/measurement/classifier
dimensions, held-out GT or a breeder-confirmed output sample, see the `evaluation` and
`cv-research` skills; a tile scale needs a geometry basis, a physical scale a physical
measurement, and no kind stands in for another), read from the predictions' own
`operating_point.json` sidecar, not a caller-asserted string.

- `compute_phenology` and the web `/export_csv` phenology branch both reconcile the positive-state
  classifier (from `classifier_operating_point.json`, see `calibrate_classifier_operating_point`)
  and the count operating point, then hand the reconciled state to `write_phenology_csv` /
  `write_phenology_curve_csv` (`phenology.py`), the one writer both doors share: it runs the gate
  itself, composes every provenance cell and records the delivery event, so a CSV from either door
  carries the same schema and the same composition. The MCP door alone can carry a caller-stated
  `operating_point_conf` and a caller-asserted validity floor (`phenology_tools.py:745`, `:882-886`),
  fields the web door has none for. The producer tail (`producer_model_sha256`,
  `producer_experiment_id`, `validation_record`) is filled from the verified bindings, so a bucket
  whose claim no record answers for delivers those cells blank rather than repeating the names its
  stamp asserted.
- `tabulate_counts` gates the count operating point on the run's resolved bundle and hands
  `export_detection_csv` the already-reconciled state, plus the bucket it persisted when it was
  given a `predictions_dir` to persist one into.
  `export_detection_csv` and `export_aggregated_csv` both gate at the writer the same way: pass
  `pred_dirs` for a count trait so the validity is reconciled from each bucket's
  `operating_point.json` sidecar rather than trusting a bare caller string. Without `pred_dirs`,
  either writer takes `measurement_validated` directly as a caller-asserted string with no on-disk
  reconciliation. A continuous/ordinal trait has no on-disk measurement-validity producer today, so
  without `pred_dirs` the only route to delivery through `export_aggregated_csv` is the explicit
  acknowledge below.
- A tiled run's `tile_size` is a second gating dimension of the same count operating point, at
  every one of those doors: the tile edge scales the per-image counts, so a run with no persisted
  training geometry, no recoverable native-frame edge, and no explicit caller override refuses
  exactly as an uncalibrated conf does. Three routes reach a validated tile scale, ranked strongest
  to weakest: a checkpoint whose training tile geometry was persisted; a checkpoint that trained
  untiled on frames of one square size (that frame's own size, each tile run through the resize its
  recorded augmentation config applied), a real basis mechanically inferred from the checkpoint's
  own recorded frame, never stated by a caller, and sufficient on its own to clear the gate; or an
  explicit caller `tile_size` that the checkpoint's own recorded geometry (persisted, or absent
  one, the native frame) does not contradict, every tiling door refuses outright when it does,
  naming both edges. A delivery that mixes buckets across tiers travels under the weakest
  tier present. An untiled run is never gated on it.
- To ship a provisional result, pass `acknowledge_unvalidated=True`: the door writes but stamps
  `measurement_validated=false` so the un-trustworthiness travels with the CSV. This is for an honest
  provisional delivery, never for silently shipping a bare number. A tile scale with no real basis at
  all has no value to ship provisionally either, so `acknowledge_unvalidated` cannot admit that case:
  it refuses unconditionally.

`run_inference` stays fully ungated but honestly stamped: the review loop must be able to produce
unvalidated predictions to *reach* a validated measurement, so it never refuses on conf; it does
refuse (unconditionally, no `acknowledge_unvalidated`) when tiling is requested and the checkpoint's
tile scale has no real basis at all, since there is no number to tile at. `export_predictions`, the
door that actually persists a prediction bucket other doors treat as ground truth, applies the same
tile_size gate on top, which any of the three bases clears. Conf itself is never gated at either
door; the measurement gate lives at the final phenotype door.

## Quality Control

Before delivery, verify:
1. Completeness: Every plant has values for all expected traits
2. Range: Values within biological plausibility (e.g., fruit diameter 1-15cm)
3. Outliers: Flag statistical outliers for manual review
4. Confidence: Reject predictions below a confidence operating point derived from the data in hand (per the "derive, don't pin" rule), not a frozen constant; the right cutoff varies by dataset, model, and trait
5. No duplicates: One value per plant per trait per date
