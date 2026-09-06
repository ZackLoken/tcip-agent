---
name: delivery
description: "How to deliver phenotyping results: the per-plant CSV schema, per-image-to-per-plant aggregation rules by trait type, the export tools, and pre-delivery quality control. Load when exporting results, building a per-plant deliverable, aggregating per-image measurements to per-plant, or quality-checking a breeding-program CSV before hand-off."
---

# Results Delivery

## Per-Plant CSV Schema

The final deliverable is a CSV with one row per plant per trait:

Columns, in the order `export_aggregated_csv` writes them (its `fieldnames` is the authority; see
`pipelines/postprocessing/aggregation.py`):

| Column | Type | Description |
|--------|------|-------------|
| plant_id | string | Unique plant identifier |
| crop | string | Crop species |
| delivered_phenotype | string | Delivered phenotype name (from crop skill) |
| value | float/string | Measurement value |
| units | string | Physical unit implied by the value's own `value_key`, blank for a count trait or a trait crops.yml declares no unit for. Under `operating_point`, a unit-declared trait's `value_key` must itself imply a unit; a px-space value refuses rather than shipping blank. Under a scalar head (`ordinal_operating_point`/`regression_operating_point`), a `value_key` implying no unit takes the trait's own declared unit from `crops.yml` instead of blank |
| value_key | string | Which aggregated field `value` came from (e.g. `count`, `area_mm2`), so a reader can detect a px/mm mismatch independently |
| measurement_document | string | Which sidecar document (`operating_point`, `ordinal_operating_point`, `regression_operating_point`) answers for the measurement that produced `value`, stated by the caller's records and checked against every named bucket |
| scale_document | string | `resolve_scale` when a per-pixel physical scale produced `value`, blank otherwise |
| confidence | float | Model confidence [0,1] |
| n_images | int | Number of source images |
| pipeline_version | string | Pipeline that produced this result |
| plant_id_source | string | How the plant identity was resolved for this plant's images (`"mixed"` when they disagree); blank when the records carried no identity provenance |
| plant_attribution | string | The granularity objects were attributed to plants at: `"image"` for `build_plant_mapping`'s walked-capture mapping, `"detection"` for an orthomosaic's nearest-neighbour per-detection mapping, `"segment"` for an orthomosaic's canopy-segment mapping (a detection's box centroid fell inside a canopy boundary a person accepted, never a mask-level or area measurement). Distinct from `plant_id_source`, which names the matching method, not the granularity |
| plant_id_distance_m_max | float | Worst per-image plant-assignment distance, in metres, across this plant's images; the identity-confidence signal `build_plant_mapping` produces |
| producer_model_sha256 | string | Checkpoint hash of the model that produced the predictions; blank when the bucket names an experiment nothing outside it can corroborate |
| producing_experiment_id | string | The run that produced the predictions, blank when there was none or nothing corroborates it; never the calibration a claim was earned under |
| produced_at | string | Timestamp this CSV was written, stamped by the shared tail composition; never the producing run's own timestamp |
| operating_point_validated | string | The named dimension's cleared reference (paired with `measurement_document`, e.g. `measurement_document=ordinal_operating_point` beside this column naming which document it answers for), floored false whenever any other gated dimension with no column of its own (tile_size, scale, claim_scope) is unvalidated |
| unvalidated_dimensions | string | Every gated dimension that did not validate, `;`-joined; blank when everything cleared |
| validation_record | string | `experiment:digest` of the validation record the delivered claim was verified against; blank when the numbers rest on no record |
| acknowledged_by | string | Who shipped this delivery unvalidated, from the gate's own `Acknowledgement`; blank on a fully validated delivery, even one posted with one (the gate discards an acknowledgement that cleared nothing) |
| acknowledgement_reason | string | Why, from the same `Acknowledgement`; blank together with `acknowledged_by` |

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
| a one-off script, in the project's own directory | Produces the Per-Plant CSV Schema above by chaining `launch_training` + `run_inference` then the importable postprocessing libs `aggregate_per_plant` / `export_aggregated_csv`; see `pipeline-design` skill for the chaining mechanics |
| `run_inference` | Run a checkpoint over images or a raster and persist predictions as per-image JSON (COCO-shaped) or, for a raster, one whole-mosaic prediction file. A bucket (`output_dir`, the run's own prediction directory, not a score bin) with review verdicts is immutable; the run redirects to a fresh `<dir>@r2` bucket (see the response's `output_dir`); `overwrite=True` forces in-place but is refused when verdicts exist. A bucket that already holds a prediction document from an earlier publish, with no verdict yet recorded, refuses this run outright, whatever `overwrite` says, naming the document count and a suggested fresh bucket; a completed experiment's bucket in that state has no audited route to clear it for republication |
| `deliver_phenology_milestones` | Per-plant bloom CSV (05/50/95-per-date) from classified preds + plant mapping; its own column schema; see `phenology` skill |
| `register_plant_registry` | Names a plant-locations CSV set once (per-file `sha256`/`n_plants`, `crop`, `site`, a content digest over the parsed rows), so `deliver_orthomosaic_plant_counts` and `build_plant_mapping` read the same registered version by name (`plant_registry`) instead of re-asserting file paths |
| `deliver_orthomosaic_plant_counts` | Per-plant detection counts from a persisted whole-raster prediction bucket plus a `plant_registry` name; georeferences the boxes and delivers through `export_aggregated_csv`, so it inherits the same gate and provenance columns; refuses a bucket that cannot vouch for the caller's raster. Nearest-neighbour by default; `canopy_subject` switches to containment in an accepted canopy boundary instead (refused alongside a stated `nn_tolerance_m`). Fewer rows than the registry names can ship under either regime, the absent plants named on the delivery event (outside the raster's frame under both; with no segment, or with an ambiguous detection, under the segment regime). This tool builds no acknowledgement, so an unvalidated dimension always refuses here; the Results tab's count export (`/api/results/export_count_csv`) is the one surface that can acknowledge and ship this kind unvalidated |
| `deliver_per_image_counts` | Per-image detection-count CSV; see the Per-Image CSV Schema above. Builds no acknowledgement either, so its bucket regime's unvalidated dimension always refuses here too; the same Results tab count export serves it |
| `deliver_per_plant_csv` | The general per-plant CSV door: takes `aggregate_per_plant`'s own output (a caller's own composition of buckets plus a plant mapping) and calls `export_aggregated_csv` directly, for the case neither specialist door's own composition covers; `predictions_by_date` names the buckets a delivery reads, whether or not a mapping is named; a named `plant_mapping` requires it in the same call and is resolved, refused by name when unknown, when it does not cover a delivered date, when it was built over a different dataset than the buckets belong to, or when its own recorded inputs no longer verify, through the same preamble the phenology doors share; every delivered `plant_id` must also appear among a plot the mapping actually assigned on the delivered dates; a delivery either fully verifies the mapping it names or names none |
| `supersede_delivery` | Records that an already-shipped delivery's number is withdrawn or replaced (`delivery_supersessions`, keyed by the superseded event's id, naming its `output_sha256`, the replacement event when one exists, and the reason); never deletes or rewrites the file or the event |
| `calibrate_scalar_operating_point` | Calibrate and validate a continuous or ordinal trait's prediction against a disjoint held-out split; stamps `ordinal_operating_point.json` / `regression_operating_point.json`, the on-disk producer `export_aggregated_csv` reconciles against |
| `calibrate_count_operating_point` | Calibrate and validate a count trait's confidence operating point against a disjoint held-out split, and merge the earned `conf` and validation pointer into an already-published bucket's own `operating_point.json`, the on-disk producer every count reconciler reads; a calibration that does not clear its gate merges `validated=false` and earns nothing; refuses a bucket that already carries a validated stamp |
| `calibrate_physical_scale` | Derive and validate a per-pixel physical scale against a breeder-supplied reference, and stamp it into a bucket's `resolve_scale.json`, the `scale_document` cell's producer |
| `materialize_review_dataset` | Turn human review verdicts into a curated training set for re-delivery after correction; see `annotation` skill |

`deliver_per_image_counts` produces a different, per-image CSV, not the per-plant schema above
(the `image` cell is the source basename with its extension, this platform's image identity
everywhere else). The bucket regime resolves it from the stamp's `image_filenames` map (each
prediction document's stem mapped to its source image's filename, recorded at publication) and
falls back to the bare stem, disclosed in the response's `image_note`, for a bucket stamped before
that map existed or for a stem the map does not name. Don't reach for it when the per-plant schema
is what's wanted. Two source regimes: live (`checkpoint_path` + `images_dir`, routing through the
same verified pass `run_inference` runs, optionally persisting the counted predictions into
`predictions_dir` under `run_inference`'s own publish contract) or bucket (`predictions_dir` alone,
no GPU, reading an existing reviewed bucket's own stamp), exactly one stated.

### Per-Image CSV Schema

Columns, in the order `export_detection_csv` writes them (its `fieldnames` is the authority; see
`pipelines/postprocessing/export.py`):

| Column | Type | Description |
|--------|------|-------------|
| image | string | Source image basename with its extension |
| detection_count | int | Real detections in this image (a zero-extent box excluded) |
| avg_confidence | float | Mean score across this image's kept detections |
| measurement_document | string | Always `operating_point`: a detection count never rests on a scalar head or a physical scale |
| producer_model_sha256 | string | Checkpoint hash of the model that produced the predictions; blank when the bucket names an experiment nothing outside it can corroborate |
| producing_experiment_id | string | The run that produced the predictions, blank when there was none or nothing corroborates it |
| operating_point_conf | float | The count operating point's confidence threshold |
| produced_at | string | Timestamp this CSV was written, stamped by the shared tail composition |
| operating_point_validated | string | The count operating point's cleared reference |
| unvalidated_dimensions | string | Every gated dimension that did not validate, `;`-joined; blank when everything cleared |
| validation_record | string | `experiment:digest` of the validation record the delivered claim was verified against; blank when the numbers rest on no record |
| acknowledged_by | string | Who shipped this delivery unvalidated; blank on a fully validated delivery |
| acknowledgement_reason | string | Why, from the same `Acknowledgement`; blank together with `acknowledged_by` |

## The meaning door (what the number is)

Before the evidence gate below, every count and per-plant delivery answers a different question:
what the delivered number means, recorded per project and confirmed by the breeder. `crops.yml`
gives a field criterion, which is not something a model can realize on its own, so the record says
what the number means, what decides it in the imagery, and which subject it is about. No record, or
one nobody confirmed, or one whose spec fields moved since, and the door refuses and names the
primitive that fixes it. An acknowledgement does not reach this: it says a number's error is
uncharacterized, which is a claim about a quantity that has been defined.

- `export_detection_csv` and `deliver_per_image_counts` take a required, keyword-only `trait` and rest on
  its `per_image_count` record, in either of `deliver_per_image_counts`'s two source regimes. That record
  names no delivered phenotype, because the per-image CSV carries no phenotype column; what it
  names is the counted subject, checked against the counted subjects of every bucket that
  recorded one (a classified bucket's own scope subject, its recorded `id_map`'s keys otherwise,
  since a classified map is keyed by attribute values, not object classes). A bucket-regime call
  also refuses a bucket whose own stamp names a different, non-`None` trait, validated or not.
- `export_aggregated_csv` and `deliver_orthomosaic_plant_counts` take `delivered_phenotype`, a
  crop-vocabulary delivered-phenotype name (the `delivered_phenotype` CSV column and the unit
  cross-check read it), and resolve it to the registered trait whose spec `delivers` it: none or
  more than one refuses. Which record applies follows from the records' own `measurement_document`
  (`operating_point`, `ordinal_operating_point` or `regression_operating_point`), since a count, an
  ordinal and a regression aggregate rest on three different spec floors and are three separate
  confirmations. Every row carries a value key and every one has to be inside the confirmed set.
- A `state_crossing_dates` statement and every delivery under it are checked against the delivered
  dataset's own class registry, never a bare spec value: `state_trait_operationalization` resolves
  it from `dataset_root` (given explicitly, or the project root's own registry when that project is
  unambiguously one dataset), and a positive class the registry does not declare for the measured
  subject refuses at the statement, or supersedes an existing confirmation at delivery.

## The delivery gate (measurement integrity)

Every phenotype-delivery door refuses a bare write: an unvalidated measurement number with no
acknowledgement. The count/date/value is the phenotype, so each door reconciles every dimension
the deliverable rests on against a reference of that dimension's own kind (for the
operating_point/classifier dimensions, held-out GT or a breeder-confirmed output sample, see the
`evaluation` and `cv-research` skills; a tile scale needs a geometry basis, a physical scale a
physical measurement, and no kind stands in for another), reading the predictions' own sidecar,
one of the five measurement-document kinds a bucket can carry (`operating_point.json`,
`classifier_operating_point.json`, `ordinal_operating_point.json`, `regression_operating_point.json`,
`resolve_scale.json`), then hands the resolved states to one shared
`check_delivery_gate`, which does no I/O of its own: it judges the already-resolved dict and ships
only when every dimension it was handed clears, or two independent escapes cover what didn't. An
`Acknowledgement` (a real name and a non-empty reason, built only by a web delivering route)
clears any dimension, stamped false. The
phenology writer, `export_detection_csv` and `export_aggregated_csv` all take one now, each from
the web results route that composes it for its own delivery kind (`/export_csv` for phenology,
`/export_count_csv` for the per-image bucket regime and the per-plant orthomosaic composition);
no MCP tool builds one.
`allow_unvalidated_staging` clears only `tile_size`/`claim_scope`, the pre-pass gate a raw
prediction bucket is written under, and can never clear a phenotype's own delivered dimension
(`operating_point`, `classifier`, `scale`).

`record_delivery_binding_event`'s own project-scoped `delivery_events` record files under the
caller's project (an MCP tool's process-pinned root, or a web route's guarded, resolved one); its
dataset-scoped audit-log line files under the buckets' own shared dataset root regardless, and is
unaffected by a caller's project root.

- `deliver_phenology_milestones` and the web `/export_csv` phenology branch both reconcile the positive-state
  classifier (from `classifier_operating_point.json`, see `calibrate_classifier_operating_point`)
  and the count operating point, then hand the reconciled state to `write_phenology_csv` /
  `write_phenology_curve_csv` (`phenology.py`), the one writer both doors share: it runs the gate
  itself, composes every provenance cell and records the delivery event, so a CSV from either door
  carries the same schema and the same composition. The delivery event is a best-effort second
  write after the CSV already exists, so an already-delivered file can outlive a failed event
  write; the web export route's own `X-TCIP-Delivery-Event-Recorded` response header says whether
  this delivery's event actually landed. The MCP door alone can carry a caller-stated
  `operating_point_conf` and a caller-asserted validity floor (`phenology_tools.py`),
  fields the web door has none for. The producer tail (`producer_model_sha256`,
  `producing_experiment_id`, `validation_record`) is filled from the verified bindings, so a
  bucket whose claim no record answers for delivers those cells blank rather than repeating the
  names its stamp asserted; `produced_at` is always the write's own timestamp, never blank and
  never read from the bindings. Both writers also carry `dates_delivered` and
  `images_unattributed`, scoped to this delivery's own delivered dates (never the mapping's own
  `n_dates_missing_images` span, which is delivery-independent), and `plant_attribution`, the
  granularity `build_plant_mapping` attributed captures to plants at; the curve CSV repeats the
  same delivery-wide `images_unattributed` count beside its own per-row `n_images`.
  `deliver_orthomosaic_plant_counts`'s own delivery event carries the same idea in a different
  shape: no walked `build_plant_mapping` build exists for a whole-raster frame, so its own
  `plant_mapping` names the registry it read (byte-verified against what `register_plant_registry`
  recorded, refusing a rewritten file by name before any plant or prediction is read), the raster
  identity every count is attributed through, and `detections_unattributed` scoped to the
  delivered raster, the raster-frame counterpart of `images_unattributed`. The nearest-neighbour
  regime's disclosure adds the matched tolerance and its source, and every registry plant outside
  the raster's own frame, by name (`plants_outside_raster`; "outside the raster" is the registry's
  own point, never the tree's real canopy extent). The canopy-segment regime's disclosure instead
  names the boundary document it read (path, digest, subject, segment count), the resolved
  segment-to-plant ties with each one's own `clearance_m` (the margin a displaced registry
  position would have to exceed to leave its tied segment; the tie itself rests on nothing but
  that disclosed clearance, since no breeder-confirmed tie or validated position-error bound
  exists yet), and every plant its own rows do not cover: outside the raster, inside no segment,
  or inside a segment whose own detection was ambiguous. Either regime's per-plant count is a
  raster-visible count, not a whole-tree total: where the crop's own knowledge document states 2D
  occlusion undercounts a canopy-borne quantity (the chestnut document does, for `n_burrs`/
  `burrs_density`), a raster count inherits that undercount.
- `deliver_per_image_counts`'s live regime, given a `predictions_dir`, publishes into it through the same
  bracket `run_inference` publishes with (the same bucket-immutability resolution, refusing on a
  verdict or on a document a prior run left with none; the tile gate, count-claim gate,
  frozen-lineage-pointer refusal, write, lineage link, gated only by `allow_unvalidated_staging`,
  never a route to ship the CSV unvalidated), then hands `export_detection_csv` that bucket; a
  document refusal returns before the checkpoint is loaded. The CSV's own delivery gate
  then runs exactly once, inside the writer, never a second time at the door. Without a
  `predictions_dir`, the door calls the writer with no bucket at all, and the writer's own
  no-`pred_dirs` floor always refuses: an acknowledgement clears an unvalidated dimension, never
  the absence of a bucket to reconcile one from at all. The door's own
  response then reports the live run's own narrowed conf reference under
  `run_conf_validated_against` (accepted-or-false), a different fact from
  `operating_point_validated`, which floors false on this path since nothing on disk backs it
  without a bucket for a second gate call to reconcile against. With a bucket (either regime),
  `operating_point_validated`/`tile_size_validated` instead quote the writer's own returned
  gate-and-reconciliation summary, never a second gate call, and `unvalidated_dimensions` names
  every gated dimension that did not validate. Without a bucket, tile_size is never one of the
  gate's flags at all, so the CSV's own `unvalidated_dimensions` cell can only ever name
  `operating_point` on that path; `tile_size_validated` there is the run's own in-memory
  tile-scale flag, which never passed the gate. The bucket regime (`predictions_dir` alone) reads an
  existing bucket's own stamp and hands the writer that same bucket, with no run at all standing
  behind it. `export_detection_csv` and `export_aggregated_csv` both gate at the writer the same
  way: pass `pred_dirs` so the validity is reconciled from each bucket's own sidecar. Without
  `pred_dirs`, either writer floors
  `operating_point_validated` to unvalidated regardless of what the caller asserted; there is no
  carve-out. A continuous or
  ordinal trait's on-disk measurement-validity producer is
  `calibrate_scalar_operating_point`, which stamps `ordinal_operating_point.json` /
  `regression_operating_point.json`; `export_aggregated_csv` reconciles against it the same way it
  reconciles a count trait's `operating_point.json`, keyed off each result's own
  `measurement_document`.
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
- To ship a provisional result, the web `/export_csv` route (phenology) or `/export_count_csv`
  route (the per-image bucket regime and the per-plant orthomosaic composition) builds a real
  `Acknowledgement` from the breeder's own reason and identity; the door writes but stamps
  `operating_point_validated=false` (and every other unvalidated dimension the same way). No MCP
  tool builds an `Acknowledgement` itself; the ordinal and regression aggregates have no
  acknowledged route since their per-plant strategy is the agent's own choice, reached only
  through `deliver_per_plant_csv`, which accepts a caller-composed table the principle above
  refuses a web route for; nor does a per-plant count aggregate over walked captures, since no
  structured field states its across-dates aggregation strategy for a breeder to confirm.

The private pass beneath `run_inference` stays fully ungated: the review loop
must be able to produce unvalidated predictions to *reach* a validated measurement, so it never
refuses on conf; it does refuse (unconditionally, no staging escape) when tiling is requested and
the checkpoint's tile scale has no real basis at all, since there is no number to tile at.
`run_inference` itself, the door that actually persists a prediction bucket other doors treat as
ground truth, applies a further gate on top of that pass' own result over the two staging
dimensions, tile_size and claim_scope: any of the three tile-size bases clears the first, and a
raster pass whose training mosaic identity matches clears the second. A dimension the gate reads
as unvalidated refuses the write unless `allow_unvalidated_staging=True`, which stamps only the
raw bucket unvalidated and is never a route to a delivered phenotype. Conf itself is never gated
at either layer; the measurement gate lives at the final phenotype door.

## Quality Control

Before delivery, verify:
1. Completeness: Every plant has values for all expected traits
2. Range: Values within biological plausibility. `crops.yml` carries no range field on any
   trait; the plausible range is the breeder's own account, elicited and checked at review
3. Outliers: Flag statistical outliers for manual review
4. Confidence: Reject predictions below a confidence operating point derived from the data in hand (per the "derive, don't pin" rule), not a frozen constant; the right cutoff varies by dataset, model, and trait
5. No duplicates: One value per plant per trait per date
