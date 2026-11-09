---
name: delivery
description: "How to deliver phenotyping results: the per-plant CSV schema, per-image-to-per-plant aggregation rules by trait type, the export tools, and pre-delivery quality control. Load when exporting results, building a per-plant deliverable, aggregating per-image measurements to per-plant, or quality-checking a breeding-program CSV before hand-off."
---

# Results Delivery

## Per-Plant CSV Schema

The final deliverable is a CSV with one row per plant per trait:

Columns, in the order the per-plant doors write them (see
`pipelines/postprocessing/aggregation.py`). `deliver_orthomosaic_plant_counts` writes every column
but the three marked per-image aggregate only: its value is one plant's count on one orthomosaic,
which no per-plant statistic reduces. Everything the delivery rests on beyond the row (the
producing checkpoint and its origin, each bucket's gate finding and the assessment behind it, the
acknowledgment, the population) is recorded once on the delivery event the last column names:

| Column | Type | Description |
|--------|------|-------------|
| plant_id | string | Unique plant identifier |
| crop | string | The crop of the plant registry the rows' plant identities came from |
| delivered_phenotype | string | Delivered phenotype name (from crop skill) |
| value | float/string | Measurement value |
| units | string | Physical unit implied by the value's own `value_key`, blank for a count trait or a trait crops.yml declares no unit for. In a count delivery a unit-declared trait's `value_key` must itself imply a unit, and a dimensional value rests on a physical-scale assessment; in an ordinal or regression delivery a `value_key` implying no unit takes the trait's own declared unit from `crops.yml` |
| value_key | string | The quantity `value` is, as the delivery names it (e.g. `count`, `area_mm2`): the quantity of the per-image values a per-image aggregate reduces, or `count` for an orthomosaic count, so a reader can detect a px/mm mismatch independently |
| plant_attribution | string | The granularity objects were attributed to plants at: `"image"` for `build_plant_mapping`'s walked-capture mapping, `"detection"` for an orthomosaic's nearest-neighbor per-detection mapping, `"segment"` for an orthomosaic's canopy-segment mapping (a detection's box centroid fell inside a canopy boundary a person accepted, never a mask-level or area measurement). Distinct from `plant_id_source`, which names the matching method, not the granularity |
| n_images | int | Per-image aggregate only: number of source images |
| plant_id_source | string | Per-image aggregate only: how the plant identity was resolved for this plant's images, the `source` of the mapping's assignments of them (`"mixed"` when they disagree) |
| plant_id_distance_m_max | float | Per-image aggregate only: worst per-image plant-assignment distance, in meters, across this plant's images, the largest `distance_m` of the mapping's assignments of them |
| validated | bool | Whether every delivered bucket's assessment cleared the one delivery gate; false only on a delivery a breeder acknowledged shipping unvalidated |
| trait | string | The trait the delivery ships under |
| trait_revision | int | The confirmed trait revision the delivery ships under |
| trait_revision_sha256 | string | That revision's entry digest, the one the breeder confirmed |
| delivery_event_id | string | The delivery event recording this file's provenance |

## Aggregation Rules

The statistic a plant's per-image values combine into is part of the trait's definition: the
entry's `plant_aggregation`, which every per-plant aggregate kind rests on and the breeder
confirms. Ask the breeder its question when it is unauthored and propose their answer with
`propose_trait`; a per-plant aggregate delivery takes the per-image records and aggregates them
itself under the confirmed statistic, so nothing outside the delivery picks one. The table below
is a starting reference for that conversation, never the answer (a skewed count may want a
median; a date wants a crossing, whose rule is the entry's `crossing_rule`).

A per-plant count on an orthomosaic is a kind of its own, `per_plant_orthomosaic_count`, resting on
the count fields and no `plant_aggregation`: a mosaic is stitched from many aerial frames, so its
count is neither one image's nor a statistic over several, and the door delivers each plant's
attributed detection count as it is.

Examples use real `crops.yml` trait names; verify any trait against `crops.yml`, served whole by
`serve_domain_knowledge(name='crops')`, before use.

| Trait Type | Common choice | Example |
|-----------|-------------|---------|
| Count | Median across images | `stem_count` → median across the plant's images |
| Date (bloom) | Elongated-fraction crossing | `catkin_50per_date` → date the elongated fraction crosses 50% (see `phenology` skill) |
| Ordinal | Mode | `efb_damage` → most common rating |
| Continuous | Mean | `fruit_diameter` → average across images |
| Area | Sum / 3D model | `plant_surface_area` → planimetric crown area from a 3D canopy model, out of current build scope (3D LiDAR/SfM is not built today). A validated 2D mask yields a calibrated pixel area, not this trait |

## Tools

| Tool | Purpose |
|------|---------|
| `assess_checkpoint` | Assess a checkpoint against the calibration and holdout sides of a drawn selection, for one delivery kind of a trait's latest confirmed revision; records the assessment (its execution record, the retained reference, the disjointness from the producing run, the criterion and its failures, none when it passed) under `.tcip/assessments/<id>/`, write-once |
| `assess_reserved_regions` | The same assessment for a whole-mosaic checkpoint, over the reserved regions its training run held out of one mosaic, for a count kind (`per_image_count`, `per_plant_count_aggregate` or `per_plant_orthomosaic_count`) |
| `calibrate_physical_scale` | Derive a per-pixel physical scale from reference objects of breeder-measured length on a selection's calibration side and check it on its holdout side; recorded as an assessment a dimensional delivery names |
| `run_inference` | Run a checkpoint over images or a raster and publish the predictions as a new bucket named `bucket` under the images' dataset root: one document per image (one for a raster) and the bucket's record of the checkpoint, class scope, execution record, capture and assessment behind them, in one commit. A bucket is published once; a name already published refuses. With `assessment_id` the pass runs exactly that assessment's execution record and the bucket names it |
| `deliver_per_image_counts` | Per-image detection-count CSV from one published bucket; see the Per-Image CSV Schema below |
| `deliver_per_plant_csv` | The general per-plant CSV door: per-image records, each carrying its image's capture `date`, its `stem` and its `value` (any other key refuses), with the `value_key` naming the quantity, aggregated per plant under the trait's confirmed `plant_aggregation` over the published buckets they came from, for the case neither specialist door's own composition covers; the required `plant_mapping` is resolved and verified against the delivered buckets, every delivered plant must be one it assigned on the delivered dates, each record counts toward the plant the mapping assigned its image, and every row's `plant_attribution` is the mapping's own. Every image the delivery reads of a delivered plant (attributed, with a document in its date's bucket, on disk) needs exactly one record, and a record of any other image refuses, naming it |
| `deliver_phenology_milestones` | Per-plant milestone CSV from buckets whose scope classifies the trait's positive state, and a plant mapping; its own column schema; reads the same images the per-plant door reads, and a plant with an assigned image it did not read gets no milestones; see `phenology` skill |
| `register_plant_registry` | Names a plant-locations CSV set once (per-file `sha256`/`n_plants`, `crop`, `site`, a content digest over the parsed rows), so `deliver_orthomosaic_plant_counts` and `build_plant_mapping` read the same registered version by name (`plant_registry`) instead of re-asserting file paths; a shapefile is converted first by `shp_to_plant_csv` |
| `deliver_orthomosaic_plant_counts` | Per-plant detection counts from a published whole-raster bucket plus a `plant_registry` name, delivered under the trait's `per_plant_orthomosaic_count` operationalization. Nearest-neighbor by default; `canopy_subject` switches to containment in an accepted canopy boundary instead (refused alongside a stated `nn_tolerance_m`). Fewer rows than the registry names can ship under either regime, the absent plants named on the delivery event |

A table or script you write yourself before a delivery door (the per-image records
`deliver_per_plant_csv` takes, for one) lives where the `project-setup` skill places your own work
for a project.

An unvalidated delivery ships only under a breeder's recorded acknowledgment of exactly the rows
and disclosure it writes. A breeder records one in the Results tab, never through these tools; each
delivery tool takes the recorded act by its `acknowledgment_id` and executes it.

`deliver_per_image_counts` produces a different, per-image CSV, not the per-plant schema above:
one row per document of the bucket, the `image` cell the source file name the bucket's record names
for it. A mosaic bucket refuses there, naming the per-plant door.

### Per-Image CSV Schema

Columns, in the order `deliver_per_image_counts_csv` writes them (see
`pipelines/postprocessing/export.py`):

| Column | Type | Description |
|--------|------|-------------|
| image | string | Source image file name, as the bucket's record names it |
| detection_count | int | Real detections in this image (a zero-extent box is never written, and the bucket records how many were dropped) |
| avg_confidence | float | Mean score across this image's detections |
| validated | bool | Whether the bucket's assessment cleared the one delivery gate |
| trait | string | The trait the delivery ships under |
| trait_revision | int | The confirmed trait revision the delivery ships under |
| trait_revision_sha256 | string | That revision's entry digest, the one the breeder confirmed |
| delivery_event_id | string | The delivery event recording this file's provenance |

## The meaning door (what the number is)

Before the gate below, every delivery answers a different question: what the delivered number
means. A trait is one entry per project, its spec fields and an operationalization per delivery
kind (what the number means, what decides it in the imagery, which subject it is about), proposed
as a revision with `propose_trait` and confirmed by the breeder as a whole in the Setup tab.
`crops.yml` gives a field criterion, which is not something a model can realize on its own. A
delivery ships under the trait's latest confirmed revision and names it on its delivery event; a
later, unconfirmed revision changes nothing until the breeder confirms it. No confirmed revision,
or one stating no operationalization for this delivery's kind, and the door refuses and names the
primitive that fixes it. An acknowledgment does not reach this: it says a number's error is
uncharacterized, which is a claim about a quantity that has been defined.

- `deliver_per_image_counts` takes a required `trait` and rests on its `per_image_count`
  operationalization.
- The per-plant doors take `delivered_phenotype`, a crop-vocabulary delivered-phenotype name, and
  resolve it to the trait whose latest confirmed revision `delivers` it: none or more than one
  refuses. `deliver_per_plant_csv`'s delivery kind, a kind whose per-plant value is the trait's
  `plant_aggregation`, names the operationalization, the records' value key has to be inside its
  confirmed set, and the delivery aggregates each plant's records under the revision's
  `plant_aggregation`. `deliver_orthomosaic_plant_counts` rests on the
  `per_plant_orthomosaic_count` operationalization, which names no value key: the door writes
  `count` itself.
- A `state_crossing_dates` operationalization and every delivery under it are checked against the
  delivered dataset's own subject registry, never a bare spec value: a delivery whose buckets'
  dataset carries no registry refuses, and a positive state naming an attribute the registry does
  not declare for the measured subject, or a value that attribute does not list, refuses the
  proposal, or the delivery when the registry changed since.

## The delivery gate (measurement integrity)

Every delivery clears one gate over the buckets it reads; a delivery over no bucket refuses. A
count or phenology delivery's every bucket must cover the operationalization's measured subject,
its own scope's subject. A bucket is validated when the
assessment it was published under passed, measured this delivery's kind under the revision the
delivery ships under, was measured over a reference that has not changed since, measured the
checkpoint and execution record the bucket's own record states, and covers the bucket's capture
(one of its captures for an image bucket, the same mosaic for a raster bucket). The gate refuses
outright buckets naming more than one producing checkpoint or run, staged proposals, a phenology
delivery over a bucket whose scope declares no attribute listing the positive state's value,
a dimensional detector delivery that names no physical-scale assessment, and a named
physical-scale assessment when the delivered values are in no physical unit. A named
physical-scale assessment that does not answer (it did not pass, was judged under another
revision, was measured over a reference that has changed since or does not cover every delivered
capture, is not a physical scale or not in the delivered unit, or will not read) is a finding like
an unvalidated bucket.

A delivery with any unvalidated finding refuses, one sentence per finding, and the refusal carries
the digest of the result it would write: its rows, population and plant-mapping disclosure
beside its kind, revision, producer and every finding. A breeder's acknowledgment,
recorded once in the Results tab under the identity the backend runs as, with a non-empty reason
and that digest, and audited as `delivery_acknowledged`, ships exactly that result with
`validated` false; one given for any other result refuses, and no tool or request from outside a
browser records one.

Every delivery composes its CSV and its one `delivery_events` record, validated against its schema
before anything is written, then writes the file and the record once: the door, the kind, the
revision, the delivered file and its digest, the producer once, each bucket's finding, the finding
for the named physical-scale assessment (its id when it answers, its id and the gate's sentence
when it does not, none when the delivery names none), whether it is validated, the
acknowledgment, the population, and the plant-mapping disclosure. Its one audit line, naming only
the record's `event_id`, files in the log of the dataset the buckets sit in.

A per-plant count from a whole raster is a raster-visible count, not a whole-tree total: where the
crop's own knowledge document states 2D occlusion undercounts a canopy-borne quantity, a raster
count inherits that undercount.

## Quality Control

Before delivery, verify:
1. Completeness: Every plant has values for all expected traits
2. Range: Values within biological plausibility. `crops.yml` carries no range field on any
   trait; the plausible range is the breeder's own account, elicited and checked at review
3. Outliers: Flag statistical outliers for manual review
4. Confidence: Reject predictions below a confidence operating point derived from the data in hand (per the "derive, don't pin" rule), not a frozen constant; the right cutoff varies by dataset, model, and trait
5. No duplicates: One value per plant per trait per date
