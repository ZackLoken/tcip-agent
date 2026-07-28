---
name: delivery
description: "How to deliver phenotyping results — the per-plant CSV schema, per-image-to-per-plant aggregation rules by trait type, the export tools, and pre-delivery quality control. Load when exporting results, building a per-plant deliverable, aggregating per-image measurements to per-plant, or quality-checking a breeding-program CSV before hand-off."
---

# Results Delivery

## Per-Plant CSV Schema

The final deliverable is a CSV with one row per plant per trait:

| Column | Type | Description |
|--------|------|-------------|
| plant_id | string | Unique plant identifier |
| crop | string | Crop species |
| trait_name | string | Measured trait (from crop skill) |
| value | float/string | Measurement value |
| confidence | float | Model confidence [0,1] |
| n_images | int | Number of source images |
| pipeline_version | string | Pipeline that produced this result |

## Aggregation Rules

Aggregating per-image measurements into a per-plant value is an agent **choice keyed to the trait's
read-semantics**, not a frozen one-function-per-type map. The table below is a starting reference of
common choices — pick (or compose) the aggregation the trait's definition actually calls for (a
skewed count may want a median, a robust mean, or a yield-model estimate; a date wants a crossing),
and record which you used. When the trait carries read-semantics fields (`TraitSpec`), those govern.

Examples use real `crops.yml` trait names — verify any trait against `crops.yml` before use.

| Trait Type | Common choice | Example |
|-----------|-------------|---------|
| Count | Median across images | `stem_count` → median across the plant's images |
| Date (bloom) | Elongated-fraction crossing | `catkin_50per_date` → date the elongated fraction crosses 50% (see `phenology` skill) |
| Ordinal | Mode | `efb_damage` → most common rating |
| Continuous | Mean | `fruit_diameter` → average across images |
| Area | Sum / 3D model | `plant_surface_area` → planimetric crown area from a 3D canopy model — **out of current build scope** (3D LiDAR/SfM is not built today; see CLAUDE.md Scope). A validated 2D mask yields a calibrated pixel area, not this trait |

## Tools

| Tool | Purpose |
|------|---------|
| logged `scripts/` script | Produces the Per-Plant CSV Schema above by chaining `launch_training` + `run_inference` then the importable postprocessing libs `aggregate_per_plant` / `export_aggregated_csv` — see `pipeline-design` skill for the chaining mechanics |
| `export_predictions` | Export predictions as per-image JSON (COCO-shaped). A bucket (`output_dir`) with review verdicts is immutable — the export redirects to a fresh `<dir>@r2` bucket (see the response's `output_dir`); `overwrite=True` forces in-place but is refused when verdicts exist |
| `run_inference` | Run batch inference on images |
| `compute_phenology` | Per-plant bloom CSV (05/50/95-per-date) from classified preds + plant mapping — its own column schema; see `phenology` skill |
| `materialize_review_dataset` | Turn human review verdicts into a curated training set for re-delivery after correction — see `annotation` skill |

`tabulate_counts` produces a **different**, per-image `image, detection_count,
avg_confidence` CSV — not the per-plant schema above. Don't reach for it when the
per-plant schema is what's wanted.

## The delivery gate (measurement integrity)

Every phenotype-delivery door refuses a **bare write** — an unvalidated measurement number with no
acknowledgement. The count/date/value is the phenotype, so each door runs one shared
`check_delivery_gate`: it ships only when each measurement dimension the deliverable rests on is
validated against a reference sized to the trait (held-out GT **or** a breeder-confirmed output
sample — see the `evaluation` and `cv-research` skills), read from the predictions' own
`operating_point.json` sidecar, not a caller-asserted string.

- `compute_phenology` gates both the elongation classifier (reconciled from
  `classifier_operating_point.json` — see `calibrate_classifier_operating_point`) and the count
  operating point; the web `/export_csv` phenology branch gates the same, per row.
- `tabulate_counts` gates the count operating point on the run's resolved bundle. `export_detection_csv`
  and `export_aggregated_csv` gate at the writer: pass `pred_dirs` for a count trait so the validity is
  reconciled from each bucket's `operating_point.json` sidecar. **`measurement_validated` alone, with
  no `pred_dirs`, is never trusted as a bare caller string** — a continuous/ordinal trait has no
  on-disk measurement-validity producer today, so without `pred_dirs` the only route to delivery is
  the explicit acknowledge below.
- To ship a **provisional** result, pass `acknowledge_unvalidated=True`: the door writes but stamps
  `measurement_validated=false` so the un-trustworthiness travels with the CSV. This is for an honest
  provisional delivery, never for silently shipping a bare number.

The **prediction-bucket writers stay ungated but honestly stamped** (`export_predictions` /
`run_inference`): the review loop must be able to produce unvalidated predictions to *reach* a
validated measurement — the gate lives at the final phenotype door, not on the intermediate labels.

## Quality Control

Before delivery, verify:
1. **Completeness**: Every plant has values for all expected traits
2. **Range**: Values within biological plausibility (e.g., fruit diameter 1-15cm)
3. **Outliers**: Flag statistical outliers for manual review
4. **Confidence**: Reject predictions below a confidence operating point **derived from the data in hand** (per the "derive, don't pin" rule) — not a frozen constant; the right cutoff varies by dataset, model, and trait
5. **No duplicates**: One value per plant per trait per date
