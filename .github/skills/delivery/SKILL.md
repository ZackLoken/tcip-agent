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

Different trait types require different aggregation from per-image to per-plant:

Examples use real `crops.yml` trait names — verify any trait against `crops.yml` before use.

| Trait Type | Aggregation | Example |
|-----------|-------------|---------|
| Count | Median across images | `stem_count` → median across the plant's images |
| Date (bloom) | Elongated-fraction crossing | `catkin_50per_date` → date the elongated fraction crosses 50% (see `phenology` skill) |
| Ordinal | Mode | `efb_damage` → most common rating |
| Continuous | Mean | `fruit_diameter` → average across images |
| Area | Sum / 3D model | `plant_surface_area` → planimetric crown area (3D canopy model; not a raw 2D pixel sum) |

## Tools

| Tool | Purpose |
|------|---------|
| `export_results_csv` | Generate per-plant CSV from inference results |
| `export_predictions_yolo` | Export predictions in YOLO format |
| `run_inference` | Run batch inference on images |
| `compute_phenology` | Per-plant bloom CSV (05/50/95-per-date) from classified preds + plant mapping — its own column schema; see `phenology` skill |

## Quality Control

Before delivery, verify:
1. **Completeness**: Every plant has values for all expected traits
2. **Range**: Values within biological plausibility (e.g., fruit diameter 1-15cm)
3. **Outliers**: Flag statistical outliers for manual review
4. **Confidence**: Reject predictions below a confidence operating point **derived from the data in hand** (per the "derive, don't pin" rule) — not a frozen constant; the right cutoff varies by dataset, model, and trait
5. **No duplicates**: One value per plant per trait per date
