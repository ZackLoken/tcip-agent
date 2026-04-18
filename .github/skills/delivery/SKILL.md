---
name: delivery
description: "Per-plant CSV export, results delivery, aggregation rules by trait type, and quality control for breeding program deliverables."
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

| Trait Type | Aggregation | Example |
|-----------|-------------|---------|
| Count | Median across images | catkin_count → median of 5 flight images |
| Date | Sigmoid curve fitting | bloom_date → inflection point of sigmoid fit |
| Ordinal | Mode | disease_severity → most common rating |
| Continuous | Mean | fruit_diameter → average across images |
| Area | Sum | canopy_area → sum of segmented pixels |

## Tools

| Tool | Purpose |
|------|---------|
| `export_results_csv` | Generate per-plant CSV from inference results |
| `export_predictions_yolo` | Export predictions in YOLO format |
| `run_inference` | Run batch inference on images |

## Quality Control

Before delivery, verify:
1. **Completeness**: Every plant has values for all expected traits
2. **Range**: Values within biological plausibility (e.g., fruit diameter 1-15cm)
3. **Outliers**: Flag statistical outliers for manual review
4. **Confidence**: Reject predictions below confidence threshold (default: 0.3)
5. **No duplicates**: One value per plant per trait per date
