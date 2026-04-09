---
name: delivery
description: "Per-plant CSV output format, aggregation from predictions to plant-level measurements, QC checks, and breeder-facing deliverable requirements."
triggers:
  - CSV
  - output
  - deliver
  - aggregate
  - per-plant
  - breeder
  - genotype
  - results
  - export
  - final
modes: [PipelineDesigner, ResultsAnalyzer, CodeGenerator]
priority: high
max_chars: 3000
---

# Delivery

## Purpose

Define the final per-plant CSV deliverable that breeders consume. Every pipeline must end with a CSV mapping predictions back to individual plants/genotypes.

## Per-Plant CSV Schema

Required columns:

| Column | Type | Description |
|--------|------|-------------|
| plant_id | string | Unique plant/genotype identifier |
| crop | string | One of: hazelnut, chestnut, currant, elderberry, persimmon, black_locust |
| trait_name | string | Exact trait name from registry (e.g. `catkin_50per_date`) |
| value | numeric/string | Predicted measurement (count, date, ordinal score, etc.) |
| confidence | float | Model confidence or prediction uncertainty [0, 1] |
| n_images | int | Number of images contributing to this prediction |
| pipeline_version | string | Git hash or version tag of the pipeline that produced this |

Optional columns: `date_collected`, `block`, `row`, `position`, `notes`.

## Aggregation Rules

### Count traits (detection → count)
Per image: count detections above confidence threshold.
Per plant: median count across images (robust to outliers). Report n_images.

### Date traits (detection → temporal sigmoid)
Collect per-date counts → fit sigmoid → extract threshold dates (5%, 50%, 95%).
Value = date string (YYYY-MM-DD). Confidence = sigmoid R² fit quality.

### Ordinal traits (classification → score)
Per image: predicted class (integer 0-9).
Per plant: mode (most frequent prediction). If tied, use mean and round. Report agreement ratio as confidence.

### Continuous traits (regression)
Per image: predicted value.
Per plant: mean across images. Confidence = 1 - (std / mean) if mean > 0.

### Area/volume traits (segmentation → measure)
Per image: sum of instance mask areas (in calibrated units).
Per plant: mean across images.

## QC Checks Before Delivery

Run before handing CSV to breeders:

1. **Completeness**: every plant_id in the plot map has a row (flag missing)
2. **Range check**: values within biologically plausible range for the trait (query registry)
3. **Outlier flag**: values > 3σ from population mean get `outlier=True` column
4. **Confidence floor**: flag rows where confidence < 0.5 — breeder should verify
5. **No duplicates**: one row per (plant_id, trait_name) pair
6. **Format**: CSV with UTF-8 encoding, header row, no trailing commas

## Delivery Checkpoint

HITL checkpoint #5: present summary statistics before delivering CSV:
- Total plants, total traits
- Completeness % (plants with predictions / total plants)
- Mean confidence across all predictions
- Number of outlier-flagged rows
- Sample of 5 rows for visual inspection

## Key Constraints

- Every pipeline MUST produce a per-plant CSV — this is the deliverable
- Use exact trait names from registry — never abbreviate or rename
- Breeders rank genotypes by trait values — relative accuracy matters more than absolute
- Include confidence so breeders can filter low-quality predictions
- CSV must be reproducible: same inputs + same pipeline version = same output
