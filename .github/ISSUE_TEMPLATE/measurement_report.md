---
name: Measurement report
about: A delivered phenotype (a CSV row from tabulate_counts, export_aggregated_csv,
  deliver_phenology_milestones, or deliver_orthomosaic_plant_counts) is wrong or overclaims
title: "[measurement] "
labels: measurement
---

Measurement integrity is the platform's highest rule, so this report gets read closely. Fill in
everything you have; leave the rest blank rather than guessing.

## Project and data

- Project:
- Dataset capture date (the `images/<YYYY-MM-DD>/` bucket the delivery drew from):
- Subject (the class registry name the delivery measured):
- Delivered phenotype or trait name:

## The delivery's own provenance tail

Copy the values from the delivered CSV's own columns; a blank cell is itself informative, so
report it as blank rather than omitting the row.

- `producer_model_sha256`:
- `producing_experiment_id`:
- `produced_at`:
- `validation_record`:
- `operating_point_conf` / `operating_point_validated`:
- `positive_state_classifier_validated` (phenology deliveries only):
- `unvalidated_dimensions`:
- `plant_mapping_sha256` (phenology deliveries only):
- `captures_unverified` / `plant_csvs_unverified` (phenology deliveries only):
- `dates_delivered` / `images_unattributed` (phenology deliveries only):
- `plant_attribution`:

## What is wrong

Describe the specific value(s) that are wrong or that overclaim confidence, and against what
ground truth or breeder judgment you are checking them.

## What the breeder expected

What the breeder told you the delivered number should mean, and how what you got differs from
that. If `state_trait_operationalization` was confirmed for this trait and delivery kind, say
so and quote the confirmed statement if you have it.
