---
name: Measurement report
about: A delivered phenotype (a CSV row from deliver_per_image_counts, deliver_per_plant_csv,
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

## The delivery's own record

Every delivered CSV carries, beside its measurement, the columns `DELIVERY_COLUMNS` names in
`packages/tcip-mcp/src/tcip_mcp/delivery.py`, the same on every row. Copy their cells from one row;
a blank cell is itself informative, so report it as blank rather than omitting it. The row's
`delivery_event_id` is the key to the delivery event the platform recorded, which holds the
producer, each bucket's gate finding, the scale assessment and how the delivery shipped, among
the rest. The GUI's Results tab lists the events, and `tcip dump-store <project> <out_dir>`
writes each out as a file to attach.

- The `DELIVERY_COLUMNS` cells:
- The delivery event, if you can attach it:

## What is wrong

Describe the specific value(s) that are wrong or that overclaim confidence, and against what ground
truth or breeder judgment you are checking them.

## What the breeder expected

What the breeder told you the delivered number should mean, and how what you got differs from that.
Name the trait revision the delivery event records, whether the breeder had confirmed it, and quote
its operationalization statement for this delivery kind if you have it.
