---
mode: agent
description: "Export per-plant CSV results from inference"
tools: ["tcip-pipeline"]
---

Export per-plant CSV deliverables:

1. Use `inspect_project` to verify inference results exist
2. Write a logged `scripts/` script that chains `run_inference` with the importable
   postprocessing libs (`aggregate_per_plant` / `export_aggregated_csv`) to generate the
   per-plant CSV (see the `delivery` and `pipeline-design` skills for the schema and chaining
   pattern — `tabulate_counts` produces a different, per-image count CSV, not this schema)
3. The export refuses a bare write (the count/value is a phenotype): pass `pred_dirs` so the count
   operating point's validity is read from each bucket's `operating_point.json` (or
   `measurement_validated` for a continuous/ordinal trait), or `acknowledge_unvalidated=True` for an
   honestly-flagged provisional CSV — see the delivery gate in the `delivery` skill
4. Verify CSV completeness: all plants have values for expected traits
5. Check value ranges against crop trait definitions
6. Flag outliers and low-confidence predictions for review
