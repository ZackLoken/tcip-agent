---
mode: agent
description: "Export per-plant CSV results from inference"
tools: ["tcip-pipeline"]
---

Export per-plant CSV deliverables:

1. Use `get_project_status` to verify inference results exist
2. Use `run_pipeline` with an `aggregation`-type phase to generate the per-plant CSV (see the
   `delivery` skill for the schema — `export_results_csv` produces a different, per-image
   count CSV, not this schema)
3. Verify CSV completeness: all plants have values for expected traits
4. Check value ranges against crop trait definitions
5. Flag outliers and low-confidence predictions for review
