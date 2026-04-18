---
mode: agent
description: "Review annotations and assess label quality"
tools: ["tcip-pipeline"]
---

Start an annotation review session:

1. Use `load_dataset` to identify the dataset
2. Use `load_annotations` to load ground truth labels
3. Use `run_matching` to match predictions against ground truth
4. Use `evaluate_detections` to compute precision/recall/AP
5. Use `push_panel_data` to send review data to the review panel
6. Identify images needing correction with `get_worst_predictions`
7. Use `get_review_queue` for active learning prioritization
