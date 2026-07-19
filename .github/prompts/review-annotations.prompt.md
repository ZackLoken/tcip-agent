---
mode: agent
description: "Review annotations and assess label quality"
tools: ["tcip-pipeline"]
---

Start an annotation review session. See `.github/skills/annotation` for the full review
channel — the agent proposes on canvas and never writes GT the human hasn't seen:

1. Use `load_dataset` to identify the dataset
2. Use `load_annotations` to load ground truth labels
3. Use `evaluate_predictions` (`detail=True`) to match predictions against ground truth and
   compute precision/recall/AP
4. For agent-proposed detections (not yet predictions from a model run), `stage_proposals`
   them to the predictions tree — never write to `annotations/` directly
5. Use `push_panel_data` or `focus_review` to send the human straight to the flagged frames
6. Identify images needing correction with `get_worst_predictions`
7. Use `prioritize_review_queue` for active learning prioritization
