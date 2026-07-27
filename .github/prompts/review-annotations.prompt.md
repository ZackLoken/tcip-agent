---
mode: agent
description: "Review annotations and assess label quality"
tools: ["tcip-pipeline"]
---

Start an annotation review session. See `.github/skills/annotation` for the full review
channel — the agent proposes on canvas and never writes GT the human hasn't seen:

1. Use `scan_dataset` to identify the dataset
2. Use `read_annotations` to load ground truth labels
3. Use `score_predictions` (`detail=True`) to match predictions against ground truth and
   compute precision/recall/AP
4. For agent-proposed detections (not yet predictions from a model run), `stage_proposals`
   them to the predictions tree — never write to `annotations/` directly
5. Use `push_panel_data` or `focus(tab='review')` to send the human straight to the flagged frames
6. Use `render_failure_cases` to surface likely-bad frames (count-mismatch heuristic, not
   IoU-matched); confirm with `score_predictions`(`detail=True`) before deciding what to correct
7. Use `prioritize_review_queue` for active learning prioritization
