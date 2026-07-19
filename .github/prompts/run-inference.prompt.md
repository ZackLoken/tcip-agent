---
mode: agent
description: "Run inference on images with a trained model"
tools: ["tcip-pipeline"]
---

Run batch inference:

1. Use `inspect_project` to check current state
2. Use `select_best_model` or `list_registered_models` to find the model
3. Use `run_inference` to generate predictions
4. Use `export_predictions` to save per-image JSON prediction files
5. Use `push_panel_data` to visualize results in the inference panel
6. Use `score_predictions` (on the dataset dir) if ground truth is available for comparison
