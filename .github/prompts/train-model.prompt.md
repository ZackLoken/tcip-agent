---
mode: agent
description: "Configure and launch a model training run"
tools: ["tcip-pipeline"]
---

Help configure and launch a training run:

1. Use `get_project_status` to understand current state
2. Use `load_dataset` to check available data
3. Use `recommend_model` for architecture guidance
4. Build a training config with model_spec, data, and training sections
5. Use `validate_config` to verify the configuration
6. Use `create_experiment` to set up experiment tracking
7. Use `launch_training` to start the run
8. Monitor with `check_training_status`
