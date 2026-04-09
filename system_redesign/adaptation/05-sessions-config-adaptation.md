# 05 — Sessions & Config Adaptation

## Sessions — carries over as-is

- **JSONL format** — append-only, durable. Keep exactly.
- **Session struct** — id, timestamps, messages, compaction, fork. All useful.
- **ConversationMessage / ContentBlock** — universal message format. Keep.
- **Rotation** (> 256KB → backup files). Keep.
- **Fork** — create a branch of a conversation. Useful for "try different approach."

### Modification: project-scoped sessions
Claw-code sessions are per-working-directory. We want sessions scoped to a
**project** (crop + trait combination). Directory structure:

```
~/.tcip/sessions/
├── hazelnut_catkin_phenology/
│   ├── session_001.jsonl
│   └── session_002.jsonl
├── chestnut_burr_detection/
│   └── session_001.jsonl
```

This lets the agent resume context for a specific trait pipeline across work days.

## Config — adaptation needed

### Claw-code's 5 levels
```
~/.claw.json → ~/.claw/settings.json → .claw.json → .claw/settings.json → .claw/settings.local.json
```

### Our 4 levels (simplified for single workstation)
```
1. Built-in defaults                  (compiled into Rust agent + MCP server)
2. ~/.tcip/settings.json              (user preferences: model, API key ref, theme)
3. <project>/.tcip/config.json        (project: crop, trait, pipeline config, GPU settings)
4. CLI flags                           (runtime overrides)
```

Drop the "legacy" and "local" levels. Single workstation = no need for separate
per-machine overrides. GPU settings go in project config.

### Config fields specific to TCIP

```json
{
  "api": {
    "provider": "anthropic",
    "model_default": "claude-sonnet-4-20250514",
    "model_reasoning": "claude-opus-4-20250514",
    "max_budget_usd": 5.0
  },
  "training": {
    "device": "cuda:0",
    "default_batch_size": 16,
    "tensorboard_dir": "runs/",
    "checkpoint_dir": "checkpoints/"
  },
  "annotation": {
    "default_format": "yolo_txt",
    "polygon_mode": true,
    "iou_threshold": 0.5,
    "confidence_threshold": 0.25
  },
  "permissions": {
    "mode": "workspace-write",
    "allow": ["list_crops", "get_trait_info", "read_file"],
    "ask": ["launch_training", "launch_hpo", "register_model"],
    "deny": []
  },
  "mcp": {
    "pipeline_server": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "tcip_mcp"]
    }
  }
}
```
