# 04 — Permissions Adaptation

## What claw-code has

5 permission modes (ReadOnly → DangerFullAccess → Prompt → Allow).
PermissionPolicy with allow/deny/ask rule patterns.
PermissionEnforcer with workspace boundary + bash validation.
PermissionPrompter trait for UI-specific prompting.

## What carries over as-is

- **Permission modes** — the ReadOnly/WorkspaceWrite/FullAccess hierarchy
  maps directly. Registry queries = ReadOnly. File writes = WorkspaceWrite.
  Training launch = FullAccess.
- **Per-tool permission requirements** — each tool declares its minimum mode.
- **PermissionPrompter trait** — we implement it in the GUI bridge instead of terminal.
- **Deny/Allow/Ask rules** — configurable patterns. Useful for locking down
  what colleagues can do vs what we can do.

## What needs modification

### GUI-based permission prompting
Claw-code prompts in the terminal. We route permission requests over the stdio
bridge to PyQt6 and show a dialog:

```
Agent wants to: launch_training
Config: hazelnut_catkin_phenology_v2.yaml
Estimated GPU time: ~4 hours
[Allow] [Deny] [Review Config First]
```

The `PermissionPrompter` trait is already the right abstraction. We just implement
it differently — serialize the request as JSON over stdio, deserialize the response.

### Colleague-safe mode
Colleagues write prompts and review work. They should NOT be able to:
- Delete datasets or trained models
- Modify registry/schema
- Run arbitrary bash commands
- Change system configuration

Default colleague mode: `WorkspaceWrite` with deny rules for destructive ops.
My mode: `FullAccess` (or `Allow` when I'm confident).

This is already supported by claw-code's permission rules:
```json
{
  "deny": ["bash rm *", "bash git push *"],
  "ask": ["launch_training", "launch_hpo"],
  "allow": ["list_crops", "get_trait_info", "read_file"]
}
```

### HITL checkpoints as permission gates
Pipeline checkpoints (pipeline design, pipeline config review, training launch, results
review, model deployment) can be implemented as permission checks on specific tools.
When the agent calls `launch_training`, the permission system routes it to the
GUI as a rich checkpoint dialog rather than a simple allow/deny prompt.

This is a natural extension of the PermissionPrompter — the prompter inspects the
tool name and renders a checkpoint-specific UI when appropriate.

## What's new

### Audit trail
Every permission decision (allow, deny, who, when, what tool, what input) should
be logged to a JSONL audit file. Claw-code has telemetry but not a compliance-grade
audit trail. Add this for production accountability.
