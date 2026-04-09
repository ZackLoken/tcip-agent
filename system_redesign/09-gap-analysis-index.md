# Adaptation Gap Analysis — Index

Maps each claw-code subsystem to TCIP phenotyping needs.

## Consultation Answers (context for decisions)

| Question | Answer |
|----------|--------|
| Rust experience | Some — can read/modify, not fluent |
| Language choice | Rust agent + Python MCP |
| GUI | PyQt6 desktop |
| Colleague access | They write prompts to agent and review its work |
| Deployment | Single workstation (my desk) |
| Agent-annotation | Agent assists + human refines |
| MVP target | Hazelnut catkin phenology (4 traits: catkin_05per_date, catkin_50per_date, catkin_95per_date, catkin_elongation_date) |
| Existing code fate | Start fresh with learnings |
| Process model | GUI spawns both agent and MCP server as siblings |
| Subagents | From the start (4 modes: PipelineDesigner, CodeGenerator, TrainingOrchestrator, ResultsAnalyzer) |
| Two-layer pipeline | Both layers from the start, plus temporal aggregation |
| Schema approach | Drop schema.py/registry.py, YAML queried directly by MCP tools |
| Config levels | 4: defaults → user → project → CLI |
| Permissions | ReadOnly / WorkspaceWrite / FullAccess |

## Gap Analysis Documents

| # | Subsystem | File |
|---|-----------|------|
| 1 | Conversation Runtime | [01-runtime-adaptation.md](adaptation/01-runtime-adaptation.md) |
| 2 | Tool System | [02-tools-adaptation.md](adaptation/02-tools-adaptation.md) |
| 3 | API Client | [03-api-adaptation.md](adaptation/03-api-adaptation.md) |
| 4 | Permissions | [04-permissions-adaptation.md](adaptation/04-permissions-adaptation.md) |
| 5 | Sessions & Config | [05-sessions-config-adaptation.md](adaptation/05-sessions-config-adaptation.md) |
| 6 | MCP Subsystem | [06-mcp-adaptation.md](adaptation/06-mcp-adaptation.md) |
| 7 | CLI → GUI | [07-gui-adaptation.md](adaptation/07-gui-adaptation.md) |
| 8 | New: ML Domain | [08-ml-domain-additions.md](adaptation/08-ml-domain-additions.md) |
| 9 | New: Schema/Registry Redesign | [09-schema-registry-redesign.md](adaptation/09-schema-registry-redesign.md) |
| 10 | New: Skills Architecture | [10-skills-architecture.md](adaptation/10-skills-architecture.md) |
