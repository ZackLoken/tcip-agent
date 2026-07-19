---
mode: agent
description: "Initialize a new crop analysis project"
tools: ["tcip-pipeline"]
---

Initialize a new TCIP project. See `.github/skills/project-setup` for the full front-door
arc (naming convention, `ingest_images`, format confirmation, `set_active_project`) — this is
the short form:

1. Name the project `{crop}_{trait}_{site}` (ask the human for anything missing — crop, trait,
   or site — don't invent it)
2. Use `ingest_images` to structure the raw photo pile into the canonical layout (it scaffolds
   the project's `.tcip/` too, so a separate `init_project` isn't needed here)
3. Use `inspect_project` to verify the project structure
4. Use `scan_dataset` to explore any existing data
5. Use `validate_data_quality` to check data integrity
6. `set_active_project` once the project is ready, so the GUI opens what you built
7. Recommend next steps based on data availability
