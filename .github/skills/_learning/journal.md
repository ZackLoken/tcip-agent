# Learning journal

A **rolling capture buffer** of undistilled friction → proposed repo improvements (see the
`self-improvement` skill). Capture the moment friction happens (observation → proposal → status);
at the end of substantial work, distill open entries into skills / `CLAUDE.md` diffs / tool
proposals for the owner to approve, then **prune** the graduated entry — add a line under
**Graduated** and move its text verbatim to `archive/<YYYY-MM>.md`. Keep this file short: only
**open** items live here; the full history lives in `archive/`.

---

## Graduated (distilled → durable artifact; full narratives in `archive/2026-07.md`)

- **L1 / L9** — phenology made agent-facing → `phenology` skill + `build_plant_mapping` /
  `compute_phenology` tools + `pipelines/postprocessing/phenology.py`
- **L2** — `run_inference(tile=True)` already does SAHI tiling → `phenology` skill note
- **L3** — the tcip MCP tools need a repo-root cwd → CLAUDE.md startup note
- **L4 + audits** — invented bbox-height & count-of-peak "science" purged → CLAUDE.md
  measurement-integrity invariant (the first hard rule)
- **L5** — RTK-grid vs fuzzy image-GPS framing + carry genotype through → `phenology` /
  `crops/hazelnut` / `crop-science`
- **L6 / L7 / Lo2 / Lo3** — CUDA preflight, path portability, `evaluate_model` iou_type default,
  `$TCIP_PROJECT_ROOT` state anchoring → applied in code
- **Session-2 fence + GUI** (G-A/B/C, terminal copy) → PowerShell + Bash exec/write guards,
  `focus_annotate` tool, dataset-select advisories
- **Skills fabrication** — 6 crop + `crop-science` skills rebuilt from `crops.yml` + cited research;
  `verify_skill_traits.py` + `tests/test_skill_trait_fidelity.py` guardrail
- **L-SKILL-1** — `catkin_elongation_date` = the 95% majority crossing → `phenology.plant_milestones`
  + phenology skill (provisional, pending breeder confirmation)
- **L-SKILL-2** — trait-fidelity guardrail made permanent CI → `tests/test_skill_trait_fidelity.py`

---

## Open — undistilled friction (act, then graduate + prune)

### L-SKILL-3-residual — deferred non-core skill edits (2026-07-14)
Two non-core skill body edits were noted on the skills-rebuild pass but not applied:
1. `pipeline-design` Patterns C/E need explicit **out-of-2D-scope** notes (they imply capabilities
   beyond the current 2D-only scope).
2. `model_spec` guidance is **duplicated** between `pipeline-design` and `training` — de-dup to one home.

Low priority; apply or consciously discard on the next skills pass.

### self-improve-commit-model — breeders shouldn't commit to the repo (2026-07-15)
The journal reaches every machine by being **git-committed**, which assumes commit access — fine for
the owner, but breeder users shouldn't be committing to the repo. Viable future model (the plumbing
half-exists): breeders' friction lands only in machine-local `.tcip/learning_capture.jsonl` (the
`SessionEnd` hook) + `scripts/distill_learnings.py`; the **owner is the sole distiller/committer**, so
learning still reaches the team without breeders touching git. Explore once multi-user is real.

> **Deferred by owner (pointer only, not agent-distillable):** a real execution **sandbox** to
> replace the string-matching terminal fence, and an *enforced* SessionEnd capture hook — governance
> decisions tracked in the platform status / governance plan, not here.
