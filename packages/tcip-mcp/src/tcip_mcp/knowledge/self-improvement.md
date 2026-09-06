---
name: self-improvement
description: "How the agent turns friction and findings into a durable record that the next session picks up. Load this whenever the user pushes back, corrects you, repeats an annoyance, when you discover machinery you didn't know existed, nearly reinvent something, or hit a missing tool/skill, and at the end of substantial work. Everything lands with the project, via report_friction and write_retrospective, and load_project_memory reads it back."
---

# Self-improvement: learning that stays with the project

Learning lands with the project it came from, in that project's `.tcip/` store. Nothing a
session learns writes into the repo: skills, `CLAUDE.md`, and platform code change only when the
owner changes them.

Where something is genuinely general, the owner is the transport: they read these records and
decide what becomes a platform change.

This document is itself one of the platform's domain knowledge documents. Claude Code reaches it
as a generated skill; a client without generated skills reaches the identical text through the
`serve_domain_knowledge` tool.

The loop is three live, audited tools, no separate journal file:

| When | Tool | Lands |
|---|---|---|
| The moment friction happens | `report_friction` | `.tcip/reports/` |
| End of substantial work, even if incomplete | `write_retrospective` | `.tcip/retrospectives/` |
| Start of the next session | `load_project_memory` (`kind='reports'`, then `'retrospectives'`) | read back into context |

## Capture: the moment friction happens

Call `report_friction` when you notice any of these. One line is enough in the moment; the free-text
`detail` matters far more than the category.

- Pushback / correction: "no, do it this way", "don't do X", "actually it's Y". Call
  `report_friction` with `user_disagreement=True` for these; it's a separate signal from
  `category`.
- Repetition: you're told the same thing a second time.
- Wrong assumption: something you assumed about the data, domain, or tools turned out false
  (e.g. "GPS is too coarse for per-plant" when the plant grid is RTK-accurate).
- Re-discovery: you spent real effort finding machinery that already existed.
- Near-reinvention: you were about to write something the toolkit already does.
- Missing / hard tool: a capability that should exist but is buried in a web route, a script,
  or nowhere.
- Environment / setup friction: a missing dependency, a stale path, a slow default.
- A blocked or failed mandated action: a ritual call that errored, a guard that denied a
  read-only command, `doctor.py` refusing to run. Never skip one silently.

Over-report.

## Record: what the retrospective should contain

At the end of substantial work, call `write_retrospective`. Write what a future session working
this project would need, and be honest about what failed.

- What you measured about this dataset: object scale and elongation from the GT, capture
  cadence and missing dates, class imbalance, where the operating point resolved and on what
  reference, what the validation gate said.
- What you tried that did not work: dead ends, approaches abandoned and why.
- Assumptions that turned out wrong.
- Domain questions for the breeder: anything about what a trait *means* that you could not
  settle from `crops.yml`. Trait semantics are the expert's to confirm; surface the question,
  never invent the answer.
- Tools that were missing or unusable.

### What must not go in a retrospective

- A reusable pipeline shape. Record what you *measured* that led you to a shape, not the shape.
- An inventory of existing machinery. Read the source, or run `scripts/list_tools.py`.

## Distill: closing the loop toward a platform change

`scripts/distill_learnings.py` gathers one project's (or, with `--workspace`, every project's)
reports and retrospectives into a worksheet of recurring themes; it only reads, nothing is
written, applied, or promoted. After reviewing a worksheet, call `record_distillation_pass(project_path)` per project covered:
the one audited write in this loop, kept out of the script. It only resets that
project's distillation-backlog counters; turning a recurring theme into a skill line, a
`CLAUDE.md` rule, or a tool change stays your own separate edit, per the scoping above.

## Reading the record back

`read_audit_log(scope=None, tool=None, since=None, until=None, status=None, limit=200)` answers
a different question than the two memory tools above: not what a session learned, but which
door touched a dataset, a project, or the platform log, when, and with what status. It reads the
same log every `@audited` tool and `record_event` call already writes to, filtered in memory,
newest entries first; a corrupt or unreadable page is refused rather than answered partially.

## Honesty

Report what you actually did. If you skipped a capture, say so.
