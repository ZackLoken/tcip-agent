---
name: self-improvement
description: "How the agent turns friction and findings into a durable record that the next session picks up. Load this whenever the user pushes back, corrects you, repeats an annoyance, when you discover machinery you didn't know existed, nearly reinvent something, or hit a missing tool/skill, and at the end of substantial work. Everything lands with the project, via claude_reports and project_retrospective, and load_project_memory reads it back."
---

# Self-improvement: learning that stays with the project

Learning lands **with the project it came from**, in that project's `.tcip/` store. Nothing a
session learns writes into the repo: skills, `CLAUDE.md`, and platform code change only when the
owner changes them.

That scoping is deliberate. What a session learns is almost always *about this data*: this block's
object scale, this season's capture cadence, where the operating point resolved and why. Recording
it project-side keeps it attached to the dataset that produced it, and means the next project
derives its own answer from its own data rather than inheriting one measured somewhere else. Where
something is genuinely general, the owner is the transport: they read these records and decide what
becomes a platform change.

The loop is three live, audited tools, no separate journal file:

| When | Tool | Lands |
|---|---|---|
| The moment friction happens | `claude_reports` | `.tcip/reports/` |
| End of substantial work, even if incomplete | `project_retrospective` | `.tcip/retrospectives/` |
| Start of the next session | `load_project_memory` (`kind='reports'`, then `'retrospectives'`) | read back into context |

## Capture: the moment friction happens

Call `claude_reports` when you notice any of these. One line is enough in the moment; the free-text
`detail` matters far more than the category.

- **Pushback / correction**: "no, do it this way", "don't do X", "actually it's Y". Call
  `claude_reports` with `user_disagreement=True` for these; it's a separate signal from
  `category`, so a later distill pass can pull every place the owner and you disagreed out of the
  pile on its own, rather than mixed into general friction.
- **Repetition**: you're told the same thing a second time.
- **Wrong assumption**: something you assumed about the data, domain, or tools turned out false
  (e.g. "GPS is too coarse for per-plant" when the plant grid is RTK-accurate).
- **Re-discovery**: you spent real effort finding machinery that already existed. If you had to
  dig, so will the next session.
- **Near-reinvention**: you were about to write something the toolkit already does.
- **Missing / hard tool**: a capability that should exist but is buried in a web route, a script,
  or nowhere.
- **Environment / setup friction**: a missing dependency, a stale path, a slow default.
- **A blocked or failed mandated action**: a ritual call that errored, a guard that denied a
  read-only command, `doctor.py` refusing to run. Never skip one silently.

Over-report. A report is cheap; a silent guess is not.

## Record: what the retrospective should contain

At the end of substantial work, call `project_retrospective`. Write what a future session working
**this project** would need, and be honest about what failed; that is the most useful part.

- **What you measured about this dataset**: object scale and elongation from the GT, capture
  cadence and missing dates, class imbalance, where the operating point resolved and on what
  reference, what the validation gate said.
- **What you tried that did not work**: dead ends, approaches abandoned and why. This is what
  stops the next session repeating them.
- **Assumptions that turned out wrong.**
- **Domain questions for the breeder**: anything about what a trait *means* that you could not
  settle from `crops.yml`. Trait semantics are the expert's to confirm; surface the question,
  never invent the answer.
- **Tools that were missing or unusable.**

### What must not go in a retrospective

- **A reusable pipeline shape.** A decomposition that worked here becomes a recipe for a problem it
  was never measured against, the ceiling this platform exists to avoid. Record what you *measured*
  that led you to a shape, not the shape.
- **An inventory of existing machinery.** It goes stale the first time the source moves. Read the
  source, or run `scripts/list_tools.py`.

## Distill: closing the loop toward a platform change

`scripts/distill_learnings.py` gathers one project's (or, with `--workspace`, every project's)
reports and retrospectives into a worksheet of recurring themes; it only reads, nothing is
written, applied, or promoted. After reviewing a worksheet, call `record_distillation_pass(project_path)` per project covered:
the one audited write in this loop, kept out of the script on purpose. It only resets that
project's distillation-backlog counters; turning a recurring theme into a skill line, a
`CLAUDE.md` rule, or a tool change stays your own separate edit, per the scoping above.

## Honesty

Report what you actually did. If you skipped a capture, say so. A learning record that pretends to
have learned is worse than none; its whole value is that the next session, and the owner, can trust
it.
