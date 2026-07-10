---
name: self-improvement
description: "How the agent turns friction into durable, team-shared platform improvements. Load this whenever the user pushes back, corrects you, repeats an annoyance, when you discover machinery you didn't know existed, nearly reinvent something, or hit a missing tool/skill — and at the end of substantial work. Captures the signal to the learning journal and distills it into proposed skills / CLAUDE.md edits / tool changes for the human to approve. The goal: the platform gets better without the owner in the loop."
---

# Self-improvement — learning that reaches the whole team

The owner (Zack) is stepping back; the breeders direct Claude as their ML/CV engineer. For
that to hold **without him translating**, what you learn has to land in the **repo**
(`.github/skills/`, `CLAUDE.md`, tools) — the only place that reaches every breeder's
machine. Your `~/.claude` memory and a project's `.tcip/` retrospectives are *machine-local*
and never travel. This skill is how a correction today becomes a capability tomorrow.

## Capture — the moment friction happens

Append to the journal (`.github/skills/_learning/journal.md`) the moment you notice any of:

- **Pushback / correction** — the user says "no, do it this way", "don't do X", "actually
  it's Y". The thing you got wrong is a candidate rule.
- **Repetition** — you're told the same thing a second time (a sign it belongs in a skill or
  CLAUDE.md, not just this conversation).
- **Wrong assumption** — you assumed something about the data/domain/tools that turned out
  false (e.g. "GPS is too coarse for per-plant" when the plant grid is RTK-accurate).
- **Re-discovery** — you spent real effort (searches, Explore agents, reading) finding
  machinery that already existed. If *you* had to dig for it, the next session will too:
  it wants a skill entry or a pointer.
- **Near-reinvention** — you were about to write code for something the toolkit already does
  (e.g. a SAHI tiling script when `run_inference(tile=True, ...)` exists).
- **Missing / hard tool** — a capability that should be a tool or skill but is buried in a web
  route, a script, or nowhere.
- **Environment / setup friction** — a missing dependency, a stale config path, a slow default.

One line is enough in the moment; precision beats volume. The free-text detail matters more
than the category.

## Distill — turn the journal into proposals

At the end of substantial work (or when asked for a "learning review"), read the journal plus
the session and draft **concrete, reviewable artifacts**, not vague notes:

- **New or updated skill files** (`.github/skills/<name>/SKILL.md`) — the primary output. A
  skill is the right home for a repeated pattern, a domain definition, or an inventory of
  existing machinery so no one re-discovers it.
- **A `CLAUDE.md` diff** — for a behavior/invariant that should govern every session. Proposed
  as a diff; **never applied silently** (CLAUDE.md is governance).
- **A tool / architecture proposal** — when a capability should be lifted from a web route or a
  script into a shared `pipelines/` module and an MCP tool the agent can compose. Describe the
  seam; the owner decides whether it's worth the build.
- **An environment / config fix** — a dependency to add, a machine-specific path to relativize.

Each proposal states: the observation (what happened), the artifact (exactly what to
create/change), and why it helps the *next* session.

## Approve — the human gate (the fence, in spirit)

You **propose**; the owner **approves and applies** anything touching governance
(`CLAUDE.md`) or platform code. Skill files can be applied once approved. This mirrors the
permission fence: the breeder-lane agent may only *write to the journal* (proposing); the
dev-lane owner distills and applies. Distillation touches the contract, so it is a dev-lane
act — not a breeder-facing one.

## Close the loop

Approved skills live in `.github/skills/` and are discoverable — CLAUDE.md instructs agents to
load the relevant skill, and you can `ls .github/skills/` to find new ones (they are **not**
auto-loaded by the runtime, so a genuinely new capability should also get a one-line pointer in
CLAUDE.md's skills list). Approved CLAUDE.md edits change behaviour for everyone. Both are
committed → they reach the whole team. That is the difference between learning and remembering.

## Honesty

Report what you actually did. If you skipped a capture, say so. A learning system that pretends
to have learned is worse than none — the whole point is that the owner can trust the platform to
compound its own competence while he's away.
