"""Distill worksheet — gather one project's learning record in one place.

Learning lands with the project (see the `self-improvement` skill): friction goes to
`.tcip/reports/` via ``claude_reports``, and end-of-work findings to `.tcip/retrospectives/` via
``project_retrospective``. This gathers both, plus the SessionEnd capture backstop, so a review is
cheap and nothing is dropped.

    conda activate tcip-agent
    python scripts/distill_learnings.py [--project <root>]

Output is a Markdown worksheet: recurring themes across the project's reports and retrospectives,
then the records themselves. It *gathers* — nothing is written, applied, or promoted anywhere.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Generic English function words — filtered out so recurrence-counting surfaces whatever
# actually recurs in a project's own reports/retrospectives, not a fixed, maintained,
# domain-specific vocabulary that goes stale the moment this project's vocabulary shifts.
_STOPWORDS = frozenset("""
a an the and or but if then than so to of in on at by for with as is was were are be been being
it its this that these those there here which what when where who whom how why i you we they he
she him her them his their my your our do does did doing have has had having would could should
can will just also not no yes very more most some any all each every other such into out up down
off over under again once about against between through during before after above below from
because while still only own same too now one two
""".split())

_WORD = re.compile(r"[a-z][a-z0-9']{2,}")


def _themes(*texts: str, top: int = 12, min_count: int = 2) -> list[tuple[str, int]]:
    """Recurring words/phrases across free text — generic, no fixed vocabulary to maintain.

    Unigrams and bigrams built from a stopword-filtered token stream (bigrams keep both
    words so a real recurring phrase like "operating point" survives, not just single words).
    Recurrence is the signal, so anything mentioned only once is dropped here — it's still
    printed in full further down in the friction-reports/retrospectives sections either way.
    """
    raw = [w for t in texts for w in _WORD.findall(t.lower())]
    counts: Counter[str] = Counter(w for w in raw if w not in _STOPWORDS)
    counts.update(
        f"{a} {b}" for a, b in zip(raw, raw[1:])
        if a not in _STOPWORDS and b not in _STOPWORDS
    )
    return [(term, n) for term, n in counts.most_common(top) if n >= min_count]


def _read_jsonl_dir(d: Path) -> list[dict]:
    rows: list[dict] = []
    if not d.is_dir():
        return rows
    for f in sorted(d.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def build_worksheet(project_root: Path) -> str:
    """Assemble the Markdown distill worksheet (pure — no writes)."""
    lines: list[str] = [f"# Learning-review worksheet — {project_root}", ""]

    reports = _read_jsonl_dir(project_root / ".tcip" / "reports")
    retros_dir = project_root / ".tcip" / "retrospectives"
    retros = sorted(retros_dir.glob("*.md")) if retros_dir.is_dir() else []
    report_text = " ".join(str(r.get("detail", "")) for r in reports)
    retro_text = " ".join(p.read_text(encoding="utf-8") for p in retros)

    themes = _themes(report_text, retro_text)
    if themes:
        lines.append("\n## Recurring themes (candidates for a skill line or a CLAUDE.md rule)")
        for word, n in themes:
            lines.append(f"- **{word}** ×{n}")

    disagreements = [r for r in reports if r.get("user_disagreement")]
    if disagreements:
        lines.append(f"\n## Disagreements ({len(disagreements)}) — the owner pushed back or disagreed")
        for r in disagreements[-15:]:
            cat = r.get("category", "?")
            detail = str(r.get("detail", "")).replace("\n", " ")[:200]
            lines.append(f"- [{cat}] {detail}")

    if reports:
        lines.append(f"\n## Friction reports ({len(reports)}) — machine-local, won't reach the repo alone")
        for r in reports[-15:]:
            cat = r.get("category", "?")
            detail = str(r.get("detail", "")).replace("\n", " ")[:160]
            lines.append(f"- [{cat}] {detail}")

    if retros:
        lines.append(f"\n## Retrospectives ({len(retros)})")
        for p in retros:
            lines.append(f"- {p.name}")

    # SessionEnd capture backstop (machine-local; agent_learning_capture.py writes it).
    cap_file = project_root / ".tcip" / "learning_capture.jsonl"
    captures: list[dict] = []
    if cap_file.is_file():
        for line in cap_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    captures.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if captures:
        lines.append(f"\n## Session captures ({len(captures)}) — SessionEnd backstop")
        for c in captures[-10:]:
            lines.append(f"- {c.get('ts', '?')}  session {c.get('session_id', '?')}")

    lines.append(
        "\n---\nNow (per the self-improvement skill) draft the concrete artifacts — new/updated "
        "`.github/skills/<name>/SKILL.md`, a proposed CLAUDE.md diff, or a tool proposal — for the "
        "owner to approve. This script gathers; the judgment is yours. Nothing here is applied."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Gather learning-review material into one worksheet.")
    ap.add_argument("--project", default=str(REPO_ROOT),
                    help="project root holding .tcip/reports + .tcip/retrospectives (default: repo root)")
    args = ap.parse_args()
    print(build_worksheet(Path(args.project)))


if __name__ == "__main__":
    main()
