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

# Words that mark friction worth turning into a durable rule/skill (recurrence = signal).
_THEME_WORDS = re.compile(
    r"\b(exif|orientation|tile|tiling|sahi|nms|max_dets|operating point|calibrat|"
    r"validat|firewall|checkpoint|kind|yolo|ultralytics|sandbox|fence|governance|"
    r"format:check|prettier|gate|ci|type|mypy|pyright|two-root|active project|"
    r"phenology|catkin|elongat|plant.mapping|classifier|derive|pin)\b",
    re.IGNORECASE,
)


def _themes(*texts: str, top: int = 12) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for t in texts:
        for m in _THEME_WORDS.findall(t):
            counter[m.lower()] += 1
    return counter.most_common(top)


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
