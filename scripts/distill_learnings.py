"""Distill worksheet — gather the raw material for a learning review in one place.

The *judgment* step (drafting new `.github/skills/*` files + a CLAUDE.md diff) is the agent's
job, per the `self-improvement` skill — a script can't write a skill. What this does is collect
and structure the inputs so that review is cheap and nothing is dropped: the learning journal,
plus the machine-local friction reports and retrospectives that never reach the repo on their
own. Run it on demand ("review what I've learned") or from the SessionEnd capture hook.

    conda activate tcip-agent
    python scripts/distill_learnings.py [--project <root>]

Output is a Markdown worksheet: undistilled journal entries first, then recurring themes across
journal + reports + retrospectives, then the reports/retros the next skill/CLAUDE.md proposals
should draw from. Nothing is written or applied — governance stays human.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
JOURNAL = REPO_ROOT / ".github" / "skills" / "_learning" / "journal.md"

# Words that mark friction worth turning into a durable rule/skill (recurrence = signal).
_THEME_WORDS = re.compile(
    r"\b(exif|orientation|tile|tiling|sahi|nms|max_dets|operating point|calibrat|"
    r"validat|firewall|checkpoint|kind|yolo|ultralytics|sandbox|fence|governance|"
    r"format:check|prettier|gate|ci|type|mypy|pyright|two-root|active project|"
    r"phenology|catkin|elongat|plant.mapping|classifier|derive|pin)\b",
    re.IGNORECASE,
)


def _journal_sections(text: str) -> list[tuple[str, str]]:
    """Split the journal into (heading, body) sections at ``## `` boundaries."""
    parts = re.split(r"(?m)^## ", text)
    out: list[tuple[str, str]] = []
    for p in parts[1:]:  # parts[0] is the file preamble
        head, _, body = p.partition("\n")
        out.append((head.strip(), body))
    return out


def _undistilled(sections: list[tuple[str, str]]) -> list[str]:
    """Journal headings whose body never records a distilled/applied status."""
    done = re.compile(r"\b(distilled|applied|committed|status:\s*(done|applied|committed))\b",
                      re.IGNORECASE)
    return [h for h, body in sections if not done.search(body)]


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
    lines: list[str] = ["# Learning-review worksheet", ""]

    journal_text = JOURNAL.read_text(encoding="utf-8") if JOURNAL.is_file() else ""
    sections = _journal_sections(journal_text)
    undistilled = _undistilled(sections)

    lines.append(f"**Journal:** {len(sections)} entries, {len(undistilled)} not yet distilled.")
    if undistilled:
        lines.append("\n## Undistilled journal entries (draft skills / CLAUDE.md diffs from these)")
        for h in undistilled:
            lines.append(f"- {h}")

    reports = _read_jsonl_dir(project_root / ".tcip" / "reports")
    retros_dir = project_root / ".tcip" / "retrospectives"
    retros = sorted(retros_dir.glob("*.md")) if retros_dir.is_dir() else []
    report_text = " ".join(str(r.get("detail", "")) for r in reports)
    retro_text = " ".join(p.read_text(encoding="utf-8") for p in retros)

    themes = _themes(journal_text, report_text, retro_text)
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
        lines.append(f"\n## Session captures ({len(captures)}) — SessionEnd backstop; distill any real signal into the journal")
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
