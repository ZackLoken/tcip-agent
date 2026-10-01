"""Distill worksheet: one project's friction reports and retrospectives, or every workspace
project's, printed as Markdown with their recurring themes; writes nothing.

    tcip distill-learnings --project <project>
    tcip distill-learnings --workspace
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# Generic English function words, filtered out so recurrence-counting surfaces whatever
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


def _terms(text: str) -> list[str]:
    """The non-stopword words of ``text`` and its adjacent non-stopword word pairs, repeats kept."""
    raw = _WORD.findall(text.lower())
    return [w for w in raw if w not in _STOPWORDS] + [
        f"{a} {b}" for a, b in zip(raw, raw[1:]) if a not in _STOPWORDS and b not in _STOPWORDS]


def _ranked(counts: Counter[str], top: int, floor: int) -> list[tuple[str, int]]:
    """The ``top`` most counted entries of ``counts``, each counted at least ``floor`` times."""
    return [(term, n) for term, n in counts.most_common() if n >= floor][:top]


def _themes(text: str, top: int = 12, min_count: int = 2) -> list[tuple[str, int]]:
    """The ``top`` most frequent of ``text``'s terms, each mentioned at least ``min_count`` times."""
    return _ranked(Counter(_terms(text)), top, min_count)


def _cross_project_themes(
    per_project_tokens: dict[str, set[str]], top: int = 12, min_projects: int = 2
) -> list[tuple[str, int]]:
    """The ``top`` terms present in at least ``min_projects`` of the projects' term sets, counted
    once per project."""
    counts: Counter[str] = Counter()
    for tokens in per_project_tokens.values():
        counts.update(tokens)
    return _ranked(counts, top, min_projects)


def build_workspace_worksheet(workspace: Path) -> str:
    """The cross-project Markdown worksheet for every project under ``workspace``; writes
    nothing."""
    from tcip_mcp.workspace import project_dirs

    lines: list[str] = [f"# Cross-project learning-review worksheet: {workspace}", ""]
    projects = project_dirs(workspace)
    if not projects:
        lines.append("\nNo projects with a `.tcip/` directory found under this workspace.")
        return "\n".join(lines) + "\n"

    per_project_tokens: dict[str, set[str]] = {}
    per_project_report_count: dict[str, int] = {}
    per_project_retro_count: dict[str, int] = {}
    for proj in projects:
        reports, retros, text = _gather(proj)
        per_project_tokens[proj.name] = set(_terms(text))
        per_project_report_count[proj.name] = len(reports)
        per_project_retro_count[proj.name] = len(retros)

    cross_themes = _cross_project_themes(per_project_tokens)
    if cross_themes:
        lines.append(
            "\n## Cross-project recurring themes (candidates for a skill line or a CLAUDE.md rule)"
        )
        lines.append(
            "A theme here appeared in reports/retrospectives from multiple distinct projects, a "
            "stronger platform-change signal than one project's own recurrence."
        )
        for word, n_projects in cross_themes:
            lines.append(f"- {word}: {n_projects} projects")
    else:
        lines.append(
            "\n## Cross-project recurring themes\nNone found (or only one project has data)."
        )

    lines.append(f"\n## Projects covered ({len(projects)})")
    for proj in projects:
        lines.append(
            f"- {proj.name}: {per_project_report_count[proj.name]} report(s), "
            f"{per_project_retro_count[proj.name]} retrospective(s)"
        )

    lines.append(
        "\n---\nSame as the single-project worksheet: this gathers, the judgment is yours. Nothing "
        "here is applied. After reviewing, call the `record_distillation_pass` MCP tool for each "
        "project covered so its distillation-backlog counters reset."
    )
    return "\n".join(lines) + "\n"


def _gather(project: Path) -> tuple[list[dict], list, str]:
    """``project``'s decoded friction reports (an unreadable one skipped), its retrospectives
    (latest stated section first), and the free text of both joined."""
    from tcip_mcp.tools.meta_tools import report_documents, retrospective_documents

    reports = [document.value for document in report_documents(str(project))
               if not document.value.get("malformed")]
    retros = retrospective_documents(str(project))
    text = " ".join([*(str(r.get("detail", "")) for r in reports),
                     *(document.value for document in retros)])
    return reports, retros, text


def _report_line(report: dict, width: int) -> str:
    """One report as a bullet: its category and its detail on one line, cut to ``width``."""
    detail = str(report.get("detail", "")).replace("\n", " ")[:width]
    return f"- [{report.get('category', '?')}] {detail}"


def build_worksheet(project: Path) -> str:
    """Assemble the Markdown distill worksheet (pure: no writes)."""
    lines: list[str] = [f"# Learning-review worksheet: {project}", ""]

    reports, retros, text = _gather(project)

    themes = _themes(text)
    if themes:
        lines.append("\n## Recurring themes (candidates for a skill line or a CLAUDE.md rule)")
        for word, n in themes:
            lines.append(f"- {word} ×{n}")

    disagreements = [r for r in reports if r.get("user_disagreement")]
    if disagreements:
        lines.append(f"\n## Disagreements ({len(disagreements)}): the owner pushed back or disagreed")
        lines.extend(_report_line(r, 200) for r in disagreements[:15])

    if reports:
        lines.append(f"\n## Friction reports ({len(reports)}): machine-local, won't reach the repo alone")
        lines.extend(_report_line(r, 160) for r in reports[:15])

    if retros:
        lines.append(f"\n## Retrospectives ({len(retros)})")
        for document in retros:
            lines.append(f"- {document.name}")

    lines.append(
        "\n---\nNow (per the self-improvement skill) draft the concrete artifacts, a new/updated "
        "`packages/tcip-mcp/src/tcip_mcp/knowledge/<name>.md`, a proposed CLAUDE.md diff, or a "
        "tool proposal, for the owner to approve. This script gathers; the judgment is yours. "
        "Nothing here is applied."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Gather learning-review material into one worksheet.", prog=prog)
    scope = ap.add_mutually_exclusive_group(required=True)
    scope.add_argument("--project", default=None,
                       help="the project holding the friction reports and retrospectives")
    scope.add_argument("--workspace", action="store_true",
                       help="cross-project mode: gather across every project under the TCIP "
                            "workspace TCIP_WORKSPACE names")
    args = ap.parse_args(argv)

    if args.workspace:
        from tcip_store.binding import bind_default

        from tcip_mcp.workspace import workspace_from_environment

        bind_default()
        print(build_workspace_worksheet(workspace_from_environment()))
        return 0

    from tcip_mcp.cli import bound_project

    print(build_worksheet(bound_project(args.project)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
