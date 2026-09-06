"""Distill worksheet: gather one project's learning record in one place.

Learning lands with the project (see the `self-improvement` skill): friction goes to the friction
reports via ``report_friction``, and end-of-work findings to the retrospectives via
``write_retrospective``. This gathers both, plus the SessionEnd capture backstop, so a review is
cheap and nothing is dropped.

    conda activate tcip-agent
    tcip distill-learnings [--project <root>]
    tcip distill-learnings --workspace

Output is a Markdown worksheet: recurring themes across the project's reports and retrospectives,
then the records themselves. It *gathers*: nothing is written, applied, or promoted anywhere; this
stays true in ``--workspace`` mode too. ``--workspace`` gathers across every project under the TCIP
workspace instead of one project root, and surfaces themes recurring across multiple distinct
projects, a stronger "candidate for a platform change" signal than one project's own recurrence,
which a single project's own accumulated friction/retrospectives could produce on its own. After
reviewing a worksheet (either mode), call the ``record_distillation_pass`` MCP tool per project
covered so its distillation-backlog counters reset; that's the one write in this loop, and it's
audited, kept out of this script on purpose.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path


def _find_repo_root(start: Path) -> Path | None:
    cur = start
    while cur != cur.parent:
        if (cur / ".git").exists():
            return cur
        cur = cur.parent
    return None

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


def _themes(*texts: str, top: int = 12, min_count: int = 2) -> list[tuple[str, int]]:
    """Recurring words/phrases across free text: generic, no fixed vocabulary to maintain.

    Unigrams and bigrams built from a stopword-filtered token stream (bigrams keep both
    words so a real recurring phrase like "operating point" survives, not just single words).
    Recurrence is the signal, so anything mentioned only once is dropped here; it's still
    printed in full further down in the friction-reports/retrospectives sections either way.
    """
    raw = [w for t in texts for w in _WORD.findall(t.lower())]
    counts: Counter[str] = Counter(w for w in raw if w not in _STOPWORDS)
    counts.update(
        f"{a} {b}" for a, b in zip(raw, raw[1:])
        if a not in _STOPWORDS and b not in _STOPWORDS
    )
    return [(term, n) for term, n in counts.most_common(top) if n >= min_count]


def _project_token_set(*texts: str) -> set[str]:
    """Distinct unigrams/bigrams present in one project's text: membership, not frequency. The
    building block for cross-project recurrence (_cross_project_themes), which asks "how many
    distinct projects", not "how many times in one project" (that's _themes' own job)."""
    raw = [w for t in texts for w in _WORD.findall(t.lower())]
    tokens = {w for w in raw if w not in _STOPWORDS}
    tokens.update(
        f"{a} {b}" for a, b in zip(raw, raw[1:])
        if a not in _STOPWORDS and b not in _STOPWORDS
    )
    return tokens


def _cross_project_themes(
    per_project_tokens: dict[str, set[str]], top: int = 12, min_projects: int = 2
) -> list[tuple[str, int]]:
    """Themes appearing in >= min_projects distinct projects' token sets.

    Built from each project's own set (one project can only ever contribute 1 to a token's count,
    no matter how many times it repeats that token internally), so a single verbose project can
    never clear the bar alone the way a pooled frequency count over concatenated text would let it.
    """
    counts: Counter[str] = Counter()
    for tokens in per_project_tokens.values():
        counts.update(tokens)
    return [(term, n) for term, n in counts.most_common() if n >= min_projects][:top]


def build_workspace_worksheet(workspace_root: Path) -> str:
    """Cross-project distill worksheet: gathers across every project under the workspace.

    Still pure gather, same as :func:`build_worksheet`: nothing is written, applied, or promoted.
    """
    lines: list[str] = [f"# Cross-project learning-review worksheet: {workspace_root}", ""]

    projects = sorted(
        p for p in workspace_root.iterdir() if p.is_dir() and (p / ".tcip").is_dir()
    )
    if not projects:
        lines.append("\nNo projects with a `.tcip/` directory found under this workspace.")
        return "\n".join(lines) + "\n"

    per_project_tokens: dict[str, set[str]] = {}
    per_project_report_count: dict[str, int] = {}
    per_project_retro_count: dict[str, int] = {}
    for proj in projects:
        reports = _read_reports(proj)
        retros = _read_retrospectives(proj)
        report_text = " ".join(str(r.get("detail", "")) for r in reports)
        retro_text = " ".join(document.value for document in retros)
        per_project_tokens[proj.name] = _project_token_set(report_text, retro_text)
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


def _read_reports(project_root: Path) -> list[dict]:
    """Every friction report of one project, decoded; an unreadable one is skipped.

    The corpus, its decode and its order all come from the store's own owner rather than a
    directory walk restated here, so this worksheet, the GUI panel and the agent's memory tool
    read one project in one order.
    """
    from tcip_mcp.tools.meta_tools import report_documents

    return [
        document.value
        for document in report_documents(str(project_root))
        if not document.value.get("malformed")
    ]


def _read_retrospectives(project_root: Path) -> list:
    """Every retrospective of one project, latest stated section first, from the store's owner."""
    from tcip_mcp.tools.meta_tools import retrospective_documents

    return retrospective_documents(str(project_root))


def _read_captures(project_root: Path) -> list[dict]:
    """Every SessionEnd capture entry for this project's root, through the store the hook
    (`agent_learning_capture.py`) appends through, under whichever backend this process bound.

    Importing the hook's module registers the log's store descriptor as a side effect, the
    same way ``tcip_mcp.store_catalogue`` does for the commands that must cover every store.
    An undecodable entry is excluded from what ``read_log`` returns here exactly as the old
    direct file read skipped one it could not parse; that page also carries a `corrupt` count
    of such entries this worksheet does not otherwise surface.
    """
    from tcip_store import read_log
    from tcip_web.agent_learning_capture import learning_capture_key

    page = read_log(learning_capture_key(project_root))
    return [dict(r) for r in page.records]


def build_worksheet(project_root: Path) -> str:
    """Assemble the Markdown distill worksheet (pure: no writes)."""
    lines: list[str] = [f"# Learning-review worksheet: {project_root}", ""]

    reports = _read_reports(project_root)
    retros = _read_retrospectives(project_root)
    report_text = " ".join(str(r.get("detail", "")) for r in reports)
    retro_text = " ".join(document.value for document in retros)

    themes = _themes(report_text, retro_text)
    if themes:
        lines.append("\n## Recurring themes (candidates for a skill line or a CLAUDE.md rule)")
        for word, n in themes:
            lines.append(f"- {word} ×{n}")

    disagreements = [r for r in reports if r.get("user_disagreement")]
    if disagreements:
        lines.append(f"\n## Disagreements ({len(disagreements)}): the owner pushed back or disagreed")
        for r in disagreements[:15]:
            cat = r.get("category", "?")
            detail = str(r.get("detail", "")).replace("\n", " ")[:200]
            lines.append(f"- [{cat}] {detail}")

    if reports:
        lines.append(f"\n## Friction reports ({len(reports)}): machine-local, won't reach the repo alone")
        for r in reports[:15]:
            cat = r.get("category", "?")
            detail = str(r.get("detail", "")).replace("\n", " ")[:160]
            lines.append(f"- [{cat}] {detail}")

    if retros:
        lines.append(f"\n## Retrospectives ({len(retros)})")
        for document in retros:
            lines.append(f"- {document.name}")

    # SessionEnd capture backstop (machine-local; agent_learning_capture.py writes it).
    captures = _read_captures(project_root)
    if captures:
        lines.append(f"\n## Session captures ({len(captures)}): SessionEnd backstop")
        for c in captures[-10:]:
            lines.append(f"- {c.get('ts', '?')}  session {c.get('session_id', '?')}")

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
    ap.add_argument("--project", default=None,
                    help="project root holding the friction reports and retrospectives "
                         "(default: this platform checkout's own repo root)")
    ap.add_argument("--workspace", action="store_true",
                    help="cross-project mode: gather across every project under the TCIP workspace "
                         "(TCIP_WORKSPACE, default ~/tcip-projects/) instead of one --project root")
    args = ap.parse_args(argv)

    # Its own process entry point, so it binds the storage backend the seam has no default for.
    from tcip_store.binding import bind_default

    bind_default()

    if args.workspace:
        from tcip_mcp.workspace import workspace_root

        print(build_workspace_worksheet(workspace_root()))
        return 0

    project = args.project
    if project is None:
        repo_root = _find_repo_root(Path(__file__).resolve())
        if repo_root is None:
            print("error: --project not given and no repo root (.git ancestor) found to "
                  "default to; name a project root explicitly")
            return 2
        project = str(repo_root)
    print(build_worksheet(Path(project)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
