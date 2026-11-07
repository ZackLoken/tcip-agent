"""Generate the repository's harness files from the knowledge documents and the harness paths.

`tcip_mcp.knowledge.harness_files` writes a `SKILL.md` per document of
`tcip_mcp.knowledge.list_documents()` under `.claude/skills/` and `.agents/skills/`, and the
operating contract's block of `AGENTS.md`. Claude Code's row settings file gets its `Edit`
denials from `protected_paths()` and its `WebFetch(domain:...)` grants from the hosts the
`cv-research` document's source list names.

    python -m tools.generate_harness_discovery
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from tests import REPO_ROOT

if TYPE_CHECKING:
    from pathlib import Path

    from tcip_mcp.knowledge import DocumentInfo


def stale_skills(skills_dir: Path, documents: list[DocumentInfo]) -> list[str]:
    """Return the document names whose `SKILL.md` under `skills_dir` is missing or stale.

    Stale is not equal to `render_skill(document)`.
    """
    from tcip_mcp.knowledge.harness_files import render_skill

    return [
        document.name
        for document in documents
        if not (path := skills_dir / document.name / "SKILL.md").is_file()
        or path.read_text(encoding="utf-8") != render_skill(document)
    ]


def stray_skill_directories(
    skills_dir: Path, documents: list[DocumentInfo]
) -> set[str]:
    """Return the directory names directly under `skills_dir` naming no document in `documents`."""
    documented_names = {document.name for document in documents}
    on_disk = {entry.name for entry in skills_dir.iterdir() if entry.is_dir()}
    return on_disk - documented_names


RESEARCH_DOCUMENT = "cv-research"
_SOURCE_LIST = re.compile(
    r"The allowed/preferred set:\s*\n\n(.*?)\n\nSearch discipline:", re.DOTALL
)
_BACKTICK_HOST = re.compile(r"`([a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,})`")
_WEBFETCH = "WebFetch(domain:"


def research_hosts(document_text: str) -> list[str]:
    """Return the backticked hosts of the research document's source list, in order.

    Raises `ValueError` when the document carries no source list.
    """
    block = _SOURCE_LIST.search(document_text)
    if block is None:
        raise ValueError(
            f"the {RESEARCH_DOCUMENT} document carries no source list"
        )
    return _BACKTICK_HOST.findall(block.group(1))


_EDIT = "Edit("


def protected_paths() -> list[str]:
    """Return the workspace paths whose edits Claude's row denies, relative to the workspace.

    They are the state directory `tcip_store`'s `database_file` names under a project, then, for
    each provider row's `files`, its instruction file unless it sits in a directory another
    path names, and the directory its skills sit in, each directory with everything under it.
    """
    from tcip_store.file_backend import database_file
    from tcip_web.terminal import PROVIDERS

    state = database_file(str(REPO_ROOT)).parent.relative_to(REPO_ROOT)
    directories = {row.files.skills.parent for row in PROVIDERS}
    paths = [f"{state.as_posix()}/**"]
    for row in PROVIDERS:
        if not directories.intersection(row.files.instructions.parents):
            paths.append(row.files.instructions.as_posix())
        paths.append(f"{row.files.skills.parent.as_posix()}/**")
    return list(dict.fromkeys(paths))


def render_claude_settings(
    settings_text: str, hosts: list[str], protected: list[str]
) -> str:
    """Return `settings_text` with its `Edit` denials and its `WebFetch` grants replaced.

    One `Edit` denial per path of `protected` comes before every other deny entry, and one
    `WebFetch` grant per host of `hosts` after every other allow entry.
    """
    settings = json.loads(settings_text)
    permissions = settings["permissions"]
    permissions["deny"] = [
        *(f"{_EDIT}{path})" for path in protected),
        *(rule for rule in permissions["deny"] if not rule.startswith(_EDIT)),
    ]
    permissions["allow"] = [
        rule for rule in permissions["allow"] if not rule.startswith(_WEBFETCH)
    ]
    permissions["allow"] += [f"{_WEBFETCH}{host})" for host in hosts]
    return json.dumps(settings, indent=2) + "\n"


def expected_claude_settings() -> str:
    """Return Claude Code's row settings file text, its denials and its grants generated.

    The denials are `protected_paths()`; the grants are the research document's hosts.
    """
    from tcip_mcp.knowledge import document_path
    from tcip_web.terminal import CLAUDE_SETTINGS

    hosts = research_hosts(
        document_path(RESEARCH_DOCUMENT).read_text(encoding="utf-8")
    )
    return render_claude_settings(
        CLAUDE_SETTINGS.read_text(encoding="utf-8"), hosts, protected_paths()
    )


def generate(root: Path = REPO_ROOT) -> list[Path]:
    """Write every generated file at its repository-relative place under `root`.

    Returns the paths written.
    """
    from tcip_mcp.knowledge.harness_files import (
        AGENTS,
        CLAUDE_CODE,
        write_block,
        write_skills,
        write_text,
    )
    from tcip_web.terminal import CLAUDE_SETTINGS

    return [
        *write_skills(root / CLAUDE_CODE.skills),
        *write_skills(root / AGENTS.skills),
        write_block(root / AGENTS.instructions),
        write_text(
            root / CLAUDE_SETTINGS.relative_to(REPO_ROOT),
            expected_claude_settings(),
        ),
    ]


def main() -> int:
    """Generate every file and return 0."""
    generate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
