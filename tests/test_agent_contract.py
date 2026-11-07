"""The agent's operating contract, where every launch puts it and what binds the agent beside it.

The contract is one knowledge document; the developer's root ``CLAUDE.md`` imports it and states
none of its sentences, and Claude's row's deny list protects what a launch writes into the
workspace and every project's store.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pathspec import GitIgnoreSpec

from tcip_mcp.knowledge import read_document
from tcip_mcp.knowledge.harness_files import CONTRACT
from tcip_web import terminal as pty_host
from tests import REPO_ROOT

if TYPE_CHECKING:
    from tcip_web.state import OpenProject

_PATH_TOOLS = re.compile(r"^(Edit|Read)\((.+)\)$")
"""A permission rule naming a path: Claude Code's ``Edit`` and ``Read`` rules."""


def _sentences(markdown: str) -> set[str]:
    """Return every sentence of ``markdown``, its whitespace collapsed and list markers dropped."""
    prose = re.sub(r"(?m)^\s*(?:[-*]|#+)\s+", "", markdown)
    return {
        sentence
        for sentence in re.split(r"(?<=[.?!])\s+", " ".join(prose.split()))
        if sentence
    }


def test_the_root_claude_md_states_none_of_the_operating_contracts_sentences() -> (
    None
):
    """The developer's ``CLAUDE.md`` imports the contract and restates none of its sentences."""
    contract = _sentences(read_document(CONTRACT))
    developer = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")

    assert contract, "the contract holds no sentence to compare"
    assert sorted(contract & _sentences(developer)) == []


def _claude_posix(path: Path) -> str:
    """Return ``path`` as Claude Code's permissions page normalizes it, drive ``C:`` as ``/c``."""
    if path.drive:
        return "/".join(["", path.drive[0].lower(), *path.parts[1:]])
    return path.as_posix()


def _denies(pattern: str, file: Path, settings: Path, workspace: Path) -> bool:
    """Return whether the deny rule ``pattern`` of the settings file ``settings`` matches ``file``.

    The pattern is a gitignore pattern, matched by pathspec, under its anchor as Claude Code's
    permissions page states it: ``//`` the filesystem root, ``~/`` the home directory, ``/`` the
    directory of a ``--settings`` file, and any other pattern the agent's working directory, the
    workspace, where a deny rule's pattern of a single directory segment, such as ``src/**``,
    matches that directory at any depth.
    """
    if pattern.startswith("//"):
        anchor, pattern = "", pattern[2:]
    elif pattern.startswith("~/"):
        anchor, pattern = _claude_posix(Path.home()), pattern[2:]
    elif pattern.startswith("/"):
        anchor, pattern = _claude_posix(settings.parent), pattern[1:]
    else:
        anchor, pattern = _claude_posix(workspace), pattern.removeprefix("./")
        if pattern.partition("/")[2] == "**":
            pattern = f"**/{pattern}"
    target = _claude_posix(file)
    return target.startswith(f"{anchor}/") and GitIgnoreSpec.from_lines(
        [pattern]
    ).match_file(target.removeprefix(f"{anchor}/"))


def test_the_deny_list_protects_what_a_launch_writes_and_every_projects_store(
    opened: OpenProject,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claude's row denies edits of every file a launch writes and of a project's store.

    The rules are the deny list of the settings file the row hands its harness, read by
    :func:`_denies`; the files are every one each row's preparation writes into the workspace
    and the open project's database, which the tools alone write. A script in the project's own
    directory stays editable.
    """
    from tcip_mcp.web_client import record_backend_port
    from tcip_store.file_backend import database_file
    from tcip_web.state import store

    monkeypatch.setattr(
        pty_host,
        "ANTIGRAVITY_SETTINGS",
        tmp_path.parent.parent / "agy-settings.json",
    )
    monkeypatch.setattr(pty_host, "run_to_completion", lambda argv: "added")
    record_backend_port(store.workspace, 8765)
    before = set(store.workspace.rglob("*"))
    for row in pty_host.PROVIDERS:
        row.prepare(row.files, row.executable, opened, "term_contract")
    written = [
        p for p in set(store.workspace.rglob("*")) - before if p.is_file()
    ]
    settings = pty_host.write_claude_settings()
    deny = json.loads(settings.read_text(encoding="utf-8"))["permissions"][
        "deny"
    ]
    patterns = [m[2] for rule in deny if (m := _PATH_TOOLS.match(rule))]

    def denied(file: Path) -> bool:
        return any(
            _denies(pattern, file, settings, store.workspace)
            for pattern in patterns
        )

    assert written
    protected = [*written, database_file(str(opened.root))]
    assert sorted(p for p in protected if not denied(p)) == []
    assert not denied(opened.root / "code" / "measure.py")


def test_the_settings_a_launch_hands_refuse_file_tool_writes_outside_the_workspace() -> (
    None
):
    """Claude's row settings refuse the file tools' writes outside its working directories.

    The key is read from the settings file a launch writes, where Claude Code honors it in every
    permission mode.
    """
    settings = pty_host.write_claude_settings()
    permissions = json.loads(settings.read_text(encoding="utf-8"))[
        "permissions"
    ]

    assert permissions["blockWritesOutsideWorkingDirectories"] is True
