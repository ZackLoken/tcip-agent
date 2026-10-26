"""Agent-facing prose names no tool the registry does not hold and omits no registered one.

A cell of a Tools table (a table whose header's first column is literally "Tool") naming an
unregistered tool fails; a registered tool no surface names as `` `name` `` or `` `name( ``,
anywhere in its text, fails. Tool names outside a Tools table are not checked for fabrication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tools import verify_skill_tools

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from mcp.types import Tool


def _listed(*names: str) -> list[Tool]:
    """Return a listing of tools named ``names``, each with an empty input schema."""
    from mcp.types import Tool

    return [Tool(name=name, input_schema={}) for name in names]


def test_landed_prose_carries_no_fabricated_tool_name() -> None:
    from tcip_web.cli import tool_listing

    fabricated = verify_skill_tools.fabricated_tool_names(tool_listing())
    assert not fabricated, (
        f"Tools-table cell(s) name a tool the registry does not hold: {fabricated}. Either "
        "it's a fabricated or retired name (fix the surface) or a real tool missing from the "
        "registry (fix the registration)."
    )


def test_landed_prose_leaves_no_registered_tool_orphaned() -> None:
    from tcip_web.cli import tool_listing

    orphans = verify_skill_tools.orphan_tool_names(tool_listing())
    assert not orphans, (
        f"registered tool(s) named nowhere in any agent-facing surface: {orphans}. Name each "
        "in the skill that owns its domain."
    )


def test_orphan_allow_list_is_empty() -> None:
    """The orphan allow-list is empty."""
    assert verify_skill_tools.ORPHAN_ALLOW == frozenset()


def test_fabricated_tool_names_catches_a_retired_name_in_a_fixture_table(
    tmp_path: Path,
) -> None:
    """A Tools table naming a retired tool is reported as fabricated."""
    fixture = tmp_path / "SKILL.md"
    fixture.write_text(
        "## Tools\n\n"
        "| Tool | Purpose |\n"
        "|------|---------|\n"
        "| `sam_auto_label` | a name retired long ago |\n"
        "| `run_inference` | a real, registered tool |\n",
        encoding="utf-8",
    )
    fabricated = verify_skill_tools.fabricated_tool_names(
        _listed("run_inference"), [fixture]
    )
    assert fabricated == {str(fixture): ["sam_auto_label"]}


def test_fabricated_tool_names_skips_a_non_identifier_first_token(
    tmp_path: Path,
) -> None:
    """A Tools-table cell whose backticked token is no identifier is skipped, not reported.

    A cell backticking the path fragment ``tools/`` reports nothing.
    """
    fixture = tmp_path / "SKILL.md"
    fixture.write_text(
        "## Tools\n\n"
        "| Tool | Purpose |\n"
        "|------|---------|\n"
        "| logged `tools/` script | not a tool name |\n",
        encoding="utf-8",
    )
    assert verify_skill_tools.fabricated_tool_names(_listed(), [fixture]) == {}


def test_fabricated_tool_names_extracts_the_name_from_a_call_signature(
    tmp_path: Path,
) -> None:
    """A Tools-table cell showing a registered tool's whole call reports no fabrication.

    A first cell of `` `get_experiment(experiment_id, view='lineage')` `` reports nothing.
    """
    fixture = tmp_path / "SKILL.md"
    fixture.write_text(
        "## Tools\n\n"
        "| Tool | Purpose |\n"
        "|------|---------|\n"
        "| `get_experiment(experiment_id, view='lineage')` | a real tool, whole call shown |\n",
        encoding="utf-8",
    )
    assert (
        verify_skill_tools.fabricated_tool_names(
            _listed("get_experiment"), [fixture]
        )
        == {}
    )


def test_fabrication_check_does_not_reach_a_retired_name_in_running_prose(
    tmp_path: Path,
) -> None:
    """The fabrication check does not reach a retired name in running prose.

    A retired name in a plain sentence outside any Tools table reports nothing.
    """
    fixture = tmp_path / "SKILL.md"
    fixture.write_text(
        "The agent calls `sam_auto_label` to do the labeling, a plain sentence, no table.\n",
        encoding="utf-8",
    )
    assert verify_skill_tools.fabricated_tool_names(_listed(), [fixture]) == {}


def test_fabrication_check_does_not_reach_a_differently_headed_tool_table(
    tmp_path: Path,
) -> None:
    """The fabrication check does not reach a tool table headed other than "Tool".

    A table documenting tools under the header "Piece" reports nothing, though it names a
    retired tool.
    """
    fixture = tmp_path / "SKILL.md"
    fixture.write_text(
        "## Pieces\n\n"
        "| Piece | Where | Role |\n"
        "|-------|-------|------|\n"
        "| `sam_auto_label` | somewhere | a name retired long ago |\n"
        "| `run_inference` | somewhere | a real, registered tool |\n",
        encoding="utf-8",
    )
    assert verify_skill_tools.fabricated_tool_names(_listed(), [fixture]) == {}


def test_fabricated_tool_names_catches_an_invented_console_command(
    tmp_path: Path,
) -> None:
    """An invented console command in a Tools table is reported as a fabrication.

    A `` `tcip <command>` `` cell naming no console command is reported as a fabrication.
    """
    fixture = tmp_path / "SKILL.md"
    fixture.write_text(
        "## Tools\n\n"
        "| Tool | Purpose |\n"
        "|------|---------|\n"
        "| `tcip frobnicate` | an invented command |\n",
        encoding="utf-8",
    )
    assert verify_skill_tools.fabricated_tool_names(_listed(), [fixture]) == {
        str(fixture): ["tcip frobnicate"]
    }


def test_fabricated_tool_names_admits_a_real_console_command(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "SKILL.md"
    fixture.write_text(
        "## Tools\n\n"
        "| Tool | Purpose |\n"
        "|------|---------|\n"
        "| `tcip dump-store` | a real operator command |\n"
        "| `tcip scan-dataset` | a registered tool's command |\n",
        encoding="utf-8",
    )
    assert (
        verify_skill_tools.fabricated_tool_names(
            _listed("scan_dataset"), [fixture]
        )
        == {}
    )


def test_orphan_tool_names_catches_a_registered_name_missing_from_every_surface(
    tmp_path: Path,
) -> None:
    """A registered tool named nowhere in any surface is reported as an orphan."""
    fixture = tmp_path / "SKILL.md"
    fixture.write_text(
        "Only `run_inference` is named here.\n", encoding="utf-8"
    )
    assert verify_skill_tools.orphan_tool_names(
        _listed("run_inference", "orphaned_tool"), [fixture]
    ) == ["orphaned_tool"]


def test_orphan_tool_names_accepts_the_call_form_too(tmp_path: Path) -> None:
    """A tool documented in its call form alone counts as named.

    A tool whose one mention is `` `name(`` (its call form, arguments shown) is no orphan.
    """
    fixture = tmp_path / "SKILL.md"
    fixture.write_text(
        "Call `run_inference(checkpoint_path=...)` to score a batch.\n",
        encoding="utf-8",
    )
    assert (
        verify_skill_tools.orphan_tool_names(
            _listed("run_inference"), [fixture]
        )
        == []
    )


def test_one_checker_invocation_lists_the_tools_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The checker's two checks and its command validation read one server's listing."""
    import tcip_mcp.server as server

    built: list[server.Session] = []
    build = server.build_server

    def counted(session: server.Session) -> object:
        built.append(session)
        return build(session)

    monkeypatch.setattr(server, "build_server", counted)

    assert verify_skill_tools.main() == 0
    assert len(built) == 1
