"""Agent-facing prose names no tool the registry does not hold and omits no registered one.

A cell of a Tools table (a table whose header's first column is literally "Tool") naming an
unregistered tool fails; a registered tool no surface names as `` `name` `` or `` `name( ``,
anywhere in its text, fails. Tool names outside a Tools table are not checked for fabrication.
"""

from __future__ import annotations

import importlib.util

from tests import REPO_ROOT


def _load_guardrail():
    spec = importlib.util.spec_from_file_location(
        "verify_skill_tools", REPO_ROOT / "tools" / "verify_skill_tools.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


guardrail = _load_guardrail()


def test_landed_prose_carries_no_fabricated_tool_name():
    fabricated = guardrail.fabricated_tool_names()
    assert not fabricated, (
        f"Tools-table cell(s) name a tool the registry does not hold: {fabricated}. Either "
        "it's a fabricated or retired name (fix the surface) or a real tool missing from the "
        "registry (fix the registration)."
    )


def test_landed_prose_leaves_no_registered_tool_orphaned():
    orphans = guardrail.orphan_tool_names()
    assert not orphans, (
        f"registered tool(s) named nowhere in any agent-facing surface: {orphans}. Name each "
        "in the skill that owns its domain."
    )


def test_orphan_allow_list_is_empty():
    """The orphan allow-list is empty."""
    assert guardrail.ORPHAN_ALLOW == frozenset()


def test_fabricated_tool_names_catches_a_retired_name_in_a_fixture_table(
    tmp_path,
):
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
    fabricated = guardrail.fabricated_tool_names([fixture])
    assert fabricated == {str(fixture): ["sam_auto_label"]}


def test_fabricated_tool_names_skips_a_non_identifier_first_token(tmp_path):
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
    assert guardrail.fabricated_tool_names([fixture]) == {}


def test_fabricated_tool_names_extracts_the_name_from_a_call_signature(
    tmp_path,
):
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
    assert guardrail.fabricated_tool_names([fixture]) == {}


def test_fabrication_check_does_not_reach_a_retired_name_in_running_prose(
    tmp_path,
):
    """The fabrication check does not reach a retired name in running prose.

    A retired name in a plain sentence outside any Tools table reports nothing.
    """
    fixture = tmp_path / "SKILL.md"
    fixture.write_text(
        "The agent calls `sam_auto_label` to do the labeling, a plain sentence, no table.\n",
        encoding="utf-8",
    )
    assert guardrail.fabricated_tool_names([fixture]) == {}


def test_fabrication_check_does_not_reach_a_differently_headed_tool_table(
    tmp_path,
):
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
    assert guardrail.fabricated_tool_names([fixture]) == {}


def test_fabricated_tool_names_catches_an_invented_console_command(tmp_path):
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
    assert guardrail.fabricated_tool_names([fixture]) == {
        str(fixture): ["tcip frobnicate"]
    }


def test_fabricated_tool_names_admits_a_real_console_command(tmp_path):
    fixture = tmp_path / "SKILL.md"
    fixture.write_text(
        "## Tools\n\n"
        "| Tool | Purpose |\n"
        "|------|---------|\n"
        "| `tcip doctor` | a real console command |\n",
        encoding="utf-8",
    )
    assert guardrail.fabricated_tool_names([fixture]) == {}


def test_orphan_tool_names_catches_a_registered_name_missing_from_every_surface(
    monkeypatch, tmp_path
):
    """A registered tool named nowhere in any surface is reported as an orphan."""
    from tcip_mcp import server

    monkeypatch.setattr(
        server,
        "list_registered_tools",
        lambda: ["run_inference", "orphaned_tool"],
    )

    fixture = tmp_path / "SKILL.md"
    fixture.write_text(
        "Only `run_inference` is named here.\n", encoding="utf-8"
    )
    assert guardrail.orphan_tool_names([fixture]) == ["orphaned_tool"]


def test_orphan_tool_names_accepts_the_call_form_too(monkeypatch, tmp_path):
    """A tool documented in its call form alone counts as named.

    A tool whose one mention is `` `name(`` (its call form, arguments shown) is no orphan.
    """
    from tcip_mcp import server

    monkeypatch.setattr(
        server, "list_registered_tools", lambda: ["run_inference"]
    )

    fixture = tmp_path / "SKILL.md"
    fixture.write_text(
        "Call `run_inference(checkpoint_path=...)` to score a batch.\n",
        encoding="utf-8",
    )
    assert guardrail.orphan_tool_names([fixture]) == []
