"""CI guardrail: agent-facing prose must not name a fabricated tool or omit a registered one.

Mirrors `tests/test_skill_trait_fidelity.py` for tool names: a Tools-table cell naming a tool
the registry does not hold is a fabrication or a rename residue; a registered tool no surface
names as `` `name` `` or `` `name( `` is an orphan, usually a rename whose new name nobody wrote
down. Reach is stated here, not just in the script: the fabrication half only reads Tools
tables (a table whose header's first column is literally "Tool"), so a fabricated or retired
name sitting in ordinary running prose, outside any such table, is out of its reach; the same
half also misses a table documenting tools under a different header, the phenology skill's
piece inventory (headed "Piece") being the live case, since that table names real tools
alongside internal module names and a content-based match flags the module names too. A green
run over either shape is not coverage of it. The orphan half has no such gap, since it searches
every surface's whole text, table or prose alike.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_guardrail():
    spec = importlib.util.spec_from_file_location(
        "verify_skill_tools", REPO / "tools" / "verify_skill_tools.py"
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
    """Every registered tool is named somewhere at HEAD; the allow-list exists only for a
    future, consciously-reviewed exception, never to paper over a rename nobody wrote down."""
    assert guardrail.ORPHAN_ALLOW == frozenset()


def test_fabricated_tool_names_catches_a_retired_name_in_a_fixture_table(tmp_path):
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
    """A cell whose backticked token is not a plain identifier (a path fragment, not a name)
    never claimed to be a tool, so it is skipped rather than reported."""
    fixture = tmp_path / "SKILL.md"
    fixture.write_text(
        "## Tools\n\n"
        "| Tool | Purpose |\n"
        "|------|---------|\n"
        "| logged `tools/` script | not a tool name |\n",
        encoding="utf-8",
    )
    assert guardrail.fabricated_tool_names([fixture]) == {}


def test_fabricated_tool_names_extracts_the_name_from_a_call_signature(tmp_path):
    """A documented call signature or a keyword note in the first cell still yields the plain
    name, so `get_experiment` (`view='lineage'`)-shaped cells check correctly."""
    fixture = tmp_path / "SKILL.md"
    fixture.write_text(
        "## Tools\n\n"
        "| Tool | Purpose |\n"
        "|------|---------|\n"
        "| `get_experiment(experiment_id, view='lineage')` | a real tool, whole call shown |\n",
        encoding="utf-8",
    )
    assert guardrail.fabricated_tool_names([fixture]) == {}


def test_fabrication_check_does_not_reach_a_retired_name_in_running_prose(tmp_path):
    """The stated reach: a fabricated or retired name outside any Tools table is invisible to
    the fabrication half, so a green run over prose-only text is not coverage of that shape."""
    fixture = tmp_path / "SKILL.md"
    fixture.write_text(
        "The agent calls `sam_auto_label` to do the labeling, a plain sentence, no table.\n",
        encoding="utf-8",
    )
    assert guardrail.fabricated_tool_names([fixture]) == {}


def test_fabrication_check_does_not_reach_a_differently_headed_tool_table(tmp_path):
    """The stated second reach gap: a table documenting tools under a header other than "Tool"
    (the phenology skill's piece inventory, headed "Piece", is the live case) is invisible to
    the fabrication half even when it names a retired tool, so a green run over that shape is
    not coverage of it either."""
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
    """A `` `tcip <command>` `` cell is never an MCP tool name, so it is checked against
    ``tcip_web.cli.COMMANDS`` instead of the tool registry: an invented command is reported as a
    fabrication, and a real one passes clean."""
    fixture = tmp_path / "SKILL.md"
    fixture.write_text(
        "## Tools\n\n"
        "| Tool | Purpose |\n"
        "|------|---------|\n"
        "| `tcip frobnicate` | an invented command |\n",
        encoding="utf-8",
    )
    assert guardrail.fabricated_tool_names([fixture]) == {str(fixture): ["tcip frobnicate"]}


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


def test_orphan_tool_names_catches_a_registered_name_missing_from_every_surface(monkeypatch, tmp_path):
    """A registered tool named nowhere in any surface is reported as an orphan: the shape a
    rename leaves behind when every stale mention of the old name is fixed but the new name is
    never written down anywhere."""
    from tcip_mcp import server

    monkeypatch.setattr(server, "list_registered_tools", lambda: ["run_inference", "orphaned_tool"])

    fixture = tmp_path / "SKILL.md"
    fixture.write_text("Only `run_inference` is named here.\n", encoding="utf-8")
    assert guardrail.orphan_tool_names([fixture]) == ["orphaned_tool"]


def test_orphan_tool_names_accepts_the_call_form_too(monkeypatch, tmp_path):
    """A tool documented only as `` `name(`` (its call form, arguments shown) still counts as
    named: tools are conventionally documented with their arguments, not the bare name."""
    from tcip_mcp import server

    monkeypatch.setattr(server, "list_registered_tools", lambda: ["run_inference"])

    fixture = tmp_path / "SKILL.md"
    fixture.write_text("Call `run_inference(checkpoint_path=...)` to score a batch.\n", encoding="utf-8")
    assert guardrail.orphan_tool_names([fixture]) == []
