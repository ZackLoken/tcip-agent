"""The statement tool: what it records, where its audit entry lands, and what it cannot write.

The statement is the agent's half of the rail and the confirmation is the breeder's. This pins
that the tool surface carries no way to write the breeder's half, and that a statement about a
project's own measurement is recorded in that project's own log rather than the platform's.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import tcip_mcp.audit as audit_module
import tcip_store as ts
from tcip_mcp import operationalization as op
from tcip_mcp.audit import audit_log_key
from tcip_mcp.tools.operationalization_tools import state_trait_operationalization
from tests import _operationalization_fixtures as fx


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return fx.seed_project(tmp_path / "project")


@pytest.fixture
def platform_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The platform's own audit root, somewhere the project's records are not."""
    root = tmp_path / "platform"
    root.mkdir()
    monkeypatch.setattr(audit_module, "AUDIT_ROOT", root)
    return root


def _entries(root: Path, tool: str) -> list[dict]:
    page = ts.read_log(audit_log_key(root))
    return [row for row in page.records if row["tool"] == tool]


def _state(project: Path, **overrides) -> dict:
    payload = {
        "project_root": str(project),
        "trait": fx.CROSSING_TRAIT,
        "delivery_kind": op.STATE_CROSSING_DATES,
        "statement": "the date each plant reached the state the breeder scores in the field",
        "mechanism": "the calibrated state classifier over the isolated flowers of one plant",
        "measured_subject": "flower",
        "delivered_phenotypes": ["bloom_05per_date"],
    }
    payload.update(overrides)
    return state_trait_operationalization(**payload)


def test_one_audit_event_per_statement_in_the_project_log(project: Path, platform_root: Path):
    result = _state(project)

    assert "error" not in result
    rows = _entries(project, "state_trait_operationalization")
    assert len(rows) == 1, rows
    assert rows[0]["scope"] == str(project.resolve())
    assert rows[0]["arguments"]["trait"] == fx.CROSSING_TRAIT
    assert _entries(platform_root, "state_trait_operationalization") == []


def test_a_refused_statement_records_the_call_it_refused_and_writes_nothing(
    project: Path, platform_root: Path
):
    result = _state(project, delivered_phenotypes=["bloom_95per_date"])

    assert "error" in result
    rows = _entries(project, "state_trait_operationalization")
    assert len(rows) == 1 and rows[0]["status"] == "error"
    _, stored, _ = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)
    assert stored.value is None


def test_the_statement_tool_declares_no_confirmation_parameter():
    """The agent's tool surface has no way to write the breeder's half of the record."""
    parameters = set(inspect.signature(state_trait_operationalization).parameters)

    assert parameters.isdisjoint(op.CONFIRMATION_FIELDS)
    assert "stated_by" not in parameters


def test_no_registered_tool_reaches_the_confirmation_writer():
    """Asserted against the live registry rather than a hand-maintained list, so it survives a new tool."""
    from tcip_mcp.server import list_registered_tools

    assert "state_trait_operationalization" in list_registered_tools()
    assert not [name for name in list_registered_tools() if "confirm_trait" in name]


def test_the_tool_returns_the_record_and_the_hash_the_confirming_surface_compares(project: Path):
    result = _state(project, relayed_note="said at the fence line, not in the app")

    assert result["stated_by"] == op.STATEMENT_SURFACE
    assert result["relayed_note"] == "said at the fence line, not in the app"
    assert all(result[field] is None for field in op.CONFIRMATION_FIELDS)
    _, stored, _ = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)
    assert result["record_seen"] == op.record_seen_hash(stored.value)


def test_the_tool_reports_an_unregistered_trait_as_an_error_rather_than_raising(project: Path):
    result = _state(project, trait="not_registered_here")

    assert "Unknown trait" in result["error"]
