"""An audit line whose arguments its reader's model refuses is refused at the audit log's place.

Each reader of a line's arguments (a run's launch declarations, a project's recent activity, a
plant mapping's receipts, a terminal session's record) is handed a line the audit writer appended
with no arguments, which the reader's model refuses. The refused read names the place the audit
reader names for an entry of the same log that will not decode, and a run's cancel over such a
launch line is recorded at that place.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import pytest

import tcip_store as ts
from tcip_mcp.audit import (
    append_or_raise,
    audit_entries,
    audit_line,
    audit_log_key,
)
from tests._record_damage_fixtures import damage_last_log_entry

if TYPE_CHECKING:
    from pathlib import Path

    from tcip_web.state import OpenProject


def _launch_declarations(made: OpenProject) -> object:
    from tcip_mcp.experiments import launch_declarations

    return launch_declarations(made.root)


def _inspect_project(made: OpenProject) -> object:
    from tcip_mcp.tools.project_tools import inspect_project

    return inspect_project(made.root)


def _map_valley(made: OpenProject) -> None:
    from tests._mapping_fixtures import map_captures

    map_captures(made.root, made.root / "ds", ["2026-02-11"], name="valley")


def _load_valley(made: OpenProject) -> object:
    from tcip_mcp.pipelines.postprocessing.plant_mapping import load_mapping

    return load_mapping(made.root, "valley")


def _recorded_session(made: OpenProject) -> object:
    from tcip_web import terminal as pty_host
    from tcip_web.routes.terminal import recorded_session

    return recorded_session(pty_host.PROVIDERS[0].id, made)


def _nothing(made: OpenProject) -> None:
    """Prepare nothing beyond the project itself."""


class ArgumentsReader(NamedTuple):
    """A reader of ``tool``'s audit lines, reached by ``read`` once ``prepare`` has run."""

    tool: str
    read: Callable[[OpenProject], object]
    prepare: Callable[[OpenProject], None] = _nothing


READERS = [
    ArgumentsReader("launch_training", _launch_declarations),
    ArgumentsReader("write_retrospective", _inspect_project),
    ArgumentsReader("plant_mapping_built", _load_valley, _map_valley),
    ArgumentsReader("agent_terminal_started", _recorded_session),
]


@pytest.mark.parametrize("reader", READERS, ids=lambda r: r.tool)
def test_a_line_whose_arguments_its_reader_refuses_is_refused_at_the_audit_logs_place(
    made: OpenProject, reader: ArgumentsReader
) -> None:
    """A line of ``reader.tool`` with no arguments refuses its reader as invalid at the log.

    The refusal's place equals the place :func:`~tcip_mcp.audit.audit_entries` raises for the
    same log once its last entry no longer decodes.
    """
    reader.prepare(made)
    append_or_raise(audit_line(reader.tool, {}, None, "ok"), scope=made.root)

    with pytest.raises(ts.ReadRefusedError) as refused:
        reader.read(made)
    damage_last_log_entry(audit_log_key(made.root))
    with pytest.raises(ts.ReadRefusedError) as log_refused:
        audit_entries(made.root)

    assert (
        refused.value.served.where,
        refused.value.served.refusal.kind,
    ) == (log_refused.value.served.where, "invalid")


def test_a_run_cancel_over_a_launch_line_naming_no_run_is_recorded_at_the_audit_logs_place(
    project: Path,
) -> None:
    """A run's cancel over a log holding a launch line with no arguments answers recorded.

    The run's detail reads the log's launch lines, so that line refuses the read after the cancel
    was written; the answer carries the refusal as invalid at the place
    :func:`~tcip_mcp.audit.audit_entries` raises for the log, the cancel stands and its own line
    lands ``ok``.
    """
    from tcip_mcp.experiments import CANCEL_FILE
    from tests._audit_fixtures import audit_rows
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        opened_run,
    )
    from tests.test_hpo_cancel import cancel_unread

    run = opened_run(
        project, detection_config(project / "ds", project=project)
    )
    append_or_raise(
        audit_line("launch_training", {}, None, "ok"), scope=project
    )

    unread = cancel_unread(project, run)

    assert (run / CANCEL_FILE).is_file()
    assert [
        row["status"] for row in audit_rows(project, "cancel_training")
    ] == ["ok"]
    damage_last_log_entry(audit_log_key(project))
    with pytest.raises(ts.ReadRefusedError) as log_refused:
        audit_entries(project)
    assert (unread["where"], unread["refusal"]["kind"]) == (
        log_refused.value.served.where,
        "invalid",
    )
