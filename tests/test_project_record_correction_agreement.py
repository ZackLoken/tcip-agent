"""The console door and the web door record one correction of a project's invariances alike.

``tcip write-project-record invariances`` and ``POST /api/projects/invariances`` each correct a
project made through ``initialize_project`` with the same block by the same person; the blocks
they persist and the audit lines they leave are compared with each other, the two differences
being each line's time and the minted record id it carries.
"""

from __future__ import annotations

from tcip_mcp.project_record import CaptureInvariances
from tests._audit_fixtures import audit_rows
from tests._cli_fixtures import run_tcip
from tests._web_fixtures import INVARIANCES, named_project


def _flags(block: CaptureInvariances) -> list[str]:
    """``block`` as the console door's flags take it."""
    flags: list[str] = []
    for name, value in block.model_dump().items():
        flag = name.replace("_", "-")
        if isinstance(value, bool):
            flags.append(f"--{flag}" if value else f"--no-{flag}")
        else:
            flags += [f"--{flag}", str(value)]
    return flags


def _recorded(line: dict, record_id: str) -> dict:
    """``line`` without what differs between two acts of one correction: its time, and the
    record id its arguments carry, which must be ``record_id``.
    """
    assert line["arguments"]["id"] == record_id
    return {
        **{k: v for k, v in line.items() if k != "timestamp"},
        "arguments": {k: v for k, v in line["arguments"].items() if k != "id"},
    }


def test_the_console_and_web_doors_record_one_correction_alike(
    client, tmp_path
):
    """The same block corrected by the same person through the console door on one project and
    the web door on another persists the same block on both, and leaves on each one
    ``project_record_corrected`` line that matches the other's once its time is dropped and its
    record id, checked to be its own project's, is set aside.
    """
    from tcip_mcp.project_record import read_record
    from tcip_web.routes.projects import InvariancesRequest

    console = named_project(tmp_path.parent / "console_block", "Console block")
    web = named_project(tmp_path.parent / "web_block", "Web block")
    corrected = CaptureInvariances.model_validate(
        INVARIANCES.model_dump() | {"vertical_flip": False, "rotation": 5.0}
    )

    form = ["invariances", str(console.root), *_flags(corrected)]
    ran = run_tcip("write-project-record", [*form, "--user", "tester"])
    resp = client.post(
        "/api/projects/invariances",
        json=InvariancesRequest(
            id=web.id, invariances=corrected, user="tester"
        ).model_dump(),
    )

    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert resp.status_code == 200, resp.text
    assert read_record(console.root).invariances == (
        read_record(web.root).invariances
    )
    (by_console,) = audit_rows(console.root, "project_record_corrected")
    (by_web,) = audit_rows(web.root, "project_record_corrected")
    assert _recorded(by_console, console.id) == _recorded(by_web, web.id)
