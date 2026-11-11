"""A call carrying an argument its tool's listed schema does not hold is refused, never run.

Each case runs on the two routes a call takes to a tool: the server's own ``call_tool``, which
the suite and the console command make, and a client session over streams. A refused call names
each argument the tool does not take and the arguments it does, runs no body, and leaves no
record; a call carrying listed arguments alone answers as it always has.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import anyio
import pytest
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent

from tcip_mcp import dataset_layout, traits
from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
from tcip_mcp.workspace import BoundProject
from tests import _trait_fixtures as fx
from tests._audit_fixtures import audit_rows
from tests._producer_fixtures import write_image
from tests._server_calls import client_server, results_through_handshake

DIRECT, WIRE = "direct", "wire"
ROUTES = (DIRECT, WIRE)
"""The server's own ``call_tool``, and a client session over in-memory streams."""


def _text(result: CallToolResult) -> str:
    """Return the text of the one block a call answered with."""
    (block,) = result.content
    assert isinstance(block, TextContent), block
    return block.text


def _call(
    route: str, tool: str, arguments: dict, bound: BoundProject
) -> tuple[bool, str]:
    """Return whether ``tool`` called with ``arguments`` on ``route`` was refused, and its text."""
    if route == WIRE:
        (result,) = results_through_handshake([(tool, arguments)], bound=bound)
        return bool(result.is_error), _text(result)
    try:
        answer = asyncio.run(client_server(bound).call_tool(tool, arguments))
    except ToolError as exc:
        return True, str(exc)
    assert isinstance(answer, CallToolResult), answer
    return False, _text(answer)


def _listed(bound: BoundProject, tool: str) -> list[str]:
    """Return the argument names ``tool``'s schema lists, from the server's own listing."""
    (listed,) = [
        t for t in anyio.run(client_server(bound).list_tools) if t.name == tool
    ]
    return list(listed.input_schema["properties"])


def _raw_images(project: Path) -> Path:
    """Write one image into a source folder in ``project`` and return the folder."""
    write_image(project / "raw" / "a.png")
    return project / "raw"


def _ingested_file(project: Path) -> Path:
    """Return where the image ``_raw_images`` writes lands when it is ingested."""
    return image_dir(project, UNDATED_BUCKET) / "a.png"


UNLISTED = ["dataset_root", "destination", "project", "bound", "actor"]
"""Arguments ``ingest_images`` does not list: two it never took, three the server supplies."""

REFUSED = [
    *(("ingest_images", [name]) for name in UNLISTED),
    ("ingest_images", ["dataset_root", "destination"]),
    ("inspect_project", ["dataset_root"]),
]
"""A tool and the unlisted names one call to it states: one, two, and to a tool taking none."""


def _refusal(
    route: str,
    tool: str,
    unlisted: list[str],
    project: Path,
    bound: BoundProject,
) -> str:
    """Return the sentence ``tool`` is refused with when called naming ``unlisted`` on ``route``.

    The call also states a source folder, which ``tool`` takes where it lists one. The refusal
    is asserted to have run no body: no image is stored and the audit log is as it was.
    """
    before = audit_rows(project)
    arguments = {name: str(project / "elsewhere") for name in unlisted}
    if "source" in _listed(bound, tool):
        arguments["source"] = str(_raw_images(project))

    refused, sentence = _call(route, tool, arguments, bound)

    assert refused, sentence
    assert not _ingested_file(project).exists()
    assert dataset_layout.list_dates(project) == []
    assert audit_rows(project) == before
    return sentence


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(("tool", "unlisted"), REFUSED)
def test_a_call_naming_arguments_the_tool_does_not_list_is_refused_before_it_runs(
    project: Path,
    bound: BoundProject,
    route: str,
    tool: str,
    unlisted: list[str],
) -> None:
    """The refusal names each unlisted argument, then lists the tool's own or says it takes none."""
    sentence = _refusal(route, tool, unlisted, project, bound)

    named, _, takes = sentence.partition("; ")
    assert [repr(name) in named for name in unlisted] == [True] * len(unlisted)
    listed = _listed(bound, tool)
    assert (
        takes == f"it takes {', '.join(listed)}"
        if listed
        else takes == "it takes no argument"
    ), sentence


@pytest.mark.parametrize(("tool", "unlisted"), REFUSED)
def test_the_two_routes_refuse_a_call_in_the_same_sentence(
    project: Path, bound: BoundProject, tool: str, unlisted: list[str]
) -> None:
    direct, wire = (
        _refusal(route, tool, unlisted, project, bound) for route in ROUTES
    )

    assert direct == wire


@pytest.mark.parametrize("route", ROUTES)
def test_a_tool_taking_no_argument_is_called_with_none(
    bound: BoundProject, route: str
) -> None:
    assert _listed(bound, "inspect_project") == []

    refused, answer = _call(route, "inspect_project", {}, bound)

    assert not refused, answer


@pytest.mark.parametrize("route", ROUTES)
def test_a_call_naming_a_tool_the_server_does_not_hold_answers_the_unknown_tool_error(
    bound: BoundProject, route: str
) -> None:
    refused, sentence = _call(
        route, "no_such_tool", {"dataset_root": "ds"}, bound
    )

    assert refused
    assert sentence == "Unknown tool: no_such_tool"


@pytest.mark.parametrize("route", ROUTES)
def test_an_ingest_naming_listed_arguments_alone_writes_the_image_and_leaves_its_line(
    project: Path, bound: BoundProject, route: str
) -> None:
    refused, answer = _call(
        route,
        "ingest_images",
        {
            "source": str(_raw_images(project)),
            "copy": True,
            "recursive": False,
        },
        bound,
    )

    assert not refused, answer
    assert _ingested_file(project).is_file()
    (line,) = audit_rows(project, "ingest_images")
    assert line["status"] == "ok"


@pytest.mark.parametrize("route", ROUTES)
def test_a_listed_argument_holding_a_model_is_admitted_with_the_models_own_keys(
    project: Path, bound: BoundProject, route: str
) -> None:
    entry = fx.with_operationalization(
        fx.COUNT_SPEC,
        traits.PER_IMAGE_COUNT,
        measured_subject=fx.COUNT_SUBJECT,
    )

    refused, answer = _call(
        route,
        "propose_trait",
        {
            "entry": entry.model_dump(mode="json"),
            "rationale": "the breeder described the count in their own field-scoring terms",
        },
        bound,
    )

    assert not refused, answer
    assert traits.read_trait(fx.COUNT_TRAIT, project).latest.entry == entry
