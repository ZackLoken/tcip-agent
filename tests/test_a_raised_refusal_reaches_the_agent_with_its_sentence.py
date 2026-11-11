"""A refusal a tool raises reaches the agent as an error answer carrying its class and sentence.

The door cases run the real ``tcip-pipeline`` server in memory through a handshake
(``tests._server_calls.results_through_handshake``), each over a project the
platform's own producers set up and a record of it then damaged through
``tests._record_damage_fixtures``; the expected sentence is the one the platform's own reader of
that record raises. The cases no door reaches (an empty sentence, an unknown trait's ``KeyError``,
the SDK's resource and protocol errors) declare one function as a server's only tool.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import tcip_store as ts
from tcip_mcp.dataset_layout import image_dir, label_key_of
from tests._audit_fixtures import audit_rows
from tests._producer_fixtures import label_image, write_image
from tests._record_damage_fixtures import damage_record
from tests._server_calls import results_through_handshake

if TYPE_CHECKING:
    from collections.abc import Callable

    from tcip_mcp.workspace import BoundProject

REFUSED_DOCUMENT = b'{"frame": "not a label document"}'
"""Bytes that decode as JSON and that the label-document reader refuses."""

UNDECODABLE = b"{not valid json"
"""Bytes that do not decode as JSON."""


def _text(result: object) -> str:
    """Return the one text block of a refused call's result, asserting it is an error."""
    assert result.is_error, result  # type: ignore[attr-defined]
    (content,) = result.content  # type: ignore[attr-defined]
    return content.text


def _photograph(project: Path) -> Path:
    """Write a 16 px photograph into ``project``'s ``2026-04-01`` capture and return its path."""
    return write_image(
        image_dir(project, "2026-04-01") / "a_1.png", (16, 16), (90, 90, 90)
    )


def test_a_label_document_the_save_cannot_read_answers_its_class_and_sentence(
    project: Path, bound: BoundProject
) -> None:
    from tcip_annotation.json_io import (
        UnreadableLabelDocumentError,
        read_label_document,
    )

    image = _photograph(project)
    label_image(image, [], project=project)
    key = label_key_of(str(image))
    damage_record(key, REFUSED_DOCUMENT)
    with pytest.raises(UnreadableLabelDocumentError) as refused:
        read_label_document(key)

    (result,) = results_through_handshake(
        [("save_annotations", {"image_path": str(image), "annotations": []})],
        bound=bound,
    )

    assert _text(result) == (
        "Error executing tool save_annotations: UnreadableLabelDocumentError: "
        f"{refused.value}"
    )


def test_a_gui_snapshot_that_will_not_read_answers_its_class_and_sentence_to_agent_and_console(
    project: Path, bound: BoundProject, capsys: pytest.CaptureFixture[str]
) -> None:
    from tcip_mcp.web_client import (
        GUI_STATE_DEFAULTS,
        gui_snapshot_key,
        read_gui_snapshot,
        write_gui_snapshot,
    )
    from tcip_web.cli import main

    write_gui_snapshot(project, GUI_STATE_DEFAULTS)
    damage_record(gui_snapshot_key(project), b'{"active_tab": "not-a-tab"}')
    with pytest.raises(ts.ReadRefusedError) as refused:
        read_gui_snapshot(project)
    said = f"Error executing tool view_gui_state: ReadRefusedError: {refused.value}"

    (result,) = results_through_handshake(
        [("view_gui_state", {})], bound=bound
    )
    code = main(
        ["view-gui-state", "--project", str(project), "--user", "tester"]
    )

    assert _text(result) == said
    assert code == 2
    assert capsys.readouterr().err.strip() == said


def test_an_audited_door_raising_past_its_handlers_answers_the_sentence_beside_one_exception_line(
    project: Path, bound: BoundProject
) -> None:
    """A door's raise answers its class and sentence, and its log holds the raw sentence once.

    A photograph is ingested and labeled through the platform, then its label document's bytes
    are damaged: ``export_labels`` and ``register_dataset`` read it with the store's plain read,
    whose ``DecodeError`` no handler of theirs catches. ``register_plant_registry`` meets the same
    raise reading a damaged registry record before a second registration under its name.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        plant_registry_key,
    )
    from tcip_mcp.tools.ingest_tools import IngestManifest, ingest_images
    from tcip_mcp.tools.phenology_tools import register_plant_registry
    from tcip_mcp.traits import registered_crops
    from tests._mapping_fixtures import PLANTS, write_plant_csv

    source = write_image(project / "raw" / "IMG_0001.png", (16, 16))
    ingested = ingest_images(project, source=str(source.parent), actor=None)
    assert isinstance(ingested, IngestManifest), ingested
    (image,) = (project / "images").rglob("IMG_0001.png")
    label_image(
        image, [{"subject": "bud", "bbox": [2, 2, 8, 8]}], project=project
    )
    key = label_key_of(str(image))
    damage_record(key, UNDECODABLE)
    with pytest.raises(ts.DecodeError) as refused:
        ts.read(key)

    crop = sorted(registered_crops())[0]
    plants = write_plant_csv(project / "plants.csv", PLANTS)
    registry = {
        "name": "north-block",
        "csv_paths": [str(plants)],
        "crop": crop,
        "site": "north orchard",
    }
    first = register_plant_registry(project, **registry, actor=None)
    assert "error" not in first, first
    registry_key = plant_registry_key(project, "north-block")
    damage_record(registry_key, UNDECODABLE)
    with pytest.raises(ts.DecodeError) as registry_refused:
        ts.read(registry_key)

    exported, registered, reregistered = results_through_handshake(
        [
            (
                "export_labels",
                {"dataset_root": str(project), "destination": "labels.zip"},
            ),
            (
                "register_dataset",
                {"dataset_root": str(project), "crop": crop},
            ),
            ("register_plant_registry", registry),
        ],
        bound=bound,
    )

    for door, result, sentence in (
        ("export_labels", exported, refused.value),
        ("register_dataset", registered, refused.value),
        ("register_plant_registry", reregistered, registry_refused.value),
    ):
        assert _text(result) == (
            f"Error executing tool {door}: DecodeError: {sentence}"
        )
        (line,) = audit_rows(project, door)[-1:]
        assert (line["status"], line["error"]) == ("exception", str(sentence))
    assert len(audit_rows(project, "export_labels")) == 1
    assert len(audit_rows(project, "register_dataset")) == 1
    assert not (project / "labels.zip").exists()


def test_a_creation_whose_binding_refuses_answers_its_class_and_sentence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server for no project creates one, and the binding that follows refuses with its sentence.

    With ``TCIP_WORKSPACE`` blank the creation door writes the project, and the binding the server
    takes of it (``server.binding_for``) refuses naming the variable.
    """
    from tcip_mcp.workspace import workspace_from_environment
    from tests._web_fixtures import INVARIANCES

    monkeypatch.setenv("TCIP_WORKSPACE", " ")
    with pytest.raises(ValueError) as refused:
        workspace_from_environment()

    (result,) = results_through_handshake(
        [
            (
                "initialize_project",
                {
                    "project_path": str(tmp_path.parent / "ridge_block"),
                    "display_name": "Ridge block",
                    "site": "north orchard",
                    "invariances": INVARIANCES.model_dump(),
                },
            )
        ],
        bound=None,
    )

    assert _text(result) == (
        f"Error executing tool initialize_project: ValueError: {refused.value}"
    )


# ── the adapter's own contract ───────────────────────────────────────────────────────────


def _refusal_of_the_one_declared_tool(
    fn: Callable[..., object],
    arguments: dict,
    monkeypatch: pytest.MonkeyPatch,
    *,
    bound: BoundProject | None = None,
) -> BaseException:
    """Call ``fn`` with ``arguments`` through a server for ``bound`` declaring ``fn`` alone.

    ``fn`` is declared by the server's own ``tool()`` over an emptied declaration list, so the
    server's registration serves it and nothing else. Returns what the call raised; asserts it
    raised.
    """
    from tcip_mcp import server
    from tests._server_calls import client_server

    monkeypatch.setattr(server, "_TOOLS", [])
    server.tool()(fn)
    with pytest.raises(BaseException) as raised:
        asyncio.run(client_server(bound).call_tool(fn.__name__, arguments))
    return raised.value


def silent() -> dict:
    """Raise a ``ValueError`` with no sentence."""
    raise ValueError


def refused_resource() -> dict:
    """Raise the SDK's ``ResourceError``."""
    from mcp.server.mcpserver.exceptions import ResourceError

    raise ResourceError("the resource refused")


def protocol_refusal() -> dict:
    """Raise the SDK's protocol error, ``MCPError``."""
    from mcp.shared.exceptions import MCPError

    raise MCPError(-32602, "the request is malformed")


def test_an_exception_with_no_sentence_answers_its_class_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        str(_refusal_of_the_one_declared_tool(silent, {}, monkeypatch))
        == "Error executing tool silent: ValueError"
    )


def test_a_key_error_answers_its_quoted_sentence_caused_by_it_and_logged_with_its_traceback(
    project: Path,
    bound: BoundProject,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown trait, a ``KeyError``, answers its sentence as Python spells it, quoted.

    The trait reader is declared as the one tool of a server for the project. The answer's cause
    chain reaches the reader's own exception, and the server's ERROR record carries its traceback.
    """
    from tcip_mcp.traits import TraitUnknownError, read_trait

    with pytest.raises(TraitUnknownError) as refused:
        read_trait("unicorn", project)
    caplog.set_level(logging.ERROR, logger="tcip_mcp.server")

    answer = _refusal_of_the_one_declared_tool(
        read_trait, {"trait": "unicorn"}, monkeypatch, bound=bound
    )

    sentence = str(refused.value)
    assert sentence[0] == sentence[-1] and sentence[0] in "'\""
    assert str(answer) == (
        f"Error executing tool read_trait: TraitUnknownError: {sentence}"
    )
    assert answer.__cause__ is not None
    cause = answer.__cause__.__cause__
    assert isinstance(cause, TraitUnknownError)
    assert cause.args == refused.value.args
    (logged,) = [r for r in caplog.records if r.name == "tcip_mcp.server"]
    assert logged.levelno == logging.ERROR
    assert logged.exc_info is not None and logged.exc_info[1] is cause


def test_a_resource_error_and_a_protocol_error_pass_through_as_the_sdk_answers_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp.shared.exceptions import MCPError

    resource = _refusal_of_the_one_declared_tool(
        refused_resource, {}, monkeypatch
    )
    protocol = _refusal_of_the_one_declared_tool(
        protocol_refusal, {}, monkeypatch
    )

    assert str(resource) == (
        "Error executing tool refused_resource: the resource refused"
    )
    assert isinstance(protocol, MCPError)
    assert protocol.message == "the request is malformed"
