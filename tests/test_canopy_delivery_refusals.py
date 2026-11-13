"""The canopy delivery tells an absent canopy document from one that could not be read.

Each scene is ``test_orthomosaic_tools``'s canopy scene: a raster bucket published through the
platform, a registry registered through it, and the canopy boundaries written by the platform's
own label writer, then damaged where the case is a stored record that will not read
(``tests._record_damage_fixtures``). The tool cases call ``deliver_orthomosaic_plant_counts``
through the server; the route case posts to the count route.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir, label_key_of
from tests import _trait_fixtures as fx
from tests._mapping_fixtures import POSITION_ERROR_M
from tests._record_damage_fixtures import damage_record
from tests._server_calls import called_through_the_server
from tests.test_a_raised_refusal_reaches_the_agent_with_its_sentence import (
    REFUSED_DOCUMENT,
    UNDECODABLE,
)
from tests.test_orthomosaic_tools import (
    _canopy_setup,
    _plant_registry,
    _plants_csv_at,
    _write_canopy_document,
)

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from tcip_mcp.workspace import BoundProject

TOOL = "deliver_orthomosaic_plant_counts"
CANOPY_BOXES = [(5.0, 5.0, 15.0, 15.0)]
"""One canopy boundary around plot0, the registry's one plant."""


def _scene(project: Path) -> tuple[Path, dict, BoundProject]:
    """Return the canopy scene's raster, the tool's arguments over it, and the project's binding.

    The project is made through the platform's creation door with the delivery traits confirmed,
    a one-detection raster bucket published over a 64 by 64 raster of dataset ``ds``, and a
    registry of one plant, ``plot0``, inside the boundary :data:`CANOPY_BOXES` holds. No canopy
    document is written.
    """
    from tcip_mcp.server import binding_for
    from tests._web_fixtures import named_project

    named_project(project, "Orchard")
    fx.seed_delivery_traits(project)
    fx.seed_confirmed_orthomosaic_count(project, "stem_count")
    _root, raster_path, bucket = _canopy_setup(
        project, [(8.0, 8.0, 12.0, 12.0)]
    )
    registry = _plant_registry(
        project,
        _plants_csv_at(project, raster_path, [("plot0", 10.0, 10.0)]),
    )
    arguments = {
        "dataset_root": str(bucket.root),
        "bucket": bucket.name,
        "plant_registry": registry,
        "output_csv_path": str(project / "counts.csv"),
        "delivered_phenotype": "stem_count",
        "plants": ["plot0"],
        "canopy_subject": "canopy",
        "position_error_m": POSITION_ERROR_M,
    }
    return raster_path, arguments, binding_for(project)


def _refusal(answer: object) -> str:
    """Return the error value a tool answered, asserting it answered one rather than raised."""
    assert isinstance(answer, dict) and set(answer) == {"error"}, answer
    return answer["error"]


def _reader_sentence(raster_path: Path) -> str:
    """Return the sentence the label-document reader refuses the raster's stored document with."""
    from tcip_annotation.json_io import (
        UnreadableLabelDocumentError,
        read_label_document,
    )

    with pytest.raises(UnreadableLabelDocumentError) as refused:
        read_label_document(label_key_of(str(raster_path)))
    return str(refused.value)


def test_an_absent_canopy_document_is_refused_with_the_advice_to_author_it(
    tmp_path: Path,
) -> None:
    raster_path, arguments, bound = _scene(tmp_path)

    error = _refusal(called_through_the_server(TOOL, arguments, bound=bound))

    assert "'canopy'" in error and f"'{raster_path.stem}'" in error
    assert "author the canopy boundaries" in error
    assert "do not read" not in error
    assert "Pass default=" not in error
    assert not (tmp_path / "counts.csv").exists()


def test_a_canopy_document_that_does_not_decode_is_refused_with_the_stores_sentence(
    tmp_path: Path,
) -> None:
    raster_path, arguments, bound = _scene(tmp_path)
    _write_canopy_document(raster_path, CANOPY_BOXES)
    damage_record(label_key_of(str(raster_path)), UNDECODABLE)
    sentence = _reader_sentence(raster_path)
    assert "exists but does not decode" in sentence

    error = _refusal(called_through_the_server(TOOL, arguments, bound=bound))

    assert "could not be read" in error and sentence in error
    assert "author" not in error
    assert not (tmp_path / "counts.csv").exists()


@pytest.mark.parametrize(
    "stored", [REFUSED_DOCUMENT, b"null"], ids=["not-a-document", "null"]
)
def test_a_canopy_document_the_reader_refuses_is_refused_with_the_readers_sentence(
    tmp_path: Path, stored: bytes
) -> None:
    raster_path, arguments, bound = _scene(tmp_path)
    _write_canopy_document(raster_path, CANOPY_BOXES)
    damage_record(label_key_of(str(raster_path)), stored)
    sentence = _reader_sentence(raster_path)

    error = _refusal(called_through_the_server(TOOL, arguments, bound=bound))

    assert "could not be read" in error and sentence in error
    assert "author" not in error
    assert not (tmp_path / "counts.csv").exists()


def test_an_intact_canopy_document_delivers_through_the_server(
    tmp_path: Path,
) -> None:
    """A control: the scene with its canopy boundaries intact delivers under an acknowledgment."""
    from tcip_mcp.pipelines.postprocessing.plant_mapping import load_registry
    from tcip_mcp.tools.orthomosaic_tools import orthomosaic_plant_counts
    from tests._chain_fixtures import acknowledged

    raster_path, arguments, bound = _scene(tmp_path)
    _write_canopy_document(raster_path, CANOPY_BOXES)

    def deliver(acknowledgment_id: str | None) -> object:
        if acknowledgment_id is None:
            return orthomosaic_plant_counts(
                tmp_path,
                Path(arguments["dataset_root"]),
                arguments["bucket"],
                load_registry(tmp_path, arguments["plant_registry"]),
                arguments["output_csv_path"],
                arguments["delivered_phenotype"],
                arguments["plants"],
                canopy_subject=arguments["canopy_subject"],
                position_error_m=arguments["position_error_m"],
                door=TOOL,
                actor=None,
            )
        return called_through_the_server(
            TOOL,
            {**arguments, "acknowledgment_id": acknowledgment_id},
            bound=bound,
        )

    answer = acknowledged(tmp_path, deliver)

    assert isinstance(answer, dict) and "error" not in answer, answer
    assert (tmp_path / "counts.csv").exists()


def test_the_count_route_refuses_an_undecodable_canopy_document_with_the_stores_sentence(
    client: TestClient, tmp_path: Path
) -> None:
    """The count route refuses it before the gate: no acknowledgment is sent, none recorded."""
    from tcip_mcp.delivery import DeliveryRefusal
    from tests.test_tcip_web_results_routes import (
        _export_count,
        _orthomosaic,
        _per_plant,
    )

    bucket, registry = _orthomosaic(tmp_path)
    raster_path = image_dir(tmp_path, UNDATED_BUCKET) / "mosaic.tif"
    _write_canopy_document(raster_path, CANOPY_BOXES)
    damage_record(label_key_of(str(raster_path)), UNDECODABLE)
    sentence = _reader_sentence(raster_path)
    body = _per_plant(bucket, registry)
    body["delivery"].update(
        canopy_subject="canopy", position_error_m=POSITION_ERROR_M
    )

    resp = _export_count(client, body)

    assert resp.status_code == 400, resp.text[:300]
    message = DeliveryRefusal.model_validate(resp.json()["detail"]).message
    assert "could not be read" in message and sentence in message
    assert "author" not in message
    assert client.get("/api/results/delivery-events").json()["records"] == []
