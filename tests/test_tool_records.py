"""The tool modules' stored records, each written by its producer and read back through its model.

Each record kind is written through the platform's own door and read back through the model its
reader validates with, which must equal what the producer was handed; a stored record missing a
field the model requires is refused by that reader, naming the field.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

import tcip_store as ts
from tcip_mcp.dataset_layout import image_dir, image_root
from tests._producer_fixtures import kept

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from tcip_web.state import OpenProject


def test_a_friction_report_reads_back_as_the_report_its_door_was_handed(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.meta_tools import (
        FrictionReport,
        read_memory,
        report_friction,
    )

    result = report_friction(
        tmp_path,
        category="ambiguous_data",
        detail="two captures share one date",
        context={"capture": "2026-05-01"},
        user_disagreement=True,
        actor=None,
    )

    assert read_memory(
        FrictionReport, str(tmp_path), result["id"]
    ) == FrictionReport(
        timestamp=result["timestamp"],
        category="ambiguous_data",
        detail="two captures share one date",
        context={"capture": "2026-05-01"},
        user_disagreement=True,
    )


def test_a_friction_report_missing_its_category_is_refused_naming_it(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.meta_tools import (
        FrictionReport,
        memory_key,
        read_memory,
        report_friction,
    )

    result = report_friction(
        tmp_path, category="missing_tool", detail="x", actor=None
    )
    key = memory_key(FrictionReport, str(tmp_path), result["id"])
    stored = dict(ts.read(key))
    del stored["category"]
    ts.replace(key, stored)

    with pytest.raises(ValidationError, match="category"):
        read_memory(FrictionReport, str(tmp_path), result["id"])


def test_a_retrospective_reads_back_as_the_record_its_door_was_handed(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.meta_tools import (
        Retrospective,
        read_memory,
        write_retrospective,
    )

    result = write_retrospective(
        tmp_path,
        project_id="p",
        task="t",
        worked="w",
        did_not_work="d",
        assumptions_wrong="a",
        knowledge_for_future="k",
        missing_or_hard_tools="m",
        would_do_differently="o",
        actor=None,
    )

    assert read_memory(
        Retrospective, str(tmp_path), result["id"]
    ) == Retrospective(
        timestamp=result["timestamp"],
        project_id="p",
        task="t",
        worked="w",
        did_not_work="d",
        assumptions_wrong="a",
        knowledge_for_future="k",
        missing_or_hard_tools="m",
        would_do_differently="o",
    )


def test_a_retrospective_missing_a_section_is_refused_naming_it(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.meta_tools import (
        Retrospective,
        memory_key,
        read_memory,
        write_retrospective,
    )

    result = write_retrospective(
        tmp_path,
        project_id="p",
        task="t",
        worked="w",
        did_not_work="d",
        actor=None,
    )
    key = memory_key(Retrospective, str(tmp_path), result["id"])
    stored = dict(ts.read(key))
    del stored["worked"]
    ts.replace(key, stored)

    with pytest.raises(ValidationError, match="worked"):
        read_memory(Retrospective, str(tmp_path), result["id"])


def test_a_registered_dataset_reads_back_as_the_entry_its_door_registered(
    tmp_path: Path,
) -> None:
    from tcip_mcp.registry_paths import stored_path
    from tcip_mcp.tools.project_tools import (
        DatasetEntry,
        read_datasets,
        register_dataset,
    )
    from tests.test_project_tools import _make_dataset

    root = tmp_path / "ds"
    _make_dataset(root)

    result = register_dataset(tmp_path, str(root), crop="currant", actor=None)

    assert read_datasets(tmp_path) == [
        DatasetEntry(id=result["id"], path=stored_path(root, tmp_path))
    ]


def test_a_dataset_entry_missing_its_path_is_refused_naming_it(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.project_tools import (
        dataset_registry_key,
        read_datasets,
        register_dataset,
    )
    from tests.test_project_tools import _make_dataset

    root = tmp_path / "ds"
    _make_dataset(root)
    register_dataset(tmp_path, str(root), crop="currant", actor=None)
    ts.replace(
        dataset_registry_key(tmp_path),
        [{"id": entry.id} for entry in read_datasets(tmp_path)],
    )

    with pytest.raises(ValidationError, match="path"):
        read_datasets(tmp_path)


def _proposed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Propose the stub engine's one candidate on a dated image.

    Returns the image, the candidate and the run's answer.
    """
    from tcip_mcp.tools.proposal_tools import propose_annotations
    from tests._producer_fixtures import install_stub_proposer
    from tests.test_proposal_staging_record import _candidate, _make_image

    image = image_dir(tmp_path, "2026-04-01") / "leaf.jpg"
    _make_image(image)
    candidate = _candidate(0, 5.0)
    install_stub_proposer(monkeypatch, [candidate])
    result = propose_annotations(
        tmp_path, image_path=str(image), engine="stub", actor=None
    )
    assert result.staging_id is not None, result
    return image, candidate, result


def test_a_staged_proposal_reads_back_as_the_engine_candidates_and_image_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcip_mcp.pipelines.raster_source import SourceHeader, content_identity
    from tcip_mcp.tools.proposal_tools import ProposalStaging, _staging_key_for

    image, candidate, proposed = _proposed(tmp_path, monkeypatch)

    staging = ProposalStaging.model_validate(
        ts.read(_staging_key_for(str(image), proposed.staging_id))
    )
    with SourceHeader(image).open_at_route_count() as raster:
        identity = content_identity(raster)
    assert staging.engine == "stub"
    assert [c.model_dump(mode="json") for c in staging.candidates] == [
        json.loads(json.dumps(candidate))
    ]
    assert staging.image_identity == identity


def test_a_staged_proposal_missing_its_engine_is_refused_naming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcip_mcp.tools.proposal_tools import _staging_key_for, stage_proposals

    image, _candidate, proposed = _proposed(tmp_path, monkeypatch)
    key = _staging_key_for(str(image), proposed.staging_id)
    stored = dict(ts.read(key))
    del stored["engine"]
    ts.replace(key, stored)

    with pytest.raises(ValidationError, match="engine"):
        stage_proposals(
            tmp_path,
            image_path=str(image),
            assignments=kept(proposed, (0, "bud")),
            actor=None,
        )


def test_an_interrupted_raster_pass_reads_back_as_its_identity_and_first_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.generic_predictor import SliceBatch
    from tcip_mcp.pipelines.slicing import slice_lattice
    from tcip_mcp.tools.inference_tools import (
        RasterPassIdentity,
        _progress_key,
    )
    from tests.test_orthomosaic_tools import TILE
    from tests.test_raster_pass_resume import _interrupted, _setup

    ckpt, raster_path = _setup(tmp_path)
    bucket = "preds/2026-01-01"
    _interrupted(tmp_path, monkeypatch, bucket, ckpt, raster_path)
    root = tmp_path / "ds"

    identity = RasterPassIdentity.model_validate(
        ts.read(_progress_key(tmp_path, root, bucket, "identity"))
    )
    batch = SliceBatch.model_validate(
        ts.read(_progress_key(tmp_path, root, bucket, "batch-000000"))
    )

    checkpoint = load_registered_checkpoint(Path(ckpt), project=tmp_path)
    assert identity.producer == checkpoint.producer
    assert identity.assessment_id is None
    assert identity.tile_batch_size == 1
    assert (
        identity.raster_identity.width,
        identity.raster_identity.height,
    ) == (
        64,
        64,
    )
    assert identity.execution.tile_size == TILE
    assert batch.slices == slice_lattice(64, 64, TILE, 0.0)[:1]


def test_a_raster_pass_identity_missing_its_tile_batch_size_refuses_the_resume_naming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    from tcip_mcp.tools.inference_tools import _progress_key
    from tests.test_raster_pass_resume import _interrupted, _run, _setup

    ckpt, raster_path = _setup(tmp_path)
    bucket = "preds/2026-01-01"
    _interrupted(tmp_path, monkeypatch, bucket, ckpt, raster_path)
    key = _progress_key(tmp_path, tmp_path / "ds", bucket, "identity")
    stored = dict(ts.read(key))
    del stored["tile_batch_size"]
    with ts.transaction(key) as txn:
        txn.write(key, stored)

    result = _run(tmp_path, ckpt, raster_path, bucket, resume=True)

    assert "tile_batch_size" in result["error"]


_SHAPE = {
    "kind": "box",
    "xyxy": [1, 2, 30, 40],
    "color": "#FF0000",
    "tag": "gt",
}


def _pushed(
    client: TestClient, opened: OpenProject, opened_project: Path
) -> None:
    """Push one full canvas state for ``opened`` through the backend's route."""
    payload = {
        "project_id": opened.id,
        "tab": "annotate",
        "image_path": str(image_root(opened_project) / "a.jpg"),
        "viewport": {"x": 0, "y": 0, "w": 200, "h": 100, "scale": 1.0},
        "mode": "polygon",
        "active_subject": None,
        "cut_armed": False,
        "dirty": True,
        "user": "breeder",
        "classes": [{"name": "bud", "color": "#FF0000"}],
        "counts": {"boxes": 1},
        "shapes": [_SHAPE],
    }
    response = client.post("/api/canvas/state", json=payload)
    assert response.status_code == 200, response.text


def test_a_canvas_meta_record_missing_its_user_refuses_the_capture_naming_it(
    client: TestClient, opened: OpenProject, opened_project: Path
) -> None:
    pytest.importorskip("fastapi")
    from tcip_mcp.tools.vision_tools import capture_live_canvas
    from tcip_mcp.web_client import canvas_meta_key
    from tests._web_fixtures import bound_to

    _pushed(client, opened, opened_project)
    key = canvas_meta_key(str(opened_project))
    stored = dict(ts.read(key))
    del stored["user"]
    ts.replace(key, stored)

    with pytest.raises(ValidationError, match="user"):
        capture_live_canvas(bound_to(opened), refresh=False)
