"""Tests of the plant-mapping record's binding to its inputs, its dataset and its receipt.

The record is bound to its inputs, to a registered dataset by minted id, and to the receipt its
build leaves. Scenes are built through the platform's own producers (initialize_project,
register_dataset, build_plant_mapping, publish and the phenology measurement and delivery), over a
second registered trait; a hand-written record is written straight through the store.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import tcip_store as ts
from tcip_mcp.pipelines.postprocessing import plant_mapping
from tcip_mcp.tools.phenology_tools import build_plant_mapping
from tcip_mcp.tools.project_tools import register_dataset
from tcip_mcp.traits import registered_crops
from tests._audit_fixtures import held_by_another_writer
from tests._image_fixtures import write_geo_image
from tests._mapping_fixtures import (
    PHOTO_POSITION_ERROR_M,
    register_plant_registry_for,
    write_plant_csv,
)
from tests._web_fixtures import named_project
from tests.test_second_trait_acceptance import (
    BLOOM_STATE,
    FLOWERS,
    _seed_currant_bloom_trait,
)

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from tcip_web.state import OpenProject

PLANTS = [
    {"plot": "P1", "accession": "acc-A", "lat": 43.19670, "lon": -90.058000},
    {"plot": "P2", "accession": "acc-B", "lat": 43.19670, "lon": -90.058037},
]
DATES = ["2026-02-11", "2026-02-25"]
UNREAD_P2 = frozenset({f"{PLANTS[1]['plot']}_{DATES[0].replace('-', '')}"})
"""The second plant's first-date capture, left without a prediction document."""

POPULATION = [p["plot"] for p in PLANTS]
"""The plants every delivery here states, the ones the scene's registry places."""


def _init(tmp_path: Path) -> OpenProject:
    """Make ``tmp_path`` a project named "Orchard" (``tests._web_fixtures.named_project``)."""
    return named_project(tmp_path, "Orchard")


def _dataset(tmp_path: Path, name: str = "ds") -> Path:
    """Register a dataset root in ``tmp_path`` and return it.

    The dataset's own subject registry is ``FLOWERS``, declaring the second trait's positive state.
    """
    from tests._producer_fixtures import registry_over

    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    result = register_dataset(
        tmp_path, str(root), crop=sorted(registered_crops())[0]
    )
    assert "error" not in result, result
    registry_over(root, FLOWERS)
    return root


def _publish(project: Path, bucket: str, images: list[Path]) -> str:
    """Publish one bucket named ``bucket`` holding an open flower on each of ``images``.

    The bucket is published unassessed under ``project`` (``_chain_fixtures.published``) from a
    checkpoint whose scope declares the bloom state. Return its name.
    """
    from tests._chain_fixtures import published

    results = [
        {
            "image": str(p),
            "width": 8,
            "height": 8,
            "boxes": [[1.0, 1.0, 3.0, 3.0]],
            "scores": [0.9],
            "labels": [1],
            "cap": 2,
            "attributes": [[BLOOM_STATE.values.index("open")]],
        }
        for p in images
    ]
    return published(
        project, bucket, results, scope={"subject": "flower"}, registry=FLOWERS
    ).name


def _deliver(
    project: Path,
    *,
    trait: str,
    mapping_name: str,
    plants: list[str],
    dataset_root: Path,
    buckets: Iterable[str],
    output_csv_path: str,
) -> dict:
    """Run ``deliver_phenology_milestones``'s measurement and delivery through the library.

    The delivery ships under a breeder's acknowledgment; a refusal answers ``{"error": ...}``.
    """
    from tcip_mcp.pipelines.postprocessing import phenology
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        MappingDeliveryError,
    )
    from tests._chain_fixtures import acknowledged

    try:
        measurement = phenology.measure_phenology(
            project,
            trait=trait,
            mapping_name=mapping_name,
            dataset_root=dataset_root,
            buckets=list(buckets),
            plants=plants,
        )
        return acknowledged(
            project,
            lambda ack: phenology.deliver_phenology(
                project,
                measurement,
                curves=False,
                output_path=Path(output_csv_path),
                acknowledgment_id=ack,
                door=DOOR,
                actor=None,
            ),
            reason="mapping rails, not the assessment",
        )
    except (
        ValueError,
        MappingDeliveryError,
        *phenology.measurement_refusals(),
    ) as exc:
        return {"error": str(exc)}


DOOR = "test_mapping_delivery"


def _events(project: Path) -> list[dict]:
    """Return the delivery events :func:`_deliver` recorded under ``project``, as JSON forms."""
    from tcip_mcp.delivery import read_delivery_events

    return [
        e.model_dump(mode="json")
        for e in read_delivery_events(project)
        if e.door == DOOR
    ]


def _write_scene(
    dataset_root: Path,
    *,
    dates: list[str] = DATES,
    plants: list[dict] | None = None,
    unpredicted: frozenset[str] = frozenset(),
) -> tuple[Path, Path, dict[str, str]]:
    """Write real geolocated images for ``plants`` across ``dates`` and publish their buckets.

    ``plants`` defaults to ``PLANTS``. One bucket per date is published under the project the
    dataset sits in, holding a document for every image but the stems ``unpredicted`` names.
    Returns (images_root, plant_csv, preds_by_date).
    """
    pytest.importorskip("torch")
    plants = PLANTS if plants is None else plants
    images_root = dataset_root / "images"
    preds_by_date: dict[str, str] = {}
    for date in dates:
        base_time = datetime.strptime(date, "%Y-%m-%d").replace(
            hour=9, minute=30
        )
        images = []
        for j, plant in enumerate(plants):
            stem = f"{plant['plot']}_{date.replace('-', '')}"
            write_geo_image(
                images_root / date / f"{stem}.jpg",
                plant["lat"],
                plant["lon"],
                base_time + timedelta(minutes=j),
                h_pos_err=PHOTO_POSITION_ERROR_M,
            )
            if stem not in unpredicted:
                images.append(images_root / date / f"{stem}.jpg")
        preds_by_date[date] = _publish(
            dataset_root.parent, f"live/{date}", images
        )

    plant_csv = write_plant_csv(
        dataset_root.parent / f"{dataset_root.name}_plants.csv", plants
    )
    return images_root, plant_csv, preds_by_date


# ── rail 6: no project record ────────────────────────────────────────────


# ── rail 4: dataset identity, NAME_SEGMENT, variously-spelled roots, dataset mismatch ────


def test_build_plant_mapping_refuses_a_name_outside_name_segment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    res = build_plant_mapping(
        tmp_path,
        name="Not Legal!",
        images_root=str(tmp_path),
        plant_registry="unregistered",
    )
    assert "error" in res
    assert "lowercase" in res["error"]


def test_build_plant_mapping_over_an_unregistered_images_dir_names_register_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    images_root = tmp_path / "ds" / "images"
    (images_root / DATES[0]).mkdir(parents=True)
    res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry="unregistered",
    )
    assert "error" in res
    assert "register_dataset" in res["error"]
    assert not (tmp_path / ".tcip" / "state" / "plant_mappings").exists()


def test_build_plant_mapping_admits_the_dataset_images_root_spelled_variously(
    tmp_path: Path,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root)

    trailing = str(images_root) + os.sep
    res = build_plant_mapping(
        tmp_path,
        name="trailing-sep",
        images_root=trailing,
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in res, res

    forward = str(images_root).replace("\\", "/")
    res = build_plant_mapping(
        tmp_path,
        name="forward-slash",
        images_root=forward,
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in res, res

    res = build_plant_mapping(
        tmp_path,
        name="relative",
        images_root=str(images_root.relative_to(tmp_path)),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in res, res


def test_deliver_phenology_milestones_refuses_predictions_from_a_different_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root)
    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    other_root = _dataset(tmp_path, name="ds2")
    _, _, other_preds = _write_scene(other_root)

    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=other_root,
        buckets=other_preds.values(),
        output_csv_path=str(tmp_path / "out.csv"),
    )
    assert "error" in res
    assert "different dataset" in res["error"]


# ── rail 5: a bucket dated on a day the mapping does not name ──────────────────────────


def test_deliver_phenology_milestones_refuses_a_date_the_mapping_does_not_cover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(
        dataset_root, dates=[DATES[0]]
    )
    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    extra_date = "2026-03-01"
    extra_image = images_root / extra_date / "P1_20260301.jpg"
    write_geo_image(
        extra_image,
        PLANTS[0]["lat"],
        PLANTS[0]["lon"],
        datetime(2026, 3, 1, 9, 30),
        h_pos_err=PHOTO_POSITION_ERROR_M,
    )
    preds_by_date[extra_date] = _publish(
        tmp_path, f"live/{extra_date}", [extra_image]
    )

    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(tmp_path / "out.csv"),
    )
    assert "error" in res
    assert extra_date in res["error"]


# ── rail 1: a hand-written record refuses, missing provenance or missing receipt ────────


def test_deliver_phenology_milestones_refuses_a_hand_written_record_missing_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    _, _, preds_by_date = _write_scene(dataset_root)
    _seed_currant_bloom_trait(tmp_path)

    ts.replace(
        plant_mapping.plant_mapping_key(tmp_path, "forged"),
        {
            "assignments": {d: [] for d in DATES},
        },
    )
    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="forged",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv),
    )
    assert "error" in res
    assert "is not a record this reader decodes" in res["error"]
    assert not out_csv.exists()


def test_deliver_phenology_milestones_refuses_a_record_with_provenance_and_no_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    _, _, preds_by_date = _write_scene(dataset_root)
    _seed_currant_bloom_trait(tmp_path)

    record = {
        "name": "forged",
        "dataset_root": "ds",
        "dataset_id": "whatever-id",
        "built_at": "2026-02-11T00:00:00+00:00",
        "dates_requested": None,
        "dates": list(DATES),
        "match_radius": {},
        "plant_registry": {"name": "unregistered", "digest": "0" * 64},
        "capture_digests": {d: {} for d in DATES},
        "unreadable": {d: [] for d in DATES},
        "assignments": {d: [] for d in DATES},
        "supersedes": None,
    }
    ts.replace(plant_mapping.plant_mapping_key(tmp_path, "forged"), record)
    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="forged",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv),
    )
    assert "error" in res
    assert "receipt" in res["error"]
    assert not out_csv.exists()


# ── rail 3: a plant CSV rewritten in place refuses, naming the file ─────────────────────


def test_deliver_phenology_milestones_refuses_a_plant_csv_rewritten_in_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root)
    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    write_plant_csv(plant_csv, [{**PLANTS[0], "accession": "acc-Z"}])

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv),
    )
    assert "error" in res
    assert str(plant_csv) in res["error"]
    assert not out_csv.exists()


# ── rail 7: a readability flip on a capture this delivery reads refuses, naming the file ─


def test_an_unread_captures_bytes_going_bad_is_disclosed_never_opened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unread capture whose bytes go bad is disclosed, and the delivery succeeds.

    A recorded, mapped capture this delivery's own buckets carry no document for is unread:
    corrupting its bytes afterward surfaces as this capture's own disclosure, not a readability
    refusal.
    """
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(
        dataset_root, dates=[DATES[0]], unpredicted=UNREAD_P2
    )
    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    p2_stem = f"{PLANTS[1]['plot']}_{DATES[0].replace('-', '')}"
    (images_root / DATES[0] / f"{p2_stem}.jpg").write_bytes(
        b"not a real jpeg any more"
    )

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv),
    )
    assert "error" not in res, res
    assert out_csv.exists()

    events = _events(tmp_path)
    pm = events[-1]["plant_mapping"]
    assert pm["captures_unverified"] == [f"{DATES[0]}/{p2_stem}.jpg"]


def test_an_unread_captures_bytes_changing_in_place_does_not_refuse_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unread capture changed in place to another readable image does not refuse delivery.

    The capture's prediction is absent; the delivery succeeds and discloses that capture, alone,
    as unverified.
    """
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(
        dataset_root, dates=[DATES[0]], unpredicted=UNREAD_P2
    )
    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    p2_stem = f"{PLANTS[1]['plot']}_{DATES[0].replace('-', '')}"
    write_geo_image(
        images_root / DATES[0] / f"{p2_stem}.jpg",
        PLANTS[1]["lat"],
        PLANTS[1]["lon"],
        datetime(2026, 2, 11, 10, 45),
        h_pos_err=PHOTO_POSITION_ERROR_M,
    )

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv),
    )
    assert "error" not in res, res
    assert out_csv.exists()

    events = _events(tmp_path)
    pm = events[-1]["plant_mapping"]
    assert pm["captures_unverified"] == [f"{DATES[0]}/{p2_stem}.jpg"]


def test_a_non_delivered_mapping_date_is_never_walked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mapped date the delivery's buckets omit is disclosed as a bare date.

    A capture corrupted under that date leaves the delivery succeeding with the date as its one
    unverified entry.
    """
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root)
    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    stem = f"{PLANTS[0]['plot']}_{DATES[1].replace('-', '')}"
    (images_root / DATES[1] / f"{stem}.jpg").write_bytes(
        b"garbage, never read by this delivery"
    )

    delivered_preds = {DATES[0]: preds_by_date[DATES[0]]}
    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=delivered_preds.values(),
        output_csv_path=str(out_csv),
    )
    assert "error" not in res, res
    assert out_csv.exists()

    events = _events(tmp_path)
    pm = events[-1]["plant_mapping"]
    assert pm["captures_unverified"] == [DATES[1]]


def test_a_moved_read_capture_refuses_naming_the_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(
        dataset_root, dates=[DATES[0]]
    )
    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    stem = f"{PLANTS[0]['plot']}_{DATES[0].replace('-', '')}"
    target = images_root / DATES[0] / f"{stem}.jpg"
    # Several meters east: nowhere near either recorded plant, so nothing matches the distance
    # this mapping's own row recorded for it.
    write_geo_image(
        target,
        PLANTS[0]["lat"],
        PLANTS[0]["lon"] + 0.0002,
        datetime(2026, 2, 11, 9, 30),
        h_pos_err=PHOTO_POSITION_ERROR_M,
    )

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv),
    )
    assert "error" in res
    assert target.name in res["error"]
    assert "assignment would differ" in res["error"]
    assert not out_csv.exists()


def test_full_coverage_still_catches_an_in_place_exif_timestamp_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full coverage still catches an in-place EXIF timestamp change.

    With every mapped capture's prediction present, an in-place EXIF timestamp change at the same
    GPS position refuses the delivery as changed since the mapping was built, and no CSV is
    written.
    """
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(
        dataset_root, dates=[DATES[0]]
    )
    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    stem = f"{PLANTS[0]['plot']}_{DATES[0].replace('-', '')}"
    target = images_root / DATES[0] / f"{stem}.jpg"
    write_geo_image(
        target,
        PLANTS[0]["lat"],
        PLANTS[0]["lon"],
        datetime(2026, 2, 11, 11, 0),
        h_pos_err=PHOTO_POSITION_ERROR_M,
    )

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv),
    )
    assert "error" in res
    assert "changed since this mapping was built" in res["error"]
    assert not out_csv.exists()


def test_an_unmapped_raster_does_not_block_the_whole_date_digest_from_catching_a_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unmapped raster does not block the whole-date digest from catching a change.

    An unmapped raster capture sits beside two fully-read mapped plants, and an in-place EXIF
    timestamp change on a mapped capture still refuses the delivery as changed since the mapping
    was built.
    """
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(
        dataset_root, dates=[DATES[0]]
    )
    (images_root / DATES[0] / "orthomosaic_block.tif").write_bytes(b"")

    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    stem = f"{PLANTS[0]['plot']}_{DATES[0].replace('-', '')}"
    target = images_root / DATES[0] / f"{stem}.jpg"
    write_geo_image(
        target,
        PLANTS[0]["lat"],
        PLANTS[0]["lon"],
        datetime(2026, 2, 11, 11, 0),
        h_pos_err=PHOTO_POSITION_ERROR_M,
    )

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv),
    )
    assert "error" in res
    assert "changed since this mapping was built" in res["error"]
    assert not out_csv.exists()


def test_an_unmapped_raster_beside_a_full_mapped_read_delivers_with_nothing_disclosed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unmapped raster beside a full mapped read delivers with nothing disclosed.

    Nothing changed on disk and both plants' predictions are present, so the delivery succeeds
    and lists no capture as unverified, the raster included.
    """
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(
        dataset_root, dates=[DATES[0]]
    )
    (images_root / DATES[0] / "orthomosaic_block.tif").write_bytes(b"")

    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv),
    )
    assert "error" not in res, res
    assert out_csv.exists()

    events = _events(tmp_path)
    pm = events[-1]["plant_mapping"]
    assert pm["captures_unverified"] == [], pm


def test_a_partial_delivery_delivers_with_disclosures_naming_exactly_what_it_did_not_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(
        dataset_root, unpredicted=UNREAD_P2
    )
    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    p2_stem = f"{PLANTS[1]['plot']}_{DATES[0].replace('-', '')}"

    delivered_preds = {DATES[0]: preds_by_date[DATES[0]]}
    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=delivered_preds.values(),
        output_csv_path=str(out_csv),
    )
    assert "error" not in res, res
    assert out_csv.exists()

    events = _events(tmp_path)
    pm = events[-1]["plant_mapping"]
    assert pm["captures_unverified"] == [f"{DATES[0]}/{p2_stem}.jpg", DATES[1]]


def test_a_capture_readable_at_build_and_unreadable_at_verify_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(
        dataset_root, dates=[DATES[0]]
    )
    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    stem = f"{PLANTS[0]['plot']}_{DATES[0].replace('-', '')}"
    target = images_root / DATES[0] / f"{stem}.jpg"
    target.write_bytes(b"corrupted after the build")

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv),
    )
    assert "error" in res
    assert target.name in res["error"]
    assert not out_csv.exists()


def test_a_capture_unreadable_at_build_is_never_read_so_replacing_it_only_discloses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capture unreadable at build maps to no plant, and replacing it is disclosed, not refused.

    The build records the capture unreadable and assigns it no ``plant_id``. Replaced by its
    readable, geotagged original, with another mapped plant's prediction absent, the delivery
    succeeds and discloses this capture as unverified.
    """
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    # Both dates, not just DATES[0]: DATES[1] stays fully intact so the classifier is still
    # assessed somewhere with DATES[0]'s own P2 prediction absent.
    images_root, plant_csv, preds_by_date = _write_scene(
        dataset_root, unpredicted=UNREAD_P2
    )
    stem = f"{PLANTS[0]['plot']}_{DATES[0].replace('-', '')}"
    target = images_root / DATES[0] / f"{stem}.jpg"
    original_bytes = target.read_bytes()
    target.write_bytes(b"garbage, no EXIF, unreadable at build time")

    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in build_res, build_res
    build = plant_mapping.load_mapping(tmp_path, "valley")
    assert build is not None
    assert f"{stem}.jpg" in build.unreadable[DATES[0]]
    by_stem = {a.stem: a for a in build.assignments[DATES[0]]}
    assert by_stem[stem].plant_id is None
    _seed_currant_bloom_trait(tmp_path)

    target.write_bytes(original_bytes)

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv),
    )
    assert "error" not in res, res
    assert out_csv.exists()

    events = _events(tmp_path)
    pm = events[-1]["plant_mapping"]
    assert f"{DATES[0]}/{stem}.jpg" in pm["captures_unverified"]


# ── rail 8: a receipt that cannot be written fails loudly and the record stays refused ──


def test_a_receipt_that_cannot_be_written_fails_persist_mapping_and_the_record_stays_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``persist_mapping`` raises when its receipt cannot be written, and its record stays refused.

    The receipt cannot be written under a held audit lock, and ``load_mapping`` refuses the record
    ``persist_mapping`` wrote.
    """
    from tcip_mcp.audit import AuditEntryNotWrittenError, audit_log_key
    from tcip_store.sqlite_backend import SqliteBackend

    ts.bind(SqliteBackend(lock_timeout_s=0.2))
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root)
    build = plant_mapping.build_mapping(
        images_root,
        [plant_csv],
        name="valley",
        dataset_root=dataset_root,
        dataset_id="whatever-id",
        project=tmp_path,
        plant_registry=plant_mapping.PlantRegistryReference(
            name="unregistered", digest="0" * 64
        ),
    )

    with (
        held_by_another_writer(audit_log_key(tmp_path)),
        pytest.raises(AuditEntryNotWrittenError, match="plant_mapping_built"),
    ):
        plant_mapping.persist_mapping(build, tmp_path, actor=None)

    with pytest.raises(ValueError, match="receipt"):
        plant_mapping.load_mapping(tmp_path, "valley")


def test_the_web_build_route_answers_409_when_the_receipt_cannot_be_written(
    tmp_path: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_mcp.audit import audit_log_key
    from tcip_store.sqlite_backend import SqliteBackend
    from tcip_web.state import store

    project = _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root)
    registry = register_plant_registry_for(tmp_path, [plant_csv])
    asyncio.run(store.open_project(project))

    ts.bind(SqliteBackend(lock_timeout_s=0.2))

    with held_by_another_writer(audit_log_key(tmp_path)):
        resp = client.post(
            "/api/results/plant_mapping/build",
            json={
                "name": "valley",
                "images_root": str(images_root),
                "plant_registry": registry,
                "user": "tester",
            },
        )
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, dict), detail
    assert detail.get("error") == "audit_entry_not_written"
    assert "plant_mapping_built" in (detail.get("message") or "")
    # A record no receipt names is no mapping a reader loads, so nothing is offered as committed.
    assert detail.get("committed") is None
    with pytest.raises(ValueError, match="no plant_mapping_built receipt"):
        plant_mapping.load_mapping(tmp_path, "valley")


def _cite_mapping(
    tmp_path: Path,
    name: str,
    dataset_root: Path,
    preds_by_date: dict[str, str],
) -> None:
    """Seed the second trait and deliver through the mapping ``name`` with :func:`_deliver`."""
    _seed_currant_bloom_trait(tmp_path)
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name=name,
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(tmp_path / "cited.csv"),
    )
    assert "error" not in res, res


def test_a_supersede_whose_receipt_fails_answers_409_and_the_archive_still_loads(
    tmp_path: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A supersede whose receipt fails answers 409, and the archived record still loads.

    A rebuild that supersedes a cited mapping moves the old record to its archive name, then
    writes the new record and its one receipt. When that receipt is refused the door answers 409
    with ``committed`` null, and the archived record still loads on its own build's receipt.
    """
    from tcip_web.state import store

    project = _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root)
    registry = register_plant_registry_for(tmp_path, [plant_csv])
    asyncio.run(store.open_project(project))

    first = client.post(
        "/api/results/plant_mapping/build",
        json={
            "name": "valley",
            "images_root": str(images_root),
            "plant_registry": registry,
            "user": "tester",
        },
    )
    assert first.status_code == 200, first.text
    old_record = ts.read(plant_mapping.plant_mapping_key(tmp_path, "valley"))

    # A delivery event citing this build, so the rebuild below is the supersede path.
    _cite_mapping(tmp_path, "valley", dataset_root, preds_by_date)

    import tcip_mcp.audit as audit_module

    real_append = audit_module.append
    calls = {"n": 0}

    def _refuse_first(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("audit log unwritable")
        return real_append(*args, **kwargs)

    monkeypatch.setattr(audit_module, "append", _refuse_first)
    resp = client.post(
        "/api/results/plant_mapping/build",
        json={
            "name": "valley",
            "images_root": str(images_root),
            "plant_registry": registry,
            "user": "tester",
            "supersede": True,
        },
    )
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["error"] == "audit_entry_not_written"
    assert detail["committed"] is None

    archived_digest = plant_mapping.record_digest(old_record)
    archived = plant_mapping.load_mapping(
        tmp_path,
        plant_mapping.archived_mapping_name("valley", archived_digest),
    )
    assert archived is not None and archived.record_sha256 == archived_digest


def test_a_delivery_made_from_another_directory_names_the_projects_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delivery made from another directory names the project's dataset.

    With the process working in a directory outside the project, the delivery event records the
    dataset root resolved under the project.
    """
    project = tmp_path / "project"
    _init(project)
    dataset_root = _dataset(project)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root)
    built = build_plant_mapping(
        project,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(project, [plant_csv]),
    )
    assert "error" not in built, built
    _seed_currant_bloom_trait(project)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    delivered = _deliver(
        project,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(project / "out.csv"),
    )

    assert "error" not in delivered, delivered
    (event,) = _events(project)
    assert (
        Path(event["plant_mapping"]["dataset_root"]) == dataset_root.resolve()
    )


# ── rails 9, 10: the full round trip through the platform's own producers ───────────────


def test_full_round_trip_delivers_and_a_rebuild_reads_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root)

    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in build_res, build_res
    assert build_res["summary"]["totals"]["n_dates"] == len(DATES)

    build = plant_mapping.load_mapping(tmp_path, "valley")
    assert build is not None
    assert build.name == "valley"
    assert set(build.dates) == set(DATES)
    for date in DATES:
        assert len(build.assignments[date]) == len(PLANTS)
        assert {a.plant_id for a in build.assignments[date]} == {
            p["plot"] for p in PLANTS
        }

    _seed_currant_bloom_trait(tmp_path)
    out_csv = tmp_path / "out" / "bloom_phenology.csv"
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv),
    )
    assert "error" not in res, res
    assert out_csv.exists()

    events = _events(tmp_path)
    assert len(events) == 1, events
    pm = events[0]["plant_mapping"]
    assert pm["name"] == "valley"
    assert pm["record_sha256"] == build.record_sha256
    assert pm["capture_digests"] == build.capture_digests
    assert set(pm["capture_digests"]) == set(DATES)

    # A rebuild under the same name is cited by the delivery just recorded, so it refuses
    # without supersede; with it, the rebuild reads back and still delivers.
    blocked = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" in blocked
    build_res2 = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
        supersede=True,
    )
    assert "error" not in build_res2, build_res2
    out_csv2 = tmp_path / "out2" / "bloom_phenology.csv"
    res2 = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv2),
    )
    assert "error" not in res2, res2


def test_the_delivery_events_plant_mapping_block_carries_the_match_radius(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A phenology delivery's disclosure carries the mapping's own per-date ``match_radius``.

    A delivered milestone thereby states the radius its identities were matched under.
    """
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root)

    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in build_res, build_res
    build = plant_mapping.load_mapping(tmp_path, "valley")
    assert build is not None

    _seed_currant_bloom_trait(tmp_path)
    out_csv = tmp_path / "out" / "bloom_phenology.csv"
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv),
    )
    assert "error" not in res, res

    events = _events(tmp_path)
    assert len(events) == 1, events
    assert (
        events[0]["plant_mapping"]["match_radius"]
        == build.to_record()["match_radius"]
    )


# ── rail 12: a moved plant CSV and an archived date deliver, disclosed rather than refused


def test_a_moved_plant_csv_and_an_archived_date_deliver_with_disclosures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root)
    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    moved = tmp_path / "plants_moved.csv"
    plant_csv.rename(moved)
    shutil.rmtree(images_root / DATES[0])

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv),
    )
    assert "error" not in res, res
    assert out_csv.exists()

    events = _events(tmp_path)
    pm = events[-1]["plant_mapping"]
    assert str(plant_csv) in pm["plant_csvs_unverified"]
    assert DATES[0] in pm["captures_unverified"]


def test_a_read_capture_whose_plants_own_csv_is_missing_discloses_rather_than_reports_it_moved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read capture whose plant's own CSV is missing is disclosed, not reported moved.

    One plant's verified CSV does not answer for another plant's missing one: a read capture whose
    recorded plant sits in the unreachable CSV did not move, and the delivery succeeds with that
    CSV in plant_csvs_unverified and the capture among the unverified captures.
    """
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, _plant_csv, preds_by_date = _write_scene(
        dataset_root, dates=[DATES[0]]
    )

    per_plant_csvs = [
        write_plant_csv(tmp_path / f"plants_{p['plot']}.csv", [p])
        for p in PLANTS
    ]
    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, per_plant_csvs),
    )
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    per_plant_csvs[1].unlink()

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv),
    )
    assert "error" not in res, res
    assert out_csv.exists()

    events = _events(tmp_path)
    pm = events[-1]["plant_mapping"]
    assert str(per_plant_csvs[1]) in pm["plant_csvs_unverified"]
    p2_stem = f"{PLANTS[1]['plot']}_{DATES[0].replace('-', '')}"
    assert f"{DATES[0]}/{p2_stem}.jpg" in pm["captures_unverified"]


def test_a_moved_capture_whose_own_csv_is_missing_is_disclosed_under_a_partial_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A moved capture whose own CSV is missing is disclosed under a partial read.

    Three plants, one CSV each: one plant's capture moves on disk and that plant's own CSV is
    deleted, with a third plant's prediction absent. The delivery succeeds and discloses the moved
    capture as unverified.
    """
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    plants3 = PLANTS + [
        {
            "plot": "P3",
            "accession": "acc-C",
            "lat": 43.19670,
            "lon": -90.058074,
        }
    ]
    p3_stem = f"{plants3[2]['plot']}_{DATES[0].replace('-', '')}"
    images_root, _plant_csv, preds_by_date = _write_scene(
        dataset_root,
        dates=[DATES[0]],
        plants=plants3,
        unpredicted=frozenset({p3_stem}),
    )

    per_plant_csvs = [
        write_plant_csv(tmp_path / f"plants_{p['plot']}.csv", [p])
        for p in plants3
    ]
    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, per_plant_csvs),
    )
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    p2_stem = f"{plants3[1]['plot']}_{DATES[0].replace('-', '')}"
    moved = images_root / DATES[0] / f"{p2_stem}.jpg"
    write_geo_image(
        moved,
        plants3[1]["lat"],
        plants3[1]["lon"] + 0.0002,
        datetime(2026, 2, 11, 9, 31),
        h_pos_err=PHOTO_POSITION_ERROR_M,
    )
    per_plant_csvs[1].unlink()

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv),
    )
    assert "error" not in res, res
    assert out_csv.exists()

    events = _events(tmp_path)
    pm = events[-1]["plant_mapping"]
    assert f"{DATES[0]}/{p2_stem}.jpg" in pm["captures_unverified"]


# ── rail 15: the build route's happy path, and a moved+re-registered dataset still binds ─


def test_a_moved_and_re_registered_dataset_still_delivers_through_the_earlier_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root)
    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    # Copied, not moved: the copied store carries the buckets, and id preservation only needs
    # dataset.json at the new root.
    moved_root = tmp_path / "ds_moved"
    ts.release_root(dataset_root)
    shutil.copytree(str(dataset_root), str(moved_root))
    reg = register_dataset(
        tmp_path, str(moved_root), crop=sorted(registered_crops())[0]
    )
    assert "error" not in reg, reg
    original = plant_mapping.load_mapping(tmp_path, "valley")
    assert original is not None
    assert reg["id"] == original.dataset_id, (
        "register_dataset must preserve the id across the move"
    )

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=moved_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv),
    )
    assert "error" not in res, res
    assert out_csv.exists()

    # images/ removed under the original root: the check must resolve against the delivered
    # root, not the recorded dataset_root.
    shutil.rmtree(str(images_root))
    out_csv2 = tmp_path / "out2.csv"
    res2 = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=moved_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv2),
    )
    assert "error" not in res2, res2

    events = _events(tmp_path)
    pm = events[-1]["plant_mapping"]
    assert pm["captures_unverified"] == [], pm


# ── rail 13 (listing only): legal names are listed, an illegally-named stray is not ─────


def test_plant_mapping_names_lists_legal_names_and_omits_a_stray_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root)
    res_a = build_plant_mapping(
        tmp_path,
        name="valley-a",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in res_a, res_a
    res_b = build_plant_mapping(
        tmp_path,
        name="valley-b",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in res_b, res_b

    # A record written straight through the store under a key the key producer refuses to
    # build stands in for a stray file under either backend.
    from tcip_mcp.project_paths import project_state_dir

    ts.replace(
        ts.Key(
            plant_mapping.PLANT_MAPPING_STORE,
            str(project_state_dir(tmp_path)),
            ("Not A Legal Name",),
        ),
        {},
    )

    assert plant_mapping.plant_mapping_names(tmp_path) == [
        "valley-a",
        "valley-b",
    ]


def test_two_projects_mapping_one_dataset_under_the_same_name_each_deliver_through_their_own(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two projects mapping one dataset under one name each deliver through their own record.

    Each project builds over its own dates and delivers through its own record, and a delivery
    naming a date only the other project mapped refuses, naming that date.
    """
    proj_a, proj_b = tmp_path / "proj_a", tmp_path / "proj_b"
    for proj in (proj_a, proj_b):
        named_project(proj, proj.name)

    from tests._producer_fixtures import registry_over

    dataset_root = tmp_path / "shared_ds"
    dataset_root.mkdir()
    registry_over(dataset_root, FLOWERS)
    for proj in (proj_a, proj_b):
        reg = register_dataset(
            proj, str(dataset_root), crop=sorted(registered_crops())[0]
        )
        assert "error" not in reg, reg

    images_root, plant_csv, preds_by_date = _write_scene(dataset_root)

    build_a = build_plant_mapping(
        proj_a,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(proj_a, [plant_csv]),
        dates=[DATES[0]],
    )
    assert "error" not in build_a, build_a
    _seed_currant_bloom_trait(proj_a)

    build_b = build_plant_mapping(
        proj_b,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(proj_b, [plant_csv]),
        dates=list(DATES),
    )
    assert "error" not in build_b, build_b
    _seed_currant_bloom_trait(proj_b)

    # Each project's own record: build_a saw one date, build_b saw both.
    assert plant_mapping.load_mapping(proj_a, "valley").dates == [DATES[0]]
    assert set(plant_mapping.load_mapping(proj_b, "valley").dates) == set(
        DATES
    )

    date0_preds = {DATES[0]: preds_by_date[DATES[0]]}
    out_csv_a = proj_a / "out.csv"
    res_a = _deliver(
        proj_a,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=date0_preds.values(),
        output_csv_path=str(out_csv_a),
    )
    assert "error" not in res_a, res_a
    assert out_csv_a.exists()

    out_csv_b = proj_b / "out.csv"
    res_b = _deliver(
        proj_b,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv_b),
    )
    assert "error" not in res_b, res_b
    assert out_csv_b.exists()

    # A date proj_a's own (narrower) mapping does not cover refuses through proj_a's own
    # record, unaffected by proj_b's wider mapping under the identical name.
    out_csv_a2 = proj_a / "out2.csv"
    res_a2 = _deliver(
        proj_a,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv_a2),
    )
    assert "error" in res_a2
    assert DATES[1] in res_a2["error"]

    assert plant_mapping.plant_mapping_names(proj_a) == ["valley"]
    assert plant_mapping.plant_mapping_names(proj_b) == ["valley"]


# ── rail 2: a capture added under a mapped date, through the platform's own writers ─────


def test_an_image_ingested_under_a_mapped_date_refuses_the_delivery_naming_the_date(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_mcp.tools.ingest_tools import ingest_images

    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(
        dataset_root, dates=[DATES[0]]
    )
    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    extra_source = tmp_path / "extra_source"
    write_geo_image(
        extra_source / "P3_extra.jpg",
        43.1968,
        -90.0581,
        datetime(2026, 2, 11, 9, 40),
        h_pos_err=PHOTO_POSITION_ERROR_M,
    )
    res = ingest_images(dataset_root, source=str(extra_source))
    assert "error" not in res, res
    assert [f["bucket"] for f in res["files"]] == [DATES[0]]

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv),
    )
    assert "error" in res
    assert DATES[0] in res["error"]
    assert "P3_extra.jpg" in res["error"]
    assert not out_csv.exists()


def test_a_band_group_written_under_a_mapped_date_refuses_the_delivery_the_same_way(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A band group written under a mapped date refuses the delivery, naming date and manifest.

    No band group existed at build time; ``detect_and_write_band_groups`` forms one from two
    previously-standalone files, so the date's capture set changes shape (two stems collapse into
    one), and no CSV is written.
    """
    from tcip_mcp.pipelines.data.band_groups import (
        detect_and_write_band_groups,
    )

    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(
        dataset_root, dates=[DATES[0]]
    )
    date_dir = images_root / DATES[0]
    write_geo_image(
        date_dir / "aux_b1.jpg",
        43.1968,
        -90.0581,
        datetime(2026, 2, 11, 9, 41),
        h_pos_err=PHOTO_POSITION_ERROR_M,
    )
    write_geo_image(
        date_dir / "aux_b2.jpg",
        43.1968,
        -90.0581,
        datetime(2026, 2, 11, 9, 42),
        h_pos_err=PHOTO_POSITION_ERROR_M,
    )

    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    grouped = detect_and_write_band_groups(
        date_dir,
        explicit_groups={"aux": {"b1": "aux_b1.jpg", "b2": "aux_b2.jpg"}},
    )
    assert grouped["formed"], grouped

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv),
    )
    assert "error" in res
    assert DATES[0] in res["error"]
    assert "aux.bandgroup" in res["error"]
    assert not out_csv.exists()


# ── per-capture digests ─────────────────────────────────────────────────────────────────


def test_build_mapping_persists_and_reads_back_capture_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mapping build persists one capture digest per stem and reads it back.

    A build through the platform's own producer carries one digest per capture, keyed by stem,
    and a recompute over the captures on disk agrees with it entry by entry. The receipt written
    for the record equals the record digest recomputed over it, and a delivery through the
    mapping succeeds.
    """
    from tcip_mcp.pipelines.image_utils import list_logical_images
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        _read_date_stamps,
        capture_digests,
        record_digest,
    )

    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(
        dataset_root, dates=[DATES[0]]
    )
    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    build = plant_mapping.load_mapping(tmp_path, "valley")
    assert build is not None
    date = DATES[0]
    assert set(build.capture_digests) == set(build.dates)
    recorded_stems = {a.stem for a in build.assignments[date]}
    assert set(build.capture_digests[date]) == recorded_stems

    stamps = _read_date_stamps(list_logical_images(images_root / date), date)
    assert capture_digests(stamps) == build.capture_digests[date]

    # Not build.record_sha256: that came from load_mapping's own read of this same raw document,
    # so comparing it back to record_digest(raw) would prove nothing.
    from tcip_mcp.audit import audit_log_key

    raw = ts.read(plant_mapping.plant_mapping_key(tmp_path, "valley"))
    recomputed = record_digest(raw)
    page = ts.read_log(audit_log_key(tmp_path))
    receipts = [
        arguments["record_sha256"]
        for entry in page.records
        if entry.get("tool") == "plant_mapping_built"
        and isinstance(arguments := entry.get("arguments"), dict)
        and arguments.get("name") == "valley"
    ]
    assert receipts, "no plant_mapping_built receipt found for 'valley'"
    assert receipts[-1] == recomputed

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv),
    )
    assert "error" not in res, res
    assert out_csv.exists()


def test_a_band_group_manifest_rewritten_in_place_refuses_the_delivery_naming_the_band_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A band-group manifest rewritten in place refuses the delivery, naming the band group.

    The band group's own membership is unchanged (the same two files, the same stem); the
    delivery refuses naming the date and the band group's manifest, and no CSV is written.
    """
    from tcip_mcp.pipelines.data.band_groups import (
        band_group_manifest_path,
        detect_and_write_band_groups,
    )

    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(
        dataset_root, dates=[DATES[0]]
    )
    date_dir = images_root / DATES[0]
    write_geo_image(
        date_dir / "aux_b1.jpg",
        43.1968,
        -90.0581,
        datetime(2026, 2, 11, 9, 41),
        h_pos_err=PHOTO_POSITION_ERROR_M,
    )
    write_geo_image(
        date_dir / "aux_b2.jpg",
        43.1968,
        -90.0581,
        datetime(2026, 2, 11, 9, 42),
        h_pos_err=PHOTO_POSITION_ERROR_M,
    )
    grouped = detect_and_write_band_groups(
        date_dir,
        explicit_groups={"aux": {"b1": "aux_b1.jpg", "b2": "aux_b2.jpg"}},
    )
    assert grouped["formed"], grouped

    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    manifest_path = band_group_manifest_path(date_dir, "aux")
    # Same "bands" claimed (same members, same kind): a rewrite in place, not an added/removed
    # capture, so only the whole-date identity recompute can catch it.
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv),
    )
    assert "error" in res
    assert DATES[0] in res["error"]
    assert "band group manifest" in res["error"]
    assert "aux.bandgroup" in res["error"]
    assert not out_csv.exists()


# ── rail 11: an unreadable image, a raster and a band group all build and deliver ───────


def test_a_date_with_an_unreadable_image_a_raster_and_a_band_group_builds_and_delivers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_mcp.pipelines.data.band_groups import (
        detect_and_write_band_groups,
    )
    from tcip_mcp.pipelines.image_utils import list_logical_images
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        _read_date_stamps,
    )

    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(
        dataset_root, dates=[DATES[0]]
    )
    date_dir = images_root / DATES[0]

    bad = date_dir / "bad.jpg"
    bad.write_bytes(b"not a real jpeg")
    (date_dir / "raster.npy").write_bytes(
        b"not really a numpy array, list_logical_images never decodes it"
    )
    write_geo_image(
        date_dir / "aux_b1.jpg",
        43.1968,
        -90.0581,
        datetime(2026, 2, 11, 9, 41),
        h_pos_err=PHOTO_POSITION_ERROR_M,
    )
    write_geo_image(
        date_dir / "aux_b2.jpg",
        43.1968,
        -90.0581,
        datetime(2026, 2, 11, 9, 42),
        h_pos_err=PHOTO_POSITION_ERROR_M,
    )
    grouped = detect_and_write_band_groups(
        date_dir,
        explicit_groups={"aux": {"b1": "aux_b1.jpg", "b2": "aux_b2.jpg"}},
    )
    assert grouped["formed"], grouped

    # White-box: the raster and band-group stamps carry readable=None (no EXIF to fail reading),
    # the unreadable image carries readable=False, before build_mapping ever touches a store.
    logical = list_logical_images(date_dir)
    stamps_by_stem = {s.stem: s for s in _read_date_stamps(logical, DATES[0])}
    assert (
        stamps_by_stem["bad"].kind == "photo"
        and stamps_by_stem["bad"].readable is False
    )
    assert (
        stamps_by_stem["raster"].kind == "npy"
        and stamps_by_stem["raster"].readable is None
    )
    assert (
        stamps_by_stem["aux"].kind == "group"
        and stamps_by_stem["aux"].readable is None
    )

    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in build_res, build_res
    assert build_res["unreadable"][DATES[0]] == ["bad.jpg"]
    _seed_currant_bloom_trait(tmp_path)

    build = plant_mapping.load_mapping(tmp_path, "valley")
    assert build is not None
    by_stem = {a.stem: a for a in build.assignments[DATES[0]]}
    assert by_stem["bad"].source == "unmapped"
    assert by_stem["raster"].source == "unmapped"
    assert by_stem["aux"].source == "unmapped"

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=dataset_root,
        buckets=preds_by_date.values(),
        output_csv_path=str(out_csv),
    )
    assert "error" not in res, res
    assert out_csv.exists()


# ── rail 14 (cursor memo): the second receipt scan in one process reads only what was ───
# ── appended since the first, never the whole log again ─────────────────────────────────


def test_second_receipt_scan_in_one_process_reads_only_what_was_appended(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root)

    calls: list[str | None] = []
    real_read_log = ts.read_log

    def spy(key: ts.Key, after: str | None = None) -> ts.LogPage:
        calls.append(after)
        return real_read_log(key, after=after)

    monkeypatch.setattr(ts, "read_log", spy)

    res_a = build_plant_mapping(
        tmp_path,
        name="valley-a",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in res_a, res_a
    build_a = plant_mapping.load_mapping(tmp_path, "valley-a")
    assert build_a is not None
    assert calls == [None]

    res_b = build_plant_mapping(
        tmp_path,
        name="valley-b",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in res_b, res_b
    build_b = plant_mapping.load_mapping(tmp_path, "valley-b")
    assert build_b is not None
    assert len(calls) == 2
    assert calls[1] is not None


def test_two_spellings_of_one_project_share_one_receipt_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two spellings of one project share one receipt scan.

    A load under the project's path and a load under a second spelling of the same directory
    together read the log once, from its start.
    """
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root)
    built = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]),
    )
    assert "error" not in built, built

    calls: list[str | None] = []
    real_read_log = ts.read_log

    def spy(key: ts.Key, after: str | None = None) -> ts.LogPage:
        calls.append(after)
        return real_read_log(key, after=after)

    monkeypatch.setattr(ts, "read_log", spy)
    other = Path(f"{tmp_path}/../{tmp_path.name}")
    assert str(other) != str(tmp_path) and ts.canonical_path(
        other
    ) == ts.canonical_path(tmp_path)

    assert plant_mapping.load_mapping(tmp_path, "valley") is not None
    assert plant_mapping.load_mapping(other, "valley") is not None
    assert calls == [None]


def test_a_build_leaves_one_row_its_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mapping build leaves one audit row, its receipt.

    The call adds one row to the log, the library's ``plant_mapping_built`` receipt naming the
    mapping.
    """
    from tcip_mcp.audit import audit_log_key

    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root)
    registry = register_plant_registry_for(tmp_path, [plant_csv])
    before = len(ts.read_log(audit_log_key(tmp_path)).records)

    res = build_plant_mapping(
        tmp_path,
        name="valley",
        images_root=str(images_root),
        plant_registry=registry,
    )

    assert "error" not in res, res
    rows = ts.read_log(audit_log_key(tmp_path)).records[before:]
    assert [row["tool"] for row in rows] == ["plant_mapping_built"], rows
    arguments = rows[0]["arguments"]
    assert isinstance(arguments, dict) and arguments["name"] == "valley"
