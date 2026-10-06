"""Rails for the plant-mapping binding family: the record is bound to its inputs, to a
registered dataset by minted id, and to the receipt that proves it was written by the
platform's own producers, never a hand-filed file. Every scenario is built through the
platform's own producers (initialize_project, register_dataset, build_plant_mapping, publish and
the phenology measurement and delivery), a second registered trait rather than the pilot's, so
nothing here generalizes from one trait's own vocabulary.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from datetime import datetime, timedelta
from collections.abc import Iterable
from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp.pipelines.postprocessing import plant_mapping
from tcip_mcp.tools.phenology_tools import build_plant_mapping
from tcip_mcp.tools.project_tools import initialize_project, register_dataset
from tcip_mcp.traits import registered_crops

from tests._audit_fixtures import held_by_another_writer
from tests._image_fixtures import write_geo_image
from tests._mapping_fixtures import register_plant_registry_for, write_plant_csv
from tests.test_second_trait_acceptance import BLOOM_STATE, FLOWERS, _seed_currant_bloom_trait

PLANTS = [
    {"plot": "P1", "accession": "acc-A", "lat": 43.19670, "lon": -90.058000},
    {"plot": "P2", "accession": "acc-B", "lat": 43.19670, "lon": -90.058037},
]
DATES = ["2026-02-11", "2026-02-25"]
UNREAD_P2 = frozenset({f"{PLANTS[1]['plot']}_{DATES[0].replace('-', '')}"})
"""The second plant's first-date capture, left without a prediction document."""

POPULATION = [p["plot"] for p in PLANTS]
"""The plants every delivery here states, the ones the scene's registry places."""


def _init(tmp_path: Path) -> None:
    result = initialize_project(str(tmp_path), "Orchard", site="orchard block")
    assert "error" not in result, result


def _dataset(tmp_path: Path, name: str = "ds") -> Path:
    """A dataset root registered in ``tmp_path`` whose own subject registry declares the second
    trait's positive state, the registry a phenology delivery binds against."""
    from tests._producer_fixtures import registry_over

    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    result = register_dataset(tmp_path, str(root), crop=sorted(registered_crops())[0])
    assert "error" not in result, result
    registry_over(root, FLOWERS)
    return root


def _publish(project: Path, bucket: str, images: list[Path]) -> str:
    """One bucket named ``bucket`` holding an open flower on each of ``images``, published under
    ``project`` (``_chain_fixtures.published``) from a checkpoint whose scope declares the bloom
    state, unassessed: these rails are about the mapping's own binding, not the assessment
    gate; its name."""
    from tests._chain_fixtures import published

    results = [{"image": str(p), "width": 8, "height": 8, "boxes": [[1.0, 1.0, 3.0, 3.0]],
                "scores": [0.9], "labels": [1], "attributes": [[BLOOM_STATE.values.index("open")]]}
               for p in images]
    return published(project, bucket, results, scope={"subject": "flower"},
                     registry=FLOWERS).name


def _deliver(project: Path, *, trait: str, mapping_name: str, plants: list[str],
             dataset_root: Path, buckets: Iterable[str], output_csv_path: str) -> dict:
    """``deliver_phenology_milestones``'s measurement and delivery through the library, shipped
    under a breeder's acknowledgment (these rails are the mapping's, never the assessment
    gate's); a refusal answers ``{"error": ...}`` the way the tool's does."""
    from tcip_mcp.pipelines.postprocessing import phenology
    from tcip_mcp.pipelines.postprocessing.plant_mapping import MappingDeliveryRefusal
    from tests._chain_fixtures import acknowledged

    try:
        measurement = phenology.measure_phenology(
            project, trait=trait, mapping_name=mapping_name, dataset_root=dataset_root,
            buckets=list(buckets), plants=plants,
            require_all_dates_complete=phenology.REQUIRE_ALL_DATES_COMPLETE)
        return acknowledged(project, lambda ack: phenology.deliver_phenology(
            project, measurement, curves=False, output_path=Path(output_csv_path),
            acknowledgment_id=ack, door=DOOR, actor=None),
            reason="mapping rails, not the assessment")
    except (ValueError, MappingDeliveryRefusal, *phenology.measurement_refusals()) as exc:
        return {"error": str(exc)}


DOOR = "test_mapping_delivery"


def _events(project: Path) -> list[dict]:
    """The delivery events :func:`_deliver` recorded under ``project``, each as its JSON form."""
    from tcip_mcp.delivery import read_delivery_events

    return [e.model_dump(mode="json") for e in read_delivery_events(project) if e.door == DOOR]


def _write_scene(
    dataset_root: Path, *, dates: list[str] = DATES, plants: list[dict] | None = None,
    unpredicted: frozenset[str] = frozenset(),
) -> tuple[Path, Path, dict[str, str]]:
    """Real geolocated images for ``plants`` (``PLANTS`` by default) across ``dates``, plus one
    published bucket per date holding a document for every image but the stems
    ``unpredicted`` names, under the project the dataset sits in. Returns (images_root,
    plant_csv, preds_by_date).
    """
    pytest.importorskip("torch")
    plants = PLANTS if plants is None else plants
    images_root = dataset_root / "images"
    preds_by_date: dict[str, str] = {}
    for date in dates:
        base_time = datetime.strptime(date, "%Y-%m-%d").replace(hour=9, minute=30)
        images = []
        for j, plant in enumerate(plants):
            stem = f"{plant['plot']}_{date.replace('-', '')}"
            write_geo_image(
                images_root / date / f"{stem}.jpg", plant["lat"], plant["lon"],
                base_time + timedelta(minutes=j))
            if stem not in unpredicted:
                images.append(images_root / date / f"{stem}.jpg")
        preds_by_date[date] = _publish(dataset_root.parent, f"live/{date}", images)

    plant_csv = write_plant_csv(dataset_root.parent / f"{dataset_root.name}_plants.csv", plants)
    return images_root, plant_csv, preds_by_date


# ── rail 6: no project record ────────────────────────────────────────────


# ── rail 4: dataset identity, NAME_SEGMENT, variously-spelled roots, dataset mismatch ────


def test_build_plant_mapping_refuses_a_name_outside_name_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    res = build_plant_mapping(
        tmp_path, name="Not Legal!", images_root=str(tmp_path), plant_registry="unregistered")
    assert "error" in res
    assert "lowercase" in res["error"]


def test_build_plant_mapping_over_an_unregistered_images_dir_names_register_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    images_root = tmp_path / "ds" / "images"
    (images_root / DATES[0]).mkdir(parents=True)
    res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root), plant_registry="unregistered")
    assert "error" in res
    assert "register_dataset" in res["error"]
    assert not (tmp_path / ".tcip" / "state" / "plant_mappings").exists()


def test_build_plant_mapping_admits_the_dataset_images_root_spelled_variously(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root)

    trailing = str(images_root) + os.sep
    res = build_plant_mapping(
        tmp_path, name="trailing-sep", images_root=trailing,
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in res, res

    forward = str(images_root).replace("\\", "/")
    res = build_plant_mapping(
        tmp_path, name="forward-slash", images_root=forward,
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in res, res

    monkeypatch.chdir(dataset_root)
    res = build_plant_mapping(
        tmp_path, name="relative", images_root="images",
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in res, res


def test_deliver_phenology_milestones_refuses_predictions_from_a_different_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root)
    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    other_root = _dataset(tmp_path, name="ds2")
    _, _, other_preds = _write_scene(other_root)

    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=other_root, buckets=other_preds.values(),
        output_csv_path=str(tmp_path / "out.csv"))
    assert "error" in res
    assert "different dataset" in res["error"]


# ── rail 5: a bucket dated on a day the mapping does not name ──────────────────────────


def test_deliver_phenology_milestones_refuses_a_date_the_mapping_does_not_cover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATES[0]])
    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    extra_date = "2026-03-01"
    extra_image = images_root / extra_date / "P1_20260301.jpg"
    write_geo_image(extra_image, PLANTS[0]["lat"], PLANTS[0]["lon"], datetime(2026, 3, 1, 9, 30))
    preds_by_date[extra_date] = _publish(tmp_path, f"live/{extra_date}", [extra_image])

    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(tmp_path / "out.csv"))
    assert "error" in res
    assert extra_date in res["error"]


# ── rail 1: a hand-written record refuses, missing provenance or missing receipt ────────


def test_deliver_phenology_milestones_refuses_a_hand_written_record_missing_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    _, _, preds_by_date = _write_scene(dataset_root)
    _seed_currant_bloom_trait(tmp_path)

    ts.replace(plant_mapping.plant_mapping_key(tmp_path, "forged"), {
        "assignments": {d: [] for d in DATES},
    })
    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="forged", plants=POPULATION,
        dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv))
    assert "error" in res
    assert "is not a record this reader decodes" in res["error"]
    assert not out_csv.exists()


def test_deliver_phenology_milestones_refuses_a_record_with_provenance_and_no_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    _, _, preds_by_date = _write_scene(dataset_root)
    _seed_currant_bloom_trait(tmp_path)

    record = {
        "name": "forged", "dataset_root": "ds",
        "dataset_id": "whatever-id",
        "built_at": "2026-02-11T00:00:00+00:00", "dates_requested": None, "dates": list(DATES),
        "nn_tolerance_m": {"value": 10.0, "source": "stated"},
        "plant_registry": {"name": "unregistered", "digest": "0" * 64},
        "capture_identity": {d: "0" * 16 for d in DATES},
        "capture_digests": {d: {} for d in DATES}, "unreadable": {d: [] for d in DATES},
        "assignments": {d: [] for d in DATES}, "supersedes": None,
    }
    ts.replace(plant_mapping.plant_mapping_key(tmp_path, "forged"), record)
    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="forged", plants=POPULATION,
        dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv))
    assert "error" in res
    assert "receipt" in res["error"]
    assert not out_csv.exists()


# ── rail 3: a plant CSV rewritten in place refuses, naming the file ─────────────────────


def test_deliver_phenology_milestones_refuses_a_plant_csv_rewritten_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root)
    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    write_plant_csv(plant_csv, [{**PLANTS[0], "accession": "acc-Z"}])

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv))
    assert "error" in res
    assert str(plant_csv) in res["error"]
    assert not out_csv.exists()


# ── rail 7: a readability flip on a capture this delivery reads refuses, naming the file ─


def test_an_unread_captures_bytes_going_bad_is_disclosed_never_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recorded, mapped capture this delivery's own buckets carry no
    document for is unread: corrupting its bytes afterward must surface only as this capture's
    own disclosure, never a readability refusal, since an unread capture is never opened."""
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(
        dataset_root, dates=[DATES[0]], unpredicted=UNREAD_P2)
    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    p2_stem = f"{PLANTS[1]['plot']}_{DATES[0].replace('-', '')}"
    (images_root / DATES[0] / f"{p2_stem}.jpg").write_bytes(b"not a real jpeg any more")

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv))
    assert "error" not in res, res
    assert out_csv.exists()

    events = _events(tmp_path)
    pm = events[-1]["plant_mapping"]
    assert pm["captures_unverified"] == [f"{DATES[0]}/{p2_stem}.jpg"]


def test_an_unread_captures_bytes_changing_in_place_does_not_refuse_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same unread capture as above, but changed to a different, still-readable image rather
    than garbage: the whole-date identity this would have flipped is never recomputed either,
    since this delivery does not read every capture of the date."""
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(
        dataset_root, dates=[DATES[0]], unpredicted=UNREAD_P2)
    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    p2_stem = f"{PLANTS[1]['plot']}_{DATES[0].replace('-', '')}"
    write_geo_image(
        images_root / DATES[0] / f"{p2_stem}.jpg", PLANTS[1]["lat"], PLANTS[1]["lon"],
        datetime(2026, 2, 11, 10, 45))

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv))
    assert "error" not in res, res
    assert out_csv.exists()

    events = _events(tmp_path)
    pm = events[-1]["plant_mapping"]
    assert pm["captures_unverified"] == [f"{DATES[0]}/{p2_stem}.jpg"]


def test_a_non_delivered_mapping_date_is_never_walked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mapped date this delivery's own buckets omit is disclosed as a bare
    date and never enumerated: a capture corrupted under it changes nothing about the delivery."""
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root)
    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    stem = f"{PLANTS[0]['plot']}_{DATES[1].replace('-', '')}"
    (images_root / DATES[1] / f"{stem}.jpg").write_bytes(b"garbage, never read by this delivery")

    delivered_preds = {DATES[0]: preds_by_date[DATES[0]]}
    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=dataset_root, buckets=delivered_preds.values(),
        output_csv_path=str(out_csv))
    assert "error" not in res, res
    assert out_csv.exists()

    events = _events(tmp_path)
    pm = events[-1]["plant_mapping"]
    assert pm["captures_unverified"] == [DATES[1]]


def test_a_moved_read_capture_refuses_naming_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATES[0]])
    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    stem = f"{PLANTS[0]['plot']}_{DATES[0].replace('-', '')}"
    target = images_root / DATES[0] / f"{stem}.jpg"
    # Several meters east: nowhere near either recorded plant, so nothing matches the distance
    # this mapping's own row recorded for it.
    write_geo_image(
        target, PLANTS[0]["lat"], PLANTS[0]["lon"] + 0.0002, datetime(2026, 2, 11, 9, 30))

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv))
    assert "error" in res
    assert target.name in res["error"]
    assert "assignment would differ" in res["error"]
    assert not out_csv.exists()


def test_full_coverage_still_catches_an_in_place_exif_timestamp_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With every mapped capture's prediction present, an in-place EXIF timestamp change at the
    same GPS position is caught by the whole-date identity recompute, not the moved-position
    check, since the position itself never moved."""
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATES[0]])
    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    stem = f"{PLANTS[0]['plot']}_{DATES[0].replace('-', '')}"
    target = images_root / DATES[0] / f"{stem}.jpg"
    write_geo_image(target, PLANTS[0]["lat"], PLANTS[0]["lon"], datetime(2026, 2, 11, 11, 0))

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv))
    assert "error" in res
    assert "changed since this mapping was built" in res["error"]
    assert not out_csv.exists()


def test_an_unmapped_raster_does_not_block_the_whole_date_digest_from_catching_a_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unmapped raster capture beside two fully-read mapped plants does not block the
    whole-date recompute: the trigger is every mapped capture read, so an in-place EXIF
    timestamp change on a mapped capture still refuses even with the raster never read."""
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATES[0]])
    (images_root / DATES[0] / "orthomosaic_block.tif").write_bytes(b"")

    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    stem = f"{PLANTS[0]['plot']}_{DATES[0].replace('-', '')}"
    target = images_root / DATES[0] / f"{stem}.jpg"
    write_geo_image(target, PLANTS[0]["lat"], PLANTS[0]["lon"], datetime(2026, 2, 11, 11, 0))

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv))
    assert "error" in res
    assert "changed since this mapping was built" in res["error"]
    assert not out_csv.exists()


def test_an_unmapped_raster_beside_a_full_mapped_read_delivers_with_nothing_disclosed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The admitting side of the same scene: nothing changed on disk, both plants' predictions are
    present, and the raster is verified only through the whole-date digest, so it is not also
    listed among the unverified captures."""
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATES[0]])
    (images_root / DATES[0] / "orthomosaic_block.tif").write_bytes(b"")

    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv))
    assert "error" not in res, res
    assert out_csv.exists()

    events = _events(tmp_path)
    pm = events[-1]["plant_mapping"]
    assert pm["captures_unverified"] == [], pm


def test_a_partial_delivery_delivers_with_disclosures_naming_exactly_what_it_did_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, unpredicted=UNREAD_P2)
    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    p2_stem = f"{PLANTS[1]['plot']}_{DATES[0].replace('-', '')}"

    delivered_preds = {DATES[0]: preds_by_date[DATES[0]]}
    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=dataset_root, buckets=delivered_preds.values(),
        output_csv_path=str(out_csv))
    assert "error" not in res, res
    assert out_csv.exists()

    events = _events(tmp_path)
    pm = events[-1]["plant_mapping"]
    assert pm["captures_unverified"] == [f"{DATES[0]}/{p2_stem}.jpg", DATES[1]]


def test_a_capture_readable_at_build_and_unreadable_at_verify_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATES[0]])
    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    stem = f"{PLANTS[0]['plot']}_{DATES[0].replace('-', '')}"
    target = images_root / DATES[0] / f"{stem}.jpg"
    target.write_bytes(b"corrupted after the build")

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv))
    assert "error" in res
    assert target.name in res["error"]
    assert not out_csv.exists()


def test_a_capture_unreadable_at_build_is_never_read_so_replacing_it_only_discloses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capture unreadable when the mapping was built has no GPS, so ``assign_plants`` records it
    unmapped, and an unmapped row's stem is never among what a delivery reads
    (``stems_delivery_reads`` requires a truthy ``plot_name``): replacing it with a readable,
    geotagged image later is never examined for a readability flip at all, coverage of that dead
    path rather than a guard, and under a partial read (another mapped plant's prediction absent)
    it surfaces only as this capture's own disclosure."""
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    # Both dates, not just DATES[0]: DATES[1] stays fully intact so the classifier is still
    # assessed somewhere with DATES[0]'s own P2 prediction absent.
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, unpredicted=UNREAD_P2)
    stem = f"{PLANTS[0]['plot']}_{DATES[0].replace('-', '')}"
    target = images_root / DATES[0] / f"{stem}.jpg"
    original_bytes = target.read_bytes()
    target.write_bytes(b"garbage, no EXIF, unreadable at build time")

    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in build_res, build_res
    build = plant_mapping.load_mapping(tmp_path, "valley")
    assert build is not None
    assert f"{stem}.jpg" in build.unreadable[DATES[0]]
    by_stem = {a.stem: a for a in build.assignments[DATES[0]]}
    assert by_stem[stem].plot_name is None
    _seed_currant_bloom_trait(tmp_path)

    target.write_bytes(original_bytes)

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv))
    assert "error" not in res, res
    assert out_csv.exists()

    events = _events(tmp_path)
    pm = events[-1]["plant_mapping"]
    assert f"{DATES[0]}/{stem}.jpg" in pm["captures_unverified"]


# ── rail 8: a receipt that cannot be written fails loudly and the record stays refused ──


def test_a_receipt_that_cannot_be_written_fails_persist_mapping_and_the_record_stays_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At the pipeline level: ``persist_mapping``'s own contract, not the MCP tool's ``@audited``
    wrapper (which writes its own, separate audit line for the call and would otherwise also
    contend for the same lock this test holds for the whole body). This calls
    ``plant_mapping.build_mapping`` directly rather than through ``build_plant_mapping`` so the
    tool's wrapper does not contend for that lock."""
    from tcip_mcp.audit import AuditEntryNotWritten, audit_log_key
    from tcip_store.sqlite_backend import SqliteBackend

    ts.bind(SqliteBackend(lock_timeout_s=0.2))
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root)
    build = plant_mapping.build_mapping(
        images_root, [plant_csv], name="valley", dataset_root=dataset_root,
        dataset_id="whatever-id", project=tmp_path,
        plant_registry={"name": "unregistered", "digest": "0" * 64})

    with held_by_another_writer(audit_log_key(tmp_path)), pytest.raises(
            AuditEntryNotWritten, match="plant_mapping_built"):
        plant_mapping.persist_mapping(build, tmp_path, actor=None)

    with pytest.raises(ValueError, match="receipt"):
        plant_mapping.load_mapping(tmp_path, "valley")


def test_the_web_build_route_answers_409_when_the_receipt_cannot_be_written(
    tmp_path: Path, client, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_mcp.audit import audit_log_key
    from tcip_store.sqlite_backend import SqliteBackend
    from tcip_web.state import store

    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root)
    registry = register_plant_registry_for(tmp_path, [plant_csv])
    asyncio.run(store.open_project(tmp_path.resolve()))

    ts.bind(SqliteBackend(lock_timeout_s=0.2))

    with held_by_another_writer(audit_log_key(tmp_path)):
        resp = client.post("/api/results/plant_mapping/build", json={
            "name": "valley", "images_root": str(images_root), "plant_registry": registry,
            "user": "tester",
        })
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, dict), detail
    assert detail.get("error") == "audit_entry_not_written"
    assert "plant_mapping_built" in (detail.get("message") or "")
    # A record no receipt names is no mapping a reader loads, so nothing is offered as committed.
    assert detail.get("committed") is None
    with pytest.raises(ValueError, match="no plant_mapping_built receipt"):
        plant_mapping.load_mapping(tmp_path, "valley")


def _cite_mapping(tmp_path: Path, name: str, dataset_root: Path,
                  preds_by_date: dict[str, str]) -> None:
    """A real delivery through the mapping under ``name`` (:func:`_deliver`), whose event cites
    it."""
    _seed_currant_bloom_trait(tmp_path)
    res = _deliver(tmp_path, trait="currant_bloom", mapping_name=name,
                   plants=POPULATION, dataset_root=dataset_root, buckets=preds_by_date.values(),
                   output_csv_path=str(tmp_path / "cited.csv"))
    assert "error" not in res, res


def test_a_supersede_whose_receipt_fails_answers_409_and_the_archive_still_loads(
    tmp_path: Path, client, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rebuild that supersedes a cited mapping moves the old record to its archive name, then
    writes the new record and its one receipt. When that receipt is refused the door answers 409
    with ``committed`` null, and the archived record still loads on its own build's receipt."""
    from tcip_web.state import store

    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root)
    registry = register_plant_registry_for(tmp_path, [plant_csv])
    asyncio.run(store.open_project(tmp_path.resolve()))

    first = client.post("/api/results/plant_mapping/build", json={
        "name": "valley", "images_root": str(images_root), "plant_registry": registry,
        "user": "tester",
    })
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
    resp = client.post("/api/results/plant_mapping/build", json={
        "name": "valley", "images_root": str(images_root), "plant_registry": registry,
        "user": "tester",
        "supersede": True,
    })
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["error"] == "audit_entry_not_written"
    assert detail["committed"] is None

    archived_digest = plant_mapping.record_digest(old_record)
    archived = plant_mapping.load_mapping(
        tmp_path, plant_mapping.archived_mapping_name("valley", archived_digest))
    assert archived is not None and archived.record_sha256 == archived_digest


# ── rails 9, 10: the full round trip through the platform's own producers ───────────────


def test_full_round_trip_delivers_and_a_rebuild_reads_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root)

    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in build_res, build_res
    assert build_res["summary"]["totals"]["n_dates"] == len(DATES)

    build = plant_mapping.load_mapping(tmp_path, "valley")
    assert build is not None
    assert build.name == "valley"
    assert set(build.dates) == set(DATES)
    for date in DATES:
        assert len(build.assignments[date]) == len(PLANTS)
        assert {a.plot_name for a in build.assignments[date]} == {p["plot"] for p in PLANTS}

    _seed_currant_bloom_trait(tmp_path)
    out_csv = tmp_path / "out" / "bloom_phenology.csv"
    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv))
    assert "error" not in res, res
    assert out_csv.exists()

    events = _events(tmp_path)
    assert len(events) == 1, events
    pm = events[0]["plant_mapping"]
    assert pm["name"] == "valley"
    assert pm["record_sha256"] == build.record_sha256
    assert set(pm["capture_identity"].keys()) == set(DATES)

    # A rebuild under the same name is cited by the delivery just recorded, so it refuses
    # without supersede; with it, the rebuild reads back and still delivers.
    blocked = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" in blocked
    build_res2 = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]), supersede=True)
    assert "error" not in build_res2, build_res2
    out_csv2 = tmp_path / "out2" / "bloom_phenology.csv"
    res2 = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv2))
    assert "error" not in res2, res2


def test_the_delivery_events_plant_mapping_block_carries_the_tolerance_dict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coverage: the disclosure a phenology delivery records carries the mapping's own
    ``nn_tolerance_m``, so a delivered milestone states the radius its identities were matched
    under."""
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root)

    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in build_res, build_res
    build = plant_mapping.load_mapping(tmp_path, "valley")
    assert build is not None

    _seed_currant_bloom_trait(tmp_path)
    out_csv = tmp_path / "out" / "bloom_phenology.csv"
    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv))
    assert "error" not in res, res

    events = _events(tmp_path)
    assert len(events) == 1, events
    assert events[0]["plant_mapping"]["nn_tolerance_m"] == build.nn_tolerance_m


# ── rail 12: a moved plant CSV and an archived date deliver, disclosed rather than refused


def test_a_moved_plant_csv_and_an_archived_date_deliver_with_disclosures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root)
    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    moved = tmp_path / "plants_moved.csv"
    plant_csv.rename(moved)
    shutil.rmtree(images_root / DATES[0])

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv))
    assert "error" not in res, res
    assert out_csv.exists()

    events = _events(tmp_path)
    pm = events[-1]["plant_mapping"]
    assert str(plant_csv) in pm["plant_csvs_unverified"]
    assert DATES[0] in pm["captures_unverified"]


def test_a_read_capture_whose_plants_own_csv_is_missing_discloses_rather_than_reports_it_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One plant's verified CSV must not answer for another plant's missing one: a read capture
    whose recorded plant sits in the unreachable CSV did not move, and the delivery proceeds with
    that CSV in plant_csvs_unverified rather than refusing as if the capture had moved."""
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, _plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATES[0]])

    per_plant_csvs = [write_plant_csv(tmp_path / f"plants_{p['plot']}.csv", [p]) for p in PLANTS]
    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, per_plant_csvs))
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    per_plant_csvs[1].unlink()

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv))
    assert "error" not in res, res
    assert out_csv.exists()

    events = _events(tmp_path)
    pm = events[-1]["plant_mapping"]
    assert str(per_plant_csvs[1]) in pm["plant_csvs_unverified"]
    p2_stem = f"{PLANTS[1]['plot']}_{DATES[0].replace('-', '')}"
    assert f"{DATES[0]}/{p2_stem}.jpg" in pm["captures_unverified"]


def test_a_moved_capture_whose_own_csv_is_missing_is_disclosed_under_a_partial_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three plants, one CSV each: one plant's capture moves on disk and that plant's own CSV is
    deleted, while a partial read (a third plant's prediction absent) means the delivery cannot
    fall back on the whole-date digest either. The moved capture is disclosed as unverified rather
    than refused as moved, since its own plant's CSV cannot answer for whether it moved."""
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    plants3 = PLANTS + [{"plot": "P3", "accession": "acc-C", "lat": 43.19670, "lon": -90.058074}]
    p3_stem = f"{plants3[2]['plot']}_{DATES[0].replace('-', '')}"
    images_root, _plant_csv, preds_by_date = _write_scene(
        dataset_root, dates=[DATES[0]], plants=plants3, unpredicted=frozenset({p3_stem}))

    per_plant_csvs = [write_plant_csv(tmp_path / f"plants_{p['plot']}.csv", [p]) for p in plants3]
    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, per_plant_csvs))
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    p2_stem = f"{plants3[1]['plot']}_{DATES[0].replace('-', '')}"
    moved = images_root / DATES[0] / f"{p2_stem}.jpg"
    write_geo_image(
        moved, plants3[1]["lat"], plants3[1]["lon"] + 0.0002, datetime(2026, 2, 11, 9, 31))
    per_plant_csvs[1].unlink()

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv))
    assert "error" not in res, res
    assert out_csv.exists()

    events = _events(tmp_path)
    pm = events[-1]["plant_mapping"]
    assert f"{DATES[0]}/{p2_stem}.jpg" in pm["captures_unverified"]


# ── rail 15: the build route's happy path, and a moved+re-registered dataset still binds ─


def test_a_moved_and_re_registered_dataset_still_delivers_through_the_earlier_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root)
    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    # Copied, not moved: the copied store carries the buckets, and id preservation only needs
    # dataset.json at the new root.
    moved_root = tmp_path / "ds_moved"
    ts.release_root(dataset_root)
    shutil.copytree(str(dataset_root), str(moved_root))
    reg = register_dataset(tmp_path, str(moved_root), crop=sorted(registered_crops())[0])
    assert "error" not in reg, reg
    original = plant_mapping.load_mapping(tmp_path, "valley")
    assert original is not None
    assert reg["id"] == original.dataset_id, "register_dataset must preserve the id across the move"

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=moved_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv))
    assert "error" not in res, res
    assert out_csv.exists()

    # images/ removed under the original root: the check must resolve against the delivered
    # root, not the recorded dataset_root.
    shutil.rmtree(str(images_root))
    out_csv2 = tmp_path / "out2.csv"
    res2 = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=moved_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv2))
    assert "error" not in res2, res2

    events = _events(tmp_path)
    pm = events[-1]["plant_mapping"]
    assert pm["captures_unverified"] == [], pm


# ── rail 13 (listing only): legal names are listed, an illegally-named stray is not ─────


def test_plant_mapping_names_lists_legal_names_and_omits_a_stray_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root)
    res_a = build_plant_mapping(
        tmp_path, name="valley-a", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in res_a, res_a
    res_b = build_plant_mapping(
        tmp_path, name="valley-b", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in res_b, res_b

    # A record written straight through the store under a key the key producer refuses to
    # build stands in for a stray file under either backend.
    from tcip_mcp.project_paths import project_state_dir

    ts.replace(ts.Key(plant_mapping.PLANT_MAPPING_STORE, str(project_state_dir(tmp_path)),
                      ("Not A Legal Name",)), {})

    assert plant_mapping.plant_mapping_names(tmp_path) == ["valley-a", "valley-b"]


def test_two_projects_mapping_one_dataset_under_the_same_name_each_deliver_through_their_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rail 13's delivery half: two projects mapping one registered dataset under the same
    mapping name each build and deliver through their own record, never the other's."""
    proj_a, proj_b = tmp_path / "proj_a", tmp_path / "proj_b"
    for proj in (proj_a, proj_b):
        res = initialize_project(str(proj), proj.name, site=f"orchard {proj.name}")
        assert "error" not in res, res

    from tests._producer_fixtures import registry_over

    dataset_root = tmp_path / "shared_ds"
    dataset_root.mkdir()
    registry_over(dataset_root, FLOWERS)
    for proj in (proj_a, proj_b):
        reg = register_dataset(proj, str(dataset_root), crop=sorted(registered_crops())[0])
        assert "error" not in reg, reg

    images_root, plant_csv, preds_by_date = _write_scene(dataset_root)

    build_a = build_plant_mapping(
        proj_a, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(proj_a, [plant_csv]), dates=[DATES[0]])
    assert "error" not in build_a, build_a
    _seed_currant_bloom_trait(proj_a)

    build_b = build_plant_mapping(
        proj_b, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(proj_b, [plant_csv]), dates=list(DATES))
    assert "error" not in build_b, build_b
    _seed_currant_bloom_trait(proj_b)

    # Each project's own record: build_a saw one date, build_b saw both.
    assert plant_mapping.load_mapping(proj_a, "valley").dates == [DATES[0]]
    assert set(plant_mapping.load_mapping(proj_b, "valley").dates) == set(DATES)

    date0_preds = {DATES[0]: preds_by_date[DATES[0]]}
    out_csv_a = proj_a / "out.csv"
    res_a = _deliver(
        proj_a, trait="currant_bloom", mapping_name="valley",
        plants=POPULATION, dataset_root=dataset_root, buckets=date0_preds.values(),
        output_csv_path=str(out_csv_a))
    assert "error" not in res_a, res_a
    assert out_csv_a.exists()

    out_csv_b = proj_b / "out.csv"
    res_b = _deliver(
        proj_b, trait="currant_bloom", mapping_name="valley",
        plants=POPULATION, dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv_b))
    assert "error" not in res_b, res_b
    assert out_csv_b.exists()

    # A date proj_a's own (narrower) mapping does not cover refuses through proj_a's own
    # record, unaffected by proj_b's wider mapping under the identical name.
    out_csv_a2 = proj_a / "out2.csv"
    res_a2 = _deliver(
        proj_a, trait="currant_bloom", mapping_name="valley",
        plants=POPULATION, dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv_a2))
    assert "error" in res_a2
    assert DATES[1] in res_a2["error"]

    assert plant_mapping.plant_mapping_names(proj_a) == ["valley"]
    assert plant_mapping.plant_mapping_names(proj_b) == ["valley"]


# ── rail 2: a capture added under a mapped date, through the platform's own writers ─────


def test_an_image_ingested_under_a_mapped_date_refuses_the_delivery_naming_the_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_mcp.tools.ingest_tools import ingest_images

    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATES[0]])
    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    extra_source = tmp_path / "extra_source"
    write_geo_image(extra_source / "P3_extra.jpg", 43.1968, -90.0581, datetime(2026, 2, 11, 9, 40))
    res = ingest_images(dataset_root, source=str(extra_source), date_from=DATES[0])
    assert "error" not in res, res
    assert res["buckets"].get(DATES[0]) == 1

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv))
    assert "error" in res
    assert DATES[0] in res["error"]
    assert "P3_extra.jpg" in res["error"]
    assert not out_csv.exists()


def test_a_band_group_written_under_a_mapped_date_refuses_the_delivery_the_same_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No band group existed at build time; ``detect_and_write_band_groups`` forms one from two
    previously-standalone files, so the date's capture set changes shape (two stems collapse
    into one) the same way an added image does: refused, naming the date and the manifest's own
    file name, through the added-capture check, since the mapping's own assignment rows name
    the two original standalone stems, never the grouped one.
    """
    from tcip_mcp.pipelines.data.band_groups import detect_and_write_band_groups

    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATES[0]])
    date_dir = images_root / DATES[0]
    write_geo_image(date_dir / "aux_b1.jpg", 43.1968, -90.0581, datetime(2026, 2, 11, 9, 41))
    write_geo_image(date_dir / "aux_b2.jpg", 43.1968, -90.0581, datetime(2026, 2, 11, 9, 42))

    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    grouped = detect_and_write_band_groups(
        date_dir, explicit_groups={"aux": {"b1": "aux_b1.jpg", "b2": "aux_b2.jpg"}})
    assert grouped["formed"], grouped

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv))
    assert "error" in res
    assert DATES[0] in res["error"]
    assert "aux.bandgroup" in res["error"]
    assert not out_csv.exists()


# ── the mapping rider: per-capture digests beside capture_identity ──────────────────────


def test_build_mapping_persists_and_reads_back_capture_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A build through the platform's own producer carries one digest per capture, keyed by
    stem, alongside the per-date capture_identity; both derive from the same row builder, so
    capture_identity's own recompute (round-tripped here) agrees with capture_digests entry by
    entry. The receipt still verifies over the new record shape: load_mapping would refuse
    before returning anything if record_sha256, computed at persist over the whole record,
    disagreed with the receipt written for it."""
    from tcip_mcp.pipelines.image_utils import list_logical_images
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        _read_date_stamps,
        capture_identity,
        record_digest,
    )

    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATES[0]])
    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    build = plant_mapping.load_mapping(tmp_path, "valley")
    assert build is not None
    date = DATES[0]
    assert set(build.capture_digests) == set(build.dates)
    recorded_stems = {a.stem for a in build.assignments[date]}
    assert set(build.capture_digests[date]) == recorded_stems

    stamps = _read_date_stamps(list_logical_images(images_root / date), date)
    assert capture_identity(stamps) == build.capture_identity[date]

    # Not build.record_sha256: that came from load_mapping's own read of this same raw document,
    # so comparing it back to record_digest(raw) would prove nothing.
    from tcip_mcp.audit import audit_log_key

    raw = ts.read(plant_mapping.plant_mapping_key(tmp_path, "valley"))
    recomputed = record_digest(raw)
    page = ts.read_log(audit_log_key(tmp_path))
    receipts = [
        entry["arguments"]["record_sha256"] for entry in page.records
        if entry.get("tool") == "plant_mapping_built"
        and (entry.get("arguments") or {}).get("name") == "valley"
    ]
    assert receipts, "no plant_mapping_built receipt found for 'valley'"
    assert receipts[-1] == recomputed

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv))
    assert "error" not in res, res
    assert out_csv.exists()


def test_a_band_group_manifest_rewritten_in_place_refuses_the_delivery_naming_the_band_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The band group's own membership is unchanged (the same two files, the same stem), so the
    added/missing-stem checks never fire; only the whole-date identity recompute catches a
    manifest rewritten in place, and capture_digests now lets the refusal name the band group's
    own manifest rather than only the date."""
    from tcip_mcp.pipelines.data.band_groups import (
        band_group_manifest_path,
        detect_and_write_band_groups,
    )

    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATES[0]])
    date_dir = images_root / DATES[0]
    write_geo_image(date_dir / "aux_b1.jpg", 43.1968, -90.0581, datetime(2026, 2, 11, 9, 41))
    write_geo_image(date_dir / "aux_b2.jpg", 43.1968, -90.0581, datetime(2026, 2, 11, 9, 42))
    grouped = detect_and_write_band_groups(
        date_dir, explicit_groups={"aux": {"b1": "aux_b1.jpg", "b2": "aux_b2.jpg"}})
    assert grouped["formed"], grouped

    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    manifest_path = band_group_manifest_path(date_dir, "aux")
    # Same "bands" claimed (same members, same kind): a rewrite in place, not an added/removed
    # capture, so only the whole-date identity recompute can catch it.
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    out_csv = tmp_path / "out.csv"
    res = _deliver(
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv))
    assert "error" in res
    assert DATES[0] in res["error"]
    assert "band group manifest" in res["error"]
    assert "aux.bandgroup" in res["error"]
    assert not out_csv.exists()


# ── rail 11: an unreadable image, a raster and a band group all build and deliver ───────


def test_a_date_with_an_unreadable_image_a_raster_and_a_band_group_builds_and_delivers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_mcp.pipelines.data.band_groups import detect_and_write_band_groups
    from tcip_mcp.pipelines.image_utils import list_logical_images
    from tcip_mcp.pipelines.postprocessing.plant_mapping import _read_date_stamps

    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATES[0]])
    date_dir = images_root / DATES[0]

    bad = date_dir / "bad.jpg"
    bad.write_bytes(b"not a real jpeg")
    (date_dir / "raster.npy").write_bytes(b"not really a numpy array, list_logical_images "
                                           b"never decodes it")
    write_geo_image(date_dir / "aux_b1.jpg", 43.1968, -90.0581, datetime(2026, 2, 11, 9, 41))
    write_geo_image(date_dir / "aux_b2.jpg", 43.1968, -90.0581, datetime(2026, 2, 11, 9, 42))
    grouped = detect_and_write_band_groups(
        date_dir, explicit_groups={"aux": {"b1": "aux_b1.jpg", "b2": "aux_b2.jpg"}})
    assert grouped["formed"], grouped

    # White-box: the raster and band-group stamps carry readable=None (no EXIF to fail reading),
    # the unreadable image carries readable=False, before build_mapping ever touches a store.
    logical = list_logical_images(date_dir)
    stamps_by_stem = {s.stem: s for s in _read_date_stamps(logical, DATES[0])}
    assert stamps_by_stem["bad"].kind == "image" and stamps_by_stem["bad"].readable is False
    assert stamps_by_stem["raster"].kind == "raster" and stamps_by_stem["raster"].readable is None
    assert stamps_by_stem["aux"].kind == "band_group" and stamps_by_stem["aux"].readable is None

    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
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
        tmp_path, trait="currant_bloom", mapping_name="valley", plants=POPULATION,
        dataset_root=dataset_root, buckets=preds_by_date.values(),
        output_csv_path=str(out_csv))
    assert "error" not in res, res
    assert out_csv.exists()


# ── rail 14 (cursor memo): the second receipt scan in one process reads only what was ───
# ── appended since the first, never the whole log again ─────────────────────────────────


def test_second_receipt_scan_in_one_process_reads_only_what_was_appended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root)

    calls: list[str | None] = []
    real_read_log = ts.read_log

    def spy(key, after=None):
        calls.append(after)
        return real_read_log(key, after=after)

    monkeypatch.setattr(ts, "read_log", spy)

    res_a = build_plant_mapping(
        tmp_path, name="valley-a", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in res_a, res_a
    build_a = plant_mapping.load_mapping(tmp_path, "valley-a")
    assert build_a is not None
    assert calls == [None]

    res_b = build_plant_mapping(
        tmp_path, name="valley-b", images_root=str(images_root),
        plant_registry=register_plant_registry_for(tmp_path, [plant_csv]))
    assert "error" not in res_b, res_b
    build_b = plant_mapping.load_mapping(tmp_path, "valley-b")
    assert build_b is not None
    assert len(calls) == 2
    assert calls[1] is not None


def test_a_build_leaves_one_row_its_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The door's one line is the library's receipt, which carries the record digest a decorator
    line could not: every row the call adds to the log is counted, and there is one."""
    from tcip_mcp.audit import audit_log_key

    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root)
    registry = register_plant_registry_for(tmp_path, [plant_csv])
    before = len(ts.read_log(audit_log_key(tmp_path)).records)

    res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root), plant_registry=registry)

    assert "error" not in res, res
    rows = ts.read_log(audit_log_key(tmp_path)).records[before:]
    assert [row["tool"] for row in rows] == ["plant_mapping_built"], rows
    assert rows[0]["arguments"]["name"] == "valley"
