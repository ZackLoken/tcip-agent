"""Unit tests for the plant-mapping pipeline module."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import tcip_store
from tcip_mcp.pipelines.postprocessing.plant_mapping import (
    MappingBuild,
    PlantRecord,
    _segment_runs,
    assign_plants,
    build_mapping,
    haversine_m,
    load_mapping,
    persist_mapping,
    plant_mapping_key,
    read_image_stamp,
    read_plant_csvs,
)
from tests._image_fixtures import write_geo_image
from tests._mapping_fixtures import PLOTS, map_captures
from tests._web_fixtures import named_project

DATE = "2026-02-11"


@pytest.fixture
def stamp(tmp_path: Path):
    """Stamps read by the platform's own reader off geotagged photographs written under
    ``tmp_path``, each captured ``seconds`` after one start time."""
    def make(stem: str, lat: float, lon: float, seconds: int):
        path = tmp_path / DATE / f"{stem}.JPG"
        write_geo_image(path, lat, lon, datetime(2026, 2, 11, 9) + timedelta(seconds=seconds))
        return read_image_stamp(path, DATE)

    return make


def _one_build(project: Path) -> MappingBuild:
    """One mapping named "mapping" of a registered dataset's captures on :data:`DATE`, built,
    registered and persisted through the platform's own producers (``map_captures``), as
    ``load_mapping`` reads it back."""
    named_project(project, "Orchard")
    map_captures(project, project / "ds", [DATE], name="mapping")
    build = load_mapping(project, "mapping")
    assert build is not None
    return build


def _plant(plot: str, accession: str, lat: float, lon: float) -> PlantRecord:
    return PlantRecord(
        plot_name=plot,
        accession_name=accession,
        plot_number=None,
        row_number=None,
        col_number=None,
        lat=lat,
        lon=lon,
    )


def test_haversine_known_distance() -> None:
    # Two points ~111 km apart at the equator
    d = haversine_m(0.0, 0.0, 1.0, 0.0)
    assert 110_500 < d < 112_000


def test_segment_runs_splits_on_large_jumps(stamp) -> None:
    stamps = [
        stamp("a", 43.1960, -90.0580, 0),
        stamp("b", 43.1961, -90.0580, 2),
        stamp("c", 43.2000, -90.0580, 4),  # ~440 m jump -> new run
        stamp("d", 43.2001, -90.0580, 6),
    ]
    runs = _segment_runs(stamps)
    assert len(runs) == 2
    assert [s.stem for s in runs[0]] == ["a", "b"]
    assert [s.stem for s in runs[1]] == ["c", "d"]


def test_segment_runs_splits_on_multiple_jumps(stamp) -> None:
    # Three row runs of ~2 m in-row steps separated by two ~15 m row-transition jumps.
    # 15 m is well under a fixed 25 m threshold, so this only splits under a derivation
    # that reads the break relative to this date's own ~2 m walking pace.
    stamps = [
        stamp("a", 43.19600000, -90.0580, 0),
        stamp("b", 43.19601797, -90.0580, 2),
        stamp("c", 43.19603593, -90.0580, 4),
        stamp("d", 43.19617068, -90.0580, 6),  # ~15 m jump -> new run
        stamp("e", 43.19618865, -90.0580, 8),
        stamp("f", 43.19632339, -90.0580, 10),  # ~15 m jump -> new run
        stamp("g", 43.19634136, -90.0580, 12),
    ]
    runs = _segment_runs(stamps)
    assert [[s.stem for s in r] for r in runs] == [
        ["a", "b", "c"],
        ["d", "e"],
        ["f", "g"],
    ]


def test_segment_runs_curved_row_not_fragile_to_uneven_steps(stamp) -> None:
    # In-row steps vary (20-28 m, simulating a curved row's uneven pace) and one includes a step
    # larger than a fixed 25 m threshold, followed by one unambiguous ~300 m row-transition
    # jump. A fixed-distance rule would split on every step over 25 m; the derivation should not.
    stamps = [
        stamp("a", 43.196000, -90.0580, 0),
        stamp("b", 43.196180, -90.0580, 2),  # ~20 m
        stamp("c", 43.196431, -90.0580, 4),  # ~28 m
        stamp("d", 43.196628, -90.0580, 6),  # ~22 m
        stamp("e", 43.196862, -90.0580, 8),  # ~26 m
        stamp("f", 43.199556, -90.0580, 10),  # ~300 m jump -> new run
    ]
    runs = _segment_runs(stamps)
    assert len(runs) == 2
    assert [s.stem for s in runs[0]] == ["a", "b", "c", "d", "e"]
    assert [s.stem for s in runs[1]] == ["f"]


def test_segment_runs_uniform_gaps_stay_one_run(stamp) -> None:
    # Roughly uniform ~27-32 m gaps throughout, all above a fixed 25 m threshold (which
    # would split every consecutive pair) but with no real bimodal break in the sequence.
    stamps = [
        stamp("a", 43.196000, -90.0580, 0),
        stamp("b", 43.196252, -90.0580, 2),  # ~28 m
        stamp("c", 43.196521, -90.0580, 4),  # ~30 m
        stamp("d", 43.196808, -90.0580, 6),  # ~32 m
        stamp("e", 43.197069, -90.0580, 8),  # ~29 m
        stamp("f", 43.197347, -90.0580, 10),  # ~31 m
        stamp("g", 43.197590, -90.0580, 12),  # ~27 m
    ]
    runs = _segment_runs(stamps)
    assert len(runs) == 1
    assert [s.stem for s in runs[0]] == ["a", "b", "c", "d", "e", "f", "g"]


def test_segment_runs_duplicate_gps_gap_does_not_hijack_the_split(stamp) -> None:
    # A duplicate/cached EXIF GPS reading between two rapid captures (an exact 0 m gap) carries
    # no walking-pace information; it must not be read as an "infinitely large" jump that wins
    # over a genuine row-transition jump elsewhere in the sequence.
    stamps = [
        stamp("a", 43.19600000, -90.0580, 0),
        stamp("b", 43.19601797, -90.0580, 2),  # ~2 m
        stamp("c", 43.19603593, -90.0580, 4),  # ~2 m
        stamp("d", 43.19603593, -90.0580, 6),  # duplicate of c: 0 m gap
        stamp("e", 43.19605390, -90.0580, 8),  # ~2 m
        stamp("f", 43.19874790, -90.0580, 10),  # ~300 m jump -> new run
        stamp("g", 43.19876587, -90.0580, 12),  # ~2 m
    ]
    runs = _segment_runs(stamps)
    assert len(runs) == 2
    assert [s.stem for s in runs[0]] == ["a", "b", "c", "d", "e"]
    assert [s.stem for s in runs[1]] == ["f", "g"]


def test_assign_plants_one_to_one(stamp) -> None:
    plants = [
        _plant("P1", "A", 43.1960, -90.0580),
        _plant("P2", "B", 43.1961, -90.0580),
    ]
    stamps = [
        stamp("img1", 43.1960, -90.0580, 0),
        stamp("img2", 43.1961, -90.0580, 2),
    ]
    assignments = assign_plants(stamps, plants, nn_tolerance_m=10.0)
    # Each image should claim a different plant
    assert len({a.plot_name for a in assignments}) == 2
    assert all(a.source == "sequence" for a in assignments)


def test_assign_plants_handles_missing_gps(tmp_path: Path) -> None:
    plants = [_plant("P1", "A", 43.1960, -90.0580)]
    path = tmp_path / DATE / "img1.JPG"
    path.parent.mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (8, 8)).save(path)
    stamps = [read_image_stamp(path, DATE)]
    assignments = assign_plants(stamps, plants, nn_tolerance_m=10.0)
    assert assignments[0].source == "unmapped"
    assert assignments[0].plot_name is None


def test_assign_plants_no_plants_returns_unmapped(stamp) -> None:
    stamps = [stamp("img1", 43.1960, -90.0580, 0)]
    assignments = assign_plants(stamps, [], nn_tolerance_m=10.0)
    assert all(a.source == "unmapped" for a in assignments)


def test_assign_plants_far_image_falls_back_or_unmapped(stamp) -> None:
    plants = [_plant("P1", "A", 43.1960, -90.0580)]
    # ~111 km away
    stamps = [stamp("img1", 44.1960, -90.0580, 0)]
    assignments = assign_plants(stamps, plants, nn_tolerance_m=10.0)
    assert assignments[0].source == "unmapped"


def test_read_plant_csvs(tmp_path: Path) -> None:
    csv_path = tmp_path / "plants.csv"
    csv_path.write_text(
        "plot_name,accession_name,plot_number,block_number,is_a_control,rep_number,range_number,"
        "row_number,col_number,seedlot_name,num_seed_per_plot,weight_gram_seed_per_plot,entry_number,"
        "WGS84_centroid_x,WGS84_centroid_y\n"
        "2026_VF_MWxMW_PLOT1,MN_20_0305_602,1101002001.0,1.0,,1.0,,2,1,,,,,"
        "-90.05808906799996,43.19684267700006\n",
        encoding="utf-8",
    )
    records = read_plant_csvs([csv_path])
    assert len(records) == 1
    r = records[0]
    assert r.plot_name == "2026_VF_MWxMW_PLOT1"
    assert r.lat == 43.19684267700006
    assert r.lon == -90.05808906799996


def test_persist_and_load_mapping_round_trip(tmp_path: Path) -> None:
    build = replace(_one_build(tmp_path), name="again")
    persist_mapping(build, tmp_path, actor=None)
    loaded = load_mapping(tmp_path, "again")
    assert loaded is not None
    assert loaded.assignments == build.assignments
    assert loaded.capture_digests == build.capture_digests
    assert {a.plot_name for a in loaded.assignments[DATE]} == set(PLOTS)


def test_load_mapping_refuses_a_record_missing_capture_digests(tmp_path: Path) -> None:
    """A record without its capture digests (one the producer wrote, then stripped) fails the
    required-keys read naming the missing key."""
    _one_build(tmp_path)
    key = plant_mapping_key(tmp_path, "mapping")
    record = tcip_store.read(key)
    del record["capture_digests"]
    tcip_store.replace(key, record)

    with pytest.raises(ValueError, match="capture_digests"):
        load_mapping(tmp_path, "mapping")


def test_build_mapping_empty_dir_refuses_naming_no_capture(tmp_path: Path) -> None:
    """No capture at all under the requested dates refuses by name, naming the images root."""
    with pytest.raises(Exception, match="no capture under") as exc:
        build_mapping(
            tmp_path / "nope", [], name="mapping", dataset_root=tmp_path / "ds",
            dataset_id="ds-1", project=tmp_path,
            plant_registry={"name": "unregistered", "digest": "0" * 64}, nn_tolerance_m=10.0,
        )
    assert type(exc.value).__name__ == "UngeoreferencedCaptureError"


def test_persisting_a_mapping_into_a_directory_that_does_not_exist_yet_still_lands(
    tmp_path: Path
) -> None:
    """The first mapping of a fresh project, whose state directory holds nothing yet, lands
    where it is named and reads back."""
    build = _one_build(tmp_path)

    assert tcip_store.exists(plant_mapping_key(tmp_path, "mapping"))
    assert {a.plot_name for a in build.assignments[DATE]} == set(PLOTS)


def test_persisting_a_mapping_waits_on_the_lock_its_record_is_written_under(
    tmp_path: Path
) -> None:
    """The write takes the database's write lock, and reports the contention rather than
    writing past a holder of it."""
    from tcip_store import StoreBusyError
    from tcip_store.sqlite_backend import SqliteBackend

    from tests._audit_fixtures import held_by_another_writer

    build = replace(_one_build(tmp_path), name="again")
    key = plant_mapping_key(tmp_path, "again")
    backend = tcip_store.bind(SqliteBackend(lock_timeout_s=0.2))

    with held_by_another_writer(key), pytest.raises(StoreBusyError):
        persist_mapping(build, tmp_path, actor=None)
    assert not tcip_store.exists(key)
    backend.close()
