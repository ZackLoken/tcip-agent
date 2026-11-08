"""Unit tests for the plant-mapping pipeline module."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import tcip_store
from tcip_mcp.dataset_layout import image_dir, image_root, list_dates
from tcip_mcp.pipelines.postprocessing.plant_mapping import (
    MappingBuild,
    PlantRecord,
    StatedRadius,
    assign_plants,
    date_match_radius,
    load_mapping,
    nearest_plant,
    persist_mapping,
    plant_mapping_key,
    read_image_stamp,
    read_plant_csvs,
)
from tests._image_fixtures import write_geo_image
from tests._mapping_fixtures import (
    PLOTS,
    geolocated_captures,
    map_captures,
    mapping_over,
    write_plots_csv,
)
from tests._producer_fixtures import write_image
from tests._web_fixtures import named_project

if TYPE_CHECKING:
    from collections.abc import Callable

    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        Assignment,
        ImageStamp,
    )

DATE = "2026-02-11"
WIDE = StatedRadius(value=10.0, rationale="reaches every plant a test places")
"""A stated radius past every plant the assignment tests place near their photographs."""


def _assign(
    stamps: list[ImageStamp], plants: list[PlantRecord]
) -> list[Assignment]:
    """Run ``assign_plants`` over ``stamps`` within their date's radius stated as :data:`WIDE`."""
    return assign_plants(
        stamps, plants, radius=date_match_radius(stamps, WIDE)
    )


@pytest.fixture
def stamp(
    tmp_path: Path,
) -> Callable[[str, float, float, int], ImageStamp]:
    """Return a maker of stamps read off geotagged photographs written under ``tmp_path``.

    The platform's own reader reads each stamp, each photograph captured ``seconds`` after one
    start time.
    """

    def make(stem: str, lat: float, lon: float, seconds: int) -> ImageStamp:
        path = tmp_path / DATE / f"{stem}.JPG"
        write_geo_image(
            path,
            lat,
            lon,
            datetime(2026, 2, 11, 9) + timedelta(seconds=seconds),
        )
        return read_image_stamp(path, DATE)

    return make


def _one_build(project: Path) -> MappingBuild:
    """Return ``load_mapping``'s read of a mapping named "mapping" of the captures on :data:`DATE`.

    ``project`` is made through ``named_project`` and the mapping built by ``map_captures`` over
    its dataset ``ds``.
    """
    named_project(project, "Orchard")
    map_captures(project, project / "ds", [DATE], name="mapping")
    build = load_mapping(project, "mapping")
    assert build is not None
    return build


def _plant(plot: str, accession: str, lat: float, lon: float) -> PlantRecord:
    return PlantRecord(
        plant_id=plot,
        accession_name=accession,
        plot_number=None,
        row_number=None,
        col_number=None,
        lat=lat,
        lon=lon,
    )


def test_a_plants_distance_is_the_wgs84_geodesic() -> None:
    """A plant's distance is the WGS84 geodesic.

    One degree of longitude along the equator is the WGS84 semi-major axis times pi over 180
    (6,378,137 m, the datum's own definition), the distance the nearest plant is reported at.
    """
    import math

    _index, distance = nearest_plant(
        (0.0, 0.0), [_plant("P1", "A", 0.0, 1.0)], within_m=0.0
    )
    assert distance == pytest.approx(6_378_137.0 * math.pi / 180, abs=1e-3)


def test_assign_plants_one_to_one(
    stamp: Callable[[str, float, float, int], ImageStamp],
) -> None:
    plants = [
        _plant("P1", "A", 43.1960, -90.0580),
        _plant("P2", "B", 43.1961, -90.0580),
    ]
    stamps = [
        stamp("img1", 43.1960, -90.0580, 0),
        stamp("img2", 43.1961, -90.0580, 2),
    ]
    assignments = _assign(stamps, plants)
    # Each image should claim a different plant
    assert len({a.plant_id for a in assignments}) == 2
    assert all(a.source == "sequence" for a in assignments)


def test_the_earlier_capture_claims_the_plant_two_captures_are_nearest_to(
    stamp: Callable[[str, float, float, int], ImageStamp],
) -> None:
    """Two captures nearest to one plant: the earlier claims it, the later takes the next nearest.

    The stamps are given later capture first; the assignments come back in that order.
    """
    plants = [
        _plant("P1", "A", 43.1960, -90.0580),
        _plant("P2", "B", 43.19603, -90.0580),
    ]
    later = stamp("later", 43.1960, -90.0580, 10)
    earlier = stamp("earlier", 43.1960, -90.0580, 0)
    assigned = _assign([later, earlier], plants)
    assert [(a.plant_id, a.source) for a in assigned] == [
        ("P2", "sequence"),
        ("P1", "sequence"),
    ]


def test_the_later_capture_takes_the_claimed_plant_when_no_other_is_in_range(
    stamp: Callable[[str, float, float, int], ImageStamp],
) -> None:
    """With no unclaimed plant within the radius, the later capture takes the claimed nearest."""
    plants = [_plant("P1", "A", 43.1960, -90.0580)]
    later = stamp("later", 43.1960, -90.0580, 10)
    earlier = stamp("earlier", 43.1960, -90.0580, 0)
    assigned = _assign([later, earlier], plants)
    assert [(a.plant_id, a.source) for a in assigned] == [
        ("P1", "nearest_neighbor"),
        ("P1", "sequence"),
    ]


def test_assign_plants_handles_missing_gps(tmp_path: Path) -> None:
    plants = [_plant("P1", "A", 43.1960, -90.0580)]
    path = write_image(tmp_path / DATE / "img1.JPG", (8, 8))
    stamps = [read_image_stamp(path, DATE)]
    assignments = _assign(stamps, plants)
    assert assignments[0].source == "unmapped"
    assert assignments[0].plant_id is None


def test_assign_plants_no_plants_returns_unmapped(
    stamp: Callable[[str, float, float, int], ImageStamp],
) -> None:
    stamps = [stamp("img1", 43.1960, -90.0580, 0)]
    assignments = _assign(stamps, [])
    assert all(a.source == "unmapped" for a in assignments)


def test_assign_plants_far_image_falls_back_or_unmapped(
    stamp: Callable[[str, float, float, int], ImageStamp],
) -> None:
    plants = [_plant("P1", "A", 43.1960, -90.0580)]
    # ~111 km away
    stamps = [stamp("img1", 44.1960, -90.0580, 0)]
    assignments = _assign(stamps, plants)
    assert assignments[0].source == "unmapped"


def test_read_plant_csvs(tmp_path: Path) -> None:
    csv_path = tmp_path / "plants.csv"
    csv_path.write_text(
        "plant_id,accession_name,plot_number,block_number,is_a_control,rep_number,range_number,"
        "row_number,col_number,seedlot_name,num_seed_per_plot,weight_gram_seed_per_plot,entry_number,"
        "WGS84_centroid_x,WGS84_centroid_y\n"
        "2026_VF_MWxMW_PLOT1,MN_20_0305_602,1101002001.0,1.0,,1.0,,2,1,,,,,"
        "-90.05808906799996,43.19684267700006\n",
        encoding="utf-8",
    )
    records = read_plant_csvs([csv_path])
    assert len(records) == 1
    r = records[0]
    assert r.plant_id == "2026_VF_MWxMW_PLOT1"
    assert r.lat == 43.19684267700006
    assert r.lon == -90.05808906799996


def test_persist_and_load_mapping_round_trip(tmp_path: Path) -> None:
    build = replace(_one_build(tmp_path), name="again")
    persist_mapping(build, tmp_path, actor=None)
    loaded = load_mapping(tmp_path, "again")
    assert loaded is not None
    assert loaded.assignments == build.assignments
    assert loaded.capture_digests == build.capture_digests
    assert {a.plant_id for a in loaded.assignments[DATE]} == set(PLOTS)


def test_load_mapping_refuses_a_record_missing_capture_digests(
    tmp_path: Path,
) -> None:
    """``load_mapping`` refuses a record missing its capture digests, naming the key.

    The record is one the producer wrote, then stripped.
    """
    _one_build(tmp_path)
    key = plant_mapping_key(tmp_path, "mapping")
    record = tcip_store.read(key)
    del record["capture_digests"]
    tcip_store.replace(key, record)

    with pytest.raises(ValueError, match="capture_digests"):
        load_mapping(tmp_path, "mapping")


def test_build_mapping_empty_dir_refuses_naming_no_capture(
    tmp_path: Path,
) -> None:
    """``build_mapping`` over no capture refuses with ``UngeoreferencedCaptureError``, saying so."""
    with pytest.raises(Exception, match="no capture under") as exc:
        mapping_over(tmp_path / "ds", [], tmp_path)
    assert type(exc.value).__name__ == "UngeoreferencedCaptureError"


def test_a_mapping_walks_the_captures_its_dataset_lists(
    tmp_path: Path,
) -> None:
    """A mapping's dates are its dataset's ``list_dates``; a hidden directory is no capture."""
    named_project(tmp_path, "Orchard")
    dataset_root = tmp_path / "ds"
    write_geo_image(
        image_root(dataset_root) / ".staging" / "P1_staged.jpg",
        *PLOTS["P1"],
        datetime(2026, 2, 11, 9),
    )
    map_captures(tmp_path, dataset_root, [DATE], name="mapping")

    built = load_mapping(tmp_path, "mapping")
    assert built.dates == list_dates(dataset_root) == [DATE]


def test_a_mapping_of_named_dates_maps_them_past_an_ambiguous_capture(
    tmp_path: Path,
) -> None:
    """A mapping naming its dates maps them whatever a capture it does not name holds.

    The unnamed capture holds two images of one stem, which a walk scanning it refuses.
    """
    dataset_root = tmp_path / "ds"
    geolocated_captures(dataset_root, [DATE])
    ambiguous = image_dir(dataset_root, "2026-02-18")
    write_image(ambiguous / "P1_again.jpg", (8, 8))
    write_image(ambiguous / "P1_again.png", (8, 8))

    built = mapping_over(
        dataset_root,
        [write_plots_csv(tmp_path / "plants.csv")],
        tmp_path,
        dates=[DATE],
    )

    assert built.dates == [DATE]
    assert {a.plant_id for a in built.assignments[DATE]} == set(PLOTS)


def test_persisting_a_mapping_into_a_directory_that_does_not_exist_yet_still_lands(
    tmp_path: Path,
) -> None:
    """A fresh project's first mapping lands where it is named and reads back.

    The project's state directory holds nothing yet.
    """
    build = _one_build(tmp_path)

    assert tcip_store.exists(plant_mapping_key(tmp_path, "mapping"))
    assert {a.plant_id for a in build.assignments[DATE]} == set(PLOTS)


def test_persisting_a_mapping_waits_on_the_lock_its_record_is_written_under(
    tmp_path: Path,
) -> None:
    """Persisting a mapping under another writer's lock raises ``StoreBusyError``, writing none."""
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
