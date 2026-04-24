"""Unit tests for tcip_web.plant_mapping."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tcip_web.plant_mapping import (
    Assignment,
    ImageStamp,
    PlantRecord,
    _segment_runs,
    assign_plants,
    build_mapping,
    haversine_m,
    load_mapping,
    persist_mapping,
    read_plant_csvs,
)


def _stamp(
    stem: str, lat: float, lon: float, seconds: int, *, path: str | None = None
) -> ImageStamp:
    return ImageStamp(
        path=path or f"/data/{stem}.JPG",
        stem=stem,
        date_folder="2-11-26",
        timestamp=datetime(2026, 2, 11, 9, 0, 0) + timedelta(seconds=seconds),
        lat=lat,
        lon=lon,
        h_pos_err=5.0,
    )


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


def test_segment_runs_splits_on_large_jumps() -> None:
    stamps = [
        _stamp("a", 43.1960, -90.0580, 0),
        _stamp("b", 43.1961, -90.0580, 2),
        _stamp("c", 43.2000, -90.0580, 4),  # ~440 m jump -> new run
        _stamp("d", 43.2001, -90.0580, 6),
    ]
    runs = _segment_runs(stamps)
    assert len(runs) == 2
    assert [s.stem for s in runs[0]] == ["a", "b"]
    assert [s.stem for s in runs[1]] == ["c", "d"]


def test_assign_plants_one_to_one() -> None:
    plants = [
        _plant("P1", "A", 43.1960, -90.0580),
        _plant("P2", "B", 43.1961, -90.0580),
    ]
    stamps = [
        _stamp("img1", 43.1960, -90.0580, 0),
        _stamp("img2", 43.1961, -90.0580, 2),
    ]
    assignments = assign_plants(stamps, plants, nn_tolerance_m=10.0)
    # Each image should claim a different plant
    assert len({a.plot_name for a in assignments}) == 2
    assert all(a.source == "sequence" for a in assignments)


def test_assign_plants_handles_missing_gps() -> None:
    plants = [_plant("P1", "A", 43.1960, -90.0580)]
    stamps = [_stamp("img1", 43.1960, -90.0580, 0)]
    stamps[0].lat = None
    stamps[0].lon = None
    assignments = assign_plants(stamps, plants, nn_tolerance_m=10.0)
    assert assignments[0].source == "unmapped"
    assert assignments[0].plot_name is None


def test_assign_plants_no_plants_returns_unmapped() -> None:
    stamps = [_stamp("img1", 43.1960, -90.0580, 0)]
    assignments = assign_plants(stamps, [], nn_tolerance_m=10.0)
    assert all(a.source == "unmapped" for a in assignments)


def test_assign_plants_far_image_falls_back_or_unmapped() -> None:
    plants = [_plant("P1", "A", 43.1960, -90.0580)]
    # ~111 km away
    stamps = [_stamp("img1", 44.1960, -90.0580, 0)]
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
    mapping = {
        "2-11-26": [
            Assignment(
                image_path="/data/IMG.JPG",
                stem="IMG",
                date_folder="2-11-26",
                plot_name="PLOT1",
                accession_name="A",
                confidence=0.8,
                source="sequence",
                distance_m=1.2,
            )
        ]
    }
    out = tmp_path / "mapping.json"
    persist_mapping(mapping, out)
    assert out.exists()
    loaded = load_mapping(out)
    assert list(loaded.keys()) == ["2-11-26"]
    assert loaded["2-11-26"][0].plot_name == "PLOT1"
    assert loaded["2-11-26"][0].confidence == pytest.approx(0.8)
    assert loaded["2-11-26"][0].distance_m == pytest.approx(1.2)


def test_build_mapping_empty_dir(tmp_path: Path) -> None:
    mapping = build_mapping(tmp_path / "nope", [])
    assert mapping == {}
