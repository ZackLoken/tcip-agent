"""Unit tests for the plant-mapping pipeline module."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

import tcip_store
from tcip_mcp.pipelines.postprocessing.plant_mapping import (
    Assignment,
    ImageStamp,
    MappingBuild,
    PlantRecord,
    _segment_runs,
    assign_plants,
    build_mapping,
    haversine_m,
    load_mapping,
    persist_mapping,
    plant_mapping_key,
    read_plant_csvs,
)


def _stamp(
    stem: str, lat: float, lon: float, seconds: int, *, path: str | None = None
) -> ImageStamp:
    return ImageStamp(
        path=path or f"/data/{stem}.JPG",
        stem=stem,
        date_folder="2-11-26",
        kind="image",
        name=f"{stem}.JPG",
        timestamp=datetime(2026, 2, 11, 9, 0, 0) + timedelta(seconds=seconds),
        lat=lat,
        lon=lon,
        h_pos_err=5.0,
        readable=True,
    )


def _build(assignments: dict[str, list[Assignment]]) -> MappingBuild:
    """A ``MappingBuild`` around hand-composed assignments, for testing the pipeline module's
    own ``persist_mapping``/``load_mapping`` round trip: every other provenance field is a
    placeholder these tests do not exercise."""
    return MappingBuild(
        name="mapping", project_root="/proj", dataset_root="/proj/ds", dataset_id="ds-1",
        built_by="build_plant_mapping", built_at="2026-02-11T00:00:00+00:00",
        dates_requested=None, dates=sorted(assignments),
        nn_tolerance_m={"value": 10.0, "source": "fallback"}, plant_csvs=[],
        capture_identity={d: "0" * 16 for d in assignments},
        capture_digests={d: {} for d in assignments}, unreadable={d: [] for d in assignments},
        assignments=assignments,
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


def test_segment_runs_splits_on_multiple_jumps() -> None:
    # Three row runs of ~2 m in-row steps separated by two ~15 m row-transition jumps.
    # 15 m is well under the old fixed 25 m constant, so this only splits under a derivation
    # that reads the break relative to this date's own ~2 m walking pace.
    stamps = [
        _stamp("a", 43.19600000, -90.0580, 0),
        _stamp("b", 43.19601797, -90.0580, 2),
        _stamp("c", 43.19603593, -90.0580, 4),
        _stamp("d", 43.19617068, -90.0580, 6),  # ~15 m jump -> new run
        _stamp("e", 43.19618865, -90.0580, 8),
        _stamp("f", 43.19632339, -90.0580, 10),  # ~15 m jump -> new run
        _stamp("g", 43.19634136, -90.0580, 12),
    ]
    runs = _segment_runs(stamps)
    assert [[s.stem for s in r] for r in runs] == [
        ["a", "b", "c"],
        ["d", "e"],
        ["f", "g"],
    ]


def test_segment_runs_curved_row_not_fragile_to_uneven_steps() -> None:
    # In-row steps vary (20-28 m, simulating a curved row's uneven pace) and one includes a step
    # larger than the old fixed 25 m constant, followed by one unambiguous ~300 m row-transition
    # jump. A fixed-distance rule would split on every step over 25 m; the derivation should not.
    stamps = [
        _stamp("a", 43.196000, -90.0580, 0),
        _stamp("b", 43.196180, -90.0580, 2),  # ~20 m
        _stamp("c", 43.196431, -90.0580, 4),  # ~28 m
        _stamp("d", 43.196628, -90.0580, 6),  # ~22 m
        _stamp("e", 43.196862, -90.0580, 8),  # ~26 m
        _stamp("f", 43.199556, -90.0580, 10),  # ~300 m jump -> new run
    ]
    runs = _segment_runs(stamps)
    assert len(runs) == 2
    assert [s.stem for s in runs[0]] == ["a", "b", "c", "d", "e"]
    assert [s.stem for s in runs[1]] == ["f"]


def test_segment_runs_uniform_gaps_stay_one_run() -> None:
    # Roughly uniform ~27-32 m gaps throughout, all above the old fixed 25 m constant (which
    # would have split every consecutive pair) but with no real bimodal break in the sequence.
    stamps = [
        _stamp("a", 43.196000, -90.0580, 0),
        _stamp("b", 43.196252, -90.0580, 2),  # ~28 m
        _stamp("c", 43.196521, -90.0580, 4),  # ~30 m
        _stamp("d", 43.196808, -90.0580, 6),  # ~32 m
        _stamp("e", 43.197069, -90.0580, 8),  # ~29 m
        _stamp("f", 43.197347, -90.0580, 10),  # ~31 m
        _stamp("g", 43.197590, -90.0580, 12),  # ~27 m
    ]
    runs = _segment_runs(stamps)
    assert len(runs) == 1
    assert [s.stem for s in runs[0]] == ["a", "b", "c", "d", "e", "f", "g"]


def test_segment_runs_duplicate_gps_gap_does_not_hijack_the_split() -> None:
    # A duplicate/cached EXIF GPS reading between two rapid captures (an exact 0 m gap) carries
    # no walking-pace information; it must not be read as an "infinitely large" jump that wins
    # over a genuine row-transition jump elsewhere in the sequence.
    stamps = [
        _stamp("a", 43.19600000, -90.0580, 0),
        _stamp("b", 43.19601797, -90.0580, 2),  # ~2 m
        _stamp("c", 43.19603593, -90.0580, 4),  # ~2 m
        _stamp("d", 43.19603593, -90.0580, 6),  # duplicate of c: 0 m gap
        _stamp("e", 43.19605390, -90.0580, 8),  # ~2 m
        _stamp("f", 43.19874790, -90.0580, 10),  # ~300 m jump -> new run
        _stamp("g", 43.19876587, -90.0580, 12),  # ~2 m
    ]
    runs = _segment_runs(stamps)
    assert len(runs) == 2
    assert [s.stem for s in runs[0]] == ["a", "b", "c", "d", "e"]
    assert [s.stem for s in runs[1]] == ["f", "g"]


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
    build = _build({
        "2-11-26": [
            Assignment(
                image_path="/data/IMG.JPG",
                stem="IMG",
                date_folder="2-11-26",
                plot_name="PLOT1",
                accession_name="A",
                source="sequence",
                distance_m=1.2,
            )
        ]
    })
    persist_mapping(build, tmp_path, "mapping")
    assert tcip_store.exists(plant_mapping_key(tmp_path, "mapping"))
    loaded = load_mapping(tmp_path, "mapping")
    assert loaded is not None
    assert list(loaded.assignments.keys()) == ["2-11-26"]
    assert loaded.assignments["2-11-26"][0].plot_name == "PLOT1"
    assert loaded.assignments["2-11-26"][0].source == "sequence"
    assert loaded.assignments["2-11-26"][0].distance_m == pytest.approx(1.2)


def test_load_mapping_refuses_an_older_record_missing_capture_digests(tmp_path: Path) -> None:
    """A record built before capture_digests joined the required top-level keys (here, one this
    same producer wrote, then stripped, standing in for that older shape honestly: the bytes a
    pre-rider build actually left are gone) fails the required-keys read naming the missing key
    and the rebuild remedy, rather than silently defaulting it."""
    build = _build({
        "2-11-26": [
            Assignment(
                image_path="/data/IMG.JPG", stem="IMG", date_folder="2-11-26",
                plot_name="PLOT1", accession_name="A", source="sequence", distance_m=1.2,
            )
        ]
    })
    persist_mapping(build, tmp_path, "mapping")
    key = plant_mapping_key(tmp_path, "mapping")
    record = tcip_store.read(key)
    del record["capture_digests"]
    tcip_store.replace(key, record)

    with pytest.raises(ValueError, match="capture_digests"):
        load_mapping(tmp_path, "mapping")


def test_load_mapping_refuses_a_date_capture_identity_names_but_capture_digests_omits(
    tmp_path: Path,
) -> None:
    """capture_digests present as a dict but missing the entry for a date capture_identity
    still names: verify_mapping_inputs' own .get(date, {}) would otherwise silently read that
    date as having no digests at all, rather than the coverage gap it actually is."""
    build = _build({
        "2-11-26": [
            Assignment(
                image_path="/data/IMG.JPG", stem="IMG", date_folder="2-11-26",
                plot_name="PLOT1", accession_name="A", source="sequence", distance_m=1.2,
            )
        ]
    })
    persist_mapping(build, tmp_path, "mapping")
    key = plant_mapping_key(tmp_path, "mapping")
    record = tcip_store.read(key)
    del record["capture_digests"]["2-11-26"]
    tcip_store.replace(key, record)

    with pytest.raises(ValueError, match="capture_digests"):
        load_mapping(tmp_path, "mapping")


def test_scan_receipts_refuses_a_version_refused_log_line_not_as_corruption(
    tmp_path: Path,
) -> None:
    """A version-refused audit line is a policy fact, not corruption: it must still block the
    receipt scan (an entry could be hiding behind it unread), naming schema_version rather than
    the corrupt-log wording. Planted as bytes on the file backend's own log, since the seam's
    writer refuses to append the line itself."""
    from tcip_mcp.audit import audit_log_key
    from tcip_store.file_backend import FileBackend

    tcip_store.bind(FileBackend())
    try:
        build = _build({
            "2-11-26": [
                Assignment(
                    image_path="/data/IMG.JPG", stem="IMG", date_folder="2-11-26",
                    plot_name="PLOT1", accession_name="A", source="sequence", distance_m=1.2,
                )
            ]
        })
        persist_mapping(build, tmp_path, "mapping")
        key = audit_log_key(tmp_path)
        poisoned = tcip_store.get_descriptor(key.store).codec.encode(
            {"tool": "a_future_tool", "schema_version": 99})
        with open(FileBackend().path_for(key), "ab") as handle:
            handle.write(poisoned + b"\n")

        with pytest.raises(ValueError, match="schema_version"):
            load_mapping(tmp_path, "mapping")
    finally:
        tcip_store.unbind()


def test_scan_receipts_still_admits_a_real_receipt_with_no_version_refused_lines(
    tmp_path: Path,
) -> None:
    build = _build({
        "2-11-26": [
            Assignment(
                image_path="/data/IMG.JPG", stem="IMG", date_folder="2-11-26",
                plot_name="PLOT1", accession_name="A", source="sequence", distance_m=1.2,
            )
        ]
    })
    persist_mapping(build, tmp_path, "mapping")

    loaded = load_mapping(tmp_path, "mapping")

    assert loaded is not None
    assert loaded.assignments["2-11-26"][0].plot_name == "PLOT1"


def test_build_mapping_empty_dir_refuses_naming_no_capture(tmp_path: Path) -> None:
    """No capture at all under the requested dates refuses by name, rather than returning an
    empty, uselessly-successful build: an empty or absent images_root can never deliver, so the
    door refuses at the source instead of a caller discovering it downstream."""
    with pytest.raises(Exception, match="no capture under") as exc:
        build_mapping(
            tmp_path / "nope", [], name="mapping", dataset_root=tmp_path / "ds",
            dataset_id="ds-1", project_root=tmp_path, built_by="build_plant_mapping",
        )
    assert type(exc.value).__name__ == "UngeoreferencedCaptureRefusal"


def _one_build() -> MappingBuild:
    return _build({
        "2-11-26": [
            Assignment(
                image_path="/data/IMG.JPG",
                stem="IMG",
                date_folder="2-11-26",
                plot_name="PLOT1",
                accession_name="A",
                source="sequence",
                distance_m=1.2,
            )
        ]
    })


def test_persisting_a_mapping_into_a_directory_that_does_not_exist_yet_still_lands(
    tmp_path: Path
) -> None:
    """A caller names where the mapping goes, and the location is made ready for them.

    The state directory of a fresh project has nothing in it, so a persist that required the
    location to exist already would refuse the very first build.
    """
    persist_mapping(_one_build(), tmp_path, "mapping")

    assert tcip_store.exists(plant_mapping_key(tmp_path, "mapping"))
    loaded = load_mapping(tmp_path, "mapping")
    assert loaded is not None
    assert loaded.assignments["2-11-26"][0].plot_name == "PLOT1"


def test_persisting_a_mapping_waits_on_the_lock_its_record_is_written_under(
    tmp_path: Path
) -> None:
    """The write takes the mapping record's own lock, and reports the contention rather than
    writing past it.

    Two processes can be handed the same mapping path, and an unguarded write would interleave
    with the other's bytes and leave a document that parses as a mapping while holding neither
    build's assignments.

    Bound to the file backend on purpose: a per-path lock held from outside the write is the
    file backend's own exclusion, and the contention a database backend reports is its own.
    """
    import threading

    from tcip_store import StoreBusy
    from tcip_store.file_backend import FileBackend, path_lock

    out = tmp_path / ".tcip" / "state" / "plant_mappings" / "mapping.json"
    tcip_store.bind(FileBackend(lock_timeout_s=0.2))

    holding, release = threading.Event(), threading.Event()

    def hold() -> None:
        with path_lock(out, timeout_s=30):
            holding.set()
            release.wait(30)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert holding.wait(30)
        with pytest.raises(StoreBusy):
            persist_mapping(_one_build(), tmp_path, "mapping")
        assert not out.exists()
    finally:
        release.set()
        holder.join(30)
