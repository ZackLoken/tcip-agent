"""Tests of a mapping's match radius and an orthomosaic attribution's tolerance.

Each date's radius is derived from the positioning errors its photographs report and checked
against the figure numpy computes from the same errors, a stated one recorded as stated, a date
whose photographs report none refused naming the argument; and the nearest-neighbor tolerance an
orthomosaic attribution resolves is bounded by half the grid pitch.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pyproj import Geod

from tcip_mcp.dataset_layout import image_dir
from tcip_mcp.pipelines.postprocessing.plant_mapping import (
    MatchRadius,
    MatchTolerance,
    NoMatchRadiusError,
    NoMatchToleranceError,
    StatedRadius,
    grid_pitch_m,
    read_plant_csvs,
    resolve_nn_tolerance_m,
)
from tests._mapping_fixtures import (
    mapping_over,
    plant_rows,
    write_photos,
    write_plant_csv,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        MappingBuild,
        PlantRecord,
    )

DATE = "2026-02-11"
ROW_ORIGIN = (43.19670, -90.058000)
PLANT_SPACING_M = 3.0
"""Sample data: four plants a row apart by this many meters, east of :data:`ROW_ORIGIN`."""
GEOD = Geod(ellps="WGS84")


def _east(meters: float, north: float = 0.0) -> tuple[float, float]:
    """Return the ``(lat, lon)`` ``meters`` east of :data:`ROW_ORIGIN`, then ``north`` north."""
    lon, lat, _ = GEOD.fwd(ROW_ORIGIN[1], ROW_ORIGIN[0], 90, meters)
    if north:
        lon, lat, _ = GEOD.fwd(lon, lat, 0, north)
    return lat, lon


PLANTS = plant_rows({f"P{i}": _east(i * PLANT_SPACING_M) for i in range(4)})
STRAY_NORTH_M = 4.0
"""How far north of the last plant the one stray photograph was taken."""


def _build(
    tmp_path: Path,
    errors: Sequence[float | None],
    match_radius: StatedRadius | None = None,
) -> MappingBuild:
    """Map a date of up to four photographs through ``build_mapping`` and return the build.

    One photograph sits at each of the first three plants and one :data:`STRAY_NORTH_M` north of
    the last, taken in that order; ``errors`` gives each its reported positioning error, and only
    as many photographs as ``errors`` has entries are written.
    """
    dataset_root = tmp_path / "ds"
    positions = [(PLANTS[i]["lat"], PLANTS[i]["lon"]) for i in range(3)] + [
        _east(3 * PLANT_SPACING_M, STRAY_NORTH_M)
    ]
    write_photos(
        image_dir(dataset_root, DATE),
        {
            f"photo_{i}.jpg": (lat, lon, err)
            for i, ((lat, lon), err) in enumerate(zip(positions, errors))
        },
        datetime(2026, 2, 11, 9),
        timedelta(seconds=1),
    )
    return mapping_over(
        dataset_root,
        [write_plant_csv(tmp_path / "plants.csv", PLANTS)],
        tmp_path,
        match_radius=match_radius,
    )


def test_a_dates_radius_is_the_largest_error_its_photographs_report(
    tmp_path: Path,
) -> None:
    errors = [1.2, 2.5, 0.8, 1.0]

    build = _build(tmp_path, errors)

    radius = float(np.max(errors))
    assert build.match_radius == {
        DATE: MatchRadius(value=radius, source="h_pos_err", rationale=None)
    }
    assert (
        radius > grid_pitch_m(read_plant_csvs([tmp_path / "plants.csv"])) / 2
    )
    by_image = {a.image: a for a in build.assignments[DATE]}
    assert [by_image[f"photo_{i}.jpg"].plant_id for i in range(3)] == [
        "P0",
        "P1",
        "P2",
    ]
    stray = by_image["photo_3.jpg"]
    assert stray.plant_id is None
    assert stray.distance_m == pytest.approx(STRAY_NORTH_M, abs=1e-3)
    assert stray.distance_m > radius


def test_the_served_build_and_its_disclosure_read_the_recorded_radius(
    tmp_path: Path,
) -> None:
    build = _build(tmp_path, [1.2, 2.5, 0.8, 1.0])

    recorded = build.to_record()["match_radius"]
    assert recorded == {
        DATE: {"value": 2.5, "source": "h_pos_err", "rationale": None}
    }
    assert build.served()["match_radius"] == recorded
    disclosure = build.delivery_disclosure(
        {"captures_unverified": [], "plant_csvs_unverified": []}, [DATE]
    )
    assert disclosure["match_radius"] == recorded


def test_a_date_whose_photographs_report_no_error_refuses_naming_the_argument(
    tmp_path: Path,
) -> None:
    with pytest.raises(NoMatchRadiusError, match="match_radius"):
        _build(tmp_path, [None, 1.0, None, 1.0])


def test_a_stated_radius_is_recorded_as_stated_with_its_reason(
    tmp_path: Path,
) -> None:
    stated = StatedRadius(value=5.0, rationale="the receiver's own spec sheet")

    build = _build(tmp_path, [None, None, None, None], match_radius=stated)

    assert build.match_radius == {
        DATE: MatchRadius(
            value=5.0,
            source="stated",
            rationale="the receiver's own spec sheet",
        )
    }
    stray = {a.image: a for a in build.assignments[DATE]}["photo_3.jpg"]
    assert stray.plant_id == "P3"


def test_a_stated_radius_is_recorded_on_a_date_with_no_positioned_capture(
    tmp_path: Path,
) -> None:
    """A stated radius is recorded on every walked date, one with no positioned capture included."""
    from tests._producer_fixtures import write_image

    write_image(image_dir(tmp_path / "ds", "2026-02-12") / "nofix.jpg", (8, 8))
    stated = StatedRadius(value=5.0, rationale="the receiver's own spec sheet")

    build = _build(tmp_path, [None, None, None, None], match_radius=stated)

    assert set(build.match_radius) == {DATE, "2026-02-12"}
    assert build.match_radius["2026-02-12"] == build.match_radius[DATE]


def _plants(tmp_path: Path, plants: list[dict]) -> list[PlantRecord]:
    return read_plant_csvs([write_plant_csv(tmp_path / "p.csv", plants)])


def test_the_nn_tolerance_derives_as_half_the_grid_pitch(
    tmp_path: Path,
) -> None:
    plants = _plants(tmp_path, PLANTS)

    tolerance = resolve_nn_tolerance_m(plants)

    assert tolerance.source == "grid_pitch"
    assert tolerance.value == pytest.approx(PLANT_SPACING_M / 2, rel=1e-3)


def test_the_grid_pitch_is_numpys_median_of_unequal_neighbor_distances(
    tmp_path: Path,
) -> None:
    """The grid pitch is numpy's median of unequal nearest-neighbor distances.

    Four plants 0, 1, 10 and 30 meters east of the origin have nearest-neighbor distances of about
    1, 1, 9 and 20 meters: the pitch is their median, 5.
    """
    plants = _plants(
        tmp_path,
        [
            {"plot": f"P{i}", "accession": f"acc-{i}", "lat": lat, "lon": lon}
            for i, (lat, lon) in enumerate(
                _east(m) for m in (0.0, 1.0, 10.0, 30.0)
            )
        ],
    )
    nearest = [
        min(
            GEOD.inv(p.lon, p.lat, q.lon, q.lat)[2]
            for q in plants
            if q is not p
        )
        for p in plants
    ]

    pitch = grid_pitch_m(plants)

    assert pitch == pytest.approx(float(np.median(nearest)), rel=1e-9)
    assert pitch == pytest.approx(5.0, rel=1e-3)
    assert resolve_nn_tolerance_m(plants).value == pytest.approx(pitch / 2)


def test_the_nn_tolerance_refuses_naming_the_parameter_without_a_pitch(
    tmp_path: Path,
) -> None:
    with pytest.raises(NoMatchToleranceError, match="nn_tolerance_m"):
        resolve_nn_tolerance_m(_plants(tmp_path, PLANTS[:1]))


def test_a_stated_nn_tolerance_is_honored_up_to_half_the_pitch(
    tmp_path: Path,
) -> None:
    plants = _plants(tmp_path, PLANTS)
    half = grid_pitch_m(plants) / 2

    assert resolve_nn_tolerance_m(plants, 1.0) == MatchTolerance(
        value=1.0, source="stated"
    )
    assert resolve_nn_tolerance_m(plants, 5.0) == MatchTolerance(
        value=half, source="stated_capped"
    )
    assert resolve_nn_tolerance_m(plants[:1], 2.0) == MatchTolerance(
        value=2.0, source="stated"
    )
