"""A mapping's match radius: each date's derived from the positioning errors its photographs
report and checked against the figure numpy computes from the same errors, a stated one recorded
as stated, a date whose photographs report none refused naming the argument; and the
nearest-neighbor tolerance an orthomosaic attribution resolves, bounded by half the grid pitch.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from pyproj import Geod

from tcip_mcp.pipelines.postprocessing.plant_mapping import (
    MatchRadius,
    NoMatchRadiusError,
    NoMatchToleranceError,
    StatedRadius,
    build_mapping,
    grid_pitch_m,
    read_plant_csvs,
    resolve_nn_tolerance_m,
)
from tests._image_fixtures import write_geo_image
from tests._mapping_fixtures import write_plant_csv

DATE = "2026-02-11"
ROW_ORIGIN = (43.19670, -90.058000)
PLANT_SPACING_M = 3.0
"""Sample data: four plants a row apart by this many meters, east of :data:`ROW_ORIGIN`."""
GEOD = Geod(ellps="WGS84")


def _east(meters: float, north: float = 0.0) -> tuple[float, float]:
    """``(lat, lon)`` ``meters`` east of :data:`ROW_ORIGIN`, then ``north`` meters north."""
    lon, lat, _ = GEOD.fwd(ROW_ORIGIN[1], ROW_ORIGIN[0], 90, meters)
    if north:
        lon, lat, _ = GEOD.fwd(lon, lat, 0, north)
    return lat, lon


PLANTS = [
    {"plot": f"P{i}", "accession": f"acc-{i}", "lat": lat, "lon": lon}
    for i, (lat, lon) in enumerate(
        _east(i * PLANT_SPACING_M) for i in range(4)
    )
]
STRAY_NORTH_M = 4.0
"""How far north of the last plant the one stray photograph was taken."""


def _build(tmp_path: Path, errors, match_radius: StatedRadius | None = None):
    """A date of photographs, one at each of the first three plants and one
    :data:`STRAY_NORTH_M` north of the last, taken in that order with ``errors`` as their
    reported positioning errors, mapped through ``build_mapping``.
    """
    images_root = tmp_path / "ds" / "images"
    positions = [(PLANTS[i]["lat"], PLANTS[i]["lon"]) for i in range(3)] + [
        _east(3 * PLANT_SPACING_M, STRAY_NORTH_M)
    ]
    for i, ((lat, lon), err) in enumerate(zip(positions, errors)):
        write_geo_image(
            images_root / DATE / f"photo_{i}.jpg",
            lat,
            lon,
            datetime(2026, 2, 11, 9) + timedelta(seconds=i),
            h_pos_err=err,
        )
    return build_mapping(
        images_root,
        [write_plant_csv(tmp_path / "plants.csv", PLANTS)],
        name="valley",
        dataset_root=tmp_path / "ds",
        dataset_id="ds-1",
        project=tmp_path,
        plant_registry={"name": "unregistered", "digest": "0" * 64},
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
    assert [by_image[f"photo_{i}.jpg"].plot_name for i in range(3)] == [
        "P0",
        "P1",
        "P2",
    ]
    stray = by_image["photo_3.jpg"]
    assert stray.plot_name is None
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
    assert stray.plot_name == "P3"


def test_a_stated_radius_is_recorded_on_a_date_with_no_positioned_capture(
    tmp_path: Path,
) -> None:
    """A caller's statement reaches the record on every date the build walked, a date whose one
    photograph carries no position included.
    """
    from PIL import Image

    unpositioned = tmp_path / "ds" / "images" / "2026-02-12" / "nofix.jpg"
    unpositioned.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(unpositioned)
    stated = StatedRadius(value=5.0, rationale="the receiver's own spec sheet")

    build = _build(tmp_path, [None, None, None, None], match_radius=stated)

    assert set(build.match_radius) == {DATE, "2026-02-12"}
    assert build.match_radius["2026-02-12"] == build.match_radius[DATE]


def _plants(tmp_path: Path, plants: list[dict]):
    return read_plant_csvs([write_plant_csv(tmp_path / "p.csv", plants)])


def test_the_nn_tolerance_derives_as_half_the_grid_pitch(
    tmp_path: Path,
) -> None:
    plants = _plants(tmp_path, PLANTS)

    tolerance = resolve_nn_tolerance_m(plants)

    assert tolerance["source"] == "grid_pitch"
    assert tolerance["value"] == pytest.approx(PLANT_SPACING_M / 2, rel=1e-3)


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

    assert resolve_nn_tolerance_m(plants, 1.0) == {
        "value": 1.0,
        "source": "stated",
    }
    assert resolve_nn_tolerance_m(plants, 5.0) == {
        "value": half,
        "source": "stated_capped",
    }
    assert resolve_nn_tolerance_m(plants[:1], 2.0) == {
        "value": 2.0,
        "source": "stated",
    }
