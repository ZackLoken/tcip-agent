"""A capture's GPS stamp and the position it states.

A coordinate whose hemisphere reference is missing refuses naming the image, and a stamp with a
latitude and no longitude has no position, so its capture stays unmapped beside a plant at
longitude 0.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from tests._image_fixtures import write_geo_image

_WHEN = datetime(2024, 5, 1, 10, 0, 0)


def test_a_latitude_missing_its_hemisphere_reference_refuses_naming_the_image(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        read_image_stamp,
    )

    path = tmp_path / "2024-05-01" / "tree_a.jpg"
    write_geo_image(path, 42.0, -93.0, _WHEN, omit=(0x0001,))

    with pytest.raises(ValueError, match="reference") as refused:
        read_image_stamp(path, "2024-05-01")
    assert str(path) in str(refused.value)


def test_a_stamp_with_every_reference_reads_its_position(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        read_image_stamp,
    )

    path = tmp_path / "2024-05-01" / "tree_a.jpg"
    write_geo_image(path, 42.0, -93.0, _WHEN)

    position = read_image_stamp(path, "2024-05-01").position
    assert position is not None
    assert position[0] == pytest.approx(42.0) and position[1] == pytest.approx(
        -93.0
    )


def test_a_latitude_with_no_longitude_is_unattributed_never_measured_at_longitude_zero(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        MatchRadius,
        PlantRecord,
        assign_plants,
        read_image_stamp,
    )

    lone = tmp_path / "2024-05-01" / "tree_a.jpg"
    write_geo_image(lone, 0.00001, 0.0, _WHEN, omit=(0x0003, 0x0004))
    stamp = read_image_stamp(lone, "2024-05-01")
    assert stamp.position is None

    # A plant at longitude 0 beside the latitude: a defaulted longitude would match it.
    plant = PlantRecord(
        plant_id="P1",
        accession_name="A",
        plot_number=None,
        row_number=None,
        col_number=None,
        lat=0.00001,
        lon=0.0,
    )
    (assignment,) = assign_plants(
        [stamp],
        [plant],
        radius=MatchRadius(
            value=1.0, source="stated", rationale="a generous radius"
        ),
    )
    assert assignment.source == "unmapped"
    assert assignment.plant_id is None and assignment.distance_m is None
