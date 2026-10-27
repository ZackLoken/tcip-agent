"""Plant registries and plant mappings a test registers through the platform's own producers."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from tcip_mcp.dataset_layout import image_dir

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        MappingBuild,
        StatedRadius,
    )

PLOTS = {"P1": (43.19670, -90.058000), "P2": (43.19670, -90.058037)}
"""Two plots one row apart, each at its ``(lat, lon)``."""

GRID_COLUMNS = ("plot_number", "row_number", "col_number")
"""The plant-grid position columns a breeding layout's registry CSV carries beside each plant's
location, written as :func:`write_plant_csv`'s ``extra``."""

POSITION_ERROR_M = 0.5
"""A sample error bound in meters a project states for its registry positions in the canopy
regime."""

PHOTO_POSITION_ERROR_M = 1.0
"""A sample horizontal positioning error (meters) a capture's EXIF reports, under the spacing of
:data:`PLOTS`."""


def write_plant_csv(
    path: Path, plants: list[dict], *, extra: tuple[str, ...] = ()
) -> Path:
    """Write a plant-locations CSV at ``path``, one row per entry of ``plants``.

    Each row carries its ``plot``, ``accession``, ``lon`` and ``lat`` under the columns
    ``PLANT_CSV_COLUMN_FOR_FIELD`` names for ``plant_id``, ``accession_name``, ``lon`` and
    ``lat``, then each ``extra`` column it carries under that name. Returns ``path``.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        PLANT_CSV_COLUMN_FOR_FIELD,
    )

    column = PLANT_CSV_COLUMN_FOR_FIELD
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                column["plant_id"],
                column["accession_name"],
                column["lon"],
                column["lat"],
                *extra,
            ]
        )
        for p in plants:
            w.writerow(
                [
                    p["plot"],
                    p["accession"],
                    p["lon"],
                    p["lat"],
                    *(p[c] for c in extra),
                ]
            )
    return path


def register_plant_registry_for(
    project: Path,
    csv_paths: list[str | Path],
    *,
    name: str = "reg",
    crop: str = "currant",
    site: str = "orchard",
) -> str:
    """Register ``csv_paths`` under ``name`` in ``project`` and return ``name``.

    The registration goes through ``register_plant_registry`` at ``crop`` and ``site``,
    asserting it returned no error.
    """
    from tcip_mcp.tools.phenology_tools import register_plant_registry

    res = register_plant_registry(
        project,
        name=name,
        csv_paths=[str(p) for p in csv_paths],
        crop=crop,
        site=site,
        actor=None,
    )
    assert "error" not in res, res
    return name


def plant_rows(
    plots: Mapping[str, tuple[float, float]] = PLOTS,
) -> list[dict]:
    """Return one plant row per entry of ``plots``: its plot, ``acc-<plot>`` and position."""
    return [
        {"plot": plot, "accession": f"acc-{plot}", "lat": lat, "lon": lon}
        for plot, (lat, lon) in plots.items()
    ]


PLANTS = plant_rows()
"""The plant rows of :data:`PLOTS` (:func:`plant_rows`)."""


def write_plots_csv(
    path: Path, plots: Mapping[str, tuple[float, float]] = PLOTS
) -> Path:
    """Write ``plots`` as a plant-locations CSV at ``path`` (:func:`plant_rows`)."""
    return write_plant_csv(path, plant_rows(plots))


def write_photos(
    images_dir: Path,
    photos: Mapping[str, tuple[float, float, float | None]],
    start: datetime,
    step: timedelta,
) -> list[Path]:
    """Write one geotagged photograph per entry of ``photos`` in ``images_dir``, in order.

    Each entry maps the file name to the ``(lat, lon, reported positioning error)`` that
    :func:`~tests._image_fixtures.write_geo_image` writes, the photographs taken ``step`` apart
    from ``start``. Returns their paths.
    """
    from tests._image_fixtures import write_geo_image

    paths = []
    for i, (name, (lat, lon, error)) in enumerate(photos.items()):
        paths.append(images_dir / name)
        write_geo_image(paths[-1], lat, lon, start + i * step, h_pos_err=error)
    return paths


def geolocated_captures(
    dataset_root: Path, dates: list[str], plants: list[dict] = PLANTS
) -> dict[str, list[Path]]:
    """Write one geolocated image per row of ``plants`` on each of ``dates`` (:func:`write_photos`).

    Each is ``images/<date>/<plot>_<yyyymmdd>.jpg`` under ``dataset_root``, a minute apart from
    09:30, reporting :data:`PHOTO_POSITION_ERROR_M`. Returns each date's images.
    """
    return {
        date: write_photos(
            image_dir(dataset_root, date),
            {
                f"{plant['plot']}_{date.replace('-', '')}.jpg": (
                    plant["lat"],
                    plant["lon"],
                    PHOTO_POSITION_ERROR_M,
                )
                for plant in plants
            },
            datetime.fromisoformat(date).replace(hour=9, minute=30),
            timedelta(minutes=1),
        )
        for date in dates
    }


def mapping_over(
    dataset_root: Path,
    plant_csvs: list[Path],
    project: Path,
    *,
    name: str = "mapping",
    dates: list[str] | None = None,
    match_radius: StatedRadius | None = None,
) -> MappingBuild:
    """Return ``build_mapping``'s build ``name`` of ``dataset_root`` against ``plant_csvs``.

    It is ``project``'s, of the dataset ``ds-1``, under a registry reference no registration
    names, over ``dates`` within ``match_radius`` as ``build_mapping`` takes them.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        PlantRegistryReference,
        build_mapping,
    )

    return build_mapping(
        dataset_root,
        plant_csvs,
        name=name,
        dataset_id="ds-1",
        project=project,
        plant_registry=PlantRegistryReference(
            name="unregistered", digest="0" * 64
        ),
        dates=dates,
        match_radius=match_radius,
    )


def map_captures(
    project: Path,
    dataset_root: Path,
    dates: list[str],
    *,
    name: str = "valley",
) -> dict[str, list[Path]]:
    """Register ``dataset_root`` in ``project`` and map geolocated captures under ``name``.

    The captures are :func:`geolocated_captures` on each of ``dates``, mapped against
    :func:`write_plots_csv` (``register_plant_registry`` then ``build_plant_mapping``). Returns
    each date's captures.
    """
    from tcip_mcp.tools.phenology_tools import build_plant_mapping
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.traits import registered_crops

    dataset_root.mkdir(parents=True, exist_ok=True)
    registered = register_dataset(
        project,
        str(dataset_root),
        crop=sorted(registered_crops())[0],
        actor=None,
    )
    assert "error" not in registered, registered
    captures = geolocated_captures(dataset_root, dates)
    plant_csv = write_plots_csv(project / f"{name}_plants.csv")
    built = build_plant_mapping(
        project,
        name=name,
        dataset_root=str(dataset_root),
        plant_registry=register_plant_registry_for(project, [plant_csv]),
        actor=None,
    )
    assert "error" not in built, built
    return captures
