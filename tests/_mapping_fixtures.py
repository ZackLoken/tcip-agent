"""Plant registries and plant mappings a test registers through the platform's own producers."""

from __future__ import annotations

import csv
from pathlib import Path

PLOTS = {"P1": (43.19670, -90.058000), "P2": (43.19670, -90.058037)}
"""Two plots one row apart, the plant grid :func:`map_captures` places captures on."""

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

    Each row carries its ``plot``, ``accession``, ``lon`` and ``lat`` under ``plant_id``,
    ``accession_name``, ``WGS84_centroid_x`` and ``WGS84_centroid_y``, then each ``extra`` column
    it carries under that name. Returns ``path``.
    """
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "plant_id",
                "accession_name",
                "WGS84_centroid_x",
                "WGS84_centroid_y",
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


def map_captures(
    project: Path,
    dataset_root: Path,
    dates: list[str],
    *,
    name: str = "valley",
) -> dict[str, list[Path]]:
    """Register ``dataset_root`` in ``project`` and map geolocated captures under ``name``.

    :func:`~tests._image_fixtures.write_geo_image` writes one image per entry of :data:`PLOTS` on
    each of ``dates`` at ``images/<date>/<plot>_<yyyymmdd>.jpg`` under the root, and they are mapped
    (``register_plant_registry`` then ``build_plant_mapping``). Returns each date's captures.
    """
    from datetime import datetime, timedelta

    from tcip_mcp.tools.phenology_tools import build_plant_mapping
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.traits import registered_crops
    from tests._image_fixtures import write_geo_image

    dataset_root.mkdir(parents=True, exist_ok=True)
    registered = register_dataset(
        project,
        str(dataset_root),
        crop=sorted(registered_crops())[0],
        actor=None,
    )
    assert "error" not in registered, registered
    captures: dict[str, list[Path]] = {}
    for date in dates:
        when = datetime.fromisoformat(date).replace(hour=9, minute=30)
        captures[date] = []
        for j, (plot, (lat, lon)) in enumerate(PLOTS.items()):
            image = (
                dataset_root
                / "images"
                / date
                / f"{plot}_{date.replace('-', '')}.jpg"
            )
            write_geo_image(
                image,
                lat,
                lon,
                when + timedelta(minutes=j),
                h_pos_err=PHOTO_POSITION_ERROR_M,
            )
            captures[date].append(image)
    plant_csv = write_plant_csv(
        project / f"{name}_plants.csv",
        [
            {"plot": plot, "accession": f"acc-{plot}", "lat": lat, "lon": lon}
            for plot, (lat, lon) in PLOTS.items()
        ],
    )
    built = build_plant_mapping(
        project,
        name=name,
        images_root=str(dataset_root / "images"),
        plant_registry=register_plant_registry_for(project, [plant_csv]),
        actor=None,
    )
    assert "error" not in built, built
    return captures
