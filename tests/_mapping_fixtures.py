"""Plant registries and plant mappings a test registers through the platform's own producers."""

from __future__ import annotations

import csv
from pathlib import Path

PLOTS = {"P1": (43.19670, -90.058000), "P2": (43.19670, -90.058037)}
"""Two plots one row apart, the plant grid :func:`map_captures` places captures on."""

GRID_COLUMNS = ("plot_number", "row_number", "col_number")
"""The plant-grid position columns a breeding layout's registry CSV carries beside each plant's
location, written as :func:`write_plant_csv`'s ``extra``."""


def write_plant_csv(path: Path, plants: list[dict], *, extra: tuple[str, ...] = ()) -> Path:
    """A plant-locations CSV at ``path``, one row per entry of ``plants``: its ``plot``,
    ``accession``, ``lat`` and ``lon`` under the registry's column names, then each ``extra``
    column it carries under that name. Returns ``path``."""
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["plot_name", "accession_name", "WGS84_centroid_x", "WGS84_centroid_y", *extra])
        for p in plants:
            w.writerow([p["plot"], p["accession"], p["lon"], p["lat"], *(p[c] for c in extra)])
    return path


def register_plant_registry_for(
    project: Path, csv_paths: list[str | Path], *, name: str = "reg", crop: str = "currant",
    site: str = "orchard",
) -> str:
    """Register ``csv_paths`` under ``name`` in ``project`` through ``register_plant_registry``,
    and return ``name``. Idempotent under the same content: a second call with the same paths
    under the same name is a no-op.
    """
    from tcip_mcp.tools.phenology_tools import register_plant_registry

    res = register_plant_registry(
        project, name=name, csv_paths=[str(p) for p in csv_paths], crop=crop, site=site)
    assert "error" not in res, res
    return name


def map_captures(project: Path, dataset_root: Path, dates: list[str], *,
                 name: str = "valley") -> dict[str, list[Path]]:
    """Register ``dataset_root`` in ``project``, write one geolocated capture of each of
    :data:`PLOTS` on each of ``dates`` under its ``images/``, and map them under ``name``
    (``register_plant_registry`` then ``build_plant_mapping``); each date's captures."""
    from datetime import datetime, timedelta

    from tcip_mcp.tools.phenology_tools import build_plant_mapping
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.traits import registered_crops
    from tests._image_fixtures import write_geo_image

    dataset_root.mkdir(parents=True, exist_ok=True)
    registered = register_dataset(project, str(dataset_root), crop=sorted(registered_crops())[0])
    assert "error" not in registered, registered
    captures: dict[str, list[Path]] = {}
    for date in dates:
        when = datetime.fromisoformat(date).replace(hour=9, minute=30)
        captures[date] = []
        for j, (plot, (lat, lon)) in enumerate(PLOTS.items()):
            image = dataset_root / "images" / date / f"{plot}_{date.replace('-', '')}.jpg"
            write_geo_image(image, lat, lon, when + timedelta(minutes=j))
            captures[date].append(image)
    plant_csv = write_plant_csv(project / f"{name}_plants.csv", [
        {"plot": plot, "accession": f"acc-{plot}", "lat": lat, "lon": lon}
        for plot, (lat, lon) in PLOTS.items()])
    built = build_plant_mapping(project, name=name, images_root=str(dataset_root / "images"),
                                plant_registry=register_plant_registry_for(project, [plant_csv]))
    assert "error" not in built, built
    return captures
