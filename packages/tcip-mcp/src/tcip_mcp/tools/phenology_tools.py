"""Phenology MCP tools — the agent-facing surface for the per-plant bloom pipeline.

Two composable steps over the canonical ``pipelines.postprocessing`` modules, so the agent
composes tools instead of scripting into the web backend (and a milestone date means exactly
what it means in the web Results tab):

    build_plant_mapping   geolocated images + plant CSVs → persisted plant_mapping.json
    compute_phenology     that mapping + classified predictions → catkin_phenology.csv

See the ``phenology`` skill for the whole pattern (isolate → detect → classify elongation →
per-plant fraction → crossings).
"""

from __future__ import annotations

import json
from pathlib import Path

from tcip_mcp.audit import audited
from tcip_mcp.pipelines.postprocessing import phenology
from tcip_mcp.server import mcp


@mcp.tool()
@audited
def build_plant_mapping(
    images_root: str,
    plant_csv_paths: list[str],
    output_mapping_path: str,
    dates: list[str] | None = None,
    nn_tolerance_m: float = 10.0,
) -> dict:
    """Assign each geolocated image to a plant, then persist the mapping for phenology.

    Image GPS (handheld EXIF) carries ~5 m error while the plant grid is ~2.8 m between
    adjacent plots, so nearest-neighbour GPS alone is ambiguous. This orders each date's
    images by EXIF capture time (the walker's sequence), splits into row runs on large GPS
    jumps, and assigns along the row — falling back to nearest-neighbour when the sequence
    signal is weak. Each assignment records its ``source`` and GPS ``distance_m`` (no
    fabricated "confidence"). The persisted ``plant_mapping.json`` is what ``compute_phenology``
    consumes. See the ``phenology`` skill.

    Args:
        images_root: Directory whose immediate subfolders are ``<YYYY-MM-DD>/`` image buckets
            (the ingest layout).
        plant_csv_paths: One or more plant-locations CSVs (columns ``plot_name``,
            ``accession_name``, ``WGS84_centroid_x/y``, …).
        output_mapping_path: Where to persist the mapping JSON (e.g.
            ``<project>/.tcip/state/plant_mapping.json``).
        dates: Optional subset of date folders to map (default: all under ``images_root``).
        nn_tolerance_m: Nearest-neighbour tolerance (m); larger allows looser GPS matches.

    Returns a compact per-date summary (images, mapped count, avg GPS distance) plus totals
    and the persisted path — not the full per-image mapping (that lives in the JSON).
    """
    from tcip_mcp.pipelines.postprocessing import plant_mapping

    root = Path(images_root)
    if not root.is_dir():
        return {"error": f"images_root not found: {images_root}"}
    missing = [p for p in plant_csv_paths if not Path(p).is_file()]
    if missing:
        return {"error": f"plant CSV(s) not found: {missing}"}

    mapping = plant_mapping.build_mapping(
        root,
        [Path(p) for p in plant_csv_paths],
        dates=dates,
        nn_tolerance_m=nn_tolerance_m,
    )
    if not mapping:
        return {"error": f"no date folders with images under {images_root}"}

    plant_mapping.persist_mapping(mapping, Path(output_mapping_path))

    per_date: dict[str, dict] = {}
    total_images = 0
    total_mapped = 0
    for date_str, assignments in mapping.items():
        n_images = len(assignments)
        n_mapped = sum(1 for a in assignments if a.plot_name)
        dists = [a.distance_m for a in assignments if a.distance_m is not None]
        per_date[date_str] = {
            "n_images": n_images,
            "n_mapped": n_mapped,
            "avg_distance_m": (round(sum(dists) / len(dists), 2) if dists else None),
        }
        total_images += n_images
        total_mapped += n_mapped

    return {
        "mapping_path": str(output_mapping_path),
        "n_dates": len(mapping),
        "n_images": total_images,
        "n_mapped": total_mapped,
        "n_unmapped": total_images - total_mapped,
        "per_date": per_date,
    }


@mcp.tool()
@audited
def compute_phenology(
    mapping_path: str,
    predictions_by_date: dict[str, str],
    output_csv_path: str,
    elongated_class_id: int = 1,
) -> dict:
    """Per-plant catkin bloom milestones from classified predictions + a plant mapping.

    Bloom is the **fraction of a plant's detected catkins that are elongated** — where
    "elongated" is an expert-defined morphological stage emitted by a *validated* 2-class
    classifier (class ``elongated_class_id``), never a geometric proxy such as bounding-box
    height. For each plant this reports:

        catkin_elongation_date   first date any elongation appears (fraction > 0)
        catkin_05/50/95per_date  dates the elongated fraction crosses 5/50/95%

    Args:
        mapping_path: Path to a persisted plant-mapping JSON (``{date: [assignment, ...]}``
            with ``stem`` / ``plot_name`` / ``accession_name`` per assignment) — produced by
            the web plant-mapping step or ``run_matching``.
        predictions_by_date: ``{date: predictions_dir}`` — each dir holds YOLO ``.txt``
            prediction files (one per image ``stem``) from the elongation classifier.
        output_csv_path: Where to write the delivered per-plant ``catkin_phenology.csv``.
        elongated_class_id: Class id the classifier assigns to "elongated" (default 1).

    Returns a summary. **Measurement-integrity guard:** if the predictions carry no
    elongation class anywhere, the elongated fraction is not a valid bloom measurement — the
    tool refuses to write the CSV and returns ``error`` with ``elongation_classified: false``
    so an unvalidated curve is never delivered (see the CLAUDE.md measurement-integrity
    invariant).
    """
    mp = Path(mapping_path)
    if not mp.is_file():
        return {"error": f"mapping not found: {mapping_path}"}
    try:
        mapping = json.loads(mp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return {"error": f"could not read mapping {mapping_path}: {e}"}
    if not isinstance(mapping, dict) or not mapping:
        return {"error": f"mapping at {mapping_path} is empty or malformed"}

    result = phenology.per_plant_phenology(
        mapping, predictions_by_date, elongated_class_id=elongated_class_id
    )
    rows = result["rows"]

    if not result["elongation_classified"]:
        return {
            "error": (
                "predictions carry no elongation class "
                f"(class {elongated_class_id}); classes seen: {result['classes_seen']}. "
                "The elongated fraction is not a valid bloom measurement — run and validate "
                "the 2-class elongation classifier before computing phenology."
            ),
            "elongation_classified": False,
            "classes_seen": result["classes_seen"],
            "n_plants": len(rows),
        }

    csv_path = phenology.write_phenology_csv(rows, Path(output_csv_path))
    n_with_50 = sum(1 for r in rows if r.get("catkin_50per_date"))
    return {
        "csv_path": csv_path,
        "n_plants": len(rows),
        "n_plants_reached_50per": n_with_50,
        "elongation_classified": True,
        "classes_seen": result["classes_seen"],
        "columns": phenology.PHENOLOGY_CSV_COLUMNS,
    }
