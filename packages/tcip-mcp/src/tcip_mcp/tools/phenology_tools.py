"""Phenology MCP tools for per-plant phenology delivery.

The tools over the canonical ``pipelines.postprocessing`` modules:

    shp_to_plant_csv               a plant-locations shapefile → a plant-locations CSV
    register_plant_registry        plant-locations CSVs → a named, registered registry
    build_plant_mapping            geolocated images + a plant registry → a named mapping under
                                   the project
    deliver_phenology_milestones   that mapping + prediction buckets whose scope declares the
                                   positive state's attribute →
                                   <phenology_prefix>_phenology.csv
"""

from __future__ import annotations

from pathlib import Path

from tcip_mcp.audit import audited
from tcip_mcp.pipelines.postprocessing import phenology
from tcip_mcp.pipelines.postprocessing.plant_mapping import (
    PlantRegistry,
    ShapefileField,
    StatedRadius,
)
from tcip_mcp.registry_paths import located
from tcip_mcp.server import tool


def _validate_round_trip(csv_path: Path, n_written: int) -> int:
    """Read ``csv_path`` back through ``read_plant_csvs`` and fail loudly, naming the cause."""
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        PLANT_CSV_COLUMNS,
        read_plant_csvs,
    )

    parsed = read_plant_csvs([csv_path])
    if len(parsed) == 0:
        raise ValueError(
            f"{csv_path}: wrote {n_written} row(s) but read_plant_csvs parsed 0 plant records back "
            f"from it. The likely cause is a WGS84_centroid_x/WGS84_centroid_y value that failed "
            f"float() parsing, or a header not matching the expected columns {PLANT_CSV_COLUMNS}."
        )
    if len(parsed) != n_written:
        raise ValueError(
            f"{csv_path}: wrote {n_written} row(s) but read_plant_csvs parsed only {len(parsed)} "
            "back; one or more rows' WGS84_centroid_x/WGS84_centroid_y failed float() parsing."
        )
    return len(parsed)


@tool()
@audited
def shp_to_plant_csv(
    project: Path,
    shp_path: str,
    csv_path: str,
    field_map: dict[ShapefileField, str] | None = None,
    *,
    actor: str | None,
) -> dict:
    """Convert a plant-locations shapefile to a CSV in ``read_plant_csvs``' schema, by ``actor``.

    Reads through ``plant_mapping.read_plant_shapefile`` (the CRS refusal, the
    point/polygon/multipolygon geometry rule, the DBF field resolution and truncation): a point's
    own coordinate, or a polygon's centroid, reprojected to WGS84. Writes its rows under
    ``PLANT_CSV_COLUMNS``, the header ``read_plant_csv_bytes`` reads, and reads the written CSV
    back through ``read_plant_csvs`` before answering.

    Args:
        shp_path: The point or polygon shapefile (:func:`~tcip_mcp.registry_paths.located`
            against the project).
        csv_path: Where the CSV is written (located against the project).
        field_map: Each pass-through record field (``plant_mapping.ShapefileField``) to the
            source attribute it is read from, where the shapefile names it otherwise.

    Returns ``{csv_path, n_features, n_parsed, geometry_kinds, missing_fields,
    skipped_null_geometry}``: ``n_features`` the rows written, which excludes the null-geometry
    features ``skipped_null_geometry`` counts, and ``missing_fields`` every column with no source
    attribute, written empty. Refuses (``{"error": ...}``) a source with no resolvable CRS, a
    feature whose geometry is neither point, polygon nor multipolygon, and zero features with
    readable geometry, each before writing, and a written CSV that fails to round-trip.

    """
    import csv

    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        PLANT_CSV_COLUMNS,
        read_plant_shapefile,
    )

    source, target = located(shp_path, project), located(csv_path, project)
    try:
        result = read_plant_shapefile(source, field_map=field_map)
    except ValueError as exc:
        return {"error": str(exc)}
    if not result.rows:
        return {
            "error": f"{source}: zero features with readable geometry; nothing to convert."
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PLANT_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(result.rows)
    try:
        n_parsed = _validate_round_trip(target, len(result.rows))
    except ValueError as exc:
        return {"error": str(exc)}
    return {
        "csv_path": str(target),
        "n_features": len(result.rows),
        "n_parsed": n_parsed,
        "geometry_kinds": result.geometry_kinds,
        "missing_fields": result.missing_fields,
        "skipped_null_geometry": result.skipped_null_geometry,
    }


@tool()
@audited
def register_plant_registry(
    project: Path,
    name: str,
    csv_paths: list[str],
    *,
    crop: str,
    site: str,
    actor: str | None,
) -> PlantRegistry | dict[str, str]:
    """Register a plant-locations CSV set under a name in this project, by ``actor``.

    Reads every path in ``csv_paths`` through ``read_plant_csvs``, one file at a time, and refuses
    naming any file that parses to no georeferenced, named plant. The stored, frozen,
    project-scoped record holds each file's ``{path, sha256, n_plants}``, ``crop``, ``site``,
    ``registered_at``, and a content digest over every parsed row.

    A second registration under a taken ``name`` is read before it writes (this store's own
    ``concurrency="cas"``): it returns the existing record unchanged when the digest matches (the
    same plants again, from a different path or row order), and refuses naming both digests when it
    does not.

    Args:
        name: The registry's name within this project (``plant_mapping.NAME_SEGMENT``:
            lowercase letters, digits, single hyphens).
        csv_paths: One or more plant-locations CSVs (columns ``plant_id``, ``accession_name``,
            ``WGS84_centroid_x/y``, …).
        crop: The crop these plants are of. The expert's fact, never inferred.
        site: The site or block these plants are of. The expert's fact, never inferred.

    Refuses (``{"error": ...}``) naming any missing file, naming any shapefile part by suffix
    (``.shp``/``.shx``/``.dbf``/``.prj``; convert it first with :func:`shp_to_plant_csv`), naming
    any file that parsed no georeferenced, named plant or is not UTF-8 text, and naming a name
    conflict's two digests or a name outside ``NAME_SEGMENT``. This does not require a project
    record.

    """
    from tcip_mcp.pipelines.postprocessing import plant_mapping

    paths = [located(p, project) for p in csv_paths]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        return {"error": f"plant CSV(s) not found: {missing}"}

    shapefile_suffixes = {".shp", ".shx", ".dbf", ".prj"}
    shapefile_paths = [
        str(p) for p in paths if p.suffix.casefold() in shapefile_suffixes
    ]
    if shapefile_paths:
        return {
            "error": (
                f"{shapefile_paths} look like shapefile parts, not plant-locations CSVs (this "
                "registry re-verifies and re-parses each file from its own one-snapshot bytes at "
                "delivery, which a multi-file shapefile has no single-file form for); convert "
                "it with shp_to_plant_csv and register the CSV instead"
            )
        }

    try:
        return plant_mapping.register_plant_registry_record(
            project, name, paths, crop=crop, site=site
        )
    except (
        plant_mapping.NoGeoreferencedPlantsError,
        plant_mapping.PlantRegistryNameConflictError,
        ValueError,
    ) as exc:
        return {"error": str(exc)}


@tool()
def build_plant_mapping(
    project: Path,
    name: str,
    dataset_root: str,
    plant_registry: str,
    dates: list[str] | None = None,
    match_radius: StatedRadius | None = None,
    supersede: bool = False,
    *,
    actor: str | None,
) -> dict:
    """Assign each geolocated image to a plant, then persist the mapping under this project.

    The build is ``actor``'s.

    Orders each date's images by EXIF capture time (the walker's sequence), splits into row runs on
    large GPS jumps, and assigns along the row, falling back to nearest-neighbor when the sequence
    signal is weak, both within the date's match radius. Each assignment records its ``source``
    and GPS ``distance_m``. The mapping is project state, persisted
    under the project by ``name``.

    Args:
        name: The mapping's name within this project (``plant_mapping_key``'s own naming rule). A
            rebuild under the same name replaces this project's own mapping of that name.
        dataset_root: The registered dataset whose captures are mapped.
        plant_registry: The name of a plant registry already registered under this project by
            ``register_plant_registry``.
        dates: Optional subset of the dataset's captures to map (default: all of them).
        match_radius: ``{value, rationale}``, the radius (meters) every date's captures are
            matched within and how it is known. Omitted, each date's radius is the largest
            horizontal positioning error its photographs' EXIF reports; a date whose positioned
            photographs report none refuses naming this argument.
        supersede: A rebuild under ``name`` whose current record is still cited by a delivery event
            under this project refuses by name, listing the citing events, unless this is ``True``:
            the current record is then archived first (never overwritten), the new record's own
            ``supersedes`` names the archived digest, and the archived record stays readable by
            that digest. An uncited rebuild replaces, ignoring this.

    Refuses (a plain ``{"error": ...}``) with each refusal of
    :func:`~tcip_mcp.pipelines.postprocessing.plant_mapping.build_plant_mapping`; a rebuild a
    delivery event still cites, with ``supersede`` left ``False``, also names those events as
    ``citing_events``. A receipt that cannot be written fails the call naming the receipt: the
    record it would have named is left on disk but
    :func:`~tcip_mcp.pipelines.postprocessing.plant_mapping.load_mapping` refuses to read it until
    a rebuild replaces it.

    Returns the build as ``MappingBuild.served`` states it, never the full per-image mapping (that
    lives in the persisted record).

    """
    from tcip_mcp.audit import AuditEntryNotWrittenError
    from tcip_mcp.pipelines.postprocessing import plant_mapping

    try:
        return plant_mapping.build_plant_mapping(
            project,
            name,
            located(dataset_root, project),
            plant_registry,
            dates=dates,
            match_radius=match_radius,
            supersede=supersede,
            actor=actor,
        ).served()
    except plant_mapping.MappingRebuildError as exc:
        return {"error": str(exc), "citing_events": exc.event_ids}
    except (
        plant_mapping.UngeoreferencedCaptureError,
        AuditEntryNotWrittenError,
        ValueError,
    ) as exc:
        return {"error": str(exc)}


@tool()
def deliver_phenology_milestones(
    project: Path,
    trait: str,
    mapping_name: str,
    dataset_root: str,
    buckets: list[str],
    output_csv_path: str,
    plants: list[str],
    acknowledgment_id: str | None = None,
    *,
    actor: str | None,
) -> dict:
    """Deliver per-plant phenology milestones from prediction buckets and a plant mapping.

    The buckets are ones whose scope declares the positive state's attribute; the delivery is
    ``actor``'s.

    A phenology milestone is a crossing of the fraction of a plant's detected objects a classifier
    calls the trait's positive class, read under the crossing rule the trait's latest confirmed
    revision states. The CSV carries each milestone date that revision declares and its
    evidentiary bound, one row per plant in ``plants``, a plant with any incomplete date carrying
    none, and the delivery's ``validated``, ``trait_revision`` and ``delivery_event_id``. Every
    bucket must have been published under an assessment that answers for a
    ``state_crossing_dates`` delivery of that revision, or the delivery ships only under
    ``acknowledgment_id``, a breeder's recorded acknowledgment of exactly this result, which this
    door executes and never records. The buckets must name one producer.

    Args:
        trait: A trait in this project.
        mapping_name: A plant mapping persisted under this project.
        dataset_root: The dataset root the buckets are published under.
        buckets: The name of the published bucket of each delivered date, one per date; each
            stands for the capture date its own record states.
        output_csv_path: Where to write the CSV.
        plants: The delivery's population, the mapping's ``plant_id`` values it is for; an empty
            list refuses.
        acknowledgment_id: A breeder's recorded acknowledgment of this unvalidated result.

    """
    from tcip_mcp.delivery import DeliveryRefusedError
    from tcip_mcp.operationalization import OperationalizationRefusedError
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        MappingDeliveryError,
    )
    from tcip_mcp.subject_registry import RegistryError
    from tcip_mcp.traits import TraitUnknownError

    try:
        measurement = phenology.measure_phenology(
            project,
            trait=trait,
            mapping_name=mapping_name,
            dataset_root=located(dataset_root, project),
            buckets=buckets,
            plants=plants,
        )
        delivered = phenology.deliver_phenology(
            project,
            measurement,
            curves=False,
            output_path=located(output_csv_path, project),
            acknowledgment_id=acknowledgment_id,
            door="deliver_phenology_milestones",
            actor=actor,
        )
    except phenology.measurement_refusals() as exc:
        return {"error": str(exc)}
    except (
        DeliveryRefusedError,
        OperationalizationRefusedError,
        TraitUnknownError,
        RegistryError,
        MappingDeliveryError,
        ValueError,
    ) as exc:
        return {"error": str(exc)}
    disclosure = measurement.plant_mapping
    return {
        **delivered,
        "n_plants": len(measurement.rows),
        "n_images_unattributed": disclosure["images_unattributed"],
        "dates_delivered": disclosure["dates_delivered"],
        "captures_unverified": disclosure["captures_unverified"],
        "plant_csvs_unverified": disclosure["plant_csvs_unverified"],
    }
