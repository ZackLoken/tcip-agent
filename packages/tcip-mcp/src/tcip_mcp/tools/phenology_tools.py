"""Phenology MCP tools, the agent-facing surface for the per-plant phenology pipeline.

Three tools over the canonical ``pipelines.postprocessing`` modules:

    register_plant_registry        plant-locations CSVs → a named, registered registry
    build_plant_mapping            geolocated images + a plant registry → a named mapping under
                                   the project
    deliver_phenology_milestones   that mapping + prediction buckets whose scope declares the
                                   positive state's attribute →
                                   <phenology_prefix>_phenology.csv

See the ``phenology`` skill for the whole pattern (isolate → detect → classify state → per-plant
fraction → crossings).
"""

from __future__ import annotations

from pathlib import Path

from tcip_mcp.audit import audited
from tcip_mcp.pipelines.postprocessing import phenology
from tcip_mcp.server import tool


@tool()
@audited
def register_plant_registry(project: Path, name: str, csv_paths: list[str], *, crop: str,
                            site: str) -> dict:
    """Register a plant-locations CSV set under a name that later doors cite instead of file paths.

    Reads every path in ``csv_paths`` through ``read_plant_csvs``, one file at a time, and refuses
    naming any file that parses to no georeferenced, named plant. The stored, frozen,
    project-scoped record holds each file's ``{path, sha256, n_plants}``, ``crop``, ``site``,
    ``registered_at``, and a content digest over every parsed row.

    A second registration under a taken ``name`` is read before it writes (this store's own
    ``concurrency="cas"``): it returns the existing record unchanged when the digest matches (the
    same plants again, from a different path or row order), and refuses naming both digests when it
    does not.

    Args:
        name: The registry's name within this project (``tcip_store.layout_claims.NAME_SEGMENT``:
            lowercase letters, digits, single hyphens).
        csv_paths: One or more plant-locations CSVs (columns ``plot_name``, ``accession_name``,
            ``WGS84_centroid_x/y``, …).
        crop: The crop these plants are of. The expert's fact, never inferred.
        site: The site or block these plants are of. The expert's fact, never inferred.

    Refuses (``{"error": ...}``) naming any missing file, naming any shapefile part by suffix
    (``.shp``/``.shx``/``.dbf``/``.prj``; convert it first with ``tcip shp-to-plant-csv``), naming
    any file that parsed no georeferenced, named plant or is not UTF-8 text, and naming a name
    conflict's two digests or a name outside ``NAME_SEGMENT``. This does not require a project
    record.
    """
    from tcip_mcp.pipelines.postprocessing import plant_mapping

    missing = [p for p in csv_paths if not Path(p).is_file()]
    if missing:
        return {"error": f"plant CSV(s) not found: {missing}"}

    shapefile_suffixes = {".shp", ".shx", ".dbf", ".prj"}
    shapefile_paths = [p for p in csv_paths if Path(p).suffix.casefold() in shapefile_suffixes]
    if shapefile_paths:
        return {"error": (
            f"{shapefile_paths} look like shapefile parts, not plant-locations CSVs (this "
            "registry re-verifies and re-parses each file from its own one-snapshot bytes at "
            "delivery, which a multi-file shapefile has no single-file form for); run "
            "tcip shp-to-plant-csv <plants.shp> <plants.csv> and register the CSV instead")}

    try:
        return plant_mapping.register_plant_registry_record(
            project, name, [Path(p) for p in csv_paths], crop=crop, site=site)
    except (plant_mapping.NoGeoreferencedPlantsRefusal, plant_mapping.PlantRegistryNameConflict,
            ValueError) as exc:
        return {"error": str(exc)}


@tool()
def build_plant_mapping(
    project: Path,
    name: str,
    images_root: str,
    plant_registry: str,
    dates: list[str] | None = None,
    nn_tolerance_m: float | None = None,
    supersede: bool = False,
) -> dict:
    """Assign each geolocated image to a plant, then persist the mapping under this project.

    Orders each date's images by EXIF capture time (the walker's sequence), splits into row runs on
    large GPS jumps, and assigns along the row, falling back to nearest-neighbor when the sequence
    signal is weak. Each assignment records its ``source`` and GPS ``distance_m`` (no fabricated
    "confidence"). The mapping is project state, persisted under the project by ``name``.

    Args:
        name: The mapping's name within this project (``plant_mapping_key``'s own naming rule). A
            rebuild under the same name replaces this project's own mapping of that name.
        images_root: Directory whose immediate subfolders are ``<YYYY-MM-DD>/`` image buckets (the
            ingest layout).
        plant_registry: The name of a plant registry already registered under this project by
            ``register_plant_registry``.
        dates: Optional subset of date folders to map (default: all under ``images_root``).
        nn_tolerance_m: Nearest-neighbor tolerance (m). ``None`` (default) derives it from the
            plot's grid pitch (pitch/6) so the match radius stays within half a grid cell; an
            explicit value is honored but still capped at that pitch-derived ceiling.
        supersede: A rebuild under ``name`` whose current record is still cited by a delivery event
            under this project refuses by name, listing the citing events, unless this is ``True``:
            the current record is then archived first (never overwritten), the new record's own
            ``supersedes`` names the archived digest, and the archived record stays readable by
            that digest. An uncited rebuild replaces as it always has, ignoring this.

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
    from tcip_mcp.audit import AuditEntryNotWritten
    from tcip_mcp.pipelines.postprocessing import plant_mapping

    try:
        return plant_mapping.build_plant_mapping(
            project, name, images_root, plant_registry, dates=dates,
            nn_tolerance_m=nn_tolerance_m, supersede=supersede).served()
    except plant_mapping.MappingRebuildRefusal as exc:
        return {"error": str(exc), "citing_events": exc.event_ids}
    except (plant_mapping.UngeoreferencedCaptureRefusal, AuditEntryNotWritten,
            ValueError) as exc:
        return {"error": str(exc)}


@tool()
def deliver_phenology_milestones(
    project: Path,
    trait: str,
    mapping_name: str,
    buckets: list[str],
    output_csv_path: str,
    plants: list[str],
    require_all_dates_complete: bool = phenology.REQUIRE_ALL_DATES_COMPLETE,
    acknowledgment_id: str | None = None,
) -> dict:
    """Per-plant phenology milestones from prediction buckets whose scope declares the positive
    state's attribute, and a plant mapping.

    A phenology milestone is a crossing of the fraction of a plant's detected objects a classifier
    calls the trait's positive class. The CSV carries each milestone date the trait's latest
    confirmed revision declares and its evidentiary bound, one row per plant in ``plants``, and
    the delivery's ``validated``, ``trait_revision`` and ``delivery_event_id``. Every bucket must
    have been published under an assessment that answers for a ``state_crossing_dates`` delivery
    of that revision, or the delivery ships only under ``acknowledgment_id``, a breeder's recorded
    acknowledgment of exactly this result, which this door executes and never records. The buckets
    must name one producer.

    Args:
        trait: A trait in this project.
        mapping_name: A plant mapping persisted under this project.
        buckets: The published bucket of each delivered date, one per date; each stands for the
            capture date its own record states.
        output_csv_path: Where to write the CSV; a relative path is under the project.
        plants: The delivery's population, the plant ids (the mapping's ``plot_name`` values) it
            is for; an empty list refuses.
        require_all_dates_complete: Compute a plant's milestones only when every one of its dates
            carries the positive state's attribute on every detection and is fully observed;
            ``False`` computes them from its complete dates alone. Defaults to ``phenology.REQUIRE_ALL_DATES_COMPLETE``; recorded on the delivery
            event.
        acknowledgment_id: A breeder's recorded acknowledgment of this unvalidated result.
    """
    from tcip_mcp.delivery import DeliveryRefused
    from tcip_mcp.operationalization import OperationalizationRefused
    from tcip_mcp.pipelines.postprocessing.plant_mapping import MappingDeliveryRefusal
    from tcip_mcp.subject_registry import RegistryError
    from tcip_mcp.traits import TraitUnknownError

    try:
        measurement = phenology.measure_phenology(
            project, trait=trait, mapping_name=mapping_name,
            buckets=[Path(project, b) for b in buckets], plants=plants,
            require_all_dates_complete=require_all_dates_complete)
        delivered = phenology.deliver_phenology(
            project, measurement, curves=False, output_path=Path(project, output_csv_path),
            acknowledgment_id=acknowledgment_id, door="deliver_phenology_milestones")
    except phenology.measurement_refusals() as exc:
        return {"error": str(exc)}
    except (DeliveryRefused, OperationalizationRefused, TraitUnknownError, RegistryError,
            MappingDeliveryRefusal, ValueError) as exc:
        return {"error": str(exc)}
    disclosure = measurement.plant_mapping
    return {**delivered, "n_plants": len(measurement.rows),
            "n_images_unattributed": disclosure["images_unattributed"],
            "dates_delivered": disclosure["dates_delivered"],
            "captures_unverified": disclosure["captures_unverified"],
            "plant_csvs_unverified": disclosure["plant_csvs_unverified"]}
