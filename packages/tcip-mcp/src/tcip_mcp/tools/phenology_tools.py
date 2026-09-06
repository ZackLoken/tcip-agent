"""Phenology MCP tools, the agent-facing surface for the per-plant phenology pipeline.

Three tools over the canonical ``pipelines.postprocessing`` modules, so the agent composes tools
instead of scripting into the web backend (and a milestone date means exactly what it means in
the web Results tab):

    build_plant_mapping                   geolocated images + plant CSVs → a named mapping
                                           under the project
    calibrate_classifier_operating_point   the positive-state classifier's own held-out
                                           validation gate, distinct from the count operating
                                           point run_inference calibrates
    deliver_phenology_milestones           that mapping + classified predictions →
                                           <phenology_prefix>_phenology.csv

See the ``phenology`` skill for the whole pattern (isolate → detect → classify state →
per-plant fraction → crossings).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from tcip_mcp.audit import audited
from tcip_mcp.pipelines.postprocessing import phenology
from tcip_mcp.server import mcp


@mcp.tool()
@audited
def register_plant_registry(name: str, csv_paths: list[str], *, crop: str, site: str) -> dict:
    """Register a plant-locations CSV set under a name, so ``build_plant_mapping`` and
    ``deliver_orthomosaic_plant_counts`` name which version of a plant list they read rather than
    re-asserting file paths on every call.

    Reads every path in ``csv_paths`` through the same parser those two doors already use
    (``read_plant_csvs``), one file at a time, and refuses naming any file that parses to no
    georeferenced, named plant. The stored, frozen, project-scoped record holds each file's
    ``{path, sha256, n_plants}``, ``crop``, ``site``, ``registered_by`` (always this door's own
    identity: no HTTP request reaches this tool, so there is no breeder identity to stamp
    instead), ``registered_at``, and a content digest over every parsed row.

    A second registration under a taken ``name`` is read before it writes (this store's own
    ``concurrency="cas"``): it returns the existing record unchanged when the digest matches (the
    same plants again, from a different path or row order), and refuses naming both digests when
    it does not, since silently moving what a taken name means would strand every mapping already
    citing it.

    Args:
        name: The registry's name within this project (``tcip_store.layout_claims.NAME_SEGMENT``:
            lowercase letters, digits, single hyphens).
        csv_paths: One or more plant-locations CSVs (columns ``plot_name``, ``accession_name``,
            ``WGS84_centroid_x/y``, …).
        crop: The crop these plants are of. The expert's fact, never inferred.
        site: The site or block these plants are of. The expert's fact, never inferred.

    Refuses (``{"error": ...}``) naming any missing file, naming any file that parsed no
    georeferenced plant, and naming a name conflict's two digests. A name outside
    ``NAME_SEGMENT`` refuses at the door. Unlike ``build_plant_mapping``, this does not require a
    project record: ``deliver_orthomosaic_plant_counts`` reads a registry without one, and a
    registry has to be registrable before either door can name it.
    """
    from tcip_store.layout_claims import NAME_SEGMENT

    from tcip_mcp.pipelines.postprocessing import plant_mapping
    from tcip_mcp.project_paths import platform_state_root

    if not NAME_SEGMENT.fullmatch(name):
        return {"error": (
            f"name {name!r} is not lowercase letters, digits and single hyphens "
            f"({NAME_SEGMENT.pattern})")}

    platform_root = platform_state_root()
    missing = [p for p in csv_paths if not Path(p).is_file()]
    if missing:
        return {"error": f"plant CSV(s) not found: {missing}"}

    try:
        record = plant_mapping.register_plant_registry_record(
            platform_root, name, [Path(p) for p in csv_paths],
            crop=crop, site=site, registered_by="register_plant_registry",
        )
    except (plant_mapping.NoGeoreferencedPlantsRefusal,
            plant_mapping.PlantRegistryNameConflict) as exc:
        return {"error": str(exc)}

    return {
        "name": record["name"],
        "project_root": str(platform_root),
        "crop": record["crop"],
        "site": record["site"],
        "n_plants": record["n_plants"],
        "digest": record["digest"],
        "csvs": record["csvs"],
        "registered_by": record["registered_by"],
        "registered_at": record["registered_at"],
    }


@mcp.tool()
@audited
def build_plant_mapping(
    name: str,
    images_root: str,
    plant_registry: str,
    dates: list[str] | None = None,
    nn_tolerance_m: float | None = None,
    supersede: bool = False,
) -> dict:
    """Assign each geolocated image to a plant, then persist the mapping under this project.

    Image GPS (handheld EXIF) carries ~5 m error while the plant grid is ~2.8 m between
    adjacent plots, so nearest-neighbour GPS alone is ambiguous. This orders each date's
    images by EXIF capture time (the walker's sequence), splits into row runs on large GPS
    jumps, and assigns along the row, falling back to nearest-neighbour when the sequence
    signal is weak. Each assignment records its ``source`` and GPS ``distance_m`` (no
    fabricated "confidence"). The mapping is project state, persisted under the resolved
    platform state root by ``name``; ``deliver_phenology_milestones`` reads it back the same way. See the
    ``phenology`` skill.

    Args:
        name: The mapping's name within this project (``plant_mapping_key``'s own naming
            rule). A rebuild under the same name replaces this project's own mapping of that
            name.
        images_root: Directory whose immediate subfolders are ``<YYYY-MM-DD>/`` image buckets
            (the ingest layout).
        plant_registry: The name of a plant registry already registered under this project by
            ``register_plant_registry``.
        dates: Optional subset of date folders to map (default: all under ``images_root``).
        nn_tolerance_m: Nearest-neighbour tolerance (m). ``None`` (default) derives it from the
            plot's grid pitch (pitch/6) so the match radius stays within half a grid cell; an
            explicit value is honored but still capped at that pitch-derived ceiling.
        supersede: A rebuild under ``name`` whose current record is still cited by a delivery
            event under this project refuses by name, listing the citing events, unless this is
            ``True``: the current record is then archived first (never overwritten), the new
            record's own ``supersedes`` names the archived digest, and the archived record stays
            readable by that digest. An uncited rebuild replaces as it always has, ignoring this.

    Refuses (a plain ``{"error": ...}``) naming ``register_dataset`` when ``images_root`` is not
    a registered dataset's own ``images/`` directory, naming ``initialize_project``/
    ``activate_project`` when the resolved platform state root carries no project record, and
    naming ``register_plant_registry`` when ``plant_registry`` names no stored registry. A name
    outside ``tcip_store.layout_claims.NAME_SEGMENT`` (lowercase letters, digits, single hyphens)
    refuses at the door. No capture at all under the requested dates, or captures that carry no
    position this door reads (no GPS EXIF, or a raster/band-group capture), also refuses, naming
    the plant-tag mechanism the platform does not yet have. A receipt that cannot be written fails
    the call naming the receipt: the record it would have named is left on disk but
    :func:`~tcip_mcp.pipelines.postprocessing.plant_mapping.load_mapping` refuses to read it until
    a rebuild replaces it. A rebuild a delivery event still cites, with ``supersede`` left
    ``False``, refuses naming those events.

    Returns a compact per-date summary (images, mapped count, unattributed count, avg GPS
    distance) plus totals, the mapping's ``name``, the resolved ``project_root`` and
    ``dataset_root``, ``nn_tolerance_m`` (the persisted record's own ``{"value": ..., "source":
    ...}``, never recomputed here), ``max_match_distance_m`` (the tolerance's own loosest accepted
    distance, derived from it through ``plant_mapping.match_gates``), and ``unreadable`` (per date,
    the captures PIL could not open), not the full per-image mapping (that lives in the persisted
    record).
    """
    from tcip_store.layout_claims import NAME_SEGMENT

    from tcip_mcp.audit import AuditEntryNotWritten
    from tcip_mcp.dataset_layout import dataset_root_of, image_root, require_dataset_identity
    from tcip_mcp.pipelines.data.splits import same_directory
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem
    from tcip_mcp.pipelines.postprocessing import plant_mapping
    from tcip_mcp.project_paths import platform_state_root
    from tcip_mcp.project_record import ProjectRecordMissing, read_record

    if not NAME_SEGMENT.fullmatch(name):
        return {"error": (
            f"name {name!r} is not lowercase letters, digits and single hyphens "
            f"({NAME_SEGMENT.pattern})")}

    platform_root = platform_state_root()
    try:
        read_record(platform_root)
    except ProjectRecordMissing as exc:
        return {"error": str(exc)}

    resolved_images_root = Path(images_root).resolve()
    if not resolved_images_root.is_dir():
        return {"error": f"images_root not found: {images_root}"}

    candidate = dataset_root_of(resolved_images_root)
    if candidate is None or not same_directory(image_root(candidate), resolved_images_root):
        return {"error": (
            f"{images_root} is not a dataset's own images/ root; build_plant_mapping maps a "
            "registered dataset's image tree")}
    try:
        identity = require_dataset_identity(candidate)
    except ValueError as exc:
        return {"error": str(exc)}

    registry_record = plant_mapping.load_registry(platform_root, plant_registry)
    if registry_record is None:
        return {"error": (
            f"plant registry not found: {plant_registry!r} under {platform_root}; register it "
            "with register_plant_registry before build_plant_mapping reads it")}
    registry_ref = {"name": plant_registry, "digest": registry_record["digest"]}
    registry_paths = [Path(e["path"]) for e in plant_mapping.registry_csv_entries(registry_record)]

    try:
        build = plant_mapping.build_mapping(
            resolved_images_root, registry_paths,
            name=name, dataset_root=candidate, dataset_id=identity["id"],
            project_root=platform_root, built_by="build_plant_mapping",
            plant_registry=registry_ref, dates=dates, nn_tolerance_m=nn_tolerance_m,
        )
    except (AmbiguousImageStem, plant_mapping.UngeoreferencedCaptureRefusal) as exc:
        return {"error": str(exc)}

    try:
        plant_mapping.persist_mapping(build, platform_root, name, supersede=supersede)
    except AuditEntryNotWritten as exc:
        return {"error": str(exc)}
    except plant_mapping.MappingRebuildRefusal as exc:
        return {"error": str(exc), "citing_events": exc.event_ids}

    summary = build.summary()
    return {
        "name": name,
        "project_root": str(platform_root),
        "dataset_root": str(candidate),
        "unreadable": build.unreadable,
        "n_dates": summary["totals"]["n_dates"],
        "n_images": summary["totals"]["n_images"],
        "n_mapped": summary["totals"]["n_mapped"],
        "n_unattributed": summary["totals"]["n_unattributed"],
        "per_date": summary["per_date"],
        "nn_tolerance_m": build.nn_tolerance_m,
        "max_match_distance_m": plant_mapping.match_gates(
            build.nn_tolerance_m["value"])["max_match_distance_m"],
    }


def _resolve_positive_class_id(trait_name: str, predictions_by_date: dict[str, str]) -> tuple[int | None, str]:
    """Thin wrapper over ``phenology.resolve_positive_class_id`` (the one resolution both delivery
    doors' positive-class-id surfaces call), for this tool's trait-name-based callers."""
    from tcip_mcp.traits import get_trait

    return phenology.resolve_positive_class_id(get_trait(trait_name), predictions_by_date)


def _resolve_producer_identity(predictions_by_date: dict[str, str]) -> dict:
    """Collect producing-model identity from each date's ``operating_point.json`` sidecar.

    A single producer across dates carries through; differing producers collapse to ``"multiple"``
    so a curve spliced from two models is not silently attributed to one. Best-effort, a missing
    sidecar contributes nothing rather than failing the delivery.
    """
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    shas: set[str] = set()
    exps: set[str] = set()
    for pred_dir in predictions_by_date.values():
        data = read_operating_point_sidecar(pred_dir)
        if not data:
            continue
        if data.get("checkpoint_sha256"):
            shas.add(str(data["checkpoint_sha256"]))
        if data.get("experiment_id"):
            exps.add(str(data["experiment_id"]))

    def _one(vals: set[str]) -> str | None:
        if not vals:
            return None
        return next(iter(vals)) if len(vals) == 1 else "multiple"

    return {"sha256": _one(shas), "experiment_id": _one(exps)}


def _greedy_match(gt: list, preds: list, gt_boxes: list, pred_boxes: list, *,
                  score: Callable[[Any, Any], float], tolerance: float) -> list[tuple]:
    """Greedy 1:1 IoU assignment over every (gt, pred) pair, highest IoU claimed first.

    The classifier calibration's ``iou_match`` branch only: ``center_match`` calls
    ``evaluation.center_match_pairs`` directly instead (see ``_match_gt_to_predictions``), so this
    keeps a single accept direction (``score >= tolerance``) rather than the two directions a
    shared higher/lower-is-better loop used to carry. Equal IoUs are claimed by (gt index,
    pred index) descending here, where ``tcip_annotation.matching.compute_matches`` keeps them
    ascending: the two agree on the IoU primitive, not on the tie order.
    """
    pairs = sorted(
        ((score(g, p), gi, pi) for gi, g in gt_boxes for pi, p in pred_boxes),
        reverse=True,
    )
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple] = []
    for s, gi, pi in pairs:
        if s < tolerance or gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append((gt[gi], preds[pi]))
    return matches


def _center(a) -> tuple[float, float]:
    b = a.geometry
    return ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0)


def _match_gt_to_predictions(gt: list, preds: list, *, kind: str,
                             center_match_tolerance: float | None = None,
                             iou_threshold: float = 0.5) -> list[tuple]:
    """Match GT to predictions using the criterion resolved by the caller, never a pinned IoU
    threshold (a pinned IoU=0.3 silently paired objects by a criterion different from the trait's
    own choice, dropping real classification calibration pairs from the reference with no
    disclosure).

    Unlike ``tcip_annotation.matching.compute_matches`` (which only matches within the same
    ``subject`` name), a classification calibration matches on box geometry alone, not
    ``subject``: a GT box carries the trait's object type in ``subject`` and its confirmed value
    under ``attributes[attribute]``, and a prediction box under a classified scope carries the
    same shape, so matching by ``subject`` name would already agree by construction rather than
    testing anything (both sides read as the same object class). Box geometries only (polygon GT
    is out of scope for a box-detector's classification calibration).

    ``kind``/``center_match_tolerance``/``iou_threshold`` come from
    ``evaluation.resolve_match_criterion``, the same resolver every other localization consumer
    goes through, computed once across the whole reference split by the caller
    (``_classification_items``), never re-derived per image (a per-image average lets one atypical
    annotation both drop real pairs, on a smaller-than-typical image, and fabricate a false match,
    on a larger-than-typical one). For ``center_match``, calls ``evaluation.center_match_pairs``
    directly under its ``distance_first`` policy: acceptance drops a prediction's score once it is
    confirmed (``review.py``), so a partly reviewed bucket holds records with no place in a score
    order, and geometry alone is the evidence for which ground truth a prediction identifies
    (``_center_match_image`` returns aggregate TP/FP/FN counts only, under the count's own
    score-first policy, so this pairing goes through the shared primitive under its own policy
    instead). For ``iou_match``, reuses
    ``tcip_annotation.matching.box_iou`` (the same primitive ``compute_matches`` itself calls)
    rather than a second IoU implementation.
    """
    from tcip_annotation.matching import box_iou
    from tcip_annotation.state import BBox

    gt_boxes = [(i, a) for i, a in enumerate(gt) if isinstance(a.geometry, BBox)]
    pred_boxes = [(i, a) for i, a in enumerate(preds) if isinstance(a.geometry, BBox)]

    if kind == "center_match":
        from tcip_mcp.pipelines.training.evaluation import center_match_pairs

        tolerance = center_match_tolerance if center_match_tolerance is not None else 0.0
        gt_centers = [_center(g) for _, g in gt_boxes]
        pred_centers = [_center(p) for _, p in pred_boxes]
        # The identity question: acceptance drops a prediction's score, so geometry alone, not
        # confidence order, decides which ground truth a prediction is.
        pairs = center_match_pairs(gt_centers, pred_centers, tolerance, policy="distance_first")
        return [(gt[gt_boxes[gi][0]], preds[pred_boxes[pi][0]]) for gi, pi in pairs]

    return _greedy_match(gt, preds, gt_boxes, pred_boxes,
                         score=lambda g, p: box_iou(g.geometry, p.geometry),
                         tolerance=iou_threshold)


def _classification_items(gt_dir: str, pred_dir: str, *, trait_name: str, subject: str,
                          positive_value: str, attribute: str) -> list[dict]:
    """Build classification calibration/holdout items for one split from paired GT + prediction dirs.

    For every ``<stem>.json`` present in both dirs, matches GT annotations against predictions by
    the trait's own localization criterion (``_match_gt_to_predictions``) and yields one item per
    matched pair: ``{"image_id": stem, "is_true_positive": <GT subject/attribute == positive_value>,
    "is_pred_positive": <prediction subject == positive_value>, "bbox": <the GT box, x1,y1,x2,y2>}``.
    ``subject`` scopes the GT side to the run's own object class: a labels dir isn't guaranteed to
    hold only one kind of annotation, a dataset that also isolates an enabling subject (e.g.
    ``bush``, per the root CLAUDE.md's "a subject is not a trait" rule) would otherwise let an
    unrelated object's box enter the match pool and pair against the classifier's prediction
    purely on proximity. Predictions are held to the object class the same way ground truth is
    (:func:`~tcip_annotation.json_io.require_classified_record`), never subject-filtered out of
    that check. ``attribute`` is the per-run fact naming which
    GT attribute carries the trait's positive-class axis,
    threaded by the caller, never hardcoded here (a hardcoded attribute name would pin this
    producer to one trait). ``bbox`` is the matched instance's own GT geometry, carried through
    so ``resolve_classifier_operating_point``'s content-overlap check has real per-instance
    content to hash (a placeholder box would collapse every item of the same class to one
    identical hash, defeating that check entirely; see its own docstring). An unmatched GT or
    prediction (the detector itself missed or hallucinated an object) is not a classification-call
    disagreement and is excluded, this calibrates the classifier's call, not the detector's, the
    same separation the platform's own detect-then-classify decomposition makes elsewhere.

    ``pred_dir``'s own recorded scope governs when it has one: no stamp at all (the hand-split
    calibration/holdout workflow) leaves the caller's stated ``(subject, attribute)`` in force; a
    classified stamp must agree with it, else this refuses naming both; a detector stamp refuses
    outright, a detector bucket carries no value to calibrate. The vocabulary a bare ``pred_dir``,
    or a classified stamp recording no usable ``id_map``, is held to is the registry ``gt_dir``'s
    own dataset root carries; with none, this refuses naming ``pred_dir``, ``gt_dir`` and the
    remedy (place the split under its dataset root), since the vocabulary a classifier is
    calibrated against is the registry the reference belongs to, never the values the reference
    happens to carry, and a registry borrowed from elsewhere would be a claim the data does not
    carry.

    ``gt_dir`` is read as a measurement reference, so it goes through
    ``json_io.require_reference_ground_truth`` first: a GT dir pointed at a prediction bucket would
    match the classifier's calls against the classifier's own calls and agree with itself
    perfectly, which no numeric gate below can catch.

    The match criterion (kind + tolerance/iou_threshold) is resolved once across the whole split
    via ``evaluation.resolve_match_criterion``, the same resolver every other localization
    consumer goes through, never a second independent computation, built from the same
    subject-scoped GT the matching itself uses, so an unrelated subject's typical size can't skew
    it either, and never re-derived per image.
    """
    from tcip_annotation import json_io
    from tcip_annotation.json_io import prediction_documents
    from tcip_annotation.state import BBox
    from tcip_mcp.pipelines.postprocessing.phenology import bucket_id_map
    from tcip_mcp.pipelines.resolution import bucket_scope
    from tcip_mcp.pipelines.training.evaluation import resolve_match_criterion

    gt_p, pred_p = Path(gt_dir), Path(pred_dir)
    json_io.require_reference_ground_truth(gt_p)  # the prediction side is never held to this
    scope = bucket_scope(pred_p)
    if scope is None:
        vocabulary_map = None
    elif scope.classified:
        if (scope.subject, scope.attribute) != (subject, attribute):
            raise ValueError(
                f"{pred_p}'s stamp records scope (subject={scope.subject!r}, "
                f"attribute={scope.attribute!r}), not the stated (subject={subject!r}, "
                f"attribute={attribute!r})."
            )
        vocabulary_map = bucket_id_map(pred_p)
    else:
        raise ValueError(f"{pred_p} is a detector bucket: it carries no value to calibrate.")
    if vocabulary_map is None:
        from tcip_mcp.pipelines.data.label_queries import (
            resolve_registry_id_map,
            resolved_classes_path,
        )

        if resolved_classes_path(gt_dir) is None:
            absent = (
                "carries no stamp at all" if scope is None
                else "records a classified scope with no usable id_map"
            )
            raise ValueError(
                f"{pred_p} {absent}, and {gt_p} resolves no dataset registry of its own: place "
                "the split under its dataset root (<root>/annotations/<date>/, or a labels/ tree "
                "directly under a root carrying classes.json), since the vocabulary a classifier "
                "is calibrated against is the registry the reference belongs to, never the values "
                "the reference happens to carry."
            )
        _reg, vocabulary_map = resolve_registry_id_map(gt_dir, subject, attribute)
    vocabulary = set(vocabulary_map or {})
    if positive_value not in vocabulary:
        raise ValueError(
            f"{gt_p}'s registry declares values {sorted(vocabulary)} for (subject={subject!r}, "
            f"attribute={attribute!r}), which does not include the positive value "
            f"{positive_value!r}: declare it in the registry before calibrating against this split."
        )
    # gt_dir/pred_dir may themselves be prediction buckets (a calibration/holdout split of one),
    # so both are walked through prediction_documents, their own sidecar stamps excluded.
    paired = [f for f in prediction_documents(gt_p) if (pred_p / f.name).is_file()]

    def _scoped_gt(path: str) -> list:
        return [a for a in json_io.read_annotations(path) if a.subject == subject]

    def _xywh(a) -> list[float]:
        b = a.geometry
        return [b.x1, b.y1, max(b.x2 - b.x1, 0.0), max(b.y2 - b.y1, 0.0)]

    per_image = [
        {"gt": [{"bbox": _xywh(a), "category_id": 0}
                for a in _scoped_gt(str(gt_file)) if isinstance(a.geometry, BBox)]}
        for gt_file in paired
    ]
    criterion = resolve_match_criterion(trait_name, per_image)
    kind = criterion["kind"]
    center_match_tolerance = criterion.get("tolerance")
    iou_threshold = criterion.get("iou_threshold", 0.5)

    items: list[dict] = []
    for gt_file in paired:
        pred_file = pred_p / gt_file.name
        gt_annots = _scoped_gt(str(gt_file))
        pred_annots = json_io.read_annotations(str(pred_file))
        for gt_a, pred_a in _match_gt_to_predictions(
            gt_annots, pred_annots, kind=kind, center_match_tolerance=center_match_tolerance,
            iou_threshold=iou_threshold,
        ):
            gt_value = gt_a.attributes.get(attribute) if gt_a.attributes else None
            if gt_value is None:
                # Never assessed for `attribute` yet: a soft, expected gap, not a confirmed
                # negative, which would fabricate a disagreement against a perfect classifier.
                continue
            if gt_value not in vocabulary:
                raise ValueError(
                    f"{gt_file}: ground-truth value {gt_value!r} under (subject={subject!r}, "
                    f"attribute={attribute!r}) is outside the registry's declared values "
                    f"{sorted(vocabulary)}."
                )
            pred_value = json_io.require_classified_record(
                pred_a, subject=subject, attribute=attribute, vocabulary=vocabulary,
                source=f"{pred_file}")
            box = gt_a.geometry
            items.append({
                "image_id": gt_file.stem,
                "is_true_positive": gt_value == positive_value,
                "is_pred_positive": pred_value == positive_value,
                "bbox": [box.x1, box.y1, box.x2, box.y2],
            })
    return items


def _stated_root_disagreement(dataset_root: str, candidates: dict[str, str]) -> str | None:
    """The refusal for a stated dataset root a caller-supplied directory's own root contradicts.

    Only a positive disagreement refuses: a directory the dataset layout cannot place answers
    nothing, and a bespoke calibration over loose directories with a stated root is legitimate work
    for the two detector doors this check otherwise serves alone. The classifier door additionally
    requires, past this check, that a bare or map-less prediction bucket's ground truth resolve the
    registry its own dataset root carries (``_classification_items``): a loose ground-truth
    directory with no relation to the stated root refuses there even when this check alone does not.

    Two other modules (``scale_tools.py``, ``calibration_tools.py``) import this beside its own
    caller here, the same cross-module-consumer shape that moved ``calibrate_operating_point``/
    ``gate_evidence_summary`` out of their donor module; it stays here deliberately, since it is this
    module's own refusal wording, not a shared primitive those callers each restate.
    """
    from tcip_mcp.dataset_layout import dataset_root_of

    stated = Path(dataset_root).resolve()
    for role, path in candidates.items():
        derived = dataset_root_of(path)
        if derived is not None and derived.resolve() != stated:
            return (f"{role} {str(path)!r} sits under dataset root {str(derived.resolve())!r}, "
                    f"while dataset_root states {str(stated)!r}. The claim, its covered locations "
                    "and its reference are all recorded against one root, so state the root the "
                    "calibration's own directories live under.")
    return None


def _agreed_checkpoint_identity(pred_dirs: list[str]) -> str | None:
    """The checkpoint identity the calibration evidence itself carried, or ``None``.

    Copied from the prediction buckets' own stamps, never re-resolved from a checkpoint file:
    re-hashing a file proves a file with that content exists somewhere, not that these predictions
    came from it. Buckets carrying none, or disagreeing, record the identity absent, which the
    reader compares as absence equal to absence rather than skipping the comparison.
    """
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    carried = {(read_operating_point_sidecar(d) or {}).get("checkpoint_sha256") for d in pred_dirs}
    return carried.pop() if len(carried) == 1 else None


@mcp.tool()
@audited
def calibrate_classifier_operating_point(
    trait_name: str,
    subject: str,
    attribute: str,
    calibration_gt_dir: str,
    calibration_pred_dir: str,
    holdout_gt_dir: str,
    holdout_pred_dir: str,
    output_dir: str,
    dataset_root: str,
    experiment_id: str | None = None,
) -> dict:
    """Calibrate and validate the trait's positive-class classifier against held-out GT.

    Builds classification calibration/holdout items by matching each split's GT against its
    predictions via the trait's own localization criterion (``_classification_items``), runs the
    same-rigor classification-mode gate (``operating_point.resolve_classifier_operating_point``,
    disjointness, train-disjointness, content-duplication, count-bias, and a derived
    compensating-error floor, mirroring the detector calibration path), and stamps the result into
    ``<output_dir>/classifier_operating_point.json``, a file distinct from the count operating
    point's own sidecar, never conflatable with it. Without this producer, a classifier-validated
    stamp can never be earned on disk, and the gate floors every caller to unvalidated forever.

    A stamp that claims validation names the record it was earned from: the gate runs once through
    ``resolution.open_validation`` over the evidence, ``seal_validation`` files the row and returns
    the stamp with its pointer merged in, and the stamp is written last. A calibration that does not
    clear its gate stamps unvalidated, with its failures, and earns nothing.

    Refuses (a plain ``{"error": ...}``) when either GT dir holds the model's own predictions
    rather than a measurement; the pred dirs are predictions by definition and are not held to it,
    and when a GT dir's own dataset root contradicts the stated ``dataset_root``.

    This door takes no split manifest: it draws no universe at all, the caller already hands it
    four already-split directories.

    Args:
        trait_name: The registered trait whose positive class is being calibrated.
        subject: The GT annotation subject naming this trait's object type, a per-run fact the
            caller supplies: the (subject, attribute) axis threads from the run's own config, never
            from ``TraitSpec`` (pinning it here would reintroduce a trait-vocabulary leak at the
            public surface). Scopes the GT side of
            matching so an unrelated subject sharing the same labels dir (e.g. an enabling subject
            like ``bush``, root CLAUDE.md's "a subject is not a trait") can't enter the match pool.
        attribute: The GT annotation attribute carrying this trait's positive-class axis, a
            per-run fact the caller supplies, same rationale as ``subject`` above.
        calibration_gt_dir / calibration_pred_dir: Paired per-image JSON dirs for the calibration
            split (same stems).
        holdout_gt_dir / holdout_pred_dir: Paired per-image JSON dirs for the disjoint held-out split.
        output_dir: Where to write ``classifier_operating_point.json``.
        dataset_root: The dataset this calibration's claim hangs off, stated by the caller: the
            record's reference locations are written against it, and it is the root a reader
            resolves them from. Refuses when either GT dir's own layout places it under a different
            root; loose directories the layout cannot place refuse nothing here. Past this check, a
            GT dir whose prediction bucket carries no usable vocabulary of its own (a bare bucket,
            or a classified stamp with no id_map) must itself resolve the registry its own dataset
            root carries (``_classification_items``'s own refusal, naming both directories and the
            remedy), so the two ground-truth directories cannot be loose unless their buckets
            already carry a vocabulary.
        experiment_id: The classifier checkpoint's training-run id, if known, gates train-
            disjointness the same way the detector calibration path does. ``None`` (a foreign/
            unregistered checkpoint) skips that check rather than failing closed.
    """
    from tcip_mcp.dataset_layout import annotation_date
    from tcip_mcp.pipelines.data.splits import manifest_date_key
    from tcip_mcp.pipelines.operating_point import resolve_classifier_operating_point
    from tcip_mcp.traits import TraitUnknownError, get_trait

    try:
        spec = get_trait(trait_name)
    except TraitUnknownError as e:
        return {"error": str(e)}
    if not spec.positive_class_name:
        return {"error": f"trait {trait_name!r} defines no positive_class_name to calibrate"}
    disagreement = _stated_root_disagreement(
        dataset_root, {"calibration_gt_dir": calibration_gt_dir, "holdout_gt_dir": holdout_gt_dir})
    if disagreement:
        return {"error": disagreement}

    from tcip_annotation.json_io import UnreadableLabelDocument
    from tcip_store import StoreError

    try:
        cal_items = _classification_items(calibration_gt_dir, calibration_pred_dir, trait_name=trait_name,
                                          subject=subject, positive_value=spec.positive_class_name,
                                          attribute=attribute)
        hold_items = _classification_items(holdout_gt_dir, holdout_pred_dir, trait_name=trait_name,
                                           subject=subject, positive_value=spec.positive_class_name,
                                           attribute=attribute)
    except (ValueError, UnreadableLabelDocument, StoreError) as exc:
        return {"error": str(exc)}
    result = resolve_classifier_operating_point(
        trait_name, calibration_items=cal_items, holdout_items=hold_items,
        experiment_id=experiment_id,
        calibration_date=manifest_date_key(annotation_date(calibration_gt_dir)),
        calibration_labels_dir=calibration_gt_dir,
    )

    from tcip_mcp.project_paths import resolve_output_path

    from tcip_mcp.pipelines.resolution import open_validation, seal_validation, write_sidecar

    out = resolve_output_path(output_dir)
    checkpoint_sha256 = _agreed_checkpoint_identity([calibration_pred_dir, holdout_pred_dir])
    stamp = {
        "operating_point": {"classifier": {"validated_against": result["validated_against"],
                                           "value": spec.positive_class_name}},
        "validated": result["passed"],
        "validated_by": None,
        "failures": result["failures"],
        "gate_evidence": result["gate_evidence"],
        "checkpoint_sha256": checkpoint_sha256,
        "experiment_id": experiment_id,
        "trait": trait_name,
    }
    if result["passed"]:
        draft = open_validation(
            document="classifier_operating_point",
            # Named off the function this door reported from, so record and report share one gate.
            evidence={"resolver": resolve_classifier_operating_point.__name__,
                      "inputs": {"calibration_items": cal_items, "holdout_items": hold_items}},
            trait=trait_name, checkpoint_sha256=checkpoint_sha256,
            producing_experiment_id=experiment_id,
            reference_inputs={
                "dataset_root": dataset_root,
                "label_dirs": {"calibration": calibration_gt_dir, "holdout": holdout_gt_dir},
                "reference_buckets": {"calibration": calibration_pred_dir,
                                      "holdout": holdout_pred_dir},
            },
        )
        _, stamp = seal_validation(draft, dataset_root=dataset_root, bucket_dirs=[],
                                   stamp_body=stamp)
    write_sidecar(out, stamp, "classifier_operating_point")
    return {
        "output_dir": str(out),
        "validated_against": result["validated_against"],
        "passed": result["passed"],
        "failures": result["failures"],
        "validated_by": stamp["validated_by"],
        "n_calibration_items": len(cal_items),
        "n_holdout_items": len(hold_items),
    }


@mcp.tool()
@audited
def deliver_phenology_milestones(
    trait: str,
    mapping_name: str,
    predictions_by_date: dict[str, str],
    output_csv_path: str,
    classifier_pred_dirs: list[str] | None = None,
) -> dict:
    """Per-plant phenology milestones from classified predictions + a plant mapping.

    A phenology milestone is a crossing of the **fraction of a plant's detected objects that are in
    the trait's positive/measured state**, an expert-defined morphological stage emitted by a
    *validated* classifier (the trait's positive class), never a geometric proxy such as bounding-box
    height. For a registered trait whose positive state is, say, ``<majority_label>`` this reports:

        <phenology_prefix>_<majority_label>_date   date most objects reached that state
                                 (crops.yml) = the 95% crossing (crossing-unconfirmed reading,
                                 pending breeder confirmation)
        <phenology_prefix>_05/50/95per_date  dates the positive-state fraction crosses 5/50/95%

    Column names and crossing fractions come from ``trait``'s ``TraitSpec``, a different registered
    trait yields its own prefixed columns without a code change.

    Args:
        trait: A registered trait name (``registered_traits()``), required, no default. The
            positive class id is resolved from the prediction buckets' own recorded ``id_map`` by
            this trait's ``positive_class_name`` (a mapping fact read from the labels the run
            actually decoded through, never a pinned default or a separate registry re-derivation
            that could disagree with it).
        mapping_name: Name of a plant mapping persisted under this project (``{date:
            [assignment, ...]}`` with ``stem`` / ``plot_name`` / ``accession_name`` per
            assignment), produced by the web plant-mapping step or ``build_plant_mapping``.
        predictions_by_date: ``{date: predictions_dir}``, each dir holds per-image COCO/JSON
            prediction files (``<stem>.json``) from the state classifier.
        output_csv_path: Where to write the delivered per-plant CSV (e.g.
            ``<phenology_prefix>_phenology.csv``). A relative path resolves against the
            platform state root, never the server process's cwd.
        classifier_pred_dirs: Bucket(s) carrying the trait's classifier-validity stamp
            (``classifier_operating_point.json``, written by ``calibrate_classifier_operating_point``)
, reconciled from disk, never trusted from a caller-asserted string. ``None``
            or a bucket with no such stamp floors the classifier dimension to unvalidated.

    The count operating point's confidence and validity are both read from each prediction
    bucket's own ``operating_point.json``, never from a caller: this tool takes no
    acknowledgement, so an unvalidated dimension always refuses here. Only the Results tab's
    ``export_csv`` route can deliver this trait's ``state_crossing_dates`` unvalidated, through a
    breeder's own acknowledged act.

    A bucket produced by a tiled run also gates on its ``tile_size``: the tile edge scales the
    per-image counts the positive fraction is built from, so a run with no persisted training
    geometry and no explicit caller override refuses here, the same way an uncalibrated conf does.
    Buckets from untiled runs are never gated on it.

    The delivered CSV's producer tail (``producer_model_sha256``, ``producing_experiment_id``,
    ``produced_at``, ``validation_record``) is built from the bindings the reconciliation verified,
    so a bucket whose validation claim no record answers for delivers those cells empty rather than
    carrying the names it asserted for itself, and a delivery every bucket of which is bound names
    the records.

    Returns a summary. Measurement-integrity guard: if no bucket, anywhere in the delivery, ever
    classified along the trait's positive-class axis, the positive fraction is not a valid measurement
    anywhere, the tool refuses to write the CSV and returns ``error`` with
    ``positive_class_assessed: false``. Rows for a plant with a partially-unclassified or partially-
    missing date still ship (with the gap disclosed via ``n_dates_unclassified``/
    ``n_dates_missing_images``) but carry no fabricated milestone dates for that plant (see
    CLAUDE.md's measurement-integrity invariant).
    """
    from tcip_mcp.class_registry import RegistryError, registry_for_pred_dirs
    from tcip_mcp.operationalization import (
        STATE_CROSSING_DATES,
        check_operationalization,
        resolve_trait_and_record,
    )
    from tcip_mcp.project_paths import resolve_output_path
    from tcip_mcp.traits import TraitUnknownError

    output_csv_path = str(resolve_output_path(output_csv_path))
    try:
        spec, record, _specs_dir = resolve_trait_and_record(trait, STATE_CROSSING_DATES)
    except TraitUnknownError as e:
        return {"error": str(e), "n_plants": 0}

    # The one dataset every one of this delivery's buckets belongs to; refuses if they disagree.
    try:
        registry = registry_for_pred_dirs(list(predictions_by_date.values()))
    except RegistryError as e:
        return {"error": str(e), "n_plants": 0}
    # Ahead of the positive class id, so an unstated trait's class-id failure never names the wrong problem.
    stated = check_operationalization(spec, record, STATE_CROSSING_DATES, registry=registry)
    if not stated.ok:
        return {"error": stated.message, "n_plants": 0}
    pos = spec.positive_class_name

    from tcip_mcp.pipelines.postprocessing import plant_mapping
    from tcip_mcp.project_paths import platform_state_root

    platform_root = platform_state_root()
    try:
        mapping_build, verified = plant_mapping.resolve_delivery_mapping(
            platform_root, mapping_name, predictions_by_date)
    except plant_mapping.MappingDeliveryRefusal as e:
        return {"error": str(e), "n_plants": 0}

    mapping = mapping_build.rows()
    dates_delivered = list(predictions_by_date)
    disclosure = mapping_build.delivery_disclosure(verified, dates_delivered)

    positive_class_id, msg = _resolve_positive_class_id(trait, predictions_by_date)
    if positive_class_id is None:
        return {"error": (f"could not resolve the {pos} class id from any prediction bucket's "
                          f"own recorded id_map ({msg})."),
                "n_plants": 0}

    from tcip_annotation.json_io import ClassifiedRecordRefused, UnreadableLabelDocument
    from tcip_mcp.pipelines.resolution import StampScopeUnstated
    from tcip_store import StoreError

    try:
        result = phenology.per_plant_phenology(
            mapping, predictions_by_date, positive_class_name=pos, spec=spec,
        )
    except (UnreadableLabelDocument, StampScopeUnstated, ClassifiedRecordRefused,
            StoreError) as exc:
        return {"error": str(exc), "n_plants": 0}
    rows = result["rows"]

    if not result["positive_class_assessed"]:
        return {
            "error": (
                f"predictions carry no {pos} class anywhere in this delivery, the classifier that "
                f"produced them never assessed this trait's positive class. "
                f"The {pos} fraction is not a valid measurement, run and validate "
                f"the {pos}-state classifier before computing phenology."
            ),
            "positive_class_assessed": False,
            "n_plants": len(rows),
            "n_images_unattributed": disclosure["images_unattributed"],
            "dates_delivered": disclosure["dates_delivered"],
        }

    # Measurement-integrity gate: a delivery requires a classifier validated against held-out GT.
    # This tool builds no acknowledgement, so an unvalidated dimension always refuses here.
    from tcip_mcp.pipelines.resolution import (
        bind_classifier_validity,
        binding_notes_text,
        check_delivery_gate,
        reconcile_classifier_validity,
        reconcile_operating_point_validity,
        reconcile_tile_size_validity,
    )

    # The count operating point's validity is read from each prediction bucket's operating_point.json
    # (stamped by run_inference), with no caller floor to reconcile: this door takes no assertion.
    recon = reconcile_operating_point_validity(list(predictions_by_date.values()), trait=trait)
    op_state = recon["validated"]

    # The tile scale is the second gating dimension of the same count operating point: a tile edge
    # with no real basis at all is as untrustworthy as an uncalibrated conf.
    tile_recon = reconcile_tile_size_validity(list(predictions_by_date.values()))

    # The classifier's validity is read the same way, from classifier_operating_point.json, never a
    # caller-asserted string. No producer stamp anywhere -> floors to unvalidated, same as
    # the count dimension with no on-disk backing.
    classifier_recon = reconcile_classifier_validity(classifier_pred_dirs or [])
    classifier_state = classifier_recon["validated"]

    # Bind the classifier stamp to this delivery: unlike the count dimension (which reconciles from
    # the same predictions_by_date buckets it delivers), classifier_pred_dirs is a separate,
    # caller-supplied list, reconcile_classifier_validity alone can't see whether a
    # genuinely-validated stamp was calibrated for an unrelated model or trait. A sidecar's own
    # recorded `trait`/`experiment_id` (written by calibrate_classifier_operating_point) must agree
    # with what's actually being delivered here; a foreign/unregistered checkpoint calibration
    # (experiment_id=None) is deliberately not rejected for lacking one to compare against.
    classifier_state, classifier_binding_note = bind_classifier_validity(
        classifier_state, classifier_pred_dirs, list(predictions_by_date.values()), trait=trait,
    )

    # A delivered phenotype needs both dimensions validated; this tool passes no acknowledgement.
    flags = phenology.phenology_delivery_flags(classifier_state, op_state, tile_recon)
    gate = check_delivery_gate(flags)
    if not gate.ok:
        floor_note = ""
        if recon["missing_sidecars"] or recon["unvalidated_buckets"]:
            floor_note = (f" On-disk operating-point reconciliation floored the count to invalid "
                          f"(missing sidecars: {recon['missing_sidecars']}; unvalidated buckets: "
                          f"{recon['unvalidated_buckets']}).")
        if tile_recon["unvalidated_buckets"]:
            floor_note += (
                f" Tiled bucket(s) {tile_recon['unvalidated_buckets']} carry a tile_size with no "
                "persisted training geometry, no recoverable native-frame edge, and no explicit "
                "caller override, so the scale the counts were produced at has no basis. "
                "Re-export with an explicit tile_size, or from a checkpoint whose training tile "
                "geometry was persisted.")
        if classifier_recon["missing_sidecars"]:
            floor_note += (f" No classifier_operating_point.json found in "
                           f"{classifier_recon['missing_sidecars']}, calibrate the classifier via "
                           "calibrate_classifier_operating_point before delivering.")
        if recon["binding_notes"]:
            floor_note += f" {binding_notes_text(recon['binding_notes'])}"
        if classifier_binding_note:
            floor_note += f" {classifier_binding_note}"
        return {
            "error": (
                "a delivered phenotype requires both a validated positive-state classifier "
                f"(reconciled from classifier_operating_point.json = {classifier_state!r}) and a "
                f"validated count operating point (reconciled from operating_point.json = "
                f"{op_state!r})." + floor_note
                + " Validate both (calibrate_classifier_operating_point for the classifier; a "
                "calibrated run_inference for the count). This tool takes no acknowledgement; "
                "acknowledge and deliver a clearly-flagged provisional CSV from the Results tab "
                "instead."
            ),
            "positive_state_classifier_validated": gate.stamp["classifier"],
            "operating_point_validated": op_state,
            "tile_size_validated": tile_recon["validated"],
            "operating_point_missing_sidecars": recon["missing_sidecars"],
            "n_plants": len(rows),
        }

    # What the sidecars assert about the producer, corroborated by the writer against the verified
    # bindings.
    producer = _resolve_producer_identity(predictions_by_date)

    # A confirmation withdrawn or a field moved while this ran refuses here, with nothing written.
    # The registry is re-read too, not reused from the first check, to catch a racing registry edit.
    spec_now, record_now, _ = resolve_trait_and_record(trait, STATE_CROSSING_DATES)
    try:
        registry_now = registry_for_pred_dirs(list(predictions_by_date.values()))
    except RegistryError as e:
        return {"error": str(e), "n_plants": len(rows)}
    still_stated = check_operationalization(
        spec_now, record_now, STATE_CROSSING_DATES, registry=registry_now, basis=stated.basis)
    if not still_stated.ok:
        return {"error": still_stated.message, "n_plants": len(rows)}

    from tcip_mcp.project_paths import platform_state_root

    # write_phenology_csv re-runs the same gate over these flags, composes every provenance cell
    # (including the majority crossing-unconfirmed marker) and records the delivery.
    cells = phenology.write_phenology_csv(
        "deliver_phenology_milestones", rows, Path(output_csv_path), spec,
        flags=flags, acknowledgement=None, basis=still_stated.basis,
        operating_point_confs=recon["confs"], producer=producer, bindings=recon["bindings"],
        predictions_by_date=predictions_by_date, project_root=platform_state_root(),
        plant_mapping=disclosure)
    # Per-milestone summary: report reached-counts for each milestone the spec actually declares.
    n_reached: dict[str, int] = {}
    for key in phenology._milestone_targets(spec):
        col = f"{spec.phenology_prefix}_{key}_date"
        n_reached[key] = sum(1 for r in rows if r.get(col))
    return {
        "csv_path": output_csv_path,
        "n_plants": len(rows),
        "n_plants_reached_milestone": n_reached,
        "positive_class_assessed": True,
        "positive_state_classifier_validated": cells["positive_state_classifier_validated"],
        "operating_point_validated": cells["operating_point_validated"],
        "unvalidated_dimensions": cells["unvalidated_dimensions"],
        "tile_size_validated": gate.stamp.get("tile_size"),
        "n_images_unattributed": disclosure["images_unattributed"],
        "dates_delivered": disclosure["dates_delivered"],
        "columns": phenology.phenology_csv_columns(spec),
        "captures_unverified": verified["captures_unverified"],
        "plant_csvs_unverified": verified["plant_csvs_unverified"],
    }
