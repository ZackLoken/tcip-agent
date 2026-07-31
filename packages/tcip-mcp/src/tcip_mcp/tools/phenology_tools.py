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
from typing import Any, Callable

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
    nn_tolerance_m: float | None = None,
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
        nn_tolerance_m: Nearest-neighbour tolerance (m). ``None`` (default) derives it from the
            plot's grid pitch (pitch/6) so the match radius stays within half a grid cell; an
            explicit value is honored but still capped at that pitch-derived ceiling.

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
def update_trait_spec_fields(trait_name: str, fields: dict, provenance_entries: list[str]) -> dict:
    """Update one or more fields on an already-registered trait's spec, recording who asserted
    the change and how firmly (K18 B2.5).

    Before this tool, a trait spec was authored by hand-writing YAML directly — no audited
    record, no re-validation. This refuses if the trait has no existing spec file (creating a new
    trait is a separate, still-manual authoring step) or if the merged result would fail the same
    crops.yml cross-check every config-authored spec already goes through. Returns the updated
    spec.

    This is what a real localization-kind derivation (from actual GT box geometry) or a real
    breeder-answered count objective gets recorded through — never a silent default and never
    copied from another trait's values, both durable, audited facts instead of living only in a
    session's memory.
    """
    import dataclasses

    from tcip_mcp import traits

    spec = traits.write_trait_spec_fields(trait_name, fields, provenance_entries)
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(spec).items()}


def _resolve_positive_class_id(trait_name: str, predictions_by_date: dict[str, str]) -> tuple[int | None, str]:
    """Thin wrapper over ``phenology.resolve_positive_class_id`` (the one resolution both delivery
    doors' positive-class-id surfaces call — K4/K5/K6), for this tool's trait-name-based callers."""
    from tcip_mcp.traits import get_trait

    return phenology.resolve_positive_class_id(get_trait(trait_name), predictions_by_date)


def _resolve_producer_identity(predictions_by_date: dict[str, str]) -> dict:
    """Collect producing-model identity from each date's ``operating_point.json`` sidecar.

    A single producer across dates carries through; differing producers collapse to ``"multiple"``
    so a curve spliced from two models is not silently attributed to one. Best-effort — a missing
    sidecar contributes nothing rather than failing the delivery.
    """
    shas: set[str] = set()
    exps: set[str] = set()
    for pred_dir in predictions_by_date.values():
        sidecar = Path(pred_dir) / "operating_point.json"
        if not sidecar.is_file():
            continue
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
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
                  score: Callable[[Any, Any], float], tolerance: float,
                  best_first: bool) -> list[tuple]:
    """Greedy 1:1 assignment over every (gt, pred) pair, best-scoring pair claimed first.

    ``best_first=True`` (e.g. IoU, higher is better) keeps a pair while ``score >= tolerance``;
    ``False`` (e.g. center distance, lower is better) keeps a pair while ``score <= tolerance``. One
    shared assignment loop for both of ``_match_gt_to_predictions``'s match kinds, which differ only
    in the pairwise score and its accept direction, not in the greedy logic itself.
    """
    pairs = sorted(
        ((score(g, p), gi, pi) for gi, g in gt_boxes for pi, p in pred_boxes),
        reverse=best_first,
    )
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple] = []
    for s, gi, pi in pairs:
        ok = s >= tolerance if best_first else s <= tolerance
        if not ok or gi in matched_gt or pi in matched_pred:
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
    """Match GT to predictions using the criterion resolved by the caller — never a pinned IoU
    threshold (stage-6 review, K3: a pinned IoU=0.3 silently paired catkins by a criterion
    different from the trait's own choice, dropping real classification calibration pairs from
    the reference with no disclosure).

    Unlike ``tcip_annotation.matching.compute_matches`` (which only matches within the same
    ``subject`` name), a classification calibration must match a GT box (subject = the trait's
    object type, e.g. ``"catkin"``) against a prediction box whose ``subject`` IS the classifier's
    verdict (e.g. ``"elongated"``) — different names by design (K3/K4/K5). Box geometries only
    (polygon GT is out of scope for a box-detector's classification calibration).

    ``kind``/``center_match_tolerance``/``iou_threshold`` come from
    ``evaluation.resolve_match_criterion`` — the same resolver every other localization consumer
    goes through (K18 B3) — computed ONCE across the whole reference split by the caller
    (``_classification_items``), never re-derived per image (stage-6 review Finding A/N5: a
    per-image average lets one atypical annotation both drop real pairs, on a smaller-than-typical
    image, and fabricate a false match, on a larger-than-typical one). For ``center_match``,
    mirrors ``evaluation.py``'s own center-match algorithm (adapted, via ``_greedy_match``, to
    return matched ``(gt, pred)`` PAIRS — ``_center_match_image`` returns aggregate TP/FP/FN counts
    only, so pairing needs its own pass). For ``iou_match``, reuses
    ``tcip_annotation.matching.box_iou`` (the same primitive ``compute_matches`` itself calls)
    rather than a second IoU implementation.
    """
    from tcip_annotation.matching import box_iou
    from tcip_annotation.state import BBox

    gt_boxes = [(i, a) for i, a in enumerate(gt) if isinstance(a.geometry, BBox)]
    pred_boxes = [(i, a) for i, a in enumerate(preds) if isinstance(a.geometry, BBox)]

    if kind == "center_match":
        def _dist(g, p):
            gx, gy = _center(g)
            px, py = _center(p)
            return ((gx - px) ** 2 + (gy - py) ** 2) ** 0.5

        tolerance = center_match_tolerance if center_match_tolerance is not None else 0.0
        return _greedy_match(gt, preds, gt_boxes, pred_boxes,
                             score=_dist, tolerance=tolerance, best_first=False)

    return _greedy_match(gt, preds, gt_boxes, pred_boxes,
                         score=lambda g, p: box_iou(g.geometry, p.geometry),
                         tolerance=iou_threshold, best_first=True)


def _classification_items(gt_dir: str, pred_dir: str, *, trait_name: str, subject: str,
                          positive_value: str, attribute: str) -> list[dict]:
    """Build classification calibration/holdout items for one split from paired GT + prediction dirs.

    For every ``<stem>.json`` present in both dirs, matches GT annotations against predictions by
    the trait's own localization criterion (``_match_gt_to_predictions``) and yields one item per
    matched pair: ``{"image_id": stem, "is_true_positive": <GT subject/attribute == positive_value>,
    "is_pred_positive": <prediction subject == positive_value>, "bbox": <the GT box, x1,y1,x2,y2>}``.
    ``subject`` scopes the GT side to the run's own object class (stage-6 review NEW-5, round 4): a
    labels dir isn't guaranteed to hold only one kind of annotation — a dataset that also isolates
    an enabling subject (e.g. ``bush``, per the root CLAUDE.md's "a subject is not a trait" rule)
    would otherwise let an unrelated object's box enter the match pool and pair against a catkin
    classifier's prediction purely on proximity. Predictions are never subject-filtered here: a
    prediction's ``subject`` already carries the classifier's decoded VERDICT (e.g. ``"elongated"``),
    not an object-type name, so there is nothing to scope it against. ``attribute`` is the per-run
    fact naming which GT attribute carries the trait's positive-class axis (e.g. ``"elongation"`` for
    catkin) — threaded by the caller, never hardcoded here (K3/K4/K5: a hardcoded attribute name pins
    this producer to one trait exactly the defect K5 fixed elsewhere). ``bbox`` is the matched
    instance's own GT geometry — carried through so ``resolve_classifier_operating_point``'s
    content-overlap check has real per-instance content to hash (a placeholder box would collapse
    every item of the same class to one identical hash, defeating that check entirely; see its own
    docstring). An unmatched GT or prediction (the detector itself missed or hallucinated an object)
    is not a classification-call disagreement and is excluded — this calibrates the CLASSIFIER's
    call, not the detector's, the same separation the platform's own detect-then-classify
    decomposition makes elsewhere.

    The match criterion (kind + tolerance/iou_threshold) is resolved ONCE across the whole split
    via ``evaluation.resolve_match_criterion`` (K18 B3) — the same resolver every other
    localization consumer goes through, never a second independent computation — built from the
    SAME subject-scoped GT the matching itself uses, so an unrelated subject's typical size can't
    skew it either, and never re-derived per image (stage-6 review Finding A/N5).
    """
    from tcip_annotation import json_io
    from tcip_annotation.state import BBox
    from tcip_mcp.pipelines.training.evaluation import resolve_match_criterion

    gt_p, pred_p = Path(gt_dir), Path(pred_dir)
    paired = [f for f in sorted(gt_p.glob("*.json")) if (pred_p / f.name).is_file()]

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
                # Never assessed for `attribute` yet -- a soft, expected gap (stage-6 review N1),
                # not a confirmed negative. Coercing an unassessed instance into "not positive"
                # fabricates a disagreement against a perfect classifier -- the exact
                # unlabeled-vs-undecodable distinction json_io.UNLABELED exists for elsewhere in
                # this diff, not previously applied to this reference builder.
                continue
            box = gt_a.geometry
            items.append({
                "image_id": gt_file.stem,
                "is_true_positive": gt_value == positive_value,
                "is_pred_positive": pred_a.subject == positive_value,
                "bbox": [box.x1, box.y1, box.x2, box.y2],
            })
    return items


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
    experiment_id: str | None = None,
) -> dict:
    """Calibrate and validate the trait's positive-class classifier against held-out GT (K3).

    Builds classification calibration/holdout items by matching each split's GT against its
    predictions via the trait's own localization criterion (``_classification_items``), runs the
    same-rigor classification-mode gate (``operating_point.resolve_classifier_operating_point`` —
    disjointness, train-disjointness, content-duplication, count-bias, and a derived
    compensating-error floor, mirroring the detector calibration path), and stamps the result into
    ``<output_dir>/classifier_operating_point.json`` — a file distinct from the count operating
    point's own sidecar, never conflatable with it. This is the producer TRAP 2 requires: without
    it, a classifier-validated stamp can never be earned on disk, and the gate floors every caller
    to unvalidated forever.

    Args:
        trait_name: The registered trait whose positive class is being calibrated.
        subject: The GT annotation subject naming this trait's object type (e.g. ``"catkin"``) —
            a per-run fact the caller supplies, per Zack's decision that the (subject, attribute)
            axis threads from the run's own config, never from ``TraitSpec`` (K3/K4/K5: pinning it
            here would reintroduce the same trait-vocabulary leak K5 fixed at the public surface).
            Scopes the GT side of matching so an unrelated subject sharing the same labels dir
            (e.g. an enabling subject like ``bush`` — root CLAUDE.md's "a subject is not a trait")
            can't enter the match pool (stage-6 review NEW-5).
        attribute: The GT annotation attribute carrying this trait's positive-class axis (e.g.
            ``"elongation"`` for catkin) — a per-run fact the caller supplies, same rationale as
            ``subject`` above.
        calibration_gt_dir / calibration_pred_dir: Paired per-image JSON dirs for the calibration
            split (same stems).
        holdout_gt_dir / holdout_pred_dir: Paired per-image JSON dirs for the disjoint held-out split.
        output_dir: Where to write ``classifier_operating_point.json``.
        experiment_id: The classifier checkpoint's training-run id, if known — gates train-
            disjointness the same way the detector calibration path does. ``None`` (a foreign/
            unregistered checkpoint) skips that check rather than failing closed.
    """
    from tcip_mcp.pipelines.operating_point import resolve_classifier_operating_point
    from tcip_mcp.traits import TraitUnknownError, get_trait
    from tcip_mcp.utils.atomic_io import atomic_write_json

    try:
        spec = get_trait(trait_name)
    except TraitUnknownError as e:
        return {"error": str(e)}
    if not spec.positive_class_name:
        return {"error": f"trait {trait_name!r} defines no positive_class_name to calibrate"}

    cal_items = _classification_items(calibration_gt_dir, calibration_pred_dir, trait_name=trait_name,
                                      subject=subject, positive_value=spec.positive_class_name,
                                      attribute=attribute)
    hold_items = _classification_items(holdout_gt_dir, holdout_pred_dir, trait_name=trait_name,
                                       subject=subject, positive_value=spec.positive_class_name,
                                       attribute=attribute)
    result = resolve_classifier_operating_point(
        trait_name, calibration_items=cal_items, holdout_items=hold_items,
        experiment_id=experiment_id,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "classifier_operating_point.json", {
        "operating_point": {"classifier": {"validated_against": result["validated_against"],
                                           "value": spec.positive_class_name}},
        "validated": result["passed"],
        "failures": result["failures"],
        "sweep_data": result["sweep_data"],
        "experiment_id": experiment_id,
        "trait": trait_name,
    })
    return {
        "output_dir": str(out),
        "validated_against": result["validated_against"],
        "passed": result["passed"],
        "failures": result["failures"],
        "n_calibration_items": len(cal_items),
        "n_holdout_items": len(hold_items),
    }


@mcp.tool()
@audited
def compute_phenology(
    trait: str,
    mapping_path: str,
    predictions_by_date: dict[str, str],
    output_csv_path: str,
    classifier_pred_dirs: list[str] | None = None,
    operating_point_conf: float | None = None,
    operating_point_validated: str | None = None,
    acknowledge_unvalidated: bool = False,
) -> dict:
    """Per-plant phenology milestones from classified predictions + a plant mapping.

    A phenology milestone is a crossing of the **fraction of a plant's detected objects that are in
    the trait's positive/measured state** — an expert-defined morphological stage emitted by a
    *validated* classifier (the trait's positive class), never a geometric proxy such as bounding-box
    height. For the Phase-1 catkin trait the positive state is ``elongated`` and this reports:

        catkin_elongation_date   date most catkins have elongated (crops.yml) = the 95% crossing
                                 (provisional reading, pending breeder confirmation)
        catkin_05/50/95per_date  dates the elongated fraction crosses 5/50/95%

    Column names and crossing fractions come from ``trait``'s ``TraitSpec`` — a different registered
    trait yields its own prefixed columns without a code change (K4/K5's generalization).

    Args:
        trait: A registered trait name (``registered_traits()``) — required, no default. The
            positive class id is resolved from the prediction buckets' own recorded ``id_map`` by
            this trait's ``positive_class_name`` (a mapping fact read from the labels the run
            actually decoded through, never a pinned default or a separate registry re-derivation
            that could disagree with it).
        mapping_path: Path to a persisted plant-mapping JSON (``{date: [assignment, ...]}``
            with ``stem`` / ``plot_name`` / ``accession_name`` per assignment) — produced by
            the web plant-mapping step or ``build_plant_mapping``.
        predictions_by_date: ``{date: predictions_dir}`` — each dir holds per-image COCO/JSON
            prediction files (``<stem>.json``) from the state classifier.
        output_csv_path: Where to write the delivered per-plant CSV (e.g. ``catkin_phenology.csv``).
        classifier_pred_dirs: Bucket(s) carrying the trait's classifier-validity stamp
            (``classifier_operating_point.json``, written by ``calibrate_classifier_operating_point``)
            — reconciled from disk, never trusted from a caller-asserted string (TRAP 2). ``None``
            or a bucket with no such stamp floors the classifier dimension to unvalidated.
        operating_point_conf: The count operating point (conf) the predictions were produced
            at — stamped into the CSV; the on-disk sidecar value is preferred when present.
        operating_point_validated: An optional caller assertion of the count operating point's
            validity. It only *lowers* the result: the real state is read from each bucket's
            ``operating_point.json`` and floored against this (a missing/unvalidated sidecar
            floors the curve to ``false``). Must reconcile to a reference
            ``accepted_references("annotations")`` recognizes to deliver unacknowledged.
        acknowledge_unvalidated: Override the gate — write the CSV even when the classifier or
            operating point is unvalidated, stamping the un-validated dimension as ``false`` so
            the un-trustworthiness travels with the delivery.

    Returns a summary. **Measurement-integrity guard:** if no bucket, anywhere in the delivery, ever
    classified along the trait's positive-class axis, the positive fraction is not a valid measurement
    anywhere — the tool refuses to write the CSV and returns ``error`` with
    ``elongation_classified: false``. Rows for a plant with a partially-unclassified or partially-
    missing date still ship (with the gap disclosed via ``n_dates_unclassified``/
    ``n_dates_missing_images``) but carry no fabricated milestone dates for that plant (see
    CLAUDE.md's measurement-integrity invariant).
    """
    from tcip_mcp.traits import TraitUnknownError, get_trait

    try:
        spec = get_trait(trait)
    except TraitUnknownError as e:
        return {"error": str(e), "n_plants": 0}
    pos = spec.positive_class_name or "positive"

    mp = Path(mapping_path)
    if not mp.is_file():
        return {"error": f"mapping not found: {mapping_path}"}
    try:
        mapping = json.loads(mp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return {"error": f"could not read mapping {mapping_path}: {e}"}
    if not isinstance(mapping, dict) or not mapping:
        return {"error": f"mapping at {mapping_path} is empty or malformed"}

    positive_class_id, msg = _resolve_positive_class_id(trait, predictions_by_date)
    if positive_class_id is None:
        return {"error": (f"could not resolve the {pos} class id from any prediction bucket's "
                          f"own recorded id_map ({msg})."),
                "n_plants": 0}

    result = phenology.per_plant_phenology(
        mapping, predictions_by_date, positive_class_name=pos, spec=spec,
    )
    rows = result["rows"]

    if not result["elongation_classified"]:
        return {
            "error": (
                f"predictions carry no {pos} class anywhere in this delivery — the classifier that "
                f"produced them never assessed this trait's positive class. "
                f"The {pos} fraction is not a valid measurement — run and validate "
                f"the {pos}-state classifier before computing phenology."
            ),
            "elongation_classified": False,
            "n_plants": len(rows),
        }

    # Measurement-integrity gate (the numerator's validity): the phenotype rests on the
    # elongated/dormant call being right, so a delivery requires a classifier validated against
    # held-out GT — presence of the class is not enough. Refuse unless explicitly acknowledged,
    # and in that case stamp the CSV validated=false so the un-trustworthiness travels downstream.
    from tcip_mcp.pipelines.resolution import (
        bind_classifier_validity,
        check_delivery_gate,
        reconcile_classifier_validity,
        reconcile_operating_point_validity,
    )

    # The count operating point's validity is read from each prediction bucket's operating_point.json
    # (stamped by export_predictions), floored against any caller assertion — never trusted from the
    # caller's string alone (T5-3). A missing/unvalidated sidecar floors the whole curve to false.
    recon = reconcile_operating_point_validity(
        list(predictions_by_date.values()), asserted=operating_point_validated)
    op_state = recon["validated"]
    if operating_point_conf is None and recon["conf"] is not None:
        operating_point_conf = recon["conf"]  # prefer the on-disk conf over a caller string

    # The classifier's validity is read the same way, from classifier_operating_point.json — never a
    # caller-asserted string (K3/TRAP 2). No producer stamp anywhere -> floors to unvalidated, same as
    # the count dimension with no on-disk backing.
    classifier_recon = reconcile_classifier_validity(classifier_pred_dirs or [])
    classifier_state = classifier_recon["validated"]

    # Bind the classifier stamp to THIS delivery (stage-6 review N3): unlike the count dimension
    # (which reconciles from the SAME predictions_by_date buckets it delivers), classifier_pred_dirs
    # is a separate, caller-supplied list — reconcile_classifier_validity alone can't see whether a
    # genuinely-validated stamp was calibrated for an unrelated model or trait. A sidecar's own
    # recorded `trait`/`experiment_id` (written by calibrate_classifier_operating_point) must agree
    # with what's actually being delivered here; a foreign/unregistered checkpoint calibration
    # (experiment_id=None, the K3 owner decision) is not rejected for lacking one to compare against.
    classifier_state, classifier_binding_note = bind_classifier_validity(
        classifier_state, classifier_pred_dirs, list(predictions_by_date.values()), trait=trait,
    )

    # A delivered phenotype needs BOTH the classifier and the count operating point validated against a
    # reference sized to the trait — the one shared refuse-or-stamp gate, or an explicit acknowledge.
    gate = check_delivery_gate(
        {"classifier": classifier_state, "operating_point": op_state},
        acknowledge_unvalidated=acknowledge_unvalidated,
    )
    if not gate.ok:
        floor_note = ""
        if recon["missing_sidecars"] or recon["unvalidated_buckets"]:
            floor_note = (f" On-disk operating-point reconciliation floored the count to invalid "
                          f"(missing sidecars: {recon['missing_sidecars']}; unvalidated buckets: "
                          f"{recon['unvalidated_buckets']}).")
        if classifier_recon["missing_sidecars"]:
            floor_note += (f" No classifier_operating_point.json found in "
                           f"{classifier_recon['missing_sidecars']} — calibrate the classifier via "
                           "calibrate_classifier_operating_point before delivering.")
        if classifier_binding_note:
            floor_note += f" {classifier_binding_note}"
        return {
            "error": (
                "a delivered bloom phenotype requires BOTH a validated elongation classifier "
                f"(reconciled from classifier_operating_point.json = {classifier_state!r}) AND a "
                f"validated count operating point (reconciled from operating_point.json = "
                f"{op_state!r})." + floor_note
                + " Validate both (calibrate_classifier_operating_point for the classifier; a "
                "calibrated export_predictions for the count), or pass acknowledge_unvalidated=True "
                "to write a clearly-flagged provisional CSV."
            ),
            "positive_state_classifier_validated": gate.stamp["classifier"],
            "operating_point_validated": op_state,
            "operating_point_missing_sidecars": recon["missing_sidecars"],
            "n_plants": len(rows),
        }

    # Producing-model identity is recovered from the prediction dirs' operating_point.json sidecars
    # (stamped by export_predictions) so the delivered curve names the exact checkpoint + run behind
    # its counts. Distinct producers across dates collapse to "multiple"; absent -> left empty.
    producer = _resolve_producer_identity(predictions_by_date)

    # Carry the majority-date read-semantics marker with the delivery: whether the trait's "most in
    # state" mapping to a milestone crossing is still provisional (breeders to confirm), read from the
    # spec. The column name derives from the spec too, matching phenology_csv_columns.
    stamp = {
        "operating_point_conf": operating_point_conf,
        "operating_point_validated": gate.stamp["operating_point"],
        "positive_state_classifier_validated": gate.stamp["classifier"],
        "producer_model_sha256": producer.get("sha256"),
        "producer_experiment_id": producer.get("experiment_id"),
    }
    # Only when the spec names a majority crossing — the marker qualifies that alias, and
    # phenology_csv_columns declares the column under the same condition, so stamping it for a trait
    # without one would raise on an unknown stamp key rather than ship a permanently-blank column.
    if spec.majority_milestone:
        stamp[f"{spec.phenology_prefix}_{spec.majority_label}_provisional"] = (
            "true" if spec.majority_provisional else "false")
    csv_path = phenology.write_phenology_csv(rows, Path(output_csv_path), spec, stamp=stamp)
    # Per-milestone summary (TRAP 3 fix, K5): report reached-counts for each milestone the SPEC
    # actually declares, not a single hardcoded "50per" key — a trait authored with different
    # milestone fractions has no fabricated zero for a crossing it was never asked to report.
    n_reached: dict[str, int] = {}
    for key in phenology._milestone_targets(spec):
        col = f"{spec.phenology_prefix}_{key}_date"
        n_reached[key] = sum(1 for r in rows if r.get(col))
    return {
        "csv_path": csv_path,
        "n_plants": len(rows),
        "n_plants_reached_milestone": n_reached,
        "elongation_classified": True,
        "positive_state_classifier_validated": stamp["positive_state_classifier_validated"],
        "operating_point_validated": stamp["operating_point_validated"],
        "columns": phenology.phenology_csv_columns(spec),
    }
