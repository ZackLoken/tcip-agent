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


def _resolve_positive_class_id(trait_name: str, classes_json_path: str | None) -> tuple[int | None, str]:
    """Resolve the trait's positive (elongated) class id from ``classes.json`` BY NAME.

    The id is a mapping fact read from the labels' class map, never a pinned magic default. Returns
    ``(class_id, message)``; ``class_id`` is ``None`` when it cannot be resolved honestly (no class
    map, or the trait's ``positive_class_name`` is absent from it) so the caller refuses rather than
    silently falling back to a guessed id.
    """
    from tcip_mcp.project_paths import resolve_state
    from tcip_mcp.traits import get_trait

    name = get_trait(trait_name).positive_class_name
    if not name:
        return None, f"trait {trait_name!r} defines no positive_class_name"
    if classes_json_path:
        candidates = [Path(classes_json_path)]
    else:
        state = Path(".tcip") / "state"
        candidates = [resolve_state(state / "classes" / f"{trait_name}.json"),
                      resolve_state(state / "classes.json")]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        for cid, entry in data.items():
            if isinstance(entry, dict) and entry.get("name") == name:
                try:
                    return int(cid), f"resolved {name!r} -> class {cid} from {path}"
                except (TypeError, ValueError):
                    continue
        present = [e.get("name") for e in data.values() if isinstance(e, dict)]
        return None, f"class map {path} has no class named {name!r} (classes: {present})"
    return None, ("no class map found — name the elongated class via write_class_map or the GUI so "
                  "the id is derived from labels")


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


@mcp.tool()
@audited
def compute_phenology(
    mapping_path: str,
    predictions_by_date: dict[str, str],
    output_csv_path: str,
    positive_class_id: int | None = None,
    classes_json_path: str | None = None,
    classifier_validated: str | None = None,
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

    Column names and crossing fractions come from the trait's ``TraitSpec``; a different trait yields
    its own prefixed columns without a code change.

    Args:
        mapping_path: Path to a persisted plant-mapping JSON (``{date: [assignment, ...]}``
            with ``stem`` / ``plot_name`` / ``accession_name`` per assignment) — produced by
            the web plant-mapping step or ``build_plant_mapping``.
        predictions_by_date: ``{date: predictions_dir}`` — each dir holds per-image COCO/JSON
            prediction files (``<stem>.json``) from the state classifier.
        output_csv_path: Where to write the delivered per-plant CSV (e.g. ``catkin_phenology.csv``).
        positive_class_id: Class id the classifier assigns to the trait's positive/measured state
            (for catkin, "elongated"). ``None`` (default) derives it from ``classes.json`` by the
            trait's positive class name (a mapping fact from the labels, never a pinned default) —
            the tool refuses if that name is absent rather than guessing an id. An explicit id is
            honored as-is.
        classes_json_path: Optional explicit path to the class map used to resolve the positive
            class id; ``None`` uses the trait's canonical ``.tcip/state/classes`` map.
        classifier_validated: The state classifier's ``validated_vs_gt`` state; a CSV
            is only written unacknowledged when this is ``validated_held_out``.
        operating_point_conf: The count operating point (conf) the predictions were produced
            at — stamped into the CSV; the on-disk sidecar value is preferred when present.
        operating_point_validated: An optional caller assertion of the count operating point's
            validity. It only *lowers* the result: the real state is read from each bucket's
            ``operating_point.json`` and floored against this (a missing/unvalidated sidecar
            floors the curve to ``false``). Must reconcile to a shippable reference
            (``validated_held_out`` / ``review_confirmed``) to deliver unacknowledged.
        acknowledge_unvalidated: Override the gate — write the CSV even when the classifier or
            operating point is unvalidated, stamping the un-validated dimension as ``false`` so
            the un-trustworthiness travels with the delivery.

    Returns a summary. **Measurement-integrity guard:** if the predictions carry no positive-state
    class anywhere (for catkin, the elongation class), the positive fraction is not a valid
    measurement — the tool refuses to write the CSV and returns ``error`` with
    ``elongation_classified: false`` so an unvalidated curve is never delivered (see the CLAUDE.md
    measurement-integrity invariant).
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

    from tcip_mcp.traits import get_trait

    trait_name = "catkin"
    spec = get_trait(trait_name)
    pos = spec.positive_class_name or "positive"
    if positive_class_id is None:
        positive_class_id, msg = _resolve_positive_class_id(trait_name, classes_json_path)
        if positive_class_id is None:
            return {"error": (f"could not resolve the {pos} class id from the class map by name "
                              f"({msg}). Name the {pos} class so the id is derived from the "
                              "labels, or pass positive_class_id explicitly."),
                    "n_plants": 0}

    result = phenology.per_plant_phenology(
        mapping, predictions_by_date, elongated_class_id=positive_class_id
    )
    rows = result["rows"]

    if not result["elongation_classified"]:
        return {
            "error": (
                f"predictions carry no {pos} class "
                f"(class {positive_class_id}); classes seen: {result['classes_seen']}. "
                f"The {pos} fraction is not a valid measurement — run and validate "
                f"the {pos}-state classifier before computing phenology."
            ),
            "elongation_classified": False,
            "classes_seen": result["classes_seen"],
            "n_plants": len(rows),
        }

    # Measurement-integrity gate (the numerator's validity): the phenotype rests on the
    # elongated/dormant call being right, so a delivery requires a classifier validated against
    # held-out GT — presence of the class is not enough. Refuse unless explicitly acknowledged,
    # and in that case stamp the CSV validated=false so the un-trustworthiness travels downstream.
    from tcip_mcp.pipelines.resolution import (
        check_delivery_gate,
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

    # A delivered phenotype needs BOTH the classifier and the count operating point validated against a
    # reference sized to the trait — the one shared refuse-or-stamp gate, or an explicit acknowledge.
    gate = check_delivery_gate(
        {"classifier": classifier_validated, "operating_point": op_state},
        acknowledge_unvalidated=acknowledge_unvalidated,
    )
    if not gate.ok:
        floor_note = ""
        if recon["missing_sidecars"] or recon["unvalidated_buckets"]:
            floor_note = (f" On-disk operating-point reconciliation floored the count to invalid "
                          f"(missing sidecars: {recon['missing_sidecars']}; unvalidated buckets: "
                          f"{recon['unvalidated_buckets']}).")
        return {
            "error": (
                "a delivered bloom phenotype requires BOTH a validated elongation classifier "
                f"(got classifier_validated={classifier_validated!r}) AND a validated count operating "
                f"point (reconciled from the buckets' operating_point.json = {op_state!r})." + floor_note
                + " Validate both (evaluate_model task='classification' for the classifier; a calibrated "
                "export_predictions for the count), or pass acknowledge_unvalidated=True to write a "
                "clearly-flagged provisional CSV."
            ),
            "elongation_classifier_validated": gate.stamp["classifier"],
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
    provisional = "true" if spec.majority_provisional else "false"
    provisional_col = f"{spec.phenology_prefix}_{spec.majority_label}_provisional"
    stamp = {
        "operating_point_conf": operating_point_conf,
        "operating_point_validated": gate.stamp["operating_point"],
        "elongation_classifier_validated": gate.stamp["classifier"],
        provisional_col: provisional,
        "producer_model_sha256": producer.get("sha256"),
        "producer_experiment_id": producer.get("experiment_id"),
    }
    csv_path = phenology.write_phenology_csv(rows, Path(output_csv_path), stamp=stamp)
    n_with_50 = sum(1 for r in rows if r.get("catkin_50per_date"))
    return {
        "csv_path": csv_path,
        "n_plants": len(rows),
        "n_plants_reached_50per": n_with_50,
        "elongation_classified": True,
        "elongation_classifier_validated": stamp["elongation_classifier_validated"],
        "operating_point_validated": stamp["operating_point_validated"],
        "classes_seen": result["classes_seen"],
        "columns": phenology.PHENOLOGY_CSV_COLUMNS,
    }
