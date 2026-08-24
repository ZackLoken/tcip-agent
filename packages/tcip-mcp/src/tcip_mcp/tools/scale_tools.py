"""Physical per-pixel scale calibration: the delivery-gating producer for ``resolve_scale.json``.

Before this tool, no platform code produced a validated physical scale; every ``resolve_scale.json``
on disk was hand-authored and floored at delivery for want of a record that answered for it (see
``pipelines.measurement.mask_geometry.resolve_scale`` and ``pipelines.resolution.
reconcile_scale_validity``). ``calibrate_physical_scale`` closes that gap: it reads a breeder's own
reference measurements, runs ``pipelines.measurement.scale_calibration.resolve_physical_scale``'s
locked calibration/holdout gate over them, and stamps the result into a prediction bucket's
``resolve_scale.json``, an audit seam and a delivery-gating write and therefore a tool rather than a
script.
"""

from __future__ import annotations

import csv
import hashlib
import logging
from pathlib import Path

from tcip_mcp.audit import audited
from tcip_mcp.server import mcp

logger = logging.getLogger(__name__)


def _read_reference_csv(csv_path: str) -> dict[str, dict[str, float | str]]:
    """``stem -> {"physical_extent": float, "unit": str}`` from the breeder's reference CSV
    (``image_stem, physical_extent, unit``, one row per reference image), the same standing a
    ground-truth CSV has for ``calibrate_ordinal_regression_operating_point``."""
    out: dict[str, dict[str, float | str]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            stem, extent, row_unit = row[0].strip(), row[1].strip(), row[2].strip()
            out[stem] = {"physical_extent": float(extent), "unit": row_unit}
    return out


@mcp.tool()
@audited
def calibrate_physical_scale(
    trait: str,
    pred_dir: str,
    dataset_root: str,
    unit: str,
    reference_subject: str,
    labels_dir: str,
    reference_csv: str,
    capture_id: str | None = None,
    group_by: str = "tile_prefix",
    group_key_map: dict[str, str] | None = None,
    seed: int = 0,
    holdout_ratio: float = 0.5,
) -> dict:
    """Derive and validate a physical per-pixel scale, and stamp it into ``pred_dir``'s
    ``resolve_scale.json``.

    The reference has two halves, both real evidence, never an agent-supplied candidate: the pixel
    extent comes from ``reference_subject`` annotated as a polygon or mask on each reference image
    under ``labels_dir`` (its principal-axis extent, orientation-independent, never a bounding box's
    long side, which is refused outright); the physical extent comes from the breeder, in
    ``reference_csv`` (``image_stem, physical_extent, unit``). Every reference stem must be an image
    of ``pred_dir`` itself: a scale claim is a fact about the bucket's own imagery, and a reference
    photographed in some other capture says nothing about this one.

    The scale is derived on a locked calibration half of the references and validated against the
    holdout half it was not derived from (``scale_calibration.resolve_physical_scale``), against
    ``trait``'s own authored ``scale_tolerance_frac``; a trait with none authored refuses rather than
    validating against a platform-invented number.

    On a pass, ``open_validation``/``seal_validation`` earn and record the claim the same two-phase
    way every other calibration door does, with ``covered_buckets`` keyed by ``pred_dir``'s own image
    stems (never its prediction bytes): re-exporting predictions over the same images leaves the
    scale claim standing, while an image added to or removed from the bucket floors it, a real
    reason to re-run this tool. ``write_sidecar`` writes the stamp last, whether or not the gate
    passed, so a failed calibration still leaves a readable, honestly-unvalidated record of what was
    tried.

    Args:
        trait: The registered trait this physical scale is earned for; a delivery reading a
            different trait's scale floors (``reconcile_scale_validity``).
        pred_dir: The prediction bucket to stamp; the scale claim binds to this bucket's own image
            stems.
        dataset_root: The dataset this calibration's claim hangs off; the reference locations and
            the locked split are recorded and stored against it.
        unit: The physical unit every reference (and the stamped scale) is in; a reference CSV row
            in a different unit refuses rather than being silently converted.
        reference_subject: The ``classes.json`` subject the reference object is annotated as.
        labels_dir: Directory holding one ``<stem>.json`` annotation file per reference image.
        reference_csv: The breeder's own physical-measurement CSV.
        capture_id: The capture this scale is scoped to, when it is capture-specific (a handheld
            standoff that can vary image to image); ``None`` (default) scopes it to the whole
            bucket.
        group_by / group_key_map / seed / holdout_ratio: The locked cal/holdout split's grouping
            policy, same semantics as the ordinal/regression calibrator's own arguments; only the
            first call for this reference's identity draws the split.
    """
    from tcip_mcp.traits import TraitUnknownError, get_trait

    try:
        spec = get_trait(trait)
    except TraitUnknownError as e:
        return {"error": str(e)}
    if spec.scale_tolerance_frac is None:
        return {"error": (
            f"trait {trait!r} has no authored scale_tolerance_frac; a physical-scale gate has no "
            "platform-provisional fallback for how much reference disagreement is acceptable. "
            "Author it via update_trait_spec_fields before calibrating this trait's scale."
        )}

    from tcip_mcp.tools.phenology_tools import _stated_root_disagreement

    disagreement = _stated_root_disagreement(
        dataset_root, {"labels_dir": labels_dir, "pred_dir": pred_dir})
    if disagreement:
        return {"error": disagreement}

    from tcip_annotation import json_io
    from tcip_annotation.state import Polygon
    from tcip_mcp.prediction_buckets import bucket_stems

    bucket_image_stems = bucket_stems(pred_dir)
    if not bucket_image_stems:
        return {"error": f"no prediction file(s) found in {pred_dir}"}

    try:
        references_raw = _read_reference_csv(reference_csv)
    except (OSError, ValueError) as exc:
        return {"error": f"could not read reference_csv {reference_csv!r}: {exc}"}
    if not references_raw:
        return {"error": f"{reference_csv} names no reference image"}

    outside = sorted(set(references_raw) - bucket_image_stems)
    if outside:
        return {"error": (
            f"reference stem(s) {outside} named in {reference_csv} are not images of the bucket "
            f"at {pred_dir}; a physical-scale claim binds to the imagery it is stamped into, and a "
            "reference photographed in some other capture says nothing about this one."
        )}

    from tcip_mcp.pipelines.measurement.mask_geometry import principal_axis_extent_of_points

    references: dict[str, dict] = {}
    for stem, row in sorted(references_raw.items()):
        label_path = Path(labels_dir) / f"{stem}.json"
        if not label_path.is_file():
            return {"error": f"no annotation file for reference image {stem!r} at {label_path}"}
        annotations = [a for a in json_io.read_annotations(str(label_path))
                      if a.subject == reference_subject]
        if len(annotations) != 1:
            return {"error": (
                f"{label_path} carries {len(annotations)} {reference_subject!r} annotation(s); a "
                "reference image must carry exactly one."
            )}
        geometry = annotations[0].geometry
        if not isinstance(geometry, Polygon):
            return {"error": (
                f"{label_path}'s {reference_subject!r} annotation is a "
                f"{type(geometry).__name__ if geometry is not None else 'None'}, not a Polygon/"
                "mask; a bounding box's long side is the object's projected extent (up to "
                "root-two times the true length at 45 degrees), refused as a reference geometry. "
                "Annotate the reference object as a polygon or mask instead."
            )}
        points = [p for ring in geometry.rings for p in ring]
        pixel_extent = principal_axis_extent_of_points(points)
        references[stem] = {
            "physical_extent": row["physical_extent"], "unit": row["unit"],
            "pixel_extent": pixel_extent,
        }

    from tcip_mcp.pipelines.measurement.scale_calibration import resolve_physical_scale
    from tcip_mcp.pipelines.resolution import csv_dataset_hash, dataset_hash

    identity_hash = hashlib.sha256(
        f"{csv_dataset_hash(reference_csv)}:"
        f"{dataset_hash(labels_dir, stems=sorted(references_raw))}".encode()
    ).hexdigest()[:16]

    result = resolve_physical_scale(
        unit=unit, references=references, tolerance_frac=spec.scale_tolerance_frac,
        dataset_root=dataset_root, identity_hash=identity_hash, group_by=group_by,
        group_key_map=group_key_map, seed=seed, holdout_ratio=holdout_ratio,
        capture_id=capture_id,
    )

    from tcip_mcp.pipelines.resolution import open_validation, seal_validation, write_sidecar

    n_cal_stems = len(result["sweep_data"].get("calibration_implied_scales") or {})
    stamp = {
        "operating_point": {"scale": {
            "name": f"scale_{unit}_per_px", "value": result["value"], "unit": unit,
            "source": "derived",
            "derived_from": f"mean of {n_cal_stems} {reference_subject!r} reference object(s), "
                            "the calibration half of the locked reference split",
            "requires_validation": True, "validation_kind": "physical",
            "validated_against": result["validated_against"],
            "capture_scoped": capture_id is not None, "capture_id": capture_id,
        }},
        "validated": result["passed"], "validated_by": None,
        "failures": result["failures"], "sweep_data": result["sweep_data"],
        "trait": trait, "reference_subject": reference_subject,
        "reference_csv": _relative_to_root(reference_csv, dataset_root),
        "produced_at": _now_iso(),
    }
    if result["passed"]:
        draft = open_validation(
            document="resolve_scale",
            evidence={"resolver": "resolve_physical_scale",
                      "inputs": {"unit": unit, "references": references,
                                 "tolerance_frac": spec.scale_tolerance_frac,
                                 "dataset_root": dataset_root, "identity_hash": identity_hash,
                                 "group_by": group_by, "group_key_map": group_key_map,
                                 "seed": seed, "holdout_ratio": holdout_ratio,
                                 "capture_id": capture_id}},
            trait=trait, checkpoint_sha256=None, producing_experiment_id=None,
            reference_inputs={
                "dataset_root": dataset_root,
                "label_dirs": {"reference": labels_dir},
                "label_csvs": {"reference": reference_csv},
                "stated_values": {"split_identity": identity_hash},
            },
        )
        _, stamp = seal_validation(draft, dataset_root=dataset_root, bucket_dirs=[pred_dir],
                                   stamp_body=stamp)
    write_sidecar(pred_dir, stamp, "resolve_scale")
    return {
        "pred_dir": pred_dir,
        "validated_against": result["validated_against"],
        "passed": result["passed"],
        "failures": result["failures"],
        "validated_by": stamp["validated_by"],
        "value": result["value"],
        "unit": unit,
        "n_references": len(references),
    }


def _relative_to_root(path: str, dataset_root: str) -> str:
    """``path`` expressed against ``dataset_root``, or the resolved absolute path when it does not
    sit under it (a CSV over a loose images directory is legitimate work, see
    ``resolution._relative_location``, the same reasoning applied here to the stamp's own
    ``reference_csv`` field)."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(dataset_root).resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
