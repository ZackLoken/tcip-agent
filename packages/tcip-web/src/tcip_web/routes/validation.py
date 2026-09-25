"""Validation routes: promote a completed review into a validation reference.

validate_reference reconstructs a review's verdicts into the COCO records resolve_operating_point
consumes (the review_calibration adapter) and runs them through the disjoint-split + count-bias
gate and conf-censoring guard, so a review can only stamp a bucket's operating_point.json
VALIDATED_REVIEW_CONFIRMED with a record outside the bucket answering for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_mcp.audit import AuditEntryNotWritten
from tcip_web.routes.audit_gap import audit_gap_409

from tcip_web.routes.review import (
    _bucket_of_dir, _get_engine, _guard_path, _prediction_digest,
)

router = APIRouter(prefix="/api/review", tags=["review"])


def _recorded_prediction_digests(image_state: dict) -> set[Optional[str]]:
    """Every prediction-document identity recorded against one reviewed image at review time.

    The image-level producer fact a confirmed negative carries, plus the one on each verdict entry.
    A recorded identity carrying no digest reads as None, the same value an image with no
    prediction document records.
    """
    from tcip_annotation.verdicts import decode_verdict

    identities = [image_state.get("producer_identity")]
    identities += [decode_verdict(d).producer_identity for d in image_state.get("detections") or []]
    return {i.get("prediction_digest") for i in identities if isinstance(i, dict)}


# ── Promote a completed review into a validation reference ─────────────────


class ValidateReferenceRequest(BaseModel):
    dataset_root: str
    trait: str
    # The prediction bucket whose review is being promoted: the per-image prediction dir the
    # delivery gate reads an ``operating_point.json`` from.
    pred_dir: Optional[str] = None
    # The object identity this reference validates. Required by the route; kept optional here so
    # an absent one earns the route's own named 400 rather than a generic pydantic error.
    subject: Optional[str] = None


class ValidateReferenceResponse(BaseModel):
    # True only when the review cleared the identical gate the backend uses (or the bucket was already
    # validated). A refusal is surfaced honestly here, never silently upgraded.
    validated: bool
    reference: Optional[str]  # a resolution.py validated_against value ("false" when unvalidated)
    reviewed_image_count: int
    conf: Optional[float]  # the derived count operating point (for transparency)
    reason: str  # plain-language, breeder-facing, always present
    buckets_stamped: list[str]


@router.post("/validate_reference")
def validate_reference(req: ValidateReferenceRequest) -> ValidateReferenceResponse:
    """Promote a completed review of one prediction bucket into a validation reference for its
    (model, trait, date).

    The review's verdicts become the COCO records ``resolve_operating_point`` consumes (the
    ``review_calibration`` adapter) and run through the disjoint-split, count-bias and
    conf-censoring gate. A passing gate is earned through ``open_validation``/``seal_validation``,
    so the bucket's ``operating_point.json`` claims ``VALIDATED_REVIEW_CONFIRMED`` only with a
    record outside the bucket answering for it; a refusal stamps an honest ``validated=false`` and
    returns the reason.

    A stamp whose claim no record answers for is promotable over. Refuses (400) a request naming no
    dataset root or subject, a bucket under another dataset root, a bucket with no
    ``operating_point.json`` stamp, and an explicit tile edge whose stamp omits why it is trusted;
    a review whose prediction documents changed since the verdicts earns nothing. A lost audit line
    after a record or stamp landed answers 409 with the buckets stamped so far.
    """
    if not req.dataset_root:
        raise HTTPException(
            400,
            "validate_reference requires the dataset root this reference is scoped to; name "
            "one rather than leaving it unstated.",
        )
    if not req.subject:
        raise HTTPException(
            400,
            "validate_reference requires the subject this reference validates; name one rather "
            "than leaving it unstated.",
        )
    pred_dir = _guard_path(req.pred_dir)
    if not pred_dir:
        return ValidateReferenceResponse(
            validated=False, reference=None, reviewed_image_count=0, conf=None,
            reason="No predictions are selected to validate. Choose a model with predictions for "
                   "this dataset, then try again.",
            buckets_stamped=[])

    from tcip_mcp.dataset_layout import bucket_dataset_root
    from tcip_mcp.pipelines.data.splits import same_directory
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar, verify_stamp_binding
    from tcip_mcp.prediction_buckets import bucket_stems
    from tcip_store.errors import DecodeError, SchemaVersionRefused, StoreBusy

    # A bucket answering a different root than the stated one is another dataset's evidence; a
    # bucket under no dataset root answers nothing to cross-check and is legitimate work.
    named_root = bucket_dataset_root(pred_dir)
    if named_root is not None and not same_directory(named_root, req.dataset_root):
        raise HTTPException(
            400,
            f"the predictions at {pred_dir} belong to dataset {named_root}, not to "
            f"{req.dataset_root}, the dataset this request names. Validate a bucket under its own "
            "dataset root, so the verdicts, the validation record and the stamp all hang off one "
            "dataset.")

    stems = bucket_stems(pred_dir)
    engine = _get_engine(req.dataset_root)
    # The verdicts recorded against this bucket, so a stem under two buckets contributes only
    # what was reviewed here.
    reviewed = {
        name: data
        for name, data in engine.image_states(_bucket_of_dir(pred_dir)).items()
        if data.get("img_status") == "completed"
    }
    completed = {name: data for name, data in reviewed.items() if Path(name).stem in stems}
    n = len(completed)

    # Strict: an undecodable stamp is never one a review answers for the way an absent one is.
    try:
        sidecar = read_operating_point_sidecar(pred_dir, strict=True) or {}
    except StoreBusy as exc:
        raise HTTPException(503, str(exc)) from exc
    except (DecodeError, SchemaVersionRefused) as exc:
        raise HTTPException(400, str(exc)) from None
    digest_memo: dict[str, str] = {}
    binding = verify_stamp_binding(sidecar, pred_dir, document="operating_point",
                                   digest_memo=digest_memo)
    stamped_op = sidecar.get("operating_point") or {}
    if binding.claimed and binding.ok:
        return ValidateReferenceResponse(
            validated=True, reference=(stamped_op.get("conf") or {}).get("validated_against"),
            reviewed_image_count=n, conf=None,
            reason="These predictions are already validated, so a review reference isn't needed here.",
            buckets_stamped=[])

    if n == 0:
        return ValidateReferenceResponse(
            validated=False, reference=None, reviewed_image_count=0, conf=None,
            reason="No completed reviews yet for this model on this date. Review the predictions and "
                   "mark the images Reviewed, then try again.",
            buckets_stamped=[])

    if not sidecar:
        raise HTTPException(
            400,
            f"{pred_dir} carries no operating_point.json stamp: no producer wrote a checkpoint, "
            "experiment or generation conf this validation could rest on. A staged bucket is "
            "reviewed through the accept path and is never promoted to a validation reference."
        )

    # A prediction document that changed, appeared or vanished since review is evidence for nothing.
    diverged = sorted(
        name for name, data in reviewed.items()
        if any(recorded != _prediction_digest(pred_dir, name)
               for recorded in _recorded_prediction_digests(data))
    )
    if diverged:
        return ValidateReferenceResponse(
            validated=False, reference=None, reviewed_image_count=n, conf=None,
            reason=f"The predictions for {', '.join(diverged)} are no longer the ones that were "
                   "reviewed: a prediction file has been added, replaced or removed in this bucket "
                   "since those verdicts were recorded. Re-running inference on a reviewed bucket "
                   "writes the next free variant of it instead (the same '@r2' redirect the "
                   "immutability guard makes), which keeps this review intact and can be reviewed "
                   "and validated on its own.",
            buckets_stamped=[])

    from tcip_mcp.dataset_layout import annotation_dir, prediction_bucket_date
    from tcip_mcp.pipelines.feedback import (
        describe_review_validation,
        resolve_operating_point_from_review,
        review_conf_threshold,
        review_reference_hash,
        review_to_records,
    )
    from tcip_mcp.pipelines.resolution import tile_size_source_of
    from tcip_mcp.traits import TraitUnknownError

    review_state = {"image": completed}
    # The producing run named by the bucket's own stamp, never asserted, so the calibration's
    # train-disjointness gate checks the reviewed images against that run's training split.
    review_experiment_id = sidecar["experiment_id"]
    bucket_identities = [{"checkpoint_sha256": sidecar["checkpoint_sha256"],
                          "experiment_id": review_experiment_id}]
    # The effective staging floor is max(generation conf, the verdicts' own recorded
    # conf_threshold); either half unknown makes it None (fails closed).
    generation_conf = (stamped_op.get("conf") or {}).get("value")
    review_conf = review_conf_threshold(review_state, bucket_identities=bucket_identities,
                                        only_completed=True)
    staged_conf_floor = (
        max(float(generation_conf), review_conf)
        if isinstance(generation_conf, (int, float)) and review_conf is not None
        else None
    )

    tile_size_prov = stamped_op.get("tile_size") or {}
    tiled_prov = stamped_op.get("tiled") or {}
    review_tile_size = tile_size_prov.get("value")
    # From validated_against, not the bare source field, which a native-ratio edge shares with a
    # real persisted one: reading source alone would silently re-validate native-ratio on review.
    review_tile_size_source = tile_size_source_of(
        tile_size_prov.get("validated_against") if review_tile_size is not None else None,
        tile_size=review_tile_size)
    review_tile_size_derived_from = (
        tile_size_prov.get("derived_from") if review_tile_size is not None else None)
    review_tiled = tiled_prov.get("value")
    recorded_tiled_source = tiled_prov.get("source") if review_tiled is not None else None
    review_tiled_source = (
        recorded_tiled_source if isinstance(recorded_tiled_source, str) else "default")

    if review_tile_size_source == "explicit" and review_tile_size_derived_from is None:
        raise HTTPException(
            400,
            f"the predictions at {pred_dir} carry an explicit tile edge but their stamp omits why "
            "it is trusted, so the review promotion cannot state a derivation for the validated "
            "claim. Re-export the predictions from a run that records one.",
        )

    # One spelling of the evidence, shared by the resolver call and open_validation's own record.
    resolver_inputs: dict[str, Any] = {
        "review_state": review_state,
        "only_completed": True,
        "bucket_identities": bucket_identities,
        "staged_conf_floor": staged_conf_floor,
        "tile_size": review_tile_size,
        "tile_size_source": review_tile_size_source,
        "tile_size_derived_from": review_tile_size_derived_from,
        "tiled": review_tiled,
        "tiled_source": review_tiled_source,
        # The root the verdict store was opened on, so the split lock travels with the verdicts.
        "scope_root": req.dataset_root,
        # Where the reviewed images' own ground truth lives: a bound checkpoint's selection check
        # narrows the run's val members to this directory, as the calibration door does.
        "calibration_labels_dir": str(
            annotation_dir(req.dataset_root, prediction_bucket_date(pred_dir))),
        "subject": req.subject,
    }
    try:
        bundle = resolve_operating_point_from_review(
            trait_name=req.trait, experiment_id=review_experiment_id, **resolver_inputs)
    except TraitUnknownError:
        raise HTTPException(
            400,
            f"a validation reference is not defined for trait {req.trait!r} yet. This action is "
            "available for traits the platform can calibrate a count operating point for.",
        ) from None
    except ValueError as exc:
        # A locked cal/holdout split refusing this call: a reviewed image was deleted/renamed
        # since the split locked, or the lock file itself is corrupt.
        raise HTTPException(400, str(exc)) from None
    except AuditEntryNotWritten as exc:
        # The split lock was drawn and its receipt lost; nothing was validated or sealed.
        raise audit_gap_409(exc, {exc.tool: exc.arguments, "validated": False}) from None

    result = describe_review_validation(bundle, reviewed_image_count=n)

    from tcip_mcp.pipelines.resolution import (
        claim_payload,
        open_validation,
        seal_validation,
        update_sidecar,
    )
    from tcip_mcp.prediction_buckets import review_state_dir_of

    ref_hash = review_reference_hash(
        review_to_records(review_state, bucket_identities=bucket_identities, subject=req.subject))
    draft = None
    if result["validated"]:
        try:
            draft = open_validation(
                document="operating_point",
                evidence={"resolver": "resolve_operating_point_from_review",
                          "inputs": resolver_inputs},
                trait=req.trait,
                checkpoint_sha256=sidecar["checkpoint_sha256"],
                producing_experiment_id=review_experiment_id,
                reference_inputs={
                    "dataset_root": req.dataset_root,
                    "scope_roots": {"verdicts": str(review_state_dir_of(req.dataset_root))},
                    "stated_values": {"review_reference_hash": ref_hash, "review_image_count": n},
                },
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None

    def _stamp_body(stored: dict) -> dict:
        """This promotion merged over whatever the producing run left in ``stored``: the trait only
        when a gate was cleared, so an honest placeholder claims no scope at all.
        """
        merged = {**stored, "operating_point": bundle.to_provenance()["operating_point"],
                  "validated": result["validated"],
                  "shippable_issues": bundle.shippable_issues()}
        if draft is not None:
            merged["trait"] = req.trait
        return merged

    stamped: list[str] = []
    gap: AuditEntryNotWritten | None = None
    try:
        Path(pred_dir).mkdir(parents=True, exist_ok=True)
        # Sealed outside the stamp's lock: no store write may open inside another's transaction.
        earned = _stamp_body(sidecar)
        if draft is not None:
            earned = seal_validation(
                draft, dataset_root=req.dataset_root, bucket_dirs=[pred_dir], stamp_body=earned)

        def _promote(stored: dict) -> dict | None:
            """Merge this promotion into the stamp as stored, inside its lock: a stamp a record
            now answers for is left as it is, and the pointer is merged only while the stamp
            still makes the claim ``earned`` was sealed over."""
            now = verify_stamp_binding(stored, pred_dir, document="operating_point",
                                       digest_memo=digest_memo)
            if now.claimed and now.ok:
                return None
            merged = _stamp_body(stored)
            if draft is None:
                return merged
            if claim_payload(merged, document="operating_point") != claim_payload(
                    earned, document="operating_point"):
                return None
            merged["validated_by"] = earned["validated_by"]
            return merged

        try:
            if update_sidecar(pred_dir, _promote):
                stamped.append(pred_dir)
        except AuditEntryNotWritten:
            stamped.append(pred_dir)  # the stamp landed; only its line was lost
            raise
    except StoreBusy as exc:
        # Contention is a retryable infrastructure fault, never a malformed request.
        raise HTTPException(503, str(exc)) from exc
    except (ValueError, SchemaVersionRefused, DecodeError) as exc:
        raise HTTPException(400, str(exc)) from None
    except AuditEntryNotWritten as exc:
        # A record or stamp already landed; only its library's audit line failed.
        gap = exc
    committed = ValidateReferenceResponse(
        validated=bool(result["validated"]),
        reference=result["reference"],
        reviewed_image_count=n,
        conf=result["conf"],
        reason=result["reason"],
        buckets_stamped=stamped,
    )
    if gap is not None:
        raise audit_gap_409(gap, committed) from gap
    return committed
