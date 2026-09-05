"""Validation routes: promote a completed review into a validation reference.

validate_reference reconstructs a review's verdicts into the COCO records
resolve_operating_point consumes (the review_calibration adapter) and runs them through the
identical disjoint-split + count-bias gate and conf-censoring guard the held-out-GT path uses,
so a review can only stamp a bucket's operating_point.json VALIDATED_REVIEW_CONFIRMED with a
record outside the bucket answering for it.

Shares review.py's engine cache, audit writer and bucket-key helpers, the same verdict store
the review routes read and write, rather than a second implementation of any of them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_web.routes.review import (
    _audit, _bucket_of_dir, _get_engine, _guard_path, _prediction_digest,
)

router = APIRouter(prefix="/api/review", tags=["review"])


def _dataset_root_of_all(paths: Iterable[Optional[str]]) -> Optional[str]:
    """The one dataset root every path in paths belongs to, or None when that is not a single
    answer.

    A cross-check that a request naming one dataset is not pointing at another's files, never a
    source of the root itself: the request states that. None means the paths answer nothing to
    cross-check against (a bucket under no dataset root is legitimate work), or that they answer
    several things, which is the refusal a request accepting more than one prediction directory
    would need.
    """
    from tcip_mcp.dataset_layout import dataset_root_of

    roots = {str(root) for p in paths if p and (root := dataset_root_of(p)) is not None}
    return roots.pop() if len(roots) == 1 else None


def _recorded_prediction_digests(image_state: dict) -> set[Optional[str]]:
    """Every prediction-document identity recorded against one reviewed image at review time.

    The image-level producer fact a confirmed negative carries, plus the one on each verdict
    entry. A recorded identity carrying no digest reads as None, the same value an image with no
    prediction document records, so an unrecorded identity is compared rather than waved through.
    """
    identities = [image_state.get("producer_identity")]
    identities += [d.get("producer_identity") for d in image_state.get("detections") or []]
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
    """Promote a completed review session into a validation reference for its (model, trait, date-set).

    Reconstructs the review verdicts into the COCO records ``resolve_operating_point`` consumes (the
    ``review_calibration`` adapter) and runs them through the identical disjoint-split + count-bias
    gate and conf-censoring guard the held-out-GT path uses: no shortcut to "validated". A passing
    gate is earned through ``open_validation``/``seal_validation``, which append the validation
    record and hand back the stamp carrying its pointer, so the bucket's ``operating_point.json`` can
    only claim ``VALIDATED_REVIEW_CONFIRMED`` with a record outside the bucket answering for it; on
    refusal an honest ``validated=false`` placeholder is written and the reason is returned.

    The promotion verifies before it decides. A bucket whose stamp claims validation that no record
    answers for is treated as unvalidated and is promotable over, and a review whose prediction
    documents are no longer the ones the reviewer saw earns nothing at all.
    """
    if not req.subject:
        raise HTTPException(
            400,
            "validate_reference requires the subject this reference validates; name one rather "
            "than leaving it unstated.",
        )
    pred_dir = _guard_path(req.pred_dir)
    bucket_dirs = [pred_dir] if pred_dir else []
    if not bucket_dirs:
        return ValidateReferenceResponse(
            validated=False, reference=None, reviewed_image_count=0, conf=None,
            reason="No predictions are selected to validate. Choose a model with predictions for "
                   "this dataset, then try again.",
            buckets_stamped=[])

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar, verify_stamp_binding
    from tcip_mcp.prediction_buckets import bucket_stems

    # A bucket answering a different root than the stated one is another dataset's evidence.
    named_root = _dataset_root_of_all(bucket_dirs)
    if named_root is not None and Path(named_root).resolve() != Path(req.dataset_root).resolve():
        raise HTTPException(
            400,
            f"the predictions at {pred_dir} belong to dataset {named_root}, not to "
            f"{req.dataset_root}, the dataset this request names. Validate a bucket under its own "
            "dataset root, so the verdicts, the validation record and the stamp all hang off one "
            "dataset.")

    stems = bucket_stems(*bucket_dirs)
    engine = _get_engine(req.dataset_root)
    # The verdicts recorded against the bucket being promoted, so a stem that exists under two
    # buckets contributes only what was reviewed here.
    reviewed = {
        name: data
        for name, data in engine.image_states(_bucket_of_dir(pred_dir)).items()
        if data.get("img_status") == "completed"
    }
    completed = {name: data for name, data in reviewed.items() if Path(name).stem in stems}
    n = len(completed)

    # A claim no record answers for is an assertion: unvalidated, and promotable over.
    sidecars = {d: (read_operating_point_sidecar(d) or {}) for d in bucket_dirs}
    digest_memo: dict[str, str] = {}
    bindings = {d: verify_stamp_binding(sc, d, document="operating_point", digest_memo=digest_memo)
                for d, sc in sidecars.items()}
    if all(b.claimed and b.ok for b in bindings.values()):
        ref = next((((sc.get("operating_point") or {}).get("conf") or {}).get("validated_against")
                    for sc in sidecars.values()), None)
        return ValidateReferenceResponse(
            validated=True, reference=ref, reviewed_image_count=n, conf=None,
            reason="These predictions are already validated, so a review reference isn't needed here.",
            buckets_stamped=[])

    if n == 0:
        return ValidateReferenceResponse(
            validated=False, reference=None, reviewed_image_count=0, conf=None,
            reason="No completed reviews yet for this model on this date. Review the predictions and "
                   "mark the images Reviewed, then try again.",
            buckets_stamped=[])

    # A prediction document that changed, appeared or vanished since review is evidence for nothing.
    diverged = sorted(
        name for name, data in reviewed.items()
        if any(recorded != _prediction_digest(req.pred_dir, name)
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

    from tcip_mcp.dataset_layout import prediction_bucket_date
    from tcip_mcp.pipelines.feedback import (
        describe_review_validation,
        resolve_operating_point_from_review,
        review_conf_threshold,
        review_reference_hash,
        review_to_records,
    )
    from tcip_mcp.traits import TraitUnknownError

    review_state = {"image": completed}
    # Thread the producing run's experiment_id through so the calibration's train-disjointness
    # gate can check the reviewed images against that run's training split. Sourced from the
    # buckets' own operating_point.json sidecars (stamped by run_inference), never asserted:
    # when multiple buckets disagree on which run produced them, pass None (mixed-provenance
    # shouldn't silently vouch for one run's disjointness) rather than raising, so this route keeps
    # working for a legitimate multi-bucket review call.
    bucket_exp_ids = {sc.get("experiment_id") for sc in sidecars.values() if sc.get("experiment_id")}
    review_experiment_id = next(iter(bucket_exp_ids)) if len(bucket_exp_ids) == 1 else None

    # Scope every verdict/negative record to the bucket(s) actually being validated, at the
    # deepest choke point (resolve_operating_point_from_review), not just here.
    bucket_identities = [
        {"checkpoint_sha256": sc.get("checkpoint_sha256"), "experiment_id": sc.get("experiment_id")}
        for sc in sidecars.values()
    ]
    # The review path's effective staging floor is max(generation_conf, review_conf_threshold):
    # the generation half read off the same sidecars already loaded above, the review half read
    # off the verdicts' own recorded conf_threshold (scoped identically to the bucket(s) above).
    # Either half unknown makes the combined floor None (fails closed).
    gen_confs = [
        v for sc in sidecars.values()
        if isinstance(v := ((sc.get("operating_point") or {}).get("conf") or {}).get("value"),
                     (int, float))
    ]
    generation_conf = max(float(v) for v in gen_confs) if gen_confs else None
    review_conf = review_conf_threshold(review_state, bucket_identities=bucket_identities,
                                        only_completed=True)
    staged_conf_floor = (
        max(generation_conf, review_conf)
        if generation_conf is not None and review_conf is not None
        else None
    )

    # Thread tile_size/tiled + their sources off the same sidecars already loaded above, so a
    # review-confirmed bundle honestly reports "derived"/"explicit" instead of always falling
    # back to "default" regardless of what the buckets actually carry. A single bucket's own
    # stamp is used; a mixed set of sources across buckets is not resolvable to one fact, so it
    # falls back to the honest default.
    from tcip_mcp.pipelines.resolution import tile_size_source_of

    tile_sizes = {((sc.get("operating_point") or {}).get("tile_size") or {}).get("value")
                  for sc in sidecars.values()}
    # From validated_against, not the bare source field, which a native-ratio edge shares with a
    # real persisted one: reading source alone would silently re-validate native-ratio on review.
    tile_size_valid_refs = {
        ((sc.get("operating_point") or {}).get("tile_size") or {}).get("validated_against")
        for sc in sidecars.values()}
    tiled_vals = {((sc.get("operating_point") or {}).get("tiled") or {}).get("value")
                 for sc in sidecars.values()}
    tiled_sources = {((sc.get("operating_point") or {}).get("tiled") or {}).get("source")
                     for sc in sidecars.values()}
    tile_size_derived_froms = {
        ((sc.get("operating_point") or {}).get("tile_size") or {}).get("derived_from")
        for sc in sidecars.values()}
    review_tile_size = next(iter(tile_sizes)) if len(tile_sizes) == 1 else None
    review_tile_size_valid_ref = (
        next(iter(tile_size_valid_refs)) if len(tile_size_valid_refs) == 1
        and review_tile_size is not None else None)
    review_tile_size_source = tile_size_source_of(
        review_tile_size_valid_ref, tile_size=review_tile_size)
    # The stamp's own derived_from text, carried forward unchanged: this route holds no predictor
    # to compose one from, only the record the producing run already wrote.
    review_tile_size_derived_from = (
        next(iter(tile_size_derived_froms)) if len(tile_size_derived_froms) == 1
        and review_tile_size is not None else None)
    review_tiled = next(iter(tiled_vals)) if len(tiled_vals) == 1 else None
    review_tiled_source = (next(iter(tiled_sources)) if len(tiled_sources) == 1
                           and review_tiled is not None else "default")

    # Refuse here, naming the bucket(s), rather than let the resolver's bare ValueError surface.
    if review_tile_size_source == "explicit" and review_tile_size_derived_from is None:
        per_bucket_derived_from = {
            d: ((sc.get("operating_point") or {}).get("tile_size") or {}).get("derived_from")
            for d, sc in sidecars.items()
        }
        raise HTTPException(
            400,
            "these predictions carry an explicit tile edge but disagree about, or omit, why it is "
            f"trusted ({per_bucket_derived_from}), so the review promotion cannot state one "
            "derivation for the validated claim. Validate the disagreeing bucket separately, or "
            "re-export the predictions from one run so their stamps agree.",
        )

    # One spelling of the evidence, shared by the description and open_validation's own resolver run.
    resolver_inputs = {
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
        # The reviewed bucket's own date, when it has one: a bound checkpoint's selection check
        # scopes itself to this the same way the calibration door scopes to labels_dir's date.
        "calibration_date": prediction_bucket_date(pred_dir),
        # True when the buckets named more than one producing run, false when none named one: both
        # collapse review_experiment_id to None above, but only the first is a real disagreement.
        "experiment_id_ambiguous": len(bucket_exp_ids) > 1,
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
        # since the split locked, or the lock file itself is corrupt. Either way
        # this is an honest refusal, not a 500: surface it as such.
        raise HTTPException(400, str(exc)) from None

    result = describe_review_validation(bundle, reviewed_image_count=n)

    # Stamp each bucket's provenance sidecar (operating_point.json is not a label, so this never
    # touches the reviewed per-image predictions or the verdict-immutability guard).
    from tcip_mcp.pipelines.resolution import (
        claim_payload,
        open_validation,
        seal_validation,
        update_sidecar,
    )
    from tcip_annotation.json_io import UnreadableLabelDocument
    from tcip_mcp.prediction_buckets import review_state_dir_of
    from tcip_store.errors import DecodeError, SchemaVersionRefused, StoreBusy

    op_prov = bundle.to_provenance()["operating_point"]
    ref_hash = review_reference_hash(
        review_to_records(review_state, bucket_identities=bucket_identities, subject=req.subject))
    now_iso = datetime.now(timezone.utc).isoformat()
    record_digests: dict[str, str] = {}
    stamped: list[str] = []

    try:
        draft = None
        if result["validated"]:
            shas = {sc.get("checkpoint_sha256") for sc in sidecars.values()
                    if sc.get("checkpoint_sha256")}
            draft = open_validation(
                document="operating_point",
                evidence={"resolver": "resolve_operating_point_from_review",
                          "inputs": resolver_inputs},
                trait=req.trait,
                checkpoint_sha256=next(iter(shas)) if len(shas) == 1 else None,
                producing_experiment_id=review_experiment_id,
                reference_inputs={
                    "dataset_root": req.dataset_root,
                    "scope_roots": {"verdicts": str(review_state_dir_of(req.dataset_root))},
                    "stated_values": {"review_reference_hash": ref_hash, "review_image_count": n},
                },
            )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None

    def _require_bare_bucket_subject(bucket_dir: str, subject: str) -> None:
        """A bare directory (no stamp at all) has no recorded map to carry a promoted stamp's
        subject claim, so every prediction record in it must positively carry ``subject``.

        Walks the directory the same way ``require_reference_ground_truth`` does
        (``prediction_documents``, each document's own annotations); a directory holding another
        subject refuses by name: a review reference is one subject's, and a staged bucket holding
        several is promoted by staging one subject per bucket.
        """
        from tcip_annotation.json_io import prediction_documents, read_annotations

        others: set[str] = set()
        for path in prediction_documents(Path(bucket_dir)):
            for a in read_annotations(str(path)):
                if a.subject != subject:
                    others.add(a.subject)
        if others:
            raise ValueError(
                f"{bucket_dir} holds annotations of {sorted(others)} besides {subject!r}: a "
                "review reference is one subject's. Stage one subject per bucket before "
                "promoting it."
            )

    def _stamp_body(stored: dict) -> dict:
        """This promotion merged over whatever the producing run left in ``stored``.

        The trait is written only when a gate was cleared, so a bucket carries the trait its claim
        was earned for and an honest placeholder claims no scope at all.

        A bare directory (``stored`` empty, no producer ever stamped it) has no pair of its own to
        carry forward: the review it promotes was a detector review (a bare directory admits no
        classified one), so this writes ``subject=req.subject``/``attribute=None``, after the
        caller has verified every record in the directory positively carries that subject
        (``_require_bare_bucket_subject``). A stored stamp's own pair (present or, for a pre-key
        stamp, absent) is carried forward unchanged by the plain ``dict(stored)`` below; the rail
        refuses a merge that ends up with neither key.

        ``schema_version`` marks the promotion's own writing vintage, not every value the merged
        record carries: a member ``stored`` already held under an older provenance vocabulary
        (unchanged by this merge) keeps reading under that older spelling, exactly as
        ``operating_point_stamp`` documents for its own producers.
        """
        merged = dict(stored)
        if not stored:
            merged["subject"] = req.subject
            merged["attribute"] = None
        merged.update({
            "schema_version": 2,
            "operating_point": op_prov,
            "validated": result["validated"],
            "validated_reference": result["reference"],
            "validation_source": "review_confirmed",
            "review_reference_hash": ref_hash,
            "review_image_count": n,
            "shippable_issues": bundle.shippable_issues(),
            "validated_at": now_iso,
        })
        merged.setdefault("produced_at", now_iso)
        if draft is not None:
            merged["trait"] = req.trait
        return merged

    def _promotion_of(pred_dir: str, earned: dict) -> Callable[[dict], Optional[dict]]:
        """The merge one bucket's stamp is promoted through, run inside that stamp's own lock."""

        def _promote(stored: dict) -> dict | None:
            """Merge this promotion into whatever the producing run left, inside the stamp's lock.

            The no-downgrade decision is made against the stored stamp, not the copy read before the
            lock: predictions whose validation a record answers for (held-out GT, an earlier review)
            stay as they are, and a producer that stamped the bucket while this review was being
            reconciled is not overwritten. ``earned`` is the body the record was sealed over, so the
            pointer is merged only while the stamp still makes the claim that record answers for; a
            claim that moved under the lock leaves the record inert rather than misnamed.
            """
            binding = verify_stamp_binding(stored, pred_dir, document="operating_point",
                                           digest_memo=digest_memo)
            if binding.claimed and binding.ok:
                return None
            merged = _stamp_body(stored)
            if draft is None:
                return merged
            if claim_payload(merged, document="operating_point") != claim_payload(
                    earned, document="operating_point"):
                return None
            merged["validated_by"] = earned["validated_by"]
            return merged

        return _promote

    try:
        for d in bucket_dirs:
            if bindings[d].claimed and bindings[d].ok:
                continue  # a mixed set: a bucket whose validation a record answers for is left alone
            Path(d).mkdir(parents=True, exist_ok=True)
            if not sidecars[d]:
                _require_bare_bucket_subject(d, req.subject)
            # Sealed outside the stamp's lock: no store write may open inside another's transaction.
            earned = _stamp_body(sidecars[d])
            if draft is not None:
                record_digests[d], earned = seal_validation(
                    draft, dataset_root=req.dataset_root, bucket_dirs=list(bucket_dirs),
                    stamp_body=earned)
            if update_sidecar(d, _promotion_of(d, earned)):
                stamped.append(d)
    except StoreBusy as exc:
        # Contention is a retryable infrastructure fault, never a malformed request; the
        # dataset select route's own StoreBusy handling is the platform's precedent.
        raise HTTPException(503, str(exc)) from exc
    except (ValueError, SchemaVersionRefused, DecodeError, UnreadableLabelDocument) as exc:
        raise HTTPException(400, str(exc)) from None

    # The sidecar this stamps sits in the prediction bucket, which travels with the dataset.
    _audit(req.dataset_root, "gui_review_validate_reference", {
        "trait": req.trait,
        "validated": result["validated"],
        "reference": result["reference"],
        "reviewed_image_count": n,
        "buckets_stamped": stamped,
        "record_digests": record_digests,
    })
    return ValidateReferenceResponse(
        validated=bool(result["validated"]),
        reference=result["reference"],
        reviewed_image_count=n,
        conf=result["conf"],
        reason=result["reason"],
        buckets_stamped=stamped,
    )
