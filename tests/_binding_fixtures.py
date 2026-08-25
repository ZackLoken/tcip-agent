"""Minting a real validation record for a hand-built prediction-bucket stamp.

A stamp that claims validation is refused at write time and floors at read time unless a record
outside the bucket answers for it. Plenty of tests need a validated bucket for a subject that is
not the binding at all (a delivery door's arithmetic, a chronology, a lock), so they file a genuine
record here instead of each learning the record's shape. What this does by hand is what
``seal_validation`` does for a producer: it files the row and hands back the stamp with its pointer
merged in.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_HOST = object()
"""Default for ``producing_experiment_id``: the run that produced the predictions is the experiment
the row is filed on, which is the ordinary case for a bucket a training run's checkpoint produced."""


def write_prediction(pred_dir: str | Path, stem: str, *, count: int = 1) -> Path:
    """One per-image prediction document in a bucket, enough to give the bucket content to hash."""
    d = Path(pred_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{stem}.json"
    path.write_text(json.dumps({"image": f"{stem}.png", "count": count}), encoding="utf-8")
    return path


def file_validation_record(
    stamp: dict,
    *,
    document: str = "operating_point",
    dataset_root: str | Path,
    pred_dirs: list[str | Path] | tuple[str | Path, ...] = (),
    images_dir: str | Path | None = None,
    experiment_id: str = "exp-binding-reference",
    producing_experiment_id: Any = _HOST,
    trait: str | None = None,
    reference_identity: dict | None = None,
) -> dict:
    """File the record ``stamp`` claims, and return the stamp with its pointer merged in.

    ``pred_dirs`` are the buckets a claim covers, hashed as they are on disk now, so they must
    already hold what the claim is about: prediction bytes for ``operating_point``, image bytes for
    ``resolve_scale`` (a scale claim is a fact about the bucket's imagery, not its predictions, the
    same distinction ``seal_validation`` draws; ``images_dir`` is required for a ``resolve_scale``
    claim over a non-empty ``pred_dirs``). The other documents cover no bucket and take none.
    """
    from tcip_mcp.experiments import _append_validation, create_experiment, experiment_exists
    from tcip_mcp.pipelines.resolution import _DOCUMENT_PARAM, claim_payload, cleared_reference
    from tcip_mcp.prediction_buckets import bucket_content_digest, bucket_stems_digest

    param_key, validation_kind = _DOCUMENT_PARAM[document]
    reference = cleared_reference(
        ((stamp.get("operating_point") or {}).get(param_key) or {}).get("validated_against"),
        validation_kind=validation_kind,
    )
    root = Path(dataset_root).resolve()
    if document == "resolve_scale" and pred_dirs and images_dir is None:
        raise ValueError(
            "file_validation_record needs images_dir to hash a resolve_scale claim's covered "
            "bucket(s)"
        )

    def digest_fn(d: str | Path) -> str:
        if document == "operating_point":
            return bucket_content_digest(d)
        assert images_dir is not None
        return bucket_stems_digest(d, images_dir=images_dir)

    covered = {Path(d).resolve().relative_to(root).as_posix(): digest_fn(d)
               for d in pred_dirs}
    host = producing_experiment_id if producing_experiment_id is not _HOST else experiment_id

    if not experiment_exists(experiment_id):
        create_experiment(experiment_id, {"derived_from": "a reference for a test whose subject is "
                                                          "not the binding itself"})
    body = {
        "document": document,
        "trait": trait if trait is not None else stamp.get("trait"),
        "claim": claim_payload(stamp, document=document),
        "validated_against": reference,
        "checkpoint_sha256": stamp.get("checkpoint_sha256"),
        "producing_experiment_id": host,
        "reference_identity": reference_identity or {"stated_values": {"reference": "hand-filed"}},
        "covered_buckets": covered,
        "dataset_root": str(root),
        "recorded_at": "2026-03-04T12:00:00+00:00",
    }
    appended = _append_validation(experiment_id, body)
    assert "error" not in appended, appended
    return {**stamp, "validated_by": {"experiment_id": experiment_id,
                                      "record_digest": appended["record_digest"]}}


def write_bound_sidecar(
    pred_dir: str | Path,
    stamp: dict,
    *,
    document: str = "operating_point",
    dataset_root: str | Path,
    pred_dirs: list[str | Path] | tuple[str | Path, ...] | None = None,
    images_dir: str | Path | None = None,
    **record: Any,
) -> dict:
    """File the record and write the bound stamp, the two steps a producer does in that order."""
    from tcip_mcp.pipelines.resolution import write_sidecar

    covered = pred_dirs if pred_dirs is not None else (
        [pred_dir] if document in ("operating_point", "resolve_scale") else [])
    bound = file_validation_record(
        stamp, document=document, dataset_root=dataset_root, pred_dirs=covered,
        images_dir=images_dir, **record)
    write_sidecar(pred_dir, bound, document)
    return bound


PRODUCER_WEIGHTS = b"the weights a producing run filed under the experiment its predictions name"
PRODUCER_CHECKPOINT_SHA256 = hashlib.sha256(PRODUCER_WEIGHTS).hexdigest()
"""The content hash of those weights, so a fixture's stamp and a byte golden name one identity."""


def record_producing_run(weights_dir: str | Path, experiment_id: str) -> str:
    """File the run a bucket's stamp names as its producer, and return the checkpoint hash it filed.

    A delivered producer column is emitted only where something outside the prediction bucket
    corroborates the identity the stamp asserts: the experiment has to exist, and the checkpoint it
    recorded has to be the one the stamp names. A fixture that wants the populated case has to leave
    both behind, which is what a completed training run leaves behind for itself.
    """
    from tcip_mcp.experiments import create_experiment, experiment_exists, update_lineage
    from tcip_mcp.model_registry import checkpoint_sha256

    ckpt = Path(weights_dir) / "model_best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(PRODUCER_WEIGHTS)
    if not experiment_exists(experiment_id):
        create_experiment(experiment_id, {"note": "a producing run standing behind a delivery"})
    update_lineage(experiment_id, model_weights=str(ckpt))
    return checkpoint_sha256(ckpt)


# --- the export doors: a stand-in run whose validated count they can actually earn a record for ---

def calibrated_run_fields(
    trait: str = "catkin",
    *,
    labels_dir: str | Path,
    identity: str = "evidence-for-a-standin-run",
    tiled: bool = False,
    tile_size: int | None = None,
    tile_size_source: str = "default",
) -> dict:
    """The fields a calibrated run's result carries for a delivery door to earn its record from.

    A test standing in for the inference pass still has to leave behind what the door reopens the
    gate over, or the door has nothing to earn with and says so. This resolves a real held-out
    operating point over a dense synthetic reference, files the evidence beside its sweep exactly
    where a calibrated run files it, and hands back the result fields that carry it. The producing
    experiment is ``None``, the ordinary bespoke-checkpoint case, so the door earns through a
    created calibration experiment.
    """
    from tcip_store import store

    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tcip_mcp.tools.inference_tools import confidence_sweep_key
    from tests._dense_op_fixtures import dense_records

    n_images, objects = 20, 80
    inputs = {
        "dataset_hash": "H",
        "calibration_records": dense_records(n_images=n_images, objects_per_image=objects,
                                             id_prefix="c", fp_pattern=[1] * n_images, score=0.9,
                                             fp_score=0.05),
        "holdout_records": dense_records(n_images=n_images, objects_per_image=objects,
                                         id_prefix="h", shift=5.0, fp_pattern=[1] * n_images,
                                         score=0.9, fp_score=0.05),
        "tiled": tiled,
        "tile_size": tile_size,
        "tile_size_source": tile_size_source,
        "staged_conf_floor": 0.01,
    }
    bundle = resolve_operating_point(trait, experiment_id=None, **inputs)
    evidence = {"resolver": "resolve_operating_point", "inputs": inputs,
                "reference_inputs": {"label_dirs": {"calibration": str(labels_dir)}}}
    store.replace(confidence_sweep_key(identity), {"calibration_evidence": evidence})
    return {
        "operating_point": bundle.to_provenance()["operating_point"],
        "validated": True,
        "conf_source": "calibration",
        "dataset_hash": "H",
        "shippable_issues": [],
        "experiment_id": None,
        "calibration_evidence_key": identity,
    }
