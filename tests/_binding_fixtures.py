"""Minting a real validation record for a hand-built prediction-bucket stamp.

A stamp that claims validation is refused at write time and floors at read time unless a record
outside the bucket answers for it. Plenty of tests need a validated bucket for a subject that is
not the binding at all (a delivery door's arithmetic, a chronology, a lock), so they file a genuine
record here instead of each learning the record's shape. What this does by hand is what
``seal_validation`` does for a producer: it files the row and hands back the stamp with its pointer
merged in.
"""

from __future__ import annotations

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
    experiment_id: str = "exp-binding-reference",
    producing_experiment_id: Any = _HOST,
    trait: str | None = None,
    reference_identity: dict | None = None,
) -> dict:
    """File the record ``stamp`` claims, and return the stamp with its pointer merged in.

    ``pred_dirs`` are the buckets a count claim covers, hashed as they are on disk now, so they must
    already hold the prediction files the claim is about. The other documents cover no bucket's
    content and take none.
    """
    from tcip_mcp.experiments import _append_validation, create_experiment, experiment_exists
    from tcip_mcp.pipelines.resolution import _DOCUMENT_PARAM, claim_payload, cleared_reference
    from tcip_mcp.prediction_buckets import bucket_content_digest

    param_key, validation_kind = _DOCUMENT_PARAM[document]
    reference = cleared_reference(
        ((stamp.get("operating_point") or {}).get(param_key) or {}).get("validated_against"),
        validation_kind=validation_kind,
    )
    root = Path(dataset_root).resolve()
    covered = {Path(d).resolve().relative_to(root).as_posix(): bucket_content_digest(d)
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
    **record: Any,
) -> dict:
    """File the record and write the bound stamp, the two steps a producer does in that order."""
    from tcip_mcp.pipelines.resolution import write_sidecar

    covered = pred_dirs if pred_dirs is not None else (
        [pred_dir] if document == "operating_point" else [])
    bound = file_validation_record(
        stamp, document=document, dataset_root=dataset_root, pred_dirs=covered, **record)
    write_sidecar(pred_dir, bound, document)
    return bound
