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

_UNSTATED = object()
"""Default for ``train_disjointness``: the same shape an unchecked, foreign-checkpoint row carries,
picked per document since ``resolve_scale`` has no training run to check at all."""

_UNSTATED_SELECTION = object()
"""Default for ``selection_disjointness``: the same not-applicable shape a calibration naming no
split manifest carries, picked per document since ``resolve_scale`` has no training run to check
at all."""


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
    train_disjointness: Any = _UNSTATED,
    selection_disjointness: Any = _UNSTATED_SELECTION,
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
    if train_disjointness is _UNSTATED:
        td = None if document == "resolve_scale" else {"checked": False, "group_check": None}
    else:
        td = train_disjointness
    if selection_disjointness is _UNSTATED_SELECTION:
        sd = None if document == "resolve_scale" else {
            "applicable": False, "reason": "no split manifest named for this hand-filed record",
            "checked": False, "group_check": None,
        }
    else:
        sd = selection_disjointness

    if not experiment_exists(experiment_id):
        create_experiment(experiment_id, {"derived_from": "a reference for a test whose subject is "
                                                          "not the binding itself"})
    body = {
        "schema_version": 2,  # mirrors seal_validation's own row literal (resolution.py)
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
        "train_disjointness": td,
        "selection_disjointness": sd,
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


def _producer_bytes(experiment_id: str) -> bytes:
    """The per-experiment bytes :func:`record_producing_run` files and
    :func:`producer_checkpoint_sha256` digests: ``PRODUCER_WEIGHTS`` plus the id, so two
    producing runs never share one digest by construction."""
    return PRODUCER_WEIGHTS + experiment_id.encode("utf-8")


def producer_checkpoint_sha256(experiment_id: str) -> str:
    """The digest :func:`record_producing_run` files for ``experiment_id``, so a golden asserting
    the delivered cell can compute its own expectation without re-running the fixture."""
    from tcip_mcp.model_registry import _sha256_of_bytes

    return _sha256_of_bytes(_producer_bytes(experiment_id))


def record_producing_run(weights_dir: str | Path, experiment_id: str) -> str:
    """File the run a bucket's stamp names as its producer, bound through the platform's own
    registration, and return the registered entry's checkpoint hash.

    A delivered producer column is emitted only where something outside the prediction bucket
    corroborates the identity the stamp asserts: the experiment has to exist, be completed with a
    recorded digest, and be bound to a registry entry naming that digest. A fixture that wants the
    populated case has to leave all of that behind, which is what a completed, registered training
    run leaves behind for itself. Idempotent under a repeat call for the same ``experiment_id``
    (some callers file more than one bucket behind one producing run): a run completion cannot be
    repeated once terminal, so a second call skips straight to registration, itself idempotent for
    the recorded bytes.
    """
    from tcip_mcp.experiments import (
        complete_run, create_experiment, experiment_exists, read_member,
        register_model_from_experiment, status_key,
    )

    ckpt = Path(weights_dir) / "model_best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(_producer_bytes(experiment_id))
    if not experiment_exists(experiment_id):
        create_experiment(experiment_id, {"note": "a producing run standing behind a delivery"})
    status = read_member(status_key(experiment_id), {})
    if not (isinstance(status, dict) and status.get("state") == "completed"):
        completed = complete_run(experiment_id, str(ckpt))
        assert "error" not in completed, completed
    registered = register_model_from_experiment(experiment_id, str(ckpt))
    assert "error" not in registered, registered
    return registered["sha256"]


def register_plant_registry_for(
    csv_paths: list[str | Path], *, name: str = "reg", crop: str = "currant", site: str = "orchard",
) -> str:
    """Register ``csv_paths`` under ``name`` in whichever project the process is pinned to
    (``register_plant_registry``'s own resolution, the one ``build_plant_mapping`` and
    ``deliver_orthomosaic_plant_counts`` share), and return ``name``. Idempotent under the same
    content: a second call with the same paths under the same name is a no-op.
    """
    from tcip_mcp.tools.phenology_tools import register_plant_registry

    res = register_plant_registry(
        name=name, csv_paths=[str(p) for p in csv_paths], crop=crop, site=site)
    assert "error" not in res, res
    return name


def write_plant_mapping(
    project_root: str | Path, name: str, mapping: dict[str, list[dict]],
    *, dataset_root: str | Path,
) -> str:
    """Persist a hand-composed ``{date: [assignment dict, ...]}`` mapping through the platform's
    own producer (``persist_mapping``, record then receipt), and return the dataset id it minted.

    A fixture's real interest is only ``mapping``; every other provenance field is a placeholder
    a delivery's dataset-identity and receipt checks require but do not otherwise inspect.
    ``dataset_root`` is registered (``register_dataset``) if it carries no identity yet, so a
    delivery reading these predictions binds on a real minted id. ``plant_registry`` names a real,
    empty registry (``register_plant_registry_record`` over no CSVs, idempotent under a repeat
    call), so a delivery's own registry check (``registry_entries_or_refusal``) finds a real
    record to load rather than refusing a fixture's placeholder name as vanished.
    """
    from datetime import datetime, timezone

    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        Assignment,
        MappingBuild,
        persist_mapping,
        register_plant_registry_record,
    )
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.traits import registered_crops

    def _row(row: dict, date: str) -> Assignment:
        # Tolerant of a fixture's partial row (just the fields its own test cares about): the
        # rest take the same honest defaults a real sequence-anchored match would carry.
        return Assignment(
            image_path=row.get("image_path", f"{row.get('stem', '')}.jpg"),
            stem=row["stem"], date_folder=row.get("date_folder", date),
            plot_name=row.get("plot_name"), accession_name=row.get("accession_name"),
            source=row.get("source", "sequence"), distance_m=row.get("distance_m", 1.0),
        )

    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    crop = sorted(registered_crops())[0]
    reg = register_dataset(str(root), crop=crop, project_root=str(project_root))
    registry = register_plant_registry_record(
        project_root, "unregistered", [], crop=crop, site="fixture",
        registered_by="write_plant_mapping")
    build = MappingBuild(
        name=name, project_root=str(project_root), dataset_root=str(root), dataset_id=reg["id"],
        built_by="build_plant_mapping", built_at=datetime.now(timezone.utc).isoformat(),
        dates_requested=None, dates=sorted(mapping),
        nn_tolerance_m={"value": 10.0, "source": "fallback"},
        plant_registry={"name": "unregistered", "digest": registry["digest"]},
        capture_identity={d: "0" * 16 for d in mapping},
        capture_digests={d: {} for d in mapping}, unreadable={d: [] for d in mapping},
        assignments={d: [_row(row, d) for row in rows] for d, rows in mapping.items()},
    )
    persist_mapping(build, project_root, name)
    return reg["id"]


# --- the export doors: a stand-in run whose validated count they can actually earn a record for ---

def calibrated_run_fields(
    trait: str = "bud_opening",
    *,
    checkpoint_sha256: str,
    labels_dir: str | Path,
    tiled: bool = False,
    tile_size: int | None = None,
    tile_size_source: str = "default",
) -> dict:
    """The fields a calibrated run's result carries for a delivery door to earn its record from.

    A test standing in for the inference pass still has to leave behind what the door reopens the
    gate over, or the door has nothing to earn with and says so. This resolves a real held-out
    operating point over a dense synthetic reference, files the record under its own identity
    (``calibration_curve_identity``, the same key a real run's write and a delivery door's read
    agree on) exactly where a calibrated run files it, and hands back the result fields that carry
    it. The producing experiment is ``None``, the ordinary bespoke-checkpoint case, so the door
    earns through a created calibration experiment.
    """
    from tcip_store import store

    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tcip_mcp.tools.inference_tools import calibration_curve_identity, calibration_curve_key
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
    # Mirrors _run_inference_verified's own persisted body (inference_tools.py).
    body = {
        "schema_version": 2,
        "trait": trait,
        "dataset_hash": "H",
        "checkpoint_sha256": checkpoint_sha256,
        "predictor_path": {
            "tile": tiled, "tile_size": tile_size, "overlap": None,
            "postprocess": "nms", "global_nms_iou": None, "max_dets": None,
        },
        "gate_evidence": bundle.get("conf").gate_evidence,
        "calibration_evidence": evidence,
    }
    identity = calibration_curve_identity(body)
    store.replace(calibration_curve_key(identity), body)
    return {
        "operating_point": bundle.to_provenance()["operating_point"],
        "validated": True,
        "conf_source": "calibration",
        "dataset_hash": "H",
        "shippable_issues": [],
        "experiment_id": None,
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_evidence_key": identity,
    }
