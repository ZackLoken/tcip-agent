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
"""Default for ``producing_experiment_id``: the run that produced the predictions is the run the
row is filed on, which is the ordinary case for a bucket a training run's checkpoint produced."""

_UNSTATED = object()
"""Default for ``train_disjointness``: the same shape an unchecked, foreign-checkpoint row carries,
picked per document since ``resolve_scale`` has no training run to check at all."""

_UNSTATED_SELECTION = object()
"""Default for ``selection_disjointness``: the same not-applicable shape a calibration naming no
selection carries, picked per document since ``resolve_scale`` has no training run to check
at all."""


def document_reconciliation(
    bindings: dict[str, Any],
    *,
    validated: str,
    per_bucket: dict[str, str],
    unvalidated_buckets: list[str],
    missing_sidecars: list[str],
    on_disk_validated: bool,
    conf: float | None = None,
    confs: dict[str, float | None] | None = None,
) -> dict:
    """A ``_reconcile_validity``-shaped mapping over ``bindings``, a ``StampBinding`` mapping kept
    as given, with ``binding_notes`` one note per bucket whose binding does not hold and every
    other key the named argument of the same name."""
    return {
        "validated": validated,
        "on_disk_validated": on_disk_validated,
        "missing_sidecars": list(missing_sidecars),
        "unvalidated_buckets": list(unvalidated_buckets),
        "binding_notes": {bucket: b.note for bucket, b in bindings.items() if not b.ok},
        "bindings": dict(bindings),
        "conf": conf,
        "confs": dict(confs) if confs is not None else {},
        "per_bucket": dict(per_bucket),
    }


def write_prediction(pred_dir: str | Path, stem: str, *, count: int = 1) -> Path:
    """One per-image prediction document in a bucket, enough to give the bucket content to hash."""
    d = Path(pred_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{stem}.json"
    path.write_text(json.dumps({"image": f"{stem}.png", "count": count}), encoding="utf-8")
    return path


def file_validation_record(
    project: Path,
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
    """File the record ``stamp`` claims on a run of ``project``, and return the stamp with its
    pointer merged in.

    ``pred_dirs`` are the buckets a claim covers, hashed as they are on disk now, so they must
    already hold what the claim is about: prediction bytes for ``operating_point``, image bytes for
    ``resolve_scale`` (a scale claim is a fact about the bucket's imagery, not its predictions, the
    same distinction ``seal_validation`` draws; ``images_dir`` is required for a ``resolve_scale``
    claim over a non-empty ``pred_dirs``). The other documents cover no bucket and take none. An
    ``operating_point`` stamp is completed over the producer's own skeleton
    (:func:`complete_stamp`) first.
    """
    if document == "operating_point":
        stamp = complete_stamp(stamp)
    from tcip_mcp.experiments import append_validation, experiment_dir, find_run
    from tcip_mcp.pipelines.resolution import _DOCUMENT_PARAM, claim_payload, cleared_reference
    from tests._verified_checkpoint_fixtures import detection_config, fixture_data_dir, opened_run
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

    covered = {str(Path(d).resolve()): digest_fn(d) for d in pred_dirs}
    host = producing_experiment_id if producing_experiment_id is not _HOST else experiment_id
    if train_disjointness is _UNSTATED:
        td = None if document == "resolve_scale" else {"checked": False, "group_check": None}
    else:
        td = train_disjointness
    if selection_disjointness is _UNSTATED_SELECTION:
        sd = None if document == "resolve_scale" else {
            "applicable": False, "reason": "no selection named for this hand-filed record",
            "checked": False, "group_check": None,
        }
    else:
        sd = selection_disjointness

    if find_run(experiment_id, project=project) is None:
        opened_run(project, detection_config(fixture_data_dir(project, experiment_id)),
                   experiment_id=experiment_id)
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
        "train_disjointness": td,
        "selection_disjointness": sd,
    }
    digest = append_validation(experiment_dir(experiment_id, project=project), body)
    return {**stamp, "validated_by": {"experiment_id": experiment_id, "record_digest": digest}}


def complete_stamp(partial: dict) -> dict:
    """``partial`` over ``operating_point_stamp``'s own skeleton at its unset values, so a fixture
    naming only the fields it cares about still writes every key the producer writes."""
    from tcip_mcp.pipelines.resolution import _SKELETON_ARGS, operating_point_stamp

    return {**operating_point_stamp(None, **_SKELETON_ARGS), **partial}


def write_bound_sidecar(
    project: Path,
    pred_dir: str | Path,
    stamp: dict,
    *,
    document: str = "operating_point",
    dataset_root: str | Path,
    pred_dirs: list[str | Path] | tuple[str | Path, ...] | None = None,
    images_dir: str | Path | None = None,
    **record: Any,
) -> dict:
    """File the record on a run of ``project`` and write the bound stamp, the two steps a
    producer does in that order."""
    from tcip_mcp.pipelines.resolution import write_sidecar

    covered = pred_dirs if pred_dirs is not None else (
        [pred_dir] if document in ("operating_point", "resolve_scale") else [])
    bound = file_validation_record(
        project, stamp, document=document, dataset_root=dataset_root, pred_dirs=covered,
        images_dir=images_dir, **record)
    write_sidecar(pred_dir, bound, document, project=project)
    return bound


def validated_bucket(tmp_path: Path, trait: str, *, document: str = "operating_point",
                     tag: str = "a") -> str:
    """A prediction bucket under ``<tmp_path>/ds_<tag>`` whose sidecar carries a genuine
    held-out-validated claim for ``trait``, filed on a run of the project ``tmp_path``."""
    from tcip_mcp.pipelines.resolution import VALIDATED_HELD_OUT

    root = tmp_path / f"ds_{tag}"
    bucket = root / "predictions" / "preds"
    write_prediction(bucket, "img_a")
    param_key = {"operating_point": "conf", "regression_operating_point": "regression"}[document]
    stamp: dict = {
        "validated": True, "trait": trait,
        "operating_point": {param_key: {"value": 0.4, "requires_validation": True,
                                        "validation_kind": "annotations",
                                        "validated_against": VALIDATED_HELD_OUT}},
    }
    if document == "operating_point":
        stamp["scope"] = {"subject": trait, "attribute": None, "id_map": {trait: 0}}
    write_bound_sidecar(tmp_path, bucket, stamp, document=document, dataset_root=root,
                        experiment_id=f"exp-validated-{tag}")
    return str(bucket)


def producer_checkpoint_sha256(project: Path, experiment_id: str) -> str:
    """The digest of the checkpoint the completed run ``experiment_id`` of ``project`` names, so a
    golden asserting the delivered cell reads its own expectation off the run."""
    from tcip_mcp.experiments import experiment_dir, observe

    checkpoint = observe(experiment_dir(experiment_id, project=project)).checkpoint
    assert checkpoint is not None, f"{experiment_id} did not complete"
    return checkpoint["sha256"]


def record_producing_run(project: Path, experiment_id: str) -> str:
    """Complete the run of ``project`` a bucket's stamp names as its producer, through the
    training envelope whose final status registers its checkpoint, and return that checkpoint's
    hash.

    A delivered producer column is emitted only where something outside the prediction bucket
    corroborates the identity the stamp asserts: the run has to have completed naming the digest.
    Idempotent under a repeat call for the same ``experiment_id`` (some callers file more than one
    bucket behind one producing run): a completed run is read, never run twice.
    """
    from tcip_mcp.experiments import find_run
    from tests._verified_checkpoint_fixtures import finished_run

    if find_run(experiment_id, project=project) is None:
        finished_run(project, experiment_id=experiment_id)
    return producer_checkpoint_sha256(project, experiment_id)


def register_plant_registry_for(
    project: Path, csv_paths: list[str | Path], *, name: str = "reg", crop: str = "currant",
    site: str = "orchard",
) -> str:
    """Register ``csv_paths`` under ``name`` in ``project`` through ``register_plant_registry``,
    and return ``name``. Idempotent under the same content: a second call with the same paths
    under the same name is a no-op.
    """
    from tcip_mcp.tools.phenology_tools import register_plant_registry

    res = register_plant_registry(
        project, name=name, csv_paths=[str(p) for p in csv_paths], crop=crop, site=site)
    assert "error" not in res, res
    return name


def _deg_to_dms(value: float) -> tuple[float, float, float]:
    v = abs(value)
    d = int(v)
    m_full = (v - d) * 60
    m = int(m_full)
    return (float(d), float(m), round((m_full - m) * 60, 4))


def write_geo_image(path: Path, lat: float, lon: float, when: Any) -> None:
    """A tiny JPEG at ``path`` carrying EXIF DateTimeOriginal ``when`` and GPS ``lat``/``lon``."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    exif[0x8769] = {0x9003: when.strftime("%Y:%m:%d %H:%M:%S")}
    exif[0x8825] = {
        0x0001: "N" if lat >= 0 else "S", 0x0002: _deg_to_dms(lat),
        0x0003: "E" if lon >= 0 else "W", 0x0004: _deg_to_dms(lon),
    }
    Image.new("RGB", (8, 8)).save(path, exif=exif)


def write_plant_mapping(
    project_root: str | Path, name: str, mapping: dict[str, list[dict]],
    *, dataset_root: str | Path,
) -> str:
    """Persist a hand-composed ``{date: [assignment dict, ...]}`` mapping through the platform's
    own producer (``persist_mapping``, record then receipt), and return the dataset id it minted.
    ``dataset_root`` is registered (``register_dataset``), and the mapping names a registered,
    empty plant registry (``register_plant_registry_record`` over no CSVs).
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
            image=row.get("image", f"{row.get('stem', '')}.jpg"),
            stem=row["stem"], date_folder=row.get("date_folder", date),
            plot_name=row.get("plot_name"), accession_name=row.get("accession_name"),
            source=row.get("source", "sequence"), distance_m=row.get("distance_m", 1.0),
        )

    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    crop = sorted(registered_crops())[0]
    from tcip_mcp.registry_paths import stored_path

    reg = register_dataset(Path(project_root), str(root), crop=crop)
    registry = register_plant_registry_record(
        project_root, "unregistered", [], crop=crop, site="fixture",
        registered_by="write_plant_mapping")
    build = MappingBuild(
        name=name, dataset_root=stored_path(root, project_root), dataset_id=reg["id"],
        built_by="build_plant_mapping", built_at=datetime.now(timezone.utc).isoformat(),
        dates_requested=None, dates=sorted(mapping),
        nn_tolerance_m={"value": 10.0, "source": "stated"},
        plant_registry={"name": "unregistered", "digest": registry["digest"]},
        capture_identity={d: "0" * 16 for d in mapping},
        capture_digests={d: {} for d in mapping}, unreadable={d: [] for d in mapping},
        assignments={d: [_row(row, d) for row in rows] for d, rows in mapping.items()},
    )
    persist_mapping(build, project_root, name)
    return reg["id"]


# --- the export doors: a stand-in run whose validated count they can actually earn a record for ---

def calibrated_run_fields(
    project: Path,
    trait: str = "bud_opening",
    *,
    checkpoint_sha256: str,
    labels_dir: str | Path,
    postprocess: str | None = None,
    tile_size: int | None = None,
    tile_size_source: str = "default",
) -> dict:
    """The fields a calibrated run's result carries for a delivery door to earn its record from.

    A test standing in for the inference pass still has to leave behind what the door reopens the
    gate over, or the door has nothing to earn with and says so. This resolves a real held-out
    operating point over a dense synthetic reference, collected untiled or, with ``postprocess``
    named, under a tiled regime merging by it (``tiled_regime``), files the record under its own
    identity (``calibration_curve_identity``, the same key a real run's write and a delivery door's
    read agree on) exactly where a calibrated run of ``project`` files it, and hands back the
    result fields that
    carry it, its ``slicing`` record included. The producing experiment is ``None``, the ordinary
    bespoke-checkpoint case, so the door earns through a calibration run directory of its own.
    """
    from tcip_store import store

    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tcip_mcp.tools.inference_tools import calibration_curve_identity, calibration_curve_key
    from tests._dense_op_fixtures import dense_records
    from tests._regime_fixtures import tiled_regime

    n_images, objects = 20, 80
    inputs = {
        "dataset_hash": "H",
        "calibration_records": dense_records(n_images=n_images, objects_per_image=objects,
                                             id_prefix="c", fp_pattern=[1] * n_images, score=0.9,
                                             fp_score=0.05),
        "holdout_records": dense_records(n_images=n_images, objects_per_image=objects,
                                         id_prefix="h", shift=5.0, fp_pattern=[1] * n_images,
                                         score=0.9, fp_score=0.05),
        **(tiled_regime(postprocess=postprocess) if postprocess else {"slicing": None}),
        "tile_size": tile_size,
        "tile_size_source": tile_size_source,
        "staged_conf_floor": 0.01,
    }
    bundle = resolve_operating_point(trait, project=project, experiment_id=None, **inputs)
    evidence = {"resolver": "resolve_operating_point", "inputs": inputs,
                "reference_inputs": {"label_dirs": {"calibration": str(labels_dir)}}}
    # Mirrors _run_inference_verified's own persisted body (inference_tools.py).
    body = {
        "trait": trait,
        "dataset_hash": "H",
        "checkpoint_sha256": checkpoint_sha256,
        "gate_evidence": bundle.get("conf").gate_evidence,
        "calibration_evidence": evidence,
    }
    identity = calibration_curve_identity(body)
    store.replace(calibration_curve_key(project, identity), body)
    return {
        "operating_point": bundle.to_provenance()["operating_point"],
        "slicing": bundle.slicing,
        "validated": True,
        "conf_source": "calibration",
        "dataset_hash": "H",
        "shippable_issues": [],
        "experiment_id": None,
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_evidence_key": identity,
    }


def run_result(
    operating_point: dict,
    results: list[dict],
    *,
    slicing: dict | None = None,
    checkpoint_sha256: str = "deadbeef",
    experiment_id: str | None = None,
    subject: str = "bud",
    attribute: str | None = None,
    id_map: dict | None = None,
    images_dir: str = "images",
    **fields: Any,
) -> dict:
    """A run's own facts built through the pass' own skeleton (``_PreparedPass.result``), for a
    test standing in for the inference pass: ``operating_point`` and ``slicing`` are what its
    bundle states, ``fields`` the run's calibrated or raw extras (``validated``, ``conf_source``
    and the like), ``results`` its per-image predictions. Its scope is ``subject`` under
    ``attribute`` with ``id_map``, a detector's one-subject map when no map is given."""
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tests._regime_fixtures import stub_pass

    prepared = stub_pass(None)
    prepared.checkpoint_path, prepared.images_dir = "model_best.pt", images_dir
    prepared.identity = {"sha256": checkpoint_sha256, "experiment_id": experiment_id}
    prepared.scope = ClassScope(subject=subject, attribute=attribute,
                                id_map=id_map if id_map is not None else {subject: 0})
    return {**prepared.result({"operating_point": operating_point, "slicing": slicing}, fields),
            "results": results}
