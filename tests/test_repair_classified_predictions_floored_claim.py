"""``tcip repair-classified-predictions`` rule 7: rewriting a bucket's documents changes its
content digest, so a count claim sealed over the bucket's old bytes floors the moment the rewrite
lands. The report names the floor beside the stamp's own stored ``validated`` so a stale ``true``
is never read as still validated, and names ``calibrate_count_operating_point`` as how to re-earn
the claim over the rewritten content.
"""

from __future__ import annotations

from pathlib import Path

import tcip_store as ts
from tcip_store.binding import bind_default

from tcip_annotation.json_io import write_annotations
from tcip_annotation.state import Annotation, BBox

from tcip_mcp.experiments import config_key
from tcip_mcp.pipelines.resolution import sidecar_key, verify_stamp_binding
from tcip_mcp.tools.project_tools import upsert_dataset

SUBJECT = "bud"
ATTRIBUTE = "opening"
VALUE_ID_MAP = {"open": 0, "closed": 1}


def _load_script():
    from tcip_mcp.cli import repair_classified_predictions

    return repair_classified_predictions


def _base_stamp(*, id_map: dict, experiment_id: str = "exp-classified", **overrides) -> dict:
    from tcip_mcp.pipelines.resolution import VALIDATED_HELD_OUT

    stamp = {
        "trait": ATTRIBUTE, "dataset_hash": "h",
        "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_HELD_OUT}},
        "id_map": id_map,
        "validated": False, "validated_by": None, "tile_size_validated": None,
        "shippable_issues": [], "checkpoint": "m", "checkpoint_sha256": "f" * 64,
        "experiment_id": experiment_id, "images_dir": None, "raster_path": None,
        "produced_at": "2026-01-01T00:00:00+00:00",
    }
    stamp.update(overrides)
    return stamp


def _write_doc(bucket: Path, stem: str, annotations: list[Annotation], w=100, h=80) -> Path:
    bucket.mkdir(parents=True, exist_ok=True)
    path = bucket / f"{stem}.json"
    write_annotations(str(path), annotations, w, h, keep_empty=True)
    return path


def _register(project_root: Path, dataset_root: Path, *, dataset_id: str = "ds-1") -> None:
    (project_root / ".tcip").mkdir(parents=True, exist_ok=True)
    upsert_dataset(project_root, {"id": dataset_id, "path": str(dataset_root), "crop": "currant",
                                  "fingerprint": None})


def _write_experiment_config(experiment_id: str, root: Path, data: dict) -> None:
    ts.replace(config_key(experiment_id, root=root), {"data": data}, expect=ts.Version.ABSENT)


def test_a_rewrite_floors_a_count_claim_sealed_over_the_old_content(tmp_path):
    from tests._binding_fixtures import file_validation_record

    bind_default()
    module = _load_script()
    project = tmp_path / "project"
    dataset_root = tmp_path / "dataset"
    _register(project, dataset_root)
    bucket = dataset_root / "predictions" / "classifier" / "2026-01-01"
    _write_doc(bucket, "img1", [Annotation(subject="open", geometry=BBox(0, 0, 10, 10), score=0.9)])
    _write_doc(bucket, "img2", [Annotation(subject="closed", geometry=BBox(5, 5, 15, 15), score=0.8)])
    sealed = file_validation_record(
        _base_stamp(id_map=VALUE_ID_MAP, validated=True), dataset_root=dataset_root,
        pred_dirs=[bucket])
    ts.replace(sidecar_key(bucket, "operating_point"), sealed, expect=ts.Version.ABSENT)
    _write_experiment_config("exp-classified", project,
                              {"subject": SUBJECT, "attribute": ATTRIBUTE, "id_map": VALUE_ID_MAP})

    before = verify_stamp_binding(sealed, bucket, document="operating_point")
    assert before.claimed is True and before.ok is True  # sealed cleanly over the old content

    outcomes, refused = module.process_project_root(
        project, plan=False, operator_subject=None, operator_attribute=None)

    assert refused is False
    joined = "\n".join(outcomes)
    assert "count claim over this bucket floors" in joined
    assert "stored validated=True" in joined
    assert "calibrate_count_operating_point" in joined

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    stamp_after = read_operating_point_sidecar(bucket)
    assert stamp_after["validated"] is True  # the stored assertion is left exactly as written
    after = verify_stamp_binding(stamp_after, bucket, document="operating_point")
    assert after.claimed is True and after.ok is False  # the effective binding is floored
