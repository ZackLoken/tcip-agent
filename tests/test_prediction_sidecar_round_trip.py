"""A written prediction bucket has to be readable by the readers the delivery doors use.

``run_inference`` writes ``operating_point.json`` beside the predictions, and every door that
later assembles a phenotype from that bucket reads its validity back out of that file rather than
from a caller's word. These tests drive the real writer and the real readers against each other, so
a stamp the writer produces but no reader can find is a failure here rather than downstream.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

CONF_FROM_THE_DENSE_REFERENCE = 0.9  # every correct detection in the fixture below scores this


class _BucketStub:
    """A predictor returning one detection per image, enough to write a real bucket."""

    def __init__(self) -> None:
        from types import SimpleNamespace

        self.model = SimpleNamespace(score_thresh=0.5, nms_thresh=0.5, detections_per_img=100)
        self.device = "cpu"
        self.score_threshold = 0.5
        self.train_tile_size = None
        self.train_overlap = None

    def predict_batch(self, paths, **kw):
        return [{"image": p, "width": 160, "height": 120,
                 "boxes": [[10, 10, 30, 30]], "scores": [0.95], "labels": [1], "count": 1}
                for p in paths]


def _held_out_calibration(*, tiled: bool, tile_size: int | None = None,
                          tile_size_source: str = "default",
                          tile_size_derived_from: str | None = None):
    """The calibrated bundle plus the resolver arguments behind it, the pair a real calibration
    hands its caller so a delivery door can reopen the same gate."""
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tests._dense_op_fixtures import dense_records

    n_images, objects_per_image = 20, 80
    miss, fp = [0] * n_images, [1] * n_images
    inputs = {
        "dataset_hash": "H",
        "calibration_records": dense_records(
            n_images=n_images, objects_per_image=objects_per_image, id_prefix="c",
            miss_pattern=miss, fp_pattern=fp, score=CONF_FROM_THE_DENSE_REFERENCE, fp_score=0.05),
        "holdout_records": dense_records(
            n_images=n_images, objects_per_image=objects_per_image, id_prefix="h", shift=5.0,
            miss_pattern=miss, fp_pattern=fp, score=CONF_FROM_THE_DENSE_REFERENCE, fp_score=0.05),
        "tiled": tiled,
        "tile_size": tile_size,
        "tile_size_source": tile_size_source,
        "tile_size_derived_from": tile_size_derived_from,
        "staged_conf_floor": 0.01,
    }
    return resolve_operating_point("bud_opening", experiment_id=None, **inputs), inputs


def _export(tmp_path, monkeypatch, *, calibration, tile, tile_size=None):
    import tcip_mcp.pipelines.calibration as calibration_pipeline
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    import tcip_mcp.tools.inference_tools as itools
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    from PIL import Image

    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "images" / "2026-03-01"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (160, 120), color=(70, 90, 110)).save(images_dir / "capture_a.png")

    bundle, inputs = calibration
    evidence = {"resolver": "resolve_operating_point", "inputs": inputs,
                "reference_inputs": {"label_dirs": {"calibration": str(images_dir)}}}
    monkeypatch.setattr(calibration_pipeline, "calibrate_operating_point",
                        lambda *a, **k: (bundle, "H", 0, evidence))
    monkeypatch.setattr(predictor_mod, "build_predictor", lambda checkpoint, **kw: _BucketStub())
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path)
    result = itools.run_inference(
        str(ckpt), images_dir=str(images_dir),
        output_dir=str(dataset_root / "predictions" / "baseline" / "2026-03-01"),
        device="cpu", tile=tile, tile_size=tile_size, trait="bud_opening",
        calibration_labels_dir=str(images_dir))
    assert "error" not in result, result
    return result


def test_the_count_operating_points_validity_survives_the_round_trip_to_disk(tmp_path, monkeypatch):
    """The conf a calibrated run resolved is recoverable from the written bucket by the same
    reconciliation the delivery doors gate on, with its held-out reference intact."""
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_HELD_OUT, read_operating_point_sidecar, reconcile_operating_point_validity,
    )

    result = _export(tmp_path, monkeypatch, calibration=_held_out_calibration(tiled=False),
                     tile=False)
    bucket = result["output_dir"]

    assert read_operating_point_sidecar(bucket) is not None
    reconciled = reconcile_operating_point_validity([bucket], trait="bud_opening")
    assert reconciled["missing_sidecars"] == []
    assert reconciled["unvalidated_buckets"] == []
    assert reconciled["on_disk_validated"] is True
    assert reconciled["validated"] == VALIDATED_HELD_OUT
    assert reconciled["conf"] == pytest.approx(CONF_FROM_THE_DENSE_REFERENCE)


def test_the_validated_stamps_pointer_leads_to_a_record_that_answers_for_its_claim(
    tmp_path, monkeypatch,
):
    """The stamp a validated export writes names a row outside the bucket, and the reader's own
    verification of that binding passes against the bucket as it was actually written."""
    from tcip_mcp.pipelines.resolution import (
        read_operating_point_sidecar, verify_stamp_binding, well_formed_validated_by,
    )

    result = _export(tmp_path, monkeypatch, calibration=_held_out_calibration(tiled=False),
                     tile=False)
    bucket = result["output_dir"]

    stamp = read_operating_point_sidecar(bucket)
    assert well_formed_validated_by(stamp) is not None
    binding = verify_stamp_binding(stamp, bucket, document="operating_point", trait="bud_opening")
    assert binding.ok is True
    assert binding.claimed is True
    assert binding.note == ""


def test_a_registered_bespoke_checkpoint_exports_and_earns_its_own_calibration_record(
    tmp_path, monkeypatch,
):
    """A bespoke checkpoint registered through the register_model tool's explicit mode, with no
    experiment behind it, exports predictions and earns a validated count against an experiment
    created for the calibration, with the producing run recorded as absent rather than invented.
    An unregistered checkpoint is refused before any of this runs; that refusal is
    test_run_inference_refuses_an_unregistered_checkpoint_and_writes_nothing in
    test_checkpoint_digest_rails.py."""
    from tcip_mcp.experiments import experiment_exists, find_validation
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    result = _export(tmp_path, monkeypatch, calibration=_held_out_calibration(tiled=False),
                     tile=False)
    bucket = Path(result["output_dir"])

    assert (bucket / "capture_a.json").is_file()  # written once the registered checkpoint admits
    pointer = read_operating_point_sidecar(bucket)["validated_by"]
    assert experiment_exists(pointer["experiment_id"])
    row = find_validation(pointer["experiment_id"], pointer["record_digest"])
    assert row["producing_experiment_id"] is None
    assert row["trait"] == "bud_opening"
    assert list(row["covered_buckets"]) == ["predictions/baseline/2026-03-01"]


def test_a_run_that_dies_before_its_record_leaves_predictions_that_floor(tmp_path, monkeypatch):
    """The publication order fails closed at every partial state: files written and no stamp is a
    bucket that delivers nothing, which is the safe direction."""
    import tcip_mcp.pipelines.resolution as resolution

    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, reconcile_operating_point_validity

    def _die(*a, **kw):
        raise RuntimeError("the process died between the prediction files and the record")

    monkeypatch.setattr(resolution, "seal_validation", _die)
    with pytest.raises(RuntimeError):
        _export(tmp_path, monkeypatch, calibration=_held_out_calibration(tiled=False), tile=False)

    bucket = tmp_path / "dataset" / "predictions" / "baseline" / "2026-03-01"
    assert (bucket / "capture_a.json").is_file()
    assert not (bucket / "operating_point.json").exists()
    assert reconcile_operating_point_validity([str(bucket)], trait="bud_opening")["validated"] == VALIDATED_FALSE


def test_a_run_that_dies_after_its_record_leaves_a_row_no_stamp_names(tmp_path, monkeypatch):
    """The other partial state: the row is appended and inert, because nothing points at it, and
    the bucket still floors."""
    import tcip_mcp.pipelines.resolution as resolution

    from tcip_mcp.experiments import find_validation
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, reconcile_operating_point_validity

    sealed: dict = {}
    real_seal = resolution.seal_validation

    def _seal_then_die(draft, **kw):
        digest, body = real_seal(draft, **kw)
        sealed.update(body["validated_by"])
        raise RuntimeError("the process died between the record and the stamp")

    monkeypatch.setattr(resolution, "seal_validation", _seal_then_die)
    with pytest.raises(RuntimeError):
        _export(tmp_path, monkeypatch, calibration=_held_out_calibration(tiled=False), tile=False)

    bucket = tmp_path / "dataset" / "predictions" / "baseline" / "2026-03-01"
    assert find_validation(sealed["experiment_id"], sealed["record_digest"]) is not None
    assert not (bucket / "operating_point.json").exists()
    assert reconcile_operating_point_validity([str(bucket)], trait="bud_opening")["validated"] == VALIDATED_FALSE


def test_the_tile_geometrys_basis_survives_the_round_trip_to_disk(tmp_path, monkeypatch):
    """A tiled bucket's tile scale is readable back as the operative, explicitly-stated basis it
    was written at, the dimension a multi-bucket delivery floors itself against."""
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_EXPLICIT_GEOMETRY, reconcile_tile_size_validity,
    )

    calibration = _held_out_calibration(
        tiled=True, tile_size=64, tile_size_source="explicit",
        tile_size_derived_from="stated on a checkpoint that records no tile geometry")
    result = _export(tmp_path, monkeypatch, calibration=calibration, tile=True, tile_size=64)
    bucket = result["output_dir"]

    reconciled = reconcile_tile_size_validity([bucket])
    assert reconciled["operative"] is True
    assert reconciled["unvalidated_buckets"] == []
    assert reconciled["validated"] == VALIDATED_EXPLICIT_GEOMETRY
    assert reconciled["per_bucket"] == {str(Path(bucket)): VALIDATED_EXPLICIT_GEOMETRY}
