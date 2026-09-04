"""``calibrate_count_operating_point``: the re-admitted count calibrator earns (or honestly
declines) a validation claim over an already-published prediction bucket, running the identical
resolution ``scripts/calibrate_operating_point.py`` prints and writes nothing for.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")


def _stub_predictor(monkeypatch) -> None:
    import tcip_mcp.model_registry as model_registry_mod

    from tests._verified_checkpoint_fixtures import stub_verified_checkpoint

    monkeypatch.setattr(model_registry_mod, "load_registered_checkpoint",
                        lambda path, *a, **kw: stub_verified_checkpoint(str(path)))

    class _Predictor:
        def __init__(self):
            self.model = SimpleNamespace(detector=SimpleNamespace(
                roi_heads=SimpleNamespace(score_thresh=0.5, nms_thresh=0.5, detections_per_img=100)))
            self.device = "cpu"
            self.train_tile_size = None

    monkeypatch.setattr("tcip_mcp.pipelines.inference.predictor.build_predictor",
                        lambda checkpoint=None, **kw: _Predictor())

    class _Probe:
        stems = ["a", "b"]

    monkeypatch.setattr("tcip_mcp.pipelines.data.datasets.build_dataset", lambda *a, **kw: _Probe())
    monkeypatch.setattr("tcip_mcp.pipelines.data.splits.count_label_lines",
                        lambda labels_dir, s, **kw: 1)
    monkeypatch.setattr("tcip_mcp.pipelines.data.splits.resolve_locked_cal_holdout_split",
                        lambda stems, **kw: {"calibration": ["a"], "holdout": ["b"]})
    monkeypatch.setattr("torch.utils.data.DataLoader", lambda ds, **kw: ds)
    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.records_over_loader",
                        lambda model, loader, device, task: [])
    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.attach_split_policy_provenance",
                        lambda b, locked: None)


def _label_pair(tmp_path):
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    for stem in ("a", "b"):
        json_io.write_annotations(
            str(labels_dir / f"{stem}.json"),
            [Annotation(subject="catkin", geometry=BBox(1, 1, 5, 5))], 10, 10)
    return labels_dir


def _existing_bucket(tmp_path, *, checkpoint_sha256="stub-sha256", tile_size_validated="false"):
    from tcip_mcp.pipelines.resolution import (
        ResolvedBundle, derived, operating_point_stamp, write_sidecar,
    )

    dataset_root = tmp_path / "dataset"
    pred_dir = dataset_root / "predictions"
    pred_dir.mkdir(parents=True)
    op = ResolvedBundle(trait="catkin", dataset_hash="H", params={
        "conf": derived("conf", 0.3, requires_validation=True, validation_kind="annotations",
                        derived_from="x", validated_against="false"),
    }).to_provenance()["operating_point"]
    stamp = operating_point_stamp(
        op, validated=False, validated_by=None, tile_size_validated=tile_size_validated,
        shippable_issues=[], id_map={"1": "catkin"}, trait="catkin", dataset_hash="H",
        checkpoint="m", checkpoint_sha256=checkpoint_sha256, experiment_id=None,
        images_dir=str(tmp_path / "images"), raster_path=None,
        produced_at="2024-01-01T00:00:00Z",
    )
    write_sidecar(pred_dir, stamp)
    return dataset_root, pred_dir


def _resolve_op_shippable(trait_name, **kw):
    from tcip_mcp.pipelines.resolution import ResolvedBundle, VALIDATED_HELD_OUT, derived

    conf = derived("conf", 0.42, requires_validation=True, validation_kind="annotations",
                   derived_from="held-out sweep", validated_against=VALIDATED_HELD_OUT,
                   gate_evidence={"passed_holdout": True})
    return ResolvedBundle(trait=trait_name, dataset_hash=kw.get("dataset_hash"), params={"conf": conf})


def _resolve_op_unshippable(trait_name, **kw):
    from tcip_mcp.pipelines.resolution import ResolvedBundle, derived

    conf = derived("conf", 0.3, requires_validation=True, validation_kind="annotations",
                   derived_from="held-out sweep", validated_against="false",
                   gate_evidence={"passed_holdout": False, "failures": ["holdout_count_bias"]})
    return ResolvedBundle(trait=trait_name, dataset_hash=kw.get("dataset_hash"), params={"conf": conf})


def test_calibrate_count_operating_point_earns_a_validated_stamp(monkeypatch, tmp_path):
    """The gate clearing merges the earned conf and validated_by into the bucket's existing
    stamp, preserving every field this door never touches (id_map, images_dir, checkpoint).

    The existing bucket's tile geometry is stated (``tile_size_validated=None``, not the
    floored ``"false"``) so this exercises the genuine earn path; the tile floor itself is
    pinned separately below."""
    _stub_predictor(monkeypatch)
    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.resolve_operating_point",
                        _resolve_op_shippable)
    labels_dir = _label_pair(tmp_path)
    dataset_root, pred_dir = _existing_bucket(tmp_path, tile_size_validated=None)

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar
    from tcip_mcp.tools.calibration_tools import calibrate_count_operating_point

    result = calibrate_count_operating_point(
        checkpoint_path="x.pt", trait="catkin", labels_dir=str(labels_dir),
        images_dir=str(tmp_path / "images"), dataset_root=str(dataset_root),
        pred_dir=str(pred_dir),
    )

    assert "error" not in result, result
    assert result["validated"] is True
    assert result["validated_by"]["experiment_id"]
    assert result["validated_against"] == "held_out_annotations"

    on_disk = read_operating_point_sidecar(pred_dir)
    assert on_disk["validated"] is True
    assert on_disk["operating_point"]["conf"]["value"] == pytest.approx(0.42)
    assert on_disk["validated_by"]["experiment_id"]
    assert on_disk["id_map"] == {"1": "catkin"}
    assert on_disk["images_dir"] == str(tmp_path / "images")
    assert on_disk["checkpoint_sha256"] == "stub-sha256"


def test_calibrate_count_operating_point_folds_the_tile_floor_into_validated(monkeypatch, tmp_path):
    """A bucket whose tile geometry never validated (``tile_size_validated="false"``) cannot earn
    a validated count operating point no matter how clean the count gate itself is:
    ``operating_point_stamp``'s own floor (``fold_tile_validation``) decides ``validated`` here
    too, so the conf still merges in but the claim does not."""
    _stub_predictor(monkeypatch)
    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.resolve_operating_point",
                        _resolve_op_shippable)
    labels_dir = _label_pair(tmp_path)
    dataset_root, pred_dir = _existing_bucket(tmp_path)  # tile_size_validated="false" default

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar
    from tcip_mcp.tools.calibration_tools import calibrate_count_operating_point

    result = calibrate_count_operating_point(
        checkpoint_path="x.pt", trait="catkin", labels_dir=str(labels_dir),
        images_dir=str(tmp_path / "images"), dataset_root=str(dataset_root),
        pred_dir=str(pred_dir),
    )

    assert "error" not in result, result
    assert result["validated"] is False
    assert result["validated_by"] is None

    on_disk = read_operating_point_sidecar(pred_dir)
    assert on_disk["validated"] is False
    assert on_disk["validated_by"] is None
    assert on_disk["operating_point"]["conf"]["value"] == pytest.approx(0.42)
    from tcip_mcp.project_paths import platform_state_root

    assert not (platform_state_root() / ".tcip" / "experiments").exists()


def test_calibrate_count_operating_point_writes_an_honest_unvalidated_stamp(monkeypatch, tmp_path):
    """A calibration that does not clear its gate merges the honest, unvalidated conf and earns
    no validation record."""
    _stub_predictor(monkeypatch)
    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.resolve_operating_point",
                        _resolve_op_unshippable)
    labels_dir = _label_pair(tmp_path)
    dataset_root, pred_dir = _existing_bucket(tmp_path)

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar
    from tcip_mcp.tools.calibration_tools import calibrate_count_operating_point

    result = calibrate_count_operating_point(
        checkpoint_path="x.pt", trait="catkin", labels_dir=str(labels_dir),
        images_dir=str(tmp_path / "images"), dataset_root=str(dataset_root),
        pred_dir=str(pred_dir),
    )

    assert "error" not in result, result
    assert result["validated"] is False
    assert result["validated_by"] is None

    on_disk = read_operating_point_sidecar(pred_dir)
    assert on_disk["validated"] is False
    assert on_disk["validated_by"] is None
    assert on_disk["operating_point"]["conf"]["value"] == pytest.approx(0.3)
    assert on_disk["gate_evidence_summary"]["passed_holdout"] is False

    from tcip_mcp.project_paths import platform_state_root

    assert not (platform_state_root() / ".tcip" / "experiments").exists()


def test_calibrate_count_operating_point_refuses_a_bucket_outside_dataset_root(tmp_path):
    from tcip_mcp.tools.calibration_tools import calibrate_count_operating_point

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()

    result = calibrate_count_operating_point(
        checkpoint_path="x.pt", trait="catkin", labels_dir=str(tmp_path / "labels"),
        images_dir=str(tmp_path / "images"), dataset_root=str(dataset_root),
        pred_dir=str(outside),
    )

    assert "error" in result
    assert "not under dataset_root" in result["error"]


def test_calibrate_count_operating_point_refuses_a_bucket_with_no_stamp(tmp_path):
    from tcip_mcp.tools.calibration_tools import calibrate_count_operating_point

    dataset_root = tmp_path / "dataset"
    pred_dir = dataset_root / "predictions"
    pred_dir.mkdir(parents=True)

    result = calibrate_count_operating_point(
        checkpoint_path="x.pt", trait="catkin", labels_dir=str(tmp_path / "labels"),
        images_dir=str(tmp_path / "images"), dataset_root=str(dataset_root),
        pred_dir=str(pred_dir),
    )

    assert "error" in result
    assert "no operating_point.json stamp" in result["error"]


def test_calibrate_count_operating_point_refuses_an_already_validated_bucket(tmp_path):
    from tcip_mcp.pipelines.resolution import ResolvedBundle, derived, operating_point_stamp
    from tcip_mcp.tools.calibration_tools import calibrate_count_operating_point

    from tests._binding_fixtures import write_bound_sidecar, write_prediction

    dataset_root = tmp_path / "dataset"
    pred_dir = dataset_root / "predictions"
    write_prediction(pred_dir, "a")
    op = ResolvedBundle(trait="catkin", dataset_hash="H", params={
        "conf": derived("conf", 0.4, requires_validation=True, validation_kind="annotations",
                        derived_from="x", validated_against="held_out_annotations"),
    }).to_provenance()["operating_point"]
    stamp = operating_point_stamp(
        op, validated=True, validated_by=None, tile_size_validated=None, shippable_issues=[],
        id_map={"1": "catkin"}, trait="catkin", dataset_hash="H", checkpoint="m",
        checkpoint_sha256="stub-sha256", experiment_id=None,
        images_dir=str(tmp_path / "images"), raster_path=None,
        produced_at="2024-01-01T00:00:00Z",
    )
    # A genuine, producer-filed record behind validated_by (tests._binding_fixtures does what
    # seal_validation does for a producer), not a hand-typed pointer naming no real record.
    write_bound_sidecar(pred_dir, stamp, dataset_root=dataset_root)

    result = calibrate_count_operating_point(
        checkpoint_path="x.pt", trait="catkin", labels_dir=str(tmp_path / "labels"),
        images_dir=str(tmp_path / "images"), dataset_root=str(dataset_root),
        pred_dir=str(pred_dir),
    )

    assert "error" in result
    assert "already carries a validated" in result["error"]


def test_calibrate_count_operating_point_refuses_a_checkpoint_mismatch_before_the_pass(
    monkeypatch, tmp_path,
):
    """The checkpoint identity is derived and checked before the calibration pass draws its
    cal/holdout lock: a mismatch refuses having run no pass at all."""
    import tcip_mcp.model_registry as model_registry_mod

    from tests._verified_checkpoint_fixtures import stub_verified_checkpoint

    monkeypatch.setattr(
        model_registry_mod, "load_registered_checkpoint",
        lambda path, *a, **kw: stub_verified_checkpoint(str(path), sha256="new-sha256"))

    def _never(**kw):
        raise AssertionError("the pass must not run before the checkpoint identity is checked")

    monkeypatch.setattr("tcip_mcp.pipelines.count_calibration.resolve_count_operating_point", _never)

    dataset_root, pred_dir = _existing_bucket(tmp_path, checkpoint_sha256="old-sha256")

    from tcip_mcp.tools.calibration_tools import calibrate_count_operating_point

    result = calibrate_count_operating_point(
        checkpoint_path="x.pt", trait="catkin", labels_dir=str(tmp_path / "labels"),
        images_dir=str(tmp_path / "images"), dataset_root=str(dataset_root),
        pred_dir=str(pred_dir),
    )

    assert "error" in result
    assert "was produced by checkpoint" in result["error"]


def test_calibrate_count_operating_point_refuses_a_stamp_with_no_checkpoint_sha256(
    monkeypatch, tmp_path,
):
    """A stamp sealed under a digest it does not carry could never bind at delivery, so a bucket
    with no ``checkpoint_sha256`` at all refuses the same as a disagreeing one."""
    import tcip_mcp.model_registry as model_registry_mod

    from tcip_mcp.tools.calibration_tools import calibrate_count_operating_point
    from tests._verified_checkpoint_fixtures import stub_verified_checkpoint

    monkeypatch.setattr(model_registry_mod, "load_registered_checkpoint",
                        lambda path, *a, **kw: stub_verified_checkpoint(str(path)))

    dataset_root, pred_dir = _existing_bucket(tmp_path, checkpoint_sha256=None)

    result = calibrate_count_operating_point(
        checkpoint_path="x.pt", trait="catkin", labels_dir=str(tmp_path / "labels"),
        images_dir=str(tmp_path / "images"), dataset_root=str(dataset_root),
        pred_dir=str(pred_dir),
    )

    assert "error" in result
    assert "carries no checkpoint_sha256" in result["error"]


def test_calibrate_count_operating_point_refuses_a_stamp_validated_mid_pass(monkeypatch, tmp_path):
    """The merge decides against the stamp as it is actually stored when the lock is taken, not
    the copy read before the (potentially long) calibration pass: a stamp another process
    validates while this pass is running is left exactly as that process left it."""
    _stub_predictor(monkeypatch)
    labels_dir = _label_pair(tmp_path)
    dataset_root, pred_dir = _existing_bucket(tmp_path, tile_size_validated=None)

    from tcip_mcp.pipelines.resolution import (
        VALIDATED_HELD_OUT, ResolvedBundle, derived, read_operating_point_sidecar, write_sidecar,
    )

    def _resolve_op_and_race(trait_name, **kw):
        stamp = read_operating_point_sidecar(pred_dir)
        stamp["validated"] = True
        stamp["trait"] = trait_name
        stamp["validated_by"] = {"experiment_id": "exp_other", "record_digest": "other123"}
        write_sidecar(pred_dir, stamp)
        conf = derived("conf", 0.42, requires_validation=True, validation_kind="annotations",
                       derived_from="held-out sweep", validated_against=VALIDATED_HELD_OUT,
                       gate_evidence={"passed_holdout": True})
        return ResolvedBundle(trait=trait_name, dataset_hash=kw.get("dataset_hash"),
                              params={"conf": conf})

    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.resolve_operating_point",
                        _resolve_op_and_race)

    from tcip_mcp.tools.calibration_tools import calibrate_count_operating_point

    result = calibrate_count_operating_point(
        checkpoint_path="x.pt", trait="catkin", labels_dir=str(labels_dir),
        images_dir=str(tmp_path / "images"), dataset_root=str(dataset_root),
        pred_dir=str(pred_dir),
    )

    assert "error" in result
    assert "changed while" in result["error"]

    on_disk = read_operating_point_sidecar(pred_dir)
    assert on_disk["validated_by"] == {"experiment_id": "exp_other", "record_digest": "other123"}


def test_script_and_tool_call_the_same_count_calibration_function(monkeypatch, tmp_path):
    """Import identity, not a second implementation: both entry points resolve
    ``resolve_count_operating_point`` off the one module at call time."""
    calls = []

    def _stub(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("stub-count-calibration-called")

    monkeypatch.setattr("tcip_mcp.pipelines.count_calibration.resolve_count_operating_point", _stub)

    from scripts.calibrate_operating_point import main

    with pytest.raises(RuntimeError, match="stub-count-calibration-called"):
        main(["--checkpoint", "x.pt", "--trait", "catkin",
              "--labels-dir", str(tmp_path / "labels"), "--images-dir", str(tmp_path / "images"),
              "--dataset-root", str(tmp_path), "--project-root", str(tmp_path)])
    assert len(calls) == 1

    dataset_root, pred_dir = _existing_bucket(tmp_path)

    import tcip_mcp.model_registry as model_registry_mod

    from tcip_mcp.tools.calibration_tools import calibrate_count_operating_point
    from tests._verified_checkpoint_fixtures import stub_verified_checkpoint

    # Matches _existing_bucket's checkpoint_sha256, so this door's own pre-pass identity check
    # clears and the stub below is reached, same as the script's own call above.
    monkeypatch.setattr(model_registry_mod, "load_registered_checkpoint",
                        lambda path, *a, **kw: stub_verified_checkpoint(str(path)))

    with pytest.raises(RuntimeError, match="stub-count-calibration-called"):
        calibrate_count_operating_point(
            checkpoint_path="x.pt", trait="catkin", labels_dir=str(tmp_path / "labels"),
            images_dir=str(tmp_path / "images"), dataset_root=str(dataset_root),
            pred_dir=str(pred_dir),
        )
    assert len(calls) == 2
