"""``calibrate_count_operating_point``: the re-admitted count calibrator earns (or honestly
declines) a validation claim over an already-published prediction bucket, running the identical
resolution ``tcip calibrate-operating-point`` prints and writes nothing for.
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


def _stub_dense_pass(monkeypatch, cal_stems, hold_stems, cal_records, hold_records) -> None:
    """Stub every model-pass mechanic ``resolve_count_operating_point`` composes around its own
    ``resolve_operating_point`` call (the checkpoint, the predictor, dataset probing and record
    collection), leaving the resolver itself to run for real over the records supplied.
    """
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

    all_stems = cal_stems + hold_stems

    def _build_dataset(*a, stems=None, **kw):
        return SimpleNamespace(stems=stems if stems is not None else all_stems)

    monkeypatch.setattr("tcip_mcp.pipelines.data.datasets.build_dataset", _build_dataset)
    monkeypatch.setattr("tcip_mcp.pipelines.data.splits.count_label_lines",
                        lambda labels_dir, s, **kw: 1)
    monkeypatch.setattr(
        "tcip_mcp.pipelines.data.splits.resolve_locked_cal_holdout_split",
        lambda stems, **kw: {"calibration": cal_stems, "holdout": hold_stems})
    monkeypatch.setattr("torch.utils.data.DataLoader", lambda ds, **kw: ds)

    def _records_over_loader(model, loader, device, task):
        if list(loader.stems) == cal_stems:
            return cal_records
        if list(loader.stems) == hold_stems:
            return hold_records
        raise AssertionError(f"unexpected stems requested from records_over_loader: {loader.stems}")

    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.records_over_loader",
                        _records_over_loader)
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
            [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))], 10, 10)
    return labels_dir


def _existing_bucket(tmp_path, *, checkpoint_sha256="stub-sha256", tile_size_validated="false",
                     conf_value=0.3, trait="bud_opening", with_prediction=True):
    from tcip_mcp.pipelines.resolution import (
        ResolvedBundle, derived, operating_point_stamp, write_sidecar,
    )

    dataset_root = tmp_path / "dataset"
    pred_dir = dataset_root / "predictions"
    pred_dir.mkdir(parents=True)
    op = ResolvedBundle(trait=trait, dataset_hash="H", params={
        "conf": derived("conf", conf_value, requires_validation=True,
                        validation_kind="annotations", derived_from="x",
                        validated_against="false"),
    }).to_provenance()["operating_point"]
    stamp = operating_point_stamp(
        op, validated=False, validated_by=None, tile_size_validated=tile_size_validated,
        shippable_issues=[], id_map={trait: 0}, trait=trait, dataset_hash="H",
        checkpoint="m", checkpoint_sha256=checkpoint_sha256, experiment_id=None,
        images_dir=str(tmp_path / "images"), raster_path=None,
        produced_at="2024-01-01T00:00:00Z", subject=trait, attribute=None,
    )
    write_sidecar(pred_dir, stamp)
    if with_prediction:
        from tests._binding_fixtures import write_prediction

        write_prediction(pred_dir, "a")
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


def test_calibrate_count_operating_point_earns_a_validated_stamp(
    monkeypatch, tmp_path, seed_bud_trait_spec,
):
    """The gate clears through the real ``resolve_operating_point`` (no resolver stub) over a
    dense held-out reference, and the merge earns a validated stamp only because the bucket's
    own stamp already records production at the exact conf this calibration resolves to.
    """
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tests._dense_op_fixtures import dense_records

    n_images, objects = 20, 80
    cal_records = dense_records(n_images=n_images, objects_per_image=objects, id_prefix="c",
                                fp_pattern=[1] * n_images, score=0.9, fp_score=0.05)
    hold_records = dense_records(n_images=n_images, objects_per_image=objects, id_prefix="h",
                                 shift=5.0, fp_pattern=[1] * n_images, score=0.9, fp_score=0.05)
    cal_stems = [r["image_id"] for r in cal_records]
    hold_stems = [r["image_id"] for r in hold_records]

    probe = resolve_operating_point("bud_opening", dataset_hash="H", calibration_records=cal_records,
                                    holdout_records=hold_records, tiled=False,
                                    staged_conf_floor=0.01)
    assert probe.is_shippable, probe.shippable_issues()
    production_conf = probe.get("conf").to_provenance()["value"]

    _stub_dense_pass(monkeypatch, cal_stems, hold_stems, cal_records, hold_records)
    labels_dir = _label_pair(tmp_path)
    dataset_root, pred_dir = _existing_bucket(
        tmp_path, tile_size_validated=None, conf_value=production_conf)

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar
    from tcip_mcp.tools.calibration_tools import calibrate_count_operating_point

    result = calibrate_count_operating_point(
        checkpoint_path="x.pt", trait="bud_opening", labels_dir=str(labels_dir),
        images_dir=str(tmp_path / "images"), dataset_root=str(dataset_root),
        pred_dir=str(pred_dir),
    )

    assert "error" not in result, result
    assert result["validated"] is True
    assert result["validated_by"]["experiment_id"]
    assert result["validated_against"] == "held_out_annotations"

    on_disk = read_operating_point_sidecar(pred_dir)
    assert on_disk["validated"] is True
    assert on_disk["operating_point"]["conf"]["value"] == pytest.approx(production_conf)
    assert on_disk["validated_by"]["experiment_id"]
    assert on_disk["id_map"] == {"bud_opening": 0}
    assert on_disk["images_dir"] == str(tmp_path / "images")
    assert on_disk["checkpoint_sha256"] == "stub-sha256"
    assert on_disk["trait"] == "bud_opening"
    assert on_disk["shippable_issues"] == []


def test_calibrate_count_operating_point_refuses_when_earned_conf_differs_from_production(
    monkeypatch, tmp_path,
):
    """A bucket's stored detections were filtered at the conf its stamp records, never at
    whatever conf a later calibration resolves: any other earned conf refuses by name rather
    than silently overwriting the production conf the stored detections never saw. The
    production conf is read before the pass runs, so this ordinary mismatch is caught before
    open_validation/seal_validation ever run and mints no calibration experiment.
    """
    _stub_predictor(monkeypatch)
    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.resolve_operating_point",
                        _resolve_op_shippable)  # earns conf=0.42
    labels_dir = _label_pair(tmp_path)
    dataset_root, pred_dir = _existing_bucket(tmp_path, tile_size_validated=None, conf_value=0.3)

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar
    from tcip_mcp.tools.calibration_tools import calibrate_count_operating_point

    result = calibrate_count_operating_point(
        checkpoint_path="x.pt", trait="bud_opening", labels_dir=str(labels_dir),
        images_dir=str(tmp_path / "images"), dataset_root=str(dataset_root),
        pred_dir=str(pred_dir),
    )

    assert "error" in result
    assert "0.3" in result["error"]
    assert "0.42" in result["error"]

    on_disk = read_operating_point_sidecar(pred_dir)
    assert on_disk["validated"] is False
    assert on_disk["operating_point"]["conf"]["value"] == pytest.approx(0.3)

    from tcip_mcp.project_paths import platform_state_root

    assert not (platform_state_root() / ".tcip" / "experiments").exists()


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
        checkpoint_path="x.pt", trait="bud_opening", labels_dir=str(labels_dir),
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
        checkpoint_path="x.pt", trait="bud_opening", labels_dir=str(labels_dir),
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


def test_calibrate_count_operating_point_does_not_write_trait_on_the_unvalidated_path(
    monkeypatch, tmp_path,
):
    """The trait is written into the stamp only when this calibration actually earns the claim
    (a gate cleared): an honest unvalidated merge leaves whatever trait the bucket's own stamp
    already carried untouched, so an unvalidated placeholder never silently relabels a bucket
    produced for a different trait.
    """
    _stub_predictor(monkeypatch)
    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.resolve_operating_point",
                        _resolve_op_unshippable)
    labels_dir = _label_pair(tmp_path)
    dataset_root, pred_dir = _existing_bucket(tmp_path, trait="other-trait")

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar
    from tcip_mcp.tools.calibration_tools import calibrate_count_operating_point

    result = calibrate_count_operating_point(
        checkpoint_path="x.pt", trait="bud_opening", labels_dir=str(labels_dir),
        images_dir=str(tmp_path / "images"), dataset_root=str(dataset_root),
        pred_dir=str(pred_dir),
    )

    assert "error" not in result, result
    assert result["validated"] is False

    on_disk = read_operating_point_sidecar(pred_dir)
    assert on_disk["trait"] == "other-trait"


def test_calibrate_count_operating_point_refuses_a_bucket_outside_dataset_root(tmp_path):
    from tcip_mcp.tools.calibration_tools import calibrate_count_operating_point

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()

    result = calibrate_count_operating_point(
        checkpoint_path="x.pt", trait="bud_opening", labels_dir=str(tmp_path / "labels"),
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
        checkpoint_path="x.pt", trait="bud_opening", labels_dir=str(tmp_path / "labels"),
        images_dir=str(tmp_path / "images"), dataset_root=str(dataset_root),
        pred_dir=str(pred_dir),
    )

    assert "error" in result
    assert "no operating_point.json stamp" in result["error"]


def test_calibrate_count_operating_point_refuses_a_raster_bucket(tmp_path):
    from tcip_mcp.pipelines.resolution import (
        ResolvedBundle, derived, operating_point_stamp, write_sidecar,
    )
    from tcip_mcp.tools.calibration_tools import calibrate_count_operating_point
    from tests._binding_fixtures import write_prediction

    dataset_root = tmp_path / "dataset"
    pred_dir = dataset_root / "predictions"
    write_prediction(pred_dir, "mosaic")
    op = ResolvedBundle(trait="bud_opening", dataset_hash="H", params={
        "conf": derived("conf", 0.3, requires_validation=True, validation_kind="annotations",
                        derived_from="x", validated_against="false"),
    }).to_provenance()["operating_point"]
    stamp = operating_point_stamp(
        op, validated=False, validated_by=None, tile_size_validated=None, shippable_issues=[],
        id_map={"bud": 0}, trait="bud_opening", dataset_hash="H", checkpoint="m",
        checkpoint_sha256="stub-sha256", experiment_id=None, images_dir=None,
        raster_path=str(tmp_path / "mosaic.tif"), produced_at="2024-01-01T00:00:00Z",
        subject="bud", attribute=None,
    )
    write_sidecar(pred_dir, stamp)

    result = calibrate_count_operating_point(
        checkpoint_path="x.pt", trait="bud_opening", labels_dir=str(tmp_path / "labels"),
        images_dir=str(tmp_path / "images"), dataset_root=str(dataset_root),
        pred_dir=str(pred_dir),
    )

    assert "error" in result
    assert "raster" in result["error"]


def test_calibrate_count_operating_point_refuses_a_bucket_with_no_prediction_documents(tmp_path):
    from tcip_mcp.tools.calibration_tools import calibrate_count_operating_point

    dataset_root, pred_dir = _existing_bucket(
        tmp_path, tile_size_validated=None, with_prediction=False)

    result = calibrate_count_operating_point(
        checkpoint_path="x.pt", trait="bud_opening", labels_dir=str(tmp_path / "labels"),
        images_dir=str(tmp_path / "images"), dataset_root=str(dataset_root),
        pred_dir=str(pred_dir),
    )

    assert "error" in result
    assert "no prediction documents" in result["error"]


def test_calibrate_count_operating_point_refuses_an_already_validated_bucket(tmp_path):
    from tcip_mcp.pipelines.resolution import ResolvedBundle, derived, operating_point_stamp
    from tcip_mcp.tools.calibration_tools import calibrate_count_operating_point

    from tests._binding_fixtures import write_bound_sidecar, write_prediction

    dataset_root = tmp_path / "dataset"
    pred_dir = dataset_root / "predictions"
    write_prediction(pred_dir, "a")
    op = ResolvedBundle(trait="bud_opening", dataset_hash="H", params={
        "conf": derived("conf", 0.4, requires_validation=True, validation_kind="annotations",
                        derived_from="x", validated_against="held_out_annotations"),
    }).to_provenance()["operating_point"]
    stamp = operating_point_stamp(
        op, validated=True, validated_by=None, tile_size_validated=None, shippable_issues=[],
        id_map={"bud": 0}, trait="bud_opening", dataset_hash="H", checkpoint="m",
        checkpoint_sha256="stub-sha256", experiment_id=None,
        images_dir=str(tmp_path / "images"), raster_path=None,
        produced_at="2024-01-01T00:00:00Z", subject="bud", attribute=None,
    )
    # A genuine, producer-filed record behind validated_by (tests._binding_fixtures does what
    # seal_validation does for a producer), not a hand-typed pointer naming no real record.
    write_bound_sidecar(pred_dir, stamp, dataset_root=dataset_root)

    result = calibrate_count_operating_point(
        checkpoint_path="x.pt", trait="bud_opening", labels_dir=str(tmp_path / "labels"),
        images_dir=str(tmp_path / "images"), dataset_root=str(dataset_root),
        pred_dir=str(pred_dir),
    )

    assert "error" in result
    assert "already carries a validated" in result["error"]


def test_calibrate_count_operating_point_treats_an_unbound_validated_claim_as_unvalidated(
    monkeypatch, tmp_path,
):
    """A stamp asserting ``validated: true`` with no record answering for it (a pointer naming
    no real record, the shape ``verify_stamp_binding`` treats as unvalidated everywhere else) is
    promotable over here too, the same as the review-promotion route: the door does not refuse
    it as already validated, and proceeds to calibrate.
    """
    _stub_predictor(monkeypatch)
    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.resolve_operating_point",
                        _resolve_op_unshippable)
    labels_dir = _label_pair(tmp_path)

    from tcip_mcp.pipelines.resolution import (
        ResolvedBundle, derived, operating_point_stamp, write_sidecar,
    )
    from tests._binding_fixtures import write_prediction

    dataset_root = tmp_path / "dataset"
    pred_dir = dataset_root / "predictions"
    write_prediction(pred_dir, "a")
    op = ResolvedBundle(trait="bud_opening", dataset_hash="H", params={
        "conf": derived("conf", 0.4, requires_validation=True, validation_kind="annotations",
                        derived_from="x", validated_against="held_out_annotations"),
    }).to_provenance()["operating_point"]
    stamp = operating_point_stamp(
        op, validated=True,
        validated_by={"experiment_id": "exp-nonexistent", "record_digest": "deadbeef"},
        tile_size_validated=None, shippable_issues=[], id_map={"bud": 0}, trait="bud_opening",
        dataset_hash="H", checkpoint="m", checkpoint_sha256="stub-sha256", experiment_id=None,
        images_dir=str(tmp_path / "images"), raster_path=None,
        produced_at="2024-01-01T00:00:00Z", subject="bud", attribute=None,
    )
    write_sidecar(pred_dir, stamp)

    from tcip_mcp.tools.calibration_tools import calibrate_count_operating_point

    result = calibrate_count_operating_point(
        checkpoint_path="x.pt", trait="bud_opening", labels_dir=str(labels_dir),
        images_dir=str(tmp_path / "images"), dataset_root=str(dataset_root),
        pred_dir=str(pred_dir),
    )

    assert "error" not in result, result
    assert result["validated"] is False


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
        checkpoint_path="x.pt", trait="bud_opening", labels_dir=str(tmp_path / "labels"),
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
        checkpoint_path="x.pt", trait="bud_opening", labels_dir=str(tmp_path / "labels"),
        images_dir=str(tmp_path / "images"), dataset_root=str(dataset_root),
        pred_dir=str(pred_dir),
    )

    assert "error" in result
    assert "carries no checkpoint_sha256" in result["error"]


def test_calibrate_count_operating_point_refuses_a_stamp_validated_mid_pass(monkeypatch, tmp_path):
    """The merge decides against the stamp as it is actually stored when the lock is taken, not
    the copy read before the (potentially long) calibration pass: a stamp another process
    validates while this pass is running, through a real record that answers for it, is left
    exactly as that process left it, and the refusal names the validation row this calibration's
    own pass left behind.
    """
    _stub_predictor(monkeypatch)
    labels_dir = _label_pair(tmp_path)
    # conf_value=0.42 matches what this calibration earns below, so the pre-pass equality
    # check clears and this test reaches the race, caught only under the lock.
    dataset_root, pred_dir = _existing_bucket(tmp_path, tile_size_validated=None, conf_value=0.42)

    from tcip_mcp.pipelines.resolution import (
        VALIDATED_HELD_OUT, ResolvedBundle, derived, read_operating_point_sidecar, write_sidecar,
    )
    from tests._binding_fixtures import file_validation_record

    def _resolve_op_and_race(trait_name, **kw):
        stamp = read_operating_point_sidecar(pred_dir)
        stamp["validated"] = True
        stamp["trait"] = trait_name
        stamp["operating_point"] = {
            **stamp["operating_point"],
            "conf": {**stamp["operating_point"]["conf"], "validated_against": VALIDATED_HELD_OUT},
        }
        bound = file_validation_record(
            stamp, dataset_root=dataset_root, pred_dirs=[pred_dir], trait=trait_name,
            experiment_id="exp_other")
        write_sidecar(pred_dir, bound)
        conf = derived("conf", 0.42, requires_validation=True, validation_kind="annotations",
                       derived_from="held-out sweep", validated_against=VALIDATED_HELD_OUT,
                       gate_evidence={"passed_holdout": True})
        return ResolvedBundle(trait=trait_name, dataset_hash=kw.get("dataset_hash"),
                              params={"conf": conf})

    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.resolve_operating_point",
                        _resolve_op_and_race)

    from tcip_mcp.tools.calibration_tools import calibrate_count_operating_point

    result = calibrate_count_operating_point(
        checkpoint_path="x.pt", trait="bud_opening", labels_dir=str(labels_dir),
        images_dir=str(tmp_path / "images"), dataset_root=str(dataset_root),
        pred_dir=str(pred_dir),
    )

    assert "error" in result
    assert "changed while" in result["error"]
    assert "answers for nothing" in result["error"]

    on_disk = read_operating_point_sidecar(pred_dir)
    assert on_disk["validated_by"]["experiment_id"] == "exp_other"


def test_script_and_tool_call_the_same_count_calibration_function(monkeypatch, tmp_path):
    """Import identity, not a second implementation: both entry points resolve
    ``resolve_count_operating_point`` off the one module at call time."""
    calls = []

    def _stub(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("stub-count-calibration-called")

    monkeypatch.setattr("tcip_mcp.pipelines.count_calibration.resolve_count_operating_point", _stub)

    from tcip_mcp.cli.calibrate_operating_point import main

    with pytest.raises(RuntimeError, match="stub-count-calibration-called"):
        main(["--checkpoint", "x.pt", "--trait", "bud",
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
            checkpoint_path="x.pt", trait="bud_opening", labels_dir=str(tmp_path / "labels"),
            images_dir=str(tmp_path / "images"), dataset_root=str(dataset_root),
            pred_dir=str(pred_dir),
        )
    assert len(calls) == 2


def _producer_bucket(tmp_path, *, subject="bud", attribute=None):
    """A stamped bucket built through the platform's own writers
    (``write_predictions_json``, ``operating_point_stamp``, ``write_sidecar``), scoped to
    ``(subject, attribute)``, for the scope-agreement tests below."""
    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json
    from tcip_mcp.pipelines.resolution import (
        ResolvedBundle, derived, operating_point_stamp, write_sidecar,
    )

    dataset_root = tmp_path / "dataset"
    pred_dir = dataset_root / "predictions"
    pred_dir.mkdir(parents=True)
    id_map = {subject: 0}
    result = {"image": "a.png", "width": 100, "height": 100,
             "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1]}
    write_predictions_json(pred_dir / "a.json", result, created_by="test-producer",
                           subject=subject, attribute=attribute, id_map=id_map)
    op = ResolvedBundle(trait=subject, dataset_hash="H", params={
        "conf": derived("conf", 0.3, requires_validation=True, validation_kind="annotations",
                        derived_from="x", validated_against="false"),
    }).to_provenance()["operating_point"]
    stamp = operating_point_stamp(
        op, validated=False, validated_by=None, tile_size_validated=None, shippable_issues=[],
        id_map=id_map, trait=subject, dataset_hash="H", checkpoint="m",
        checkpoint_sha256="stub-sha256", experiment_id=None,
        images_dir=str(tmp_path / "images"), raster_path=None,
        produced_at="2024-01-01T00:00:00Z", subject=subject, attribute=attribute,
    )
    write_sidecar(pred_dir, stamp)
    return dataset_root, pred_dir


def test_calibrate_count_operating_point_defaults_to_the_bucket_own_scope_when_neither_key_stated(
    monkeypatch, tmp_path,
):
    """Ordinary detector calibration over a bucket a producer stamped states neither key: the
    bucket's own recorded scope governs the pass rather than refusing over the tool's own
    ``(None, None)`` defaults."""
    import tcip_mcp.model_registry as model_registry_mod

    from tcip_mcp.tools.calibration_tools import calibrate_count_operating_point
    from tests._verified_checkpoint_fixtures import stub_verified_checkpoint

    monkeypatch.setattr(model_registry_mod, "load_registered_checkpoint",
                        lambda path, *a, **kw: stub_verified_checkpoint(str(path)))
    dataset_root, pred_dir = _producer_bucket(tmp_path, subject="bud", attribute=None)

    calls: list[dict] = []

    def _stub(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("stub-count-calibration-called")

    monkeypatch.setattr("tcip_mcp.pipelines.count_calibration.resolve_count_operating_point", _stub)

    with pytest.raises(RuntimeError, match="stub-count-calibration-called"):
        calibrate_count_operating_point(
            checkpoint_path="x.pt", trait="bud_opening", labels_dir=str(tmp_path / "labels"),
            images_dir=str(tmp_path / "images"), dataset_root=str(dataset_root),
            pred_dir=str(pred_dir),
        )

    assert len(calls) == 1
    assert calls[0]["subject"] == "bud"
    assert calls[0]["attribute"] is None


def test_calibrate_count_operating_point_refuses_a_disagreeing_stated_pair(tmp_path):
    """A caller that states a pair disagreeing with the bucket's own recorded scope refuses by
    name, naming both pairs, rather than silently trusting the stated one."""
    from tcip_mcp.tools.calibration_tools import calibrate_count_operating_point

    dataset_root, pred_dir = _producer_bucket(tmp_path, subject="bud", attribute=None)

    result = calibrate_count_operating_point(
        checkpoint_path="x.pt", trait="bud_opening", labels_dir=str(tmp_path / "labels"),
        images_dir=str(tmp_path / "images"), dataset_root=str(dataset_root),
        pred_dir=str(pred_dir), subject="other-subject", attribute=None,
    )

    assert "error" in result
    assert "bud" in result["error"]
    assert "other-subject" in result["error"]


def test_calibrate_count_operating_point_succeeds_when_stated_pair_matches_bucket_scope(
    monkeypatch, tmp_path,
):
    """A caller that states the bucket's own scope explicitly is admitted exactly as one stating
    neither key."""
    import tcip_mcp.model_registry as model_registry_mod

    from tcip_mcp.tools.calibration_tools import calibrate_count_operating_point
    from tests._verified_checkpoint_fixtures import stub_verified_checkpoint

    monkeypatch.setattr(model_registry_mod, "load_registered_checkpoint",
                        lambda path, *a, **kw: stub_verified_checkpoint(str(path)))
    dataset_root, pred_dir = _producer_bucket(tmp_path, subject="bud", attribute=None)

    calls: list[dict] = []

    def _stub(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("stub-count-calibration-called")

    monkeypatch.setattr("tcip_mcp.pipelines.count_calibration.resolve_count_operating_point", _stub)

    with pytest.raises(RuntimeError, match="stub-count-calibration-called"):
        calibrate_count_operating_point(
            checkpoint_path="x.pt", trait="bud_opening", labels_dir=str(tmp_path / "labels"),
            images_dir=str(tmp_path / "images"), dataset_root=str(dataset_root),
            pred_dir=str(pred_dir), subject="bud", attribute=None,
        )

    assert len(calls) == 1
