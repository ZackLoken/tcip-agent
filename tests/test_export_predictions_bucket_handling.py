"""export_predictions: writes per-image prediction JSON, never overwrites a bucket with verdicts."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402


@pytest.fixture(autouse=True)
def _stub_checkpoint_verification(monkeypatch):
    """Every test in this module drives a stubbed predictor, not a real registered checkpoint;
    load_registered_checkpoint is stubbed to admit whatever path it is given, carrying the real
    file's own digest when one exists (some assertions here check that hash) and a fixed stand-in
    otherwise.
    """
    from pathlib import Path

    import tcip_mcp.model_registry as model_registry_mod

    from tests._verified_checkpoint_fixtures import stub_verified_checkpoint

    def _stub(path, *a, **kw):
        p = Path(path)
        sha = model_registry_mod._sha256_of_bytes(p.read_bytes()) if p.is_file() else "stub-sha256"
        return stub_verified_checkpoint(str(path), sha256=sha)

    monkeypatch.setattr(model_registry_mod, "load_registered_checkpoint", _stub)


def _ckpt(tmp_path, name: str = "ckpt.pt") -> str:
    """A checkpoint path that exists on disk, for the not-found check every door now runs first."""
    p = tmp_path / name
    if not p.exists():
        p.write_bytes(b"stub")
    return str(p)


def test_export_predictions_writes_json(tmp_path, monkeypatch):
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")  # only existence is checked
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.png")

    class FakePredictor:
        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, **kw):
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1], "count": 1}
                    for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakePredictor)
    from tcip_mcp.tools.inference_tools import export_predictions

    out = tmp_path / "out"
    export_predictions(str(ckpt), str(images_dir), str(out), tile=False)
    data = json.loads((out / "img.json").read_text())
    assert data["image"] == "img"
    assert (data["width"], data["height"]) == (100, 100)
    anns = data["annotations"]
    assert len(anns) == 1
    assert anns[0]["subject"] == "0"                   # label 1 -> id 0; no run id_map -> stringified id
    assert anns[0]["score"] == pytest.approx(0.9)      # confidence
    # COCO xywh (pixel) from pixel-xyxy box [10,10,30,30].
    assert anns[0]["bbox"] == pytest.approx([10.0, 10.0, 20.0, 20.0])

    import hashlib

    assert anns[0]["created_by"] == f"model:m@{hashlib.sha256(ckpt.read_bytes()).hexdigest()[:12]}"


def test_export_predictions_forwards_split_manifest_dir_to_run_inference(tmp_path, monkeypatch):
    """A manifest-restricted calibration's evidence can only earn a validation record through
    this door if the door actually forwards split_manifest_dir to run_inference."""
    import tcip_mcp.tools.inference_tools as itools

    captured = {}

    def _fake_run_inference_verified(*a, **kwargs):
        captured.update(kwargs)
        return {"error": "stop: plumbing check only"}

    monkeypatch.setattr(itools, "_run_inference_verified", _fake_run_inference_verified)

    itools.export_predictions(
        _ckpt(tmp_path), images_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
        split_manifest_dir=str(tmp_path / "m"))

    assert captured.get("split_manifest_dir") == str(tmp_path / "m")


def test_export_predictions_refuses_split_manifest_dir_with_raster_path(tmp_path):
    """The raster regime draws no split-manifest universe (block calibration validates against
    the mosaic's own reserved regions instead), so a caller-given manifest is refused by name
    rather than silently dropped before it ever reaches the raster pass."""
    from tcip_mcp.tools.inference_tools import export_predictions

    result = export_predictions(
        _ckpt(tmp_path), output_dir=str(tmp_path / "out"),
        raster_path=str(tmp_path / "mosaic.tif"), split_manifest_dir=str(tmp_path / "m"))

    assert "error" in result and "split_manifest_dir" in result["error"]


def test_tabulate_counts_forwards_split_manifest_dir_to_run_inference(tmp_path, monkeypatch):
    """A manifest-restricted calibration's evidence can only earn a validation record through
    this door if the door actually forwards split_manifest_dir to run_inference."""
    import tcip_mcp.tools.inference_tools as itools

    captured = {}

    def _fake_run_inference_verified(*a, **kwargs):
        captured.update(kwargs)
        return {"error": "stop: plumbing check only"}

    monkeypatch.setattr(itools, "_run_inference_verified", _fake_run_inference_verified)
    stated = type("Stated", (), {"ok": True, "message": ""})()
    monkeypatch.setattr(
        "tcip_mcp.operationalization.resolve_trait_and_record",
        lambda trait, kind: (object(), object(), tmp_path))
    monkeypatch.setattr(
        "tcip_mcp.operationalization.check_operationalization",
        lambda spec, record, kind, registry=None: stated)

    itools.tabulate_counts(
        _ckpt(tmp_path), str(tmp_path), str(tmp_path / "out.csv"), trait="some_trait",
        split_manifest_dir=str(tmp_path / "m"))

    assert captured.get("split_manifest_dir") == str(tmp_path / "m")


def _fake_predictor(monkeypatch):
    class FakePredictor:
        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, **kw):
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1], "count": 1}
                    for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakePredictor)


def test_a_second_image_regime_export_against_a_completed_experiment_refuses_before_writing(
        tmp_path, monkeypatch):
    """A pointer is checked before its write: a second images_dir export against an experiment
    whose lineage.predictions is already populated and terminal refuses by name, before the
    publisher writes a second bucket."""
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.png")
    _fake_predictor(monkeypatch)

    from tcip_mcp.experiments import create_experiment, update_status
    create_experiment("expImg", {"model_source": {"builder": "x:y"}})
    update_status("expImg", "running")

    from tcip_mcp.tools.inference_tools import export_predictions

    out1 = tmp_path / "out1"
    r1 = export_predictions(str(ckpt), str(images_dir), str(out1), tile=False,
                            experiment_id="expImg")
    assert "error" not in r1, r1
    update_status("expImg", "completed")

    out2 = tmp_path / "out2"
    r2 = export_predictions(str(ckpt), str(images_dir), str(out2), tile=False,
                            experiment_id="expImg")
    assert "error" in r2
    assert not out2.exists()


def test_a_same_path_image_regime_export_against_a_completed_experiment_admits_the_second_run(
        tmp_path, monkeypatch):
    """A rail must admit valid work: re-exporting into the same bucket path records the same
    lineage value the completed experiment already holds, so the additive lock's same-value
    conjunct admits it rather than refusing a legitimate re-run."""
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.png")
    _fake_predictor(monkeypatch)

    from tcip_mcp.experiments import create_experiment, update_status
    create_experiment("expImgSame", {"model_source": {"builder": "x:y"}})
    update_status("expImgSame", "running")

    from tcip_mcp.tools.inference_tools import export_predictions

    out = tmp_path / "out"
    r1 = export_predictions(str(ckpt), str(images_dir), str(out), tile=False,
                            experiment_id="expImgSame")
    assert "error" not in r1, r1
    update_status("expImgSame", "completed")

    r2 = export_predictions(str(ckpt), str(images_dir), str(out), tile=False,
                            experiment_id="expImgSame")
    assert "error" not in r2, r2
    assert (out / "img.json").is_file()


def test_export_predictions_redirects_a_bespoke_bucket_against_its_own_datasets_verdicts(
    tmp_path, monkeypatch,
):
    """A bucket that is not the canonical predictions/<model>/<date> shape but still sits inside a
    dataset is guarded against that dataset's verdict store, and redirects by its last segment."""
    from pathlib import Path

    platform_root = tmp_path / "platform"
    (platform_root / ".tcip" / "state").mkdir(parents=True)
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(platform_root))

    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "images" / "2026-01-01"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.png")

    out = dataset_root / "predictions" / "preds"
    out.mkdir(parents=True)
    # A prediction file already sits in the bucket, and a human verdict is recorded against it.
    (out / "img.json").write_text(
        json.dumps({"image": "img", "width": 100, "height": 100, "annotations": []}))
    from tcip_annotation.review_engine import ReviewContext, ReviewDetection, ReviewEngine
    from tcip_annotation.state import Annotation, BBox

    from tcip_mcp.prediction_buckets import bucket_key_of

    engine = ReviewEngine(dataset_root / ".tcip" / "state")
    ctx = ReviewContext(img_name="img.png", img_width=100, img_height=100,
                        preds=[Annotation(subject="catkin", geometry=BBox(10.0, 10.0, 30.0, 30.0),
                                          score=0.9)])
    det = ReviewDetection(det_type="fp", class_name="catkin", conf=0.9, iou=None, gt_idx=None,
                          pred_idx=0, bbox=(10.0, 10.0, 30.0, 30.0))
    engine.record_detection_action(bucket_key_of(out), det, ctx, action="accepted")

    _fake_predictor(monkeypatch)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")
    from tcip_mcp.tools.inference_tools import export_predictions

    # overwrite=True is refused with the verdict count and a suggested fresh bucket.
    res2 = export_predictions(str(ckpt), str(images_dir), str(out), overwrite=True, tile=False)
    assert "error" in res2 and res2["verdict_count"] == 1
    assert Path(res2["suggested_bucket"]).name == "preds@r2"

    # Default: redirect to a fresh @r2 bucket; the reviewed bucket is left intact.
    res = export_predictions(str(ckpt), str(images_dir), str(out), tile=False)
    assert res["bucket_redirected"] is True
    assert res["verdict_guard_operative"] is True
    assert Path(res["output_dir"]).name == "preds@r2"
    assert (Path(res["output_dir"]) / "img.json").is_file()
    assert json.loads((out / "img.json").read_text())["annotations"] == []  # untouched


def test_export_predictions_writes_a_bucket_under_no_dataset_root_and_says_the_guard_is_off(
    tmp_path, monkeypatch,
):
    """A bucket outside any dataset has no verdict store to be guarded against, so the export is
    written where it was asked for and the response says the guarantee is absent, naming the
    layout that carries it. Refusing the write would reject legitimate exploratory work."""
    from pathlib import Path

    platform_root = tmp_path / "platform"
    (platform_root / ".tcip" / "state").mkdir(parents=True)
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(platform_root))

    images_dir = tmp_path / "captures"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.png")

    _fake_predictor(monkeypatch)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")
    from tcip_mcp.tools.inference_tools import export_predictions

    out = tmp_path / "scratch_preds"
    res = export_predictions(str(ckpt), str(images_dir), str(out), tile=False)

    assert "error" not in res, res
    assert Path(res["output_dir"]) == out
    assert res["bucket_redirected"] is False
    assert res["verdict_guard_operative"] is False
    assert "predictions" in res["note"]  # the canonical layout is named as the path to the guard
    assert (out / "img.json").is_file()


def _canonical_bucket_with_a_verdict(tmp_path, monkeypatch) -> tuple:
    """A dataset holding one image, one canonical predictions/<model>/<date> bucket, and one
    review verdict recorded in that dataset's own verdict store.

    The platform root is pinned somewhere else entirely and left empty, so a guard that counted
    verdicts there instead of in the dataset's store would see none.
    """
    from tcip_annotation.review_engine import ReviewContext, ReviewDetection, ReviewEngine
    from tcip_annotation.state import Annotation, BBox

    from tcip_mcp.dataset_layout import prediction_dir
    from tcip_mcp.prediction_buckets import bucket_key_of

    platform_root = tmp_path / "platform"
    (platform_root / ".tcip" / "state").mkdir(parents=True)
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(platform_root))

    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "images" / "2026-01-01"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.png")

    out = prediction_dir(dataset_root, "baseline", "2026-01-01")
    out.mkdir(parents=True)
    (out / "img.json").write_text(
        json.dumps({"image": "img", "width": 100, "height": 100, "annotations": []}))

    engine = ReviewEngine(dataset_root / ".tcip" / "state")
    ctx = ReviewContext(img_name="img.png", img_width=100, img_height=100,
                        preds=[Annotation(subject="catkin", geometry=BBox(10.0, 10.0, 30.0, 30.0),
                                          score=0.9)])
    det = ReviewDetection(det_type="fp", class_name="catkin", conf=0.9, iou=None, gt_idx=None,
                          pred_idx=0, bbox=(10.0, 10.0, 30.0, 30.0))
    engine.record_detection_action(bucket_key_of(out), det, ctx, action="accepted")

    _fake_predictor(monkeypatch)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")
    return dataset_root, images_dir, out, ckpt


def test_export_predictions_redirect_varies_the_model_segment_for_a_canonical_bucket(
    tmp_path, monkeypatch
) -> None:
    """A caller-assembled predictions/<model>/<date> output_dir must redirect the same way the
    platform's other writers do, or the redirected bucket is invisible to every date-keyed reader."""
    from pathlib import Path

    from tcip_mcp.dataset_layout import prediction_dir

    dataset_root, images_dir, out, ckpt = _canonical_bucket_with_a_verdict(tmp_path, monkeypatch)
    from tcip_mcp.tools.inference_tools import export_predictions

    res = export_predictions(str(ckpt), str(images_dir), str(out), tile=False)
    assert res["bucket_redirected"] is True
    redirected = Path(res["output_dir"])
    # The model segment moved, not the date: still findable under the same date, a different model name.
    assert redirected == prediction_dir(dataset_root, "baseline@r2", "2026-01-01")
    assert redirected.name == "2026-01-01"
    assert redirected.parent.name == "baseline@r2"
    assert (redirected / "img.json").is_file()


def test_export_predictions_counts_a_canonical_buckets_verdicts_in_its_own_datasets_store(
    tmp_path, monkeypatch
) -> None:
    """The verdicts that freeze a dataset's prediction bucket are the ones recorded in that
    dataset's verdict store, the store stage_prediction_shapes and the review engine both use. A
    guard reading a different root counts none of them and overwrites reviewed predictions."""
    from pathlib import Path

    _dataset_root, images_dir, out, ckpt = _canonical_bucket_with_a_verdict(tmp_path, monkeypatch)
    from tcip_mcp.tools.inference_tools import export_predictions

    refused = export_predictions(str(ckpt), str(images_dir), str(out), overwrite=True, tile=False)
    assert "error" in refused and refused["verdict_count"] == 1
    assert Path(refused["suggested_bucket"]).parent.name == "baseline@r2"
    assert json.loads((out / "img.json").read_text())["annotations"] == []


def test_export_predictions_writes_a_canonical_bucket_with_no_verdicts_in_place(
    tmp_path, monkeypatch
) -> None:
    """The same dataset-scoped guard must still admit the ordinary re-run: an unreviewed bucket is
    written where it was asked for, overwrite and all, never redirected out from under its reader."""
    from pathlib import Path

    from tcip_mcp.dataset_layout import prediction_dir

    platform_root = tmp_path / "platform"
    (platform_root / ".tcip" / "state").mkdir(parents=True)
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(platform_root))

    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "images" / "2026-01-01"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.png")
    out = prediction_dir(dataset_root, "baseline", "2026-01-01")
    out.mkdir(parents=True)

    _fake_predictor(monkeypatch)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")
    from tcip_mcp.tools.inference_tools import export_predictions

    res = export_predictions(str(ckpt), str(images_dir), str(out), overwrite=True, tile=False)
    assert "error" not in res
    assert res["bucket_redirected"] is False
    assert Path(res["output_dir"]) == out
    assert (out / "img.json").is_file()


def test_a_producer_string_is_refused_without_the_checkpoints_hash():
    """A prediction's producer is spelled only from a resolved checkpoint identity: a missing
    hash is refused by name rather than stamped as a hash-less producer, and a present hash
    yields the one spelling every checkpoint-backed door writes."""
    from tcip_mcp.pipelines.resolution import prediction_producer

    with pytest.raises(ValueError, match="no checkpoint hash for .*m.pt"):
        prediction_producer("m.pt", None)  # type: ignore[arg-type]
    assert prediction_producer("m.pt", "725c546b990dabcdef") == "model:m@725c546b990d"
