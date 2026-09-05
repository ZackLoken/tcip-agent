"""run_inference: writes per-image prediction JSON, never overwrites a bucket with verdicts."""

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


def test_run_inference_writes_json(tmp_path, monkeypatch):
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
    from tcip_mcp.tools.inference_tools import run_inference

    out = tmp_path / "out"
    run_inference(str(ckpt), str(images_dir), output_dir=str(out), tile=False)
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


def test_run_inference_forwards_split_manifest_dir_to_the_verified_pass(tmp_path, monkeypatch):
    """A manifest-restricted calibration's evidence can only earn a validation record through
    this door if the door actually forwards split_manifest_dir to the verified pass."""
    import tcip_mcp.tools.inference_tools as itools

    captured = {}

    def _fake_run_inference_verified(*a, **kwargs):
        captured.update(kwargs)
        return {"error": "stop: plumbing check only"}

    monkeypatch.setattr(itools, "_run_inference_verified", _fake_run_inference_verified)

    itools.run_inference(
        _ckpt(tmp_path), images_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
        calibration_labels_dir=str(tmp_path), split_manifest_dir=str(tmp_path / "m"))

    assert captured.get("split_manifest_dir") == str(tmp_path / "m")


def test_run_inference_refuses_split_manifest_dir_with_raster_path(tmp_path):
    """The raster regime draws no split-manifest universe (block calibration validates against
    the mosaic's own reserved regions instead), so a caller-given manifest is refused by name
    rather than silently dropped before it ever reaches the raster pass."""
    from tcip_mcp.tools.inference_tools import run_inference

    result = run_inference(
        _ckpt(tmp_path), output_dir=str(tmp_path / "out"),
        raster_path=str(tmp_path / "mosaic.tif"), split_manifest_dir=str(tmp_path / "m"))

    assert "error" in result and "split_manifest_dir" in result["error"]


def test_deliver_per_image_counts_forwards_split_manifest_dir_to_run_inference(tmp_path, monkeypatch):
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

    itools.deliver_per_image_counts(
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

    from tcip_mcp.tools.inference_tools import run_inference

    out1 = tmp_path / "out1"
    r1 = run_inference(str(ckpt), str(images_dir), output_dir=str(out1), tile=False,
                       experiment_id="expImg")
    assert "error" not in r1, r1
    update_status("expImg", "completed")

    out2 = tmp_path / "out2"
    r2 = run_inference(str(ckpt), str(images_dir), output_dir=str(out2), tile=False,
                       experiment_id="expImg")
    assert "error" in r2
    assert not out2.exists()


def test_a_same_path_image_regime_export_against_a_completed_experiment_refuses_on_documents(
        tmp_path, monkeypatch):
    """The completed experiment's bucket already holds the first run's document: a second run
    into the same path refuses on the document predicate at resolution, before the pointer's own
    same-value conjunct is ever reached (a redirect or a suggestion has nowhere to go either,
    since the pointer refuses any other path for a terminal experiment)."""
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.png")
    _fake_predictor(monkeypatch)

    from tcip_mcp.experiments import create_experiment, update_status
    create_experiment("expImgSame", {"model_source": {"builder": "x:y"}})
    update_status("expImgSame", "running")

    from tcip_mcp.tools.inference_tools import run_inference

    out = tmp_path / "out"
    r1 = run_inference(str(ckpt), str(images_dir), output_dir=str(out), tile=False,
                       experiment_id="expImgSame")
    assert "error" not in r1, r1
    update_status("expImgSame", "completed")

    r2 = run_inference(str(ckpt), str(images_dir), output_dir=str(out), tile=False,
                       experiment_id="expImgSame")
    assert "error" in r2
    assert r2["document_stem_count"] == 1


def test_a_same_path_image_regime_export_against_a_completed_experiment_admits_via_an_empty_first_pass(
        tmp_path, monkeypatch):
    """A rail must admit valid work: a first run whose images_dir enumerates to no image
    publishes a stamp and no document and still records the lineage pointer; once the experiment
    completes, a second run in place over the real images is admitted, by the document predicate
    (the bucket holds no document yet) and by the pointer's own same-value conjunct (the recorded
    path is unchanged)."""
    from pathlib import Path

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar
    from tcip_mcp.prediction_buckets import bucket_stems

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")
    empty_images_dir = tmp_path / "empty_images"
    empty_images_dir.mkdir()
    real_images_dir = tmp_path / "images"
    real_images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(real_images_dir / "img.png")
    _fake_predictor(monkeypatch)

    from tcip_mcp.experiments import create_experiment, get_experiment_lineage, update_status
    create_experiment("expImgEmptyFirst", {"model_source": {"builder": "x:y"}})
    update_status("expImgEmptyFirst", "running")

    from tcip_mcp.tools.inference_tools import run_inference

    out = tmp_path / "out"
    r1 = run_inference(str(ckpt), str(empty_images_dir), output_dir=str(out), tile=False,
                       experiment_id="expImgEmptyFirst")
    assert "error" not in r1, r1
    assert r1["image_count"] == 0
    assert not (out / "img.json").exists()
    # The stamp-only boundary the document predicate rests on: a stamp with no document.
    assert read_operating_point_sidecar(out) is not None
    assert bucket_stems(out) == set()
    assert get_experiment_lineage("expImgEmptyFirst")["lineage"]["predictions"] == str(out)
    update_status("expImgEmptyFirst", "completed")

    r2 = run_inference(str(ckpt), str(real_images_dir), output_dir=str(out), tile=False,
                       experiment_id="expImgEmptyFirst")
    assert "error" not in r2, r2
    assert Path(r2["output_dir"]) == out
    assert (out / "img.json").is_file()


def test_a_run_over_an_empty_images_directory_admits_a_second_run_in_place(tmp_path, monkeypatch):
    """A run whose images_dir enumerates to no image publishes a stamp and no document, so a
    second, real run into the same output_dir is admitted: the document predicate's boundary is
    holds documents, not was published before."""
    from pathlib import Path

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar
    from tcip_mcp.prediction_buckets import bucket_stems

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")
    empty_images_dir = tmp_path / "empty_images"
    empty_images_dir.mkdir()
    real_images_dir = tmp_path / "images"
    real_images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(real_images_dir / "img.png")
    _fake_predictor(monkeypatch)

    from tcip_mcp.tools.inference_tools import run_inference

    out = tmp_path / "out"
    r1 = run_inference(str(ckpt), str(empty_images_dir), output_dir=str(out), tile=False)
    assert "error" not in r1, r1
    assert r1["image_count"] == 0
    assert not (out / "img.json").exists()
    # The stamp-only boundary the document predicate rests on: a stamp with no document.
    assert read_operating_point_sidecar(out) is not None
    assert bucket_stems(out) == set()

    r2 = run_inference(str(ckpt), str(real_images_dir), output_dir=str(out), tile=False)
    assert "error" not in r2, r2
    assert Path(r2["output_dir"]) == out
    assert (out / "img.json").is_file()


def test_run_inference_redirects_a_bespoke_bucket_against_its_own_datasets_verdicts(
    tmp_path, monkeypatch,
):
    """A bucket that is not the canonical predictions/<model>/<date> shape but still sits inside a
    dataset is guarded against that dataset's verdict store, and redirects by its last segment."""
    from pathlib import Path

    platform_root = tmp_path / "platform"
    (platform_root / ".tcip" / "state").mkdir(parents=True)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(platform_root))

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
                        preds=[Annotation(subject="bud", geometry=BBox(10.0, 10.0, 30.0, 30.0),
                                          score=0.9)])
    det = ReviewDetection(det_type="fp", class_name="bud", conf=0.9, iou=None, gt_idx=None,
                          pred_idx=0, bbox=(10.0, 10.0, 30.0, 30.0))
    engine.record_detection_action(bucket_key_of(out), det, ctx, action="accepted")

    _fake_predictor(monkeypatch)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")
    from tcip_mcp.tools.inference_tools import run_inference

    # overwrite=True is refused with the verdict count and a suggested fresh bucket.
    res2 = run_inference(str(ckpt), str(images_dir), output_dir=str(out), overwrite=True, tile=False)
    assert "error" in res2 and res2["verdict_count"] == 1
    assert Path(res2["suggested_bucket"]).name == "preds@r2"

    # Default: redirect to a fresh @r2 bucket; the reviewed bucket is left intact.
    res = run_inference(str(ckpt), str(images_dir), output_dir=str(out), tile=False)
    assert res["bucket_redirected"] is True
    assert res["verdict_guard_operative"] is True
    assert Path(res["output_dir"]).name == "preds@r2"
    assert (Path(res["output_dir"]) / "img.json").is_file()
    assert json.loads((out / "img.json").read_text())["annotations"] == []  # untouched


def test_run_inference_writes_a_bucket_under_no_dataset_root_and_says_the_guard_is_off(
    tmp_path, monkeypatch,
):
    """A bucket outside any dataset has no verdict store to be guarded against, so the export is
    written where it was asked for and the response says the guarantee is absent, naming the
    layout that carries it. Refusing the write would reject legitimate exploratory work."""
    from pathlib import Path

    platform_root = tmp_path / "platform"
    (platform_root / ".tcip" / "state").mkdir(parents=True)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(platform_root))

    images_dir = tmp_path / "captures"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.png")

    _fake_predictor(monkeypatch)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")
    from tcip_mcp.tools.inference_tools import run_inference

    out = tmp_path / "scratch_preds"
    res = run_inference(str(ckpt), str(images_dir), output_dir=str(out), tile=False)

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
    monkeypatch.setenv("TCIP_STATE_ROOT", str(platform_root))

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
                        preds=[Annotation(subject="bud", geometry=BBox(10.0, 10.0, 30.0, 30.0),
                                          score=0.9)])
    det = ReviewDetection(det_type="fp", class_name="bud", conf=0.9, iou=None, gt_idx=None,
                          pred_idx=0, bbox=(10.0, 10.0, 30.0, 30.0))
    engine.record_detection_action(bucket_key_of(out), det, ctx, action="accepted")

    _fake_predictor(monkeypatch)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")
    return dataset_root, images_dir, out, ckpt


def test_run_inference_redirect_varies_the_model_segment_for_a_canonical_bucket(
    tmp_path, monkeypatch
) -> None:
    """A caller-assembled predictions/<model>/<date> output_dir must redirect the same way the
    platform's other writers do, or the redirected bucket is invisible to every date-keyed reader."""
    from pathlib import Path

    from tcip_mcp.dataset_layout import prediction_dir

    dataset_root, images_dir, out, ckpt = _canonical_bucket_with_a_verdict(tmp_path, monkeypatch)
    from tcip_mcp.tools.inference_tools import run_inference

    res = run_inference(str(ckpt), str(images_dir), output_dir=str(out), tile=False)
    assert res["bucket_redirected"] is True
    redirected = Path(res["output_dir"])
    # The model segment moved, not the date: still findable under the same date, a different model name.
    assert redirected == prediction_dir(dataset_root, "baseline@r2", "2026-01-01")
    assert redirected.name == "2026-01-01"
    assert redirected.parent.name == "baseline@r2"
    assert (redirected / "img.json").is_file()


def test_run_inference_counts_a_canonical_buckets_verdicts_in_its_own_datasets_store(
    tmp_path, monkeypatch
) -> None:
    """The verdicts that freeze a dataset's prediction bucket are the ones recorded in that
    dataset's verdict store, the store stage_prediction_shapes and the review engine both use. A
    guard reading a different root counts none of them and overwrites reviewed predictions."""
    from pathlib import Path

    _dataset_root, images_dir, out, ckpt = _canonical_bucket_with_a_verdict(tmp_path, monkeypatch)
    from tcip_mcp.tools.inference_tools import run_inference

    refused = run_inference(str(ckpt), str(images_dir), output_dir=str(out), overwrite=True, tile=False)
    assert "error" in refused and refused["verdict_count"] == 1
    assert Path(refused["suggested_bucket"]).parent.name == "baseline@r2"
    assert json.loads((out / "img.json").read_text())["annotations"] == []


def test_overwrite_true_into_an_existing_empty_bucket_is_the_ordinary_write(
    tmp_path, monkeypatch
) -> None:
    """Coverage, not a distinct proof of overwrite=True's own effect: an empty, pre-created
    bucket holds neither a verdict nor a document, so overwrite=True writes there exactly the
    way any run does into a fresh directory, in place and never redirected."""
    from pathlib import Path

    from tcip_mcp.dataset_layout import prediction_dir

    platform_root = tmp_path / "platform"
    (platform_root / ".tcip" / "state").mkdir(parents=True)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(platform_root))

    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "images" / "2026-01-01"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.png")
    out = prediction_dir(dataset_root, "baseline", "2026-01-01")
    out.mkdir(parents=True)

    _fake_predictor(monkeypatch)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")
    from tcip_mcp.tools.inference_tools import run_inference

    res = run_inference(str(ckpt), str(images_dir), output_dir=str(out), overwrite=True, tile=False)
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


def test_dry_run_previews_the_both_sources_refusal(tmp_path):
    """A preview previews the same refusal a real call would hit: dry_run does not skip the
    mutual-exclusion check just because it loads no model."""
    from tcip_mcp.tools.inference_tools import run_inference

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    raster_path = tmp_path / "mosaic.tif"
    raster_path.write_bytes(b"stub")

    result = run_inference(
        _ckpt(tmp_path), images_dir=str(images_dir), raster_path=str(raster_path),
        output_dir=str(tmp_path / "out"), dry_run=True)

    assert "error" in result and "not both" in result["error"]


def test_dry_run_names_the_bucket_it_would_write_to_and_writes_nothing(tmp_path):
    """dry_run needs neither images_dir nor raster_path: it previews the bucket a real write would
    resolve to and the operating point it would run at, without touching disk."""
    from tcip_mcp.tools.inference_tools import run_inference

    out = tmp_path / "out"

    result = run_inference(_ckpt(tmp_path), output_dir=str(out), dry_run=True)

    assert "error" not in result, result
    assert result["output_dir"] == str(out)
    assert result["bucket_redirected"] is False
    assert not out.exists()


def test_resume_and_overwrite_together_refuse_by_name(tmp_path):
    """The two name opposite ways of handling a bucket's recorded progress; letting one silently
    win over the other would discard progress a caller asked to keep, or vice versa."""
    from tcip_mcp.tools.inference_tools import run_inference

    raster_path = tmp_path / "mosaic.tif"
    raster_path.write_bytes(b"stub")

    result = run_inference(
        _ckpt(tmp_path), raster_path=str(raster_path), output_dir=str(tmp_path / "out"),
        resume=True, overwrite=True)

    assert "error" in result
    assert "resume=True" in result["error"] and "overwrite=True" in result["error"]


# ── prediction-document immutability ────────────────────────────────────────


def test_run_inference_refuses_a_second_publish_into_a_document_holding_bucket(
    tmp_path, monkeypatch,
):
    """Two runs into one output_dir under a dataset root, with no experiment: the second refuses
    naming the document count and the suggested @r2 path, before the checkpoint is read and
    before any pass runs, leaving the first bucket's own documents and stamp unchanged (digest
    and stamp equality prove those two artifacts alone). The suggested bucket, once written
    into, admits a real re-run."""
    from pathlib import Path

    import tcip_mcp.model_registry as model_registry_mod
    from tcip_mcp.dataset_layout import prediction_dir
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar
    from tcip_mcp.prediction_buckets import bucket_content_digest
    from tcip_mcp.tools.inference_tools import run_inference

    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "images" / "2026-01-01"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.png")
    out = prediction_dir(dataset_root, "baseline", "2026-01-01")

    checkpoint_calls = {"n": 0}
    real_stub = model_registry_mod.load_registered_checkpoint

    def _counting_stub(path, *a, **kw):
        checkpoint_calls["n"] += 1
        return real_stub(path, *a, **kw)

    monkeypatch.setattr(model_registry_mod, "load_registered_checkpoint", _counting_stub)

    predict_calls = {"n": 0}

    class FakePredictor:
        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, **kw):
            predict_calls["n"] += 1
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1], "count": 1}
                    for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakePredictor)

    ckpt = _ckpt(tmp_path)
    r1 = run_inference(ckpt, str(images_dir), output_dir=str(out), tile=False)
    assert "error" not in r1, r1
    assert checkpoint_calls["n"] == 1
    assert predict_calls["n"] == 1

    digest_before = bucket_content_digest(out)
    stamp_before = read_operating_point_sidecar(out)

    r2 = run_inference(ckpt, str(images_dir), output_dir=str(out), tile=False)
    assert "error" in r2
    assert r2["document_stem_count"] == 1
    assert Path(r2["suggested_bucket"]).parent.name == "baseline@r2"
    assert not Path(r2["suggested_bucket"]).exists()
    assert checkpoint_calls["n"] == 1  # unreached on the refused call
    assert predict_calls["n"] == 1

    assert bucket_content_digest(out) == digest_before
    assert read_operating_point_sidecar(out) == stamp_before

    # The admitting case: the suggested bucket is free of both a verdict and a document.
    r3 = run_inference(ckpt, str(images_dir), output_dir=str(r2["suggested_bucket"]), tile=False)
    assert "error" not in r3, r3
    assert Path(r3["output_dir"]) == Path(r2["suggested_bucket"])


def test_run_inference_no_dataset_root_pair_refuses_through_the_shared_resolver(
    tmp_path, monkeypatch,
):
    """The no-dataset-root branch resolves through the same resolve_writable_bucket every other
    branch does, not a second, hand-built resolution: a spy on it proves it is reached, and the
    second run over the same bucket refuses on the document the first run left."""
    import tcip_mcp.prediction_buckets as buckets_mod
    from tcip_mcp.tools.inference_tools import run_inference

    platform_root = tmp_path / "platform"
    (platform_root / ".tcip" / "state").mkdir(parents=True)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(platform_root))

    images_dir = tmp_path / "captures"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.png")
    _fake_predictor(monkeypatch)
    ckpt = _ckpt(tmp_path)

    out = tmp_path / "scratch_preds"
    r1 = run_inference(ckpt, str(images_dir), output_dir=str(out), tile=False)
    assert "error" not in r1, r1
    assert r1["verdict_guard_operative"] is False

    calls = {"n": 0}
    real_resolve = buckets_mod.resolve_writable_bucket

    def _spy_resolve(*a, **kw):
        calls["n"] += 1
        return real_resolve(*a, **kw)

    monkeypatch.setattr(buckets_mod, "resolve_writable_bucket", _spy_resolve)

    r2 = run_inference(ckpt, str(images_dir), output_dir=str(out), tile=False)
    assert "error" in r2
    assert r2["document_stem_count"] == 1
    assert calls["n"] == 1


def test_dry_run_previews_the_document_refusal(tmp_path, monkeypatch):
    """A preview previews the same document refusal a real call would hit, never a bucket a real
    call would in fact refuse to write into."""
    from tcip_mcp.dataset_layout import prediction_dir
    from tcip_mcp.tools.inference_tools import run_inference

    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "images" / "2026-01-01"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.png")
    out = prediction_dir(dataset_root, "baseline", "2026-01-01")
    _fake_predictor(monkeypatch)
    ckpt = _ckpt(tmp_path)

    r1 = run_inference(ckpt, str(images_dir), output_dir=str(out), tile=False)
    assert "error" not in r1, r1

    preview = run_inference(ckpt, output_dir=str(out), dry_run=True)
    assert "error" in preview
    assert preview["document_stem_count"] == 1


def test_overwrite_true_still_refuses_a_document_holding_bucket_with_no_verdicts(
    tmp_path, monkeypatch,
):
    """overwrite=True never rescues the document refusal: it only ever changes what happens on a
    verdict, never on a document with none."""
    from tcip_mcp.dataset_layout import prediction_dir
    from tcip_mcp.tools.inference_tools import run_inference

    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "images" / "2026-01-01"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.png")
    out = prediction_dir(dataset_root, "baseline", "2026-01-01")
    _fake_predictor(monkeypatch)
    ckpt = _ckpt(tmp_path)

    r1 = run_inference(ckpt, str(images_dir), output_dir=str(out), tile=False)
    assert "error" not in r1, r1

    r2 = run_inference(ckpt, str(images_dir), output_dir=str(out), tile=False, overwrite=True)
    assert "error" in r2
    assert r2["document_stem_count"] == 1
