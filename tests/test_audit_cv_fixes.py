"""Audit CV-rigor fixes: val-loss over negatives (CV9), standard+operating mAP (CV10),
tiled evaluation (CV1), inference tile geometry (CV2), calibrated inference (CV0).

Kept in one file so the audit's measurement-integrity locks live together. Vision/state
tests here are pure-Python or monkeypatched (no GPU) so they stay xdist-safe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from tcip_mcp.pipelines.training.evaluation import (  # noqa: E402
    build_coco_image_record,
    coco_detection_metrics,
    evaluate,
)


# ======================================================================
# CV9 — detection val-loss must include all-negative images
# ======================================================================

class _StubDetector:
    """Minimal detector: records every loss-pass call so we can assert negatives are forwarded."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def train(self) -> None:  # noqa: D401
        pass

    def eval(self) -> None:  # noqa: D401
        pass

    def modules(self):
        return []

    def __call__(self, images, targets):
        self.calls.append((len(images), sum(int(t["boxes"].numel()) for t in targets)))
        return {"loss": torch.tensor(2.5)}


class _StubModel:
    """Composed-model stand-in exposing ``.detector`` and an empty-prediction forward."""

    def __init__(self, detector: _StubDetector) -> None:
        self.detector = detector

    def eval(self) -> None:  # noqa: D401
        pass

    def __call__(self, images):
        return [
            {"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)),
             "labels": torch.zeros((0,), dtype=torch.long)}
            for _ in images
        ]


def _det_batch(box_counts: list[int]):
    """One detection batch: image tensors + targets, some with empty (negative) boxes."""
    images, targets = [], []
    for n in box_counts:
        images.append(torch.zeros(3, 32, 32))
        boxes = torch.tensor([[1.0, 1.0, 5.0, 5.0]] * n, dtype=torch.float32).reshape(-1, 4)
        targets.append({"boxes": boxes, "labels": torch.ones((n,), dtype=torch.long)})
    return images, targets


def test_cv9_val_loss_forwards_all_negative_images():
    stub = _StubDetector()
    model = _StubModel(stub)
    loader = [_det_batch([1, 0]), _det_batch([0])]  # mixed batch, then all-negative batch
    evaluate(model, loader, torch.device("cpu"), "detection")
    # Both batches forwarded through the detector (full batch incl. negatives), not just foreground.
    assert stub.calls == [(2, 4), (1, 0)]


def test_cv9_all_negative_only_loader_is_not_skipped():
    stub = _StubDetector()
    model = _StubModel(stub)
    loader = [_det_batch([0, 0])]  # nothing but negatives
    result = evaluate(model, loader, torch.device("cpu"), "detection")
    assert stub.calls == [(2, 0)]  # forwarded, not skipped
    assert result["loss"] == pytest.approx(2.5)  # finite, non-zero — negatives contribute loss


# ======================================================================
# CV10 — report standard map@100 AND map@max_dets
# ======================================================================

def _rec(gt, dt, w=100, h=100):
    return build_coco_image_record(w, h, gt, dt)


def _perfect_records(n=1):
    gt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}]
    dt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}]
    return [_rec(gt, dt) for _ in range(n)]


def test_cv10_map_nonzero_at_high_maxdets():
    m = coco_detection_metrics(_perfect_records(), max_dets=1000)
    assert m["map"] == pytest.approx(1.0)      # standard map@100 no longer collapses to 0.0
    assert m["map50"] == pytest.approx(1.0)
    assert m["map_at_maxdets"] == pytest.approx(1.0)
    assert m["map50_at_maxdets"] == pytest.approx(1.0)


def test_cv10_map_invariant_across_maxdets():
    recs = _perfect_records(3)
    maps = {md: coco_detection_metrics(recs, max_dets=md) for md in (100, 300, 1000)}
    base = maps[100]
    for md in (300, 1000):
        assert maps[md]["map"] == pytest.approx(base["map"])
        assert maps[md]["map50"] == pytest.approx(base["map50"])
        assert maps[md]["map75"] == pytest.approx(base["map75"])


def test_cv10_standard_keys_unchanged_at_100():
    m = coco_detection_metrics(_perfect_records(), max_dets=100)
    assert m["map"] == pytest.approx(1.0)
    assert m["map50"] == pytest.approx(1.0)
    # at the standard cap the operating-cap figures equal the standard ones
    assert m["map_at_maxdets"] == pytest.approx(m["map"])


# ======================================================================
# CV1 — tiled evaluation regimes
# ======================================================================

def _det_dataset(tmp_path, n=3, size=128):
    from PIL import Image

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (size, size), color=(120, 120, 120)).save(images_dir / f"img{i}.png")
        json_io.write_annotations(str(labels_dir / f"img{i}.json"),
                                  [Annotation(subject="catkin", geometry=BBox(10, 10, 40, 40))], size, size)
    return images_dir, labels_dir


def _capture_run_test_evaluation(monkeypatch):
    """Patch run_test_evaluation to record the built dataset + tiling instead of loading a model."""
    import tcip_mcp.pipelines.training.evaluation as evaluation

    captured: dict = {}

    def _fake(ckpt, loader, device, task, output_dir, **kw):
        captured["ds"] = loader.dataset
        captured["tiling"] = kw.get("tiling")
        return {"tiled": bool(kw.get("tiling")), "eval_regime": "tile-level"}

    monkeypatch.setattr(evaluation, "run_test_evaluation", _fake)
    return captured


def test_cv1_run_id_reuses_training_tiling(tmp_path, monkeypatch):
    from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset
    from tcip_mcp.pipelines.training.generic_trainer import create_run
    from tcip_mcp.tools.training_tools import evaluate_model

    images_dir, labels_dir = _det_dataset(tmp_path)
    run = create_run({"data": {"tiling": {"enabled": True, "tile_size": 64}}}, str(tmp_path / "out"))
    out = Path(run.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_best.pt").write_bytes(b"x")

    captured = _capture_run_test_evaluation(monkeypatch)
    evaluate_model(run.run_id, str(images_dir), str(labels_dir), task="detection", subject="catkin")
    assert isinstance(captured["ds"], TiledDetectionDataset)
    assert captured["ds"].num_samples > 3  # more tiles than the 3 source images
    assert captured["tiling"] == {"enabled": True, "tile_size": 64}


def test_cv1_explicit_checkpoint_stays_untiled(tmp_path, monkeypatch):
    from tcip_mcp.pipelines.data.datasets import DetectionDataset, TiledDetectionDataset
    from tcip_mcp.tools.training_tools import evaluate_model

    images_dir, labels_dir = _det_dataset(tmp_path)
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"x")

    captured = _capture_run_test_evaluation(monkeypatch)
    evaluate_model(str(ckpt), str(images_dir), str(labels_dir), task="detection", subject="catkin")
    assert isinstance(captured["ds"], DetectionDataset)
    assert not isinstance(captured["ds"], TiledDetectionDataset)
    assert captured["tiling"] is None


def test_cv1_explicit_tiling_override_on_checkpoint(tmp_path, monkeypatch):
    from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset
    from tcip_mcp.tools.training_tools import evaluate_model

    images_dir, labels_dir = _det_dataset(tmp_path)
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"x")

    captured = _capture_run_test_evaluation(monkeypatch)
    evaluate_model(str(ckpt), str(images_dir), str(labels_dir), task="detection", subject="catkin",
                   tiling={"enabled": True, "tile_size": 64})
    assert isinstance(captured["ds"], TiledDetectionDataset)


def test_cv1_full_frame_counts_straddling_object_once(tmp_path, monkeypatch):
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tcip_mcp.pipelines.training.evaluation import run_full_frame_evaluation

    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (128, 128)).save(images_dir / "a.png")
    # object straddling the x=64 tile seam
    json_io.write_annotations(str(labels_dir / "a.json"),
                              [Annotation(subject="catkin", geometry=BBox(54, 54, 74, 74))], 128, 128)

    class _Stub:
        def predict_tiled(self, path, **kw):
            return {"image": path, "width": 128, "height": 128,
                    "boxes": [[54, 54, 74, 74]], "scores": [0.9], "labels": [1], "count": 1}

    monkeypatch.setattr(predictor_mod, "build_predictor", lambda **kw: _Stub())
    # K10 finding 1: this stub carries no persisted training tile geometry, so the delivery-grade
    # gate now refuses unless the caller states the geometry explicitly (the affordance a rail must
    # admit — see test_k10_gate_refuses_unresolvable_tile_geometry for the refusal itself).
    r = run_full_frame_evaluation("ckpt.pt", str(images_dir), str(labels_dir), str(tmp_path / "out"),
                                  subject="catkin", tile_size=64, overlap=0.2)
    assert r["eval_regime"] == "full-frame-tiled-inference"
    # counted once against un-fragmented full-frame GT (tile-level would split/duplicate it)
    assert r["tp"] == 1 and r["fp"] == 0 and r["fn"] == 0
    assert Path(r["results_path"]).is_file()


# ======================================================================
# CV2 — inference tile geometry derived from the checkpoint
# ======================================================================

def _stub_inference(monkeypatch, *, train_tile_size=None, train_overlap=None):
    """Patch build_predictor with a stub that records the tile geometry it was asked to run."""
    import tcip_mcp.pipelines.inference.predictor as predictor_mod

    captured: dict = {}

    class _Stub:
        def __init__(self):
            if train_tile_size is not None:
                self.train_tile_size = train_tile_size
            if train_overlap is not None:
                self.train_overlap = train_overlap

        def predict_batch(self, paths, tile=False, tile_size=None, overlap=None, **kw):
            captured["tile_size"] = tile_size
            captured["overlap"] = overlap
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [], "scores": [], "labels": [], "count": 0} for p in paths]

    monkeypatch.setattr(predictor_mod, "build_predictor", lambda **kw: _Stub())
    return captured


def _one_image(tmp_path):
    from PIL import Image
    img = tmp_path / "a.png"
    Image.new("RGB", (100, 100)).save(img)
    return str(img)


def test_cv2_derives_tile_size_from_checkpoint(tmp_path, monkeypatch):
    from tcip_mcp.tools.inference_tools import run_inference

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    captured = _stub_inference(monkeypatch, train_tile_size=224, train_overlap=0.1)
    r = run_inference(str(ckpt), image_paths=[_one_image(tmp_path)], device="cpu",
                      tile=True, tile_size=None, overlap=None)
    assert captured["tile_size"] == 224
    assert captured["overlap"] == pytest.approx(0.1)
    ts = r["operating_point"]["tile_size"]
    assert ts["value"] == 224 and ts["source"] == "derived"
    assert "warning" not in r  # geometry was recoverable
    # Stage-6 review: overlap has no home in the ResolvedBundle's tracked params — surfaced
    # directly on the result instead of silently dropped after being resolved.
    assert r["overlap"] == pytest.approx(0.1) and r["overlap_source"] == "derived"


def test_cv2_foreign_checkpoint_falls_back_to_default_with_warning(tmp_path, monkeypatch):
    from tcip_mcp.tools.inference_tools import run_inference
    from tcip_mcp.pipelines.resolution import DEFAULT_TILE_SIZE

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    captured = _stub_inference(monkeypatch)  # no train geometry
    r = run_inference(str(ckpt), image_paths=[_one_image(tmp_path)], device="cpu",
                      tile=True, tile_size=None)
    assert captured["tile_size"] == DEFAULT_TILE_SIZE
    assert r["operating_point"]["tile_size"]["source"] == "default"
    assert "no training tile geometry" in r["warning"]
    assert r["overlap"] == pytest.approx(0.2) and r["overlap_source"] == "default"


# ======================================================================
# K10 finding 1 — the delivery-gating path resolves tile geometry the SAME
# way run_inference does (via the shared resolve_tile_geometry), and REFUSES
# rather than silently defaulting when nothing can be resolved.
# ======================================================================

def test_k10_gate_refuses_unresolvable_tile_geometry(tmp_path):
    """Before this fix: a checkpoint with no persisted tiling and no explicit override silently
    scored the delivery gate at a pinned 640/0.2. Now it refuses rather than fabricate a number
    on the path the docstring calls "the number that gates a phenotype delivery"."""
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tcip_mcp.pipelines.training.evaluation import run_full_frame_evaluation

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    class _NoGeometryStub:
        train_tile_size = None
        train_overlap = None

        def predict_tiled(self, path, **kw):
            return {"image": path, "width": 100, "height": 100,
                    "boxes": [], "scores": [], "labels": [], "count": 0}

    predictor_mod_build = predictor_mod.build_predictor
    try:
        predictor_mod.build_predictor = lambda **kw: _NoGeometryStub()
        with pytest.raises(ValueError, match="tiling="):
            run_full_frame_evaluation("ckpt.pt", str(images_dir), str(labels_dir),
                                      str(tmp_path / "out"))
    finally:
        predictor_mod.build_predictor = predictor_mod_build


def test_k10_gate_derives_tile_geometry_from_checkpoint(tmp_path):
    """The checkpoint's own persisted training geometry (already sitting on the predictor object)
    governs the gate instead of the old pinned 640/0.2 — the ~2.9x scale mismatch K10 found."""
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tcip_mcp.pipelines.training.evaluation import run_full_frame_evaluation

    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (100, 100)).save(images_dir / "a.png")
    json_io.write_annotations(str(labels_dir / "a.json"),
                              [Annotation(subject="catkin", geometry=BBox(10, 10, 30, 30))], 100, 100)
    captured: dict = {}

    class _DerivedGeometryStub:
        train_tile_size = 224
        train_overlap = 0.1

        def predict_tiled(self, path, **kw):
            captured["tile_size"] = kw.get("tile_size")
            captured["overlap"] = kw.get("overlap")
            return {"image": path, "width": 100, "height": 100,
                    "boxes": [], "scores": [], "labels": [], "count": 0}

    predictor_mod_build = predictor_mod.build_predictor
    try:
        predictor_mod.build_predictor = lambda **kw: _DerivedGeometryStub()
        r = run_full_frame_evaluation("ckpt.pt", str(images_dir), str(labels_dir),
                                      str(tmp_path / "out"), subject="catkin")
    finally:
        predictor_mod.build_predictor = predictor_mod_build
    assert captured["tile_size"] == 224 and captured["overlap"] == pytest.approx(0.1)
    assert r["tile_size"] == 224 and r["tile_size_source"] == "derived"
    assert r["overlap"] == pytest.approx(0.1) and r["overlap_source"] == "derived"


def test_k10_yolo_training_imgsz_none_when_absent():
    """Before: _training_imgsz silently returned 640 on any read failure, so the delivery gate's
    refusal could never fire for a YOLO checkpoint carrying no real training geometry — the
    refusal was decorative for the whole model kind."""
    from tcip_mcp.pipelines.inference.yolo_predictor import _training_imgsz

    assert _training_imgsz("/nonexistent/checkpoint/path.pt") is None


def test_cv2_explicit_tile_size_wins(tmp_path, monkeypatch):
    from tcip_mcp.tools.inference_tools import run_inference

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    captured = _stub_inference(monkeypatch, train_tile_size=224)  # would derive 224...
    r = run_inference(str(ckpt), image_paths=[_one_image(tmp_path)], device="cpu",
                      tile=True, tile_size=512)  # ...but explicit override wins
    assert captured["tile_size"] == 512
    assert r["operating_point"]["tile_size"]["source"] == "explicit"


def test_cv2_launch_training_persists_effective_tile_geometry(tmp_path, monkeypatch):
    pytest.importorskip("torchvision")
    monkeypatch.chdir(tmp_path)
    import json

    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    import tcip_mcp.pipelines.training.generic_trainer as gt
    from tcip_mcp.tools import training_tools

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    val_images = tmp_path / "val_images"
    val_labels = tmp_path / "val_labels"
    for d in (images_dir, labels_dir, val_images, val_labels):
        d.mkdir()
    for i in range(2):
        Image.new("RGB", (128, 128)).save(images_dir / f"t{i}.png")
        json_io.write_annotations(str(labels_dir / f"t{i}.json"),
                                  [Annotation(subject="catkin", geometry=BBox(10, 10, 40, 40))], 128, 128)
    Image.new("RGB", (128, 128)).save(val_images / "v0.png")
    json_io.write_annotations(str(val_labels / "v0.json"),
                              [Annotation(subject="catkin", geometry=BBox(10, 10, 40, 40))], 128, 128)

    def _stub_train(run, *a, **k):
        run.status = "completed"
        return run

    monkeypatch.setattr(gt, "train", _stub_train)
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1, "min_size": 64, "max_size": 128},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "catkin",
                 "val_images_dir": str(val_images), "val_labels_dir": str(val_labels),
                 "tiling": {"enabled": True}},  # no tile_size -> effective default must be persisted
        "training": {"batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
                     "mixed_precision": False},
    }
    import threading
    before = set(threading.enumerate())
    res = training_tools.launch_training(cfg, str(tmp_path / "out"))
    eid = res["experiment_id"]
    persisted = json.loads((tmp_path / ".tcip" / "experiments" / eid / "config.json").read_text())
    assert persisted["data"]["tiling"]["tile_size"] == 224  # TiledDetectionDataset default
    assert persisted["data"]["tiling"]["overlap"] == pytest.approx(0.2)
    # Await the background training thread so its audit close-event lands in this test's pinned
    # root; leaking the daemon lets that event bleed into a later test's log (tests repin
    # TCIP_PROJECT_ROOT per test, so a late write resolves against whoever is running then).
    for t in set(threading.enumerate()) - before:
        t.join(timeout=60)


# ======================================================================
# CV0 — calibrated operating point wired into the delivery doors
# ======================================================================

def _op_box(cx, cy, s=20.0):
    return [cx - s / 2, cy - s / 2, s, s]


def _op_ann(cx, cy, cid=0, score=None):
    a = {"category_id": cid, "bbox": _op_box(cx, cy)}
    if score is not None:
        a["score"] = score
    return a


def _op_records(idp, *, shift=0.0):
    """Records where count-unbiased conf (0.6) is well-defined and the holdout passes (validated).

    ``shift`` (K1): offsets every GT box's center by that many px (well inside the ~10px
    center-match tolerance) so a holdout fixture's GT content genuinely differs from
    calibration's — a holdout identical in content to calibration (differing only by
    ``image_id``) now trips the content-overlap gate.
    """
    a = {"width": 400, "height": 400, "image_id": f"{idp}_a",
         "gt": [_op_ann(100 + shift, 100)],
         "dt": [_op_ann(100, 100, score=0.9), _op_ann(300, 300, score=0.6)]}
    b = {"width": 400, "height": 400, "image_id": f"{idp}_b",
         "gt": [_op_ann(100 + shift, 100), _op_ann(200 + shift, 200)],
         "dt": [_op_ann(100, 100, score=0.9), _op_ann(200, 200, score=0.3)]}
    return [a, b]


def _good_dense_cal_holdout():
    """K2 rule 17: resolve_operating_point's holdout gate (Fix B/C) needs a REALISTIC dense
    reference, not the 2-image ``_op_records`` toy (its per-image variance now trips Fix C's
    equivalence criterion at n=2). A good detector with one low-conf spurious detection per image —
    the count-unbiased pick lands at the high, correct-match score (0.9) once that FP is filtered
    out, with zero bias/dispersion and full recall/precision on the holdout.
    """
    from tests._dense_op_fixtures import dense_records

    n_images, objects_per_image = 20, 80
    miss, fp = [0] * n_images, [1] * n_images
    cal = dense_records(n_images=n_images, objects_per_image=objects_per_image, id_prefix="c",
                        miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05)
    hold = dense_records(n_images=n_images, objects_per_image=objects_per_image, id_prefix="h",
                         shift=5.0, miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05)
    return cal, hold


class _CalStub:
    """Predictor stub with the mutable operating-point surface run_inference sets."""

    def __init__(self):
        from types import SimpleNamespace
        self.model = SimpleNamespace(score_thresh=0.5, nms_thresh=0.5, detections_per_img=100)
        self.device = "cpu"
        self.score_threshold = 0.5
        self.train_tile_size = None
        self.train_overlap = None
        self.applied_conf = None

    def predict_batch(self, paths, **kw):
        self.applied_conf = self.score_threshold  # conf applied before inference
        return [{"image": p, "width": 100, "height": 100,
                 "boxes": [], "scores": [], "labels": [], "count": 0} for p in paths]


def test_cv0_default_path_unchanged(tmp_path, monkeypatch):
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tcip_mcp.tools.inference_tools import run_inference
    from tcip_mcp.pipelines.resolution import DEFAULT_CONF

    stub = _CalStub()
    monkeypatch.setattr(predictor_mod, "build_predictor", lambda **kw: stub)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    r = run_inference(str(ckpt), image_paths=[_one_image(tmp_path)], device="cpu", tile=False)
    assert r["conf_source"] == "default"
    assert r["validated"] is False
    assert r["operating_point"]["conf"]["value"] == pytest.approx(DEFAULT_CONF)
    assert r["operating_point"]["conf"]["validated_vs_gt"] == "false"
    assert "sweep_summary" not in r  # provenance shape unchanged on the default path


def test_cv0_calibration_wires_resolved_conf(tmp_path, monkeypatch):
    import tcip_mcp.tools.inference_tools as itools
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    cal, hold = _good_dense_cal_holdout()
    bundle = resolve_operating_point("catkin", dataset_hash="H",
                                     calibration_records=cal, holdout_records=hold,
                                     staged_conf_floor=0.01)
    monkeypatch.setattr(itools, "_calibrate_operating_point", lambda *a, **k: (bundle, "H"))
    stub = _CalStub()
    monkeypatch.setattr(predictor_mod, "build_predictor", lambda **kw: stub)
    monkeypatch.chdir(tmp_path)  # sweep artifact under .tcip/artifacts

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    img = _one_image(tmp_path)
    r = itools.run_inference(str(ckpt), image_paths=[img], images_dir=str(tmp_path),
                             device="cpu", tile=False, trait="catkin",
                             calibration_labels_dir=str(tmp_path))
    assert r["conf_source"] == "calibration"
    assert r["validated"] is True                                   # held-out passed
    assert r["dataset_hash"] == "H"
    assert r["operating_point"]["conf"]["value"] == pytest.approx(0.9)
    assert stub.applied_conf == pytest.approx(0.9)                  # resolved conf governs the model
    assert r["sweep_summary"]["count_unbiased_conf"] == pytest.approx(0.9)
    assert Path(r["sweep_path"]).is_file()


def test_cv0_cross_dataset_inheritance_flagged(tmp_path, monkeypatch):
    """Inferencing the SAME labeled set with a bundle scoped to a DIFFERENT hash flags inheritance and
    refuses to stamp validated=True (validated and shippable_issues stay consistent)."""
    import tcip_mcp.tools.inference_tools as itools
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    # Real labeled inference target; bundle is (mockingly) scoped to a foreign hash "H".
    img = _one_image(tmp_path)
    json_io.write_annotations(str(tmp_path / f"{Path(img).stem}.json"),
                              [Annotation(subject="catkin", geometry=BBox(10, 10, 40, 40))], 100, 100)
    bundle = resolve_operating_point("catkin", dataset_hash="H",
                                     calibration_records=_op_records("c"),
                                     holdout_records=_op_records("h", shift=3.0))
    monkeypatch.setattr(itools, "_calibrate_operating_point", lambda *a, **k: (bundle, "H"))
    monkeypatch.setattr(predictor_mod, "build_predictor", lambda **kw: _CalStub())
    monkeypatch.chdir(tmp_path)

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    r = itools.run_inference(str(ckpt), image_paths=[img], images_dir=str(tmp_path),
                             device="cpu", tile=False, trait="catkin",
                             calibration_labels_dir=str(tmp_path))
    assert r["cross_dataset_check"] == "same-labeled-set"
    assert any("inherited across a different dataset" in i for i in r["shippable_issues"])
    assert r["validated"] is False  # not shippable under the target actually used


def test_cv0_unlabeled_target_is_not_comparable_but_shippable(tmp_path, monkeypatch):
    """An unlabeled inference target has no GT hash to compare — record it as such and still ship when
    the held-out calibration passed (no validated/shippable_issues contradiction)."""
    import tcip_mcp.tools.inference_tools as itools
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    cal, hold = _good_dense_cal_holdout()
    bundle = resolve_operating_point("catkin", dataset_hash="H",
                                     calibration_records=cal, holdout_records=hold,
                                     staged_conf_floor=0.01)
    monkeypatch.setattr(itools, "_calibrate_operating_point", lambda *a, **k: (bundle, "H"))
    monkeypatch.setattr(predictor_mod, "build_predictor", lambda **kw: _CalStub())
    monkeypatch.chdir(tmp_path)

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    r = itools.run_inference(str(ckpt), image_paths=[_one_image(tmp_path)], images_dir=str(tmp_path),
                             device="cpu", tile=False, trait="catkin",
                             calibration_labels_dir=str(tmp_path))  # no labels beside the image
    assert r["cross_dataset_check"] == "not-comparable-unlabeled-target"
    assert r["shippable_issues"] == []
    assert r["validated"] is True


def test_cv0_calibration_follows_delivery_tile_regime(tmp_path, monkeypatch):
    """Regime lock: calibration must run the SAME tiled predictor path the delivery uses, not an
    untiled model forward — else the conf is validated in a different regime than it ships through."""
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tcip_mcp.tools.inference_tools import run_inference

    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    for i in range(4):
        Image.new("RGB", (128, 128)).save(images_dir / f"img{i}.png")
        json_io.write_annotations(str(labels_dir / f"img{i}.json"),
                                  [Annotation(subject="catkin", geometry=BBox(10, 10, 40, 40))], 128, 128)

    calls: list[dict] = []

    class _RegimeStub:
        def __init__(self):
            from types import SimpleNamespace
            self.model = SimpleNamespace(score_thresh=0.5, nms_thresh=0.5, detections_per_img=100)
            self.device = "cpu"
            self.score_threshold = 0.5
            self.train_tile_size = 64
            self.train_overlap = 0.2

        def predict_batch(self, paths, tile=False, tile_size=None, overlap=None,
                          tile_batch_size=96, global_nms_iou=None, postprocess="nms"):
            calls.append({"tile": tile, "tile_size": tile_size, "overlap": overlap})
            # tiled finds two boxes/img; untiled finds none — so a sweep over untiled records differs.
            boxes = [[10, 10, 40, 40], [100, 100, 130, 130]] if tile else []
            scores = [0.9, 0.6] if tile else []
            labels = [1, 1] if tile else []
            return [{"image": p, "width": 128, "height": 128, "boxes": boxes,
                     "scores": scores, "labels": labels, "count": len(boxes)} for p in paths]

    monkeypatch.setattr(predictor_mod, "build_predictor", lambda **kw: _RegimeStub())
    monkeypatch.chdir(tmp_path)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")

    run_inference(str(ckpt), images_dir=str(images_dir), device="cpu", tile=True,
                  trait="catkin", calibration_labels_dir=str(labels_dir))
    # Every predictor pass — both calibration splits AND the delivery pass — ran tiled at the same
    # geometry. Under the old untiled-sweep code there was no tiled calibration pass at all.
    assert len(calls) >= 3
    assert all(c["tile"] is True for c in calls)
    assert all(c["tile_size"] == 64 for c in calls)
    assert len({c["overlap"] for c in calls}) == 1


def test_k10_calibrated_bundle_does_not_falsely_stamp_fabricated_geometry_as_derived(
        tmp_path, monkeypatch):
    """K10 finding 3 residual (ceiling-check): before this fix, resolve_operating_point inferred
    "derived" from mere truthiness of tile_size — so a checkpoint with NO persisted geometry, whose
    tile_size silently fell back to the fabricated DEFAULT_TILE_SIZE, still got stamped "derived
    from persisted training geometry" on the calibrated (potentially validated_held_out) bundle.
    Now the real source computed by resolve_tile_geometry travels through _calibrate_operating_point
    into resolve_operating_point, so a fabricated fallback is honestly stamped "default"."""
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tcip_mcp.tools.inference_tools import run_inference

    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    for i in range(4):
        Image.new("RGB", (128, 128)).save(images_dir / f"img{i}.png")
        json_io.write_annotations(str(labels_dir / f"img{i}.json"),
                                  [Annotation(subject="catkin", geometry=BBox(10, 10, 40, 40))], 128, 128)

    class _NoGeometryStub:
        def __init__(self):
            from types import SimpleNamespace
            self.model = SimpleNamespace(score_thresh=0.5, nms_thresh=0.5, detections_per_img=100)
            self.device = "cpu"
            self.score_threshold = 0.5
            self.train_tile_size = None
            self.train_overlap = None

        def predict_batch(self, paths, tile=False, tile_size=None, overlap=None,
                          tile_batch_size=96, global_nms_iou=None, postprocess="nms"):
            boxes = [[10, 10, 40, 40], [100, 100, 130, 130]] if tile else []
            scores = [0.9, 0.6] if tile else []
            labels = [1, 1] if tile else []
            return [{"image": p, "width": 128, "height": 128, "boxes": boxes,
                     "scores": scores, "labels": labels, "count": len(boxes)} for p in paths]

    monkeypatch.setattr(predictor_mod, "build_predictor", lambda **kw: _NoGeometryStub())
    monkeypatch.chdir(tmp_path)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")

    r = run_inference(str(ckpt), images_dir=str(images_dir), device="cpu", tile=True,
                      trait="catkin", calibration_labels_dir=str(labels_dir))
    ts = r["operating_point"]["tile_size"]
    assert ts["value"] == 640  # the fabricated fallback (DEFAULT_TILE_SIZE) — real value, unchanged
    assert ts["source"] == "default"  # honestly labeled — NOT "derived" despite being truthy


def test_cv0_export_predictions_validated_from_bundle(tmp_path, monkeypatch):
    import json
    import tcip_mcp.tools.inference_tools as itools

    img = _one_image(tmp_path)
    fake_op = {"conf": {"value": 0.6, "validated_vs_gt": "validated_held_out"}}

    def _fake_run_inference(**kw):
        return {
            "results": [{"image": img, "width": 100, "height": 100,
                         "boxes": [], "scores": [], "labels": [], "count": 0}],
            "operating_point": fake_op, "validated": True, "conf_source": "calibration",
            "shippable_issues": [],
        }

    monkeypatch.setattr(itools, "run_inference", _fake_run_inference)
    out_dir = tmp_path / "out"
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    r = itools.export_predictions(str(ckpt), str(tmp_path), str(out_dir), trait="catkin",
                                  calibration_labels_dir=str(tmp_path))
    stamp = json.loads((out_dir / "operating_point.json").read_text())
    assert stamp["validated"] is True                       # not hardcoded False
    assert r["validated"] is True and r["conf_source"] == "calibration"


def test_cv0_tabulate_counts_carries_operating_point(tmp_path, monkeypatch):
    import tcip_mcp.tools.inference_tools as itools

    captured = {}

    def _fake_run_inference(**kw):
        captured.update(kw)
        return {
            "results": [{"image": "a.png", "count": 3}],
            "image_count": 1, "total_detections": 3,
            "operating_point": {"conf": {"value": 0.6}},
            "validated": True, "conf_source": "calibration",
        }

    monkeypatch.setattr(itools, "run_inference", _fake_run_inference)
    monkeypatch.setattr(
        itools, "export_detection_csv",
        lambda results, path, provenance=None, measurement_validated=None,
        acknowledge_unvalidated=False: str(path))
    r = itools.tabulate_counts("m.pt", str(tmp_path), str(tmp_path / "o.csv"),
                                  trait="catkin", calibration_labels_dir=str(tmp_path))
    assert captured["trait"] == "catkin"                    # calibration threaded through
    assert captured["calibration_labels_dir"] == str(tmp_path)
    assert r["validated"] is True and r["conf_source"] == "calibration"
    assert r["operating_point"] == {"conf": {"value": 0.6}}
