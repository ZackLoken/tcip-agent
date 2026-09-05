"""Detection measurement-integrity coverage: val-loss over negatives, standard+operating mAP,
tiled evaluation, inference tile geometry, calibrated inference.

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

# No built-in traits: seed_bud_trait_spec (conftest.py) writes a real
# bud.yml into this test's pinned platform state root so trait="bud_opening" call sites keep resolving.
pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")


def _stub_checkpoint(path: str = "ckpt.pt"):
    """A VerifiedCheckpoint stand-in for a test that stubs build_predictor: no registry lookup
    or file read behind it, since the predictor it would build is stubbed too."""
    from tests._verified_checkpoint_fixtures import stub_verified_checkpoint

    return stub_verified_checkpoint(path)


def _patch_build_predictor(monkeypatch, predictor_mod, stub_factory):
    """Stub build_predictor (its checkpoint argument is now positional) and the
    load_registered_checkpoint a calling tool resolves it through, so a stubbed-predictor test
    never needs a real registered checkpoint on disk."""
    import tcip_mcp.model_registry as model_registry_mod

    monkeypatch.setattr(predictor_mod, "build_predictor", lambda *a, **kw: stub_factory())
    monkeypatch.setattr(model_registry_mod, "load_registered_checkpoint",
                        lambda *a, **kw: _stub_checkpoint())


# ======================================================================
# Detection val-loss must include all-negative images
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


def test_val_loss_forwards_all_negative_images():
    stub = _StubDetector()
    model = _StubModel(stub)
    loader = [_det_batch([1, 0]), _det_batch([0])]  # mixed batch, then all-negative batch
    evaluate(model, loader, torch.device("cpu"), "detection")
    # Both batches forwarded through the detector (full batch incl. negatives), not just foreground.
    assert stub.calls == [(2, 4), (1, 0)]


def test_all_negative_only_loader_is_not_skipped():
    stub = _StubDetector()
    model = _StubModel(stub)
    loader = [_det_batch([0, 0])]  # nothing but negatives
    result = evaluate(model, loader, torch.device("cpu"), "detection")
    assert stub.calls == [(2, 0)]  # forwarded, not skipped
    assert result["loss"] == pytest.approx(2.5)  # finite, non-zero: negatives contribute loss


# ======================================================================
# Report standard map@100 and map@max_dets
# ======================================================================

def _rec(gt, dt, w=100, h=100):
    return build_coco_image_record(w, h, gt, dt)


def _perfect_records(n=1):
    gt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}]
    dt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}]
    return [_rec(gt, dt) for _ in range(n)]


def test_map_nonzero_at_high_maxdets():
    m = coco_detection_metrics(_perfect_records(), max_dets=1000)
    assert m["map"] == pytest.approx(1.0)      # standard map@100 no longer collapses to 0.0
    assert m["map50"] == pytest.approx(1.0)
    assert m["map_at_maxdets"] == pytest.approx(1.0)
    assert m["map50_at_maxdets"] == pytest.approx(1.0)


def test_map_invariant_across_maxdets():
    recs = _perfect_records(3)
    maps = {md: coco_detection_metrics(recs, max_dets=md) for md in (100, 300, 1000)}
    base = maps[100]
    for md in (300, 1000):
        assert maps[md]["map"] == pytest.approx(base["map"])
        assert maps[md]["map50"] == pytest.approx(base["map50"])
        assert maps[md]["map75"] == pytest.approx(base["map75"])


def test_standard_keys_unchanged_at_100():
    m = coco_detection_metrics(_perfect_records(), max_dets=100)
    assert m["map"] == pytest.approx(1.0)
    assert m["map50"] == pytest.approx(1.0)
    # at the standard cap the operating-cap figures equal the standard ones
    assert m["map_at_maxdets"] == pytest.approx(m["map"])


# ======================================================================
# Tiled evaluation regimes
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
                                  [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], size, size)
    return images_dir, labels_dir


def _capture_run_test_evaluation(monkeypatch):
    """Patch run_test_evaluation to record the built dataset + tiling instead of loading a model."""
    import tcip_mcp.pipelines.training.eval_runners as runners

    captured: dict = {}

    def _fake(ckpt, loader, device, task, output_dir, **kw):
        captured["ds"] = loader.dataset
        captured["tiling"] = kw.get("tiling")
        return {"tiled": bool(kw.get("tiling")), "eval_regime": "tile-level"}

    monkeypatch.setattr(runners, "run_test_evaluation", _fake)
    return captured


def test_run_id_reuses_training_tiling(tmp_path, monkeypatch):
    from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    images_dir, labels_dir = _det_dataset(tmp_path)
    run = create_run({"data": {"tiling": {"enabled": True, "tile_size": 64}}}, str(tmp_path / "out"))
    out = Path(run.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    registered_checkpoint(out, project_root=str(tmp_path), filename="model_best.pt")

    captured = _capture_run_test_evaluation(monkeypatch)
    evaluate_model(run.run_id, str(images_dir), str(labels_dir), task="detection", subject="bud")
    assert isinstance(captured["ds"], TiledDetectionDataset)
    assert captured["ds"].num_samples > 3  # more tiles than the 3 source images
    assert captured["tiling"] == {"enabled": True, "tile_size": 64}


def test_explicit_checkpoint_stays_untiled(tmp_path, monkeypatch):
    from tcip_mcp.pipelines.data.datasets import DetectionDataset, TiledDetectionDataset
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    images_dir, labels_dir = _det_dataset(tmp_path)
    ckpt = registered_checkpoint(tmp_path, project_root=str(tmp_path), filename="model.pt")

    captured = _capture_run_test_evaluation(monkeypatch)
    evaluate_model(ckpt, str(images_dir), str(labels_dir), task="detection", subject="bud")
    assert isinstance(captured["ds"], DetectionDataset)
    assert not isinstance(captured["ds"], TiledDetectionDataset)
    assert captured["tiling"] is None


def test_explicit_tiling_override_on_checkpoint(tmp_path, monkeypatch):
    from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    images_dir, labels_dir = _det_dataset(tmp_path)
    ckpt = registered_checkpoint(tmp_path, project_root=str(tmp_path), filename="model.pt")

    captured = _capture_run_test_evaluation(monkeypatch)
    evaluate_model(ckpt, str(images_dir), str(labels_dir), task="detection", subject="bud",
                   tiling={"enabled": True, "tile_size": 64})
    assert isinstance(captured["ds"], TiledDetectionDataset)


def test_full_frame_counts_straddling_object_once(tmp_path, monkeypatch):
    import tcip_store as ts
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tcip_mcp.pipelines.training.eval_runners import evaluation_results_key, run_full_frame_evaluation

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
                              [Annotation(subject="bud", geometry=BBox(54, 54, 74, 74))], 128, 128)

    class _Stub:
        def predict_tiled(self, path, **kw):
            return {"image": path, "width": 128, "height": 128,
                    "boxes": [[54, 54, 74, 74]], "scores": [0.9], "labels": [1], "count": 1}

    monkeypatch.setattr(predictor_mod, "build_predictor", lambda *a, **kw: _Stub())
    # This stub carries no persisted training tile geometry, so the delivery-grade
    # gate now refuses unless the caller states the geometry explicitly (the affordance a rail must
    # admit; see test_gate_refuses_unresolvable_tile_geometry for the refusal itself).
    r = run_full_frame_evaluation(_stub_checkpoint(), str(images_dir), str(labels_dir), str(tmp_path / "out"),
                                  subject="bud", tile_size=64, overlap=0.2)
    assert r["eval_regime"] == "full-frame-tiled-inference"
    # counted once against un-fragmented full-frame GT (tile-level would split/duplicate it)
    assert r["tp"] == 1 and r["fp"] == 0 and r["fn"] == 0
    assert ts.exists(evaluation_results_key(tmp_path / "out"))


def test_evaluate_scores_a_contradicted_negative_on_its_actual_content_and_names_it(
    tmp_path, monkeypatch
):
    """A stored negative whose label file now holds the subject is excluded from the negative
    count (scored on its real content, not silently dropped) and named in the result's
    contradicted_negatives so a reviewer sees the stale confirmation without a separate doctor
    pass."""
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tcip_mcp.dataset_layout import CONFIRMED_NEGATIVE, record_image_statuses, status_bucket
    from tcip_mcp.pipelines.training.eval_runners import run_full_frame_evaluation

    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (128, 128)).save(images_dir / "a.png")
    json_io.write_annotations(str(labels_dir / "a.json"), [], 128, 128, keep_empty=True)
    record_image_statuses(tmp_path, status_bucket("bud", None), {"a.png": CONFIRMED_NEGATIVE},
                          recorded_by="user:breeder")
    json_io.write_annotations(
        str(labels_dir / "a.json"),
        [Annotation(subject="bud", geometry=BBox(54, 54, 74, 74))], 128, 128,
    )

    class _Stub:
        def predict_tiled(self, path, **kw):
            return {"image": path, "width": 128, "height": 128,
                    "boxes": [[54, 54, 74, 74]], "scores": [0.9], "labels": [1], "count": 1}

    monkeypatch.setattr(predictor_mod, "build_predictor", lambda *a, **kw: _Stub())
    r = run_full_frame_evaluation(_stub_checkpoint(), str(images_dir), str(labels_dir), str(tmp_path / "out"),
                                  subject="bud", tile_size=64, overlap=0.2)
    assert r["contradicted_negatives"] == ["a.png"]
    # scored against the real content, not held out as a still-trusted negative
    assert r["scored_images"] == 1 and r["tp"] == 1 and r["fn"] == 0


def test_attribute_registry_refusal_reaches_the_caller(tmp_path, monkeypatch):
    """run_full_frame_evaluation must not let a bare `except Exception` around
    resolve_registry_id_map swallow an attribute-classification registry refusal and silently
    score against zero ground truth instead of refusing. An attribute needs a real classes.json
    to order its values (resolve_registry_id_map's own deliberate ValueError); no classes.json
    exists here, so this must propagate as a real refusal, not a quietly-empty GT read."""
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tcip_mcp.pipelines.training.eval_runners import run_full_frame_evaluation

    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (128, 128)).save(images_dir / "a.png")
    json_io.write_annotations(str(labels_dir / "a.json"),
                              [Annotation(subject="bud", geometry=BBox(54, 54, 74, 74))], 128, 128)

    class _Stub:
        def predict_tiled(self, path, **kw):
            return {"image": path, "width": 128, "height": 128,
                    "boxes": [], "scores": [], "labels": [], "count": 0}

    monkeypatch.setattr(predictor_mod, "build_predictor", lambda *a, **kw: _Stub())
    with pytest.raises(ValueError, match="classes.json"):
        run_full_frame_evaluation(_stub_checkpoint(), str(images_dir), str(labels_dir), str(tmp_path / "out"),
                                  subject="bud", attribute="opening", tile_size=64, overlap=0.2)


# ======================================================================
# Inference tile geometry derived from the checkpoint
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

    _patch_build_predictor(monkeypatch, predictor_mod, _Stub)
    return captured


def _one_image(tmp_path):
    from PIL import Image
    img = tmp_path / "a.png"
    Image.new("RGB", (100, 100)).save(img)
    return str(img)


def test_derives_tile_size_from_checkpoint(tmp_path, monkeypatch):
    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

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
    # overlap has no home in the ResolvedBundle's tracked params; surfaced
    # directly on the result instead of silently dropped after being resolved.
    assert r["overlap"] == pytest.approx(0.1) and r["overlap_source"] == "derived"


def test_foreign_checkpoint_with_no_geometry_refuses_explicit_tile(tmp_path, monkeypatch):
    """A checkpoint with no persisted training tile geometry has no real basis to tile at: an
    explicit tile=True with no explicit tile_size either must refuse (naming the missing basis),
    never silently fabricate a scale to proceed on."""
    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    _stub_inference(monkeypatch)  # no train geometry
    r = run_inference(str(ckpt), image_paths=[_one_image(tmp_path)], device="cpu",
                      tile=True, tile_size=None)
    assert "error" in r
    assert "tile_size" in r["error"]


# ======================================================================
# The delivery-gating path resolves tile geometry the same
# way run_inference does (via the shared resolve_tile_geometry), and refuses
# rather than silently defaulting when nothing can be resolved.
# ======================================================================

def test_gate_refuses_unresolvable_tile_geometry(tmp_path):
    """A checkpoint with no persisted tiling and no explicit override must refuse the delivery
    gate rather than silently score it at an ungrounded scale, on the path the docstring calls
    "the number that gates a phenotype delivery"."""
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tcip_mcp.pipelines.training.eval_runners import run_full_frame_evaluation

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
        predictor_mod.build_predictor = lambda *a, **kw: _NoGeometryStub()
        with pytest.raises(ValueError, match="tiling="):
            run_full_frame_evaluation(_stub_checkpoint(), str(images_dir), str(labels_dir),
                                      str(tmp_path / "out"))
    finally:
        predictor_mod.build_predictor = predictor_mod_build


def test_gate_derives_tile_geometry_from_checkpoint(tmp_path):
    """The checkpoint's own persisted training geometry (already sitting on the predictor object)
    governs the gate instead of an arbitrary fixed scale, avoiding a scale mismatch (~2.9x for this
    checkpoint) between the geometry the gate assumes and the geometry the model was trained at."""
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tcip_mcp.pipelines.training.eval_runners import run_full_frame_evaluation

    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (100, 100)).save(images_dir / "a.png")
    json_io.write_annotations(str(labels_dir / "a.json"),
                              [Annotation(subject="bud", geometry=BBox(10, 10, 30, 30))], 100, 100)
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
        predictor_mod.build_predictor = lambda *a, **kw: _DerivedGeometryStub()
        r = run_full_frame_evaluation(_stub_checkpoint(), str(images_dir), str(labels_dir),
                                      str(tmp_path / "out"), subject="bud")
    finally:
        predictor_mod.build_predictor = predictor_mod_build
    assert captured["tile_size"] == 224 and captured["overlap"] == pytest.approx(0.1)
    assert r["tile_size"] == 224 and r["tile_size_source"] == "derived"
    assert r["overlap"] == pytest.approx(0.1) and r["overlap_source"] == "derived"


def test_run_inference_no_registry_refuses_naming_write_class_map(tmp_path, monkeypatch):
    """An attribute-scoped run against a dataset with no classes.json refuses before the pass
    runs, naming write_class_map as the remedy: a classified run with an unresolvable id_map can
    no longer fall back to a raw-index name, since a value outside any vocabulary is worse than a
    refusal."""
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    images_dir = tmp_path / "images"  # no classes.json anywhere under this root
    images_dir.mkdir()
    from PIL import Image
    Image.new("RGB", (100, 100)).save(images_dir / "a.png")

    class _Stub:
        config = {"data": {"subject": "bud", "attribute": "state"}}

        def predict_batch(self, paths, **kw):
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [[10, 10, 20, 20]], "scores": [0.9], "labels": [1], "count": 1}
                    for p in paths]

    _patch_build_predictor(monkeypatch, predictor_mod, _Stub)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")

    r = run_inference(str(ckpt), images_dir=str(images_dir), device="cpu")
    assert "error" in r
    assert "write_class_map" in r["error"]


def test_run_inference_corrupted_registry_still_propagates(tmp_path, monkeypatch):
    """The precondition check (resolved_classes_path) only short-circuits the legitimate
    no-registry case: a classes.json that exists but is corrupted is a real, unexpected failure
    and must still raise loudly, not be silently absorbed by the same precondition that admits
    the honest degraded case."""
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    root = tmp_path / "ds"
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    (root / "classes.json").write_text("{not valid json", encoding="utf-8")
    from PIL import Image
    Image.new("RGB", (100, 100)).save(images_dir / "a.png")

    class _Stub:
        config = {"data": {"subject": "bud", "attribute": "state"}}

        def predict_batch(self, paths, **kw):
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [], "scores": [], "labels": [], "count": 0} for p in paths]

    _patch_build_predictor(monkeypatch, predictor_mod, _Stub)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")

    with pytest.raises(Exception):  # json.JSONDecodeError (a ValueError subclass), not swallowed
        run_inference(str(ckpt), images_dir=str(images_dir), device="cpu")


def test_explicit_tile_size_wins(tmp_path, monkeypatch):
    """An explicit edge that agrees with the checkpoint's own persisted geometry wins over
    deriving it, still stamped as the caller's own explicit statement."""
    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    captured = _stub_inference(monkeypatch, train_tile_size=224)
    r = run_inference(str(ckpt), image_paths=[_one_image(tmp_path)], device="cpu",
                      tile=True, tile_size=224)
    assert captured["tile_size"] == 224
    assert r["operating_point"]["tile_size"]["source"] == "explicit"


def test_explicit_tile_size_contradicting_persisted_geometry_refuses(tmp_path, monkeypatch):
    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    _stub_inference(monkeypatch, train_tile_size=224)
    r = run_inference(str(ckpt), image_paths=[_one_image(tmp_path)], device="cpu",
                      tile=True, tile_size=512)
    assert "error" in r
    assert "512" in r["error"] and "224" in r["error"]


def test_launch_training_persists_effective_tile_geometry(tmp_path, monkeypatch):
    """launch_training runs the training body in a real subprocess, so the effective
    tiling geometry can only be known (and patched into the durable experiment record) once that
    child builds the dataset, after launch_training has already returned. Polls for it instead of
    asserting synchronously.

    Also pins the isolation itself, not just the timing change: this monkeypatches
    ``generic_trainer.train`` in this process to raise if ever called. If the run executed in this
    same interpreter, that monkeypatch would poison it and the "status == completed" assertion
    below would fail, since the poisoned ``train`` would be the one actually invoked. Because the
    real subprocess re-imports fresh, the monkeypatch here has no effect and the run completes
    normally, proving the training body genuinely executes outside this process, not merely that
    the API still returns the right shape."""
    pytest.importorskip("torchvision")
    monkeypatch.chdir(tmp_path)
    import os
    import time

    import tcip_store as ts
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.experiments import config_key
    import tcip_mcp.pipelines.training.generic_trainer as gt
    from tcip_mcp.tools import training_tools

    def _poison_train(*a, **k):
        raise AssertionError(
            "generic_trainer.train ran inside the launching process: subprocess isolation broken")

    monkeypatch.setattr(gt, "train", _poison_train)

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    val_images = tmp_path / "val_images"
    val_labels = tmp_path / "val_labels"
    for d in (images_dir, labels_dir, val_images, val_labels):
        d.mkdir()
    for i in range(2):
        Image.new("RGB", (128, 128)).save(images_dir / f"t{i}.png")
        json_io.write_annotations(str(labels_dir / f"t{i}.json"),
                                  [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], 128, 128)
    Image.new("RGB", (128, 128)).save(val_images / "v0.png")
    json_io.write_annotations(str(val_labels / "v0.json"),
                              [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], 128, 128)

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1, "min_size": 64, "max_size": 128},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
                 "val_images_dir": str(val_images), "val_labels_dir": str(val_labels),
                 "tiling": {"enabled": True}},  # no tile_size -> effective default must be persisted
        "training": {"batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
                     "mixed_precision": False, "device": "cpu"},
    }
    res = training_tools.launch_training(cfg, str(tmp_path / "out"))
    assert res["pid"] != os.getpid()  # a different OS process, not this one
    eid = res["experiment_id"]
    run_id = res["run_id"]
    key = config_key(eid)

    deadline = time.monotonic() + 90
    tiling: dict = {}
    while time.monotonic() < deadline:
        if ts.exists(key):
            tiling = ts.read(key).get("data", {}).get("tiling", {})
            if "tile_size" in tiling:
                break
        status = training_tools.monitor_training(run_id)
        if status.get("status") in ("failed", "cancelled"):
            pytest.fail(f"training subprocess ended early: {status}")
        time.sleep(0.5)
    else:
        pytest.fail("timed out waiting for effective tiling geometry to be persisted")

    assert tiling["tile_size"] == 224  # TiledDetectionDataset default
    assert tiling["overlap"] == pytest.approx(0.2)

    # Let the subprocess actually finish rather than leaking it: it keeps writing to this test's
    # pinned TCIP_STATE_ROOT, and a late write after the test moves on would resolve against
    # whoever is running then (tests repin TCIP_STATE_ROOT per test, they don't isolate the OS
    # process tree). Asserting specifically on "completed" (not just any terminal state) is what
    # makes the poisoned gt.train monkeypatch load-bearing: if the child ran inside this process,
    # it would hit _poison_train and the run would be "failed", not "completed".
    final_status = None
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        final_status = training_tools.monitor_training(run_id).get("status")
        if final_status in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.5)
    else:
        pytest.fail("timed out waiting for training subprocess to finish")
    assert final_status == "completed"


# ======================================================================
# Calibrated operating point wired into the delivery doors
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

    ``shift`` offsets every GT box's center by that many px (well inside the ~10px
    center-match tolerance) so a holdout fixture's GT content genuinely differs from
    calibration's: a holdout identical in content to calibration (differing only by
    ``image_id``) trips the content-overlap gate.
    """
    a = {"width": 400, "height": 400, "image_id": f"{idp}_a",
         "gt": [_op_ann(100 + shift, 100)],
         "dt": [_op_ann(100, 100, score=0.9), _op_ann(300, 300, score=0.6)]}
    b = {"width": 400, "height": 400, "image_id": f"{idp}_b",
         "gt": [_op_ann(100 + shift, 100), _op_ann(200 + shift, 200)],
         "dt": [_op_ann(100, 100, score=0.9), _op_ann(200, 200, score=0.3)]}
    return [a, b]


def _good_dense_cal_holdout():
    """resolve_operating_point's holdout gate needs a realistic dense
    reference, not the 2-image ``_op_records`` toy (its per-image variance trips the
    equivalence criterion at n=2). A good detector with one low-conf spurious detection per image:
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


def test_default_path_unchanged(tmp_path, monkeypatch):
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference
    from tcip_mcp.pipelines.resolution import DEFAULT_CONF

    stub = _CalStub()
    _patch_build_predictor(monkeypatch, predictor_mod, lambda: stub)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    r = run_inference(str(ckpt), image_paths=[_one_image(tmp_path)], device="cpu", tile=False)
    assert r["conf_source"] == "default"
    assert r["validated"] is False
    assert r["operating_point"]["conf"]["value"] == pytest.approx(DEFAULT_CONF)
    assert r["operating_point"]["conf"]["validated_against"] == "false"
    assert "gate_evidence_summary" not in r  # provenance shape unchanged on the default path


def _stand_in_calibration(monkeypatch, calibration_pipeline, labels_dir, **overrides):
    """Stand in for the calibration pass, returning what it returns: the resolved bundle, its
    dataset identity, no excluded stems, and the evidence a delivery door reopens the gate over."""
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    cal, hold = _good_dense_cal_holdout()
    inputs = {"dataset_hash": "H", "calibration_records": cal, "holdout_records": hold,
              "staged_conf_floor": 0.01, **overrides}
    bundle = resolve_operating_point("bud_opening", experiment_id=None, **inputs)
    evidence = {"resolver": "resolve_operating_point", "inputs": inputs,
                "reference_inputs": {"label_dirs": {"calibration": str(labels_dir)}}}
    monkeypatch.setattr(calibration_pipeline, "calibrate_operating_point",
                        lambda *a, **k: (bundle, "H", 0, evidence))
    return bundle


def test_calibration_wires_resolved_conf(tmp_path, monkeypatch):
    import tcip_store as ts
    import tcip_mcp.pipelines.calibration as calibration_pipeline
    import tcip_mcp.tools.inference_tools as itools
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tests._verified_checkpoint_fixtures import run_inference_verified

    # tiled=False: this test's real verified-pass call below is tile=False (tile_size only
    # gates a bundle when tiled, so the mocked bundle must match the real regime it stands in for).
    _stand_in_calibration(monkeypatch, calibration_pipeline, tmp_path, tiled=False)
    stub = _CalStub()
    _patch_build_predictor(monkeypatch, predictor_mod, lambda: stub)
    monkeypatch.chdir(tmp_path)  # sweep artifact under .tcip/artifacts

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    img = _one_image(tmp_path)
    r = run_inference_verified(str(ckpt), image_paths=[img], images_dir=str(tmp_path),
                               device="cpu", tile=False, trait="bud_opening",
                               calibration_labels_dir=str(tmp_path))
    assert r["conf_source"] == "calibration"
    assert r["validated"] is True                                   # held-out passed
    assert r["dataset_hash"] == "H"
    assert r["operating_point"]["conf"]["value"] == pytest.approx(0.9)
    assert stub.applied_conf == pytest.approx(0.9)                  # resolved conf governs the model
    assert r["gate_evidence_summary"]["count_unbiased_conf"] == pytest.approx(0.9)
    assert ts.exists(itools.calibration_curve_key(r["calibration_evidence_key"]))


def test_sweep_artifact_is_content_addressed_not_label_hash_only(tmp_path, monkeypatch):
    """Two calibrations on the same checkpoint+labels but different predictor-path
    settings must not collide on the sweep artifact filename."""
    import tcip_store as ts
    import tcip_mcp.pipelines.calibration as calibration_pipeline
    import tcip_mcp.tools.inference_tools as itools
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tests._verified_checkpoint_fixtures import run_inference_verified

    _stand_in_calibration(monkeypatch, calibration_pipeline, tmp_path, tiled=True)
    stub = _CalStub()
    _patch_build_predictor(monkeypatch, predictor_mod, lambda: stub)
    monkeypatch.chdir(tmp_path)

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    img = _one_image(tmp_path)

    r_untiled = run_inference_verified(str(ckpt), image_paths=[img], images_dir=str(tmp_path),
                                       device="cpu", tile=False, trait="bud_opening",
                                       calibration_labels_dir=str(tmp_path))
    r_tiled = run_inference_verified(str(ckpt), image_paths=[img], images_dir=str(tmp_path),
                                     device="cpu", tile=True, tile_size=256, trait="bud_opening",
                                     calibration_labels_dir=str(tmp_path))

    assert r_untiled["calibration_curve_path"] != r_tiled["calibration_curve_path"]  # distinct predictor paths, distinct files
    assert ts.exists(itools.calibration_curve_key(r_untiled["calibration_evidence_key"]))
    assert ts.exists(itools.calibration_curve_key(r_tiled["calibration_evidence_key"]))

    sweep_body = ts.read(itools.calibration_curve_key(r_tiled["calibration_evidence_key"]))
    assert sweep_body["checkpoint_sha256"]  # identity threaded through, not omitted
    assert sweep_body["predictor_path"]["tile"] is True
    assert sweep_body["predictor_path"]["tile_size"] == 256


def test_run_inference_sidecar_carries_sweep_pointer(tmp_path, monkeypatch):
    """The delivered operating_point.json sidecar must record the sweep artifact
    that justified the shipped conf, not omit it entirely."""
    from pathlib import PurePosixPath

    import tcip_store as ts
    import tcip_mcp.pipelines.calibration as calibration_pipeline
    import tcip_mcp.tools.inference_tools as itools
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    # tiled=False matches the real tile=False run_inference runs at below, so the gate
    # never sees a mismatched tile_size dimension against the actual untiled call.
    _stand_in_calibration(monkeypatch, calibration_pipeline, tmp_path, tiled=False)
    stub = _CalStub()
    _patch_build_predictor(monkeypatch, predictor_mod, lambda: stub)
    monkeypatch.chdir(tmp_path)

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    _one_image(tmp_path)  # written directly under images_dir=tmp_path; run_inference globs it

    out = itools.run_inference(
        str(ckpt), images_dir=str(tmp_path), output_dir="dataset/predictions/baseline/2026-01-01",
        device="cpu", tile=False, trait="bud_opening", calibration_labels_dir=str(tmp_path))
    assert "error" not in out, out
    sidecar = read_operating_point_sidecar(out["output_dir"])
    assert sidecar["calibration_curve_path"]
    sweep_name = Path(sidecar["calibration_curve_path"]).name
    relative = PurePosixPath(".tcip", "artifacts", sweep_name)
    (inputs_hash,) = itools._CalibrationCurveLocator().parts_from(relative)
    assert ts.exists(itools.calibration_curve_key(inputs_hash))
    assert sidecar["gate_evidence_summary"]["count_unbiased_conf"] == pytest.approx(0.9)


def test_cross_dataset_inheritance_flagged(tmp_path, monkeypatch):
    """Inferencing the same labeled set with a bundle scoped to a different hash flags inheritance and
    refuses to stamp validated=True (validated and shippable_issues stay consistent)."""
    import tcip_mcp.pipelines.calibration as calibration_pipeline
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tests._verified_checkpoint_fixtures import run_inference_verified

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    # Real labeled inference target; bundle is (mockingly) scoped to a foreign hash "H".
    img = _one_image(tmp_path)
    json_io.write_annotations(str(tmp_path / f"{Path(img).stem}.json"),
                              [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], 100, 100)
    inputs = {"tiled": True, "dataset_hash": "H", "calibration_records": _op_records("c"),
              "holdout_records": _op_records("h", shift=3.0)}
    bundle = resolve_operating_point("bud_opening", experiment_id=None, **inputs)
    evidence = {"resolver": "resolve_operating_point", "inputs": inputs,
                "reference_inputs": {"label_dirs": {"calibration": str(tmp_path)}}}
    monkeypatch.setattr(calibration_pipeline, "calibrate_operating_point",
                        lambda *a, **k: (bundle, "H", 0, evidence))
    _patch_build_predictor(monkeypatch, predictor_mod, _CalStub)
    monkeypatch.chdir(tmp_path)

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    r = run_inference_verified(str(ckpt), image_paths=[img], images_dir=str(tmp_path),
                               device="cpu", tile=False, trait="bud_opening",
                               calibration_labels_dir=str(tmp_path))
    assert r["cross_dataset_check"] == "same-labeled-set"
    assert any("inherited across a different dataset" in i for i in r["shippable_issues"])
    assert r["validated"] is False  # not shippable under the target actually used


def test_manifest_calibration_subset_of_inference_target_is_still_comparable(tmp_path, monkeypatch):
    """Under a manifest the calibration's own universe is a held-out subset of the labelled
    directory; inferring the whole directory is still the same labelled set, so the firewall
    compares real hashes instead of reading a fully-labelled target as unlabeled."""
    import tcip_mcp.pipelines.calibration as calibration_pipeline
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tests._verified_checkpoint_fixtures import run_inference_verified

    from PIL import Image

    cal, hold = _good_dense_cal_holdout()
    inputs = {"tiled": False, "dataset_hash": "H", "calibration_records": cal,
             "holdout_records": hold, "staged_conf_floor": 0.01}
    bundle = resolve_operating_point("bud_opening", experiment_id=None, **inputs)
    evidence = {
        "resolver": "resolve_operating_point", "inputs": inputs,
        "reference_inputs": {
            "label_stems": {"calibration": {"path": str(tmp_path), "stems": ["a"]}},
            "stated_values": {"split_manifest_dir": str(tmp_path / "m")},
        },
        "calibration_stems": ["a"],
    }
    monkeypatch.setattr(calibration_pipeline, "calibrate_operating_point",
                        lambda *a, **k: (bundle, "H", 0, evidence))
    _patch_build_predictor(monkeypatch, predictor_mod, _CalStub)
    monkeypatch.chdir(tmp_path)

    img_a, img_b = tmp_path / "a.png", tmp_path / "b.png"
    Image.new("RGB", (100, 100)).save(img_a)
    Image.new("RGB", (100, 100)).save(img_b)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    r = run_inference_verified(
        str(ckpt), image_paths=[str(img_a), str(img_b)], images_dir=str(tmp_path), device="cpu",
        tile=False, trait="bud_opening", calibration_labels_dir=str(tmp_path),
        split_manifest_dir=str(tmp_path / "m"))

    assert r["cross_dataset_check"] == "same-labeled-set"


@pytest.mark.parametrize("give_calibration_images_dir", [False, True])
@pytest.mark.parametrize("case,inf_stems,expected_check", [
    ("superset", ("a", "b", "c"), "same-labeled-set"),
    ("equal", ("a", "b"), "same-labeled-set"),
    ("subset", ("a",), "not-comparable-unlabeled-target"),
    ("disjoint", ("c", "d"), "not-comparable-unlabeled-target"),
])
def test_manifest_calibration_firewall_hashes_the_universe(
        tmp_path, monkeypatch, case, inf_stems, expected_check, give_calibration_images_dir):
    """The firewall's target hash under a manifest is computed over the calibration universe (the
    same stems the bundle's own hash covers), not the inference stem list: a superset or equal
    inference target compares real, matching hashes and validates; a genuine subset or disjoint
    target stays the honest not-comparable case, in both cases regardless of whether
    calibration_images_dir is given."""
    import tcip_mcp.pipelines.calibration as calibration_pipeline
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tcip_mcp.pipelines.resolution import dataset_hash
    from tests._verified_checkpoint_fixtures import run_inference_verified

    from PIL import Image

    cal, hold = _good_dense_cal_holdout()
    universe = ["a", "b"]
    dh = dataset_hash(tmp_path, stems=universe)
    inputs = {"tiled": False, "dataset_hash": dh, "calibration_records": cal,
             "holdout_records": hold, "staged_conf_floor": 0.01}
    bundle = resolve_operating_point("bud_opening", experiment_id=None, **inputs)
    evidence = {
        "resolver": "resolve_operating_point", "inputs": inputs,
        "reference_inputs": {
            "label_stems": {"calibration": {"path": str(tmp_path), "stems": universe}},
            "stated_values": {"split_manifest_dir": str(tmp_path / "m")},
        },
        "calibration_stems": universe,
    }
    monkeypatch.setattr(calibration_pipeline, "calibrate_operating_point",
                        lambda *a, **k: (bundle, dh, 0, evidence))
    _patch_build_predictor(monkeypatch, predictor_mod, _CalStub)
    monkeypatch.chdir(tmp_path)

    image_paths = []
    for stem in inf_stems:
        p = tmp_path / f"{stem}.png"
        Image.new("RGB", (100, 100)).save(p)
        image_paths.append(str(p))
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")

    kwargs = dict(calibration_labels_dir=str(tmp_path), split_manifest_dir=str(tmp_path / "m"))
    if give_calibration_images_dir:
        kwargs["calibration_images_dir"] = str(tmp_path)

    r = run_inference_verified(
        str(ckpt), image_paths=image_paths, images_dir=str(tmp_path), device="cpu",
        tile=False, trait="bud_opening", **kwargs)

    assert r["cross_dataset_check"] == expected_check
    assert r["validated"] is True
    assert not any("inherited across a different dataset" in i for i in r["shippable_issues"])


def test_manifest_calibration_reports_its_exclusion_counts_on_the_response(tmp_path, monkeypatch):
    """A manifest-restricted calibration answers how many present stems it left out of its
    universe, the train side's, the val side's and the unassigned ones, as counts beside the
    incomplete attribute count, so the caller learns the exclusions without opening the persisted
    evidence."""
    import tcip_mcp.pipelines.calibration as calibration_pipeline
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tests._verified_checkpoint_fixtures import run_inference_verified

    from PIL import Image

    cal, hold = _good_dense_cal_holdout()
    inputs = {"tiled": False, "dataset_hash": "H", "calibration_records": cal,
              "holdout_records": hold, "staged_conf_floor": 0.01}
    bundle = resolve_operating_point("bud_opening", experiment_id=None, **inputs)
    evidence = {
        "resolver": "resolve_operating_point", "inputs": inputs,
        "reference_inputs": {
            "label_stems": {"calibration": {"path": str(tmp_path), "stems": ["a"]}},
            "stated_values": {"split_manifest_dir": str(tmp_path / "m")},
        },
        "calibration_stems": ["a"],
        "excluded": {"excluded_training_stems": ["b", "c"], "excluded_validation_stems": ["e"],
                    "excluded_unassigned_stems": ["d"]},
    }
    monkeypatch.setattr(calibration_pipeline, "calibrate_operating_point",
                        lambda *a, **k: (bundle, "H", 0, evidence))
    _patch_build_predictor(monkeypatch, predictor_mod, _CalStub)
    monkeypatch.chdir(tmp_path)

    img = tmp_path / "a.png"
    Image.new("RGB", (100, 100)).save(img)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    r = run_inference_verified(
        str(ckpt), image_paths=[str(img)], images_dir=str(tmp_path), device="cpu",
        tile=False, trait="bud_opening", calibration_labels_dir=str(tmp_path),
        split_manifest_dir=str(tmp_path / "m"))

    assert r["n_excluded_training_stems"] == 2
    assert r["n_excluded_validation_stems"] == 1
    assert r["n_excluded_unassigned_stems"] == 1
    assert r["n_excluded_incomplete_attribute"] == 0


def test_unlabeled_target_is_not_comparable_but_shippable(tmp_path, monkeypatch):
    """An unlabeled inference target has no GT hash to compare: record it as such and still ship when
    the held-out calibration passed (no validated/shippable_issues contradiction)."""
    import tcip_mcp.pipelines.calibration as calibration_pipeline
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tests._verified_checkpoint_fixtures import run_inference_verified

    # tiled=False: matches the real verified-pass call below (tile=False).
    _stand_in_calibration(monkeypatch, calibration_pipeline, tmp_path, tiled=False)
    _patch_build_predictor(monkeypatch, predictor_mod, _CalStub)
    monkeypatch.chdir(tmp_path)

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    r = run_inference_verified(str(ckpt), image_paths=[_one_image(tmp_path)], images_dir=str(tmp_path),
                               device="cpu", tile=False, trait="bud_opening",
                               calibration_labels_dir=str(tmp_path))  # no labels beside the image
    assert r["cross_dataset_check"] == "not-comparable-unlabeled-target"
    assert r["shippable_issues"] == []
    assert r["validated"] is True


def test_calibration_follows_delivery_tile_regime(tmp_path, monkeypatch):
    """Regime lock: calibration must run the same tiled predictor path the delivery uses, not an
    untiled model forward, else the conf is validated in a different regime than it ships through."""
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

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
                                  [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], 128, 128)

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
                          tile_batch_size=96, global_nms_iou=None, postprocess="nms",
                          tile_resize=None):
            calls.append({"tile": tile, "tile_size": tile_size, "overlap": overlap,
                          "tile_resize": tile_resize})
            # tiled finds two boxes/img; untiled finds none, so a sweep over untiled records differs.
            boxes = [[10, 10, 40, 40], [100, 100, 130, 130]] if tile else []
            scores = [0.9, 0.6] if tile else []
            labels = [1, 1] if tile else []
            return [{"image": p, "width": 128, "height": 128, "boxes": boxes,
                     "scores": scores, "labels": labels, "count": len(boxes)} for p in paths]

    _patch_build_predictor(monkeypatch, predictor_mod, _RegimeStub)
    monkeypatch.chdir(tmp_path)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")

    run_inference(str(ckpt), images_dir=str(images_dir), device="cpu", tile=True,
                  trait="bud_opening", calibration_labels_dir=str(labels_dir))
    # Every predictor pass, both calibration splits and the delivery pass, ran tiled at the same
    # geometry, not an untiled calibration sweep.
    assert len(calls) >= 3
    assert all(c["tile"] is True for c in calls)
    assert all(c["tile_size"] == 64 for c in calls)
    assert len({c["overlap"] for c in calls}) == 1
    # Including any train-time resize: calibrating at a geometry the delivery pass does not run at
    # would resolve the conf against a scale the shipped count never sees.
    assert len({c["tile_resize"] for c in calls}) == 1


def test_calibrated_run_refuses_when_tile_size_has_no_real_basis(tmp_path, monkeypatch):
    """A checkpoint with no persisted geometry has no real basis to tile at: an explicit tile=True
    with no explicit tile_size refuses before the (expensive) calibration pass ever runs, never
    silently calibrates (or ships) against a fabricated scale."""
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

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
                                  [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], 128, 128)

    class _NoGeometryStub:
        def __init__(self):
            from types import SimpleNamespace
            self.model = SimpleNamespace(score_thresh=0.5, nms_thresh=0.5, detections_per_img=100)
            self.device = "cpu"
            self.score_threshold = 0.5
            self.train_tile_size = None
            self.train_overlap = None

        def predict_batch(self, paths, **kw):
            raise AssertionError("must refuse before any predictor pass, calibration included")

    _patch_build_predictor(monkeypatch, predictor_mod, _NoGeometryStub)
    monkeypatch.chdir(tmp_path)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")

    r = run_inference(str(ckpt), images_dir=str(images_dir), device="cpu", tile=True,
                      trait="bud_opening", calibration_labels_dir=str(labels_dir))
    assert "error" in r
    assert "tile_size" in r["error"]


def test_run_inference_validated_from_bundle(tmp_path, monkeypatch, seed_bud_trait_spec):
    import tcip_mcp.model_registry as model_registry_mod
    import tcip_mcp.tools.inference_tools as itools
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    from tests._binding_fixtures import calibrated_run_fields

    img = _one_image(tmp_path)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    sha = "the-checkpoint-behind-this-bucket"

    def _fake_run_inference_verified(*a, **kw):
        return {
            "results": [{"image": img, "width": 100, "height": 100,
                         "boxes": [], "scores": [], "labels": [], "count": 0}],
            "checkpoint_sha256": sha,
            **calibrated_run_fields(labels_dir=tmp_path, tiled=False, checkpoint_sha256=sha),
        }

    monkeypatch.setattr(itools, "_run_inference_verified", _fake_run_inference_verified)
    monkeypatch.setattr(model_registry_mod, "load_registered_checkpoint",
                        lambda *a, **kw: _stub_checkpoint(str(ckpt)))
    out_dir = tmp_path / "dataset" / "predictions" / "baseline" / "2026-01-01"
    r = itools.run_inference(str(ckpt), str(tmp_path), output_dir=str(out_dir), trait="bud_opening",
                             calibration_labels_dir=str(tmp_path))
    assert "error" not in r, r
    stamp = read_operating_point_sidecar(out_dir)
    assert stamp["validated"] is True                       # not hardcoded False
    assert r["validated"] is True and r["conf_source"] == "calibration"


def _deliver_per_image_counts_over(monkeypatch, tmp_path, op, *, validated, captured=None):
    """Run deliver_per_image_counts against a stubbed run_inference returning ``op``/``validated``.

    The dimension under test here is which reference the conf param itself recorded. With no
    ``predictions_dir`` these counts rest on no bucket anyone can re-read, and this door takes no
    acknowledgement, so the CSV itself always refuses; the refusal still carries the operating
    point and the run's own narrowed conf reference for a caller to inspect.
    """
    import tcip_mcp.model_registry as model_registry_mod
    import tcip_mcp.tools.inference_tools as itools

    def _fake_run_inference_verified(*a, **kw):
        if captured is not None:
            captured.update(kw)
        return {
            "results": [{"image": "a.png", "count": 3}],
            "image_count": 1, "total_detections": 3,
            "operating_point": op, "validated": validated, "conf_source": "calibration",
        }

    from tests import _operationalization_fixtures as fx

    fx.seed_confirmed_count(tmp_path)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    monkeypatch.setattr(itools, "_run_inference_verified", _fake_run_inference_verified)
    monkeypatch.setattr(model_registry_mod, "load_registered_checkpoint",
                        lambda *a, **kw: _stub_checkpoint(str(ckpt)))
    return itools.deliver_per_image_counts(str(ckpt), str(tmp_path), str(tmp_path / "o.csv"),
                                  trait=fx.COUNT_TRAIT,
                                  calibration_labels_dir=str(tmp_path))


def test_deliver_per_image_counts_carries_operating_point(tmp_path, monkeypatch):
    """No predictions_dir means no bucket to back the count, so this door's own no-acknowledgement
    rule refuses the CSV outright; the refusal still threads the calibration call through and
    reports the run's own operating point and narrowed conf reference."""
    captured: dict = {}
    op = {"conf": {"value": 0.6, "validated_against": "held_out_annotations"}}
    r = _deliver_per_image_counts_over(monkeypatch, tmp_path, op, validated=True, captured=captured)
    from tests import _operationalization_fixtures as fx

    assert captured["trait"] == fx.COUNT_TRAIT              # calibration threaded through
    assert captured["calibration_labels_dir"] == str(tmp_path)
    assert "error" in r
    assert r["validated"] is False
    assert r["operating_point"] == op
    assert r["run_conf_validated_against"] == "held_out_annotations"


def test_deliver_per_image_counts_never_launders_a_bare_validated_bool_into_a_reference(tmp_path, monkeypatch):
    """The count CSV door reads the reference conf itself recorded, never the run's bare bool.

    A run whose conf param records no (or a wrong-kind) reference is unvalidated for this dimension
    even when the run's overall ``validated`` flag is true: promoting the bool to
    ``held_out_annotations`` would be a laundering path (the same shape resolution._sidecar_reference
    guards against).
    """
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, VALIDATED_PHYSICAL_MEASUREMENT

    for conf_prov in ({"value": 0.6},                                        # no reference at all
                      {"value": 0.6, "validated_against": VALIDATED_FALSE},
                      {"value": 0.6, "validated_against": "make-believe"},
                      # a real reference, but for the wrong validation kind (physical, not annotations)
                      {"value": 0.6, "validated_against": VALIDATED_PHYSICAL_MEASUREMENT}):
        r = _deliver_per_image_counts_over(monkeypatch, tmp_path, {"conf": conf_prov}, validated=True)
        assert "error" in r, conf_prov
        assert r["operating_point_validated"] == VALIDATED_FALSE
        assert r["validated"] is False
    # A real annotations reference is still reported under its own name even though this door
    # takes no acknowledgement and no bucket backs the count, so the CSV-facing column floors too.
    ok = _deliver_per_image_counts_over(
        monkeypatch, tmp_path,
        {"conf": {"value": 0.6, "validated_against": "reviewer_confirmed_annotations"}},
        validated=True)
    assert "error" in ok
    assert ok["operating_point_validated"] == VALIDATED_FALSE
    assert ok["run_conf_validated_against"] == "reviewer_confirmed_annotations"
