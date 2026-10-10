"""Detection measurement-integrity coverage: val-loss over negatives, standard+operating mAP,
tiled evaluation, and the tile geometry a pass derives from its checkpoint.

Kept in one file so the audit's measurement-integrity locks live together. Vision/state
tests here are pure-Python or monkeypatched (no GPU) so they stay xdist-safe.
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tests._producer_fixtures import checkpoint_admission

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.execution import Stated  # noqa: E402
from tcip_mcp.pipelines.model_build import CONFIG_KEY  # noqa: E402
from tests._predictor_fixtures import StubPredictor, install  # noqa: E402
from tests._producer_fixtures import label_image, seed_bud_images  # noqa: E402
from tests._verified_checkpoint_fixtures import (  # noqa: E402
    SAMPLE_DETECTOR_PASS, SCOPED_DATA, project_checkpoint, verified_checkpoint,
)
from tcip_mcp.pipelines.training.evaluation import evaluate  # noqa: E402
from tests._training_values import VALIDATION_CONF  # noqa: E402

_DIMS = {"in_chans": 3, "num_classes": 1}
"""What a one-subject detector over three-band sources is built at."""
DETECTOR_PASS = Stated(**SAMPLE_DETECTOR_PASS)
"""The execution values an evaluation of a detector here states."""


# ── detection val-loss includes all-negative images ──────────────────────

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
        targets.append({"boxes": boxes, "labels": torch.ones((n,), dtype=torch.long),
                        "iscrowd": torch.zeros((n,), dtype=torch.long)})
    return images, targets


def test_val_loss_forwards_all_negative_images():
    stub = _StubDetector()
    model = _StubModel(stub)
    loader = [_det_batch([1, 0]), _det_batch([0])]  # mixed batch, then all-negative batch
    evaluate(model, loader, torch.device("cpu"), "detection", dims=_DIMS,
             conf_threshold=VALIDATION_CONF)
    # Both batches forwarded through the detector (full batch incl. negatives), not just foreground.
    assert stub.calls == [(2, 4), (1, 0)]


def test_all_negative_only_loader_is_not_skipped():
    stub = _StubDetector()
    model = _StubModel(stub)
    loader = [_det_batch([0, 0])]  # nothing but negatives
    result = evaluate(model, loader, torch.device("cpu"), "detection", dims=_DIMS,
                      conf_threshold=VALIDATION_CONF)
    assert stub.calls == [(2, 0)]  # forwarded, not skipped
    assert result["loss"] == pytest.approx(2.5)  # finite, non-zero: negatives contribute loss


# ── tiled evaluation regimes ──────────────────────────────────────────────

def _capture_run_test_evaluation(monkeypatch):
    """Patch run_test_evaluation to record the built dataset + tiling instead of loading a model."""
    import tcip_mcp.pipelines.training.eval_runners as runners

    captured: dict = {}

    def _fake(pass_, loader, device, **kw):
        captured["ds"] = loader.dataset
        captured["tiling"] = kw.get("tiling")
        return {"eval_regime": "tile-level"}

    monkeypatch.setattr(runners, "run_test_evaluation", _fake)
    return captured


def test_run_id_reuses_training_tiling(tmp_path, monkeypatch):
    from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._verified_checkpoint_fixtures import finished_run

    images_dir = seed_bud_images(tmp_path / "images" / UNDATED_BUCKET)
    data = {**SCOPED_DATA, "tiling": {"enabled": True, "tile_size": 64, "sliver_frac": 0.5}}
    run_dir = finished_run(tmp_path, experiment_id="det-measure-tiled", data=data)

    captured = _capture_run_test_evaluation(monkeypatch)
    evaluate_model(tmp_path, run_dir.name, str(images_dir), stated=DETECTOR_PASS)
    assert isinstance(captured["ds"], TiledDetectionDataset)
    assert captured["ds"].num_samples > 3  # more tiles than the 3 source images
    from tcip_mcp.experiments import run_resolution

    assert captured["tiling"] == run_resolution(run_dir.name, project=tmp_path).data.tiling
    assert captured["tiling"].tile_size == 64


def test_evaluating_a_run_leaves_its_directory_byte_identical(tmp_path, monkeypatch):
    """An evaluation of a completed run by its id writes nothing: the run's files before and after
    are the same names holding the same bytes, no audit line is left for it, and the result comes
    back to the caller."""
    import tcip_store as ts
    from tcip_mcp.audit import audit_log_key
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._verified_checkpoint_fixtures import finished_run

    images_dir = seed_bud_images(tmp_path / "images" / UNDATED_BUCKET)
    run_dir = finished_run(tmp_path, experiment_id="det-evaluated")

    def snapshot() -> dict:
        return {str(p.relative_to(run_dir)): p.read_bytes()
                for p in sorted(run_dir.rglob("*")) if p.is_file()}

    before = snapshot()
    audit_before = list(ts.read_log(audit_log_key(tmp_path)).records)
    result = evaluate_model(tmp_path, run_dir.name, str(images_dir), stated=DETECTOR_PASS)

    assert "error" not in result, result
    assert snapshot() == before
    assert list(ts.read_log(audit_log_key(tmp_path)).records) == audit_before


def test_explicit_checkpoint_stays_untiled(tmp_path, monkeypatch):
    from tcip_mcp.pipelines.data.datasets import DetectionDataset, TiledDetectionDataset
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    images_dir = seed_bud_images(tmp_path / "images" / UNDATED_BUCKET)
    ckpt = registered_checkpoint(tmp_path)

    captured = _capture_run_test_evaluation(monkeypatch)
    evaluate_model(tmp_path, ckpt, str(images_dir), stated=DETECTOR_PASS)
    assert isinstance(captured["ds"], DetectionDataset)
    assert not isinstance(captured["ds"], TiledDetectionDataset)
    assert captured["tiling"] is None


def test_evaluate_model_reads_its_loader_at_the_checkpoints_own_width(tmp_path, monkeypatch):
    """The door measures at the width the checkpoint reads at, and a checkpoint recording none
    stops it where the predictor refuses rather than sizing the loader off the references."""
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._verified_checkpoint_fixtures import ONE_BAND_DETECTOR, registered_checkpoint

    images_dir = seed_bud_images(tmp_path / "images" / UNDATED_BUCKET)  # three-band sources
    scope = {"subject": "bud"}
    ckpt = registered_checkpoint(tmp_path, model_source=ONE_BAND_DETECTOR,
                                 data={"num_channels": 1, "scope": scope})

    captured = _capture_run_test_evaluation(monkeypatch)
    evaluate_model(tmp_path, ckpt, str(images_dir), stated=DETECTOR_PASS)
    assert captured["ds"].expected_channels == 1

    from tcip_mcp.tools.model_tools import register_model

    # A checkpoint whose own data section records no band count, written past the producer.
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    del payload[CONFIG_KEY]["data"]["num_channels"]
    unstated = tmp_path / "unstated.pt"
    torch.save(payload, str(unstated))
    assert "error" not in register_model(tmp_path, name="unstated-width",
                                         checkpoint_path=str(unstated))

    r = evaluate_model(tmp_path, str(unstated), str(images_dir), stated=DETECTOR_PASS)

    assert "no band count" in r["error"], r


def _detections_as_ground_truth(checkpoint, images_dir, *, subject: str, limit: int = 20):
    """Write each image's own strongest detections back as its ground truth, so a metric over this
    fixture is sensitive to which detections the model is allowed to emit. The registered
    ``checkpoint``'s model, built from its own spec, is read with its score floor removed so that
    it returns detections to write back.
    """
    from tcip_annotation.state import Annotation, BBox

    from tcip_mcp.pipelines.image_utils import load_image, pil_to_tensor
    from tcip_mcp.pipelines.model_build import (
        STATE_DICT_KEY, build_from_model_source, recorded_model_dims,
    )
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point

    model = build_from_model_source(checkpoint.spec.model_source, checkpoint.layout,
                                    recorded_model_dims(checkpoint.spec))
    model.load_state_dict(checkpoint.payload[STATE_DICT_KEY])
    model.eval()
    set_detector_operating_point(model, score_thresh=0.0)
    for image in sorted(Path(images_dir).glob("*.png")):
        tensor = pil_to_tensor(load_image(image, 3))
        with torch.no_grad():
            boxes = model([tensor])[0]["boxes"].tolist()
        w, h = int(tensor.shape[-1]), int(tensor.shape[-2])
        label_image(
            image,
            [Annotation(subject=subject, geometry=BBox(*(float(v) for v in box)))
             for box in boxes[:limit] if box[2] - box[0] > 1 and box[3] - box[1] > 1],
            w, h)


def test_the_scored_model_is_the_one_the_door_already_built(tmp_path, monkeypatch):
    """One build per evaluation, and the execution it reports is the one its model ran under.

    The door builds the checkpoint to read the width it measures at, and the runner scores through
    that same module governed by the execution record it reports: a separately built copy of the
    same checkpoint, set to that record's conf and cap, answers with the same numbers over the
    same loader. This checkpoint's builder declares a score floor above what it scores at, so a
    run that scored at the builder's floor while reporting the record's would report numbers
    this record does not produce, which the last assertion measures rather than assumes.
    """
    from PIL import Image

    import tcip_mcp.pipelines.inference.generic_predictor as generic_predictor
    import tcip_mcp.pipelines.model_build as model_build
    import tcip_mcp.pipelines.training.eval_runners as runners
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.model_build import STATE_DICT_KEY
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._verified_checkpoint_fixtures import built_detector, registered_checkpoint

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    for i in range(2):
        Image.new("RGB", (128, 128), color=(120, 120, 120)).save(images_dir / f"img{i}.png")
    declares_its_point = built_detector(box_score_thresh=0.6)
    # This seed's weights score just above 0.5, so the declared floor of 0.6 excludes every
    # detection and a substituted floor of 0.5 or 0.0 would not.
    ckpt = registered_checkpoint(tmp_path, model_source=declares_its_point, seed=0)
    verified = load_registered_checkpoint(ckpt, project=tmp_path)
    _detections_as_ground_truth(verified, images_dir, subject="bud")

    build = model_build.build_from_model_source
    builds = []

    def _spy(source, layout, dims):
        builds.append(source)
        return build(source, layout, dims)

    monkeypatch.setattr(model_build, "build_from_model_source", _spy)
    monkeypatch.setattr(generic_predictor, "build_from_model_source", _spy)

    recorded: dict = {}
    run_test_evaluation = runners.run_test_evaluation

    def _record(pass_, loader, device, **kw):
        recorded.update(loader=loader, device=device, task=pass_.checkpoint.task, kw=kw)
        return run_test_evaluation(pass_, loader, device, **kw)

    monkeypatch.setattr(runners, "run_test_evaluation", _record)

    # Both routes run from one seed: the loss pass samples proposals, so two unseeded passes over
    # the same weights differ in that one metric by more than float noise.
    torch.manual_seed(777)
    measured = evaluate_model(tmp_path, ckpt, str(images_dir), stated=DETECTOR_PASS)
    assert "error" not in measured, measured
    assert len(builds) == 1

    torch.manual_seed(777)
    dims = model_build.recorded_model_dims(verified.spec)
    independent_model = build(verified.spec.model_source, verified.layout, dims)
    independent_model.load_state_dict(verified.payload[STATE_DICT_KEY])
    independent_model.to(recorded["device"])
    kw = recorded["kw"]

    def _independent() -> dict:
        return evaluate(
            independent_model, recorded["loader"], recorded["device"], recorded["task"],
            dims=dims, conf_threshold=measured["execution"]["conf"],
            iou_threshold=kw["iou_threshold"], trait=kw["trait"])

    set_detector_operating_point(independent_model, score_thresh=measured["execution"]["conf"],
                                 detections_per_img=measured["execution"]["max_dets"])
    independent = _independent()
    assert independent
    for key, value in independent.items():
        # Two forward passes over the same weights are equal to float noise, not bit for bit.
        assert measured[key] == (pytest.approx(value, rel=1e-4)
                                 if isinstance(value, (int, float)) else value), key

    # What the builder's own floor reports instead, so the equality above is evidence about this
    # fixture rather than a comparison nothing could separate.
    set_detector_operating_point(
        independent_model, score_thresh=declares_its_point["builder_kwargs"]["box_score_thresh"])
    assert _independent()["map50"] != measured["map50"]


def test_explicit_tiling_override_on_checkpoint(tmp_path, monkeypatch):
    from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset
    from tcip_mcp.pipelines.schemas import TilingSpec
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    images_dir = seed_bud_images(tmp_path / "images" / UNDATED_BUCKET)
    ckpt = registered_checkpoint(tmp_path)

    captured = _capture_run_test_evaluation(monkeypatch)
    evaluate_model(tmp_path, ckpt, str(images_dir), stated=DETECTOR_PASS,
                   tiling=TilingSpec.model_validate(
                       {"enabled": True, "tile_size": 64, "sliver_frac": 0.5}))
    assert isinstance(captured["ds"], TiledDetectionDataset)


def _sliced_stub(boxes, **training) -> StubPredictor:
    """A three-band detector answering ``boxes`` at 0.9 on a 128px frame, trained at
    ``training``'s geometry."""
    return StubPredictor(task="detection", in_chans=3, width=128, height=128, boxes=boxes,
                         scores=(0.9,) * len(boxes), **training)


def _full_frame(tmp_path, monkeypatch, stub, annotations, *, checkpoint=None, **stated) -> dict:
    """``run_full_frame_evaluation`` of one 128px frame labeled with ``annotations`` through
    ``stub``, at the sample detector values unless ``stated`` names its own."""
    from tcip_mcp.pipelines.training.eval_runners import run_full_frame_evaluation
    from tests._producer_fixtures import blank_image
    from tests._verified_checkpoint_fixtures import SAMPLE_DETECTOR_PASS

    image = blank_image(tmp_path, "a.png", (128, 128))
    label_image(image, annotations, 128, 128, keep_empty=True)
    images_dir = image.parent
    install(monkeypatch, stub)
    checkpoint = checkpoint or verified_checkpoint(tmp_path)
    return run_full_frame_evaluation(
        checkpoint, checkpoint_admission(checkpoint, images_dir),
        stated=Stated(**{**SAMPLE_DETECTOR_PASS, **stated}))


def test_full_frame_counts_straddling_object_once(tmp_path, monkeypatch):
    from tcip_annotation.state import Annotation, BBox

    # An object straddling the x=64 tile seam; this stub carries no persisted geometry, so the
    # caller states it.
    r = _full_frame(tmp_path, monkeypatch, _sliced_stub([[54, 54, 74, 74]]),
                    [Annotation(subject="bud", geometry=BBox(54, 54, 74, 74))],
                    tile_size=64, overlap=0.2)

    assert r["eval_regime"] == "full-frame-tiled-inference"
    # counted once against un-fragmented full-frame GT (tile-level would split/duplicate it)
    assert r["tp"] == 1 and r["fp"] == 0 and r["fn"] == 0


def test_full_frame_scores_a_detection_in_a_crowd_region_as_neither(tmp_path, monkeypatch):
    """The delivery-grade gate reads each record's crowd flag off the loader's target: a
    detection inside a crowd region is neither a true nor a false positive."""
    from tcip_annotation.state import Annotation, BBox

    r = _full_frame(tmp_path, monkeypatch,
                    _sliced_stub([[10, 10, 30, 30], [60, 60, 120, 120]]),
                    [Annotation(subject="bud", geometry=BBox(10, 10, 30, 30)),
                     Annotation(subject="bud", geometry=BBox(60, 60, 120, 120), iscrowd=True)],
                    tile_size=64, overlap=0.2)

    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 0)


def test_full_frame_reads_each_ground_truth_box_on_the_stored_grid(tmp_path, monkeypatch):
    """The delivery-grade gate scores the box its document holds, on the stored two-decimal grid,
    never the float arithmetic of the loader's corners."""
    import tcip_mcp.pipelines.training.evaluation as evaluation
    from tcip_annotation.state import Annotation, BBox

    scored: list = []
    real_record = evaluation.image_record

    def recording(w, h, gt, dt, **kw):
        scored.append((gt, dt))
        return real_record(w, h, gt, dt, **kw)

    monkeypatch.setattr(evaluation, "image_record", recording)
    bur = {"num_channels": 3, "scope": {"subject": "bur"}}
    _full_frame(tmp_path, monkeypatch, _sliced_stub([[10.1, 10.1, 40.3, 30.3]]),
                [Annotation(subject="bur", geometry=BBox(10.1, 10.1, 40.3, 30.3))],
                checkpoint=verified_checkpoint(tmp_path, data=bur), tile_size=64, overlap=0.2)

    assert [g["bbox"] for gt, _ in scored for g in gt] == [[10.1, 10.1, 30.2, 20.2]]
    assert [d["bbox"] for _, dt in scored for d in dt] == [[10.1, 10.1, 30.2, 20.2]]


# ── the tile geometry a pass derives from its checkpoint ──────────────────

def _geometry_stub(*, train_tile_size=None, train_overlap=None) -> StubPredictor:
    """A predictor carrying the given persisted training geometry, answering empty results."""
    return StubPredictor(boxes=(), scores=(), task="detection", in_chans=3,
                         train_tile_size=train_tile_size, train_overlap=train_overlap)


def _pass_over(tmp_path, monkeypatch, stub, **stated):
    """The pass a registered checkpoint prepares over one image with ``stub`` as its predictor."""
    from PIL import Image

    from tests._verified_checkpoint_fixtures import predicted_over

    images_dir = tmp_path / "one"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100)).save(images_dir / "a.png")
    install(monkeypatch, stub)
    return predicted_over(tmp_path, project_checkpoint(tmp_path), str(images_dir), device="cpu",
                          **stated)


def test_a_tiled_pass_derives_its_tile_edge_from_the_checkpoint(tmp_path, monkeypatch):
    stub = _geometry_stub(train_tile_size=224, train_overlap=0.1)

    p, _results = _pass_over(tmp_path, monkeypatch, stub, tile=True)

    assert (p.execution.tile_size, p.execution.sources["tile_size"]) == (224, "derived")
    assert p.execution.overlap == pytest.approx(0.1)
    assert stub.executions == [p.execution]


def test_a_tiled_pass_with_no_basis_for_its_edge_refuses_naming_it(tmp_path, monkeypatch):
    """A checkpoint with no persisted training tile geometry has no real basis to tile at: a tiled
    pass with no stated edge refuses, naming the missing basis, never fabricating a scale."""
    from tcip_mcp.pipelines.execution import ExecutionRefusedError

    with pytest.raises(ExecutionRefusedError, match="tile_size"):
        _pass_over(tmp_path, monkeypatch, _geometry_stub(), tile=True)


def test_a_stated_edge_agreeing_with_the_checkpoint_is_recorded_as_stated(tmp_path, monkeypatch):
    p, _results = _pass_over(tmp_path, monkeypatch, _geometry_stub(train_tile_size=224), tile=True,
                             tile_size=224)

    assert (p.execution.tile_size, p.execution.sources["tile_size"]) == (224, "explicit")


def test_a_stated_edge_contradicting_the_checkpoint_refuses_naming_both(tmp_path, monkeypatch):
    from tcip_mcp.pipelines.execution import ExecutionRefusedError

    with pytest.raises(ExecutionRefusedError) as exc_info:
        _pass_over(tmp_path, monkeypatch, _geometry_stub(train_tile_size=224), tile=True,
                   tile_size=512)
    assert "512" in str(exc_info.value) and "224" in str(exc_info.value)


# ── the delivery-grade evaluation resolves tile geometry the same way ──────

def test_the_evaluation_refuses_an_unresolvable_tile_geometry(tmp_path, monkeypatch):
    """A checkpoint with no persisted tiling and no stated edge refuses the delivery-grade
    evaluation rather than silently scoring it at an ungrounded scale."""
    from tcip_mcp.pipelines.execution import ExecutionRefusedError

    with pytest.raises(ExecutionRefusedError, match="tile_size"):
        _full_frame(tmp_path, monkeypatch, _sliced_stub([]), [])


def test_the_evaluation_derives_tile_geometry_from_the_checkpoint(tmp_path, monkeypatch):
    """The checkpoint's own persisted training geometry governs the evaluation instead of an
    arbitrary fixed scale."""
    from tcip_annotation.state import Annotation, BBox

    stub = _sliced_stub([], train_tile_size=224, train_overlap=0.1)

    r = _full_frame(tmp_path, monkeypatch, stub,
                    [Annotation(subject="bud", geometry=BBox(10, 10, 30, 30))])

    assert [(e.tile_size, e.overlap) for e in stub.executions] == [(224, pytest.approx(0.1))]
    assert (r["execution"]["tile_size"], r["execution"]["sources"]["tile_size"]) == (224,
                                                                                      "derived")


def test_launch_training_persists_effective_tile_geometry(tmp_path, monkeypatch):
    """launch_training resolves the run before its child starts, so the effective tiling
    geometry is in the run's launch record when launch_training returns.

    Also pins the isolation of the training body: this monkeypatches
    ``generic_trainer.train`` in this process to raise if ever called. If the run executed in this
    same interpreter, that monkeypatch would poison it and the "status == completed" assertion
    below would fail, since the poisoned ``train`` would be the one actually invoked. Because the
    real subprocess re-imports fresh, the monkeypatch here has no effect and the run completes
    normally, proving the training body genuinely executes outside this process, not merely that
    the API still returns the right shape."""
    pytest.importorskip("torchvision")
    monkeypatch.chdir(tmp_path)
    import os

    from tcip_mcp.experiments import run_resolution
    import tcip_mcp.pipelines.training.generic_trainer as gt
    from tcip_mcp.tools import training_tools
    from tests._chain_fixtures import BLOB_BUILDER
    from tests._producer_fixtures import small_detection_config
    from tests._verified_checkpoint_fixtures import run_to_end

    def _poison_train(*a, **k):
        raise AssertionError(
            "generic_trainer.train ran inside the launching process: subprocess isolation broken")

    monkeypatch.setattr(gt, "train", _poison_train)

    images_dir = seed_bud_images(tmp_path / "images" / UNDATED_BUCKET, n=2)

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})

    cfg = small_detection_config(images_dir, BLOB_BUILDER)
    # no tile_size: the effective default must be persisted
    cfg["data"]["tiling"] = {"enabled": True, "sliver_frac": 0.5}
    res = training_tools.launch_training(tmp_path, cfg, actor=None)
    assert res["pid"] != os.getpid()  # a different OS process, not this one
    eid = res["experiment_id"]

    tiling = run_resolution(eid, project=tmp_path).data.tiling
    assert tiling is not None
    assert tiling.tile_size == 224  # TiledDetectionDataset default
    assert tiling.overlap == pytest.approx(0.2)

    # "completed", not any terminal state: a child run inside this process would hit _poison_train.
    assert run_to_end(tmp_path, eid)["state"] == "completed"
