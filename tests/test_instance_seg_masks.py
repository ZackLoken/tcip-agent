"""The instance_seg measurement boundary: masks reach inference and export.

Locks that instance_seg's masks travel end to end instead of being silently dropped:
``check_model_contract`` requires them for instance_seg, ``GenericPredictor._format_detection``
carries them (soft, unbinarized), ``predict_tiled`` refuses loudly for instance_seg rather than
silently dropping masks, and ``write_predictions_json`` converts a mask to a real (possibly
multi-ring) ``Polygon`` via ``resolve_binarize_threshold``, never a second hardcoded threshold.

The refusal is a rail, so it must still admit valid work: the MCP doors resolve an unset ``tile``
to untiled for an instance_seg checkpoint (and refuse an explicit ``tile=True`` with a tool-level
``{"error": ...}``), and a caller that never reads masks opts out with ``require_masks=False`` and
gets ordinary boxes-only tiled inference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
cv2 = pytest.importorskip("cv2")

from tcip_mcp.pipelines.model_contract import check_model_contract  # noqa: E402
from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor  # noqa: E402
from tcip_mcp.pipelines.measurement.mask_geometry import mask_to_polygon_points  # noqa: E402
from tests import bespoke_models  # noqa: E402


# --------------------------------------------------------------------------
# model_contract: instance_seg requires masks in the eval output
# --------------------------------------------------------------------------

def test_contract_rejects_maskless_instance_seg_model():
    """A model that emits boxes/scores/labels but no masks must fail the instance_seg contract:
    it would otherwise pass as a detector while the platform's only sanctioned dimensional
    measurement can never reach it."""
    import torch.nn as nn

    class _BoxesOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, images, targets=None):
            if self.training:
                return {"loss": self.lin(torch.rand(1, 4)).sum()}
            return [{"boxes": torch.zeros((1, 4)), "scores": torch.ones((1,)),
                    "labels": torch.ones((1,), dtype=torch.int64)}]

    report = check_model_contract(_BoxesOnly(), "instance_seg", num_classes=1, img_size=64)
    assert report["ok"] is False
    assert any("masks" in i for i in report["issues"]), report["issues"]


def test_contract_accepts_real_masked_instance_seg_model():
    model = bespoke_models.build_bespoke_instance_seg(num_classes=1, min_size=64, max_size=128)
    report = check_model_contract(model, "instance_seg", num_classes=1, img_size=64)
    assert report["ok"], report["issues"]
    assert report["eval_output_type"] == "list[dict]"


def test_contract_detection_task_unaffected_by_mask_requirement():
    """Plain detection (no masks trained/expected) must not start requiring masks: a regression
    guard that the instance_seg-only requirement stays instance_seg-only."""
    model = bespoke_models.build_bespoke_detection(num_classes=1, min_size=64, max_size=128)
    report = check_model_contract(model, "detection", num_classes=1, img_size=64)
    assert report["ok"], report["issues"]


# --------------------------------------------------------------------------
# GenericPredictor._format_detection: soft masks carried, unbinarized
# --------------------------------------------------------------------------

def _bare_predictor(task: str, score_threshold: float = 0.5) -> GenericPredictor:
    """A GenericPredictor with no real checkpoint: __init__ is never called, only the attributes
    _format_detection/predict_tiled actually read are set. Avoids loading a real model just to
    unit-test output formatting / the tiled refusal, which both happen before any inference."""
    p = GenericPredictor.__new__(GenericPredictor)
    p.task = task
    p.score_threshold = score_threshold
    p.max_dets = None
    p.in_chans = 3
    return p


def test_format_detection_carries_soft_masks_for_instance_seg():
    p = _bare_predictor("instance_seg")
    outputs = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 2.0, 2.0]]),
        "scores": torch.tensor([0.9, 0.9]),
        "labels": torch.tensor([1, 1], dtype=torch.int64),
        "masks": torch.rand(2, 1, 8, 8),  # torchvision MaskRCNN shape: [N, 1, H, W], soft
    }
    result = p._format_detection(outputs, "img.jpg", 100, 100)
    assert "masks" in result
    assert len(result["masks"]) == 2
    # Squeezed to [H, W] per instance (the singleton channel dim dropped).
    assert np.asarray(result["masks"][0]).shape == (8, 8)
    # Kept soft, not binarized here (values other than exactly 0/1 must survive).
    flat = np.asarray(result["masks"]).ravel()
    assert not np.all((flat == 0.0) | (flat == 1.0))


def test_format_detection_detection_task_never_gains_masks_key():
    """Regression guard: plain detection must not pick up a masks key even if outputs somehow
    carried one (a maskless task never asks mask_geometry/export to do anything with masks)."""
    p = _bare_predictor("detection")
    outputs = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([1], dtype=torch.int64),
        "masks": torch.rand(1, 1, 8, 8),
    }
    result = p._format_detection(outputs, "img.jpg", 100, 100)
    assert "masks" not in result


def test_format_detection_keep_mask_filters_with_scores():
    """The score threshold that filters boxes/scores/labels must filter masks in lockstep."""
    p = _bare_predictor("instance_seg", score_threshold=0.5)
    outputs = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 2.0, 2.0]]),
        "scores": torch.tensor([0.9, 0.1]),  # second below threshold
        "labels": torch.tensor([1, 1], dtype=torch.int64),
        "masks": torch.stack([torch.ones(1, 8, 8), torch.zeros(1, 8, 8)]),
    }
    result = p._format_detection(outputs, "img.jpg", 100, 100)
    assert result["count"] == 1
    assert len(result["masks"]) == 1
    assert np.asarray(result["masks"][0]).max() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# predict_tiled: refuses loudly for instance_seg rather than silently
# dropping masks through cross-tile reconstruction/merge
# --------------------------------------------------------------------------

def test_predict_tiled_refuses_for_instance_seg():
    p = _bare_predictor("instance_seg")
    with pytest.raises(NotImplementedError, match="mask"):
        p.predict_tiled("does-not-need-to-exist.jpg")


def test_predict_tiled_detection_reaches_real_tiling_path(tmp_path):
    """Sharper version of the above: detection must reach real tiling logic (fail on a bad image
    path with an image-loading error, never NotImplementedError), proving the instance_seg refusal
    is scoped to instance_seg only."""
    p = _bare_predictor("detection")
    missing = tmp_path / "missing.jpg"
    with pytest.raises(Exception) as exc_info:
        p.predict_tiled(str(missing))
    assert not isinstance(exc_info.value, NotImplementedError)


def test_predict_tiled_error_names_the_caller_facing_routes():
    """The refusal must point at real alternatives a caller can act on: the untiled library calls,
    the two MCP doors' own ``tile=False`` parameter, and the boxes-only opt-out."""
    p = _bare_predictor("instance_seg")
    with pytest.raises(NotImplementedError) as exc_info:
        p.predict_tiled("does-not-need-to-exist.jpg")
    msg = str(exc_info.value)
    assert "run_inference(..., tile=False)" in msg
    assert "export_predictions(..., tile=False)" in msg
    assert "require_masks=False" in msg


def test_predict_tiled_require_masks_false_reaches_real_tiling_path_for_instance_seg(tmp_path):
    """``require_masks=False`` must not refuse: instance_seg gets as far as loading the image, the
    same real tiling logic detection reaches. Asserting the exact FileNotFoundError (not merely
    "not NotImplementedError") is what makes this fail rather than pass on a predictor whose
    signature has no such parameter."""
    p = _bare_predictor("instance_seg")
    with pytest.raises(FileNotFoundError):
        p.predict_tiled(str(tmp_path / "missing.jpg"), require_masks=False)


# --------------------------------------------------------------------------
# The rail admits valid work: a real Mask R-CNN checkpoint through the
# boxes-only opt-out, the two MCP doors, and the delivery-gating eval
# --------------------------------------------------------------------------

TILE = 64


@pytest.fixture(scope="module")
def instance_seg_ckpt(tmp_path_factory) -> str:
    """A real bespoke instance_seg (Mask R-CNN) checkpoint, built once and only read afterwards:
    these tests exercise reachable tool/eval paths, so a stub predictor would assume the very
    dispatch under test."""
    from tcip_mcp.pipelines.model_build import build_model

    model_source = {"builder": "tests.bespoke_models:build_bespoke_instance_seg",
                    "builder_kwargs": {"num_classes": 1, "min_size": TILE, "max_size": TILE * 2},
                    "task": "instance_seg"}
    model = build_model({"model_source": model_source})
    ckpt = tmp_path_factory.mktemp("k20_instance_seg") / "model_best.pt"
    torch.save({"model_source": model_source, "model_state_dict": model.state_dict()}, str(ckpt))
    return str(ckpt)


def _image(directory: Path, name: str = "img.png", size: int = 128) -> str:
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    p = directory / name
    Image.new("RGB", (size, size), (120, 120, 120)).save(p)
    return str(p)


def test_predict_tiled_require_masks_false_returns_boxes_only(instance_seg_ckpt, tmp_path):
    """The opt-out tiles normally and returns no masks key at all, never a partial one, while the
    same predictor's untiled path still carries masks (an opt-out, not a global downgrade)."""
    pred = GenericPredictor(instance_seg_ckpt, device="cpu", score_threshold=0.5)
    assert pred.task == "instance_seg"
    img = _image(tmp_path / "images")

    tiled = pred.predict_tiled(img, tile_size=TILE, overlap=0.2, require_masks=False)
    assert "masks" not in tiled
    assert {"boxes", "scores", "labels", "count", "tiles"} <= set(tiled)
    assert tiled["tiles"] >= 4  # 128px image at tile 64 -> a 2x2+ grid, i.e. it really tiled
    assert tiled["count"] == len(tiled["boxes"]) == len(tiled["scores"])

    assert "masks" in pred.predict(img)


def test_run_inference_instance_seg_unset_tile_runs_untiled(instance_seg_ckpt, tmp_path):
    """DEFAULT_TILED is True, so an unset ``tile`` must not send an instance_seg checkpoint into
    predict_tiled (that would surface NotImplementedError as a raw traceback out of the MCP tool).
    The door resolves to untiled instead, masks intact, and says so."""
    from tcip_mcp.tools.inference_tools import run_inference

    r = run_inference(instance_seg_ckpt, image_paths=[_image(tmp_path / "images")], device="cpu")
    assert "error" not in r
    assert r["tiled"] is False
    assert r["operating_point"]["tiled"]["value"] is False
    assert len(r["results"]) == 1
    assert "instance_seg" in r["warning"]


def test_run_inference_instance_seg_explicit_tile_true_returns_error(instance_seg_ckpt, tmp_path):
    """An explicit tile=True is a refusal in the tool's own ``{"error": ...}`` contract, naming the
    tool-level parameter, not a raised NotImplementedError about library internals."""
    from tcip_mcp.tools.inference_tools import run_inference

    r = run_inference(instance_seg_ckpt, image_paths=[_image(tmp_path / "images")], device="cpu",
                      tile=True, tile_size=TILE)
    assert "error" in r
    assert "tile=False" in r["error"]
    assert "results" not in r


def test_export_predictions_instance_seg_unset_tile_writes_untiled(instance_seg_ckpt, tmp_path):
    from tcip_mcp.tools.inference_tools import export_predictions

    images_dir = tmp_path / "images"
    _image(images_dir)
    r = export_predictions(instance_seg_ckpt, str(images_dir), str(tmp_path / "preds"),
                           device="cpu")
    assert "error" not in r
    assert r["operating_point"]["tiled"]["value"] is False
    assert (Path(r["output_dir"]) / "img.json").is_file()
    assert "instance_seg" in r["warning"]


def test_tabulate_counts_instance_seg_unset_tile_carries_warning(instance_seg_ckpt, tmp_path):
    """export_predictions surfaces the instance_seg forced-untiled warning in its response;
    tabulate_counts's own docstring documents the same behavior but its success return had no
    warning key, so a count CSV (the count is the phenotype) could ship with the regime change
    disclosed only in the server log."""
    from tcip_mcp.tools.inference_tools import tabulate_counts

    images_dir = tmp_path / "images"
    _image(images_dir)
    r = tabulate_counts(instance_seg_ckpt, str(images_dir), str(tmp_path / "counts.csv"),
                        device="cpu", acknowledge_unvalidated=True)
    assert "error" not in r
    assert "instance_seg" in r["warning"]


def test_export_predictions_stamps_mask_binarize_provenance_when_masks_present(instance_seg_ckpt, tmp_path):
    """The unvalidated mask-binarize threshold must not be stamped into Annotation.attributes
    (the domain trait namespace, which would pollute GT). It travels once, as a run constant,
    in operating_point.json, the same door tiled/tile_size/conf already use."""
    import json

    from tcip_mcp.tools.inference_tools import export_predictions

    images_dir = tmp_path / "images"
    _image(images_dir)
    out = tmp_path / "preds"
    r = export_predictions(instance_seg_ckpt, str(images_dir), str(out), device="cpu",
                           conf_threshold=0.0)  # force at least one (masked) detection
    assert "error" not in r
    op = json.loads((Path(r["output_dir"]) / "operating_point.json").read_text())
    assert op["mask_binarize"]["name"] == "mask_binarize_threshold"
    assert op["mask_binarize"]["requires_validation"] is True

    pred_json = json.loads((Path(r["output_dir"]) / "img.json").read_text())
    for ann in pred_json["annotations"]:
        assert ann.get("attributes", {}) == {}  # never per-annotation


def test_export_predictions_instance_seg_explicit_tile_true_returns_error(instance_seg_ckpt, tmp_path):
    from tcip_mcp.tools.inference_tools import export_predictions

    images_dir = tmp_path / "images"
    _image(images_dir)
    out = tmp_path / "preds"
    r = export_predictions(instance_seg_ckpt, str(images_dir), str(out), device="cpu",
                           tile=True, tile_size=TILE)
    assert "error" in r
    assert "tile=False" in r["error"]
    assert not out.exists()  # refused before anything was written


def test_run_full_frame_evaluation_tiled_instance_seg_scores_boxes(instance_seg_ckpt, tmp_path):
    """The delivery-gating eval never consumed masks: it reads boxes/scores/labels only, so a
    tile-trained Mask R-CNN must still evaluate here instead of crashing on the mask refusal."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.training.evaluation import run_full_frame_evaluation

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _image(images_dir, "a.png")
    labels_dir.mkdir(parents=True)
    json_io.write_annotations(str(labels_dir / "a.json"),
                              [Annotation(subject="catkin", geometry=BBox(54, 54, 74, 74))], 128, 128)

    r = run_full_frame_evaluation(instance_seg_ckpt, str(images_dir), str(labels_dir),
                                  str(tmp_path / "out"), subject="catkin",
                                  tile_size=TILE, overlap=0.2)
    assert r["eval_regime"] == "full-frame-tiled-inference"
    assert r["scored_images"] == 1
    assert r["n_gt"] == 1
    assert Path(r["results_path"]).is_file()
    # task records the checkpoint's actual producer task, not a hardcoded "detection". iou_type
    # stays "bbox" regardless of task: this gate always scores boxes only, by design (see the
    # docstring).
    assert r["task"] == "instance_seg"
    assert r["iou_type"] == "bbox"


# --------------------------------------------------------------------------
# mask_geometry.mask_to_polygon_points: every connected component, own ring
# --------------------------------------------------------------------------

def test_mask_to_polygon_single_blob_one_ring():
    m = np.zeros((32, 32), dtype=np.uint8)
    m[5:20, 5:20] = 1
    rings = mask_to_polygon_points(m)
    assert len(rings) == 1
    assert len(rings[0]) >= 3


def test_mask_to_polygon_two_disjoint_blobs_two_rings():
    """An occlusion-split mask (two disjoint regions) must return two rings, not one truncated
    to the largest."""
    m = np.zeros((64, 64), dtype=np.uint8)
    m[5:15, 5:15] = 1
    m[40:55, 40:55] = 1  # far enough away to be a separate connected component
    rings = mask_to_polygon_points(m)
    assert len(rings) == 2


def test_mask_to_polygon_empty_mask_no_rings():
    m = np.zeros((16, 16), dtype=np.uint8)
    assert mask_to_polygon_points(m) == []


# --------------------------------------------------------------------------
# export.py write_predictions_json: masks become a real (possibly
# multi-ring) Polygon; resolve_binarize_threshold is a real caller
# --------------------------------------------------------------------------

def test_export_single_component_mask_writes_polygon(tmp_path):
    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json
    from tcip_annotation import json_io
    from tcip_annotation.state import Polygon

    mask = np.zeros((32, 32), dtype=np.float32)
    mask[5:20, 5:20] = 0.9
    result = {
        "image": "img.jpg", "width": 32, "height": 32,
        "boxes": [[5.0, 5.0, 19.0, 19.0]], "scores": [0.9], "labels": [1],
        "masks": [mask.tolist()],
    }
    out = tmp_path / "img.json"
    write_predictions_json(str(out), result)
    anns = json_io.read_annotations(str(out))
    assert len(anns) == 1
    assert isinstance(anns[0].geometry, Polygon)
    assert len(anns[0].geometry.rings) == 1


def test_export_does_not_pollute_annotation_attributes_with_binarize_threshold(tmp_path):
    """Stamping the mask-binarize threshold into Annotation.attributes (the domain trait
    namespace, not a machine-provenance one) would let it survive into GT the moment a breeder
    accepts the prediction. attributes must stay empty; the threshold travels via
    mask_binarize_provenance() into the run's operating_point.json instead (see
    test_export_predictions_stamps_mask_binarize_provenance_when_masks_present)."""
    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json
    from tcip_annotation import json_io

    mask = np.zeros((32, 32), dtype=np.float32)
    mask[5:20, 5:20] = 0.9
    result = {
        "image": "img.jpg", "width": 32, "height": 32,
        "boxes": [[5.0, 5.0, 19.0, 19.0]], "scores": [0.9], "labels": [1],
        "masks": [mask.tolist()],
    }
    out = tmp_path / "img.json"
    write_predictions_json(str(out), result)
    anns = json_io.read_annotations(str(out))
    assert anns[0].attributes == {}


def test_mask_binarize_provenance_reports_the_unvalidated_default():
    from tcip_mcp.pipelines.postprocessing.export import mask_binarize_provenance

    prov = mask_binarize_provenance()
    assert prov["name"] == "mask_binarize_threshold"
    assert prov["value"] == pytest.approx(0.5)
    assert prov["requires_validation"] is True
    assert prov["validated_against"] == "false"


def test_export_multi_component_mask_writes_multi_ring_polygon(tmp_path):
    """An occlusion-split mask must export every region as its own ring in one Polygon, never
    silently truncated to the largest component and never downgraded to a BBox that would lose
    the shape entirely."""
    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json
    from tcip_annotation import json_io
    from tcip_annotation.state import Polygon

    mask = np.zeros((64, 64), dtype=np.float32)
    mask[5:15, 5:15] = 0.9
    mask[40:55, 40:55] = 0.9
    result = {
        "image": "img.jpg", "width": 64, "height": 64,
        "boxes": [[5.0, 5.0, 54.0, 54.0]], "scores": [0.9], "labels": [1],
        "masks": [mask.tolist()],
    }
    out = tmp_path / "img.json"
    write_predictions_json(str(out), result)
    anns = json_io.read_annotations(str(out))
    assert len(anns) == 1
    assert isinstance(anns[0].geometry, Polygon)
    assert len(anns[0].geometry.rings) == 2


def test_export_empty_mask_falls_back_to_bbox(tmp_path):
    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json
    from tcip_annotation import json_io
    from tcip_annotation.state import BBox

    mask = np.zeros((16, 16), dtype=np.float32)  # binarizes to nothing at the default threshold
    result = {
        "image": "img.jpg", "width": 16, "height": 16,
        "boxes": [[1.0, 1.0, 5.0, 5.0]], "scores": [0.9], "labels": [1],
        "masks": [mask.tolist()],
    }
    out = tmp_path / "img.json"
    write_predictions_json(str(out), result)
    anns = json_io.read_annotations(str(out))
    assert len(anns) == 1
    assert isinstance(anns[0].geometry, BBox)


def test_export_no_masks_key_writes_bbox_as_before():
    """Regression guard: a plain detection result (no masks key at all) must still export BBox
    as before; masks are additive, not a behavior change for non-instance_seg."""
    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json
    from tcip_annotation import json_io
    from tcip_annotation.state import BBox
    import tempfile
    import os

    result = {
        "image": "img.jpg", "width": 32, "height": 32,
        "boxes": [[1.0, 1.0, 5.0, 5.0]], "scores": [0.9], "labels": [1],
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        write_predictions_json(path, result)
        anns = json_io.read_annotations(path)
        assert len(anns) == 1
        assert isinstance(anns[0].geometry, BBox)
    finally:
        os.unlink(path)


def test_resolve_binarize_threshold_is_a_real_caller_of_export(monkeypatch, tmp_path):
    """resolve_binarize_threshold must be the actual threshold export uses, not a bare hardcoded
    0.5 reintroduced beside it. Proven by changing the threshold and observing the binarized mask
    (and therefore the exported polygon) change accordingly."""
    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json
    from tcip_annotation import json_io
    from tcip_annotation.state import Polygon, BBox

    # A mask whose values sit strictly between 0.3 and 0.6: binarizes to a real blob at threshold
    # 0.3, and to nothing at threshold 0.6.
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[5:20, 5:20] = 0.45
    result = {
        "image": "img.jpg", "width": 32, "height": 32,
        "boxes": [[5.0, 5.0, 19.0, 19.0]], "scores": [0.9], "labels": [1],
        "masks": [mask.tolist()],
    }

    import importlib
    # tcip_mcp.pipelines.measurement.mask_geometry as a package attribute resolves to the
    # re-exported `mask_geometry` function (measurement/__init__.py shadows the submodule name
    # with a same-named function); importlib.import_module bypasses that via sys.modules and
    # returns the real submodule, which is what _mask_geometry_for_export actually imports from.
    mg = importlib.import_module("tcip_mcp.pipelines.measurement.mask_geometry")

    real_resolve = mg.resolve_binarize_threshold

    out_low = tmp_path / "low.json"
    monkeypatch.setattr(mg, "resolve_binarize_threshold", lambda *a, **k: real_resolve(0.3))
    write_predictions_json(str(out_low), result)
    anns_low = json_io.read_annotations(str(out_low))

    out_high = tmp_path / "high.json"
    monkeypatch.setattr(mg, "resolve_binarize_threshold", lambda *a, **k: real_resolve(0.6))
    write_predictions_json(str(out_high), result)
    anns_high = json_io.read_annotations(str(out_high))

    assert isinstance(anns_low[0].geometry, Polygon)   # 0.45 >= 0.3 -> real blob
    assert isinstance(anns_high[0].geometry, BBox)      # 0.45 < 0.6 -> binarizes to nothing
