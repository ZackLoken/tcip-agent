"""The instance_seg measurement boundary: masks reach inference and export.

Locks that instance_seg's masks travel end to end instead of being silently dropped:
``check_model_contract`` requires them for instance_seg, the predictor's one detection record
carries them as SAHI polygons on the untiled and the sliced path alike (see
``tests/test_sliced_inference.py``), and ``write_predictions_json`` converts each to a real
(possibly multi-ring) ``Polygon``.

``require_masks=False`` is a boxes-only opt-out for a caller that never reads masks, tested here
alongside the default mask-carrying path.
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
from tests import bespoke_models  # noqa: E402

# What the smokes below synthesize their batch at, the shape a run resolves for itself.
_SMOKE_DIMS = {"in_chans": 3, "num_classes": 1, "img_size": 64}


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

    report = check_model_contract(_BoxesOnly(), "instance_seg", dims=_SMOKE_DIMS)
    assert report["ok"] is False
    assert any("masks" in i for i in report["issues"]), report["issues"]


def test_contract_accepts_real_masked_instance_seg_model():
    model = bespoke_models.build_bespoke_instance_seg(num_classes=1, min_size=64, max_size=128)
    report = check_model_contract(model, "instance_seg", dims=_SMOKE_DIMS)
    assert report["ok"], report["issues"]
    assert report["eval_output_type"] == "list[dict]"


def test_contract_detection_task_unaffected_by_mask_requirement():
    """Plain detection (no masks trained/expected) must not start requiring masks: a regression
    guard that the instance_seg-only requirement stays instance_seg-only."""
    model = bespoke_models.build_bespoke_detection(num_classes=1, min_size=64, max_size=128)
    report = check_model_contract(model, "detection", dims=_SMOKE_DIMS)
    assert report["ok"], report["issues"]


def _bare_predictor(task: str, score_threshold: float = 0.5) -> GenericPredictor:
    """A GenericPredictor with no real checkpoint: __init__ is never called, only the attributes
    predict_sliced reads before it decodes the source are set."""
    p = GenericPredictor.__new__(GenericPredictor)
    p.task = task
    p.score_threshold = score_threshold
    p.max_dets = None
    p.in_chans = 3
    return p


# predict_sliced: instance_seg and detection reach one slicing path, masks or not.

def _sliced_kwargs(require_masks: bool = True) -> dict:
    return dict(tile_size=TILE, overlap=0.2, postprocess="nms", cross_tile_nms=0.3,
                tile_batch_size=8, tile_resize=None, require_masks=require_masks)


def test_predict_sliced_instance_seg_reaches_real_slicing_path(tmp_path):
    """instance_seg must reach real slicing logic exactly like detection does (fail on a bad image
    path with an image-loading error), not take some separate maskless code path."""
    p = _bare_predictor("instance_seg")
    missing = tmp_path / "missing.jpg"
    with pytest.raises(FileNotFoundError):
        p.predict_sliced(str(missing), **_sliced_kwargs())


def test_predict_sliced_detection_reaches_real_slicing_path(tmp_path):
    """Same real slicing logic for plain detection, so the two tasks' tiled entry points don't
    silently diverge."""
    p = _bare_predictor("detection")
    missing = tmp_path / "missing.jpg"
    with pytest.raises(FileNotFoundError):
        p.predict_sliced(str(missing), **_sliced_kwargs())


def test_predict_sliced_require_masks_false_reaches_real_slicing_path_for_instance_seg(tmp_path):
    """``require_masks=False`` reaches the same real slicing logic as the default (masks-collecting)
    path: the boxes-only opt-out is a lighter-weight rail, not a different one."""
    p = _bare_predictor("instance_seg")
    with pytest.raises(FileNotFoundError):
        p.predict_sliced(str(tmp_path / "missing.jpg"), **_sliced_kwargs(require_masks=False))


# --------------------------------------------------------------------------
# The rail admits valid work: a real Mask R-CNN checkpoint through the
# boxes-only opt-out, the two MCP doors, and the delivery-gating eval
# --------------------------------------------------------------------------

TILE = 64


@pytest.fixture(scope="module")
def instance_seg_ckpt(tmp_path_factory) -> str:
    """A real bespoke instance_seg (Mask R-CNN) checkpoint, built once and only read afterwards:
    these tests exercise reachable tool/eval paths, so a stub predictor would assume the very
    dispatch under test. Stamps its own persisted training tile geometry (``config["data"]
    ["tiling"]``), matching a real tile-trained checkpoint, so the tests below that leave ``tile``
    unset genuinely exercise the tiled default derived from *this* checkpoint's own geometry, not a
    platform-wide fallback (an untrained-tiled checkpoint has no such basis and would derive
    untiled instead, see ``resolve_tile_geometry``)."""
    from tcip_mcp.pipelines.model_build import build_model

    model_source = {"builder": "tests.bespoke_models:build_bespoke_instance_seg",
                    "builder_kwargs": {"num_classes": 1, "in_chans": 3, "min_size": TILE,
                                       "max_size": TILE * 2},
                    "task": "instance_seg"}
    model = build_model({"model_source": model_source})
    ckpt = tmp_path_factory.mktemp("instance_seg_ckpt") / "model_best.pt"
    torch.save({
        "model_source": model_source, "model_state_dict": model.state_dict(),
        "config": {"model_source": model_source,
                  "data": {"tiling": {"tile_size": TILE, "overlap": 0.2}}},
    }, str(ckpt))
    return str(ckpt)


def _image(directory: Path, name: str = "img.png", size: int = 128) -> str:
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    p = directory / name
    Image.new("RGB", (size, size), (120, 120, 120)).save(p)
    return str(p)


def _register_instance_seg_ckpt(ckpt_path: str, project_root: Path) -> None:
    """Register the module-scoped checkpoint against one test's own pinned platform state root."""
    from tcip_mcp.tools.model_tools import register_model

    result = register_model(name="instance-seg-test-model", checkpoint_path=ckpt_path,
                            config={}, project_path=str(project_root))
    assert "error" not in result, result


def test_predict_sliced_require_masks_false_returns_boxes_only(instance_seg_ckpt, tmp_path):
    """The opt-out tiles normally and returns no masks key at all, never a partial one, while the
    same predictor's untiled path still carries masks (an opt-out, not a global downgrade)."""
    from tcip_mcp.model_registry import load_registered_checkpoint

    _register_instance_seg_ckpt(instance_seg_ckpt, tmp_path)
    checkpoint = load_registered_checkpoint(instance_seg_ckpt, project_path=str(tmp_path))
    pred = GenericPredictor(checkpoint, device="cpu", score_threshold=0.5)
    assert pred.task == "instance_seg"
    img = _image(tmp_path / "images")

    tiled = pred.predict_sliced(img, **_sliced_kwargs(require_masks=False))
    assert "masks" not in tiled
    assert {"boxes", "scores", "labels", "count", "tiles"} <= set(tiled)
    assert tiled["tiles"] >= 4  # 128px image at tile 64 -> a 2x2+ lattice, i.e. it really sliced
    assert tiled["count"] == len(tiled["boxes"]) == len(tiled["scores"])

    assert "masks" in pred.predict(img)


def test_run_inference_instance_seg_unset_tile_runs_tiled_with_masks(instance_seg_ckpt, tmp_path):
    """The fixture's own persisted training tile geometry derives an unset ``tile`` to True: instance_seg
    behaves exactly as plain detection does (sliced inference merges masks across seams), and each
    result's masks are the sliced (merged polygon) shape."""
    from tests._verified_checkpoint_fixtures import run_inference_verified

    _register_instance_seg_ckpt(instance_seg_ckpt, tmp_path)
    r = run_inference_verified(instance_seg_ckpt, images_dir=str(Path(_image(tmp_path / "images")).parent),
                               device="cpu", tile_size=TILE, conf_threshold=0.0)
    assert "error" not in r
    assert r["slicing"] is not None
    assert r["slicing"] is not None
    assert len(r["results"]) == 1
    result = r["results"][0]
    assert "masks" in result
    if result["count"]:
        assert set(result["masks"][0]) == {"segmentation"}


def test_run_inference_instance_seg_explicit_tile_true_runs_tiled_with_masks(instance_seg_ckpt, tmp_path):
    """An explicit tile=True is no longer refused for instance_seg: tiled inference threads masks
    through the cross-tile reconstruction/merge now, so this checkpoint tiles like any other."""
    from tests._verified_checkpoint_fixtures import run_inference_verified

    _register_instance_seg_ckpt(instance_seg_ckpt, tmp_path)
    r = run_inference_verified(instance_seg_ckpt, images_dir=str(Path(_image(tmp_path / "images")).parent),
                               device="cpu", tile=True, tile_size=TILE, conf_threshold=0.0)
    assert "error" not in r
    assert r["slicing"] is not None
    assert len(r["results"]) == 1
    assert "masks" in r["results"][0]


def test_run_inference_instance_seg_unset_tile_writes_tiled(instance_seg_ckpt, tmp_path):
    from tcip_mcp.tools.inference_tools import run_inference

    _register_instance_seg_ckpt(instance_seg_ckpt, tmp_path)
    images_dir = tmp_path / "images"
    _image(images_dir)
    r = run_inference(instance_seg_ckpt, str(images_dir), output_dir=str(tmp_path / "preds"),
                      device="cpu", tile_size=TILE, conf_threshold=0.0)
    assert "error" not in r
    assert r["slicing"] is not None
    assert (Path(r["output_dir"]) / "img.json").is_file()


def test_deliver_per_image_counts_instance_seg_refuses_a_bare_tiled_pass(instance_seg_ckpt, tmp_path):
    """deliver_per_image_counts takes no acknowledgment for the CSV itself, so a masked tiled
    instance_seg run with no calibration behind it refuses cleanly, the same as any other detection
    checkpoint, now that tiled inference carries masks rather than being blocked."""
    from tcip_mcp.tools.inference_tools import deliver_per_image_counts

    images_dir = tmp_path / "images"
    _image(images_dir)
    out_path = tmp_path / "counts.csv"
    from tests import _operationalization_fixtures as fx

    _register_instance_seg_ckpt(instance_seg_ckpt, tmp_path)
    fx.seed_confirmed_count(tmp_path)
    r = deliver_per_image_counts(instance_seg_ckpt, str(images_dir), str(out_path),
                        trait=fx.COUNT_TRAIT, device="cpu", tile_size=TILE)
    assert "error" in r
    assert not out_path.exists()


def test_deliver_per_image_counts_instance_seg_bucket_regime_reads_agree_on_masks(
    instance_seg_ckpt, tmp_path,
):
    """The write-side geometry drop (write_predictions_json's geometry_extent_ok on the mask
    polygon) is the only extent filter a masks-backed CSV row goes through: two independent
    bucket-regime reads of the same promoted, masked, tiled bucket agree on every cell, so a
    masked detection's box-based extent (irrelevant to a polygon) cannot diverge them."""
    import csv
    from datetime import datetime

    from tcip_mcp.tools.inference_tools import deliver_per_image_counts
    from tests import _operationalization_fixtures as fx

    images_dir = tmp_path / "images"
    _image(images_dir)
    _register_instance_seg_ckpt(instance_seg_ckpt, tmp_path)
    fx.seed_confirmed_count(tmp_path)

    # This door takes no acknowledgment for the CSV itself, so the live pass refuses; the raw
    # bucket it published ahead of that refusal is what the bucket-regime reads below promote.
    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    published = deliver_per_image_counts(
        instance_seg_ckpt, str(images_dir), str(tmp_path / "seed.csv"), trait=fx.COUNT_TRAIT,
        device="cpu", tile_size=TILE, predictions_dir=str(bucket))
    assert "error" in published
    assert bucket.is_dir()

    from tcip_mcp.pipelines.resolution import VALIDATED_HELD_OUT, read_operating_point_sidecar
    from tests._binding_fixtures import write_bound_sidecar

    sidecar = read_operating_point_sidecar(bucket) or {}
    op = dict(sidecar.get("operating_point") or {})
    op["conf"] = {**op.get("conf", {}), "validated_against": VALIDATED_HELD_OUT}
    stamp = {**sidecar, "validated": True, "trait": fx.COUNT_TRAIT, "operating_point": op}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path / "ds", producing_experiment_id=None)

    csv_a = tmp_path / "a.csv"
    reread_a = deliver_per_image_counts(predictions_dir=str(bucket), output_path=str(csv_a),
                             trait=fx.COUNT_TRAIT)
    assert "error" not in reread_a, reread_a

    csv_b = tmp_path / "b.csv"
    reread_b = deliver_per_image_counts(predictions_dir=str(bucket), output_path=str(csv_b),
                             trait=fx.COUNT_TRAIT)
    assert "error" not in reread_b, reread_b

    rows_a = list(csv.DictReader(csv_a.open()))
    rows_b = list(csv.DictReader(csv_b.open()))
    assert len(rows_a) == len(rows_b) == 1
    for key in rows_a[0]:
        if key == "produced_at":
            datetime.fromisoformat(rows_a[0][key])
            datetime.fromisoformat(rows_b[0][key])
            continue
        assert rows_a[0][key] == rows_b[0][key], key


def test_run_inference_stamps_mask_binarize_provenance_when_masks_present(instance_seg_ckpt, tmp_path):
    """The unvalidated mask-binarize threshold must not be stamped into Annotation.attributes
    (the domain trait namespace, which would pollute GT). It travels once, as a run constant,
    in operating_point.json, the same door tiled/tile_size/conf already use. Exercised on the
    tiled path (the checkpoint's own default now that instance_seg tiles like any other detection
    task), so the sliced (merged polygon) mask shape reaches export too."""
    import json

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar
    from tcip_mcp.tools.inference_tools import run_inference

    _register_instance_seg_ckpt(instance_seg_ckpt, tmp_path)
    images_dir = tmp_path / "images"
    _image(images_dir)
    out = tmp_path / "preds"
    r = run_inference(instance_seg_ckpt, str(images_dir), output_dir=str(out), device="cpu",
                      tile_size=TILE, conf_threshold=0.0)  # force at least one (masked) detection
    assert "error" not in r
    op = read_operating_point_sidecar(r["output_dir"])
    # The threshold the predictor cut the masks at, carried on its own records.
    assert op["mask_binarize"]["name"] == "mask_binarize_threshold"
    assert op["mask_binarize"]["value"] == pytest.approx(0.5)
    assert op["mask_binarize"]["requires_validation"] is True
    assert op["mask_binarize"]["validated_against"] == "false"

    pred_json = json.loads((Path(r["output_dir"]) / "img.json").read_text())
    for ann in pred_json["annotations"]:
        assert ann.get("attributes", {}) == {}  # never per-annotation


def test_run_inference_instance_seg_explicit_tile_true_writes_tiled(instance_seg_ckpt, tmp_path):
    """An explicit tile=True is no longer refused for instance_seg export: masks travel through
    the tiled path into the written prediction bucket."""
    from tcip_mcp.tools.inference_tools import run_inference

    _register_instance_seg_ckpt(instance_seg_ckpt, tmp_path)
    images_dir = tmp_path / "images"
    _image(images_dir)
    out = tmp_path / "preds"
    r = run_inference(instance_seg_ckpt, str(images_dir), output_dir=str(out), device="cpu",
                      tile=True, tile_size=TILE, conf_threshold=0.0)
    assert "error" not in r
    assert (Path(r["output_dir"]) / "img.json").is_file()


def test_run_full_frame_evaluation_tiled_instance_seg_scores_boxes(instance_seg_ckpt, tmp_path):
    """The delivery-gating eval never consumed masks: it reads boxes/scores/labels only, so a
    tile-trained Mask R-CNN must still evaluate here instead of crashing on the mask refusal."""
    import tcip_store as ts
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.training.eval_runners import (
        evaluation_results_key,
        run_full_frame_evaluation,
    )

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _image(images_dir, "a.png")
    labels_dir.mkdir(parents=True)
    json_io.write_annotations(str(labels_dir / "a.json"),
                              [Annotation(subject="bud", geometry=BBox(54, 54, 74, 74))], 128, 128)

    _register_instance_seg_ckpt(instance_seg_ckpt, tmp_path)
    checkpoint = load_registered_checkpoint(instance_seg_ckpt, project_path=str(tmp_path))
    r = run_full_frame_evaluation(checkpoint, str(images_dir), str(labels_dir),
                                  str(tmp_path / "out"), subject="bud",
                                  tile_size=TILE, overlap=0.2)
    assert r["eval_regime"] == "full-frame-tiled-inference"
    assert r["scored_images"] == 1
    assert r["n_gt"] == 1
    assert ts.exists(evaluation_results_key(tmp_path / "out"))
    # task records the checkpoint's actual producer task, not a hardcoded "detection". iou_type
    # stays "bbox" regardless of task: this gate always scores boxes only, by design (see the
    # docstring).
    assert r["task"] == "instance_seg"
    assert r["iou_type"] == "bbox"


# export.py write_predictions_json: masks become a real (possibly multi-ring) Polygon.

def _mask_record(mask: np.ndarray) -> dict:
    """``mask`` as the predictor's record carries it: the shared extractor's rings of the mask
    binarized at the platform's threshold, each flattened to SAHI's ``[x0, y0, x1, y1, ...]``."""
    from tcip_annotation.mask_contours import mask_to_polygon_rings

    from tcip_mcp.pipelines.measurement.mask_geometry import resolve_binarize_threshold

    threshold = resolve_binarize_threshold().unvalidated_value(acknowledge_unvalidated=True)
    return {"segmentation": [[c for point in ring for c in point]
                             for ring in mask_to_polygon_rings(mask, threshold=threshold)]}


def test_export_single_component_mask_writes_polygon(tmp_path):
    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json
    from tcip_annotation import json_io
    from tcip_annotation.state import Polygon

    mask = np.zeros((32, 32), dtype=np.float32)
    mask[5:20, 5:20] = 0.9
    result = {
        "image": "img.jpg", "width": 32, "height": 32,
        "boxes": [[5.0, 5.0, 19.0, 19.0]], "scores": [0.9], "labels": [1],
        "masks": [_mask_record(mask)],
    }
    out = tmp_path / "img.json"
    write_predictions_json(str(out), result, subject="leaf", attribute=None)
    anns = json_io.read_annotations(str(out))
    assert len(anns) == 1
    assert isinstance(anns[0].geometry, Polygon)
    assert len(anns[0].geometry.rings) == 1


def test_export_does_not_pollute_annotation_attributes_with_binarize_threshold(tmp_path):
    """Stamping the mask-binarize threshold into Annotation.attributes (the domain trait
    namespace, not a machine-provenance one) would let it survive into GT the moment a breeder
    accepts the prediction. For a detector run (attribute=None) attributes must stay empty; a
    classified run's own decoded value would land there instead. The threshold travels once into
    the run's operating_point.json instead (see
    test_run_inference_stamps_mask_binarize_provenance_when_masks_present)."""
    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json
    from tcip_annotation import json_io

    mask = np.zeros((32, 32), dtype=np.float32)
    mask[5:20, 5:20] = 0.9
    result = {
        "image": "img.jpg", "width": 32, "height": 32,
        "boxes": [[5.0, 5.0, 19.0, 19.0]], "scores": [0.9], "labels": [1],
        "masks": [_mask_record(mask)],
    }
    out = tmp_path / "img.json"
    write_predictions_json(str(out), result, subject="leaf", attribute=None)
    anns = json_io.read_annotations(str(out))
    assert anns[0].attributes == {}


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
        "masks": [_mask_record(mask)],
    }
    out = tmp_path / "img.json"
    write_predictions_json(str(out), result, subject="leaf", attribute=None)
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
        "masks": [_mask_record(mask)],
    }
    out = tmp_path / "img.json"
    write_predictions_json(str(out), result, subject="leaf", attribute=None)
    anns = json_io.read_annotations(str(out))
    assert len(anns) == 1
    assert isinstance(anns[0].geometry, BBox)


def test_export_drops_a_mask_that_binarizes_to_a_sliver(tmp_path, monkeypatch):
    """A mask whose contour is real but collinear carries no real extent either: the writer drops
    it and reports the count, the same as a degenerate box, rather than storing a zero-area shape."""
    from tcip_mcp.pipelines.postprocessing import export
    from tcip_annotation import json_io
    from tcip_annotation.state import Polygon

    monkeypatch.setattr(
        export, "_mask_geometry_for_export",
        lambda mask, bbox_xyxy, subject, *, image_size=None: Polygon(
            rings=[[(5, 10), (8, 10), (12, 10)]]),
    )
    result = {
        "image": "img.jpg", "width": 64, "height": 64,
        "boxes": [[5.0, 10.0, 12.0, 12.0]], "scores": [0.9], "labels": [1],
        "masks": [{"segmentation": []}],
    }
    out = tmp_path / "img.json"
    dropped = export.write_predictions_json(str(out), result, subject="leaf", attribute=None)

    assert dropped == 1
    assert json_io.read_annotations(str(out)) == []


def test_export_drops_a_polygon_whose_vertices_all_round_to_one_point(tmp_path, monkeypatch):
    """A polygon with real raw extent that collapses to one point at the document's stored
    2-decimal grid must be dropped here, the same as an already-collinear contour: the writer
    would otherwise refuse it and abort the whole batch."""
    from tcip_mcp.pipelines.postprocessing import export
    from tcip_annotation import json_io
    from tcip_annotation.state import Polygon

    monkeypatch.setattr(
        export, "_mask_geometry_for_export",
        lambda mask, bbox_xyxy, subject, *, image_size=None: Polygon(
            rings=[[(0.996, 0.996), (1.004, 0.996), (1.004, 1.004)]]),
    )
    result = {
        "image": "img.jpg", "width": 64, "height": 64,
        "boxes": [[1.0, 1.0, 2.0, 2.0]], "scores": [0.9], "labels": [1],
        "masks": [{"segmentation": []}],
    }
    out = tmp_path / "img.json"
    dropped = export.write_predictions_json(str(out), result, subject="leaf", attribute=None)

    assert dropped == 1
    assert json_io.read_annotations(str(out)) == []


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
        write_predictions_json(path, result, subject="leaf", attribute=None)
        anns = json_io.read_annotations(path)
        assert len(anns) == 1
        assert isinstance(anns[0].geometry, BBox)
    finally:
        os.unlink(path)
