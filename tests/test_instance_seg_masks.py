"""Instance masks reach inference and export: ``check_model_contract`` requires them, the
predictor's detection record carries them as polygons, ``encode_predictions`` converts each to a
(possibly multi-ring) ``Polygon``, and ``require_masks=False`` answers boxes alone."""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tests._chain_fixtures import BESPOKE_INSTANCE_SEG
from tests._producer_fixtures import checkpoint_admission, gray_frame, painted_array

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.model_build import CONFIG_KEY, STATE_DICT_KEY  # noqa: E402
pytest.importorskip("torchvision")
cv2 = pytest.importorskip("cv2")

from tcip_mcp.pipelines.data.label_queries import registry_scope  # noqa: E402
from tcip_mcp.pipelines.execution import Stated  # noqa: E402
from tcip_mcp.pipelines.model_contract import check_model_contract  # noqa: E402
from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor  # noqa: E402
from tests import bespoke_models  # noqa: E402

# What the smokes below synthesize their batch at, the shape a run resolves for itself.
_SMOKE_DIMS = {"in_chans": 3, "num_classes": 1, "img_size": 64}
LEAF = registry_scope(Path(__file__).parent, "leaf")


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


def _bare_predictor(task: str) -> GenericPredictor:
    """A GenericPredictor with no real checkpoint: __init__ is never called, only the attributes
    predict_sliced reads before it decodes the source are set."""
    p = GenericPredictor.__new__(GenericPredictor)
    p.task = task
    p.in_chans = 3
    return p


# predict_sliced: instance_seg and detection reach one slicing path, masks or not.

def _sliced_kwargs(require_masks: bool = True) -> dict:
    """``predict_sliced``'s arguments for a tiled record at ``TILE``, overlap 0.2, NMS at 0.3."""
    from tests._verified_checkpoint_fixtures import tiled_record

    return dict(execution=tiled_record(tile_size=TILE, overlap=0.2, conf=0.5), tile_batch_size=8,
                require_masks=require_masks)


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
    from tcip_mcp.pipelines.model_build import build_model, recorded_model_dims

    model_source = {"builder": BESPOKE_INSTANCE_SEG,
                    "builder_kwargs": {"min_size": TILE, "max_size": TILE * 2},
                    "task": "instance_seg"}
    config = {"model_source": model_source,
              "data": {"tiling": {"tile_size": TILE, "overlap": 0.2}, "num_channels": 3,
                       "scope": {"subject": "stem", "attributes": []}}}
    model = build_model(config, recorded_model_dims(config))
    ckpt = tmp_path_factory.mktemp("instance_seg_ckpt") / "model_best.pt"
    torch.save({STATE_DICT_KEY: model.state_dict(), CONFIG_KEY: config}, str(ckpt))
    return str(ckpt)


def _register_instance_seg_ckpt(ckpt_path: str, project_root: Path) -> None:
    """Register the module-scoped checkpoint in one test's own project."""
    from tcip_mcp.tools.model_tools import register_model

    result = register_model(name="instance-seg-test-model", checkpoint_path=ckpt_path,
                            config={}, project=project_root)
    assert "error" not in result, result


def test_predict_sliced_require_masks_false_returns_boxes_only(instance_seg_ckpt, tmp_path):
    """The opt-out tiles normally and returns no masks key at all, never a partial one, while the
    same predictor's untiled path still carries masks (an opt-out, not a global downgrade)."""
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.execution import prepare_pass

    _register_instance_seg_ckpt(instance_seg_ckpt, tmp_path)
    checkpoint = load_registered_checkpoint(instance_seg_ckpt, project=tmp_path)
    tiled_pass = prepare_pass(checkpoint, Stated(tile=True, tile_size=TILE, overlap=0.2,
                                                 postprocess="nms", cross_tile_nms=0.3, conf=0.5),
                              device="cpu")
    pred = tiled_pass.predictor
    assert pred.task == "instance_seg"
    img = gray_frame(tmp_path / "images" / UNDATED_BUCKET)

    tiled = pred.predict_sliced(img, execution=tiled_pass.execution, tile_batch_size=8,
                                require_masks=False)
    assert "masks" not in tiled
    assert {"boxes", "scores", "labels", "count", "tiles"} <= set(tiled)
    assert tiled["tiles"] >= 4  # 128px image at tile 64 -> a 2x2+ lattice, i.e. it really sliced
    assert tiled["count"] == len(tiled["boxes"]) == len(tiled["scores"])

    untiled = prepare_pass(checkpoint, Stated(tile=False, conf=0.5), device="cpu")
    assert "masks" in untiled.predict([img])[0]


def test_run_inference_instance_seg_unset_tile_runs_tiled_with_masks(instance_seg_ckpt, tmp_path):
    """The fixture's own persisted training tile geometry derives an unset ``tile`` to True:
    instance_seg behaves exactly as plain detection does (sliced inference merges masks across
    seams), and each result's masks are the sliced (merged polygon) shape."""
    from tests._verified_checkpoint_fixtures import predicted_over

    _register_instance_seg_ckpt(instance_seg_ckpt, tmp_path)
    p, results = predicted_over(tmp_path, instance_seg_ckpt,
                                str(Path(gray_frame(tmp_path / "images" / UNDATED_BUCKET)).parent),
                                device="cpu", tile_size=TILE, conf=0.0)
    assert p.execution.tiled
    assert len(results) == 1
    result = results[0]
    assert "masks" in result
    if result["count"]:
        assert set(result["masks"][0]) == {"segmentation"}


def test_run_inference_instance_seg_explicit_tile_true_runs_tiled_with_masks(
    instance_seg_ckpt, tmp_path
):
    """An explicit tile=True is no longer refused for instance_seg: tiled inference threads masks
    through the cross-tile reconstruction/merge now, so this checkpoint tiles like any other."""
    from tests._verified_checkpoint_fixtures import predicted_over

    _register_instance_seg_ckpt(instance_seg_ckpt, tmp_path)
    p, results = predicted_over(tmp_path, instance_seg_ckpt,
                                str(Path(gray_frame(tmp_path / "images" / UNDATED_BUCKET)).parent),
                                device="cpu", tile=True, tile_size=TILE, conf=0.0)
    assert p.execution.tiled
    assert len(results) == 1
    assert "masks" in results[0]


def test_run_inference_instance_seg_unset_tile_writes_tiled(instance_seg_ckpt, tmp_path):
    from tcip_mcp.tools.inference_tools import run_inference

    _register_instance_seg_ckpt(instance_seg_ckpt, tmp_path)
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    gray_frame(images_dir)
    r = run_inference(tmp_path, instance_seg_ckpt, str(images_dir), bucket="preds/2026-01-01",
                      device="cpu", stated=Stated(tile_size=TILE, conf=0.0))
    assert "error" not in r
    assert r["execution"]["tile_size"] == TILE
    assert _document(r, images_dir) is not None


def _document(published: dict, images_dir: Path):
    """The key of ``img.png``'s document in the bucket a ``run_inference`` result names."""
    from tcip_mcp.buckets import read_bucket

    bucket = read_bucket(published["dataset_root"], published["bucket"])
    return bucket.document_key("img")


def test_a_masked_bucket_delivers_the_same_counts_on_every_read(instance_seg_ckpt, tmp_path):
    """The write-side geometry drop on the mask polygon is the only extent filter a masks-backed
    count goes through: two deliveries of the same published masked, tiled bucket agree on every
    count, and the door that takes no acknowledgment refuses the unassessed bucket."""
    import csv

    from tcip_mcp.pipelines.postprocessing.export import deliver_per_image_counts_csv
    from tcip_mcp.tools.inference_tools import deliver_per_image_counts, run_inference
    from tests import _trait_fixtures as fx
    from tests._chain_fixtures import acknowledged

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    gray_frame(images_dir)
    _register_instance_seg_ckpt(instance_seg_ckpt, tmp_path)
    fx.seed_confirmed_count(tmp_path)
    bucket = "baseline/2026-01-01"
    ran = run_inference(tmp_path, instance_seg_ckpt, str(images_dir), bucket=bucket,
                        device="cpu", stated=Stated(tile_size=TILE, conf=0.0))
    assert "error" not in ran, ran

    refused = deliver_per_image_counts(tmp_path, str(tmp_path), bucket,
                                       str(tmp_path / "refused.csv"), trait=fx.COUNT_TRAIT)
    rows = []
    for name in ("a", "b"):
        out = tmp_path / f"{name}.csv"
        acknowledged(tmp_path, lambda ack: deliver_per_image_counts_csv(
            tmp_path, tmp_path, bucket, str(out), trait=fx.COUNT_TRAIT, acknowledgment_id=ack,
            door="test_instance_seg", actor=None))
        rows.append([{k: v for k, v in r.items() if k != "delivery_event_id"}
                     for r in csv.DictReader(out.open(newline="", encoding="utf-8"))])

    assert "no assessment answers" in refused["error"]
    assert len(rows[0]) == 1 and rows[0] == rows[1]


def test_run_inference_never_stamps_a_mask_threshold_into_annotation_attributes(
    instance_seg_ckpt, tmp_path,
):
    """Annotation.attributes is the domain trait namespace, so no pass value lands there.
    Exercised on the tiled path, so the sliced (merged polygon) mask shape reaches export too."""
    import tcip_store

    from tcip_mcp.tools.inference_tools import run_inference

    _register_instance_seg_ckpt(instance_seg_ckpt, tmp_path)
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    gray_frame(images_dir)
    r = run_inference(tmp_path, instance_seg_ckpt, str(images_dir), bucket="preds/2026-01-01",
                      device="cpu", stated=Stated(tile_size=TILE, conf=0.0))  # a masked detection
    assert "error" not in r

    for ann in tcip_store.read(_document(r, images_dir))["annotations"]:
        assert ann.get("attributes", {}) == {}  # never per-annotation


def test_run_inference_instance_seg_explicit_tile_true_writes_tiled(instance_seg_ckpt, tmp_path):
    """An explicit tile=True is no longer refused for instance_seg export: masks travel through
    the tiled path into the written prediction bucket."""
    from tcip_mcp.tools.inference_tools import run_inference

    _register_instance_seg_ckpt(instance_seg_ckpt, tmp_path)
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    gray_frame(images_dir)
    r = run_inference(tmp_path, instance_seg_ckpt, str(images_dir), bucket="preds/2026-01-01",
                      device="cpu", stated=Stated(tile=True, tile_size=TILE, conf=0.0))
    assert "error" not in r
    assert _document(r, images_dir) is not None


def test_run_full_frame_evaluation_tiled_instance_seg_scores_masks(instance_seg_ckpt, tmp_path):
    """A tile-trained Mask R-CNN is gated full frame by its masks over the objects its own task
    selects: a box beside a polygon is no instance reference on either route."""
    from tcip_annotation.state import Annotation, BBox, Polygon
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.data.datasets import build_dataset, resolve_sizes
    from tcip_mcp.pipelines.training.eval_runners import run_full_frame_evaluation
    from tests._producer_fixtures import label_image

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    gray_frame(images_dir, name="a.png")
    label_image(images_dir / "a.png", [
        Annotation(subject="stem", geometry=BBox(10, 10, 30, 30)),
        Annotation(subject="stem", geometry=Polygon(rings=[[(54, 54), (74, 54), (74, 74)]]))],
        128, 128)

    _register_instance_seg_ckpt(instance_seg_ckpt, tmp_path)
    checkpoint = load_registered_checkpoint(instance_seg_ckpt, project=tmp_path)
    admitted = checkpoint_admission(checkpoint, images_dir)
    r = run_full_frame_evaluation(checkpoint, admitted, stated=Stated(tile_size=TILE, overlap=0.2))
    samples = admitted.every_sample()
    trained = build_dataset("instance_seg", scope=admitted.scope, samples=samples,
                            sizes=resolve_sizes("instance_seg", {"num_channels": 3}, samples))
    assert r["eval_regime"] == "full-frame-tiled-inference"
    assert r["scored_images"] == 1
    assert r["n_gt"] == len(trained[0][1]["boxes"]) == 1
    assert r["task"] == "instance_seg"
    assert r["governing_criterion"]["kind"] == "mask_iou_match"


# export.py encode_predictions: masks become a real (possibly multi-ring) Polygon.

def _encoded(result: dict) -> tuple[list, int]:
    """``result`` encoded under :data:`LEAF` and read back, and the count it dropped."""
    from tcip_annotation.json_io import label_document
    from tcip_mcp.pipelines.postprocessing.export import encode_predictions

    data, dropped = encode_predictions(result, "model:fixture", scope=LEAF)
    return label_document(data).annotations, dropped


def _mask_record(mask: np.ndarray) -> dict:
    """``mask`` as the predictor's record carries it: the shared extractor's rings of the mask
    binarized at the platform's threshold, each flattened to SAHI's ``[x0, y0, x1, y1, ...]``."""
    from tcip_annotation.mask_contours import mask_to_polygon_rings

    from tcip_mcp.pipelines.measurement.mask_geometry import resolve_binarize_threshold

    threshold = resolve_binarize_threshold()["value"]
    return {"segmentation": [[c for point in ring for c in point]
                             for ring in mask_to_polygon_rings(mask, threshold=threshold)]}


def test_export_single_component_mask_writes_polygon():
    from tcip_annotation.state import Polygon

    mask = painted_array(32, 32, [((5, 5, 20, 20), 0.9)], background=0.0, mode="F")
    result = {
        "image": "img.jpg", "width": 32, "height": 32,
        "boxes": [[5.0, 5.0, 19.0, 19.0]], "scores": [0.9], "labels": [1],
        "masks": [_mask_record(mask)],
    }
    anns, _dropped = _encoded(result)
    assert len(anns) == 1
    assert isinstance(anns[0].geometry, Polygon)
    assert len(anns[0].geometry.rings) == 1


def test_export_does_not_pollute_annotation_attributes_with_binarize_threshold(tmp_path):
    """Under a scope declaring no attribute, an exported mask annotation carries no attributes:
    the binarize threshold never lands in ``Annotation.attributes``."""
    mask = painted_array(32, 32, [((5, 5, 20, 20), 0.9)], background=0.0, mode="F")
    result = {
        "image": "img.jpg", "width": 32, "height": 32,
        "boxes": [[5.0, 5.0, 19.0, 19.0]], "scores": [0.9], "labels": [1],
        "masks": [_mask_record(mask)],
    }
    anns, _dropped = _encoded(result)
    assert anns[0].attributes == {}


def test_export_multi_component_mask_writes_multi_ring_polygon(tmp_path):
    """An occlusion-split mask must export every region as its own ring in one Polygon, never
    silently truncated to the largest component and never downgraded to a BBox that would lose
    the shape entirely."""
    from tcip_annotation.state import Polygon

    mask = painted_array(64, 64, [((5, 5, 15, 15), 0.9), ((40, 40, 55, 55), 0.9)],
                         background=0.0, mode="F")
    result = {
        "image": "img.jpg", "width": 64, "height": 64,
        "boxes": [[5.0, 5.0, 54.0, 54.0]], "scores": [0.9], "labels": [1],
        "masks": [_mask_record(mask)],
    }
    anns, _dropped = _encoded(result)
    assert len(anns) == 1
    assert isinstance(anns[0].geometry, Polygon)
    assert len(anns[0].geometry.rings) == 2


def test_export_empty_mask_falls_back_to_bbox():
    from tcip_annotation.state import BBox

    mask = np.zeros((16, 16), dtype=np.float32)  # binarizes to nothing at the default threshold
    result = {
        "image": "img.jpg", "width": 16, "height": 16,
        "boxes": [[1.0, 1.0, 5.0, 5.0]], "scores": [0.9], "labels": [1],
        "masks": [_mask_record(mask)],
    }
    anns, _dropped = _encoded(result)
    assert len(anns) == 1
    assert isinstance(anns[0].geometry, BBox)


def test_export_drops_a_mask_that_binarizes_to_a_sliver(monkeypatch):
    """A mask whose contour is real but collinear carries no real extent either: the encoder drops
    it and reports the count, the same as a degenerate box, rather than storing a zero-area
    shape."""
    from tcip_mcp.pipelines.postprocessing import export
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
    anns, dropped = _encoded(result)

    assert dropped == 1
    assert anns == []


def test_export_drops_a_polygon_whose_vertices_all_round_to_one_point(monkeypatch):
    """A polygon with real raw extent that collapses to one point at the document's stored
    2-decimal grid must be dropped here, the same as an already-collinear contour: the writer
    would otherwise refuse it and abort the whole batch."""
    from tcip_mcp.pipelines.postprocessing import export
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
    anns, dropped = _encoded(result)

    assert dropped == 1
    assert anns == []


def test_export_no_masks_key_writes_bbox_as_before():
    """Regression guard: a plain detection result (no masks key at all) must still export BBox
    as before; masks are additive, not a behavior change for non-instance_seg."""
    from tcip_annotation.state import BBox

    result = {
        "image": "img.jpg", "width": 32, "height": 32,
        "boxes": [[1.0, 1.0, 5.0, 5.0]], "scores": [0.9], "labels": [1],
    }
    anns, _dropped = _encoded(result)
    assert len(anns) == 1
    assert isinstance(anns[0].geometry, BBox)
