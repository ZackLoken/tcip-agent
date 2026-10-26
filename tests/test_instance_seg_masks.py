"""Instance masks reach inference and export.

``check_model_contract`` requires them, the predictor's detection record carries them as
polygons, ``encode_predictions`` converts each to a (possibly multi-ring) ``Polygon``, and a pass
stating ``masks`` false answers boxes alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
from tests._chain_fixtures import (
    BESPOKE_INSTANCE_SEG,
    BESPOKE_MODELS,
    FRAME_STATISTICS,
    detector_values,
)
from tests._producer_fixtures import (
    checkpoint_admission,
    gray_frame,
    write_image,
)
from tests._training_values import MASK_CUT, RATIONALE, tiling_block

if TYPE_CHECKING:
    from torch import Tensor

    import tcip_store

torch = pytest.importorskip("torch")

pytest.importorskip("torchvision")
cv2 = pytest.importorskip("cv2")

from tcip_mcp.pipelines.data.label_queries import registry_scope  # noqa: E402
from tcip_mcp.pipelines.execution import Stated  # noqa: E402
from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor  # noqa: E402
from tcip_mcp.pipelines.model_contract import check_model_contract  # noqa: E402
from tests import bespoke_models  # noqa: E402
from tests._verified_checkpoint_fixtures import (  # noqa: E402
    SAMPLE_CROSS_TILE_NMS,
    SAMPLE_DETECTOR_PASS,
    SAMPLE_INSTANCE_SEG_PASS,
    SAMPLE_OVERLAP,
)

# What the smokes below synthesize their batch at, the shape a run resolves for itself.
_SMOKE_DIMS = {"in_chans": 3, "num_classes": 1, "img_size": (64, 64)}
LEAF = registry_scope(Path(__file__).parent, "leaf")


# --------------------------------------------------------------------------
# model_contract: instance_seg requires masks in the eval output
# --------------------------------------------------------------------------


def test_contract_rejects_maskless_instance_seg_model() -> None:
    """The instance_seg contract rejects a model emitting no masks.

    A model that emits boxes, scores and labels but no masks fails the instance_seg contract
    with an issue naming the masks.
    """
    import torch.nn as nn

    class _BoxesOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(
            self, images: list[Tensor], targets: list[dict] | None = None
        ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
            if self.training:
                return {"loss": self.lin(torch.rand(1, 4)).sum()}
            return [
                {
                    "boxes": torch.zeros((1, 4)),
                    "scores": torch.ones((1,)),
                    "labels": torch.ones((1,), dtype=torch.int64),
                }
            ]

    report = check_model_contract(
        _BoxesOnly(),
        "instance_seg",
        device=torch.device("cpu"),
        dims=_SMOKE_DIMS,
    )
    assert report["ok"] is False
    assert any("masks" in i for i in report["issues"]), report["issues"]


def test_contract_accepts_real_masked_instance_seg_model() -> None:
    model = bespoke_models.build_bespoke_instance_seg(
        num_classes=1, **detector_values()
    )
    report = check_model_contract(
        model, "instance_seg", device=torch.device("cpu"), dims=_SMOKE_DIMS
    )
    assert report["ok"], report["issues"]
    assert report["eval_output_type"] == "list[dict]"


def test_contract_detection_task_unaffected_by_mask_requirement() -> None:
    """A detection model emitting no masks passes the detection contract."""
    model = bespoke_models.build_bespoke_detection(
        num_classes=1, **detector_values()
    )
    report = check_model_contract(
        model, "detection", device=torch.device("cpu"), dims=_SMOKE_DIMS
    )
    assert report["ok"], report["issues"]


def _bare_predictor(task: str) -> GenericPredictor:
    """Return a GenericPredictor with no real checkpoint.

    It is built without ``__init__``, carrying ``task`` and ``in_chans`` alone.
    """
    p = GenericPredictor.__new__(GenericPredictor)
    p.task = task
    p.in_chans = 3
    return p


# predict_sliced: instance_seg and detection reach one slicing path, masks or not.


def _sliced_kwargs() -> dict:
    """Return ``predict_sliced``'s arguments for a tiled record at ``TILE``, overlap 0.2."""
    from tests._verified_checkpoint_fixtures import tiled_record

    return dict(
        execution=tiled_record(tile_size=TILE, overlap=0.2, conf=0.5),
        tile_batch_size=8,
    )


def test_predict_sliced_instance_seg_reaches_real_slicing_path(
    tmp_path: Path,
) -> None:
    """An instance_seg slice of a missing image raises ``FileNotFoundError``."""
    p = _bare_predictor("instance_seg")
    missing = tmp_path / "missing.jpg"
    with pytest.raises(FileNotFoundError):
        p.predict_sliced(str(missing), **_sliced_kwargs())


def test_predict_sliced_detection_reaches_real_slicing_path(
    tmp_path: Path,
) -> None:
    """A detection slice of a missing image raises ``FileNotFoundError``."""
    p = _bare_predictor("detection")
    missing = tmp_path / "missing.jpg"
    with pytest.raises(FileNotFoundError):
        p.predict_sliced(str(missing), **_sliced_kwargs())


# --------------------------------------------------------------------------
# The rail admits valid work: a real Mask R-CNN checkpoint through the
# boxes-only opt-out, the two MCP doors, and the delivery-gating eval
# --------------------------------------------------------------------------

TILE = 64
TILED_PASS = Stated(
    tile=True,
    tile_size=TILE,
    overlap=SAMPLE_OVERLAP,
    conf=0.0,
    masks=True,
    mask_cut=MASK_CUT,
    postprocess="nms",
    cross_tile_nms=SAMPLE_CROSS_TILE_NMS,
    rationale=RATIONALE,
)
"""The execution values a tiled inference pass here states."""
UNSTATED_LATTICE_PASS = Stated(
    conf=0.0,
    masks=True,
    mask_cut=MASK_CUT,
    postprocess="nms",
    cross_tile_nms=SAMPLE_CROSS_TILE_NMS,
    rationale=RATIONALE,
)
"""A pass stating neither whether it tiles nor at what lattice."""
STRADDLING = [(4.0, 4.0), (124.0, 60.0), (4.0, 60.0)]
"""A stem's triangle, in a frame two :data:`TILE` px tiles wide, crossing the seam at x=64:
within the second tile its diagonal edge starts at y=32, where its bounding box starts at y=4."""
TRAINED_LATTICE = tiling_block(tile_size=TILE, overlap=0.0, sliver_frac=0.25)
"""The lattice the fixture run states: two tiles per frame; two objects derive no sliver
spread, and a quarter of the triangle's size keeps both of its pieces."""


def _stem_run(tmp_path: Path) -> Path:
    """Complete a bespoke instance_seg (Mask R-CNN) run in ``tmp_path`` and return its directory.

    The run is over two frames of its own, each two :data:`TILE` px tiles wide holding one
    :data:`STRADDLING` polygon, stating :data:`TRAINED_LATTICE`.
    """
    from tcip_annotation.state import Annotation, Polygon
    from tests._producer_fixtures import seed_labeled_images
    from tests._verified_checkpoint_fixtures import (
        finished_run,
        fixture_data_dir,
    )

    stem = Annotation(subject="stem", geometry=Polygon([STRADDLING]))
    images = seed_labeled_images(
        image_dir(fixture_data_dir(tmp_path, "stems"), UNDATED_BUCKET),
        [stem],
        n=2,
        width=2 * TILE,
        height=TILE,
    )
    return finished_run(
        tmp_path,
        model_source={
            "builder": BESPOKE_INSTANCE_SEG,
            "task": "instance_seg",
            "source_files": [BESPOKE_MODELS],
            "normalization": FRAME_STATISTICS,
        },
        data={
            "images_dir": str(images),
            "num_channels": 3,
            "scope": {"subject": "stem"},
            "tiling": TRAINED_LATTICE,
        },
    )


@pytest.fixture
def instance_seg_ckpt(tmp_path: Path) -> str:
    """Return the checkpoint :func:`_stem_run` registered by completing."""
    from tests._verified_checkpoint_fixtures import completed_checkpoint

    completed = completed_checkpoint(_stem_run(tmp_path))
    assert completed is not None
    return completed["path"]


def test_an_instance_seg_run_trains_over_its_instances_clipped_to_each_tile(
    tmp_path: Path,
) -> None:
    """An instance_seg run's tiled loaders serve its instances clipped to each tile.

    A run stating a lattice serves tiles whose masks are its instances cut at each tile's edge:
    a triangle straddling the seam appears in both of its frame's tiles, each tile's mask the
    run's own untiled loader's frame mask over that tile's rect and each tile's box the bounds of
    that mask's pixels within one pixel; its checkpoint records the lattice.
    """
    from tcip_mcp.experiments import observe_run
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.data.datasets import (
        TiledDetectionDataset,
        build_dataset,
        stated_sizes,
    )
    from tcip_mcp.pipelines.training.subprocess_worker import (
        prepare_run_context,
    )
    from tests._verified_checkpoint_fixtures import completed_checkpoint

    run_dir = _stem_run(tmp_path)
    observation = observe_run(run_dir)
    ctx = prepare_run_context(observation)
    data = ctx.run.spec.data
    frames = build_dataset(
        "instance_seg",
        scope=data.recorded_scope,
        sizes=stated_sizes(data),
        samples=observation.record.resolution.partition.samples,
    )
    frame_masks = {
        stem: frames[i][1]["masks"] for i, stem in enumerate(frames.stems)
    }
    tiled = [
        loader.dataset
        for loader in (ctx.train_loader, ctx.val_loader)
        if loader is not None
    ]

    seams: list[str] = []
    for ds in tiled:
        assert isinstance(ds, TiledDetectionDataset)
        for index, (stem, (x0, y0, x1, y1)) in enumerate(ds.tile_entries):
            target = ds[index][1]
            masks = target["masks"]
            assert masks.shape == (1, y1 - y0, x1 - x0)
            assert torch.equal(masks, frame_masks[stem][:, y0:y1, x0:x1])
            ys, xs = torch.nonzero(masks[0], as_tuple=True)
            pixels = [
                int(xs.min()),
                int(ys.min()),
                int(xs.max()) + 1,
                int(ys.max()) + 1,
            ]
            (box,) = target["boxes"].tolist()
            assert all(abs(p - b) <= 1 for p, b in zip(pixels, box)), (
                box,
                pixels,
            )
            seams.append(stem)
    assert sorted(seams) == sorted(list(frame_masks) * 2)
    completed = completed_checkpoint(run_dir)
    assert completed is not None
    recorded = load_registered_checkpoint(
        completed["path"], project=tmp_path
    ).spec.data.tiling
    assert recorded is not None
    assert recorded.model_dump(exclude_none=True) == TRAINED_LATTICE


def test_a_pass_stating_no_masks_returns_boxes_only(
    instance_seg_ckpt: str, tmp_path: Path
) -> None:
    """A pass stating no masks returns boxes and no masks key.

    A tiled pass stating ``masks`` false and no mask threshold slices into four or more tiles
    and returns no masks key; an untiled pass of the same checkpoint collecting masks carries
    them, and a pass stating neither refuses naming ``masks``.
    """
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.execution import ExecutionRefusedError, prepare

    checkpoint = load_registered_checkpoint(
        instance_seg_ckpt, project=tmp_path
    )
    tiled_pass = prepare(
        checkpoint,
        Stated(
            tile=True,
            tile_size=TILE,
            overlap=0.2,
            masks=False,
            **SAMPLE_DETECTOR_PASS,
        ),
        device="cpu",
    ).runnable()
    pred = tiled_pass.predictor
    assert pred.task == "instance_seg"
    img = gray_frame(image_dir(tmp_path, UNDATED_BUCKET))

    tiled = pred.predict_sliced(
        img,
        execution=tiled_pass.execution,
        tile_batch_size=8,
    )
    assert "masks" not in tiled
    assert {"boxes", "scores", "labels", "count", "tiles"} <= set(tiled)
    assert (
        tiled["tiles"] >= 4
    )  # 128px image at tile 64 -> a 2x2+ lattice, i.e. it really sliced
    assert tiled["count"] == len(tiled["boxes"]) == len(tiled["scores"])

    untiled = prepare(
        checkpoint,
        Stated(tile=False, **SAMPLE_INSTANCE_SEG_PASS),
        device="cpu",
    ).runnable()
    assert "masks" in untiled.predict([img])[0]

    # Whether the pass collects masks is its statement, never read off the head.
    unstated = prepare(
        checkpoint, Stated(tile=False, **SAMPLE_DETECTOR_PASS), device="cpu"
    )
    with pytest.raises(ExecutionRefusedError, match="masks None"):
        unstated.runnable()


def test_run_inference_instance_seg_explicit_tile_true_runs_tiled_with_masks(
    instance_seg_ckpt: str, tmp_path: Path
) -> None:
    """An explicitly tiled instance_seg inference runs tiled with masks.

    An explicit tile=True is admitted for instance_seg, and the tiled result carries masks in
    the sliced (merged polygon) shape.
    """
    from tests._verified_checkpoint_fixtures import predicted_over

    images_dir = image_dir(tmp_path, UNDATED_BUCKET)
    gray_frame(images_dir)
    p, results = predicted_over(
        tmp_path,
        instance_seg_ckpt,
        str(images_dir),
        device="cpu",
        tile=True,
        tile_size=TILE,
        overlap=SAMPLE_OVERLAP,
        conf=0.0,
        masks=True,
        mask_cut=MASK_CUT,
    )
    assert p.execution.tiled
    assert len(results) == 1
    result = results[0]
    assert "masks" in result
    if result["count"]:
        assert set(result["masks"][0]) == {"segmentation"}


def _document(published: dict, images_dir: Path) -> tcip_store.Key | None:
    """Return the key of ``img.png``'s document in the bucket a ``run_inference`` result names."""
    from tcip_mcp.buckets import read_bucket

    bucket = read_bucket(published["dataset_root"], published["bucket"])
    return bucket.document_key("img")


def test_a_masked_bucket_delivers_the_same_counts_on_every_read(
    instance_seg_ckpt: str, tmp_path: Path
) -> None:
    """Two deliveries of one masked bucket write identical count rows.

    Two acknowledged deliveries of one published masked, tiled bucket write identical rows, and
    the door that takes no acknowledgment refuses the unassessed bucket.
    """
    import csv

    from tcip_mcp.pipelines.postprocessing.export import (
        deliver_per_image_counts_csv,
    )
    from tcip_mcp.tools.inference_tools import (
        deliver_per_image_counts,
        run_inference,
    )
    from tests import _trait_fixtures as fx
    from tests._chain_fixtures import acknowledged

    images_dir = image_dir(tmp_path, UNDATED_BUCKET)
    gray_frame(images_dir)
    fx.seed_confirmed_count(tmp_path)
    bucket = "baseline/2026-01-01"
    ran = run_inference(
        tmp_path,
        instance_seg_ckpt,
        str(images_dir),
        bucket=bucket,
        device="cpu",
        stated=TILED_PASS,
        actor=None,
    )
    assert "error" not in ran, ran

    refused = deliver_per_image_counts(
        tmp_path,
        str(tmp_path),
        bucket,
        str(tmp_path / "refused.csv"),
        trait=fx.COUNT_TRAIT,
        actor=None,
    )
    rows = []
    for name in ("a", "b"):
        out = tmp_path / f"{name}.csv"
        acknowledged(
            tmp_path,
            lambda ack: deliver_per_image_counts_csv(
                tmp_path,
                tmp_path,
                bucket,
                str(out),
                trait=fx.COUNT_TRAIT,
                acknowledgment_id=ack,
                door="test_instance_seg",
                actor=None,
            ),
        )
        rows.append(
            [
                {k: v for k, v in r.items() if k != "delivery_event_id"}
                for r in csv.DictReader(out.open(newline="", encoding="utf-8"))
            ]
        )

    assert "no assessment answers" in refused["error"]
    assert len(rows[0]) == 1 and rows[0] == rows[1]


def test_run_inference_never_stamps_a_mask_threshold_into_annotation_attributes(
    instance_seg_ckpt: str,
    tmp_path: Path,
) -> None:
    """A tiled inference writes no attributes on its masked annotations.

    Annotation.attributes is the domain trait namespace, so no pass value lands there.
    """
    from tcip_annotation.json_io import read_predictions
    from tcip_mcp.tools.inference_tools import run_inference

    images_dir = image_dir(tmp_path, UNDATED_BUCKET)
    gray_frame(images_dir)
    r = run_inference(
        tmp_path,
        instance_seg_ckpt,
        str(images_dir),
        bucket="preds/2026-01-01",
        device="cpu",
        stated=TILED_PASS,
        actor=None,
    )  # a masked detection
    assert "error" not in r

    for ann in read_predictions(_document(r, images_dir)).annotations:
        assert ann.attributes == {}  # never per-annotation


def test_run_inference_instance_seg_explicit_tile_true_writes_tiled(
    instance_seg_ckpt: str, tmp_path: Path
) -> None:
    """An explicitly tiled instance_seg inference writes a tiled bucket.

    An explicit tile=True writes a prediction bucket at the stated tile size; a pass stating no
    lattice records the tile size and overlap the checkpoint recorded.
    """
    from tcip_mcp.tools.inference_tools import run_inference

    images_dir = image_dir(tmp_path, UNDATED_BUCKET)
    gray_frame(images_dir)
    r = run_inference(
        tmp_path,
        instance_seg_ckpt,
        str(images_dir),
        bucket="preds/2026-01-01",
        device="cpu",
        stated=TILED_PASS,
        actor=None,
    )
    assert "error" not in r
    assert r["execution"]["tile_size"] == TILE
    assert _document(r, images_dir) is not None

    implicit = run_inference(
        tmp_path,
        instance_seg_ckpt,
        str(images_dir),
        bucket="preds/2026-01-02",
        device="cpu",
        stated=UNSTATED_LATTICE_PASS,
        actor=None,
    )
    assert "error" not in implicit, implicit
    assert (
        implicit["execution"]["tile_size"],
        implicit["execution"]["overlap"],
    ) == (TILE, TRAINED_LATTICE["overlap"])


def test_run_full_frame_evaluation_tiled_instance_seg_scores_masks(
    instance_seg_ckpt: str, tmp_path: Path
) -> None:
    """A tiled full-frame evaluation of instance_seg scores its masks.

    A Mask R-CNN checkpoint evaluated with a stated tile lattice is scored full frame under a
    mask-IoU criterion over one ground truth, the count the instance_seg loader reads from an
    image holding a box and a polygon.
    """
    from tcip_annotation.state import Annotation, BBox, Polygon
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.data.datasets import build_dataset, resolve_sizes
    from tcip_mcp.pipelines.training.eval_runners import (
        run_full_frame_evaluation,
    )
    from tests._producer_fixtures import label_image
    from tests._training_values import IOU_THRESHOLD

    images_dir = image_dir(tmp_path, UNDATED_BUCKET)
    gray_frame(images_dir, name="a.png")
    label_image(
        images_dir / "a.png",
        [
            Annotation(subject="stem", geometry=BBox(10, 10, 30, 30)),
            Annotation(
                subject="stem",
                geometry=Polygon(rings=[[(54, 54), (74, 54), (74, 74)]]),
            ),
        ],
        128,
        128,
    )

    checkpoint = load_registered_checkpoint(
        instance_seg_ckpt, project=tmp_path
    )
    admitted = checkpoint_admission(checkpoint, images_dir)
    r = run_full_frame_evaluation(
        checkpoint,
        admitted,
        stated=Stated(
            tile=True, tile_size=TILE, overlap=0.2, **SAMPLE_INSTANCE_SEG_PASS
        ),
        iou_threshold=IOU_THRESHOLD,
        trait=None,
    )
    samples = admitted.every_sample()
    trained = build_dataset(
        "instance_seg",
        scope=admitted.scope,
        samples=samples,
        sizes=resolve_sizes("instance_seg", {"num_channels": 3}, samples),
    )
    assert r["eval_regime"] == "full-frame-tiled-inference"
    assert r["scored_images"] == 1
    assert r["n_gt"] == len(trained[0][1]["boxes"]) == 1
    assert r["task"] == "instance_seg"
    assert r["governing_criterion"]["kind"] == "mask_iou_match"

    implicit = run_full_frame_evaluation(
        checkpoint,
        admitted,
        stated=Stated(**SAMPLE_INSTANCE_SEG_PASS),
        iou_threshold=IOU_THRESHOLD,
        trait=None,
    )
    assert (
        implicit["execution"]["tile_size"],
        implicit["execution"]["overlap"],
    ) == (TILE, TRAINED_LATTICE["overlap"])


# export.py encode_predictions: masks become a real (possibly multi-ring) Polygon.


def _encoded(result: dict) -> tuple[list, int]:
    """Return ``result`` encoded under :data:`LEAF` and read back, and the count it dropped."""
    from tcip_annotation.json_io import label_document
    from tcip_mcp.pipelines.postprocessing.export import encode_predictions

    data, dropped = encode_predictions(result, "model:fixture", scope=LEAF)
    return label_document(data).annotations, dropped


def _predicted(
    tmp_path: Path, size: int, regions: list, *, box: list, masks: bool
) -> dict:
    """Return the record the predictor makes of a black ``size`` x ``size`` frame.

    The frame runs under a registered segmenter whose one detection, over ``box``, holds a 0.9
    soft mask over each of ``regions`` (:class:`~tests.bespoke_models.FixedMaskSegmenter`).
    With ``masks`` true the record collects the masks cut at the sample threshold; with it
    false, boxes alone.
    """
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.execution import prepare
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    checkpoint = load_registered_checkpoint(
        registered_checkpoint(
            tmp_path,
            model_source={
                "builder": "bespoke_models:build_fixed_mask_instance_seg",
                "builder_kwargs": {
                    "regions": regions,
                    "value": 0.9,
                    "box": box,
                },
                "source_files": [BESPOKE_MODELS],
                "task": "instance_seg",
                "normalization": FRAME_STATISTICS,
            },
        ),
        project=tmp_path,
    )
    frame = write_image(tmp_path / "frame.png", (size, size))
    stated = (
        Stated(tile=False, **SAMPLE_INSTANCE_SEG_PASS)
        if masks
        else Stated(tile=False, masks=False, **SAMPLE_DETECTOR_PASS)
    )
    p = prepare(checkpoint, stated, device="cpu").runnable()
    return p.predict([str(frame)])[0]


def test_export_single_component_mask_writes_polygon(tmp_path: Path) -> None:
    from tcip_annotation.state import Polygon

    result = _predicted(
        tmp_path, 32, [[5, 5, 20, 20]], box=[5, 5, 19, 19], masks=True
    )
    anns, _dropped = _encoded(result)
    assert len(anns) == 1
    assert isinstance(anns[0].geometry, Polygon)
    assert len(anns[0].geometry.rings) == 1


def test_export_does_not_pollute_annotation_attributes_with_binarize_threshold(
    tmp_path: Path,
) -> None:
    """An exported mask annotation carries no attributes under a scope declaring none."""
    result = _predicted(
        tmp_path, 32, [[5, 5, 20, 20]], box=[5, 5, 19, 19], masks=True
    )
    anns, _dropped = _encoded(result)
    assert anns[0].attributes == {}


def test_export_multi_component_mask_writes_multi_ring_polygon(
    tmp_path: Path,
) -> None:
    """A multi-component mask exports as one multi-ring polygon.

    An occlusion-split mask of two regions exports as one Polygon holding a ring per region.
    """
    from tcip_annotation.state import Polygon

    result = _predicted(
        tmp_path,
        64,
        [[5, 5, 15, 15], [40, 40, 55, 55]],
        box=[5, 5, 54, 54],
        masks=True,
    )
    anns, _dropped = _encoded(result)
    assert len(anns) == 1
    assert isinstance(anns[0].geometry, Polygon)
    assert len(anns[0].geometry.rings) == 2


def test_export_empty_mask_falls_back_to_bbox(tmp_path: Path) -> None:
    from tcip_annotation.state import BBox

    # A mask holding nothing binarizes to nothing at any threshold.
    result = _predicted(tmp_path, 16, [], box=[1, 1, 5, 5], masks=True)
    anns, _dropped = _encoded(result)
    assert len(anns) == 1
    assert isinstance(anns[0].geometry, BBox)


def test_export_drops_a_mask_that_binarizes_to_a_sliver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Export drops a mask whose geometry is a collinear ring and counts it."""
    from tcip_annotation.state import Polygon
    from tcip_mcp.pipelines.postprocessing import export

    monkeypatch.setattr(
        export,
        "_mask_geometry_for_export",
        lambda mask, bbox_xyxy, subject, *, image_size=None: Polygon(
            rings=[[(5, 10), (8, 10), (12, 10)]]
        ),
    )
    result = {
        "image": "img.jpg",
        "width": 64,
        "height": 64,
        "cap": 2,
        "boxes": [[5.0, 10.0, 12.0, 12.0]],
        "scores": [0.9],
        "labels": [1],
        "masks": [{"segmentation": []}],
    }
    anns, dropped = _encoded(result)

    assert dropped == 1
    assert anns == []


def test_export_drops_a_polygon_whose_vertices_all_round_to_one_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Export drops a polygon whose vertices all round to one point.

    A polygon with real raw extent that collapses to one point at the document's stored
    2-decimal grid is dropped and counted.
    """
    from tcip_annotation.state import Polygon
    from tcip_mcp.pipelines.postprocessing import export

    monkeypatch.setattr(
        export,
        "_mask_geometry_for_export",
        lambda mask, bbox_xyxy, subject, *, image_size=None: Polygon(
            rings=[[(0.996, 0.996), (1.004, 0.996), (1.004, 1.004)]]
        ),
    )
    result = {
        "image": "img.jpg",
        "width": 64,
        "height": 64,
        "cap": 2,
        "boxes": [[1.0, 1.0, 2.0, 2.0]],
        "scores": [0.9],
        "labels": [1],
        "masks": [{"segmentation": []}],
    }
    anns, dropped = _encoded(result)

    assert dropped == 1
    assert anns == []


def test_export_no_masks_key_writes_bbox_as_before(tmp_path: Path) -> None:
    """A result carrying no masks key exports a box.

    A pass collecting no masks returns a result with no masks key, and it exports as one BBox.
    """
    from tcip_annotation.state import BBox

    result = _predicted(
        tmp_path, 32, [[1, 1, 5, 5]], box=[1, 1, 5, 5], masks=False
    )
    assert "masks" not in result
    anns, _dropped = _encoded(result)
    assert len(anns) == 1
    assert isinstance(anns[0].geometry, BBox)
