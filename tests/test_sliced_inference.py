"""Sliced inference through SAHI.

Covers one lattice, one merge, both source kinds, and one execution record from the assessment
through the bucket published under it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tests._training_values import (
    DISPERSION_QUANTILE,
    EQUIVALENCE_Z,
    MASK_CUT,
    RATIONALE,
    collection_pass,
    drawn_split,
    tiling_block,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from sahi.postprocess.combine import PostprocessPredictions

    from tcip_mcp.model_registry import VerifiedCheckpoint
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.execution import (
        Execution,
        MaskCut,
        Pass,
        Preparation,
        Reference,
    )
    from tcip_mcp.pipelines.inference.generic_predictor import (
        GenericPredictor,
        SliceBatch,
        WindowedRasterReader,
    )

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.execution import EXECUTION_RECORD, Stated  # noqa: E402

pytest.importorskip("sahi")

TILE = 128
OVERLAP = 0.25
# A 200px frame at 128px tiles and 0.25 overlap is a 2x2 lattice whose second column and row start
# at 72: the seam blob crosses x=128, and the wide blob is wider than any slice.
FRAME = 200
SEAM_BLOB = (90, 20, 130, 60)
INNER_BLOB = (150, 140, 180, 170)
WIDE_BLOB = (20, 95, 190, 120)
BLOBS = (SEAM_BLOB, INNER_BLOB, WIDE_BLOB)


TRAINING_HEIGHT = 180
"""The height of a training frame: :data:`FRAME` cropped short of square, its blobs kept whole."""


def _checkpoint(
    tmp_path: Path,
    *,
    in_chans: int = 3,
    with_masks: bool = False,
    by_band: bool = False,
    tiling: dict | None = None,
) -> tuple[str, VerifiedCheckpoint]:
    """Return a checkpoint's path and its load through the platform's own loader.

    Two frames, ``_frame(bands=in_chans)`` cut to :data:`TRAINING_HEIGHT` rows, are written
    with a ``bud`` annotation per :data:`BLOBS` entry (a polygon with ``with_masks``, else a
    box), and a run under ``tmp_path`` of ``BLOB_BUILDER`` over them, with ``tiling`` passed
    through, registers the checkpoint by completing. With ``by_band`` the frames' registry
    declares a categorical ``color`` attribute (red, green, blue) and every annotation is
    labeled ``red``.
    """
    import tifffile

    from tcip_annotation.state import Annotation, BBox, Polygon
    from tcip_mcp import subject_registry as cr
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tests._chain_fixtures import BLOB_BUILDER
    from tests._producer_fixtures import label_image, registry_over
    from tests._verified_checkpoint_fixtures import (
        fixture_data_dir,
        registered_checkpoint,
    )

    task = "instance_seg" if with_masks else "detection"
    dataset = fixture_data_dir(tmp_path, "blobs")
    if by_band:
        color = cr.Attribute("color", "categorical", ("red", "green", "blue"))
        registry_over(
            dataset,
            cr.SubjectRegistry(
                subjects=(
                    cr.Subject(
                        name="bud", description="", attributes=(color,)
                    ),
                )
            ),
        )
    images = dataset / "images" / UNDATED_BUCKET
    images.mkdir(parents=True)
    annotations = [
        Annotation(
            subject="bud",
            attributes={"color": "red"} if by_band else {},
            geometry=Polygon([[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]])
            if with_masks
            else BBox(x0, y0, x1, y1),
        )
        for x0, y0, x1, y1 in BLOBS
    ]
    for index in range(2):
        frame = images / f"frame{index}.tif"
        tifffile.imwrite(str(frame), _frame(bands=in_chans)[:TRAINING_HEIGHT])
        label_image(frame, annotations, FRAME, TRAINING_HEIGHT)
    ckpt = registered_checkpoint(
        tmp_path,
        model_source={
            **BLOB_BUILDER,
            "builder_kwargs": {"with_masks": with_masks},
            "task": task,
        },
        data={
            "images_dir": str(images),
            "num_channels": in_chans,
            "scope": {"subject": "bud"},
            "tiling": tiling,
        },
    )
    return ckpt, load_registered_checkpoint(ckpt, project=tmp_path)


def _frame(
    bands: int = 3,
    *,
    value: int | tuple[int, ...] | np.ndarray = 255,
    blobs: Sequence[tuple[int, int, int, int]] = BLOBS,
) -> np.ndarray:
    """Return a :data:`FRAME` square array with ``bands`` bands stacked on the last axis.

    Each band is ``painted_array`` over ``blobs`` at that band's ``value``; ``value`` is one per
    band, or one broadcast to all bands.
    """
    from tests._producer_fixtures import painted_array

    return np.stack(
        [
            painted_array(FRAME, FRAME, [(b, int(v)) for b in blobs])
            for v in np.broadcast_to(np.asarray(value), (bands,))
        ],
        axis=-1,
    )


def _pass(
    checkpoint: VerifiedCheckpoint,
    reference: Reference | None = None,
    **stated: bool | int | float | str | MaskCut | None,
) -> Pass:
    """Return the tiled pass for ``checkpoint`` at the fixture's tile edge and overlap.

    The pass merges by NMM at 0.5 at conf 0.0 and, for an instance segmentation
    checkpoint, collects masks cut at the sample threshold, with ``stated`` over those, made
    runnable from ``reference`` when one is given.
    """
    from tcip_mcp.pipelines.execution import prepare

    values = dict(
        tile=True,
        tile_size=TILE,
        overlap=OVERLAP,
        postprocess="nmm",
        cross_tile_nms=0.5,
        conf=0.0,
        rationale=RATIONALE,
    )
    if checkpoint.task == "instance_seg":
        values.update(masks=True, mask_cut=MASK_CUT)
    values.update(stated)
    return prepare(
        checkpoint, Stated(**values), device="cpu", tile_batch_size=2
    ).runnable(reference)


def _sliced(
    p: Pass,
    source: str | Path | BandGroupRef | WindowedRasterReader,
    **kwargs: str | SliceBatch | Callable[[int, int, SliceBatch], None] | None,
) -> dict:
    return p.predictor.predict_sliced(
        source,
        execution=p.execution,
        tile_batch_size=p.tile_batch_size,
        **kwargs,
    )


def _whole(p: Pass, source: str) -> dict:
    """Return the untiled record of ``source`` under the pass's own execution values.

    The values are the pass's own conf, mask collection and mask cut and the checkpoint's
    density.
    """
    from tcip_mcp.pipelines.execution import execution_record

    masks = p.execution.masks if p.checkpoint.task == "instance_seg" else None
    return p.predictor.predict(
        source,
        execution_record(
            p.checkpoint,
            Stated(
                conf=p.execution.conf,
                masks=masks,
                mask_cut=p.execution.mask_cut,
                rationale=RATIONALE,
            ),
            None,
            None,
        ),
    )


def _png(directory: Path, arr: np.ndarray, name: str = "frame.png") -> str:
    from PIL import Image

    path = directory / name
    Image.fromarray(arr).save(path)
    return str(path)


@pytest.mark.parametrize(
    "height,width,tile,overlap",
    [
        (FRAME, 333, TILE, OVERLAP),
        (
            200,
            200,
            64,
            0.2,
        ),  # 64 * 0.2 is fractional, so rounding the stride either way differs
    ],
)
def test_slice_lattice_states_sahis_own_boxes(
    height: int, width: int, tile: int, overlap: float
) -> None:
    from sahi.slicing import get_slice_bboxes

    from tcip_mcp.pipelines.slicing import slice_lattice

    ours = slice_lattice(height, width, tile, overlap)
    theirs = get_slice_bboxes(
        image_height=height,
        image_width=width,
        slice_height=tile,
        slice_width=tile,
        auto_slice_resolution=False,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
    )
    assert ours == [tuple(b) for b in theirs]


def test_objects_across_a_seam_predict_once_through_both_source_kinds(
    tmp_path: Path,
) -> None:
    """Objects across a seam predict once through both source kinds.

    The whole-decode path and the windowed path merge to the same boxes, the wide blob no slice
    holds whole predicted once over its full extent.
    """
    from tcip_mcp.pipelines.raster_source import open_raster

    _path, checkpoint = _checkpoint(tmp_path)
    p = _pass(checkpoint)
    npy = tmp_path / "frame.npy"
    np.save(npy, _frame())

    whole = _sliced(p, _png(tmp_path, _frame()))
    with open_raster(str(npy), 3) as reader:
        windowed = _sliced(p, reader, source_label="frame")

    expected = sorted(list(map(float, b)) for b in BLOBS)
    assert whole["tiles"] == windowed["tiles"] == 4
    assert sorted(whole["boxes"]) == expected
    assert sorted(windowed["boxes"]) == expected


def test_a_five_band_slice_reaches_the_model_with_every_band(
    tmp_path: Path,
) -> None:
    """A five-band slice reaches the model with five channels and its bands' values.

    All eight model calls see five channels, at least one call's per-band peaks equal the five
    band values over 255, and the whole-source pass predicts one box per blob.
    """
    from tcip_mcp.pipelines.raster_source import open_raster

    band_values = [255, 200, 150, 100, 60]
    _path, checkpoint = _checkpoint(tmp_path, in_chans=5)
    p = _pass(checkpoint)
    npy = tmp_path / "five.npy"
    np.save(
        npy, _frame(bands=5, value=np.asarray(band_values, dtype=np.uint8))
    )

    whole = _sliced(p, str(npy))
    with open_raster(str(npy), 5) as reader:
        _sliced(p, reader, source_label="five")

    assert p.predictor.model.seen_channels == [5] * 8
    expected = pytest.approx([v / 255 for v in band_values], abs=1e-6)
    assert any(
        peaks == expected for peaks in p.predictor.model.seen_band_peaks
    )
    assert len(whole["boxes"]) == len(BLOBS)


def _mask_extents(record: dict) -> list[tuple]:
    extents = []
    for mask in record["masks"]:
        assert len(mask["segmentation"]) == 1
        xs, ys = mask["segmentation"][0][0::2], mask["segmentation"][0][1::2]
        extents.append((min(xs), min(ys), max(xs), max(ys)))
    return sorted(extents)


def test_instance_masks_merged_across_a_seam_are_one_polygon_per_object_whatever_its_values(
    tmp_path: Path,
) -> None:
    """Instance masks merged across a seam are one polygon per object across differing values.

    Calls of one footprint wider than any slice, differing in their attribute value: every
    slice's partial masks merge into one polygon spanning the whole object, carrying one value of
    its attribute.
    """
    _path, checkpoint = _checkpoint(tmp_path, with_masks=True, by_band=True)
    x0, y0, x1, y1 = WIDE_BLOB

    result = _sliced(
        _pass(checkpoint),
        _png(tmp_path, _frame(value=(255, 255, 0), blobs=(WIDE_BLOB,))),
    )

    assert result["labels"] == [1]
    assert result["attributes"][0][0] in (0, 1)
    # Contours run through pixel centers, so a blob's polygon ends one pixel inside its box.
    assert _mask_extents(result) == [(x0, y0, x1 - 1, y1 - 1)]


def test_an_untiled_record_carries_the_polygons_the_sliced_record_does(
    tmp_path: Path,
) -> None:
    _path, checkpoint = _checkpoint(tmp_path, with_masks=True)
    p = _pass(checkpoint)
    path = _png(tmp_path, _frame())

    whole = _whole(p, path)
    sliced = _sliced(p, path)

    assert whole["count"] == sliced["count"] == len(BLOBS)
    assert _mask_extents(whole) == _mask_extents(sliced)
    assert sorted(whole["boxes"]) == sorted(sliced["boxes"])
    assert whole["cap"] == sliced["cap"] > whole["count"]


def test_a_predicted_mask_carries_the_rings_the_shared_extractor_draws(
    tmp_path: Path,
) -> None:
    """A predicted mask carries the rings ``mask_to_polygon_rings`` draws from the same mask."""
    from tcip_annotation.mask_contours import mask_to_polygon_rings

    _path, checkpoint = _checkpoint(tmp_path, with_masks=True)
    # A disk, whose contour the shared extractor simplifies where a rectangle's has nothing to drop.
    yy, xx = np.mgrid[:FRAME, :FRAME]
    disk = ((xx - 100) ** 2 + (yy - 100) ** 2 <= 40**2).astype(np.uint8)
    path = _png(
        tmp_path, np.repeat(disk[..., None] * 255, 3, axis=2), "disk.png"
    )
    expected = [
        [c for point in ring for c in point]
        for ring in mask_to_polygon_rings(
            disk, threshold=None, vertex_spacing=MASK_CUT.vertex_spacing
        )
    ]

    record = _whole(_pass(checkpoint), path)

    assert [m["segmentation"] for m in record["masks"]] == [expected]


def test_masks_are_cut_at_the_stated_threshold_the_record_names(
    tmp_path: Path,
) -> None:
    """A pass cuts masks at the stated threshold its execution record names.

    At the sample cut the masks carry as many distinct rings as there are blobs; cut at a
    ``mask_binarize`` of zero, they carry one ring, none of the sample cut's, and the execution
    record names the stated cut with ``mask_binarize`` and ``vertex_spacing`` sourced
    ``explicit``.
    """
    from tcip_mcp.pipelines.execution import MaskCut

    _path, checkpoint = _checkpoint(tmp_path, with_masks=True)
    path = _png(tmp_path, _frame())
    every_pixel = MaskCut(
        mask_binarize=0.0, vertex_spacing=MASK_CUT.vertex_spacing
    )
    at_zero = _pass(checkpoint, mask_cut=every_pixel)

    kept = _whole(_pass(checkpoint), path)
    cut = _whole(at_zero, path)

    def rings(record: dict) -> list[str]:
        return [repr(m["segmentation"]) for m in record["masks"]]

    assert len(set(rings(kept))) == len(BLOBS)
    assert len(set(rings(cut))) == 1 and not set(rings(cut)) & set(rings(kept))
    assert (
        at_zero.execution.mask_cut,
        at_zero.execution.sources["mask_binarize"],
        at_zero.execution.sources["vertex_spacing"],
    ) == (every_pixel, "explicit", "explicit")


def test_an_instance_segmentation_pass_collecting_no_masks_evaluates_by_box(
    tmp_path: Path,
) -> None:
    """An instance segmentation pass collecting no masks evaluates by box.

    An instance segmentation checkpoint evaluated under a statement collecting no masks scores
    its boxes by box, the choice its execution record carries, and one collecting them scores
    by mask.
    """
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._training_values import IOU_THRESHOLD, collection_pass
    from tests._verified_checkpoint_fixtures import fixture_data_dir

    ckpt, _checkpoint_record = _checkpoint(tmp_path, with_masks=True)
    images = fixture_data_dir(tmp_path, "blobs") / "images" / UNDATED_BUCKET

    def evaluated(
        **stated: bool | int | float | str | MaskCut | None,
    ) -> dict:
        result = evaluate_model(
            tmp_path,
            ckpt,
            str(images),
            stated=collection_pass(**stated),
            iou_threshold=IOU_THRESHOLD,
        )
        assert "error" not in result, result
        return result

    boxes = evaluated(masks=False)
    masks = evaluated(masks=True, mask_cut=MASK_CUT)

    assert boxes["execution"]["masks"] is False
    assert boxes["governing_criterion"]["kind"] == "iou_match"
    assert masks["governing_criterion"]["kind"] == "mask_iou_match"


def test_an_untiled_detector_record_carries_no_masks(tmp_path: Path) -> None:
    _path, checkpoint = _checkpoint(tmp_path)

    record = _whole(_pass(checkpoint), _png(tmp_path, _frame()))

    assert "masks" not in record and record["count"] == len(BLOBS)


@pytest.mark.parametrize("with_masks", [False, True])
def test_a_windowed_pass_resumed_mid_raster_matches_an_uninterrupted_one(
    tmp_path: Path, with_masks: bool
) -> None:
    """A windowed pass resumed mid-raster matches an uninterrupted one.

    A pass interrupted with two slices recorded, resumed, runs the model on the remaining two
    slices and answers the uninterrupted pass's result, every mask carrying a segmentation.
    """
    from tcip_mcp.pipelines.inference.generic_predictor import SliceBatch
    from tcip_mcp.pipelines.raster_source import open_raster

    _path, checkpoint = _checkpoint(tmp_path, with_masks=with_masks)
    p = _pass(checkpoint)
    npy = tmp_path / "frame.npy"
    np.save(npy, _frame())
    recorded: list[SliceBatch] = []

    class PassInterruptedError(Exception):
        pass

    def record_then_stop(_start: int, _end: int, batch: SliceBatch) -> None:
        recorded.append(batch)
        raise PassInterruptedError

    with open_raster(str(npy), 3) as reader:
        uninterrupted = _sliced(p, reader)
        with pytest.raises(PassInterruptedError):
            _sliced(p, reader, progress=record_then_stop)
        prior = SliceBatch(
            slices=[s for b in recorded for s in b.slices],
            predictions=[row for b in recorded for row in b.predictions],
        )
        seen_before = len(p.predictor.model.seen_channels)
        resumed = _sliced(p, reader, prior=prior)

    assert len(prior.slices) == 2
    assert len(p.predictor.model.seen_channels) - seen_before == 2
    assert resumed == uninterrupted
    if with_masks:
        assert all(m["segmentation"] for m in resumed["masks"])


def test_training_and_inference_slice_one_frame_on_one_lattice(
    tmp_path: Path,
) -> None:
    """Training and inference slice one frame on one lattice.

    The tiled training dataset's slices and a sliced inference pass's recorded slices over the
    same frame agree, each read whole off its own route's record.
    """
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.raster_source import open_raster
    from tests._producer_fixtures import dataset_over, label_image

    images = tmp_path / "images" / UNDATED_BUCKET
    images.mkdir(parents=True)
    np.save(images / "frame.npy", _frame())
    label_image(
        images / "frame.npy",
        [Annotation(subject="bud", geometry=BBox(*SEAM_BLOB))],
        FRAME,
        FRAME,
        keep_empty=True,
    )
    ds = dataset_over(
        "detection",
        str(images),
        subject="bud",
        stated={"num_channels": 3},
        tiling=tiling_block(tile_size=TILE, overlap=OVERLAP, sliver_frac=0.5),
    )  # stated: one box derives no spread

    _path, checkpoint = _checkpoint(tmp_path)
    recorded: list[tuple[int, int, int, int]] = []
    with open_raster(str(images / "frame.npy"), 3) as reader:
        _sliced(
            _pass(checkpoint),
            reader,
            progress=lambda _s, _e, batch: recorded.extend(batch.slices),
        )

    assert [box for _stem, box in ds.tile_entries] == recorded


def test_a_postprocess_outside_the_vocabulary_refuses_by_name(
    tmp_path: Path,
) -> None:
    _path, checkpoint = _checkpoint(tmp_path)

    with pytest.raises(ValueError, match="greedynmm"):
        _pass(checkpoint, postprocess="soft-nms")
    admitted = _pass(checkpoint, postprocess="greedynmm")
    assert len(_sliced(admitted, _png(tmp_path, _frame()))["boxes"]) == len(
        BLOBS
    )


def test_a_tiled_pass_stating_no_merge_refuses_naming_postprocess(
    tmp_path: Path,
) -> None:
    """A tiled pass stating no merge refuses naming ``postprocess``.

    No merge is chosen for a tiled pass: one stating none refuses naming ``postprocess``, and
    the record of one stating it keeps the merge with no source label beside it.
    """
    from tcip_mcp.pipelines.execution import ExecutionRefusedError

    _path, checkpoint = _checkpoint(tmp_path)

    with pytest.raises(ExecutionRefusedError, match="postprocess"):
        _pass(checkpoint, postprocess=None)
    admitted = _pass(checkpoint, postprocess="nms")
    assert admitted.execution.postprocess == "nms"
    assert "postprocess" not in admitted.execution.sources
    assert admitted.execution.rationale == RATIONALE


def test_the_merge_leaves_the_process_environment_as_it_found_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sliced merge leaves ``CUDA_VISIBLE_DEVICES`` unset in the process environment."""
    import os

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _path, checkpoint = _checkpoint(tmp_path)

    _sliced(_pass(checkpoint), _png(tmp_path, _frame()))

    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_a_stated_merge_threshold_never_reaches_the_detectors_own_nms(
    tmp_path: Path,
) -> None:
    _path, checkpoint = _checkpoint(tmp_path)

    p = _pass(checkpoint, postprocess="nms", cross_tile_nms=0.9)

    assert (
        p.execution.cross_tile_nms,
        p.execution.sources["cross_tile_nms"],
    ) == (0.9, "explicit")
    assert p.predictor.model.nms_thresh == 0.45


def test_an_iou_merge_derives_its_threshold_and_an_ios_merge_states_one(
    tmp_path: Path,
) -> None:
    """An IoU merge derives its threshold from a reference; an IoS merge refuses one unstated.

    A box nested in another overlaps it at IoU 0.25 and IoS 1: suppression by IoU derives,
    from a reference the pass is handed, a threshold the pair stays apart under, and a pass holding
    no reference refuses it unstated; a merge by IoS stated without a threshold refuses with or
    without a reference, and merges the pair under the threshold stated for it.
    """
    from sahi.prediction import ObjectPrediction

    from tcip_mcp.pipelines.derivations import CROSS_TILE_NMS_DERIVATION
    from tcip_mcp.pipelines.execution import ExecutionRefusedError, Reference
    from tcip_mcp.pipelines.slicing import cross_tile_merge
    from tests._verified_checkpoint_fixtures import objects_over

    _path, checkpoint = _checkpoint(tmp_path)
    nested = Reference(
        regions=[
            objects_over(
                [[0.0, 0.0, 20.0, 20.0], [5.0, 5.0, 15.0, 15.0]], FRAME * FRAME
            )
        ]
    )
    predictions = [
        ObjectPrediction(bbox=[0, 0, 20, 20], category_id=1, score=0.9),
        ObjectPrediction(bbox=[5, 5, 15, 15], category_id=1, score=0.8),
    ]

    with pytest.raises(ExecutionRefusedError, match="cross_tile_nms"):
        _pass(checkpoint, postprocess="nms", cross_tile_nms=None)
    p = _pass(checkpoint, nested, postprocess="nms", cross_tile_nms=None)
    assert p.execution.sources["cross_tile_nms"] == CROSS_TILE_NMS_DERIVATION
    assert p.execution.cross_tile_nms == pytest.approx(0.30)
    assert len(cross_tile_merge(p.execution)(predictions)) == 2

    for reference in (None, nested):
        with pytest.raises(ExecutionRefusedError, match="cross_tile_nms"):
            _pass(
                checkpoint, reference, postprocess="nmm", cross_tile_nms=None
            )
    stated = _pass(checkpoint, nested, postprocess="nmm", cross_tile_nms=0.8)
    assert stated.execution.sources["cross_tile_nms"] == "explicit"
    assert len(cross_tile_merge(stated.execution)(predictions)) == 1


def test_a_preparation_runs_nothing_and_its_pass_merges_at_the_threshold_it_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preparation exposes no ``predict``, and its pass merges at the threshold it resolved.

    ``prepare`` returns a ``Preparation`` with no ``predict`` attribute and no ``Pass`` among
    its attributes. Made runnable, its pass carries the merge threshold it resolved, stated (0.5)
    or derived from an overlapping reference, and predicting two sources builds the merger with
    that threshold for each; a threshold with no statement and no reference, or one a reference
    of two apart boxes cannot derive, refuses naming ``cross_tile_nms``.
    """
    import tcip_mcp.pipelines.slicing as slicing
    from tcip_mcp.pipelines.derivations import CROSS_TILE_NMS_DERIVATION
    from tcip_mcp.pipelines.execution import (
        ExecutionRefusedError,
        Pass,
        Preparation,
        Reference,
        prepare,
    )

    _path, checkpoint = _checkpoint(tmp_path)
    received: list = []
    real_merge = slicing.cross_tile_merge

    def recording_merge(execution: Execution) -> PostprocessPredictions:
        received.append(execution.cross_tile_nms)
        return real_merge(execution)

    monkeypatch.setattr(slicing, "cross_tile_merge", recording_merge)
    sources = [
        _png(tmp_path, _frame(), "a.png"),
        _png(tmp_path, _frame(blobs=(SEAM_BLOB,)), "b.png"),
    ]

    def prepared(cross_tile_nms: float | None) -> Preparation:
        prep = prepare(
            checkpoint,
            Stated(
                tile=True,
                tile_size=TILE,
                overlap=OVERLAP,
                postprocess="nms",
                cross_tile_nms=cross_tile_nms,
                conf=0.0,
                rationale=RATIONALE,
            ),
            device="cpu",
            tile_batch_size=2,
        )
        assert isinstance(prep, Preparation) and not hasattr(prep, "predict")
        assert not any(isinstance(v, Pass) for v in vars(prep).values())
        return prep

    from tests._verified_checkpoint_fixtures import objects_over

    overlapping = Reference(
        regions=[
            objects_over(
                [[0.0, 0.0, 20.0, 20.0], [5.0, 5.0, 15.0, 15.0]], FRAME * FRAME
            )
        ]
    )
    apart = Reference(
        regions=[
            objects_over(
                [[0.0, 0.0, 20.0, 20.0], [100.0, 100.0, 120.0, 120.0]],
                FRAME * FRAME,
            )
        ]
    )
    for cross_tile_nms, reference, source in (
        (0.5, None, "explicit"),
        (None, overlapping, CROSS_TILE_NMS_DERIVATION),
    ):
        received.clear()
        p = prepared(cross_tile_nms).runnable(reference)
        assert p.execution.sources["cross_tile_nms"] == source
        p.predict(sources)
        assert received == [p.execution.cross_tile_nms] * 2
        assert all(isinstance(t, float) for t in received)
    for reference in (None, apart):
        with pytest.raises(ExecutionRefusedError, match="cross_tile_nms"):
            prepared(None).runnable(reference)


def test_the_dry_run_reports_the_record_the_bucket_keeps(
    tmp_path: Path,
) -> None:
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.tools.inference_tools import run_inference

    ckpt, _checkpoint_record = _checkpoint(tmp_path)
    images = tmp_path / "images" / UNDATED_BUCKET
    images.mkdir(parents=True)
    _png(images, _frame())
    stated = Stated(
        tile=True,
        tile_size=TILE,
        overlap=OVERLAP,
        conf=0.2,
        cross_tile_nms=0.4,
        postprocess="greedynmm",
        rationale=RATIONALE,
    )

    dry = run_inference(
        tmp_path,
        ckpt,
        images_dir=str(images),
        bucket="sliced",
        dry_run=True,
        stated=stated,
    )
    real = run_inference(
        tmp_path, ckpt, images_dir=str(images), bucket="sliced", stated=stated
    )

    assert "error" not in dry and "error" not in real, (dry, real)
    assert dry["execution"] == EXECUTION_RECORD.dump_python(
        read_bucket(real["dataset_root"], "sliced").execution, mode="json"
    )


N_CALIBRATION_IMAGES = 20


def _blob_capture(project: Path) -> Path:
    """Ingest a capture of blob frames into the dataset ``ds`` and return its images directory.

    It writes :data:`N_CALIBRATION_IMAGES` blob frames, each labeled ``bud`` by one box per
    blob, ingests them into ``project / "ds"`` and states them one capture date.
    """
    import datetime

    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.ingest_tools import ingest_images, state_capture_date
    from tests._producer_fixtures import label_image

    root, raw, date = project / "ds", project / "raw", "2026-01-01"
    raw.mkdir()
    labels = {}
    for i in range(N_CALIBRATION_IMAGES):
        corners = [
            (40 + col * 22 + i, y0) for y0 in (20, 80, 140) for col in range(4)
        ]
        boxes = [BBox(x0 - 6, y0 - 6, x0 + 26, y0 + 26) for x0, y0 in corners]
        _png(
            raw,
            _frame(blobs=[(x0, y0, x0 + 20, y0 + 20) for x0, y0 in corners]),
            f"img{i:02d}.png",
        )
        labels[f"img{i:02d}.png"] = [
            Annotation(subject="bud", geometry=b) for b in boxes
        ]
    ingested = ingest_images(root, source=str(raw))
    assert "error" not in ingested, ingested
    dated = state_capture_date(
        root,
        images=[f["dest"] for f in ingested["files"]],
        date=datetime.date.fromisoformat(date),
        rationale="the frames stand for one capture of that day",
    )
    assert "error" not in dated, dated
    images = root / "images" / date
    for name, boxed in labels.items():
        label_image(images / name, boxed, FRAME, FRAME, keep_empty=True)
    return images


def test_an_assessed_tiled_pass_and_its_bucket_run_one_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An assessed tiled pass and the bucket published under it run one merge.

    A tiled assessment merging by IoS at its stated threshold merges every reference slice and
    every slice of the bucket published under it at that one value, and the bucket records the
    assessment's execution.
    """
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tcip_mcp.tools.calibration_tools import assess_checkpoint
    from tcip_mcp.tools.data_tools import draw_splits
    from tcip_mcp.tools.inference_tools import run_inference
    from tests import _trait_fixtures as fx

    fx.seed_confirmed_count(tmp_path, measured_subject="bud")
    images = _blob_capture(tmp_path)
    selection = tmp_path / "selection"
    drawn = draw_splits(
        tmp_path,
        str(tmp_path / "ds"),
        output_path=str(selection),
        subject="bud",
        **drawn_split(
            seed=2,
            val_ratio=0.2,
            calibration_ratio=0.2,
            holdout_ratio=0.2,
            grouping="tile_prefix",
        ),
    )
    assert "error" not in drawn, drawn
    ckpt, _record = _checkpoint(tmp_path)
    merged_at: list[float] = []
    real_predict_sliced = GenericPredictor.predict_sliced

    def _capture(
        self: GenericPredictor,
        source: str | Path | BandGroupRef | WindowedRasterReader,
        **kwargs: Execution
        | int
        | str
        | dict
        | Callable[[int, int, dict], None]
        | None,
    ) -> dict:
        merged_at.append(kwargs["execution"].cross_tile_nms)
        return real_predict_sliced(self, source, **kwargs)

    monkeypatch.setattr(GenericPredictor, "predict_sliced", _capture)

    assessment = assess_checkpoint(
        tmp_path,
        checkpoint_path=ckpt,
        trait=fx.COUNT_TRAIT,
        delivery_kind="per_image_count",
        selection_dir=str(selection),
        equivalence_z=EQUIVALENCE_Z,
        dispersion_quantile=DISPERSION_QUANTILE,
        rationale=RATIONALE,
        device="cpu",
        stated=collection_pass(
            tile=True,
            tile_size=TILE,
            overlap=OVERLAP,
            postprocess="nmm",
            cross_tile_nms=0.8,
        ),
    )
    assert "error" not in assessment, assessment
    reference_passes = len(merged_at)
    published = run_inference(
        tmp_path,
        ckpt,
        stated=Stated(),
        images_dir=str(images),
        bucket="blob/2026-01-01",
        assessment_id=assessment["assessment_id"],
        device="cpu",
    )
    assert "error" not in published, published

    execution = assessment["execution"]
    assert execution["sources"]["cross_tile_nms"] == "explicit"
    assert reference_passes > 0 and len(merged_at) > reference_passes
    assert set(merged_at) == {execution["cross_tile_nms"]}
    assert (
        EXECUTION_RECORD.dump_python(
            read_bucket(
                published["dataset_root"], "blob/2026-01-01"
            ).execution,
            mode="json",
        )
        == execution
    )


@pytest.mark.parametrize(
    "with_masks", [False, True], ids=["detection", "instance_seg"]
)
def test_the_bucket_records_the_slice_geometry_the_checkpoint_derived(
    tmp_path: Path, with_masks: bool
) -> None:
    """A published bucket records the slice geometry its checkpoint derived.

    For a checkpoint of either task recording a tiling, a pass stating no tile publishes a bucket
    whose record states the checkpoint's lattice and its stated NMM merge, which builds SAHI's
    NMM postprocess by IoS at 0.6; a merge stated without its threshold refuses naming
    ``cross_tile_nms``.
    """
    import sahi
    from sahi.predict import POSTPROCESS_NAME_TO_CLASS

    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.pipelines.slicing import cross_tile_merge
    from tcip_mcp.tools.inference_tools import run_inference

    ckpt, _checkpoint_record = (
        _checkpoint(  # sliver_frac stated: three boxes derive no spread
            tmp_path,
            with_masks=with_masks,
            tiling=tiling_block(
                tile_size=TILE, overlap=OVERLAP, sliver_frac=0.5
            ),
        )
    )
    images = tmp_path / "images" / UNDATED_BUCKET
    images.mkdir(parents=True)
    _png(images, _frame())
    cut = {"masks": True, "mask_cut": MASK_CUT} if with_masks else {}

    for merge in ("nms", "nmm"):
        unstated = run_inference(
            tmp_path,
            ckpt,
            images_dir=str(images),
            bucket="sliced",
            stated=Stated(
                conf=0.0, postprocess=merge, rationale=RATIONALE, **cut
            ),
        )
        assert "cross_tile_nms" in unstated.get("error", ""), (merge, unstated)

    response = run_inference(
        tmp_path,
        ckpt,
        images_dir=str(images),
        bucket="sliced",
        stated=Stated(
            conf=0.0,
            postprocess="nmm",
            cross_tile_nms=0.6,
            rationale=RATIONALE,
            **cut,
        ),
    )

    assert "error" not in response, response
    execution = read_bucket(response["dataset_root"], "sliced").execution
    assert (execution.tile_size, execution.overlap) == (TILE, OVERLAP)
    assert execution.postprocess == "nmm"
    assert execution.sahi_version == sahi.__version__
    merger = cross_tile_merge(execution)
    assert type(merger) is POSTPROCESS_NAME_TO_CLASS["NMM"]
    assert (merger.match_metric, merger.match_threshold) == ("IOS", 0.6)
