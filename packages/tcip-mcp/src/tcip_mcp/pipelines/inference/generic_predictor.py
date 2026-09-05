"""Generic predictor for any bespoke ``model_source`` checkpoint.

Auto-detects task from the saved ``model_source``. Supports single image,
batch, and ONNX export.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Protocol, cast

import torch
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from tcip_mcp.model_registry import VerifiedCheckpoint

from tcip_mcp.pipelines.derivations import probe_channels
from tcip_mcp.pipelines.model_build import (
    MODEL_SOURCE_KEY,
    STATE_DICT_KEY,
    build_model,
    declared_in_chans,
)
from tcip_mcp.pipelines.image_utils import (
    BandGroupRef, display_source_path, load_image, pad_tile, pil_to_tensor,
)
from tcip_mcp.pipelines.inference.predictor import KIND_TCIP_MODULE
from tcip_mcp.pipelines.resolution import DEFAULT_NMS_IOU

logger = logging.getLogger(__name__)

# Detection task names that format outputs as boxes/scores/labels. A bespoke model_source declares
# the task type ``detection`` / ``instance_seg``, both route through the detection formatter.
_DETECTION_TASKS = frozenset({"detection", "instance_seg"})


from tcip_mcp.pipelines.image_utils import crop_pad_tile as _crop_pad_tile  # noqa: E402


class WindowedRasterReader(Protocol):
    """The read surface a huge-raster tile source must expose for :meth:`GenericPredictor.
    predict_tiled`: full-raster pixel dimensions, band count, and a windowed decode.
    Duck-typed rather than importing a concrete reader here, so this stays usable for any raster
    too large to load whole, not just the ``pipelines/raster_source.py`` backends (each of which
    exposes this surface; :class:`~tcip_mcp.pipelines.raster_source.GdalSource` is the one a huge
    GeoTIFF opens as).
    """

    height: int
    width: int
    num_channels: int

    def read_window(self, y0: int, y1: int, x0: int, x1: int) -> "np.ndarray": ...


class GenericPredictor:
    """Load any bespoke ``model_source`` checkpoint and run inference.

    The checkpoint must carry the model reference and the weights (``model_build``'s
    ``MODEL_SOURCE_KEY`` / ``STATE_DICT_KEY``).
    Task type is read from the model_source.

    The input geometry the run trained at travels on the checkpoint's embedded config and is
    exposed as-recorded: ``train_tile_size``/``train_overlap`` (a tiled run's tile lattice),
    ``train_native_size`` (the one frame size an untiled run's frames all shared, ``[width,
    height]``), and ``train_augmentation`` (the augmentation config that run declared, a dict or a
    preset name). Turning those into a tile geometry to infer at is
    :func:`~tcip_mcp.pipelines.inference.predictor.resolve_tile_geometry`'s job, paired with
    :func:`~tcip_mcp.pipelines.inference.predictor.native_ratio_tile_resize` for the resize half;
    nothing here picks a geometry on its own.
    """

    def __init__(
        self,
        checkpoint: "VerifiedCheckpoint",
        device: str | None = None,
        score_threshold: float = 0.5,
        nms_iou: float | None = None,
        max_dets: int | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.score_threshold = score_threshold
        self.max_dets = max_dets

        # Already read and unpickled by load_registered_checkpoint; no re-read here.
        self.checkpoint_path = checkpoint.path
        self.checkpoint_sha256 = checkpoint.sha256
        ckpt = checkpoint.payload
        # A bespoke checkpoint carries the importable-builder ref; build_model re-imports it.
        self.model_source = ckpt.get(MODEL_SOURCE_KEY)
        self.kind = KIND_TCIP_MODULE
        self.config = ckpt.get("config", {})

        # Training tile geometry, so inference can derive the tile scale from the checkpoint instead
        # of a mismatched default. None when this checkpoint carried no tiling geometry.
        _tiling = (self.config.get("data") or {}).get("tiling") or {}
        self.train_tile_size = _tiling.get("tile_size")
        self.train_overlap = _tiling.get("overlap")
        # The untiled counterpart, both as recorded: resolve_tile_geometry/native_ratio_tile_resize
        # are where they become a tile geometry, so a run that never tiles never has to read them.
        self.train_native_size = (self.config.get("data") or {}).get("train_native_size")
        self.train_augmentation = self.config.get("augmentation")

        self.model = build_model(ckpt)  # re-imported bespoke builder (no exec)
        self.model.load_state_dict(ckpt[STATE_DICT_KEY])
        self.model.to(self.device)
        self.model.eval()

        # Make the operating point govern which boxes exist (in-model thresholds), not just a
        # post-hoc filter that can never recover a box the model already discarded (the audit's
        # finding). No-op for non-detection models. See pipelines/operating_point.py.
        from tcip_mcp.pipelines.operating_point import set_detector_operating_point
        set_detector_operating_point(self.model, score_thresh=score_threshold,
                                     nms_thresh=nms_iou, detections_per_img=max_dets)

        # Task + input channels come from the bespoke model_source's declared ``task`` / ``in_chans``.
        src = self.model_source or {}
        self.task = src.get("task", "unknown")
        declared = declared_in_chans(src)
        self.in_chans = declared if declared is not None else 3

    @torch.no_grad()
    def predict(self, image_path: str | Path | BandGroupRef) -> dict:
        """Run inference on a single image.

        ``image_path`` may be a plain path/string or a :class:`BandGroupRef`, the same image
        sources ``image_utils.list_logical_images``/``resolve_image_source`` hand every other
        reader in this platform, so a band-grouped capture decodes through the channel-aware
        loader here too instead of needing its own stringified stand-in.
        """
        img = load_image(image_path, self.in_chans)
        w, h = img.size if isinstance(img, Image.Image) else (img.shape[1], img.shape[0])
        tensor = pil_to_tensor(img).to(self.device)
        disp = display_source_path(image_path)

        if self.task in _DETECTION_TASKS:
            outputs = self.model([tensor])
            if isinstance(outputs, list):
                outputs = outputs[0]
            return self._format_detection(outputs, disp, w, h)
        else:
            outputs = self.model(tensor.unsqueeze(0))
            return self._format_other(outputs, disp, w, h)

    @torch.no_grad()
    def predict_batch(
        self, image_paths: list[str | Path | BandGroupRef], tile: bool = False,
        tile_size: int | None = None, overlap: float = 0.2, tile_batch_size: int = 96,
        global_nms_iou: float = DEFAULT_NMS_IOU, batch_size: int = 16, postprocess: str = "nms",
        *, require_masks: bool = True, tile_resize: tuple[int, int] | None = None,
    ) -> list[dict]:
        """Run inference on multiple images (optionally tiled for small objects).

        For detection, images are run through the detector in batches of ``batch_size``
        (one GPU forward per batch, torchvision detectors take a list of variable-size
        images), instead of one forward per image. Non-detection heads stay per-image
        since their inputs are native-resolution (can't be stacked without resizing).

        Each element of ``image_paths`` may be a plain path/string or a :class:`BandGroupRef`
        (see :meth:`predict`). ``require_masks`` forwards to :meth:`predict_tiled` when ``tile=True``
        (see its own docstring for the tiled mask shape); ignored when ``tile=False``, since the
        untiled path always carries masks for ``instance_seg``. ``tile_size``/``tile_resize`` are
        only meaningful when ``tile=True``: ``tile_size`` is never defaulted here (see
        :meth:`predict_tiled`'s own docstring for why), a caller that tiles without resolving one
        gets that method's own clear refusal, and ``tile_resize`` forwards unchanged.
        """
        if tile:
            return [
                self.predict_tiled(p, tile_size=tile_size, overlap=overlap,
                                   tile_batch_size=tile_batch_size, global_nms_iou=global_nms_iou,
                                   postprocess=postprocess, require_masks=require_masks,
                                   tile_resize=tile_resize)
                for p in image_paths
            ]
        if self.task in _DETECTION_TASKS:
            return self._predict_batch_detection(image_paths, batch_size)
        return [self.predict(p) for p in image_paths]

    @torch.no_grad()
    def _predict_batch_detection(
        self, image_paths: list[str | Path | BandGroupRef], batch_size: int,
    ) -> list[dict]:
        results: list[dict] = []
        for start in range(0, len(image_paths), max(1, batch_size)):
            chunk = image_paths[start:start + max(1, batch_size)]
            tensors, meta = [], []
            for p in chunk:
                img = load_image(p, self.in_chans)
                w, h = img.size if isinstance(img, Image.Image) else (img.shape[1], img.shape[0])
                tensors.append(pil_to_tensor(img).to(self.device))
                meta.append((display_source_path(p), w, h))
            outputs = self.model(tensors)  # one forward over the whole chunk
            for (disp, w, h), out in zip(meta, outputs):
                results.append(self._format_detection(out, disp, w, h))
        return results

    def _tile_model_input(self, crop, tile_resize: tuple[int, int] | None, *,
                          band_interpretations: "tuple[str, ...] | None" = None):
        """One tile as the model should receive it, plus the ``(scale_x, scale_y)`` its coordinates
        were multiplied by getting there.

        ``tile_resize`` given, the tile is stretched to it through the training chain's own
        :class:`~tcip_mcp.pipelines.data.augmentations.Resize` (never a second resize
        implementation), which is a per-axis stretch: the two factors differ whenever the target's
        aspect differs from the tile's, so they travel separately and are never collapsed to one.

        A tile no PIL mode represents faithfully (uint16, 5-band, or a 4-channel tile whose
        alpha-vs-spectral-band status is unresolved) is returned untouched at ``(1.0, 1.0)``: the
        training loader's own transform chain is PIL-only and skipped such a sample too
        (``BaseImageDataset._finalize``), so resizing here would introduce a geometry training
        never applied. ``to_pil_if_faithful`` is that same shared decision, not a second one;
        ``band_interpretations`` (the windowed source's own GDAL color interpretations, absent
        for the whole-decode path since ``load_image`` already resolved it there) is the same
        real signal that decision reads elsewhere, never guessed here either.
        """
        if tile_resize is None:
            return crop, 1.0, 1.0
        from tcip_mcp.pipelines.data.augmentations import Resize
        from tcip_mcp.pipelines.image_utils import to_pil_if_faithful

        pil = to_pil_if_faithful(crop, band_interpretations=band_interpretations)
        if not isinstance(pil, Image.Image):
            return crop, 1.0, 1.0
        target = (int(tile_resize[0]), int(tile_resize[1]))
        resized, _ = Resize(size=target)(pil, {})
        return resized, target[0] / pil.width, target[1] / pil.height

    def _tiled_infer_core(
        self, height: int, width: int, get_tile: Callable[[int, int], object],
        tile_size: int, overlap: float, tile_batch_size: int, global_nms_iou: float,
        postprocess: str, *, require_masks: bool = True,
        tile_resize: tuple[int, int] | None = None,
        band_interpretations: "tuple[str, ...] | None" = None,
        prior: dict | None = None,
        progress: "Callable[[int, int, dict], None] | None" = None,
    ) -> dict:
        """Shared tiling/batching/reconstruction loop behind :meth:`predict_tiled`'s two source
        kinds (a fully decoded in-memory image, or a windowed raster reader): build tile positions
        from ``(height, width)``, pull each tile's pixels via ``get_tile(tile_x,
        tile_y)`` (already cropped-and-padded to ``tile_size`` x ``tile_size``, PIL image or ``[H,
        W, C]`` array), batch them through the model, then ``reconstruct_core``/``global_nms``/
        ``global_merge``. The two callers differ only in how ``get_tile`` sources its pixels; this
        loop is not duplicated between them.

        For ``instance_seg`` with ``require_masks=True`` (the default), each tile's soft masks
        (``outputs["masks"]``, same squeeze convention as :meth:`_format_detection`) travel through
        the same reconstruction/merge in lockstep with their boxes, and the returned dict gains a
        ``masks`` key: a list of ``{"mask_patch", "offset_x", "offset_y"}`` dicts, one per surviving
        detection (tile-local soft-mask patch + its full-image-space origin, never a dense
        full-image-sized array; see :class:`tcip_mcp.pipelines.data.tiling.MaskPatch`). This
        deliberately differs from the untiled result's own ``masks`` (one dense ``[H, W]`` array per
        detection, already in full-image coordinates): a huge orthomosaic tile source can never
        afford one full-raster-sized mask per detection, so the tiled shape stays patch-local and a
        consumer (export, visualization) adds the offset only at the point of use. Any task other
        than ``instance_seg``, or ``require_masks=False``, never collects masks and the returned
        dict carries no ``masks`` key at all (an explicit boxes-only opt-out, still worth taking
        when a caller never reads masks: a mask patch per detection is real extra memory/compute
        across a dense tile grid).

        ``tile_resize`` (a ``(width, height)``) resizes every tile to it before the forward pass and
        undoes that stretch on the way back, per axis (:meth:`_tile_model_input`), inside this loop:
        the boxes and mask patches leaving it are in the tile's own native pixel space, which is the
        space ``reconstruct_core`` shifts to full-image coordinates and runs its core-region keep
        test in. Undoing the stretch after reconstruction instead would run that keep test at the
        wrong scale and change which detections survive. What the model itself does internally to the
        tensor it is handed (a detector's own ``GeneralizedRCNNTransform`` resizes it, then maps its
        boxes back to that tensor's coordinate space) is already undone before this loop sees a box,
        so only the resize applied here is corrected here. ``band_interpretations`` passes through
        to :meth:`_tile_model_input` for the windowed-reader caller only (the whole-decode caller's
        tiles are already the type ``load_image`` resolved them to, with its own real signal).

        ``prior`` (default ``None``) seeds the accumulators with tiles a caller already has, from
        an earlier, interrupted run of this same pass: a mapping with ``tile_info``/``boxes``/
        ``scores``/``labels``, one entry per already-done tile, in the shape a batch record from
        ``progress`` below carries. Every position in ``prior["tile_info"]`` is skipped in the live
        loop, so this resumes rather than repeats it; the merge (reconstruction, cross-tile NMS or
        NMM, the ``max_dets`` cap) runs over ``prior`` and the live tiles together, and ``tiles``
        still reports the whole grid's count either way. ``progress`` (default ``None``), when
        given, is called once per flushed live batch with that batch's own first and last tile
        index into the full position grid and a mapping of its own new ``tile_info``/``boxes``/
        ``scores``/``labels`` (never ``masks``, whatever ``require_masks`` says: resuming a
        mask-bearing pass is a caller's own decision to refuse, this loop stays generic), so a
        caller can persist it and resume from it later. Neither seeds nor reports mask patches.

        Returns ``width``/``height``/``boxes``/``scores``/``labels``/``count``/``tiles`` (``masks``
        when applicable); the caller stamps its own ``image`` field (a display path or a windowed
        reader's own label).
        """
        import numpy as np
        from tcip_mcp.pipelines.data.tiling import (
            compute_stride, tile_positions, reconstruct_core, global_nms, global_merge,
        )

        stride = compute_stride(tile_size, overlap)
        positions = tile_positions(height, width, tile_size, stride)

        # The edge the model is handed, which is the resized one when a tile_resize is applied.
        model_edge = min(int(tile_resize[0]), int(tile_resize[1])) if tile_resize else tile_size
        min_size = ((self.model_source or {}).get("builder_kwargs") or {}).get("min_size")
        if min_size and abs(int(min_size) - model_edge) > model_edge:
            logger.warning("tiled inference: model min_size=%s differs greatly from the %spx tiles "
                           "it is handed (tiles will be rescaled).", min_size, model_edge)

        collect_masks = self.task == "instance_seg" and require_masks

        per_tile_boxes, per_tile_scores, per_tile_labels, tile_info = [], [], [], []
        per_tile_masks: list[np.ndarray] | None = [] if collect_masks else None
        if prior is not None:
            for info, b, s, l in zip(
                prior["tile_info"], prior["boxes"], prior["scores"], prior["labels"]):
                tile_info.append(info)
                per_tile_boxes.append(np.asarray(b, dtype=np.float32).reshape(-1, 4))
                per_tile_scores.append(np.asarray(s, dtype=np.float32))
                per_tile_labels.append(np.asarray(l, dtype=np.int64))
        done_positions = {(info["tile_x"], info["tile_y"]) for info in tile_info}

        batch_tiles: list = []
        batch_meta: list = []
        batch_scales: list = []
        batch_pos_indices: list = []

        def _flush() -> None:
            if not batch_tiles:
                return
            outputs = self.model(batch_tiles)
            for out, meta, (scale_x, scale_y) in zip(outputs, batch_meta, batch_scales):
                keep = out["scores"] >= self.score_threshold
                boxes = out["boxes"][keep].cpu().numpy()
                if (scale_x, scale_y) != (1.0, 1.0):
                    boxes = boxes.copy()
                    boxes[:, [0, 2]] /= scale_x
                    boxes[:, [1, 3]] /= scale_y
                per_tile_boxes.append(boxes)
                per_tile_scores.append(out["scores"][keep].cpu().numpy())
                per_tile_labels.append(out["labels"][keep].cpu().numpy())
                if collect_masks:
                    assert per_tile_masks is not None, "collect_masks is True only when per_tile_masks is a list"
                    m = out["masks"][keep]
                    if m.dim() == 4 and m.shape[1] == 1:  # torchvision MaskRCNN: [N, 1, H, W]
                        m = m[:, 0]
                    if (scale_x, scale_y) != (1.0, 1.0) and len(m):
                        m = torch.nn.functional.interpolate(
                            m.unsqueeze(1).float(), size=(tile_size, tile_size),
                            mode="bilinear", align_corners=False).squeeze(1)
                    per_tile_masks.append(m.cpu().numpy())
                tile_info.append(meta)
            n_new = len(batch_meta)
            if progress is not None and n_new:
                progress(batch_pos_indices[0], batch_pos_indices[-1], {
                    "tile_info": tile_info[-n_new:],
                    "boxes": [b.tolist() for b in per_tile_boxes[-n_new:]],
                    "scores": [s.tolist() for s in per_tile_scores[-n_new:]],
                    "labels": [l.tolist() for l in per_tile_labels[-n_new:]],
                })
            batch_tiles.clear()
            batch_meta.clear()
            batch_scales.clear()
            batch_pos_indices.clear()

        resize_applied = True
        for pos_index, (tile_x, tile_y) in enumerate(positions):
            if (tile_x, tile_y) in done_positions:
                continue
            raw = get_tile(tile_x, tile_y)
            crop, scale_x, scale_y = self._tile_model_input(
                raw, tile_resize, band_interpretations=band_interpretations)
            resize_applied = crop is not raw
            batch_tiles.append(pil_to_tensor(crop).to(self.device))
            batch_scales.append((scale_x, scale_y))
            batch_meta.append({"tile_x": tile_x, "tile_y": tile_y,
                               "original_width": width, "original_height": height})
            batch_pos_indices.append(pos_index)
            if len(batch_tiles) >= tile_batch_size:
                _flush()
        _flush()
        if tile_resize is not None and positions and not resize_applied:
            logger.warning("tiled inference: the checkpoint's recorded train-time resize %s was not "
                           "applied, these tiles are in no PIL mode and the training loader's own "
                           "transform chain skipped such samples too.", tuple(tile_resize))

        if collect_masks:
            assert per_tile_masks is not None, "collect_masks is True only when per_tile_masks is a list"
            boxes, scores, labels, masks = reconstruct_core(
                per_tile_boxes, per_tile_scores, per_tile_labels, tile_info, tile_size, stride,
                per_tile_masks=per_tile_masks)
        else:
            boxes, scores, labels = reconstruct_core(
                per_tile_boxes, per_tile_scores, per_tile_labels, tile_info, tile_size, stride)
            masks = None

        if len(boxes) == 0:
            pass
        elif postprocess == "nmm":
            if collect_masks:
                assert masks is not None, "collect_masks is True only when masks is a list"
                boxes, scores, labels, masks = global_merge(
                    boxes, scores, labels, global_nms_iou, per_det_masks=masks)
            else:
                boxes, scores, labels = global_merge(boxes, scores, labels, global_nms_iou)
        else:
            keep = global_nms(boxes, scores, labels, global_nms_iou)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if collect_masks:
                assert masks is not None, "collect_masks is True only when masks is a list"
                masks = [masks[i] for i in keep]

        # Full-frame cap after the cross-tile merge (highest score first); the in-model
        # detections_per_img only caps per tile. cap_hit uses >=, matching records_from_detector.
        cap_hit = bool(self.max_dets is not None and len(scores) >= self.max_dets)
        if self.max_dets is not None and len(scores) > self.max_dets:
            top = np.argsort(scores)[::-1][: self.max_dets]
            boxes, scores, labels = boxes[top], scores[top], labels[top]
            if collect_masks:
                assert masks is not None, "collect_masks is True only when masks is a list"
                masks = [masks[i] for i in top]

        result = {
            "width": width,
            "height": height,
            "boxes": boxes.tolist(),
            "scores": scores.tolist(),
            "labels": labels.tolist(),
            "count": int(len(boxes)),
            "tiles": len(positions),
            "cap_hit": cap_hit,
        }
        if collect_masks:
            assert masks is not None, "collect_masks is True only when masks is a list"
            result["masks"] = [
                {"mask_patch": mp.patch.tolist(), "offset_x": mp.offset_x, "offset_y": mp.offset_y}
                for mp in masks
            ]
        return result

    def _refuse_channel_mismatch(self, probed: int) -> None:
        if probed != self.in_chans:
            raise ValueError(
                f"source has {probed} channel(s) but the model expects in_chans={self.in_chans}; "
                "refusing to silently truncate/pad the band count the model was trained on."
            )

    def _require_tile_size(self, tile_size: int | None) -> int:
        if tile_size is None:
            raise ValueError(
                "predict_tiled requires an explicit tile_size: this raw tiling primitive never "
                "fabricates one, resolve a real basis (resolve_tile_geometry) before calling."
            )
        return tile_size

    @torch.no_grad()
    def predict_tiled(
        self, source: str | Path | BandGroupRef | WindowedRasterReader, tile_size: int | None = None,
        overlap: float = 0.2, tile_batch_size: int = 96, global_nms_iou: float = DEFAULT_NMS_IOU,
        postprocess: str = "nms", *, require_masks: bool = True, source_label: str = "",
        tile_resize: tuple[int, int] | None = None,
        prior: dict | None = None,
        progress: "Callable[[int, int, dict], None] | None" = None,
    ) -> dict:
        """Tiled (SAHI-style) detection: sliding-window tiles -> per-tile predict ->
        core-region reconstruction -> cross-tile merge -> full-image detections.

        Dispatches on ``source``'s own container layout, never its size, the same principle
        ``raster_source.open_raster`` follows: a plain path/string or :class:`BandGroupRef` decodes
        the whole image into memory once (via ``load_image``) before tiling; a
        :class:`WindowedRasterReader`-shaped object (duck-typed: has ``.read_window``, see that
        Protocol) sources each tile's pixels on demand instead, so a raster too large to load whole
        (a multi-gigabyte orthomosaic) never has to be. ``source_label`` names the windowed case's
        ``image`` result field (a windowed reader carries no single on-disk path the caller doesn't
        already know); ignored for the whole-decode case, whose own display path is used instead.

        ``postprocess`` selects the cross-tile merge: ``"nms"`` suppresses overlaps, ``"nmm"``
        unions boxes split across a seam. A whole-decode non-detection source falls back to
        :meth:`predict`; a windowed-reader source has no untiled fallback (the whole point of that
        path is a raster too large to decode whole), so a non-detection task there raises
        ``ValueError`` instead.

        A source with no safe coercion has its band count checked against ``self.in_chans`` before
        any tile is read: the reader's own declared ``num_channels`` for the windowed case, always
        checked (a raster source is never coerced); :func:`derivations.probe_channels`
        (header-only where possible) for the whole-decode case, but only when ``source`` is not a
        photographic file (:func:`raster_source.photographic_container`) - ``load_image`` converts
        any photographic frame to ``in_chans`` itself (PIL's own RGBA/RGB/L conversion), so a probed
        mismatch there is not a real one. Either way, a genuine mismatch raises ``ValueError``
        rather than silently truncating/padding the band count the model trained on.

        ``tile_size`` has no default: the caller (``resolve_tile_geometry``, upstream of every real
        entry point) resolves a real basis before calling, this raw tiling primitive never
        fabricates one.

        ``tile_resize`` (a ``(width, height)``, default ``None`` = feed each tile as it stands)
        resizes every tile to it before the forward pass and maps the boxes/masks back into the
        tile's own native pixel space per axis, so the result is in the source's real pixel space
        either way. It exists so a run can reproduce the input geometry a checkpoint trained at: the
        native-size ratio tier cuts tiles at the training frame's own size and hands them the same
        resize the training chain applied (``native_ratio_tile_resize``). A tile edge that is
        explicit or from persisted tile geometry passes ``None``: the tile as it stands is what those
        tiers mean.

        ``require_masks`` (default True) governs whether an ``instance_seg`` checkpoint's masks are
        threaded through the cross-tile reconstruction/merge at all: a mask patch per detection is
        real extra memory/compute across a dense tile grid, so a caller that never reads masks
        (``run_full_frame_evaluation`` scores boxes against full-frame GT) can pass
        ``require_masks=False`` to skip it deliberately; the result then carries no ``masks`` key
        for any task. With ``require_masks=True`` and ``instance_seg``, the returned ``masks`` are
        not the untiled result's dense ``[H, W]`` full-image arrays: each is a small tile-local
        patch plus its full-image-space offset (``{"mask_patch", "offset_x", "offset_y"}``, see
        :meth:`_tiled_infer_core`), the shape a source too large to hold one full-size mask per
        detection requires; a consumer must not assume the two ``masks`` shapes are interchangeable.

        ``prior``/``progress`` (both default ``None``) forward to :meth:`_tiled_infer_core` for the
        windowed-reader source only, the resume seam a raster too large to decode whole needs; see
        that method's own docstring. The whole-decode source never resumes (a plain directory-of-
        images pass writes its files all at the end, unchanged), so a caller of that path leaves
        both unset; passing either against a whole-decode source refuses by name rather than
        silently dropping the resume state the caller thinks it handed over.
        """
        if not hasattr(source, "read_window") and (prior is not None or progress is not None):
            raise ValueError(
                "prior/progress apply only to a windowed-reader source: a whole-decode pass writes "
                "its files all at once and has no resume seam to feed them into."
            )
        if hasattr(source, "read_window"):
            # hasattr is the actual duck-typing dispatch (see docstring); the cast only tells
            # the checker what that check already established about source's real shape.
            reader = cast("WindowedRasterReader", source)
            if self.task not in _DETECTION_TASKS:
                raise ValueError(
                    f"predict_tiled only supports detection/instance_seg tasks for a windowed "
                    f"reader source, got {self.task!r}: there is no untiled predict() fallback for "
                    "a raster too large to decode whole."
                )
            self._refuse_channel_mismatch(reader.num_channels)
            edge = self._require_tile_size(tile_size)

            def _windowed_tile(tile_x: int, tile_y: int):
                y0, y1 = tile_y, min(tile_y + edge, reader.height)
                x0, x1 = tile_x, min(tile_x + edge, reader.width)
                return pad_tile(reader.read_window(y0, y1, x0, x1), edge)

            result = self._tiled_infer_core(
                reader.height, reader.width, _windowed_tile, edge, overlap, tile_batch_size,
                global_nms_iou, postprocess, require_masks=require_masks, tile_resize=tile_resize,
                band_interpretations=getattr(reader, "band_interpretations", None),
                prior=prior, progress=progress)
            result["image"] = source_label
            return result

        if self.task not in _DETECTION_TASKS:
            return self.predict(source)

        from tcip_mcp.pipelines.raster_source import photographic_container

        # A photographic file is safely coerced to in_chans by load_image's own PIL conversion
        # below; only a genuine array/raster container has no such coercion to rely on.
        if not photographic_container(source, self.in_chans):
            self._refuse_channel_mismatch(probe_channels(source))
        edge = self._require_tile_size(tile_size)

        img = load_image(source, self.in_chans)
        w, h = img.size if isinstance(img, Image.Image) else (img.shape[1], img.shape[0])

        def _decoded_tile(tile_x: int, tile_y: int):
            return _crop_pad_tile(img, tile_x, tile_y, edge, w, h)

        result = self._tiled_infer_core(
            h, w, _decoded_tile, edge, overlap, tile_batch_size, global_nms_iou, postprocess,
            require_masks=require_masks, tile_resize=tile_resize)
        result["image"] = display_source_path(source)
        return result

    def export_onnx(self, output_path: str, opset: int = 17) -> str:
        """Export model to ONNX format with dynamic batch size."""
        dummy = torch.randn(1, 3, 640, 640).to(self.device)
        self.model.eval()
        torch.onnx.export(
            self.model,
            (dummy,),
            output_path,
            opset_version=opset,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes={"images": {0: "batch", 2: "height", 3: "width"}},
        )
        logger.info("ONNX model exported to %s", output_path)
        return output_path

    def _format_detection(self, outputs: dict, image_path: str, w: int, h: int) -> dict:
        keep = outputs["scores"] >= self.score_threshold
        result = {
            "image": image_path,
            "width": w,
            "height": h,
            "boxes": outputs["boxes"][keep].cpu().tolist(),
            "scores": outputs["scores"][keep].cpu().tolist(),
            "labels": outputs["labels"][keep].cpu().tolist(),
            "count": int(keep.sum()),
        }
        if self.task == "instance_seg" and "masks" in outputs:
            # Kept soft (unbinarized), mask_geometry()/export.py binarize via
            # resolve_binarize_threshold(), never a second hardcoded threshold here; export.py
            # stamps the (currently unvalidated) threshold it used into each prediction's own
            # attributes rather than pinning 0.5 silently. Same order as boxes/scores/labels.
            masks = outputs["masks"][keep]
            if masks.dim() == 4 and masks.shape[1] == 1:  # torchvision MaskRCNN: [N, 1, H, W]
                masks = masks[:, 0]
            result["masks"] = masks.cpu().tolist()
        return result

    def _format_other(self, outputs: dict, image_path: str, w: int, h: int) -> dict:
        result: dict = {"image": image_path, "width": w, "height": h}
        if isinstance(outputs, dict):
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    result[k] = v.cpu().tolist()
                else:
                    result[k] = v
        elif isinstance(outputs, torch.Tensor):
            result["output"] = outputs.cpu().tolist()
        return result
