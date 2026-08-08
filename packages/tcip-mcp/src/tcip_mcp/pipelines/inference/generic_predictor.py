"""Generic predictor for any bespoke ``model_source`` checkpoint.

Auto-detects task from the saved ``model_source``. Supports single image,
batch, and ONNX export.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Protocol

import torch
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

from tcip_mcp.pipelines.model_build import build_model
from tcip_mcp.pipelines.image_utils import BandGroupRef, load_image, pad_tile, pil_to_tensor
from tcip_mcp.pipelines.inference.predictor import KIND_TCIP_MODULE
from tcip_mcp.pipelines.resolution import DEFAULT_NMS_IOU, DEFAULT_TILE_SIZE

logger = logging.getLogger(__name__)

# Detection task names that format outputs as boxes/scores/labels. A bespoke model_source declares
# the task type ``detection`` / ``instance_seg``, both route through the detection formatter.
_DETECTION_TASKS = frozenset({"detection", "instance_seg"})


from tcip_mcp.pipelines.image_utils import crop_pad_tile as _crop_pad_tile  # noqa: E402


class WindowedRasterReader(Protocol):
    """The read surface a huge-raster tile source must expose for :meth:`GenericPredictor.
    predict_tiled_from_reader`: full-raster pixel dimensions, band count, and a windowed decode.
    Duck-typed rather than importing a concrete reader here, so this stays usable for any raster
    too large to load whole, not just :class:`~tcip_mcp.pipelines.raster_source.StripTiffSource`
    (``pipelines/raster_source.py``, its one implementation today).
    """

    height: int
    width: int
    num_channels: int

    def read_window(self, y0: int, y1: int, x0: int, x1: int) -> "np.ndarray": ...


def _display_path(source: str | Path | BandGroupRef) -> str:
    """A JSON-safe, human-meaningful identity string for a predict result's ``image`` field.

    A :class:`BandGroupRef` has no single sibling file that names the logical image, its own
    ``.bandgroup`` manifest path is the closest thing (stable, on disk, unique per capture); a plain
    path/string is returned as-is. Every ``load_image``/``load_multiband`` call in this module keeps
    receiving the original source object (never this string), so a band-grouped capture still
    decodes through the channel-aware loader instead of a stringified dataclass repr that no reader
    can open.
    """
    if isinstance(source, BandGroupRef):
        return str(source.manifest_path)
    return str(source)


class GenericPredictor:
    """Load any bespoke ``model_source`` checkpoint and run inference.

    The checkpoint must contain 'model_source' and 'model_state_dict'.
    Task type is read from the model_source.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str | None = None,
        score_threshold: float = 0.5,
        nms_iou: float | None = None,
        max_dets: int | None = None,
        *,
        checkpoint: dict | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.score_threshold = score_threshold
        self.max_dets = max_dets

        # ``build_predictor`` may hand us the already-loaded checkpoint (it read it to sniff the
        # kind) so the weights aren't read from disk twice; fall back to loading it ourselves.
        ckpt = checkpoint if checkpoint is not None else torch.load(
            checkpoint_path, map_location=self.device, weights_only=False)
        # A bespoke checkpoint carries the importable-builder ref; build_model re-imports it.
        self.model_source = ckpt.get("model_source")
        self.kind = KIND_TCIP_MODULE
        self.config = ckpt.get("config", {})

        # Training tile geometry, so inference can derive the tile scale from the checkpoint instead
        # of a mismatched default. None when this checkpoint carried no tiling geometry.
        _tiling = (self.config.get("data") or {}).get("tiling") or {}
        self.train_tile_size = _tiling.get("tile_size")
        self.train_overlap = _tiling.get("overlap")

        self.model = build_model(ckpt)  # re-imported bespoke builder (no exec)
        self.model.load_state_dict(ckpt["model_state_dict"])
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
        self.in_chans = int(src.get("in_chans", 3))

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
        disp = _display_path(image_path)

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
        tile_size: int = DEFAULT_TILE_SIZE, overlap: float = 0.2, tile_batch_size: int = 96,
        global_nms_iou: float = DEFAULT_NMS_IOU, batch_size: int = 16, postprocess: str = "nms",
        *, require_masks: bool = True,
    ) -> list[dict]:
        """Run inference on multiple images (optionally tiled for small objects).

        For detection, images are run through the detector in batches of ``batch_size``
        (one GPU forward per batch, torchvision detectors take a list of variable-size
        images), instead of one forward per image. Non-detection heads stay per-image
        since their inputs are native-resolution (can't be stacked without resizing).

        Each element of ``image_paths`` may be a plain path/string or a :class:`BandGroupRef`
        (see :meth:`predict`). ``require_masks`` forwards to :meth:`predict_tiled` when ``tile=True``
        (see its own docstring for the tiled mask shape); ignored when ``tile=False``, since the
        untiled path always carries masks for ``instance_seg``.
        """
        if tile:
            return [
                self.predict_tiled(p, tile_size=tile_size, overlap=overlap,
                                   tile_batch_size=tile_batch_size, global_nms_iou=global_nms_iou,
                                   postprocess=postprocess, require_masks=require_masks)
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
                meta.append((_display_path(p), w, h))
            outputs = self.model(tensors)  # one forward over the whole chunk
            for (disp, w, h), out in zip(meta, outputs):
                results.append(self._format_detection(out, disp, w, h))
        return results

    def _tiled_infer_core(
        self, height: int, width: int, get_tile: Callable[[int, int], object],
        tile_size: int, overlap: float, tile_batch_size: int, global_nms_iou: float,
        postprocess: str, *, require_masks: bool = True,
    ) -> dict:
        """Shared tiling/batching/reconstruction loop behind both :meth:`predict_tiled` (a fully
        decoded in-memory image) and :meth:`predict_tiled_from_reader` (a windowed raster reader):
        build tile positions from ``(height, width)``, pull each tile's pixels via ``get_tile(tile_x,
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

        min_size = ((self.model_source or {}).get("builder_kwargs") or {}).get("min_size")
        if min_size and abs(int(min_size) - tile_size) > tile_size:
            logger.warning("tiled inference: model min_size=%s differs greatly from tile_size=%s "
                           "(tiles will be rescaled).", min_size, tile_size)

        collect_masks = self.task == "instance_seg" and require_masks

        per_tile_boxes, per_tile_scores, per_tile_labels, tile_info = [], [], [], []
        per_tile_masks: list = [] if collect_masks else None
        batch_tiles: list = []
        batch_meta: list = []

        def _flush() -> None:
            if not batch_tiles:
                return
            outputs = self.model(batch_tiles)
            for out, meta in zip(outputs, batch_meta):
                keep = out["scores"] >= self.score_threshold
                per_tile_boxes.append(out["boxes"][keep].cpu().numpy())
                per_tile_scores.append(out["scores"][keep].cpu().numpy())
                per_tile_labels.append(out["labels"][keep].cpu().numpy())
                if collect_masks:
                    m = out["masks"][keep]
                    if m.dim() == 4 and m.shape[1] == 1:  # torchvision MaskRCNN: [N, 1, H, W]
                        m = m[:, 0]
                    per_tile_masks.append(m.cpu().numpy())
                tile_info.append(meta)
            batch_tiles.clear()
            batch_meta.clear()

        for tile_x, tile_y in positions:
            crop = get_tile(tile_x, tile_y)
            batch_tiles.append(pil_to_tensor(crop).to(self.device))
            batch_meta.append({"tile_x": tile_x, "tile_y": tile_y,
                               "original_width": width, "original_height": height})
            if len(batch_tiles) >= tile_batch_size:
                _flush()
        _flush()

        if collect_masks:
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
                boxes, scores, labels, masks = global_merge(
                    boxes, scores, labels, global_nms_iou, per_det_masks=masks)
            else:
                boxes, scores, labels = global_merge(boxes, scores, labels, global_nms_iou)
        else:
            keep = global_nms(boxes, scores, labels, global_nms_iou)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if collect_masks:
                masks = [masks[i] for i in keep]

        # Enforce the full-frame detection cap after the cross-tile merge (highest score first);
        # the in-model detections_per_img only caps per tile, so a dense image can exceed it.
        if self.max_dets is not None and len(scores) > self.max_dets:
            top = np.argsort(scores)[::-1][: self.max_dets]
            boxes, scores, labels = boxes[top], scores[top], labels[top]
            if collect_masks:
                masks = [masks[i] for i in top]

        result = {
            "width": width,
            "height": height,
            "boxes": boxes.tolist(),
            "scores": scores.tolist(),
            "labels": labels.tolist(),
            "count": int(len(boxes)),
            "tiles": len(positions),
        }
        if collect_masks:
            result["masks"] = [
                {"mask_patch": mp.patch.tolist(), "offset_x": mp.offset_x, "offset_y": mp.offset_y}
                for mp in masks
            ]
        return result

    @torch.no_grad()
    def predict_tiled(
        self, image_path: str | Path | BandGroupRef, tile_size: int = DEFAULT_TILE_SIZE,
        overlap: float = 0.2, tile_batch_size: int = 96, global_nms_iou: float = DEFAULT_NMS_IOU,
        postprocess: str = "nms", *, require_masks: bool = True,
    ) -> dict:
        """Tiled (SAHI-style) detection: sliding-window tiles -> per-tile predict ->
        core-region reconstruction -> cross-tile merge -> full-image detections.

        ``postprocess`` selects the cross-tile merge: ``"nms"`` suppresses overlaps, ``"nmm"``
        unions boxes split across a seam. Falls back to :meth:`predict` for non-detection heads.
        ``image_path`` may be a plain path/string or a :class:`BandGroupRef` (see :meth:`predict`).

        ``require_masks`` (default True) governs whether an ``instance_seg`` checkpoint's masks are
        threaded through the cross-tile reconstruction/merge at all: a mask patch per detection is
        real extra memory/compute across a dense tile grid, so a caller that never reads masks
        (``run_full_frame_evaluation`` scores boxes against full-frame GT) can pass
        ``require_masks=False`` to skip it deliberately; the result then carries no ``masks`` key
        for any task. With ``require_masks=True`` and ``instance_seg``, the returned ``masks`` are
        not the untiled result's dense ``[H, W]`` full-image arrays: each is a small tile-local
        patch plus its full-image-space offset (``{"mask_patch", "offset_x", "offset_y"}``, see
        :meth:`_tiled_infer_core`), the shape a source raster too large to hold one full-size mask
        per detection requires; a consumer must not assume the two ``masks`` shapes are
        interchangeable.

        The whole image is decoded into memory once (via ``load_image``) before tiling; for a
        raster too large for that, use :meth:`predict_tiled_from_reader`.
        """
        if self.task not in _DETECTION_TASKS:
            return self.predict(image_path)

        img = load_image(image_path, self.in_chans)
        w, h = img.size if isinstance(img, Image.Image) else (img.shape[1], img.shape[0])

        def _get_tile(tile_x: int, tile_y: int):
            return _crop_pad_tile(img, tile_x, tile_y, tile_size, w, h)

        result = self._tiled_infer_core(
            h, w, _get_tile, tile_size, overlap, tile_batch_size, global_nms_iou, postprocess,
            require_masks=require_masks)
        result["image"] = _display_path(image_path)
        return result

    @torch.no_grad()
    def predict_tiled_from_reader(
        self, reader: WindowedRasterReader, tile_size: int = DEFAULT_TILE_SIZE,
        overlap: float = 0.2, tile_batch_size: int = 96, global_nms_iou: float = DEFAULT_NMS_IOU,
        postprocess: str = "nms", *, require_masks: bool = True, source_label: str = "",
    ) -> dict:
        """The same tiled (SAHI-style) detection as :meth:`predict_tiled`, but sources each tile's
        pixels from a windowed reader instead of a fully decoded in-memory image, so a raster too
        large to load whole (a multi-gigabyte orthomosaic) never has to be.

        ``reader`` is duck-typed (see :class:`WindowedRasterReader`): anything exposing ``.height``,
        ``.width``, ``.num_channels``, and ``.read_window(y0, y1, x0, x1) -> ndarray[H, W, C]``
        works, most directly ``StripTiffSource`` (``pipelines/raster_source.py``), but this method
        itself has no georeferencing concern and no import of that module: any huge
        non-georeferenced raster hits the same memory problem and can reuse this.

        ``reader.num_channels`` must equal this model's ``in_chans``; a mismatch raises ``ValueError``
        rather than silently truncating or zero-padding the band count the model was trained on.
        Non-detection tasks raise ``ValueError`` too: unlike :meth:`predict_tiled`, there is no
        untiled ``predict()`` fallback to fall back to, since the whole point of this method is that
        the raster cannot be decoded whole.

        Returns the same shape as :meth:`predict_tiled` (full-raster pixel space); ``image`` is
        ``source_label`` if given, else ``""`` (a windowed reader carries no single on-disk path the
        caller doesn't already know). ``require_masks`` and the resulting ``masks`` shape (tile-local
        patch + full-raster-space offset, never a dense full-raster-sized array) are exactly
        :meth:`predict_tiled`'s own contract, see its docstring; it matters even more here, since
        this method's whole reason to exist is a raster too large for one dense mask per detection
        to ever be affordable.
        """
        if self.task not in _DETECTION_TASKS:
            raise ValueError(
                f"predict_tiled_from_reader only supports detection/instance_seg tasks, got "
                f"{self.task!r}: there is no untiled predict() fallback for a raster too large to "
                "decode whole."
            )
        if reader.num_channels != self.in_chans:
            raise ValueError(
                f"raster has {reader.num_channels} channel(s) but the model expects "
                f"in_chans={self.in_chans}; refusing to silently truncate/pad the band count the "
                "model was trained on."
            )

        def _get_tile(tile_x: int, tile_y: int):
            y0, y1 = tile_y, min(tile_y + tile_size, reader.height)
            x0, x1 = tile_x, min(tile_x + tile_size, reader.width)
            return pad_tile(reader.read_window(y0, y1, x0, x1), tile_size)

        result = self._tiled_infer_core(
            reader.height, reader.width, _get_tile, tile_size, overlap, tile_batch_size,
            global_nms_iou, postprocess, require_masks=require_masks)
        result["image"] = source_label
        return result

    def export_onnx(self, output_path: str, opset: int = 17) -> str:
        """Export model to ONNX format with dynamic batch size."""
        dummy = torch.randn(1, 3, 640, 640).to(self.device)
        self.model.eval()
        torch.onnx.export(
            self.model,
            dummy,
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
