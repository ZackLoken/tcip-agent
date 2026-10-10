"""Generic predictor for any bespoke ``model_source`` checkpoint: task read from the saved
``model_source``, prediction over one image, a batch, or a sliced source."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Protocol, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from pathlib import Path

    from sahi.prediction import ObjectPrediction

    from tcip_mcp.model_registry import VerifiedCheckpoint
    from tcip_mcp.pipelines.execution import Execution
    from tcip_mcp.pipelines.slicing import TcipDetectionModel

from tcip_mcp.pipelines.execution import DEFAULT_IMAGE_BATCH_SIZE, DEFAULT_TILE_BATCH_SIZE
from tcip_mcp.pipelines.model_build import (
    STATE_DICT_KEY,
    build_from_model_source,
    recorded_model_dims,
)
from tcip_mcp.pipelines.image_utils import (
    BandGroupRef, frame_size, load_image, pil_to_tensor, pixel_array, source_path_of,
)
from tcip_mcp.pipelines.model_contract import DETECTION_TASKS

logger = logging.getLogger(__name__)


class WindowedRasterReader(Protocol):
    """A source read window by window: full-raster pixel dimensions, band count, and a windowed
    decode."""

    height: int
    width: int
    num_channels: int

    def read_window(self, y0: int, y1: int, x0: int, x1: int) -> "np.ndarray": ...


class GenericPredictor:
    """Load any bespoke ``model_source`` checkpoint and run inference under the execution record
    each call is given, which alone decides the confidence threshold and the detection cap.

    The checkpoint must carry its run config, whose ``model_source`` names the builder and the
    task, and the weights (``model_build``'s ``STATE_DICT_KEY``).

    The input geometry the run trained at travels on the checkpoint's embedded config and is
    exposed as-recorded: ``train_tile_size``/``train_overlap`` (a tiled run's tile lattice),
    ``train_native_size`` (the one frame size an untiled run's frames all shared, ``[width,
    height]``), and ``train_augmentation`` (the augmentation config that run declared, a dict or a
    preset name). ``dims`` are the dimensions the model was built at
    (:func:`~tcip_mcp.pipelines.model_build.recorded_model_dims`).
    """

    def __init__(self, checkpoint: "VerifiedCheckpoint", device: str | None = None) -> None:
        from tcip_mcp.pipelines.data.datasets import stated_tiling

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = checkpoint.path
        self.checkpoint_sha256 = checkpoint.sha256
        spec = checkpoint.spec
        self.model_source = spec.model_source
        self.task = spec.model_source.task

        tiling = stated_tiling(spec.data.tiling)
        self.train_tile_size = tiling.tile_size if tiling is not None else None
        self.train_overlap = tiling.overlap if tiling is not None else None
        self.train_native_size = spec.data.train_native_size
        self.train_augmentation = spec.augmentation

        self.dims = recorded_model_dims(spec)
        self.in_chans = self.dims["in_chans"]
        self.attribute_sizes = [len(a.values) for a in self.dims.get("attributes", ())]
        self.model = build_from_model_source(spec.model_source, checkpoint.layout, self.dims)
        self.model.load_state_dict(checkpoint.payload[STATE_DICT_KEY])
        self.model.to(self.device)
        self.model.eval()

    def model_input(self, image_path: str | Path | BandGroupRef) -> tuple[torch.Tensor, int, int]:
        """One source as this model reads it: EXIF-oriented at the model's own width, as a
        tensor on its device, with the source's ``(width, height)``."""
        img = load_image(image_path, self.in_chans)
        return pil_to_tensor(img).to(self.device), *frame_size(img)

    @torch.no_grad()
    def predict(self, image_path: str | Path | BandGroupRef, execution: Execution) -> dict:
        """Run inference on a single image, a plain path/string or a :class:`BandGroupRef`, under
        the untiled ``execution`` record."""
        if self.task in DETECTION_TASKS:
            return self._predict_whole([image_path], execution)[0]
        tensor, w, h = self.model_input(image_path)
        outputs = self.model(tensor.unsqueeze(0))
        return self._format_other(outputs, source_path_of(image_path), w, h)

    @torch.no_grad()
    def predict_batch(
        self, image_paths: list[str | Path | BandGroupRef], execution: Execution,
        *, tile_batch_size: int = DEFAULT_TILE_BATCH_SIZE,
        batch_size: int = DEFAULT_IMAGE_BATCH_SIZE, require_masks: bool = True,
    ) -> list[dict]:
        """Run inference on multiple images under ``execution``, sliced when it is a tiled record.

        Detection runs ``batch_size`` images per forward; other heads run one image per forward.

        Each element of ``image_paths`` may be a plain path/string or a :class:`BandGroupRef` (see
        :meth:`predict`). Under a tiled ``execution`` every image runs through
        :meth:`predict_sliced`; untiled, ``instance_seg`` always carries masks.
        """
        if execution.tiled:
            return [self.predict_sliced(p, execution=execution, tile_batch_size=tile_batch_size,
                                        require_masks=require_masks)
                    for p in image_paths]
        if self.task not in DETECTION_TASKS:
            return [self.predict(p, execution) for p in image_paths]
        step = max(1, batch_size)
        return [r for start in range(0, len(image_paths), step)
                for r in self._predict_whole(image_paths[start:start + step], execution)]

    def governed(self, execution: Execution) -> torch.nn.Module:
        """This predictor's model with ``execution``'s conf and cap set as its in-model
        thresholds, so they govern which boxes exist."""
        from tcip_mcp.pipelines.operating_point import set_detector_operating_point

        set_detector_operating_point(self.model, score_thresh=execution.conf,
                                     detections_per_img=execution.max_dets)
        return self.model

    def _detection_model(self, execution: Execution, *, tile_resize: tuple[int, int] | None,
                         band_interpretations: tuple[str, ...] | None,
                         collect_masks: bool) -> "TcipDetectionModel":
        """This predictor wrapped as the SAHI detection model every detection pass runs through,
        its in-model thresholds and its own score filter set from ``execution``, masks cut at the
        platform's binarize threshold."""
        from tcip_mcp.pipelines.measurement.mask_geometry import resolve_binarize_threshold
        from tcip_mcp.pipelines.slicing import TcipDetectionModel

        self.governed(execution)
        return TcipDetectionModel(
            self, conf=execution.conf, tile_resize=tile_resize,
            band_interpretations=band_interpretations, collect_masks=collect_masks,
            mask_binarize=resolve_binarize_threshold())

    def _predict_whole(self, sources: list[str | Path | BandGroupRef],
                       execution: Execution) -> list[dict]:
        """One forward over ``sources`` under ``execution``, each decoded whole as one slice the
        size of its frame at shift zero, each result built by :meth:`_detection_record`."""
        model = self._detection_model(execution, tile_resize=None, band_interpretations=None,
                                      collect_masks=self.task == "instance_seg")
        arrays = [pixel_array(load_image(s, self.in_chans))[0] for s in sources]
        model.perform_batch_inference(arrays)
        model.convert_original_predictions(
            shift_amount=[[0, 0]] * len(arrays), full_shape=[list(a.shape[:2]) for a in arrays])
        return [self._detection_record(preds, source_path_of(s), *frame_size(a),
                                       model=model, max_dets=execution.max_dets)
                for preds, s, a in zip(model.object_prediction_list_per_image, sources, arrays)]

    def _detection_record(self, predictions: "list[ObjectPrediction]", label: str, width: int,
                          height: int, *, model: "TcipDetectionModel",
                          max_dets: int | None) -> dict:
        """The platform's detection record from ``ObjectPrediction``s in full-frame pixels, highest
        score first under the full-frame ``max_dets`` cap: ``image``, ``width``, ``height``,
        ``boxes`` (xyxy), ``scores``, ``labels`` (1-indexed), ``count`` and ``cap_hit``; where the
        checkpoint carries attributes, ``attributes``, one id per attribute per detection; where
        ``model`` collected masks, ``masks`` as one ``{"segmentation": [[x0, y0, x1, y1, ...],
        ...]}`` per detection, empty where the mask binarized to nothing, and ``mask_binarize``,
        the provenance of the threshold ``model`` cut them at."""
        from tcip_mcp.pipelines.slicing import prediction_rows

        ranked = sorted(predictions, key=lambda p: -p.score.value)
        cap_hit = bool(max_dets is not None and len(ranked) >= max_dets)
        kept = prediction_rows(ranked[:max_dets] if max_dets is not None else ranked,
                               self.attribute_sizes)
        record = {
            "image": label, "width": int(width), "height": int(height),
            "boxes": [row["bbox"] for row in kept], "scores": [row["score"] for row in kept],
            "labels": [row["category_id"] for row in kept], "count": len(kept), "cap_hit": cap_hit,
        }
        if self.attribute_sizes:
            record["attributes"] = [row["attributes"] for row in kept]
        if model.collect_masks:
            record["masks"] = [{"segmentation": row["segmentation"] or []} for row in kept]
            record["mask_binarize"] = model.mask_binarize
        return record

    @torch.no_grad()
    def predict_sliced(
        self, source: str | Path | BandGroupRef | WindowedRasterReader, *,
        execution: Execution, tile_batch_size: int, require_masks: bool, source_label: str = "",
        prior: dict | None = None, progress: "Callable[[int, int, dict], None] | None" = None,
    ) -> dict:
        """Sliced detection over one source under a tiled ``execution`` record: SAHI's lattice
        (:func:`~tcip_mcp.pipelines.slicing.slice_lattice`) at its tile edge and overlap over the
        whole frame, each slice cut by array indexing and predicted through
        :class:`~tcip_mcp.pipelines.slicing.TcipDetectionModel` ``tile_batch_size`` at a time with
        its shift and the frame's shape, then the record's one SAHI merge
        (:func:`~tcip_mcp.pipelines.slicing.cross_tile_merge`) over every slice's shifted
        detections and a full-frame ``max_dets`` cap, highest score first.

        A :class:`WindowedRasterReader` (has ``.read_window``) serves each slice from its own
        window and names the result by ``source_label``; any other source decodes whole once. Its
        band count must equal ``in_chans`` (the reader's ``num_channels``, or the band count a
        non-photographic file's decode serves) or it refuses before any slice is read. A
        non-detection task falls back to :meth:`predict` on a decoded source and refuses on a
        reader.

        The record's ``tile_resize`` stretches each slice PIL represents faithfully and maps the
        result back. ``require_masks`` on an ``instance_seg`` checkpoint adds ``masks``, one
        ``{"segmentation": [[x0, y0, x1, y1, ...], ...]}`` per detection: the merged SAHI polygons
        in full-frame pixels, empty where the mask binarized to nothing.

        ``prior`` seeds a reader's pass with the ``slices`` and shifted ``predictions``
        (:func:`~tcip_mcp.pipelines.slicing.prediction_rows`) an interrupted pass recorded, and
        those slices are skipped; ``progress`` is called once per live batch with its first and
        last slice index and a mapping of that shape. Both refuse on a decoded source. Returns the
        detection record with ``tiles`` (the lattice's slice count) and ``cap_hit``.
        """
        from tcip_mcp.pipelines.slicing import (
            cross_tile_merge, prediction_rows, predictions_from_rows, slice_lattice,
        )

        windowed = hasattr(source, "read_window")
        if not windowed and (prior is not None or progress is not None):
            raise ValueError(
                "prior/progress apply only to a windowed-reader source: a whole-decode pass writes "
                "its files all at once and has no resume seam to feed them into.")
        if self.task not in DETECTION_TASKS:
            if windowed:
                raise ValueError(
                    f"sliced prediction over a windowed reader needs a detection or instance_seg "
                    f"task, got {self.task!r}: a raster too large to decode whole has no untiled "
                    "fallback.")
            return self.predict(cast("str | Path | BandGroupRef", source), execution)
        reader = cast("WindowedRasterReader", source)
        decoded = cast("str | Path | BandGroupRef", source)
        # The count checked is the one the source serves at in_chans: a photograph decodes to
        # in_chans itself, an array serves its own bands.
        interpretations: tuple[str, ...] | None
        if windowed:
            height, width, label = reader.height, reader.width, source_label
            interpretations = getattr(reader, "band_interpretations", None)
            bands = reader.num_channels

            def read(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
                return reader.read_window(y0, y1, x0, x1)
        else:
            arr, interpretations = pixel_array(load_image(decoded, self.in_chans))
            (width, height), label = frame_size(arr), source_path_of(decoded)
            bands = int(arr.shape[-1])

            def read(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
                return arr[y0:y1, x0:x1]
        if bands != self.in_chans:
            raise ValueError(
                f"source has {bands} channel(s) but the model expects in_chans={self.in_chans}; "
                "refusing to silently truncate/pad the band count the model was trained on.")

        tile_size, tile_resize = cast(int, execution.tile_size), execution.tile_resize
        model_edge = min(int(tile_resize[0]), int(tile_resize[1])) if tile_resize else tile_size
        min_size = (self.model_source.builder_kwargs or {}).get("min_size")
        if min_size and abs(int(min_size) - model_edge) > model_edge:
            logger.warning("tiled inference: model min_size=%s differs greatly from the %spx tiles "
                           "it is handed (tiles will be rescaled).", min_size, model_edge)
        collect_masks = self.task == "instance_seg" and require_masks
        merge = cross_tile_merge(execution)
        model = self._detection_model(execution, tile_resize=tile_resize,
                                      band_interpretations=interpretations,
                                      collect_masks=collect_masks)
        slices = slice_lattice(height, width, tile_size, cast(float, execution.overlap))
        prior = prior or {"slices": [], "predictions": []}
        done = {tuple(s) for s in prior["slices"]}
        predictions = predictions_from_rows(prior["predictions"], [height, width],
                                            self.attribute_sizes)
        pending = [(i, s) for i, s in enumerate(slices) if s not in done]
        for start in range(0, len(pending), tile_batch_size):
            batch = pending[start:start + tile_batch_size]
            model.perform_batch_inference([read(*s) for _, s in batch])
            model.convert_original_predictions(
                shift_amount=[[s[0], s[1]] for _, s in batch],
                full_shape=[[height, width]] * len(batch))
            new = [p.get_shifted_object_prediction()
                   for per_slice in model.object_prediction_list_per_image for p in per_slice]
            predictions.extend(new)
            if progress is not None:
                progress(batch[0][0], batch[-1][0], {"slices": [list(s) for _, s in batch],
                                                     "predictions": prediction_rows(
                                                         new, self.attribute_sizes)})

        merged = merge(predictions) if predictions else []
        # The in-model cap only caps per slice; the record's cap is the full frame's.
        return {**self._detection_record(merged, label, width, height, model=model,
                                         max_dets=execution.max_dets),
                "tiles": len(slices)}

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
