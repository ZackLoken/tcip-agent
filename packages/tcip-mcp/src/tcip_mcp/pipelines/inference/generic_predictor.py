"""Generic predictor for any bespoke ``model_source`` checkpoint: task read from the saved
``model_source``, prediction over one image, a batch, or a sliced source."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Protocol, cast

import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

    from sahi.prediction import ObjectPrediction

    from tcip_mcp.model_registry import VerifiedCheckpoint
    from tcip_mcp.pipelines.slicing import TcipDetectionModel

from tcip_mcp.pipelines.derivations import probe_channels
from tcip_mcp.pipelines.model_build import (
    MODEL_SOURCE_KEY,
    STATE_DICT_KEY,
    build_model,
    run_in_chans,
)
from tcip_mcp.pipelines.image_utils import (
    BandGroupRef, display_source_path, load_image, pil_to_tensor, pixel_array,
)
from tcip_mcp.pipelines.inference.predictor import KIND_TCIP_MODULE
from tcip_mcp.pipelines.resolution import (
    DEFAULT_IMAGE_BATCH_SIZE, DEFAULT_NMS_IOU, DEFAULT_OVERLAP, DEFAULT_POSTPROCESS,
    DEFAULT_TILE_BATCH_SIZE,
)

logger = logging.getLogger(__name__)

# Detection task names that format outputs as boxes/scores/labels. A bespoke model_source declares
# the task type ``detection`` / ``instance_seg``, both route through the detection formatter.
_DETECTION_TASKS = frozenset({"detection", "instance_seg"})


class WindowedRasterReader(Protocol):
    """A source read window by window: full-raster pixel dimensions, band count, and a windowed
    decode."""

    height: int
    width: int
    num_channels: int

    def read_window(self, y0: int, y1: int, x0: int, x1: int) -> "np.ndarray": ...


class GenericPredictor:
    """Load any bespoke ``model_source`` checkpoint and run inference.

    The checkpoint must carry the model reference and the weights (``model_build``'s
    ``MODEL_SOURCE_KEY`` / ``STATE_DICT_KEY``). Task type is read from the model_source.

    The input geometry the run trained at travels on the checkpoint's embedded config and is
    exposed as-recorded: ``train_tile_size``/``train_overlap`` (a tiled run's tile lattice),
    ``train_native_size`` (the one frame size an untiled run's frames all shared, ``[width,
    height]``), and ``train_augmentation`` (the augmentation config that run declared, a dict or a
    preset name). :func:`~tcip_mcp.pipelines.inference.predictor.resolve_tile_geometry` turns
    those into an inference geometry.
    """

    def __init__(
        self,
        checkpoint: "VerifiedCheckpoint",
        device: str | None = None,
        score_threshold: float | None = 0.5,
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
        # The untiled counterpart, both as recorded; resolve_tile_geometry reads them.
        self.train_native_size = (self.config.get("data") or {}).get("train_native_size")
        self.train_augmentation = self.config.get("augmentation")

        self.model = build_model(ckpt)  # re-imported bespoke builder (no exec)
        self.model.load_state_dict(ckpt[STATE_DICT_KEY])
        self.model.to(self.device)
        self.model.eval()

        # In-model thresholds, so the operating point governs which boxes exist rather than
        # filtering ones the model already discarded; an unstated knob leaves the point as built.
        from tcip_mcp.pipelines.operating_point import set_detector_operating_point
        set_detector_operating_point(self.model, score_thresh=score_threshold,
                                     detections_per_img=max_dets)

        # The width comes from the run this checkpoint came out of, its model's declaration or its
        # data config's recorded one.
        src = self.model_source or {}
        self.task = checkpoint.task
        width = run_in_chans(src, self.config.get("data"))
        if width is None:
            raise ValueError(
                f"{checkpoint.path} records no input width: its model_source declares no in_chans "
                f"and its run config no data.num_channels, so how many bands to read an image at "
                f"is unknown, and reading at a guess would feed the model something other than "
                f"what it trained on. Re-register a checkpoint from a run this platform trained, "
                "or declare model_source.in_chans."
            )
        self.in_chans = width

    def model_input(self, image_path: str | Path | BandGroupRef) -> tuple[torch.Tensor, int, int]:
        """One source as this model reads it: EXIF-oriented at the model's own width, as a
        tensor on its device, with the source's ``(width, height)``."""
        img = load_image(image_path, self.in_chans)
        w, h = img.size if isinstance(img, Image.Image) else (img.shape[1], img.shape[0])
        return pil_to_tensor(img).to(self.device), int(w), int(h)

    @torch.no_grad()
    def predict(self, image_path: str | Path | BandGroupRef) -> dict:
        """Run inference on a single image. ``image_path`` may be a plain path/string or a
        :class:`BandGroupRef`.
        """
        if self.task in _DETECTION_TASKS:
            return self._predict_whole([image_path])[0]
        tensor, w, h = self.model_input(image_path)
        outputs = self.model(tensor.unsqueeze(0))
        return self._format_other(outputs, display_source_path(image_path), w, h)

    @torch.no_grad()
    def predict_batch(
        self, image_paths: list[str | Path | BandGroupRef], tile: bool = False,
        tile_size: int | None = None, overlap: float = DEFAULT_OVERLAP, tile_batch_size: int = DEFAULT_TILE_BATCH_SIZE,
        cross_tile_nms: float = DEFAULT_NMS_IOU, batch_size: int = DEFAULT_IMAGE_BATCH_SIZE, postprocess: str = DEFAULT_POSTPROCESS,
        *, require_masks: bool = True, tile_resize: tuple[int, int] | None = None,
    ) -> list[dict]:
        """Run inference on multiple images, optionally sliced.

        Detection runs ``batch_size`` images per forward; other heads run one image per forward.

        Each element of ``image_paths`` may be a plain path/string or a :class:`BandGroupRef` (see
        :meth:`predict`). With ``tile=True`` every image runs through :meth:`predict_sliced` with
        the tiled arguments unchanged, and a tiled call with no ``tile_size`` refuses; untiled, they
        are ignored and ``instance_seg`` always carries masks.
        """
        if tile:
            if tile_size is None:
                raise ValueError(
                    "a tiled prediction needs an explicit tile_size; resolve one "
                    "(resolve_tile_geometry) before calling.")
            return [
                self.predict_sliced(p, tile_size=tile_size, overlap=overlap,
                                    postprocess=postprocess, cross_tile_nms=cross_tile_nms,
                                    tile_batch_size=tile_batch_size, tile_resize=tile_resize,
                                    require_masks=require_masks)
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
            results.extend(self._predict_whole(image_paths[start:start + max(1, batch_size)]))
        return results

    def _detection_model(self, *, tile_resize: tuple[int, int] | None,
                         band_interpretations: tuple[str, ...] | None,
                         collect_masks: bool) -> "TcipDetectionModel":
        """This predictor wrapped as the SAHI detection model every detection pass runs through,
        masks cut at the platform's binarize threshold."""
        from tcip_mcp.pipelines.measurement.mask_geometry import resolve_binarize_threshold
        from tcip_mcp.pipelines.slicing import TcipDetectionModel

        return TcipDetectionModel(
            self, tile_resize=tile_resize, band_interpretations=band_interpretations,
            collect_masks=collect_masks, mask_binarize=resolve_binarize_threshold().to_provenance())

    def _predict_whole(self, sources: list[str | Path | BandGroupRef]) -> list[dict]:
        """One forward over ``sources``, each decoded whole as one slice the size of its frame at
        shift zero, each result built by :meth:`_detection_record`."""
        model = self._detection_model(tile_resize=None, band_interpretations=None,
                                      collect_masks=self.task == "instance_seg")
        arrays = [pixel_array(load_image(s, self.in_chans))[0] for s in sources]
        model.perform_batch_inference(arrays)
        model.convert_original_predictions(
            shift_amount=[[0, 0]] * len(arrays), full_shape=[list(a.shape[:2]) for a in arrays])
        return [self._detection_record(preds, display_source_path(s), a.shape[1], a.shape[0],
                                       model=model)
                for preds, s, a in zip(model.object_prediction_list_per_image, sources, arrays)]

    def _detection_record(self, predictions: "list[ObjectPrediction]", label: str, width: int,
                          height: int, *, model: "TcipDetectionModel") -> dict:
        """The platform's detection record from ``ObjectPrediction``s in full-frame pixels, highest
        score first under the full-frame ``max_dets`` cap: ``image``, ``width``, ``height``,
        ``boxes`` (xyxy), ``scores``, ``labels`` (1-indexed), ``count`` and ``cap_hit``; where
        ``model`` collected masks, ``masks`` as one ``{"segmentation": [[x0, y0, x1, y1, ...],
        ...]}`` per detection, empty where the mask binarized to nothing, and ``mask_binarize``,
        the provenance of the threshold ``model`` cut them at."""
        ranked = sorted(predictions, key=lambda p: -p.score.value)
        # cap_hit uses >=, matching records_from_detector.
        cap_hit = bool(self.max_dets is not None and len(ranked) >= self.max_dets)
        kept = ranked[:self.max_dets] if self.max_dets is not None else ranked
        record = {
            "image": label, "width": int(width), "height": int(height),
            "boxes": [[float(v) for v in p.bbox.to_xyxy()] for p in kept],
            "scores": [float(p.score.value) for p in kept],
            "labels": [p.category.id for p in kept],
            "count": len(kept), "cap_hit": cap_hit,
        }
        if model.collect_masks:
            record["masks"] = [{"segmentation": p.mask.segmentation if p.mask else []}
                               for p in kept]
            record["mask_binarize"] = model.mask_binarize
        return record

    @torch.no_grad()
    def predict_sliced(
        self, source: str | Path | BandGroupRef | WindowedRasterReader, *, tile_size: int,
        overlap: float, postprocess: str, cross_tile_nms: float, tile_batch_size: int,
        tile_resize: tuple[int, int] | None, require_masks: bool, source_label: str = "",
        prior: dict | None = None, progress: "Callable[[int, int, dict], None] | None" = None,
    ) -> dict:
        """Sliced detection over one source: SAHI's lattice
        (:func:`~tcip_mcp.pipelines.slicing.slice_lattice`) over the whole frame, each slice cut by
        array indexing and predicted through
        :class:`~tcip_mcp.pipelines.slicing.TcipDetectionModel` ``tile_batch_size`` at a time with
        its shift and the frame's shape, then one SAHI merge
        (:func:`~tcip_mcp.pipelines.slicing.cross_tile_merge`) over every slice's shifted
        detections and a full-frame ``max_dets`` cap, highest score first.

        A :class:`WindowedRasterReader` (has ``.read_window``) serves each slice from its own
        window and names the result by ``source_label``; any other source decodes whole once. Its
        band count must equal ``in_chans`` (the reader's ``num_channels``, or a non-photographic
        file's probed bands) or it refuses before any slice is read. A non-detection task falls
        back to :meth:`predict` on a decoded source and refuses on a reader.

        ``tile_resize`` stretches each slice PIL represents faithfully and maps the result back.
        ``require_masks`` on an ``instance_seg`` checkpoint adds ``masks``, one
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
        if self.task not in _DETECTION_TASKS:
            if windowed:
                raise ValueError(
                    f"sliced prediction over a windowed reader needs a detection or instance_seg "
                    f"task, got {self.task!r}: a raster too large to decode whole has no untiled "
                    "fallback.")
            return self.predict(cast("str | Path | BandGroupRef", source))
        from tcip_mcp.pipelines.raster_source import photographic_container

        reader = cast("WindowedRasterReader", source)
        decoded = cast("str | Path | BandGroupRef", source)
        # load_image converts a photographic frame to in_chans itself; an array has no coercion.
        bands = (reader.num_channels if windowed
                 else self.in_chans if photographic_container(decoded, self.in_chans)
                 else probe_channels(decoded))
        if bands != self.in_chans:
            raise ValueError(
                f"source has {bands} channel(s) but the model expects in_chans={self.in_chans}; "
                "refusing to silently truncate/pad the band count the model was trained on.")
        interpretations: tuple[str, ...] | None
        if windowed:
            height, width, label = reader.height, reader.width, source_label
            interpretations = getattr(reader, "band_interpretations", None)

            def read(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
                return reader.read_window(y0, y1, x0, x1)
        else:
            arr, interpretations = pixel_array(load_image(decoded, self.in_chans))
            height, width, label = arr.shape[0], arr.shape[1], display_source_path(decoded)

            def read(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
                return arr[y0:y1, x0:x1]

        model_edge = min(int(tile_resize[0]), int(tile_resize[1])) if tile_resize else tile_size
        min_size = ((self.model_source or {}).get("builder_kwargs") or {}).get("min_size")
        if min_size and abs(int(min_size) - model_edge) > model_edge:
            logger.warning("tiled inference: model min_size=%s differs greatly from the %spx tiles "
                           "it is handed (tiles will be rescaled).", min_size, model_edge)
        collect_masks = self.task == "instance_seg" and require_masks
        merge = cross_tile_merge(postprocess, cross_tile_nms)
        model = self._detection_model(tile_resize=tile_resize, band_interpretations=interpretations,
                                      collect_masks=collect_masks)
        slices = slice_lattice(height, width, tile_size, overlap)
        prior = prior or {"slices": [], "predictions": []}
        done = {tuple(s) for s in prior["slices"]}
        predictions = predictions_from_rows(prior["predictions"], [height, width])
        pending = [(i, s) for i, s in enumerate(slices) if s not in done]
        for start in range(0, len(pending), tile_batch_size):
            batch = pending[start:start + tile_batch_size]
            model.perform_batch_inference([read(*s) for _, s in batch])
            model.convert_original_predictions(
                shift_amount=[[s[0], s[1]] for _, s in batch], full_shape=[[height, width]] * len(batch))
            new = [p.get_shifted_object_prediction()
                   for per_slice in model.object_prediction_list_per_image for p in per_slice]
            predictions.extend(new)
            if progress is not None:
                progress(batch[0][0], batch[-1][0], {"slices": [list(s) for _, s in batch],
                                                     "predictions": prediction_rows(new)})

        merged = merge(predictions) if predictions else []
        # The in-model cap only caps per slice; the record's cap is the full frame's.
        return {**self._detection_record(merged, label, width, height, model=model),
                "tiles": len(slices)}

    def _kept_by_score(self, scores: torch.Tensor) -> torch.Tensor:
        """Which detections this predictor's own score threshold keeps; all of them when it was
        built with none."""
        if self.score_threshold is None:
            return torch.ones_like(scores, dtype=torch.bool)
        return scores >= self.score_threshold

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
