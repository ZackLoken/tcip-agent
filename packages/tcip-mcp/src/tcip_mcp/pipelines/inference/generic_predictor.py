"""Generic predictor for any bespoke ``model_source`` checkpoint.

The task is read from the saved ``model_source``; prediction runs over one image, a batch, or a
sliced source.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, JsonValue

if TYPE_CHECKING:
    from pathlib import Path

    from sahi.prediction import ObjectPrediction

    from tcip_mcp.model_registry import VerifiedCheckpoint
    from tcip_mcp.pipelines.execution import Execution
    from tcip_mcp.pipelines.slicing import TcipDetectionModel

from tcip_mcp.pipelines.execution import (
    DEFAULT_IMAGE_BATCH_SIZE,
    DEFAULT_TILE_BATCH_SIZE,
)
from tcip_mcp.pipelines.image_utils import (
    BandGroupRef,
    frame_size,
    load_image,
    pil_to_tensor,
    pixel_array,
    source_path_of,
)
from tcip_mcp.pipelines.model_build import (
    STATE_DICT_KEY,
    build_from_model_source,
    recorded_model_dims,
)
from tcip_mcp.pipelines.model_contract import DETECTOR_TASKS, TILING_TASKS


class SliceBatch(BaseModel):
    """Lattice slices a sliced pass predicted, with their shifted predictions.

    ``slices`` are ``(x0, y0, x1, y1)`` (:func:`~tcip_mcp.pipelines.slicing.slice_lattice`) and
    ``predictions`` their full-frame rows (:func:`~tcip_mcp.pipelines.slicing.prediction_rows`):
    one tile batch's, or every recorded batch's together as an interrupted pass's prior.
    """

    model_config = ConfigDict(extra="forbid")

    slices: list[tuple[int, int, int, int]]
    predictions: list[dict[str, JsonValue]]


class WindowedRasterReader(Protocol):
    """A source read window by window.

    It carries full-raster pixel dimensions, a band count, and a windowed decode.
    """

    height: int
    width: int
    num_channels: int

    def read_window(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Return the pixel window ``[y0:y1, x0:x1]``, rows first."""


class GenericPredictor:
    """Load any bespoke ``model_source`` checkpoint and run inference under an execution record.

    The execution record each call is given alone decides the confidence threshold and each
    frame's detection cap. The checkpoint must carry its run config, whose ``model_source`` names
    the builder and the task, and the weights (``model_build``'s ``STATE_DICT_KEY``).

    The input geometry the run trained at travels on the checkpoint and is exposed as-recorded:
    ``train_tile_size``/``train_overlap`` (a tiled run's tile lattice, off its config),
    ``train_native_size`` (its recorded geometry's ``native_size``), and ``train_augmentation``
    (the augmentation config that run declared). ``dims`` are the dimensions the model was built
    at (:func:`~tcip_mcp.pipelines.model_build.recorded_model_dims`).
    """

    def __init__(
        self, checkpoint: VerifiedCheckpoint, device: str | None = None
    ) -> None:
        """Build ``checkpoint``'s model with its weights, in eval mode on ``device``.

        ``device`` defaults to CUDA when available, else the CPU.
        """
        from tcip_mcp.pipelines.schemas import run_tiling

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_path = checkpoint.path
        self.checkpoint_sha256 = checkpoint.sha256
        spec = checkpoint.spec
        self.task = spec.model_source.task

        tiling = run_tiling(self.task, spec.data.tiling)
        self.train_tile_size = tiling.tile_size if tiling is not None else None
        self.train_overlap = tiling.overlap if tiling is not None else None
        self.train_native_size = checkpoint.geometry.native_size
        self.train_augmentation = spec.augmentation

        self.dims = recorded_model_dims(spec)
        self.in_chans = self.dims["in_chans"]
        self.attribute_sizes = [
            len(a.values) for a in self.dims.get("attributes", ())
        ]
        self.model = build_from_model_source(
            spec.model_source,
            checkpoint.layout,
            self.dims,
            checkpoint.geometry,
        )
        self.model.load_state_dict(checkpoint.payload[STATE_DICT_KEY])
        self.model.to(self.device)
        self.model.eval()

    def model_input(
        self, image_path: str | Path | BandGroupRef
    ) -> tuple[torch.Tensor, int, int]:
        """Return one source as this model reads it, with the source's width and height.

        The source is EXIF-oriented at the model's own channel width, as a tensor on its device.
        """
        img = load_image(image_path, self.in_chans)
        return pil_to_tensor(img).to(self.device), *frame_size(img)

    @torch.no_grad()
    def predict(
        self, image_path: str | Path | BandGroupRef, execution: Execution
    ) -> dict:
        """Run inference on a single image under the untiled ``execution`` record.

        The image is a plain path/string or a :class:`BandGroupRef`.
        """
        if self.task in DETECTOR_TASKS:
            return self._predict_whole([image_path], execution)[0]
        tensor, w, h = self.model_input(image_path)
        outputs = self.model(tensor.unsqueeze(0))
        return self._format_other(outputs, source_path_of(image_path), w, h)

    @torch.no_grad()
    def predict_batch(
        self,
        image_paths: list[str | Path | BandGroupRef],
        execution: Execution,
        *,
        tile_batch_size: int = DEFAULT_TILE_BATCH_SIZE,
        batch_size: int = DEFAULT_IMAGE_BATCH_SIZE,
    ) -> list[dict]:
        """Run inference on multiple images under ``execution``, sliced when it is a tiled record.

        Detection takes ``batch_size`` images at a time, one forward per distinct frame cap among
        them (:meth:`_predict_whole`); other heads run one image per forward.

        Each element of ``image_paths`` may be a plain path/string or a :class:`BandGroupRef` (see
        :meth:`predict`). Under a tiled ``execution`` every image runs through
        :meth:`predict_sliced`; masks are carried where the record collects them.
        """
        if execution.tiled:
            return [
                self.predict_sliced(
                    p,
                    execution=execution,
                    tile_batch_size=tile_batch_size,
                )
                for p in image_paths
            ]
        if self.task not in DETECTOR_TASKS:
            return [self.predict(p, execution) for p in image_paths]
        step = max(1, batch_size)
        return [
            r
            for start in range(0, len(image_paths), step)
            for r in self._predict_whole(
                image_paths[start : start + step], execution
            )
        ]

    def _predict_whole(
        self, sources: list[str | Path | BandGroupRef], execution: Execution
    ) -> list[dict]:
        """Return the detection record of each of ``sources`` under ``execution``.

        Each source is decoded whole as one slice the size of its frame at shift zero and run
        under its own frame's cap, each result built by :meth:`_detection_record` under the cap
        its forward ran at; under an :attr:`~tcip_mcp.pipelines.execution.Execution.uncut` record
        each carries ``soft_masks``, each detection's uncut mask as the model kept it
        (``slicing.TcipDetectionModel.soft_masks``).
        """
        from tcip_mcp.pipelines.derivations import detection_cap
        from tcip_mcp.pipelines.slicing import TcipDetectionModel

        density = cast(float, execution.density)
        model = TcipDetectionModel(
            self,
            execution=execution,
            cap_of=lambda a: detection_cap(density, a.shape[0] * a.shape[1]),
            band_interpretations=None,
        )
        arrays = [
            pixel_array(load_image(s, self.in_chans))[0] for s in sources
        ]
        model.perform_batch_inference(arrays)
        model.convert_original_predictions(
            shift_amount=[[0, 0]] * len(arrays),
            full_shape=[list(a.shape[:2]) for a in arrays],
        )
        uncut = execution.uncut
        return [
            {
                **self._detection_record(
                    preds,
                    source_path_of(s),
                    *frame_size(a),
                    model=model,
                    cap=cap,
                ),
                **({"soft_masks": soft} if uncut else {}),
            }
            for preds, s, a, cap, soft in zip(
                model.object_prediction_list_per_image,
                sources,
                arrays,
                model.caps,
                model.soft_masks,
                strict=True,
            )
        ]

    def _detection_record(
        self,
        predictions: list[ObjectPrediction],
        label: str,
        width: int,
        height: int,
        *,
        model: TcipDetectionModel,
        cap: int,
    ) -> dict:
        """Return the platform's detection record from ``ObjectPrediction``s in full-frame pixels.

        The predictions are as the forward or the merge that capped them at ``cap`` left them, of
        the ``width`` x ``height`` frame. The record holds ``image``, ``width``, ``height``,
        ``boxes`` (xyxy), ``scores``, ``labels`` (1-indexed), ``count`` (their number) and ``cap``;
        where the checkpoint carries attributes, ``attributes``, one id per attribute per
        detection; where ``model``'s record collects masks and cuts them, ``masks`` as one
        ``{"segmentation": [[x0, y0, x1, y1, ...], ...]}`` per detection, empty where the mask
        binarized to nothing.
        """
        from tcip_mcp.pipelines.slicing import prediction_rows

        kept = prediction_rows(predictions, self.attribute_sizes)
        record = {
            "image": label,
            "width": int(width),
            "height": int(height),
            "boxes": [row["bbox"] for row in kept],
            "scores": [row["score"] for row in kept],
            "labels": [row["category_id"] for row in kept],
            "count": len(kept),
            "cap": cap,
        }
        if self.attribute_sizes:
            record["attributes"] = [row["attributes"] for row in kept]
        if model.execution.mask_cut is not None:
            record["masks"] = [
                {"segmentation": row["segmentation"] or []} for row in kept
            ]
        return record

    @torch.no_grad()
    def predict_sliced(
        self,
        source: str | Path | BandGroupRef | WindowedRasterReader,
        *,
        execution: Execution,
        tile_batch_size: int,
        source_label: str = "",
        prior: SliceBatch | None = None,
        progress: Callable[[int, int, SliceBatch], None] | None = None,
    ) -> dict:
        """Run sliced detection over one source under a tiled ``execution`` record.

        The source is cut on SAHI's lattice (:func:`~tcip_mcp.pipelines.slicing.slice_lattice`) at
        its tile edge and overlap over the whole frame, each slice cut by array indexing and
        predicted through :class:`~tcip_mcp.pipelines.slicing.TcipDetectionModel`
        ``tile_batch_size`` at a time with its shift and the frame's shape, then the record's one
        SAHI merge (:func:`~tcip_mcp.pipelines.slicing.cross_tile_merge`) over every slice's
        shifted detections and the cap the record gives the whole frame, highest score first, each
        slice running under that same cap.

        A :class:`WindowedRasterReader` (has ``.read_window``) serves each slice from its own
        window and names the result by ``source_label``; any other source decodes whole once. Its
        band count must equal ``in_chans`` (the reader's ``num_channels``, or the band count a
        non-photographic file's decode serves) or it refuses before any slice is read. A
        non-detection task falls back to :meth:`predict` on a decoded source and refuses on a
        reader.

        The record's ``tile_resize`` stretches each slice and maps the result back; a slice PIL
        does not represent faithfully refuses (``ValueError``). A record collecting masks adds
        ``masks``, one ``{"segmentation": [[x0, y0, x1, y1, ...], ...]}`` per detection: the
        merged SAHI polygons in full-frame pixels, empty where the mask binarized to nothing.

        ``prior`` seeds a reader's pass with the :class:`SliceBatch` an interrupted pass recorded,
        and its slices are skipped; ``progress`` is called once per live batch with its first and
        last slice index and its :class:`SliceBatch`. Both refuse on a decoded source. Returns the
        detection record (:meth:`_detection_record`) with ``tiles``, the lattice's slice count.
        """
        from tcip_mcp.pipelines.slicing import (
            cross_tile_merge,
            prediction_rows,
            predictions_from_rows,
            slice_lattice,
        )

        windowed = hasattr(source, "read_window")
        if not windowed and (prior is not None or progress is not None):
            raise ValueError(
                "prior/progress apply only to a windowed-reader source: a whole-decode pass writes "
                "its files all at once and has no resume seam to feed them into."
            )
        if self.task not in TILING_TASKS:
            if windowed:
                raise ValueError(
                    f"sliced prediction over a windowed reader needs a task that tiles "
                    f"({sorted(TILING_TASKS)}), got {self.task!r}: a raster too large to decode "
                    "whole has no untiled fallback."
                )
            return self.predict(
                cast("str | Path | BandGroupRef", source), execution
            )
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
            arr, interpretations = pixel_array(
                load_image(decoded, self.in_chans)
            )
            (width, height), label = frame_size(arr), source_path_of(decoded)
            bands = int(arr.shape[-1])

            def read(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
                return arr[y0:y1, x0:x1]

        if bands != self.in_chans:
            raise ValueError(
                f"source has {bands} channel(s) but the model expects in_chans={self.in_chans}; "
                "refusing to silently truncate/pad the band count the model was trained on."
            )

        tile_size = cast(int, execution.tile_size)
        merge = cross_tile_merge(execution)
        from tcip_mcp.pipelines.derivations import detection_cap
        from tcip_mcp.pipelines.slicing import TcipDetectionModel

        # Every tile runs under the frame's cap: a tile denser than the density keeps its objects
        # until the merged frame is capped.
        frame_cap = detection_cap(
            cast(float, execution.density), width * height
        )
        model = TcipDetectionModel(
            self,
            execution=execution,
            cap_of=lambda _tile: frame_cap,
            band_interpretations=interpretations,
        )
        slices = slice_lattice(
            height, width, tile_size, cast(float, execution.overlap)
        )
        prior = prior or SliceBatch(slices=[], predictions=[])
        done = set(prior.slices)
        predictions = predictions_from_rows(
            prior.predictions, [height, width], self.attribute_sizes
        )
        pending = [(i, s) for i, s in enumerate(slices) if s not in done]
        for start in range(0, len(pending), tile_batch_size):
            batch = pending[start : start + tile_batch_size]
            model.perform_batch_inference([read(*s) for _, s in batch])
            model.convert_original_predictions(
                shift_amount=[[s[0], s[1]] for _, s in batch],
                full_shape=[[height, width]] * len(batch),
            )
            new = [
                p.get_shifted_object_prediction()
                for per_slice in model.object_prediction_list_per_image
                for p in per_slice
            ]
            predictions.extend(new)
            if progress is not None:
                progress(
                    batch[0][0],
                    batch[-1][0],
                    SliceBatch(
                        slices=[s for _, s in batch],
                        predictions=prediction_rows(new, self.attribute_sizes),
                    ),
                )

        merged = sorted(
            merge(predictions) if predictions else [],
            key=lambda p: -p.score.value,
        )[:frame_cap]
        return {
            **self._detection_record(
                merged, label, width, height, model=model, cap=frame_cap
            ),
            "tiles": len(slices),
        }

    def _format_other(
        self, outputs: dict, image_path: str, w: int, h: int
    ) -> dict:
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
