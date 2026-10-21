"""The proposal engines that ship registered: a promptable segmenter and a text-prompted detector.

Each reads its image once, EXIF-upright in RGB through the platform's raster reader, so a prompt
lands on the pixels the person pointed at and the returned geometry is in the frame the editor
draws. The segmenter runs each prompt on a window of the frame cut at its processor's
``longest_edge`` around the prompt, a box wider than that edge taking its own extent, and the
processor scales the window's longer side to that edge: a window whose longer side is the edge
reaches the model at native resolution, a frame shorter than the edge is upsampled and a wider box
downsampled by its ratio to the edge. The detector is run by the platform's tiled inference pass
on a lattice sized to its objects, which a whole-frame pass derives unless it is stated, never
finer than the model's input edge. Each runs a Hugging Face checkpoint through ``transformers``:
its files download into the user-profile cache on the checkpoint's first use, and its model and
processor, once loaded, are held for the process. torch and transformers load inside ``propose``.
"""

from __future__ import annotations

import functools
import threading
from typing import TYPE_CHECKING, Annotated, Self, TypedDict

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)
from sahi.models.base import DetectionModel

from tcip_annotation import Polygon, bbox_of
from tcip_annotation.mask_contours import mask_to_polygon_rings
from tcip_annotation.text import Text
from tcip_mcp.pipelines.execution import cross_tile_merge_of
from tcip_mcp.pipelines.model_build import placement

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import torch
    from sahi.prediction import ObjectPrediction
    from transformers import (
        GroundingDinoForObjectDetection,
        GroundingDinoProcessor,
        SamModel,
        SamProcessor,
    )

    from tcip_mcp.pipelines.proposal import Proposal

_MODEL_CALLS = threading.Lock()
"""Serializes every checkpoint load and model call of the two engines across threads."""


class SegmenterParams(BaseModel):
    """The promptable segmenter's parameters: its prompts, its mask switch and its checkpoint.

    Refuses a call stating neither a point nor a box.
    """

    model_config = ConfigDict(extra="forbid")

    points: list[tuple[float, float]] = []
    """Foreground points ``(x, y)`` in pixels of the upright frame, each a prompt of its own."""
    boxes: list[tuple[float, float, float, float]] = []
    """Boxes ``(x1, y1, x2, y2)`` in pixels of the upright frame, each a prompt of its own."""
    multimask_output: bool
    """Whether each prompt answers the model's several ranked masks, or its one best mask."""
    checkpoint: str = "facebook/sam-vit-large"
    """A SAM-family Hugging Face checkpoint; the default is the owner's choice of large SAM."""

    @model_validator(mode="after")
    def _prompted(self) -> Self:
        if not self.points and not self.boxes:
            raise ValueError(
                "the segmenter needs a prompt: state points, boxes or both"
            )
        return self


def _merge_name(name: str) -> str:
    """Return ``name``, a cross-tile merge ``CROSS_TILE_MERGES`` holds.

    Refuses (``ValueError``) any other name, naming the ones it holds
    (:func:`~tcip_mcp.pipelines.execution.cross_tile_merge_of`).
    """
    cross_tile_merge_of(name)
    return name


class DetectorParams(BaseModel):
    """The text-prompted detector's parameters: its phrase, thresholds, lattice, merge, checkpoint.

    The lattice's ``tile_size`` and ``overlap`` take ``execution.Stated``'s shape: each stated is
    taken as given and recorded ``explicit``; each unstated derives from a whole-frame pass's
    boxes (:func:`whole_frame_geometry`).
    """

    model_config = ConfigDict(extra="forbid")

    phrase: Text
    """What to find. A phrase with no period is lowercased and closed with one by the processor,
    the convention the checkpoint was trained on; one separating several classes by periods
    (``"a cat. a dog."``) is handed to it as written."""
    box_threshold: float
    """The least box confidence kept (the library's ``threshold``)."""
    text_threshold: float
    """The least token confidence that names a kept box's text label."""
    postprocess: Annotated[str, AfterValidator(_merge_name)]
    """The cross-tile merge of the slices' boxes, a name ``CROSS_TILE_MERGES`` holds."""
    cross_tile_nms: Annotated[float, Field(ge=0, le=1)]
    """The match threshold the merge joins two slices' boxes at, over its match metric, recorded
    ``explicit``."""
    tile_size: int | None = None
    """The lattice's edge in pixels, stated; unstated, derived."""
    overlap: float | None = None
    """The fraction of the edge neighboring slices share, stated; unstated, derived."""
    checkpoint: str = "IDEA-Research/grounding-dino-base"
    """A Grounding DINO Hugging Face checkpoint; the default is the owner's choice, the base one."""


class DetectorDefaults(TypedDict):
    """The detector's thresholds and merge as its libraries default them."""

    box_threshold: float
    text_threshold: float
    postprocess: str
    cross_tile_nms: float


def detector_defaults() -> DetectorDefaults:
    """Return the detector's thresholds and cross-tile merge as the libraries' own defaults.

    The two thresholds are read from the signature of transformers'
    ``GroundingDinoProcessor.post_process_grounded_object_detection``, the merge name and its
    match threshold from SAHI's ``get_sliced_prediction``, so no number is stated here.
    """
    import inspect

    from sahi.predict import get_sliced_prediction
    from transformers import GroundingDinoProcessor

    grounded = inspect.signature(
        GroundingDinoProcessor.post_process_grounded_object_detection
    ).parameters
    sliced = inspect.signature(get_sliced_prediction).parameters
    return {
        "box_threshold": grounded["threshold"].default,
        "text_threshold": grounded["text_threshold"].default,
        "postprocess": sliced["postprocess_type"].default.lower(),
        "cross_tile_nms": sliced["postprocess_match_threshold"].default,
    }


def upright_pixels(image_path: str) -> np.ndarray:
    """Return the image at ``image_path`` as ``[H, W, C]`` pixels, EXIF-upright in RGB.

    The read is :func:`~tcip_mcp.pipelines.image_utils.load_image` at three channels; the
    array is the caller's own copy, writable.
    """
    import numpy as np

    from tcip_mcp.pipelines.image_utils import load_image
    from tcip_mcp.pipelines.raster_source import hwc_array

    return np.array(hwc_array(load_image(image_path, 3)))


def mask_candidate(
    mask: np.ndarray,
    *,
    candidate_id: int,
    score: float,
    engine_meta: dict[str, JsonValue],
) -> dict | None:
    """Return the candidate one boolean ``mask`` proposes, or ``None`` when it yields no ring.

    Its ``rings`` keep every contour vertex of every connected region
    (:func:`~tcip_annotation.mask_contours.mask_to_polygon_rings` at spacing zero), its ``bbox``
    is their enclosing box and its ``area`` the mask's foreground pixel count.
    """
    import numpy as np

    rings = mask_to_polygon_rings(mask, threshold=None, vertex_spacing=0.0)
    if not rings:
        return None
    box = bbox_of(Polygon(rings))
    return {
        "candidate_id": candidate_id,
        "bbox": [box.x1, box.y1, box.x2, box.y2],
        "area": int(np.count_nonzero(mask)),
        "rings": rings,
        "score": score,
        "engine_meta": engine_meta,
    }


def prompt_window(
    prompt: Sequence[float], frame: Sequence[int], edge: int
) -> tuple[int, int, int, int]:
    """Return the half-open ``(x0, y0, x1, y1)`` window of a ``(height, width)`` frame for a prompt.

    ``prompt`` is a box ``(x1, y1, x2, y2)``, a point being a box of no extent. On each axis the
    window spans ``edge`` pixels, or the prompt's own extent where that is larger, and never more
    than the frame; it is centered on the prompt and moved inside the frame.
    """
    import math

    starts, ends = [], []
    for low, high, size in (
        (prompt[0], prompt[2], frame[1]),
        (prompt[1], prompt[3], frame[0]),
    ):
        span = min(size, max(edge, math.ceil(high - low)))
        start = min(max(round((low + high - span) / 2), 0), size - span)
        starts.append(start)
        ends.append(start + span)
    return starts[0], starts[1], ends[0], ends[1]


def shifted_candidate(candidate: dict, origin: Sequence[float]) -> dict:
    """Return ``candidate`` with its ``bbox`` and ``rings`` moved by ``origin``, ``(dx, dy)``.

    ``rings`` of ``None`` stays ``None``; every other key is kept as given.
    """
    dx, dy = origin
    x1, y1, x2, y2 = candidate["bbox"]
    rings = candidate["rings"]
    return {
        **candidate,
        "bbox": [x1 + dx, y1 + dy, x2 + dx, y2 + dy],
        "rings": None
        if rings is None
        else [[(x + dx, y + dy) for x, y in ring] for ring in rings],
    }


def box_candidate(
    box: Sequence[float],
    *,
    candidate_id: int,
    score: float,
    engine_meta: dict[str, JsonValue],
) -> dict:
    """Return the candidate one ``(x1, y1, x2, y2)`` box proposes: no rings, the box's area."""
    from tcip_annotation.matching import box_areas

    return {
        "candidate_id": candidate_id,
        "bbox": list(box),
        "area": float(box_areas([box])[0]),
        "rings": None,
        "score": score,
        "engine_meta": engine_meta,
    }


def _on_device[M: torch.nn.Module](model: M) -> M:
    """Return ``model`` moved in place to ``placement(None)`` by ``torch.nn.Module.to``.

    ``PreTrainedModel`` types its own ``to`` through ``functools.wraps``, dropping the bound self.
    """
    module: torch.nn.Module = model
    module.to(placement(None))
    return model


@functools.cache
def _segmenter(checkpoint: str) -> tuple[SamModel, SamProcessor]:
    """Return ``checkpoint``'s SAM model, on the device, and its processor."""
    from transformers import SamModel, SamProcessor

    return (
        _on_device(SamModel.from_pretrained(checkpoint)),
        SamProcessor.from_pretrained(checkpoint),
    )


@functools.cache
def _detector(
    checkpoint: str,
) -> tuple[GroundingDinoForObjectDetection, GroundingDinoProcessor]:
    """Return ``checkpoint``'s Grounding DINO model, on the device, and its processor."""
    from transformers import (
        GroundingDinoForObjectDetection,
        GroundingDinoProcessor,
    )

    return (
        _on_device(
            GroundingDinoForObjectDetection.from_pretrained(checkpoint)
        ),
        GroundingDinoProcessor.from_pretrained(checkpoint),
    )


class PromptableSegmenter:
    """A SAM-family segmenter proposing one candidate per mask each point or box prompt answers."""

    name = "sam"
    """The name the segmenter is registered and stages its proposals under."""

    def propose(self, image_path: str, **params: JsonValue) -> Proposal:
        """Return a candidate per mask the model answers for each prompt, best first per prompt.

        The proposal carries no execution record: a prompt window is no tiled pass.

        ``params`` is :class:`SegmenterParams`, refused as that model refuses it. Each prompt runs
        on its :func:`prompt_window` at the processor's own ``longest_edge``, prompts sharing a
        window sharing one image embedding. Each candidate is :func:`mask_candidate`'s moved into
        the frame (:func:`shifted_candidate`), its ``score`` the mask's predicted IoU and its
        ``engine_meta`` the ``prompt``'s index over ``points`` then ``boxes``, the mask's
        ``rank`` among that prompt's masks and the ``window`` the model saw.
        """
        stated = SegmenterParams.model_validate(params)
        import numpy as np
        import torch

        from tcip_mcp.pipelines.proposal import Proposal

        pixels = upright_pixels(image_path)
        prompts = [(x, y, x, y) for x, y in stated.points] + stated.boxes
        answers: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        with _MODEL_CALLS:
            model, processor = _segmenter(stated.checkpoint)
            edge = processor.image_processor.size["longest_edge"]
            windows = [
                prompt_window(p, pixels.shape[:2], edge) for p in prompts
            ]
            for window in dict.fromkeys(windows):
                x0, y0, x1, y1 = window
                local = {
                    i: [
                        v - o for v, o in zip(p, (x0, y0, x0, y0), strict=True)
                    ]
                    for i, p in enumerate(prompts)
                    if windows[i] == window
                }
                points = [i for i in local if i < len(stated.points)]
                boxes = [i for i in local if i >= len(stated.points)]
                inputs = processor(
                    images=pixels[y0:y1, x0:x1],
                    input_points=[[[local[i][:2]] for i in points]]
                    if points
                    else None,
                    input_boxes=[[local[i] for i in boxes]] if boxes else None,
                    return_tensors="pt",
                ).to(model.device)
                with torch.inference_mode():
                    embeddings = model.get_image_embeddings(
                        inputs["pixel_values"]
                    )
                    for kind, group in (
                        ("input_points", points),
                        ("input_boxes", boxes),
                    ):
                        if not group:
                            continue
                        answer = model(
                            image_embeddings=embeddings,
                            multimask_output=stated.multimask_output,
                            **{kind: inputs[kind]},
                        )
                        masks = processor.post_process_masks(
                            answer.pred_masks.cpu(),
                            inputs["original_sizes"].cpu(),
                            inputs["reshaped_input_sizes"].cpu(),
                        )[0].numpy()
                        scores = answer.iou_scores[0].cpu().numpy()
                        answers.update(
                            zip(group, zip(masks, scores, strict=True))
                        )

        candidates: list[dict] = []
        for prompt, window in enumerate(windows):
            masks, scores = answers[prompt]
            for rank, k in enumerate(np.argsort(-scores)):
                candidate = mask_candidate(
                    masks[k],
                    candidate_id=len(candidates),
                    score=float(scores[k]),
                    engine_meta={
                        "prompt": prompt,
                        "rank": rank,
                        "window": list(window),
                    },
                )
                if candidate is not None:
                    candidates.append(shifted_candidate(candidate, window[:2]))
        return Proposal(candidates, None)


class PhraseDetectionModel(DetectionModel):
    """A Grounding DINO model and its processor detecting one phrase, as a SAHI detection model.

    Each image, an ``[H, W, 3]`` RGB array, is detected at its own size through the processor
    and the model at the stated ``box_threshold`` and ``text_threshold``
    (``post_process_grounded_object_detection``); its detections become ``ObjectPrediction``s of
    category ``0`` named by their text label.
    """

    def __init__(
        self,
        model: GroundingDinoForObjectDetection,
        processor: GroundingDinoProcessor,
        stated: DetectorParams,
    ) -> None:
        """Wrap ``model`` and ``processor`` to detect ``stated``'s phrase at its thresholds."""
        self._processor = processor
        self._stated = stated
        self._model_device = model.device
        super().__init__(model=model)

    def set_device(self, device: str | None = None) -> None:
        """Take the device the model sits on; ``device`` is not read."""
        self.device = self._model_device

    def set_model(self, model: object, **kwargs: object) -> None:
        """Hold ``model`` as the model the images run through; ``kwargs`` are not read."""
        self.model = model

    def perform_inference(self, image: np.ndarray) -> None:
        """Detect one ``image`` through :meth:`perform_batch_inference`."""
        self.perform_batch_inference([image])

    def perform_batch_inference(self, images: list[np.ndarray]) -> None:
        """Detect each of ``images`` at its own size."""
        import torch

        found = []
        for image in images:
            inputs = self._processor(
                images=image, text=[self._stated.phrase], return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                outputs = self.model(**inputs)
            found.append(
                self._processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    threshold=self._stated.box_threshold,
                    text_threshold=self._stated.text_threshold,
                    target_sizes=[image.shape[:2]],
                )[0]
            )
        self._original_predictions = found

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int | float]] | None = None,
        full_shape_list: list[list[int | float]] | None = None,
    ) -> None:
        from sahi.prediction import ObjectPrediction

        assert shift_amount_list is not None and full_shape_list is not None
        self._object_prediction_list_per_image = [
            [
                ObjectPrediction(
                    bbox=box.tolist(),
                    score=float(score),
                    category_id=0,
                    category_name=label,
                    shift_amount=shift,
                    full_shape=full,
                )
                for box, score, label in zip(
                    found["boxes"].cpu(),
                    found["scores"].cpu(),
                    found["text_labels"],
                    strict=True,
                )
            ]
            for found, shift, full in zip(
                self._original_predictions,
                shift_amount_list,
                full_shape_list,
                strict=True,
            )
        ]


class TextPromptedDetector:
    """A Grounding DINO detector proposing one box candidate per detection of a phrase."""

    name = "grounding_dino"
    """The name the detector is registered and stages its proposals under."""

    def propose(self, image_path: str, **params: JsonValue) -> Proposal:
        """Return a box candidate per detection over the frame, through the platform's tiled pass.

        ``params`` is :class:`DetectorParams`, refused as that model refuses it. The frame runs
        through :class:`PhraseDetectionModel` by :func:`detected_proposal`, the processor's own
        ``shortest_edge`` the model's input edge, whose refusals propagate.
        """
        stated = DetectorParams.model_validate(params)
        pixels = upright_pixels(image_path)
        with _MODEL_CALLS:
            model, processor = _detector(stated.checkpoint)
            return detected_proposal(
                PhraseDetectionModel(model, processor, stated),
                pixels,
                processor.image_processor.size["shortest_edge"],
                postprocess=stated.postprocess,
                cross_tile_nms=stated.cross_tile_nms,
                tile_size=stated.tile_size,
                overlap=stated.overlap,
            )


def detected_proposal(
    detector: DetectionModel,
    pixels: np.ndarray,
    model_edge: int,
    *,
    postprocess: str,
    cross_tile_nms: float,
    tile_size: int | None,
    overlap: float | None,
) -> Proposal:
    """Return the box candidates ``detector`` answers over ``pixels``, through the tiled pass.

    The frame runs whole first (``slicing.sliced_predictions``, one slice of the frame): a frame
    within ``model_edge`` that pass finds nothing on is answered by it, with no execution record,
    and any other frame's lattice is :func:`whole_frame_geometry`'s from it and the stated
    ``tile_size`` and ``overlap``, whose refusals propagate. The record is
    ``execution.proposal_execution``'s. A lattice whose edge reaches the frame's longer side is
    the frame itself, answered by the whole-frame pass; otherwise its slices
    (``slicing.slice_lattice``) run through the same pass and merge by
    ``slicing.cross_tile_merge`` under the record. Each candidate is :func:`box_candidate`'s, its
    ``score`` the detection's confidence and its ``engine_meta`` its ``label``, the prediction's
    category name.
    """
    from tcip_mcp.pipelines import slicing
    from tcip_mcp.pipelines.execution import (
        DEFAULT_TILE_BATCH_SIZE,
        proposal_execution,
    )
    from tcip_mcp.pipelines.proposal import Proposal

    height, width = pixels.shape[:2]

    def read(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        return pixels[y0:y1, x0:x1]

    [(_batch, [whole])] = slicing.sliced_predictions(
        detector, read, [(0, (0, 0, width, height))], (height, width), 1
    )
    if not whole and max(height, width) <= model_edge:
        return Proposal([], None)
    edge, lattice_overlap, sources = whole_frame_geometry(
        whole,
        (height, width),
        model_edge,
        tile_size=tile_size,
        overlap=overlap,
    )
    execution = proposal_execution(
        tile_size=edge,
        overlap=lattice_overlap,
        geometry_sources=sources,
        postprocess=postprocess,
        cross_tile_nms=cross_tile_nms,
    )
    if edge >= max(height, width):
        return Proposal(_box_candidates(whole), execution)
    slices = slicing.slice_lattice(height, width, edge, lattice_overlap)
    predictions = [
        p
        for _batch, batch in slicing.sliced_predictions(
            detector,
            read,
            list(enumerate(slices)),
            (height, width),
            DEFAULT_TILE_BATCH_SIZE,
        )
        for found in batch
        for p in found
    ]
    merged = (
        slicing.cross_tile_merge(execution)(predictions) if predictions else []
    )
    return Proposal(_box_candidates(merged), execution)


def whole_frame_geometry(
    predictions: Sequence[ObjectPrediction],
    frame: tuple[int, int],
    model_edge: int,
    *,
    tile_size: int | None,
    overlap: float | None,
) -> tuple[int, float, dict[str, str]]:
    """Return a lattice's edge and overlap over the ``(height, width)`` frame, and their sources.

    Each is ``derivations.derive_tile_geometry``'s over a whole-frame pass's ``predictions`` as
    one region of the frame, a stated value as given and recorded ``explicit``. A derived edge
    (``TILE_EDGE_DERIVATION``) under ``model_edge``, the model's own input edge, is floored at it,
    recorded ``derived``, and an unstated overlap is derived at the edge taken
    (``TILE_OVERLAP_DERIVATION``). Refuses as the derivation refuses, naming ``tile_size`` and
    ``overlap`` to state (:class:`~tcip_mcp.pipelines.execution.ExecutionRefusedError`).
    """
    import numpy as np

    from tcip_mcp.pipelines.derivations import (
        TILE_EDGE_DERIVATION,
        TILE_OVERLAP_DERIVATION,
        Region,
        derive_tile_geometry,
    )
    from tcip_mcp.pipelines.execution import ExecutionRefusedError

    height, width = frame
    region = Region(
        boxes=np.array(
            [p.bbox.to_xyxy() for p in predictions], dtype=float
        ).reshape(-1, 4),
        labels=np.ones(len(predictions), dtype=int),
        area=float(height * width),
    )
    edge_source = "explicit" if tile_size is not None else TILE_EDGE_DERIVATION
    try:
        edge, derived_overlap = derive_tile_geometry(
            [region], tile_size=tile_size, overlap=overlap
        )
        if tile_size is None and edge < model_edge:
            edge, derived_overlap = derive_tile_geometry(
                [region], tile_size=model_edge, overlap=overlap
            )
            edge_source = "derived"
    except ValueError as exc:
        raise ExecutionRefusedError(str(exc)) from exc
    return (
        edge,
        derived_overlap,
        {
            "tile_size": edge_source,
            "overlap": "explicit"
            if overlap is not None
            else TILE_OVERLAP_DERIVATION,
        },
    )


def _box_candidates(predictions: Sequence[ObjectPrediction]) -> list[dict]:
    """Return one :func:`box_candidate` per prediction, its text label its ``engine_meta``."""
    return [
        box_candidate(
            p.bbox.to_xyxy(),
            candidate_id=i,
            score=float(p.score.value),
            engine_meta={"label": p.category.name},
        )
        for i, p in enumerate(predictions)
    ]
