"""One stub for the predictor ``GenericPredictor`` builds, answering fixed detections."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from tcip_mcp.pipelines.execution import (
    DEFAULT_IMAGE_BATCH_SIZE,
    DEFAULT_TILE_BATCH_SIZE,
)
from tcip_mcp.pipelines.image_utils import source_path_of
from tests._chain_fixtures import detected, detection_pass

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    import pytest

    from tcip_mcp.model_registry import VerifiedCheckpoint
    from tcip_mcp.pipelines.execution import Execution
    from tcip_mcp.pipelines.image_utils import BandGroupRef
    from tcip_mcp.pipelines.inference.generic_predictor import (
        SliceBatch,
        WindowedRasterReader,
    )
    from tcip_mcp.pipelines.inference.passes import Detection, DetectionPass
    from tcip_mcp.pipelines.schemas import AugmentationSpec

BOX = (10.0, 10.0, 30.0, 30.0)
"""The one box the stub answers unless told otherwise, xyxy in pixels."""


class StubPredictor:
    """A predictor answering the same detections for every image it is handed.

    The detections are ``detections`` (one at :data:`BOX` scored 0.9 unless given) on a
    ``width`` by ``height`` frame under the frame ``cap`` as stated (one above their number
    unless given). It carries a recorded training geometry (none unless given), and every other
    keyword becomes an attribute of its own (``task``, ``in_chans``). It counts its calls and
    keeps each execution it ran under and the checkpoint it was built for.
    """

    def __init__(
        self,
        *,
        width: int = 100,
        height: int = 100,
        detections: Sequence[Detection] | None = None,
        cap: int | None = None,
        train_tile_size: int | None = None,
        train_overlap: float | None = None,
        train_native_size: tuple[int, int] | None = None,
        train_augmentation: AugmentationSpec | None = None,
        **traits: str | int,
    ) -> None:
        self.train_tile_size = train_tile_size
        self.train_overlap = train_overlap
        self.train_native_size = train_native_size
        self.train_augmentation = train_augmentation
        self.frame = (width, height)
        self.detections = (
            [detected(BOX, 0.9)] if detections is None else list(detections)
        )
        self.cap = len(self.detections) + 1 if cap is None else cap
        for name, value in traits.items():
            setattr(self, name, value)
        self.calls = 0
        self.executions: list = []
        self.checkpoint = None

    def built(
        self, checkpoint: VerifiedCheckpoint, device: str | None = None
    ) -> StubPredictor:
        """Return this predictor, keeping ``checkpoint``'s path as its checkpoint."""
        self.checkpoint = getattr(checkpoint, "path", checkpoint)
        return self

    def answer(self, label: str) -> DetectionPass:
        """Return the stub's detections as the pass over the image ``label``."""
        return detection_pass(
            label, self.frame, list(self.detections), cap=self.cap
        )

    def predict_batch(
        self,
        image_paths: list[str | Path | BandGroupRef],
        execution: Execution,
        *,
        tile_batch_size: int = DEFAULT_TILE_BATCH_SIZE,
        batch_size: int = DEFAULT_IMAGE_BATCH_SIZE,
    ) -> list[DetectionPass]:
        self.calls += 1
        self.executions.append(execution)
        return [self.answer(source_path_of(p)) for p in image_paths]

    def predict_sliced(
        self,
        source: str | Path | BandGroupRef | WindowedRasterReader,
        *,
        execution: Execution,
        tile_batch_size: int,
        source_label: str = "",
        prior: SliceBatch | None = None,
        progress: Callable[[int, int, SliceBatch], None] | None = None,
    ) -> DetectionPass:
        self.calls += 1
        self.executions.append(execution)
        return self.answer(
            source_label
            if hasattr(source, "read_window")
            else source_path_of(cast("str | Path | BandGroupRef", source))
        )


def install(
    monkeypatch: pytest.MonkeyPatch, predictor: StubPredictor
) -> StubPredictor:
    """Make ``GenericPredictor`` build ``predictor``; returns it."""
    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor",
        predictor.built,
    )
    return predictor
