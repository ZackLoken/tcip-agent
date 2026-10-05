"""One stub for the predictor ``GenericPredictor`` builds, answering fixed detections."""

from __future__ import annotations

import copy
from typing import Any

BOX = (10.0, 10.0, 30.0, 30.0)
"""The one box the stub answers unless told otherwise, xyxy in pixels."""


class StubPredictor:
    """A predictor answering the same detections for every image it is handed: ``boxes`` scored
    ``scores``, each labeled 1, on a ``width`` by ``height`` frame, with ``attributes`` when
    given. It states the recorded training geometry a predictor exposes (none unless given), and
    every other keyword becomes an attribute a caller reads off a predictor (``task``). Counts its
    calls, and keeps each execution it ran under and the checkpoint it was built for."""

    def __init__(self, *, width: int = 100, height: int = 100, boxes=(BOX,), scores=(0.9,),
                 attributes=None, train_tile_size=None, train_overlap=None,
                 train_native_size=None, train_augmentation=None, **traits: Any) -> None:
        self.train_tile_size = train_tile_size
        self.train_overlap = train_overlap
        self.train_native_size = train_native_size
        self.train_augmentation = train_augmentation
        self.record: dict[str, Any] = {
            "width": width, "height": height, "boxes": [list(b) for b in boxes],
            "scores": list(scores), "labels": [1] * len(boxes), "count": len(boxes),
            "cap_hit": False}
        if attributes is not None:
            self.record["attributes"] = attributes
        for name, value in traits.items():
            setattr(self, name, value)
        self.calls = 0
        self.executions: list = []
        self.checkpoint = None

    def built(self, checkpoint_path=None, **kwargs: Any) -> StubPredictor:
        """This predictor, as ``GenericPredictor(checkpoint_path, ...)`` would build one."""
        self.checkpoint = getattr(checkpoint_path, "path", checkpoint_path)
        return self

    def predict_batch(self, paths, execution=None, **kw: Any) -> list[dict]:
        self.calls += 1
        self.executions.append(execution)
        return [{"image": p, **copy.deepcopy(self.record)} for p in paths]


def install(monkeypatch, predictor: StubPredictor) -> StubPredictor:
    """Make ``GenericPredictor`` build ``predictor``; returns it."""
    monkeypatch.setattr("tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor",
                        predictor.built)
    return predictor
