"""Importable builders of small fixed-output models, and a predictor built over one.

Each builder returns a small ``nn.Module`` with fixed outputs; :func:`predictor_for` loads one
through a registered checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class EmptyThenLowDetector(torch.nn.Module):
    """A detector answering no box on its first call and boxes scored 0.3 and 0.4 after."""

    def __init__(self) -> None:
        super().__init__()
        self._n = 0

    def forward(self, imgs):
        self._n += 1
        scores = torch.tensor([]) if self._n == 1 else torch.tensor([0.3, 0.4])
        return [
            {
                "scores": scores,
                "boxes": torch.zeros((len(scores), 4)),
                "labels": torch.zeros((len(scores),), dtype=torch.int64),
            }
        ]


class TwoHeadClassifier(torch.nn.Module):
    """Two classification heads with fixed logits."""

    def forward(self, x):
        return {
            "head0_logits": torch.tensor([[2.0, 0.0]]),
            "head1_logits": torch.tensor([[0.0, 0.0]]),
        }


class ConvBackbone(torch.nn.Module):
    """A model whose ``.backbone`` is one 3-to-8-channel 3x3 convolution."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Conv2d(3, 8, 3, padding=1)


class ScorelessDetector(torch.nn.Module):
    """A detector answering one box and its label and no ``scores``."""

    def forward(self, imgs):
        return [
            {
                "boxes": torch.zeros((1, 4)),
                "labels": torch.zeros((1,), dtype=torch.int64),
            }
        ]


def build_empty_then_low(**_: Any) -> torch.nn.Module:
    return EmptyThenLowDetector()


def build_scoreless(**_: Any) -> torch.nn.Module:
    return ScorelessDetector()


def build_two_head(**_: Any) -> torch.nn.Module:
    return TwoHeadClassifier()


def build_conv_backbone(**_: Any) -> torch.nn.Module:
    return ConvBackbone()


def build_linear(**_: Any) -> torch.nn.Module:
    return torch.nn.Linear(3, 3)


def predictor_for(tmp_path: Path, builder: str, task: str) -> Any:
    """Return a ``GenericPredictor`` over a registered checkpoint of ``scorer_models:<builder>``.

    A ``task`` run over the builder completes and registers the checkpoint, which is loaded
    through ``load_registered_checkpoint``.
    """
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tests import REPO_ROOT
    from tests._chain_fixtures import BESPOKE_MODELS, FRAME_STATISTICS
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    src = {
        "builder": f"scorer_models:{builder}",
        "builder_kwargs": {},
        "source_files": [
            str(REPO_ROOT / "tests" / "scorer_models.py"),
            BESPOKE_MODELS,
        ],
        "task": task,
        "normalization": FRAME_STATISTICS,
    }
    path = registered_checkpoint(tmp_path, model_source=src)
    return GenericPredictor(
        load_registered_checkpoint(path, project=tmp_path), device="cpu"
    )
