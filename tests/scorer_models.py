"""Importable builders for the active-learning scorer tests, and the predictor they are read through.

Each builder returns a small ``nn.Module`` whose outputs a scorer test asserts on. A test reaches
it the way every scorer caller does: the module is saved as a bespoke checkpoint, registered, and
loaded through ``load_registered_checkpoint`` and ``build_predictor``. Not a ``test_*`` module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class EmptyThenLowDetector(torch.nn.Module):
    """Returns an empty frame on the first call, a low-confidence detection on the second."""

    def __init__(self) -> None:
        super().__init__()
        self._n = 0

    def forward(self, imgs):
        self._n += 1
        scores = torch.tensor([]) if self._n == 1 else torch.tensor([0.3, 0.4])
        return [{"scores": scores, "boxes": torch.zeros((len(scores), 4)),
                 "labels": torch.zeros((len(scores),), dtype=torch.int64)}]


class TwoHeadClassifier(torch.nn.Module):
    """Two classification heads with fixed logits."""

    def forward(self, x):
        return {"head0_logits": torch.tensor([[2.0, 0.0]]),
                "head1_logits": torch.tensor([[0.0, 0.0]])}


class ConvBackbone(torch.nn.Module):
    """A model exposing a ``.backbone`` the diversity scorer embeds through."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Conv2d(3, 8, 3, padding=1)


def build_empty_then_low(**_: Any) -> torch.nn.Module:
    return EmptyThenLowDetector()


def build_two_head(**_: Any) -> torch.nn.Module:
    return TwoHeadClassifier()


def build_conv_backbone(**_: Any) -> torch.nn.Module:
    return ConvBackbone()


def build_linear(**_: Any) -> torch.nn.Module:
    return torch.nn.Linear(3, 3)


def predictor_for(tmp_path: Path, builder: str, task: str) -> Any:
    """The predictor a scorer reads, built from ``tests.scorer_models:<builder>`` through the
    platform's own checkpoint registration, load and ``build_predictor``."""
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.predictor import build_predictor
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    src = {"builder": f"tests.scorer_models:{builder}", "builder_kwargs": {"in_chans": 3},
           "task": task}
    path = registered_checkpoint(tmp_path, project_root=tmp_path, model_source=src)
    return build_predictor(load_registered_checkpoint(path, project_path=str(tmp_path)),
                           device="cpu")
