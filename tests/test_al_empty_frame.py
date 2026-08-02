"""An empty-detection frame must not flood the uncertainty queue with a max (1.0) score."""

from __future__ import annotations

import torch
from PIL import Image

from tcip_mcp.pipelines.active_learning.scorer import UncertaintyScorer


class _FakeDetector(torch.nn.Module):
    """Returns an empty frame on the first call, a low-confidence detection on the second."""

    def __init__(self) -> None:
        super().__init__()
        self._n = 0

    def forward(self, imgs):  # scorer calls model([tensor]) per image
        self._n += 1
        scores = torch.tensor([]) if self._n == 1 else torch.tensor([0.3, 0.4])
        return [{"scores": scores, "boxes": torch.zeros((len(scores), 4)),
                 "labels": torch.zeros((len(scores),), dtype=torch.int64)}]


def test_empty_frame_ranks_low_not_max(tmp_path):
    for name in ("a", "b"):
        Image.new("RGB", (64, 64), (100, 100, 100)).save(tmp_path / f"{name}.png")
    paths = [str(tmp_path / "a.png"), str(tmp_path / "b.png")]

    ranked = UncertaintyScorer(task="detection").score(paths, _FakeDetector(), torch.device("cpu"))
    by_path = dict(ranked)

    assert by_path[paths[0]] == 0.0  # the empty frame: no ambiguous decision -> does not flood
    assert by_path[paths[1]] > 0.0   # the low-confidence detection is genuinely uncertain
    assert ranked[0][0] == paths[1]  # the detection frame outranks the empty one
