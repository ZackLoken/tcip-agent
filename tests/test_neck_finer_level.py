"""Neck diversity: the FPN and PAN necks' optional finer ``add_p2`` level, off by default.

The cases run on CPU over tiny inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from torch import Tensor

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.components.necks import FPN, PAN  # noqa: E402


def _features() -> dict[str, Tensor]:
    return {
        "s0": torch.randn(1, 64, 32, 32),
        "s1": torch.randn(1, 128, 16, 16),
        "s2": torch.randn(1, 256, 8, 8),
        "s3": torch.randn(1, 512, 4, 4),
    }


# --------------------------------------------------------------------------
# Necks: add_p2
# --------------------------------------------------------------------------


def test_fpn_add_p2_adds_finer_level() -> None:
    out = FPN([64, 128, 256, 512], 64, add_p2=True)(_features())
    assert set(out.keys()) == {"p0", "p1", "p2", "p3", "p4"}
    assert (
        out["p0"].shape[-1] == 2 * out["p1"].shape[-1]
    )  # extra level is 2x finer


def test_pan_add_p2_levels_and_convs() -> None:
    pan = PAN([64, 128, 256, 512], 64, add_p2=True)
    assert len(pan(_features())) == 5
    assert len(pan.bottom_up_convs) == 4
    pan2 = PAN([64, 128, 256, 512], 64)
    assert len(pan2(_features())) == 4
    assert len(pan2.bottom_up_convs) == 3


def test_necks_default_off_unchanged() -> None:
    fpn_out = FPN([64, 128, 256, 512], 64)(_features())
    assert set(fpn_out.keys()) == {"p0", "p1", "p2", "p3"}
    assert fpn_out["p0"].shape[1] == 64
    assert set(PAN([64, 128, 256, 512], 64)(_features()).keys()) == {
        "p0",
        "p1",
        "p2",
        "p3",
    }
