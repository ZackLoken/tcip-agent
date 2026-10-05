"""The CORN loss has one body: the ordinal head trains by it and the loss registry builds it, so
the two routes answer one value for one batch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_the_ordinal_head_and_the_registered_corn_loss_agree():
    from tcip_mcp.pipelines.components.heads import OrdinalHead
    from tcip_mcp.pipelines.components.losses import build_loss

    torch.manual_seed(0)
    head = OrdinalHead(in_channels=8, num_ranks=4)
    features = torch.randn(6, 8)
    # No sample reaches rank 2, so the last classifier sees none and the average is over two.
    ranks = torch.tensor([0, 1, 0, 1, 1, 0])
    outputs = head(features)

    from_head = head.compute_loss(outputs, {"ranks": ranks})["ordinal_loss"]
    from_registry = build_loss("corn", num_ranks=4)(outputs["logits"], ranks)

    assert torch.equal(from_head, from_registry)
    assert from_registry.dtype == outputs["logits"].dtype


def test_the_corn_loss_accumulates_in_the_logits_dtype():
    from tcip_mcp.pipelines.components.losses import build_loss

    logits = torch.randn(4, 2, dtype=torch.float64)
    loss = build_loss("corn", num_ranks=3)(logits, torch.tensor([0, 1, 2, 1]))
    assert loss.dtype == torch.float64
