"""S1 — build_model indirection + the thin measurement-boundary contract.

Covers ``build_model`` dispatch (``model_source`` only) and the behavioral
``check_model_contract`` / ``overfit_check`` utilities on real bespoke models.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_mcp.pipelines.model_build import build_from_model_source, build_model  # noqa: E402
from tcip_mcp.pipelines.model_contract import (  # noqa: E402
    TCIPModel,
    check_model_contract,
    overfit_check,
)
from tests import bespoke_models  # noqa: E402


def _bespoke_builder(**kwargs):
    """An importable 'agent-written' builder — a real bespoke classification module."""
    return bespoke_models.build_bespoke_classifier(num_classes=2)


# --------------------------------------------------------------------------
# build_model dispatch
# --------------------------------------------------------------------------

def test_build_model_from_model_source_imports_builder():
    src = {"builder": f"{__name__}:_bespoke_builder", "builder_kwargs": {}, "task": "classification"}
    model = build_model({"model_source": src})
    assert isinstance(model, TCIPModel)
    # build_from_model_source is the same path, callable directly.
    assert type(build_from_model_source(src)).__name__ == type(model).__name__


def test_build_model_requires_model_source():
    with pytest.raises(ValueError, match="model_source"):
        build_model({})


def test_build_model_bad_builder_raises():
    with pytest.raises(ValueError, match="not found|Invalid dotted"):
        build_model({"model_source": {"builder": f"{__name__}:does_not_exist"}})


# --------------------------------------------------------------------------
# TCIPModel Protocol — duck-type marker, not an architecture requirement
# --------------------------------------------------------------------------

def test_tcip_model_protocol_membership():
    assert isinstance(bespoke_models.build_bespoke_classifier(num_classes=2), TCIPModel)
    assert not isinstance(object(), TCIPModel)


# --------------------------------------------------------------------------
# check_model_contract — behavioral smoke on the measurement boundary
# --------------------------------------------------------------------------

def test_check_model_contract_classification_ok():
    model = bespoke_models.build_bespoke_classifier(num_classes=2)
    report = check_model_contract(model, "classification", num_classes=2)
    assert report["ok"], report["issues"]
    assert report["eval_output_type"] == "dict"
    assert report["train_loss"] is not None


def test_check_model_contract_rejects_prediction_free_eval_output():
    """A dict is the shape, not the content — an output with no tensor is not a measurement."""
    import torch.nn as nn

    class _Empty(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, images, targets=None):
            if self.training:
                return {"loss": self.lin(torch.rand(1, 4)).sum()}
            return {"note": "done"}

    report = check_model_contract(_Empty(), "classification", num_classes=2)
    assert report["ok"] is False
    assert any("no tensor value" in i for i in report["issues"]), report["issues"]


def test_check_model_contract_accepts_a_ragged_nested_eval_output():
    """A bespoke non-detection task may legitimately return a per-image ragged list of tensors (or
    a nested dict of them) rather than one flat top-level tensor — e.g. a variable number of
    per-instance predictions per image. The eval-output check must see the tensor wherever it is
    nested, not only at the top level of the dict."""
    import torch.nn as nn

    class _Ragged(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, images, targets=None):
            if self.training:
                return {"loss": self.lin(torch.rand(1, 4)).sum()}
            # Ragged: a different number of per-instance score tensors per image.
            return {"per_image": [torch.rand(3), torch.rand(1), torch.rand(0)]}

    report = check_model_contract(_Ragged(), "classification", num_classes=2)
    assert report["ok"], report["issues"]


def test_check_model_contract_detection_ok():
    model = bespoke_models.build_bespoke_detection(num_classes=1, min_size=64, max_size=128)
    report = check_model_contract(model, "detection", num_classes=1, img_size=64)
    assert report["ok"], report["issues"]
    assert report["eval_output_type"] == "list[dict]"


# --------------------------------------------------------------------------
# overfit_check — the loss must fall on a fixed tiny batch
# --------------------------------------------------------------------------

def test_overfit_check_classification_passes():
    model = bespoke_models.build_bespoke_classifier(num_classes=2)
    report = overfit_check(model, "classification", steps=25, num_classes=2, seed=0)
    assert report["passed"], report["issue"]
    assert report["final"] < report["initial"]
    assert len(report["losses"]) == 25
