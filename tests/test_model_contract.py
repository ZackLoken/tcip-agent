"""The model_source build seam and the thin measurement-boundary contract.

Covers ``build_from_model_source`` dispatch (``model_source`` only, validated with its config)
and the behavioral ``check_model_contract`` / ``overfit_check`` utilities on real bespoke models.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_mcp.pipelines.model_build import build_from_model_source  # noqa: E402
from tcip_mcp.pipelines.schemas import checked_train_config  # noqa: E402
from tcip_mcp.pipelines.model_contract import (  # noqa: E402
    TCIPModel,
    check_model_contract,
    overfit_check,
)
from tcip_store import check_json_value  # noqa: E402
from tests import bespoke_models  # noqa: E402

# What the smokes below synthesize their batch at, the shape a run resolves for itself.
CLS_DIMS = {"in_chans": 3, "num_classes": 2, "img_size": 64}
DET_DIMS = {"in_chans": 3, "num_classes": 1, "img_size": 64}


def _bespoke_builder(**kwargs):
    """An importable 'agent-written' builder: a real bespoke classification module."""
    return bespoke_models.build_bespoke_classifier(num_classes=2)


def _source(builder: str):
    """``builder``'s classification ``model_source`` declaring this module, validated in a run
    config over no data, and the layout its admission stages (``model_build.staged_sources``)."""
    from tcip_mcp.pipelines.model_build import staged_sources
    from tcip_mcp.pipelines.schemas import train_config
    from tests import REPO_ROOT
    from tests._chain_fixtures import training_config

    spec = train_config(training_config(
        {"builder": builder, "builder_kwargs": {}, "source_files": [__file__],
         "task": "classification"}, {}))
    return spec.model_source, staged_sources(spec, REPO_ROOT).layout


def test_build_from_model_source_imports_builder():
    source, layout = _source(f"{Path(__file__).stem}:_bespoke_builder")
    model = build_from_model_source(source, layout, {"in_chans": 3, "num_classes": 2})
    assert isinstance(model, TCIPModel)


def test_a_config_stating_no_model_source_is_refused_at_validation():
    _spec, issues = checked_train_config({"data": {}})
    assert any(issue.startswith("model_source") for issue in issues), issues


def test_build_from_model_source_bad_builder_raises():
    with pytest.raises(ValueError, match="not found|Invalid dotted"):
        source, layout = _source(f"{Path(__file__).stem}:does_not_exist")
        build_from_model_source(source, layout, {"in_chans": 3})


# --------------------------------------------------------------------------
# TCIPModel Protocol: duck-type marker, not an architecture requirement
# --------------------------------------------------------------------------

def test_tcip_model_protocol_membership():
    assert isinstance(bespoke_models.build_bespoke_classifier(num_classes=2), TCIPModel)
    assert not isinstance(object(), TCIPModel)


# --------------------------------------------------------------------------
# check_model_contract: behavioral smoke on the measurement boundary
# --------------------------------------------------------------------------

def test_check_model_contract_classification_ok():
    model = bespoke_models.build_bespoke_classifier(num_classes=2)
    report = check_model_contract(model, "classification", dims=CLS_DIMS)
    assert report["ok"], report["issues"]
    assert report["eval_output_type"] == "dict"
    assert report["train_loss"] is not None


def test_check_model_contract_records_per_parameter_gradient_magnitudes():
    """A model whose parameters all took zero gradient must read as what it is: the presence
    conjunct only asks whether a .grad exists, not what it is, so the report also states each
    named parameter's gradient norm rather than gating the report on presence alone."""
    model = bespoke_models.build_bespoke_classifier(num_classes=2)
    report = check_model_contract(model, "classification", dims=CLS_DIMS)
    assert report["ok"], report["issues"]
    mags = report["gradient_magnitudes"]
    assert isinstance(mags, dict) and mags
    named = dict(model.named_parameters())
    assert set(mags) <= set(named)  # every reported name is a real parameter of this model
    assert all(isinstance(v, float) and v >= 0.0 for v in mags.values())


def test_check_model_contract_gradient_norm_survives_a_float32_overflow():
    """Each element of this parameter's gradient is finite (2e38, under float32's ~3.4e38 max), so
    the elementwise finiteness gate passes, but summing their squares in the gradient's own
    float32 dtype overflows: ``torch.tensor([2e38, 2e38]).norm()`` is ``inf`` even though every
    element is finite. The recorded magnitude must be computed at higher precision so the report
    stays the JSON-encodable value ``check_json_value`` needs downstream, never an ``inf``."""
    import torch.nn as nn

    class _NearOverflowGradient(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.ones(2))
            self.w.register_hook(lambda grad: grad * 2e38)
            self.lin = nn.Linear(4, 4)

        def forward(self, images, targets=None):
            if self.training:
                return {"loss": self.w.sum() + self.lin(torch.rand(1, 4)).sum() * 0}
            return {"logits": self.lin(torch.rand(1, 4))}

    report = check_model_contract(_NearOverflowGradient(), "classification", dims=CLS_DIMS)
    assert report["ok"], report["issues"]
    norm = report["gradient_magnitudes"]["w"]
    assert math.isfinite(norm)
    assert norm == pytest.approx(2e38 * 2 ** 0.5, rel=1e-6)


def test_check_model_contract_rejects_prediction_free_eval_output():
    """A dict is the shape, not the content: an output with no tensor is not a measurement."""
    import torch.nn as nn

    class _Empty(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, images, targets=None):
            if self.training:
                return {"loss": self.lin(torch.rand(1, 4)).sum()}
            return {"note": "done"}

    report = check_model_contract(_Empty(), "classification", dims=CLS_DIMS)
    assert report["ok"] is False
    assert any("no tensor value" in i for i in report["issues"]), report["issues"]


def test_check_model_contract_accepts_a_ragged_nested_eval_output():
    """A bespoke non-detection task may legitimately return a per-image ragged list of tensors (or
    a nested dict of them) rather than one flat top-level tensor, e.g. a variable number of
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

    report = check_model_contract(_Ragged(), "classification", dims=CLS_DIMS)
    assert report["ok"], report["issues"]


def test_check_model_contract_detection_ok():
    model = bespoke_models.build_bespoke_detection(num_classes=1, min_size=64, max_size=128)
    report = check_model_contract(model, "detection", dims=DET_DIMS)
    assert report["ok"], report["issues"]
    assert report["eval_output_type"] == "list[dict]"


# --------------------------------------------------------------------------
# overfit_check: the loss must fall on a fixed tiny batch
# --------------------------------------------------------------------------

def test_overfit_check_classification_passes():
    model = bespoke_models.build_bespoke_classifier(num_classes=2)
    report = overfit_check(model, "classification", steps=25, dims=CLS_DIMS, seed=0)
    assert report["passed"], report["issue"]
    assert report["final"] < report["initial"]
    assert len(report["losses"]) == 25


# render_overfit_report: the record-safe form of a raw overfit_check report

def test_render_overfit_report_passes_through_a_finite_report():
    from tcip_mcp.pipelines.model_contract import render_overfit_report

    model = bespoke_models.build_bespoke_classifier(num_classes=2)
    raw = overfit_check(model, "classification", steps=5, dims=CLS_DIMS, seed=0)
    rendered = render_overfit_report(raw)
    check_json_value(rendered, path="overfit_check")  # never raises for a finite report
    assert rendered["passed"] == raw["passed"]
    assert rendered["initial"] == raw["initial"]
    assert rendered["final"] == raw["final"]
    assert rendered["losses"] == raw["losses"]


def test_render_overfit_report_keeps_a_diverging_models_state():
    """A diverging model's raw report can carry nan, which check_json_value refuses. The
    rendered form keeps the state as a named companion field instead of losing it."""
    from tcip_mcp.pipelines.model_contract import render_overfit_report

    raw = {"passed": False, "issue": "loss became non-finite during overfitting",
           "losses": [1.0, float("nan"), float("inf")],
           "initial": 1.0, "final": float("nan")}
    rendered = render_overfit_report(raw)
    check_json_value(rendered, path="overfit_check")
    assert rendered["final"] is None
    assert rendered["final_state"] == "nan"
    assert rendered["initial"] == 1.0
    assert "initial_state" not in rendered
    assert rendered["losses"] == [1.0, None, None]
