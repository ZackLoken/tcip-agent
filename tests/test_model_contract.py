"""The model_source build seam and the thin measurement-boundary contract.

These tests cover ``build_from_model_source`` dispatch from a ``model_source`` validated with its
config, and the behavioral ``check_model_contract`` and ``overfit_check`` utilities on real
bespoke models.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from torch import Tensor

    from tcip_mcp.pipelines.model_build import SourceLayout
    from tcip_mcp.pipelines.schemas import ModelSourceSchema

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_mcp.pipelines.model_build import build_from_model_source  # noqa: E402
from tcip_mcp.pipelines.model_contract import (  # noqa: E402
    check_model_contract,
    overfit_check,
)
from tcip_mcp.pipelines.schemas import (  # noqa: E402
    TrainGeometry,
    checked_train_config,
)
from tcip_store import check_json_value  # noqa: E402
from tests import bespoke_models  # noqa: E402

# What the smokes below synthesize their batch at, the shape a run resolves for itself.
CLS_DIMS = {"in_chans": 3, "num_classes": 2, "img_size": (64, 64)}
DET_DIMS = {"in_chans": 3, "num_classes": 1, "img_size": (64, 64)}


def _classifier() -> bespoke_models.BespokeComposed:
    """Build a real bespoke two-class classifier at :func:`band_statistics`."""
    from tests._chain_fixtures import band_statistics

    return bespoke_models.build_bespoke_classifier(
        num_classes=2, **band_statistics()
    )


def _bespoke_builder(
    *, image_mean: list[float], image_std: list[float], **_handed: object
) -> bespoke_models.BespokeComposed:
    """Build a real bespoke classifier at the handed statistics, as an agent-written builder.

    The builder is importable and loads none of the weights it is handed.
    """
    return bespoke_models.build_bespoke_classifier(
        num_classes=2, image_mean=image_mean, image_std=image_std
    )


_UNRESOLVED = TrainGeometry(
    native_size=None, object_density=None, detector=None
)
"""The geometry of a classification run that resolved no frame."""


def _source(builder: str) -> tuple[ModelSourceSchema, SourceLayout]:
    """Return ``builder``'s classification ``model_source`` and the layout its admission stages.

    The model source declares this module, validated in a run config over no data; the layout
    is ``model_build.staged_sources``'s.
    """
    from tcip_mcp.pipelines.model_build import staged_sources
    from tcip_mcp.pipelines.schemas import train_config
    from tests import REPO_ROOT
    from tests._chain_fixtures import training_config, weights_normalization

    spec = train_config(
        training_config(
            REPO_ROOT,
            {
                "builder": builder,
                "builder_kwargs": {},
                "source_files": [__file__],
                "task": "classification",
                "normalization": weights_normalization(),
            },
            {},
        )
    )
    return spec.model_source, staged_sources(spec, REPO_ROOT).layout


def _not_a_module_builder(**_handed: object) -> dict:
    """Return a plain dict where a builder returns a module, as a mistaken agent builder."""
    return {"in_chans": _handed["in_chans"]}


_NOT_CALLABLE = 3
"""A module attribute a dotted builder name can reach that is not callable."""


def test_build_from_model_source_imports_builder() -> None:
    source, layout = _source(f"{Path(__file__).stem}:_bespoke_builder")
    model = build_from_model_source(
        source, layout, {"in_chans": 3, "num_classes": 2}, _UNRESOLVED
    )
    assert isinstance(model, torch.nn.Module)


def test_a_builder_returning_no_module_is_refused_naming_the_builder() -> None:
    """A model_source builder that returns no ``nn.Module`` is refused by its config path."""
    source, layout = _source(f"{Path(__file__).stem}:_not_a_module_builder")
    with pytest.raises(TypeError, match="model_source.builder"):
        build_from_model_source(
            source, layout, {"in_chans": 3, "num_classes": 2}, _UNRESOLVED
        )


def test_a_dotted_builder_naming_nothing_callable_is_refused() -> None:
    """A dotted builder name reaching a non-callable attribute is refused at its import."""
    source, layout = _source(f"{Path(__file__).stem}:_NOT_CALLABLE")
    with pytest.raises(ValueError, match="not callable"):
        build_from_model_source(
            source, layout, {"in_chans": 3, "num_classes": 2}, _UNRESOLVED
        )


def test_a_config_stating_no_model_source_is_refused_at_validation() -> None:
    _spec, issues = checked_train_config({"data": {}})
    assert any(issue.startswith("model_source") for issue in issues), issues


def test_build_from_model_source_bad_builder_raises() -> None:
    with pytest.raises(ValueError, match="not found|Invalid dotted"):
        source, layout = _source(f"{Path(__file__).stem}:does_not_exist")
        build_from_model_source(source, layout, {"in_chans": 3}, _UNRESOLVED)


# --------------------------------------------------------------------------
# check_model_contract: behavioral smoke on the measurement boundary
# --------------------------------------------------------------------------


def test_check_model_contract_classification_ok() -> None:
    model = _classifier()
    report = check_model_contract(model, "classification", dims=CLS_DIMS)
    assert report["ok"], report["issues"]
    assert report["eval_output_type"] == "dict"
    assert report["train_loss"] is not None


def test_check_model_contract_records_per_parameter_gradient_magnitudes() -> (
    None
):
    """The model contract records gradient magnitudes keyed by the model's parameter names.

    Each reported name is a parameter of the model, and each magnitude is a non-negative float.
    """
    model = _classifier()
    report = check_model_contract(model, "classification", dims=CLS_DIMS)
    assert report["ok"], report["issues"]
    mags = report["gradient_magnitudes"]
    assert isinstance(mags, dict) and mags
    named = dict(model.named_parameters())
    assert set(mags) <= set(
        named
    )  # every reported name is a real parameter of this model
    assert all(isinstance(v, float) and v >= 0.0 for v in mags.values())


def test_check_model_contract_gradient_norm_survives_a_float32_overflow() -> (
    None
):
    """The contract's gradient norm survives a float32 overflow as a finite value.

    Each element of the parameter's gradient is 2e38, finite in float32, and the reported norm
    is finite and equals 2e38 times the square root of two.
    """
    import torch.nn as nn

    class _NearOverflowGradient(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.ones(2))
            self.w.register_hook(lambda grad: grad * 2e38)
            self.lin = nn.Linear(4, 4)

        def forward(
            self, images: Tensor, targets: dict | None = None
        ) -> dict[str, Tensor]:
            if self.training:
                return {
                    "loss": self.w.sum() + self.lin(torch.rand(1, 4)).sum() * 0
                }
            return {"logits": self.lin(torch.rand(1, 4))}

    report = check_model_contract(
        _NearOverflowGradient(), "classification", dims=CLS_DIMS
    )
    assert report["ok"], report["issues"]
    norm = report["gradient_magnitudes"]["w"]
    assert math.isfinite(norm)
    assert norm == pytest.approx(2e38 * 2**0.5, rel=1e-6)


def test_check_model_contract_rejects_prediction_free_eval_output() -> None:
    """The model contract rejects an eval output holding no tensor."""
    import torch.nn as nn

    class _Empty(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(
            self, images: Tensor, targets: dict | None = None
        ) -> dict[str, Tensor] | dict[str, str]:
            if self.training:
                return {"loss": self.lin(torch.rand(1, 4)).sum()}
            return {"note": "done"}

    report = check_model_contract(_Empty(), "classification", dims=CLS_DIMS)
    assert report["ok"] is False
    assert any("no tensor value" in i for i in report["issues"]), report[
        "issues"
    ]


def test_check_model_contract_accepts_a_ragged_nested_eval_output() -> None:
    """The model contract accepts a ragged nested eval output.

    An eval output holding, under one key, a list of per-image tensors of different lengths
    passes the contract.
    """
    import torch.nn as nn

    class _Ragged(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(
            self, images: Tensor, targets: dict | None = None
        ) -> dict[str, Tensor] | dict[str, list[Tensor]]:
            if self.training:
                return {"loss": self.lin(torch.rand(1, 4)).sum()}
            # Ragged: a different number of per-instance score tensors per image.
            return {"per_image": [torch.rand(3), torch.rand(1), torch.rand(0)]}

    report = check_model_contract(_Ragged(), "classification", dims=CLS_DIMS)
    assert report["ok"], report["issues"]


def test_check_model_contract_detection_ok() -> None:
    from tests._chain_fixtures import detector_values

    model = bespoke_models.build_bespoke_detection(
        num_classes=1, **detector_values()
    )
    report = check_model_contract(model, "detection", dims=DET_DIMS)
    assert report["ok"], report["issues"]
    assert report["eval_output_type"] == "list[dict]"


# --------------------------------------------------------------------------
# overfit_check: the loss must fall on a fixed tiny batch
# --------------------------------------------------------------------------


def test_overfit_check_classification_passes() -> None:
    model = _classifier()
    report = overfit_check(
        model,
        "classification",
        steps=25,
        dims=CLS_DIMS,
        seed=0,
        deterministic=False,
    )
    assert report["passed"], report["issue"]
    assert report["final"] < report["initial"]
    assert len(report["losses"]) == 25


# render_overfit_report: the record-safe form of a raw overfit_check report


def test_render_overfit_report_passes_through_a_finite_report() -> None:
    from tcip_mcp.pipelines.model_contract import render_overfit_report

    model = _classifier()
    raw = overfit_check(
        model,
        "classification",
        steps=5,
        dims=CLS_DIMS,
        seed=0,
        deterministic=False,
    )
    rendered = render_overfit_report(raw)
    check_json_value(
        rendered, path="overfit_check"
    )  # never raises for a finite report
    assert rendered["passed"] == raw["passed"]
    assert rendered["initial"] == raw["initial"]
    assert rendered["final"] == raw["final"]
    assert rendered["losses"] == raw["losses"]


def test_render_overfit_report_keeps_a_diverging_models_state() -> None:
    """The rendered overfit report keeps a diverging model's state as a named field.

    A raw report carrying nan and inf renders to a form check_json_value admits, the
    non-finite values null and the final value's state named beside it.
    """
    from tcip_mcp.pipelines.model_contract import render_overfit_report

    raw = {
        "passed": False,
        "issue": "loss became non-finite during overfitting",
        "losses": [1.0, float("nan"), float("inf")],
        "initial": 1.0,
        "final": float("nan"),
    }
    rendered = render_overfit_report(raw)
    check_json_value(rendered, path="overfit_check")
    assert rendered["final"] is None
    assert rendered["final_state"] == "nan"
    assert rendered["initial"] == 1.0
    assert "initial_state" not in rendered
    assert rendered["losses"] == [1.0, None, None]
