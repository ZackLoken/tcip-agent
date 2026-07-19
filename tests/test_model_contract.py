"""S1 — build_model indirection + the thin measurement-boundary contract.

Covers ``build_model`` dispatch (model_spec | model_source, exactly one) and the behavioral
``check_model_contract`` / ``overfit_check`` utilities on real composed models.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import tcip_mcp.pipelines.components.backbones  # noqa: F401,E402
import tcip_mcp.pipelines.components.necks  # noqa: F401,E402
import tcip_mcp.pipelines.components.heads  # noqa: F401,E402
import tcip_mcp.pipelines.components.losses  # noqa: F401,E402
from tcip_mcp.pipelines.composer import compose_model  # noqa: E402
from tcip_mcp.pipelines.model_build import build_from_model_source, build_model  # noqa: E402
from tcip_mcp.pipelines.model_contract import (  # noqa: E402
    TCIPModel,
    check_model_contract,
    overfit_check,
)


def _cls_spec() -> dict:
    return {
        "backbone": {"name": "resnet18", "pretrained": False},
        "neck": {"name": "gap"},
        "heads": [{"name": "classification", "num_classes": 2}],
    }


def _det_spec() -> dict:
    return {
        "backbone": {"name": "resnet18", "pretrained": False},
        "neck": {"name": "fpn", "out_channels": 256},
        "heads": [{"name": "anchor_detection", "num_classes": 1, "min_size": 64, "max_size": 128}],
    }


def _bespoke_builder(**kwargs):
    """An importable 'agent-written' builder — here it just wraps compose_model."""
    return compose_model(_cls_spec())


# --------------------------------------------------------------------------
# build_model dispatch
# --------------------------------------------------------------------------

def test_build_model_spec_path_matches_compose_model():
    model = build_model({"model_spec": _cls_spec()})
    # Same class the composer produces — the default path is unchanged.
    assert type(model).__name__ == type(compose_model(_cls_spec())).__name__


def test_build_model_from_model_source_imports_builder():
    src = {"builder": f"{__name__}:_bespoke_builder", "builder_kwargs": {}, "task": "classification"}
    model = build_model({"model_source": src})
    assert isinstance(model, TCIPModel)
    # build_from_model_source is the same path, callable directly.
    assert type(build_from_model_source(src)).__name__ == type(model).__name__


def test_build_model_requires_exactly_one():
    with pytest.raises(ValueError, match="exactly one"):
        build_model({"model_spec": _cls_spec(), "model_source": {"builder": f"{__name__}:_bespoke_builder"}})
    with pytest.raises(ValueError, match="neither"):
        build_model({})


def test_build_model_bad_builder_raises():
    with pytest.raises(ValueError, match="not found|Invalid dotted"):
        build_model({"model_source": {"builder": f"{__name__}:does_not_exist"}})


# --------------------------------------------------------------------------
# TCIPModel Protocol — duck-type marker, not an architecture requirement
# --------------------------------------------------------------------------

def test_tcip_model_protocol_membership():
    assert isinstance(compose_model(_cls_spec()), TCIPModel)
    assert not isinstance(object(), TCIPModel)


# --------------------------------------------------------------------------
# check_model_contract — behavioral smoke on the measurement boundary
# --------------------------------------------------------------------------

def test_check_model_contract_classification_ok():
    model = compose_model(_cls_spec())
    report = check_model_contract(model, "classification", num_classes=2)
    assert report["ok"], report["issues"]
    assert report["eval_output_type"] == "dict"
    assert report["train_loss"] is not None


def test_check_model_contract_detection_ok():
    model = compose_model(_det_spec())
    report = check_model_contract(model, "detection", num_classes=1, img_size=64)
    assert report["ok"], report["issues"]
    assert report["eval_output_type"] == "list[dict]"


# --------------------------------------------------------------------------
# overfit_check — the loss must fall on a fixed tiny batch
# --------------------------------------------------------------------------

def test_overfit_check_classification_passes():
    model = compose_model(_cls_spec())
    report = overfit_check(model, "classification", steps=25, num_classes=2, seed=0)
    assert report["passed"], report["issue"]
    assert report["final"] < report["initial"]
    assert len(report["losses"]) == 25
