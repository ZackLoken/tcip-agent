"""S3 — code+env provenance for a bespoke (model_source) run.

Locks: snapshot_model_source (copy source files + sha256 + env + seed), KIND_TCIP_MODULE stamping,
the _kind_from_ckpt structural fallback, build_predictor rebuilding a bespoke model from its
importable builder (NO exec) + predicting, and register_model_from_experiment round-tripping the
bespoke kind.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import tcip_mcp.pipelines.components.backbones  # noqa: F401,E402
import tcip_mcp.pipelines.components.necks  # noqa: F401,E402
import tcip_mcp.pipelines.components.heads  # noqa: F401,E402
import tcip_mcp.pipelines.components.losses  # noqa: F401,E402
from tcip_mcp.pipelines.composer import DetectionModel, compose_model  # noqa: E402
from tcip_mcp.pipelines.inference.predictor import (  # noqa: E402
    KIND_TCIP_MODULE,
    _kind_from_ckpt,
    build_predictor,
    detect_kind,
)
from tcip_mcp.pipelines.model_build import (  # noqa: E402
    build_model,
    snapshot_model_source,
    stamp_model_ref,
)


def build_bespoke_detector(**kwargs):
    """An importable 'agent-written' builder — a from-scratch detector (here via compose_model)."""
    return compose_model({
        "backbone": {"name": "resnet18", "pretrained": False},
        "neck": {"name": "fpn", "out_channels": 64},
        "heads": [{"name": "anchor_detection", "num_classes": 1, "min_size": 64, "max_size": 128}],
    })


def _model_source() -> dict:
    return {"builder": f"{__name__}:build_bespoke_detector", "task": "detection",
            "in_chans": 3, "source_files": [__file__]}


# --------------------------------------------------------------------------
# snapshot_model_source — source files + sha256 + env + seed
# --------------------------------------------------------------------------

def test_snapshot_model_source_copies_files_and_records_provenance(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    manifest = snapshot_model_source({"model_source": _model_source(), "seed": 123}, exp_dir)

    assert manifest is not None
    assert (exp_dir / "model_src" / "manifest.json").is_file()
    assert (exp_dir / "model_src" / Path(__file__).name).is_file()  # this file was copied
    expected_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert any(e["sha256"] == expected_sha for e in manifest["files"])
    assert manifest["builder"].endswith(":build_bespoke_detector")
    assert manifest["env"]["torch"]
    assert manifest["seed"] == 123


def test_snapshot_model_source_none_for_composed(tmp_path):
    assert snapshot_model_source({"model_spec": {"backbone": {"name": "resnet18"}}}, tmp_path) is None


# --------------------------------------------------------------------------
# KIND_TCIP_MODULE stamping + structural fallback
# --------------------------------------------------------------------------

def test_stamp_and_kind_fallback():
    payload = stamp_model_ref({"model_state_dict": {}}, {"model_source": _model_source()})
    assert payload["kind"] == KIND_TCIP_MODULE
    assert payload["model_source"] == _model_source()

    # An unstamped bespoke checkpoint is recognized structurally.
    assert _kind_from_ckpt({"model_source": _model_source(), "model_state_dict": {}}, "x.pt") == KIND_TCIP_MODULE


# --------------------------------------------------------------------------
# build_predictor rebuilds the bespoke model from its builder (no exec) + predicts
# --------------------------------------------------------------------------

def test_build_predictor_rebuilds_bespoke_and_predicts(tmp_path):
    from PIL import Image

    src = _model_source()
    model = build_model({"model_source": src})
    assert isinstance(model, DetectionModel)  # built via the importable builder, not the composer

    ckpt = tmp_path / "model_best.pt"
    payload = stamp_model_ref(
        {"model_state_dict": model.state_dict(), "metrics": {"val_loss": 0.3, "epoch": 1},
         "config": {"model_source": src}}, {"model_source": src})
    torch.save(payload, ckpt)

    assert detect_kind(str(ckpt)) == KIND_TCIP_MODULE  # kind sniffed from disk

    predictor = build_predictor(checkpoint_path=str(ckpt), device="cpu", score_threshold=0.0)
    assert predictor.kind == KIND_TCIP_MODULE
    assert predictor.task == "detection"
    assert predictor.in_chans == 3

    img = tmp_path / "a.png"
    Image.new("RGB", (64, 64), (120, 120, 120)).save(img)
    out = predictor.predict(str(img))
    assert {"boxes", "scores", "labels", "count"} <= set(out)  # measurable detection output


# --------------------------------------------------------------------------
# register_model_from_experiment round-trips the bespoke kind
# --------------------------------------------------------------------------

def test_register_round_trips_bespoke_kind(tmp_path):
    from tcip_mcp.experiments import create_experiment, register_model_from_experiment
    from tcip_mcp.model_registry import ModelRegistry

    src = _model_source()
    model = build_model({"model_source": src})
    ckpt = tmp_path / "model_best.pt"
    payload = stamp_model_ref(
        {"model_state_dict": model.state_dict(), "metrics": {"val_loss": 0.3, "epoch": 1}},
        {"model_source": src})
    torch.save(payload, ckpt)

    create_experiment("expB", {"model_source": src}, data_source="imgs")
    result = register_model_from_experiment("expB", str(ckpt))
    assert result["metrics"]["val_loss"] == pytest.approx(0.3)

    entry = ModelRegistry(str(tmp_path)).get_model("expB")
    assert entry is not None
    assert entry["kind"] == KIND_TCIP_MODULE   # round-tripped from the stamped checkpoint
    assert entry["sha256"] and len(entry["sha256"]) == 64
