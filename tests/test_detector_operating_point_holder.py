"""Where a detector's operating-point knobs live: the module itself, its ``.detector.roi_heads``,
or its ``.detector``, resolved independently of what a given call actually applies.

A module exposing a knob on itself, with no ``.detector`` to route through, already reached a
validated operating point before this change (the ``getattr`` chain already fell through to the
module when it had no ``.detector``); what changed is the interface being stated once
(:func:`~tcip_mcp.pipelines.operating_point.detector_operating_point_holder`), read by both the
setter and the model contract, and the two unstated-floor producers (a module with no knob, the
review route's own unknowns) sharing one gate name distinct from a stated floor the pick does not
clear.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

IMG = 32


def test_holder_resolves_to_the_module_itself_with_no_detector():
    from tcip_mcp.pipelines.operating_point import detector_operating_point_holder

    model = SimpleNamespace(score_thresh=0.5)
    holder, path = detector_operating_point_holder(model)
    assert holder is model and path == "self"


def test_holder_resolves_to_detector_roi_heads_for_a_two_stage_wrapper():
    from tcip_mcp.pipelines.operating_point import detector_operating_point_holder

    roi_heads = SimpleNamespace(score_thresh=0.5)
    model = SimpleNamespace(detector=SimpleNamespace(roi_heads=roi_heads))
    holder, path = detector_operating_point_holder(model)
    assert holder is roi_heads and path == "detector.roi_heads"


def test_holder_resolves_to_detector_itself_for_a_one_stage_wrapper():
    from tcip_mcp.pipelines.operating_point import detector_operating_point_holder

    detector = SimpleNamespace(score_thresh=0.5)
    model = SimpleNamespace(detector=detector)
    holder, path = detector_operating_point_holder(model)
    assert holder is detector and path == "detector"


def test_holder_is_none_when_nothing_exposes_a_knob():
    from tcip_mcp.pipelines.operating_point import detector_operating_point_holder

    model = SimpleNamespace(unrelated=1)
    holder, path = detector_operating_point_holder(model)
    assert holder is None and path is None


def test_holder_refuses_when_the_module_and_its_detectors_roi_heads_both_expose_a_knob():
    """An ambiguous module, ambiguous in the same shape a real bespoke wrapper could build: a
    torchvision two-stage detector under ``.detector`` (whose ``roi_heads`` already exposes the
    knob) plus a knob restated on the wrapper itself. The platform must not silently pick one."""
    from tcip_mcp.pipelines.model_build import build_model

    model = build_model({"model_source": {
        "builder": "tests.bespoke_models:build_bespoke_detection",
        "builder_kwargs": {"num_classes": 1, "in_chans": 3, "min_size": 64, "max_size": 128},
        "task": "detection"}})
    assert hasattr(model.detector, "roi_heads") and hasattr(model.detector.roi_heads, "score_thresh")
    model.score_thresh = 0.5  # restated on the wrapper itself, ambiguous with .detector.roi_heads

    from tcip_mcp.pipelines.operating_point import detector_operating_point_holder

    with pytest.raises(ValueError, match="more than one location"):
        detector_operating_point_holder(model)


def test_set_detector_operating_point_returns_the_attribute_path():
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point

    model = SimpleNamespace(score_thresh=0.5)
    applied, attribute_path = set_detector_operating_point(model, score_thresh=0.2)
    assert applied["score_thresh"] == 0.2
    assert attribute_path == "self"


def test_set_detector_operating_point_reports_no_path_when_nothing_matches():
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point

    model = SimpleNamespace(unrelated=1)
    applied, attribute_path = set_detector_operating_point(model, score_thresh=0.2)
    assert applied.get("score_thresh") is None
    assert attribute_path is None


def _checkpoint(tmp_path, builder: str) -> str:
    from tcip_mcp.pipelines.model_build import build_model
    from tcip_mcp.tools.model_tools import register_model

    model_source = {"builder": f"tests.bespoke_models:{builder}",
                    "builder_kwargs": {"in_chans": 3}, "task": "detection"}
    ckpt = tmp_path / "model_best.pt"
    torch.save({"model_source": model_source,
                "model_state_dict": build_model({"model_source": model_source}).state_dict(),
                "config": {"data": {"tiling": {"enabled": False}}}}, str(ckpt))
    reg = register_model(name=builder, checkpoint_path=str(ckpt), config={},
                         project_path=str(tmp_path))
    assert "error" not in reg, reg
    return str(ckpt)


def _labeled_reference(tmp_path):
    """A handful of images, each its own size, GT at exactly the box
    ``BareScoreThreshDetector``/``BareNoKnobDetector`` always predict for that size, so every image
    matches perfectly and no two images collide on content.
    """
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    labels_dir = tmp_path / "labels"
    images_dir = tmp_path / "images"
    labels_dir.mkdir()
    images_dir.mkdir()
    for i, size in enumerate((32, 40, 48, 56, 64, 72)):
        Image.new("RGB", (size, size), (100, 100, 100)).save(images_dir / f"img{i}.png")
        box = BBox(size * 0.25, size * 0.25, size * 0.75, size * 0.75)
        json_io.write_annotations(str(labels_dir / f"img{i}.json"),
                                  [Annotation(subject="bud", geometry=box)], size, size)
    return images_dir, labels_dir


def _persisted_gate_evidence(run_inference_result: dict) -> dict:
    """The full gate-evidence dict a ``run_inference`` calibration persisted, read back by the same
    key the response names (``calibration_evidence_key``): the response's own
    ``gate_evidence_summary`` is a compact, response-safe subset, this is the whole thing.
    """
    from tcip_store import store

    from tcip_mcp.tools.inference_tools import calibration_curve_key

    body = store.read(calibration_curve_key(run_inference_result["calibration_evidence_key"]))
    return body["gate_evidence"]


def test_a_bespoke_module_exposing_its_own_knob_reaches_a_validated_point(tmp_path, monkeypatch):
    """Built through build_model, calibrated through run_inference: a hand-rolled, non-torchvision
    module that exposes score_thresh on itself reaches a validated operating point, with the
    attribute path it was applied on recorded. Also the admitting half of the curve-identity
    codec check: an ordinary calibration's evidence carries nothing the codec refuses, so it
    survives the check test_checkpoint_digest_rails.py's NaN-evidence test drives to a refusal."""
    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    monkeypatch.chdir(tmp_path)
    ckpt = _checkpoint(tmp_path, "build_bare_score_thresh_detector")
    images_dir, labels_dir = _labeled_reference(tmp_path)

    r = run_inference(ckpt, images_dir=str(images_dir), device="cpu", tile=False,
                      trait="bud_opening", calibration_labels_dir=str(labels_dir))

    assert "error" not in r, r
    assert r["validated"] is True
    gate_evidence = _persisted_gate_evidence(r)
    assert gate_evidence["failures"] == []
    assert gate_evidence["staged_conf_floor_attribute_path"] == "self"
    assert gate_evidence["conf_censored"] is False
    assert "conf_floor_unstated" not in gate_evidence["failures"]

    # The curve round trip: the response's own digest agrees with the record read back under it.
    from tcip_store import store

    from tcip_mcp.tools.inference_tools import (
        _calibration_evidence, calibration_curve_identity, calibration_curve_key,
    )

    key = r["calibration_evidence_key"]
    body = store.read(calibration_curve_key(key))
    assert calibration_curve_identity(body) == key
    assert _calibration_evidence(r) == body["calibration_evidence"]
    assert body["schema_version"] == 2


def test_a_module_exposing_no_knob_refuses_unstated_not_censored(tmp_path, monkeypatch):
    """A module exposing no operating-point knob under any recognized name has no floor the
    platform can state, and refuses with conf_floor_unstated, never conf_censored."""
    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    monkeypatch.chdir(tmp_path)
    ckpt = _checkpoint(tmp_path, "build_bare_no_knob_detector")
    images_dir, labels_dir = _labeled_reference(tmp_path)

    r = run_inference(ckpt, images_dir=str(images_dir), device="cpu", tile=False,
                      trait="bud_opening", calibration_labels_dir=str(labels_dir))

    assert "error" not in r, r
    assert r["validated"] is False
    gate_evidence = _persisted_gate_evidence(r)
    assert gate_evidence["staged_conf_floor_attribute_path"] is None
    assert gate_evidence["conf_censored"] is False
    assert "conf_floor_unstated" in gate_evidence["failures"]
    assert "conf_censored" not in gate_evidence["failures"]


def test_model_contract_records_the_holders_own_knobs():
    from tcip_mcp.pipelines.model_build import build_model
    from tcip_mcp.pipelines.model_contract import check_model_contract

    with_knob = build_model({"model_source": {
        "builder": "tests.bespoke_models:build_bare_score_thresh_detector",
        "builder_kwargs": {"in_chans": 3}, "task": "detection"}})
    without_knob = build_model({"model_source": {
        "builder": "tests.bespoke_models:build_bare_no_knob_detector",
        "builder_kwargs": {"in_chans": 3}, "task": "detection"}})

    assert check_model_contract(with_knob, "detection")["operating_point_knobs"] == ["score_thresh"]
    assert check_model_contract(without_knob, "detection")["operating_point_knobs"] == []
