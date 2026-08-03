"""Review->retrain MCP tools (materialize + queue + lineage + registration)."""

from __future__ import annotations

import json
from pathlib import Path

from tcip_mcp.tools.feedback_tools import materialize_review_dataset, prioritize_review_queue


def _setup(tmp_path: Path):
    from PIL import Image
    state_dir = tmp_path / "state"
    src = tmp_path / "src"
    state_dir.mkdir()
    src.mkdir()
    for name in ("imgA.png", "imgB.png"):
        Image.new("RGB", (64, 64), (120, 120, 120)).save(src / name)
    state = {"image": {
        "imgA.png": {"img_status": "completed", "detections": [
            {"action": "accepted", "class_name": "catkin", "gt_bbox_norm": [0.5, 0.5, 0.2, 0.2], "pred_bbox_norm": None}]},
        "imgB.png": {"img_status": "completed", "detections": [
            {"action": "rejected", "class_name": "catkin", "gt_bbox_norm": None, "pred_bbox_norm": [0.8, 0.8, 0.1, 0.1]}]},
    }}
    # Seed through the engine so the fixture cannot drift from the real shard format.
    from tcip_annotation.review_engine import ReviewEngine

    engine = ReviewEngine(str(state_dir))
    engine.raw_state.update(state)
    engine.save_review_state()
    return state_dir, src


def test_materialize_review_dataset_end_to_end(tmp_path):
    state_dir, src = _setup(tmp_path)
    out = tmp_path / "out"
    r = materialize_review_dataset(str(state_dir), str(src), str(out))
    assert "error" not in r
    assert r["positive"] == 1 and r["hard_negative"] == 1
    assert (out / "images" / "imgA.png").is_file()
    assert (out / "annotations" / "imgA.json").is_file()


def test_materialize_review_dataset_records_lineage(tmp_path, monkeypatch):
    import tcip_mcp.experiments as experiments
    monkeypatch.setattr(experiments, "EXPERIMENTS_DIR", tmp_path / "exp")
    experiments.create_experiment("exp1", {"x": 1})

    state_dir, src = _setup(tmp_path)
    out = tmp_path / "out"
    r = materialize_review_dataset(str(state_dir), str(src), str(out), experiment_id="exp1")
    assert r["experiment_id"] == "exp1"

    lineage = json.loads((tmp_path / "exp" / "exp1" / "lineage.json").read_text())
    assert lineage["data_source"] == str(state_dir)
    assert "review_session" in lineage
    artifacts = json.loads((tmp_path / "exp" / "exp1" / "artifacts.json").read_text())
    assert artifacts["curated_dataset"]["path"] == str(out)


def test_materialize_creates_experiment_when_absent(tmp_path, monkeypatch):
    import tcip_mcp.experiments as experiments
    monkeypatch.setattr(experiments, "EXPERIMENTS_DIR", tmp_path / "exp")

    state_dir, src = _setup(tmp_path)
    r = materialize_review_dataset(str(state_dir), str(src), str(tmp_path / "out"), experiment_id="new1")
    assert r["experiment_id"] == "new1"
    lineage = json.loads((tmp_path / "exp" / "new1" / "lineage.json").read_text())
    assert "review_session" in lineage


def test_materialize_invalid_inputs_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()  # no review/ shards
    assert "error" in materialize_review_dataset(str(empty), str(tmp_path), str(tmp_path / "o1"))

    state_dir, _src = _setup(tmp_path)
    assert "error" in materialize_review_dataset(str(state_dir), str(tmp_path / "nope"), str(tmp_path / "o2"))


def test_prioritize_review_queue_checkpoint_missing(tmp_path):
    r = prioritize_review_queue(str(tmp_path / "nope.pt"), str(tmp_path))
    assert "error" in r  # early guard, no torch import needed


def test_prioritize_review_queue_rejects_non_composed_kind(tmp_path, monkeypatch):
    """Active-learning uncertainty scoring reads model logits, which a non-composed predictor
    kind (a bespoke tcip model was never built for) doesn't expose, so it must fail with a
    clear error, not crash on ``.model``.
    """
    from types import SimpleNamespace

    import tcip_mcp.pipelines.inference.predictor as predmod

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"x")
    # prioritize_review_queue imports build_predictor from the predictor module at call time.
    monkeypatch.setattr(predmod, "build_predictor",
                        lambda *a, **k: SimpleNamespace(kind="foreign_kind"))

    r = prioritize_review_queue(checkpoint_path=str(ckpt), images_dir=str(images))
    assert "error" in r and "foreign_kind" in r["error"]


def test_prioritize_review_queue_confidence_triage_surfaces_unscoreable(tmp_path, monkeypatch):
    """A regression checkpoint's predictions carry no confidence signal at all (RegressionHead's
    point estimate, deliberately no distributional output). confidence_triage must route them
    into review rather than silently drop them from every output, and tag them distinctly via
    unscoreable_images so a caller can tell this apart from a genuinely medium-confidence item."""
    from types import SimpleNamespace

    import tcip_mcp.pipelines.inference.predictor as predmod

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"x")

    predictions = [{"image": "a.jpg", "width": 4, "height": 4, "head0_values": [0.42]}]
    monkeypatch.setattr(
        predmod, "build_predictor",
        lambda *a, **k: SimpleNamespace(predict_batch=lambda sources: predictions))

    r = prioritize_review_queue(
        checkpoint_path=str(ckpt), images_dir=str(images), strategy="confidence_triage")
    assert r["needs_review"] == 1
    assert r["review_images"] == ["a.jpg"]
    assert r["unscoreable_images"] == ["a.jpg"]
    assert r["auto_accepted_images"] == []


def test_unresolvable_scorer_raises_valueerror_not_an_import_error():
    """The refusal is a ValueError whatever the name looks like.

    ``build_scorer``'s callers catch ``ValueError`` to turn a refusal into an error dict. A dotted
    name that fails to import used to raise ``ModuleNotFoundError`` straight out of the audited
    MCP tool instead.
    """
    import pytest

    from tcip_mcp.pipelines.active_learning.helpers import build_scorer

    with pytest.raises(ValueError):
        build_scorer("no_such_scorer", "detection")
    with pytest.raises(ValueError, match="Could not import scorer"):
        build_scorer("not_a_module.at_all:make", "detection")


def test_unresolvable_proposal_engine_raises_valueerror():
    import pytest

    from tcip_mcp.pipelines.proposal import resolve_proposer

    with pytest.raises(ValueError):
        resolve_proposer("no_such_engine")
    with pytest.raises(ValueError, match="Could not import proposal engine"):
        resolve_proposer("not_a_module.at_all:make")


def test_feedback_tools_register_in_manifest():
    from tcip_mcp.server import list_registered_tools
    names = list_registered_tools()
    assert "materialize_review_dataset" in names
    assert "prioritize_review_queue" in names
