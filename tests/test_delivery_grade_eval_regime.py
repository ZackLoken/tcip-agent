"""Delivery-grade evaluation runs in a different regime than inference: it resolves tile geometry
via the same shared ``resolve_tile_geometry`` ``run_inference`` uses (refusing rather than
scoring at an ungrounded scale when nothing is resolvable), honors ``max_dets`` verbatim on both
regimes with a per-image ``cap_hit``/``max_dets_cap_saturated_frac`` signal on the gating path, and
records
``tiled``'s provenance (``raw_operating_point``/``resolve_operating_point``) to distinguish an
explicit caller choice from a documented default, mirroring the existing
``tile_size``/``tile_size_source`` pattern. See ``test_detection_measurement_integrity.py`` for the
geometry-resolution and calibrated-bundle integration tests; this file covers the
``evaluate_model`` wrapper's passthrough + refusal handling.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

# No built-in traits: seed_bud_trait_spec (conftest.py) writes a real bud.yml into this
# test's pinned platform state root so trait/subject="bud" call sites keep resolving.
pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")


# ══════════════════════════════════════════════════════════════════════════
# max_dets honored verbatim (no rescuing sentinel)
# ══════════════════════════════════════════════════════════════════════════

def _det_dataset(tmp_path, n=3, size=128):
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (size, size), color=(120, 120, 120)).save(images_dir / f"img{i}.png")
        json_io.write_annotations(str(labels_dir / f"img{i}.json"),
                                  [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], size, size)
    return images_dir, labels_dir


def test_gating_path_honors_explicit_max_dets_le_100(tmp_path, monkeypatch):
    """training_tools.evaluate_model's use_tiled_inference branch must honor an explicit max_dets
    verbatim, even at or below 100: that's the exact value _max_dets_from_density's own floor
    legitimately derives for a sparse dataset, so silently substituting 1000 would clobber a real
    value."""
    import tcip_mcp.pipelines.training.eval_runners as runners
    from tcip_mcp.tools.training_tools import evaluate_model

    captured: dict = {}

    def _fake(ckpt, images_dir, labels_dir, output_dir, **kw):
        captured.update(kw)
        return {"eval_regime": "full-frame-tiled-inference"}

    monkeypatch.setattr(runners, "run_full_frame_evaluation", _fake)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    images_dir, labels_dir = _det_dataset(tmp_path)
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path, name="gating-max-dets-le-100")

    evaluate_model(str(ckpt), str(images_dir), str(labels_dir), task="detection", subject="bud",
                   use_tiled_inference=True, max_dets=50)
    assert captured["max_dets"] == 50  # honored verbatim, not bumped to 1000


def test_gating_path_defaults_max_dets_to_1000_when_unset(tmp_path, monkeypatch):
    import tcip_mcp.pipelines.training.eval_runners as runners
    from tcip_mcp.pipelines.resolution import DEFAULT_MAX_DETS
    from tcip_mcp.tools.training_tools import evaluate_model

    captured: dict = {}

    def _fake(ckpt, images_dir, labels_dir, output_dir, **kw):
        captured.update(kw)
        return {"eval_regime": "full-frame-tiled-inference"}

    monkeypatch.setattr(runners, "run_full_frame_evaluation", _fake)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    images_dir, labels_dir = _det_dataset(tmp_path)
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path, name="gating-max-dets-default")

    evaluate_model(str(ckpt), str(images_dir), str(labels_dir), task="detection", subject="bud",
                   use_tiled_inference=True)
    assert captured["max_dets"] == DEFAULT_MAX_DETS == 1000


def test_diagnostic_path_defaults_max_dets_to_100_when_unset(tmp_path, monkeypatch):
    """The COCOeval maxDets convention default for the other (tile-level/diagnostic) regime,
    distinct from the gating regime's 1000, resolved without the two colliding via a shared
    sentinel value."""
    import tcip_mcp.pipelines.training.eval_runners as runners
    from tcip_mcp.tools.training_tools import evaluate_model

    captured: dict = {}

    def _fake(ckpt, loader, device, task, output_dir, **kw):
        captured.update(kw)
        return {"tiled": False, "eval_regime": "tile-level"}

    monkeypatch.setattr(runners, "run_test_evaluation", _fake)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    images_dir, labels_dir = _det_dataset(tmp_path)
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path, name="diagnostic-max-dets-default")

    evaluate_model(str(ckpt), str(images_dir), str(labels_dir), task="detection", subject="bud")
    assert captured["max_dets"] == 100


def test_diagnostic_path_honors_explicit_max_dets(tmp_path, monkeypatch):
    import tcip_mcp.pipelines.training.eval_runners as runners
    from tcip_mcp.tools.training_tools import evaluate_model

    captured: dict = {}

    def _fake(ckpt, loader, device, task, output_dir, **kw):
        captured.update(kw)
        return {"tiled": False, "eval_regime": "tile-level"}

    monkeypatch.setattr(runners, "run_test_evaluation", _fake)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    images_dir, labels_dir = _det_dataset(tmp_path)
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path, name="diagnostic-max-dets-explicit")

    evaluate_model(str(ckpt), str(images_dir), str(labels_dir), task="detection", subject="bud",
                   max_dets=7)
    assert captured["max_dets"] == 7


def test_bare_checkpoint_path_reuses_its_own_stamped_tiling_and_subject(tmp_path, monkeypatch):
    """A checkpoint path (not a run id) carries its own stamped config["data"] the same way a run
    id's in-memory config does: evaluate_model must not silently lose tiling/subject reuse just
    because the caller passed a path instead of a run id."""
    import tcip_mcp.pipelines.training.eval_runners as runners
    from tcip_mcp.tools.training_tools import evaluate_model

    captured: dict = {}

    def _fake(ckpt, images_dir, labels_dir, output_dir, **kw):
        captured.update(kw)
        return {"eval_regime": "full-frame-tiled-inference"}

    monkeypatch.setattr(runners, "run_full_frame_evaluation", _fake)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    images_dir, labels_dir = _det_dataset(tmp_path)
    ckpt = tmp_path / "model.pt"
    torch.save({"config": {"data": {"tiling": {"tile_size": 384, "overlap": 0.15},
                                    "subject": "bud", "attribute": None}}}, ckpt)
    from tcip_mcp.tools.model_tools import register_model

    result = register_model(name="bare-ckpt-stamped-tiling", checkpoint_path=str(ckpt), config={},
                            project_path=str(tmp_path))
    assert "error" not in result, result

    evaluate_model(str(ckpt), str(images_dir), str(labels_dir), task="detection",
                   use_tiled_inference=True)
    # subject wasn't passed explicitly; it resolves from the checkpoint's own stamped config, the
    # same reuse a run id already gets, not silently None for a bare checkpoint path.
    assert captured["subject"] == "bud"
    assert captured["tile_size"] == 384
    assert captured["overlap"] == 0.15


def test_gate_translates_geometry_refusal_to_error_dict(tmp_path, monkeypatch):
    """evaluate_model is an @mcp.tool() surface that returns {"error": ...} for every other
    failure: a bare raise from run_full_frame_evaluation would surface as an MCP exception
    instead, inconsistent with the rest of this tool's contract."""
    import tcip_mcp.pipelines.training.eval_runners as runners
    from tcip_mcp.tools.training_tools import evaluate_model

    def _refuse(*a, **kw):
        raise ValueError("Cannot resolve a trustworthy tile_size for ckpt.pt: ... tiling=")

    monkeypatch.setattr(runners, "run_full_frame_evaluation", _refuse)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    images_dir, labels_dir = _det_dataset(tmp_path)
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path, name="gate-geometry-refusal")

    r = evaluate_model(str(ckpt), str(images_dir), str(labels_dir), task="detection",
                       subject="bud", use_tiled_inference=True)
    assert "error" in r
    assert "tiling=" in r["error"]


def test_gate_translates_unreadable_label_to_error_dict(tmp_path, monkeypatch):
    """A present, unreadable label document raised out of run_full_frame_evaluation is this
    tool's own {"error": ...} shape too, not a raise through the MCP boundary."""
    import tcip_mcp.pipelines.training.eval_runners as runners
    from tcip_annotation.json_io import UnreadableLabelDocument
    from tcip_mcp.tools.training_tools import evaluate_model

    def _refuse(*a, **kw):
        raise UnreadableLabelDocument("labels/2026-03-02/IMG_0001.json does not decode as JSON")

    monkeypatch.setattr(runners, "run_full_frame_evaluation", _refuse)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    images_dir, labels_dir = _det_dataset(tmp_path)
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path, name="gate-unreadable-label")

    r = evaluate_model(str(ckpt), str(images_dir), str(labels_dir), task="detection",
                       subject="bud", use_tiled_inference=True)
    assert "error" in r
    assert "IMG_0001.json" in r["error"]


def test_cap_hit_stamped_when_explicit_max_dets_truncates(tmp_path):
    """Honoring an explicit low max_dets verbatim reopens a truncation hole unless it's at least
    detectable. A caller-explicit cap that actually binds on real detections must be visible in
    the result, not silently assumed safe."""
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tcip_mcp.pipelines.training.eval_runners import run_full_frame_evaluation

    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (200, 200)).save(images_dir / "a.png")
    json_io.write_annotations(str(labels_dir / "a.json"),
                              [Annotation(subject="bud", geometry=BBox(10, 10, 30, 30))], 200, 200)

    class _ManyDetectionsStub:
        train_tile_size = 100
        train_overlap = 0.2

        def predict_tiled(self, path, **kw):
            # 5 detections returned; max_dets below will cap the caller intentionally at 2.
            # cap_hit=True: what the real predict_tiled would stamp here, now read directly.
            boxes = [[10, 10, 30, 30], [50, 50, 70, 70], [90, 90, 110, 110],
                     [130, 130, 150, 150], [170, 170, 190, 190]]
            return {"image": path, "width": 200, "height": 200, "boxes": boxes,
                    "scores": [0.9, 0.8, 0.7, 0.6, 0.5], "labels": [1] * 5, "count": 5,
                    "cap_hit": True}

    from tests._verified_checkpoint_fixtures import stub_verified_checkpoint

    checkpoint = stub_verified_checkpoint("ckpt.pt")
    build_predictor_orig = predictor_mod.build_predictor
    try:
        predictor_mod.build_predictor = lambda *a, **kw: _ManyDetectionsStub()
        r = run_full_frame_evaluation(checkpoint, str(images_dir), str(labels_dir),
                                      str(tmp_path / "out"), subject="bud", max_dets=2)
    finally:
        predictor_mod.build_predictor = build_predictor_orig
    assert r["max_dets"] == 2  # honored verbatim
    assert r["max_dets_cap_saturated_frac"] == 1.0  # the one image hit the cap, now visible


# ══════════════════════════════════════════════════════════════════════════
# tiled provenance distinguishes explicit from default
# ══════════════════════════════════════════════════════════════════════════

def test_raw_operating_point_tiled_source_explicit_vs_default():
    from tcip_mcp.pipelines.resolution import raw_operating_point

    explicit_bundle = raw_operating_point(
        conf=0.5, cross_tile_nms=0.3, tiled=True, tile_size=640, max_dets=100,
        tiled_source="explicit",
    )
    assert explicit_bundle.get("tiled").source == "explicit"

    default_bundle = raw_operating_point(
        conf=0.5, cross_tile_nms=0.3, tiled=True, tile_size=640, max_dets=100,
    )
    assert default_bundle.get("tiled").source == "default"


def test_resolve_operating_point_tile_size_source_not_inferred_from_truthiness():
    """`tile_size`'s source is never inferred from truthiness: a truthy value alone, even a
    fabricated fallback the caller never actually derived, must not be stamped "derived". The
    caller's own resolved source travels through explicitly."""
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    # A truthy tile_size with no source claim defaults to "default", not silently "derived".
    b_default = resolve_operating_point("bud_opening", tiled=True, dataset_hash=None, tile_size=640)
    assert b_default.get("tile_size").source == "default"

    b_derived = resolve_operating_point(
        "bud_opening", tiled=True, dataset_hash=None, tile_size=224, tile_size_source="derived")
    assert b_derived.get("tile_size").source == "derived"
    assert b_derived.get("tile_size")._raw == 224

    b_explicit = resolve_operating_point(
        "bud_opening", tiled=True, dataset_hash=None, tile_size=512, tile_size_source="explicit",
        tile_size_derived_from="stated on a checkpoint that records no tile geometry")
    assert b_explicit.get("tile_size").source == "explicit"


def test_resolve_operating_point_tiled_source_explicit_vs_default():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    b_default = resolve_operating_point("bud_opening", dataset_hash=None, tiled=True)
    assert b_default.get("tiled").source == "default"

    b_explicit = resolve_operating_point(
        "bud_opening", dataset_hash=None, tiled=False, tiled_source="explicit")
    assert b_explicit.get("tiled").source == "explicit"
    assert b_explicit.get("tiled")._raw is False
