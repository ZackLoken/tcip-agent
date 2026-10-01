"""Delivery-grade evaluation runs in a different regime than inference: it resolves tile geometry
via the same shared ``resolve_tile_geometry`` ``run_inference`` uses (refusing rather than
scoring at an ungrounded scale when nothing is resolvable), honors ``max_dets`` verbatim on both
regimes with a per-image ``cap_hit``/``max_dets_cap_saturated_frac`` signal on the gating path, and
runs under the execution record ``prepare_pass`` resolves, the one ``run_inference`` runs under.
See ``test_detection_measurement_integrity.py`` for the geometry-resolution tests; this file covers
the ``evaluate_model`` wrapper's passthrough + refusal handling and the runner's own recorded
execution record.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from tcip_mcp.pipelines.execution import Stated  # noqa: E402

# seed_bud_trait_spec (conftest.py) confirms bud_opening in this test's project, so the
# trait/subject="bud" call sites resolve.
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

    def _fake(ckpt, images_dir, labels_dir, **kw):
        captured.update(kw)
        return {"eval_regime": "full-frame-tiled-inference"}

    monkeypatch.setattr(runners, "run_full_frame_evaluation", _fake)
    images_dir, labels_dir = _det_dataset(tmp_path)
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path)

    evaluate_model(tmp_path, str(ckpt), str(images_dir), str(labels_dir),
                   use_tiled_inference=True, stated=Stated(max_dets=50))
    assert captured["stated"].max_dets == 50  # honored verbatim, not bumped to 1000


def test_gating_path_defaults_max_dets_to_1000_when_unset(tmp_path, monkeypatch):
    """The door's own pass-through: an unstated max_dets reaches run_full_frame_evaluation as
    None, and the runner itself, not the door, resolves it to the delivery-grade default. Proven
    on the runner's own result rather than a fake's captured kwarg, since the door no longer
    resolves this value itself."""
    from tcip_mcp.pipelines.execution import DEFAULT_MAX_DETS
    from tcip_mcp.tools.training_tools import evaluate_model

    images_dir, labels_dir = _det_dataset(tmp_path)
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path)

    r = evaluate_model(tmp_path, str(ckpt), str(images_dir), str(labels_dir),
                       use_tiled_inference=True, stated=Stated(tile_size=128, overlap=0.0))
    assert "error" not in r, r
    assert r["execution"]["max_dets"] == DEFAULT_MAX_DETS == 1000
    assert r["execution"]["sources"]["max_dets"] == "default"


def test_diagnostic_path_hands_an_unset_cap_on_unstated(tmp_path, monkeypatch):
    """The diagnostic regime resolves no cap of its own: an unset one reaches the runner as the
    pass the one execution resolver prepared, defaulted and recorded as a default."""
    import tcip_mcp.pipelines.training.eval_runners as runners
    from tcip_mcp.pipelines.execution import DEFAULT_MAX_DETS
    from tcip_mcp.tools.training_tools import evaluate_model

    captured: dict = {}

    def _fake(pass_, loader, device, **kw):
        captured["execution"] = pass_.execution
        return {"eval_regime": "tile-level"}

    monkeypatch.setattr(runners, "run_test_evaluation", _fake)
    images_dir, labels_dir = _det_dataset(tmp_path)
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path)

    evaluate_model(tmp_path, str(ckpt), str(images_dir), str(labels_dir))
    assert captured["execution"].max_dets == DEFAULT_MAX_DETS
    assert captured["execution"].sources["max_dets"] == "default"


def test_diagnostic_path_honors_explicit_max_dets(tmp_path, monkeypatch):
    import tcip_mcp.pipelines.training.eval_runners as runners
    from tcip_mcp.tools.training_tools import evaluate_model

    captured: dict = {}

    def _fake(pass_, loader, device, **kw):
        captured["execution"] = pass_.execution
        return {"eval_regime": "tile-level"}

    monkeypatch.setattr(runners, "run_test_evaluation", _fake)
    images_dir, labels_dir = _det_dataset(tmp_path)
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path)

    evaluate_model(tmp_path, str(ckpt), str(images_dir), str(labels_dir),
                   stated=Stated(max_dets=7))
    assert captured["execution"].max_dets == 7
    assert captured["execution"].sources["max_dets"] == "explicit"


def test_bare_checkpoint_path_reuses_its_own_stamped_tiling_and_subject(tmp_path, monkeypatch):
    """A checkpoint path (not a run id) carries its own stamped config["data"]: evaluate_model
    reads its tiling and its class space from it, the one record either comes from."""
    import tcip_mcp.pipelines.training.eval_runners as runners
    from tcip_mcp.tools.training_tools import evaluate_model

    captured: dict = {}

    def _fake(ckpt, images_dir, labels_dir, **kw):
        captured.update(kw, checkpoint=ckpt)
        return {"eval_regime": "full-frame-tiled-inference"}

    monkeypatch.setattr(runners, "run_full_frame_evaluation", _fake)
    images_dir, labels_dir = _det_dataset(tmp_path)
    from tests._verified_checkpoint_fixtures import SCOPED_DATA, registered_checkpoint

    ckpt = registered_checkpoint(
        tmp_path, data={**SCOPED_DATA, "tiling": {"tile_size": 384, "overlap": 0.15,
                                                  "sliver_frac": 0.5}})

    evaluate_model(tmp_path, str(ckpt), str(images_dir), str(labels_dir),
                   use_tiled_inference=True)
    # The measurement is handed the checkpoint whose own record states its class space and its
    # tiling, and the door states no geometry of its own beside it.
    assert captured["checkpoint"].data_config["scope"]["subject"] == "bud"
    assert captured["stated"].tile_size is None and captured["stated"].overlap is None

    from tcip_mcp.pipelines.execution import prepare_pass

    record = prepare_pass(captured["checkpoint"], Stated(tile=True)).execution
    assert (record.tile_size, record.overlap) == (384, 0.15)


def test_gate_translates_geometry_refusal_to_error_dict(tmp_path, monkeypatch):
    """evaluate_model is an @mcp.tool() surface that returns {"error": ...} for every other
    failure: a bare raise from run_full_frame_evaluation would surface as an MCP exception
    instead, inconsistent with the rest of this tool's contract."""
    import tcip_mcp.pipelines.training.eval_runners as runners
    from tcip_mcp.tools.training_tools import evaluate_model

    def _refuse(*a, **kw):
        raise ValueError("Cannot resolve a trustworthy tile_size for ckpt.pt: ... tiling=")

    monkeypatch.setattr(runners, "run_full_frame_evaluation", _refuse)
    images_dir, labels_dir = _det_dataset(tmp_path)
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path)

    r = evaluate_model(tmp_path, str(ckpt), str(images_dir), str(labels_dir),
                       use_tiled_inference=True)
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
    images_dir, labels_dir = _det_dataset(tmp_path)
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path)

    r = evaluate_model(tmp_path, str(ckpt), str(images_dir), str(labels_dir),
                       use_tiled_inference=True)
    assert "error" in r
    assert "IMG_0001.json" in r["error"]


def test_cap_hit_stamped_when_explicit_max_dets_truncates(tmp_path):
    """Honoring an explicit low max_dets verbatim reopens a truncation hole unless it's at least
    detectable. A caller-explicit cap that actually binds on real detections must be visible in
    the result, not silently assumed safe."""
    import tcip_mcp.pipelines.inference.generic_predictor as predictor_mod
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
        task = "detection"
        train_tile_size = 100
        train_overlap = 0.2
        in_chans = 3

        def predict_sliced(self, path, **kw):
            # 5 detections returned; max_dets below will cap the caller intentionally at 2.
            # cap_hit=True: what the real predict_sliced would stamp here, now read directly.
            boxes = [[10, 10, 30, 30], [50, 50, 70, 70], [90, 90, 110, 110],
                     [130, 130, 150, 150], [170, 170, 190, 190]]
            return {"image": path, "width": 200, "height": 200, "boxes": boxes,
                    "scores": [0.9, 0.8, 0.7, 0.6, 0.5], "labels": [1] * 5, "count": 5,
                    "cap_hit": True}

    from tests._verified_checkpoint_fixtures import verified_checkpoint

    checkpoint = verified_checkpoint(tmp_path)
    build_predictor_orig = predictor_mod.GenericPredictor
    try:
        predictor_mod.GenericPredictor = lambda *a, **kw: _ManyDetectionsStub()
        r = run_full_frame_evaluation(checkpoint, str(images_dir), str(labels_dir),
                                      stated=Stated(max_dets=2))
    finally:
        predictor_mod.GenericPredictor = build_predictor_orig
    assert r["execution"]["max_dets"] == 2  # honored verbatim
    assert r["max_dets_cap_saturated_frac"] == 1.0  # the one image hit the cap, now visible


def test_the_gate_reads_its_references_at_the_predictors_own_width(tmp_path):
    """The delivery gate builds its loader at the width the predictor reads at, and never derives
    one: this gate scores source paths through the predictor and reads only targets off the
    loader, so references of differing band counts are a measurement it can take."""
    import numpy as np
    import tifffile
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    import tcip_mcp.pipelines.inference.generic_predictor as predictor_mod
    from tcip_mcp.pipelines.training.eval_runners import run_full_frame_evaluation
    from tests._verified_checkpoint_fixtures import verified_checkpoint

    images_dir, labels_dir = _det_dataset(tmp_path)  # three-band sources
    array = np.zeros((128, 128, 5), dtype=np.uint8)
    tifffile.imwrite(str(images_dir / "five_band.tif"), array)  # and one of five bands
    json_io.write_annotations(str(labels_dir / "five_band.json"),
                              [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], 128, 128)

    class _OneBandStub:
        task = "detection"
        train_tile_size = 100
        train_overlap = 0.2
        in_chans = 1

        def predict_sliced(self, path, **kw):
            return {"image": path, "width": 128, "height": 128, "boxes": [[10, 10, 40, 40]],
                    "scores": [0.9], "labels": [1], "count": 1, "cap_hit": False}

    build_predictor_orig = predictor_mod.GenericPredictor
    try:
        predictor_mod.GenericPredictor = lambda *a, **kw: _OneBandStub()
        r = run_full_frame_evaluation(verified_checkpoint(tmp_path), str(images_dir),
                                      str(labels_dir), stated=Stated())
    finally:
        predictor_mod.GenericPredictor = build_predictor_orig

    assert r["scored_images"] == 4
    assert r["tp"] == 4


def test_run_full_frame_evaluation_records_merge_and_execution(tmp_path):
    """The runner's record carries the execution record whose conf/max_dets/cross_tile_nms read
    source "explicit" when stated (a stated value equal to the default included) and "default"
    when not, and cross_tile_nms the merge threshold the pass ran at; a direct call stating
    max_dets=2 records 2 as explicit."""
    import tcip_mcp.pipelines.inference.generic_predictor as predictor_mod
    from tcip_mcp.pipelines.training.eval_runners import run_full_frame_evaluation

    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    class _EmptyStub:
        task = "detection"
        train_tile_size = 100
        train_overlap = 0.2
        in_chans = 3

        def predict_sliced(self, path, **kw):
            merges.append(kw["execution"].cross_tile_nms)
            return {"image": path, "width": 128, "height": 128, "boxes": [], "scores": [],
                    "labels": [], "cap_hit": False}

    merges: list[float] = []
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (128, 128)).save(images_dir / "a.png")
    json_io.write_annotations(str(labels_dir / "a.json"),
                              [Annotation(subject="bud", geometry=BBox(10, 10, 30, 30))], 128, 128)

    from tests._verified_checkpoint_fixtures import verified_checkpoint

    checkpoint = verified_checkpoint(tmp_path)
    build_predictor_orig = predictor_mod.GenericPredictor
    try:
        predictor_mod.GenericPredictor = lambda *a, **kw: _EmptyStub()
        r_default = run_full_frame_evaluation(checkpoint, str(images_dir), str(labels_dir),
                                              stated=Stated())
        r_stated = run_full_frame_evaluation(
            checkpoint, str(images_dir), str(labels_dir),
            stated=Stated(conf=0.5, cross_tile_nms=0.3, max_dets=2))
    finally:
        predictor_mod.GenericPredictor = build_predictor_orig

    assert merges == [0.3, 0.3]
    for r in (r_default, r_stated):
        assert r["execution"]["postprocess"] == "nms"
        assert r["execution"]["cross_tile_nms"] == 0.3

    for name in ("conf", "max_dets", "cross_tile_nms"):
        assert r_default["execution"]["sources"][name] == "default"
        # A stated value equal to the platform default is still recorded as explicit.
        assert r_stated["execution"]["sources"][name] == "explicit"
    assert r_stated["execution"]["max_dets"] == 2


def test_the_gate_refuses_documents_whose_geometry_a_detector_cannot_read(tmp_path):
    """The delivery gate measures over the loader a run would build, so ground truth carrying
    the subject only as points refuses in that loader's own words rather than scoring every
    image against an empty reference and reporting the number as a delivery metric."""
    import tcip_mcp.pipelines.inference.generic_predictor as predictor_mod
    from tcip_mcp.pipelines.training.eval_runners import run_full_frame_evaluation

    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox, Point

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    for index in range(8):
        Image.new("RGB", (128, 128)).save(images_dir / f"p{index}.png")
        json_io.write_annotations(
            str(labels_dir / f"p{index}.json"),
            [Annotation(subject="bud", geometry=Point(20.0, 30.0))], 128, 128)

    class _EmptyStub:
        task = "detection"
        train_tile_size = 100
        train_overlap = 0.2
        in_chans = 3

        def predict_sliced(self, path, **kw):
            return {"image": path, "width": 128, "height": 128, "boxes": [], "scores": [],
                    "labels": [], "cap_hit": False}

    from tests._verified_checkpoint_fixtures import verified_checkpoint

    checkpoint = verified_checkpoint(tmp_path)
    build_predictor_orig = predictor_mod.GenericPredictor
    try:
        predictor_mod.GenericPredictor = lambda *a, **kw: _EmptyStub()
        with pytest.raises(ValueError, match="only in geometries a detection loader"):
            run_full_frame_evaluation(checkpoint, str(images_dir), str(labels_dir),
                                      stated=Stated())

        # Admits valid work: the same eight images, their documents carrying boxes, score.
        for index in range(8):
            json_io.write_annotations(
                str(labels_dir / f"p{index}.json"),
                [Annotation(subject="bud", geometry=BBox(10, 10, 30, 30))], 128, 128)
        scored = run_full_frame_evaluation(checkpoint, str(images_dir), str(labels_dir),
                                           stated=Stated())
    finally:
        predictor_mod.GenericPredictor = build_predictor_orig
    assert scored["n_images"] == 8


def test_the_gate_refuses_an_images_tree_with_no_ground_truth(tmp_path):
    """A measurement is against a reference: with no label store there is nothing to score
    against, so the gate refuses by name rather than scoring every image against empty ground
    truth and reporting a perfect-looking miss rate."""
    import tcip_mcp.pipelines.inference.generic_predictor as predictor_mod
    from tcip_mcp.pipelines.training.eval_runners import run_full_frame_evaluation

    from PIL import Image

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (128, 128)).save(images_dir / "a.png")

    class _EmptyStub:
        task = "detection"
        train_tile_size = 100
        train_overlap = 0.2
        in_chans = 3

        def predict_sliced(self, path, **kw):
            return {"image": path, "width": 128, "height": 128, "boxes": [], "scores": [],
                    "labels": [], "cap_hit": False}

    from tests._verified_checkpoint_fixtures import verified_checkpoint

    checkpoint = verified_checkpoint(tmp_path)
    build_predictor_orig = predictor_mod.GenericPredictor
    try:
        predictor_mod.GenericPredictor = lambda *a, **kw: _EmptyStub()
        with pytest.raises(ValueError, match="neither a .csv table nor a directory"):
            run_full_frame_evaluation(checkpoint, str(images_dir), str(tmp_path / "labels"),
                                      stated=Stated())
    finally:
        predictor_mod.GenericPredictor = build_predictor_orig
