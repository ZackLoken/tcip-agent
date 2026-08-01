"""Fix J (calibration-pass detection cap) + Fix D (conf-floor threading) in
``scripts/calibrate_operating_point.py``.

Fix J: the script previously left the in-model detection cap at torchvision's raw default while
flooring conf to 0.01 for the sweep — a dense calibration image's low-conf tail could be truncated
before the sweep ever saw it. It now matches the already-safe MCP path (``run_inference`` ->
``_calibrate_operating_point``, which passes ``max_dets=DEFAULT_MAX_DETS`` to ``build_predictor``
before the floor call) exactly: same constant, both entry doors, not two independently-typed caps
that could silently drift apart.

Fix D: the script threads the floor ``set_detector_operating_point`` actually applied into
``resolve_operating_point`` as ``staged_conf_floor``, not a re-typed ``0.01`` literal.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")


def test_script_and_mcp_path_share_the_same_cap_constant(monkeypatch, tmp_path):
    """Behavioral parity, not source-text matching (stage-6 review finding): the original version of
    this test did ``inspect.getsource`` substring matching (``'max_dets=DEFAULT_MAX_DETS' in
    script_src``, etc.) — it would stay green on a cosmetic rename of an unrelated variable that
    happened to share the substring, and would NOT catch two entry doors that independently compute
    the SAME numeric value through different code (the exact divergent-defaults bug this module's own
    docstring says the whole ``operating_point`` module exists to prevent). This exercises BOTH real
    code paths — the script's ``main()`` (via the same monkeypatch approach
    ``test_script_threads_applied_floor_and_shared_cap`` above uses) and the MCP door's
    ``run_inference`` (via the same ``build_predictor`` monkeypatch
    ``test_audit_cv_fixes.py``'s CV0 tests use) — and asserts the ``max_dets`` each one ACTUALLY
    passed to ``build_predictor`` is the identical value, at runtime.
    """
    from tcip_mcp.pipelines.resolution import DEFAULT_MAX_DETS

    max_dets_calls: list = []

    class _Predictor:
        def __init__(self):
            self.model = SimpleNamespace(detector=SimpleNamespace(
                roi_heads=SimpleNamespace(score_thresh=0.5, nms_thresh=0.5, detections_per_img=100)))
            self.device = "cpu"
            self.train_tile_size = None
            self.train_overlap = None
            self.score_threshold = 0.5

        def predict_batch(self, paths, **kw):
            return [{"image": p, "width": 100, "height": 100,
                    "boxes": [], "scores": [], "labels": [], "count": 0} for p in paths]

    def _build_predictor(*, max_dets=None, **kw):
        max_dets_calls.append(max_dets)
        return _Predictor()

    # ONE patch target for BOTH doors — script and inference_tools each do
    # ``from tcip_mcp.pipelines.inference.predictor import build_predictor`` (a lazy import inside
    # the function body), so patching the defining module's attribute intercepts both calls.
    monkeypatch.setattr("tcip_mcp.pipelines.inference.predictor.build_predictor", _build_predictor)

    # ---- script path ----
    class _Probe:
        stems = ["a", "b"]

    monkeypatch.setattr("tcip_mcp.pipelines.data.datasets.build_dataset", lambda *a, **kw: _Probe())
    monkeypatch.setattr("tcip_mcp.pipelines.data.splits.count_label_lines", lambda labels_dir, s: 1)
    monkeypatch.setattr("tcip_mcp.pipelines.data.splits.resolve_locked_cal_holdout_split",
                        lambda stems, **kw: {"calibration": ["a"], "holdout": ["b"]})
    monkeypatch.setattr("torch.utils.data.DataLoader", lambda ds, **kw: ds)
    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.records_over_loader",
                        lambda model, loader, device, task: [])

    def _resolve_op(trait_name, **kw):
        from tcip_mcp.pipelines.resolution import ResolvedBundle, derived
        conf = derived("conf", 0.4, requires_validation=True, validation_kind="annotations", derived_from="x",
                       validated_against="false", sweep={})
        return ResolvedBundle(trait=trait_name, dataset_hash=kw.get("dataset_hash"), params={"conf": conf})

    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.resolve_operating_point", _resolve_op)
    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.attach_split_policy_provenance",
                        lambda b, locked: None)
    monkeypatch.setattr("tcip_mcp.project_paths.project_root", lambda: tmp_path)

    from scripts.calibrate_operating_point import main

    rc = main(["--checkpoint", "x.pt", "--trait", "catkin",
              "--labels-dir", str(tmp_path / "labels"), "--images-dir", str(tmp_path / "images")])
    assert rc == 0

    # ---- MCP path ----
    from PIL import Image

    from tcip_mcp.tools.inference_tools import run_inference

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    img = tmp_path / "a.png"
    Image.new("RGB", (100, 100)).save(img)
    run_inference(str(ckpt), image_paths=[str(img)], device="cpu", tile=False)

    assert len(max_dets_calls) == 2
    script_cap, mcp_cap = max_dets_calls
    assert script_cap == mcp_cap == DEFAULT_MAX_DETS  # the SAME effective cap, not two literals


def test_script_threads_applied_floor_and_shared_cap(monkeypatch, tmp_path):
    from tcip_mcp.pipelines.resolution import DEFAULT_MAX_DETS

    calls: dict = {}

    class _Predictor:
        def __init__(self):
            self.model = SimpleNamespace(detector=SimpleNamespace(
                roi_heads=SimpleNamespace(score_thresh=0.5, nms_thresh=0.5, detections_per_img=100)))
            self.device = "cpu"
            self.train_tile_size = None

    def _build_predictor(*, checkpoint_path, device, max_dets=None, **kw):
        calls["build_predictor_max_dets"] = max_dets
        return _Predictor()

    monkeypatch.setattr("tcip_mcp.pipelines.inference.predictor.build_predictor", _build_predictor)

    class _Probe:
        stems = ["a", "b"]

    monkeypatch.setattr("tcip_mcp.pipelines.data.datasets.build_dataset", lambda *a, **kw: _Probe())
    monkeypatch.setattr("tcip_mcp.pipelines.data.splits.count_label_lines", lambda labels_dir, s: 1)

    def _resolve_locked(stems, **kw):
        return {"calibration": ["a"], "holdout": ["b"]}

    monkeypatch.setattr("tcip_mcp.pipelines.data.splits.resolve_locked_cal_holdout_split", _resolve_locked)
    monkeypatch.setattr("torch.utils.data.DataLoader", lambda ds, **kw: ds)
    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.records_over_loader",
                        lambda model, loader, device, task: [])

    def _resolve_op(trait_name, **kw):
        calls["resolve_operating_point_kwargs"] = kw
        from tcip_mcp.pipelines.resolution import ResolvedBundle, derived
        conf = derived("conf", 0.4, requires_validation=True, validation_kind="annotations", derived_from="x",
                       validated_against="false", sweep={})
        return ResolvedBundle(trait=trait_name, dataset_hash=kw.get("dataset_hash"), params={"conf": conf})

    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.resolve_operating_point", _resolve_op)
    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.attach_split_policy_provenance",
                        lambda b, locked: None)
    monkeypatch.setattr("tcip_mcp.project_paths.project_root", lambda: tmp_path)

    from scripts.calibrate_operating_point import main

    rc = main(["--checkpoint", "x.pt", "--trait", "catkin",
              "--labels-dir", str(tmp_path / "labels"), "--images-dir", str(tmp_path / "images")])
    assert rc == 0
    # Fix J: the script's build_predictor call carries the shared cap constant, not the framework
    # default (100/300) that would otherwise truncate the 0.01-floored calibration pass.
    assert calls["build_predictor_max_dets"] == DEFAULT_MAX_DETS
    # Fix D: the applied score_thresh (0.01) is threaded through as staged_conf_floor, not a bare
    # literal re-typed a third time.
    assert calls["resolve_operating_point_kwargs"]["staged_conf_floor"] == pytest.approx(0.01)
    # K10: this script's own pass (_records) is always untiled (a plain DataLoader, never
    # predict_tiled) — tiled=False must be stated explicitly, or resolve_operating_point's
    # tiled=True default would wrongly gate (or falsely validate) a tile_size dimension that was
    # never actually operative for this untiled calibration pass.
    assert calls["resolve_operating_point_kwargs"]["tiled"] is False


def test_script_collection_cap_is_density_derived_not_the_flat_default(monkeypatch, tmp_path):
    """K7 residual (detector-cap censoring): the cap that actually governs the collection pass is
    set_detector_operating_point's detections_per_img call, which executes AFTER build_predictor's
    construction-time DEFAULT_MAX_DETS and overrides it. This split's labels are sparse (2 objects
    per stem) so the density-derived cap floors at 100 — well below DEFAULT_MAX_DETS (1000) — proving
    the collection pass is no longer capped at the flat constant."""
    from tcip_mcp.pipelines.resolution import DEFAULT_MAX_DETS

    class _Model:
        detector = SimpleNamespace(roi_heads=SimpleNamespace(
            score_thresh=0.5, nms_thresh=0.5, detections_per_img=DEFAULT_MAX_DETS))

    class _Predictor:
        def __init__(self):
            self.model = _Model()
            self.device = "cpu"
            self.train_tile_size = None

    monkeypatch.setattr("tcip_mcp.pipelines.inference.predictor.build_predictor",
                        lambda *, checkpoint_path, device, max_dets=None, **kw: _Predictor())

    class _Probe:
        stems = ["a", "b"]

    monkeypatch.setattr("tcip_mcp.pipelines.data.datasets.build_dataset", lambda *a, **kw: _Probe())
    # Sparse split: 2 objects/stem -> derive_max_dets_from_counts floors at 100, well under
    # DEFAULT_MAX_DETS (1000) — a real, visible difference from the flat constant.
    monkeypatch.setattr("tcip_mcp.pipelines.data.splits.count_label_lines", lambda labels_dir, s: 2)
    monkeypatch.setattr("tcip_mcp.pipelines.data.splits.resolve_locked_cal_holdout_split",
                        lambda stems, **kw: {"calibration": ["a"], "holdout": ["b"]})
    monkeypatch.setattr("torch.utils.data.DataLoader", lambda ds, **kw: ds)
    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.records_over_loader",
                        lambda model, loader, device, task: [])

    def _resolve_op(trait_name, **kw):
        from tcip_mcp.pipelines.resolution import ResolvedBundle, derived
        conf = derived("conf", 0.4, requires_validation=True, validation_kind="annotations",
                       derived_from="x", validated_against="false", sweep={})
        return ResolvedBundle(trait=trait_name, dataset_hash=kw.get("dataset_hash"), params={"conf": conf})

    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.resolve_operating_point", _resolve_op)
    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.attach_split_policy_provenance",
                        lambda b, locked: None)
    monkeypatch.setattr("tcip_mcp.project_paths.project_root", lambda: tmp_path)

    from scripts.calibrate_operating_point import main

    rc = main(["--checkpoint", "x.pt", "--trait", "catkin",
              "--labels-dir", str(tmp_path / "labels"), "--images-dir", str(tmp_path / "images")])
    assert rc == 0
    applied_cap = _Model.detector.roi_heads.detections_per_img
    assert applied_cap == 100  # derive_max_dets_from_counts([2, 2]) floor
    assert applied_cap != DEFAULT_MAX_DETS
