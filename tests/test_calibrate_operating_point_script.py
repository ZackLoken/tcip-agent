"""``scripts/calibrate_operating_point.py``'s calibration-pass detection cap and conf-floor threading.

The script's in-model detection cap must match the already-safe MCP path (``run_inference`` ->
``calibrate_operating_point``, which passes ``max_dets=DEFAULT_MAX_DETS`` to ``build_predictor``
before the floor call) exactly: same constant, both entry doors, not two independently-typed caps
that could silently drift apart, since flooring conf to 0.01 for the sweep without also raising the
cap would truncate a dense calibration image's low-conf tail before the sweep ever saw it.

The script also threads the floor ``set_detector_operating_point`` actually applied into
``resolve_operating_point`` as ``staged_conf_floor``, not a re-typed ``0.01`` literal.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")


def _stub_checkpoint_load(monkeypatch) -> None:
    """These tests drive a stubbed predictor over a checkpoint path ("x.pt") that never exists
    on disk; load_registered_checkpoint is stubbed so the verified-load rail never reads it."""
    import tcip_mcp.model_registry as model_registry_mod

    from tests._verified_checkpoint_fixtures import stub_verified_checkpoint

    monkeypatch.setattr(model_registry_mod, "load_registered_checkpoint",
                        lambda path, *a, **kw: stub_verified_checkpoint(str(path)))


def test_script_and_mcp_path_share_the_same_cap_constant(monkeypatch, tmp_path):
    """Behavioral parity, not source-text matching: substring-matching the script's source
    (``'max_dets=DEFAULT_MAX_DETS' in script_src``) would stay green on a cosmetic rename of an
    unrelated variable that happened to share the substring, and would not catch two entry doors
    that independently compute the same numeric value through different code (the exact
    divergent-defaults bug this module exists to prevent). This exercises both real
    code paths: the script's ``main()`` (via the same monkeypatch approach
    ``test_script_threads_applied_floor_and_shared_cap`` above uses) and the MCP door's
    ``run_inference`` (via the same ``build_predictor`` monkeypatch
    ``test_detection_measurement_integrity.py``'s calibrated-operating-point tests use), asserting the ``max_dets`` each one actually
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

    def _build_predictor(checkpoint=None, *, max_dets=None, **kw):
        max_dets_calls.append(max_dets)
        return _Predictor()

    # One patch target for both doors: script and inference_tools each do
    # ``from tcip_mcp.pipelines.inference.predictor import build_predictor`` (a lazy import inside
    # the function body), so patching the defining module's attribute intercepts both calls.
    monkeypatch.setattr("tcip_mcp.pipelines.inference.predictor.build_predictor", _build_predictor)
    _stub_checkpoint_load(monkeypatch)

    # ---- script path ----
    class _Probe:
        stems = ["a", "b"]

    monkeypatch.setattr("tcip_mcp.pipelines.data.datasets.build_dataset", lambda *a, **kw: _Probe())
    monkeypatch.setattr("tcip_mcp.pipelines.data.splits.count_label_lines", lambda labels_dir, s, **kw: 1)
    monkeypatch.setattr("tcip_mcp.pipelines.data.splits.resolve_locked_cal_holdout_split",
                        lambda stems, **kw: {"calibration": ["a"], "holdout": ["b"]})
    monkeypatch.setattr("torch.utils.data.DataLoader", lambda ds, **kw: ds)
    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.records_over_loader",
                        lambda model, loader, device, task: [])

    def _resolve_op(trait_name, **kw):
        from tcip_mcp.pipelines.resolution import ResolvedBundle, derived
        conf = derived("conf", 0.4, requires_validation=True, validation_kind="annotations", derived_from="x",
                       validated_against="false", gate_evidence={})
        return ResolvedBundle(trait=trait_name, dataset_hash=kw.get("dataset_hash"), params={"conf": conf})

    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.resolve_operating_point", _resolve_op)
    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.attach_split_policy_provenance",
                        lambda b, locked: None)
    monkeypatch.setattr("tcip_mcp.project_paths.platform_state_root", lambda: tmp_path)

    from scripts.calibrate_operating_point import main

    rc = main(["--checkpoint", "x.pt", "--trait", "bud_opening",
              "--labels-dir", str(tmp_path / "labels"), "--images-dir", str(tmp_path / "images"),
              "--dataset-root", str(tmp_path), "--project-root", str(tmp_path)])
    assert rc == 0

    # ---- MCP path ----
    from PIL import Image

    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    img = tmp_path / "a.png"
    Image.new("RGB", (100, 100)).save(img)
    run_inference(str(ckpt), image_paths=[str(img)], device="cpu", tile=False)

    assert len(max_dets_calls) == 2
    script_cap, mcp_cap = max_dets_calls
    assert script_cap == mcp_cap == DEFAULT_MAX_DETS  # the same effective cap, not two literals


def test_script_threads_applied_floor_and_shared_cap(monkeypatch, tmp_path):
    from tcip_mcp.pipelines.resolution import DEFAULT_MAX_DETS

    calls: dict = {}

    class _Predictor:
        def __init__(self):
            self.model = SimpleNamespace(detector=SimpleNamespace(
                roi_heads=SimpleNamespace(score_thresh=0.5, nms_thresh=0.5, detections_per_img=100)))
            self.device = "cpu"
            self.train_tile_size = None

    def _build_predictor(checkpoint=None, *, device, max_dets=None, **kw):
        calls["build_predictor_max_dets"] = max_dets
        return _Predictor()

    monkeypatch.setattr("tcip_mcp.pipelines.inference.predictor.build_predictor", _build_predictor)
    _stub_checkpoint_load(monkeypatch)

    class _Probe:
        stems = ["a", "b"]

    monkeypatch.setattr("tcip_mcp.pipelines.data.datasets.build_dataset", lambda *a, **kw: _Probe())
    monkeypatch.setattr("tcip_mcp.pipelines.data.splits.count_label_lines", lambda labels_dir, s, **kw: 1)

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
                       validated_against="false", gate_evidence={})
        return ResolvedBundle(trait=trait_name, dataset_hash=kw.get("dataset_hash"), params={"conf": conf})

    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.resolve_operating_point", _resolve_op)
    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.attach_split_policy_provenance",
                        lambda b, locked: None)
    monkeypatch.setattr("tcip_mcp.project_paths.platform_state_root", lambda: tmp_path)

    from scripts.calibrate_operating_point import main

    rc = main(["--checkpoint", "x.pt", "--trait", "bud_opening",
              "--labels-dir", str(tmp_path / "labels"), "--images-dir", str(tmp_path / "images"),
              "--dataset-root", str(tmp_path), "--project-root", str(tmp_path)])
    assert rc == 0
    # The script's build_predictor call carries the shared cap constant, not the framework
    # default (100/300) that would otherwise truncate the 0.01-floored calibration pass.
    assert calls["build_predictor_max_dets"] == DEFAULT_MAX_DETS
    # The applied score_thresh (0.01) is threaded through as staged_conf_floor, not a bare
    # literal re-typed a third time.
    assert calls["resolve_operating_point_kwargs"]["staged_conf_floor"] == pytest.approx(0.01)
    # This script's own pass (_records) is always untiled (a plain DataLoader, never
    # predict_tiled); tiled=False must be stated explicitly, or resolve_operating_point's
    # tiled=True default would wrongly gate (or falsely validate) a tile_size dimension that was
    # never actually operative for this untiled calibration pass.
    assert calls["resolve_operating_point_kwargs"]["tiled"] is False


def test_script_collection_cap_is_density_derived_not_the_flat_default(monkeypatch, tmp_path):
    """The cap that actually governs the collection pass is
    set_detector_operating_point's detections_per_img call, which executes after build_predictor's
    construction-time DEFAULT_MAX_DETS and overrides it. This split's labels are sparse (2 objects
    per stem) so the density-derived cap floors at 100, well below DEFAULT_MAX_DETS (1000), proving
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
                        lambda checkpoint=None, *, device, max_dets=None, **kw: _Predictor())
    _stub_checkpoint_load(monkeypatch)

    class _Probe:
        stems = ["a", "b"]

    monkeypatch.setattr("tcip_mcp.pipelines.data.datasets.build_dataset", lambda *a, **kw: _Probe())
    # Sparse split: 2 objects/stem -> derive_max_dets_from_counts floors at 100, well under
    # DEFAULT_MAX_DETS (1000), a real, visible difference from the flat constant.
    monkeypatch.setattr("tcip_mcp.pipelines.data.splits.count_label_lines", lambda labels_dir, s, **kw: 2)
    monkeypatch.setattr("tcip_mcp.pipelines.data.splits.resolve_locked_cal_holdout_split",
                        lambda stems, **kw: {"calibration": ["a"], "holdout": ["b"]})
    monkeypatch.setattr("torch.utils.data.DataLoader", lambda ds, **kw: ds)
    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.records_over_loader",
                        lambda model, loader, device, task: [])

    def _resolve_op(trait_name, **kw):
        from tcip_mcp.pipelines.resolution import ResolvedBundle, derived
        conf = derived("conf", 0.4, requires_validation=True, validation_kind="annotations",
                       derived_from="x", validated_against="false", gate_evidence={})
        return ResolvedBundle(trait=trait_name, dataset_hash=kw.get("dataset_hash"), params={"conf": conf})

    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.resolve_operating_point", _resolve_op)
    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.attach_split_policy_provenance",
                        lambda b, locked: None)
    monkeypatch.setattr("tcip_mcp.project_paths.platform_state_root", lambda: tmp_path)

    from scripts.calibrate_operating_point import main

    rc = main(["--checkpoint", "x.pt", "--trait", "bud_opening",
              "--labels-dir", str(tmp_path / "labels"), "--images-dir", str(tmp_path / "images"),
              "--dataset-root", str(tmp_path), "--project-root", str(tmp_path)])
    assert rc == 0
    applied_cap = _Model.detector.roi_heads.detections_per_img
    assert applied_cap == 100  # derive_max_dets_from_counts([2, 2]) floor
    assert applied_cap != DEFAULT_MAX_DETS

def test_script_writes_nothing_into_the_experiment_record(monkeypatch, tmp_path, capsys):
    """The script inspects a sweep; minting or replacing a validated claim belongs to the audited
    doors, so a run leaves the experiment directory exactly as it found it."""

    class _Predictor:
        def __init__(self):
            self.model = SimpleNamespace(detector=SimpleNamespace(
                roi_heads=SimpleNamespace(score_thresh=0.5, nms_thresh=0.5, detections_per_img=100)))
            self.device = "cpu"
            self.train_tile_size = None
            self.train_overlap = None
            self.score_threshold = 0.5

    monkeypatch.setattr("tcip_mcp.pipelines.inference.predictor.build_predictor",
                        lambda *a, **kw: _Predictor())
    _stub_checkpoint_load(monkeypatch)

    class _Probe:
        stems = ["a", "b"]

    monkeypatch.setattr("tcip_mcp.pipelines.data.datasets.build_dataset", lambda *a, **kw: _Probe())
    monkeypatch.setattr("tcip_mcp.pipelines.data.splits.count_label_lines", lambda labels_dir, s, **kw: 1)
    monkeypatch.setattr("tcip_mcp.pipelines.data.splits.resolve_locked_cal_holdout_split",
                        lambda stems, **kw: {"calibration": ["a"], "holdout": ["b"]})
    monkeypatch.setattr("torch.utils.data.DataLoader", lambda ds, **kw: ds)
    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.records_over_loader",
                        lambda model, loader, device, task: [])

    def _resolve_op(trait_name, **kw):
        from tcip_mcp.pipelines.resolution import ResolvedBundle, derived
        conf = derived("conf", 0.4, requires_validation=True, validation_kind="annotations",
                       derived_from="x", validated_against="false", gate_evidence={})
        return ResolvedBundle(trait=trait_name, dataset_hash=kw.get("dataset_hash"),
                              params={"conf": conf})

    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.resolve_operating_point", _resolve_op)
    monkeypatch.setattr("tcip_mcp.pipelines.operating_point.attach_split_policy_provenance",
                        lambda b, locked: None)
    monkeypatch.setattr("tcip_mcp.project_paths.platform_state_root", lambda: tmp_path)

    from scripts.calibrate_operating_point import main

    rc = main(["--checkpoint", "x.pt", "--trait", "bud_opening",
              "--labels-dir", str(tmp_path / "labels"), "--images-dir", str(tmp_path / "images"),
              "--dataset-root", str(tmp_path), "--project-root", str(tmp_path)])
    assert rc == 0
    assert not (tmp_path / ".tcip" / "experiments").exists()
    out = capsys.readouterr().out
    assert '"conf"' in out

def test_script_refuses_an_agent_authored_reference_before_touching_a_model(monkeypatch, tmp_path):
    """The labels dir is this script's measurement reference, so the admissibility rail fires
    before any model or dataset work begins."""
    import json as _json

    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.json").write_text(_json.dumps({
        "image": "a", "width": 100, "height": 100,
        "annotations": [{"subject": "bud", "bbox": [1, 1, 5, 5], "created_by": "claude"}],
    }), encoding="utf-8")

    def _never(*a, **kw):
        raise AssertionError("the rail must refuse before the predictor is built")

    monkeypatch.setattr("tcip_mcp.pipelines.inference.predictor.build_predictor", _never)

    from scripts.calibrate_operating_point import main

    with pytest.raises(ValueError) as refused:
        main(["--checkpoint", "x.pt", "--trait", "bud_opening",
              "--labels-dir", str(labels), "--images-dir", str(tmp_path / "images"),
              "--dataset-root", str(tmp_path), "--project-root", str(tmp_path)])
    assert "created_by" in str(refused.value) or "claude" in str(refused.value)


def test_script_prints_and_exits_cleanly_for_fewer_than_two_labeled_stems(monkeypatch, tmp_path, capsys):
    """A labeled split too small to divide into cal/holdout is a usage refusal, not a traceback:
    the library's CalibrationUsageError becomes a printed message and rc=2."""
    _stub_checkpoint_load(monkeypatch)

    class _Predictor:
        def __init__(self):
            self.model = SimpleNamespace(detector=SimpleNamespace(
                roi_heads=SimpleNamespace(score_thresh=0.5, nms_thresh=0.5, detections_per_img=100)))
            self.device = "cpu"
            self.train_tile_size = None

    monkeypatch.setattr("tcip_mcp.pipelines.inference.predictor.build_predictor",
                        lambda checkpoint=None, **kw: _Predictor())

    class _Probe:
        stems = ["only_one"]

    monkeypatch.setattr("tcip_mcp.pipelines.data.datasets.build_dataset", lambda *a, **kw: _Probe())
    monkeypatch.setattr("tcip_mcp.project_paths.platform_state_root", lambda: tmp_path)

    from scripts.calibrate_operating_point import main

    rc = main(["--checkpoint", "x.pt", "--trait", "bud_opening",
              "--labels-dir", str(tmp_path / "labels"), "--images-dir", str(tmp_path / "images"),
              "--dataset-root", str(tmp_path), "--project-root", str(tmp_path)])

    assert rc == 2
    assert "Need >=2 labeled stems" in capsys.readouterr().err


def test_script_split_manifest_dir_requires_subject(tmp_path):
    """--split-manifest-dir needs --subject to check the manifest's own subject against; this
    refuses before touching a checkpoint or a dataset."""
    from scripts.calibrate_operating_point import main

    rc = main(["--checkpoint", "x.pt", "--trait", "bud_opening",
              "--labels-dir", str(tmp_path / "labels"), "--images-dir", str(tmp_path / "images"),
              "--dataset-root", str(tmp_path), "--project-root", str(tmp_path),
              "--split-manifest-dir", str(tmp_path / "m")])

    assert rc == 2


def test_script_split_manifest_dir_conflicts_with_group_by(tmp_path):
    """A drawn split's own parameter beside a recorded partition is a conflict, not a silent
    choice between the two."""
    from scripts.calibrate_operating_point import main

    rc = main(["--checkpoint", "x.pt", "--trait", "bud_opening",
              "--labels-dir", str(tmp_path / "labels"), "--images-dir", str(tmp_path / "images"),
              "--dataset-root", str(tmp_path), "--project-root", str(tmp_path),
              "--split-manifest-dir", str(tmp_path / "m"),
              "--subject", "bud", "--group-by", "stem"])

    assert rc == 2


def test_script_runs_end_to_end_with_a_checkpoint_registered_under_project_root(
    tmp_path, seed_bud_trait_spec,
):
    """The admitting half of the registry rail: no stub anywhere on the checkpoint's own path,
    a real load_registered_checkpoint against --project-root, and the script completes."""
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path)

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    for i in range(3):
        Image.new("RGB", (64, 64), (100, 100, 100)).save(images_dir / f"img{i}.png")
        json_io.write_annotations(
            str(labels_dir / f"img{i}.json"),
            [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], 64, 64)

    from scripts.calibrate_operating_point import main

    rc = main([
        "--checkpoint", ckpt, "--trait", "bud_opening", "--subject", "bud",
        "--labels-dir", str(labels_dir), "--images-dir", str(images_dir),
        "--dataset-root", str(tmp_path), "--project-root", str(tmp_path),
    ])
    assert rc == 0
