"""training_tools fixes — validator/StageSpec alignment, HPO param plumbing,
HPO trial reporting (per-epoch trace + failed-trial sentinels), HPO/train regime parity,
experiment immutability on relaunch, and canonical-format confidence parsing
in get_worst_predictions."""

from __future__ import annotations

import json

import pytest

# Round 10 (2026-07-29): no built-in traits — seed_catkin_trait_spec (conftest.py) writes a real
# catkin.yml into this test's pinned project root so trait="catkin" call sites keep resolving.
pytestmark = pytest.mark.usefixtures("seed_catkin_trait_spec")


# --------------------------------------------------------------------------
# preflight_config — per-stage 'lr' is optional (trainer never reads it)
# --------------------------------------------------------------------------

def test_preflight_config_accepts_trainer_canonical_stages(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images"
    lbls = tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls)},
        # launch_training's own default stage shape: freeze_to + epochs, no lr.
        "training": {"batch_size": 2,
                     "stages": [{"freeze_to": -1, "epochs": 5}, {"freeze_to": 2, "epochs": 10}]},
    }
    r = preflight_config(cfg)
    assert r["valid"] is True, r["issues"]

    # 'epochs' is still required per provided stage.
    cfg["training"]["stages"] = [{"freeze_to": 0}]
    r2 = preflight_config(cfg)
    assert any("Stage 0 missing 'epochs'" in i for i in r2["issues"])

    # No stages at all is fine — launch_training supplies its own default schedule.
    del cfg["training"]["stages"]
    assert preflight_config(cfg)["valid"] is True


def test_preflight_config_warns_on_ignored_per_stage_lr(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images"
    lbls = tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls)},
        "training": {"batch_size": 2,
                     "stages": [{"freeze_to": -1, "epochs": 5, "lr": 1e-3}]},
    }
    r = preflight_config(cfg)
    assert r["valid"] is True  # a per-stage lr is ignored, not rejected (StageSpec extra="allow")
    assert any("stages[0].lr is set but ignored" in w for w in r["warnings"])

    # No per-stage lr -> no warning.
    cfg["training"]["stages"] = [{"freeze_to": -1, "epochs": 5}]
    assert preflight_config(cfg)["warnings"] == []


def test_preflight_config_warns_when_most_candidates_wont_train(tmp_path):
    """Round 12 (2026-07-29): trainable_stems' own partition was computed by
    DetectionDataset/InstanceSegDataset and thrown away — a run whose label store admits only a
    fraction of its candidate images used to report "valid, no warnings" with no visibility into
    what would silently train on far fewer images than the operator expects."""
    pytest.importorskip("torch")
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images"
    lbls = tmp_path / "annotations"
    imgs.mkdir()
    lbls.mkdir()
    # 1 annotated, 3 unannotated (no label file at all) -> 75% of candidates won't train.
    Image.new("RGB", (20, 20)).save(imgs / "ann.jpg")
    json_io.write_annotations(lbls / "ann.json",
                              [Annotation(subject="catkin", geometry=BBox(2, 2, 10, 10))], 20, 20)
    for stem in ("a", "b", "c"):
        Image.new("RGB", (20, 20)).save(imgs / f"{stem}.jpg")

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls), "subject": "catkin"},
        "training": {"batch_size": 2},
    }
    r = preflight_config(cfg)
    assert r["valid"] is True  # informational only, never gating
    assert any("3/4 candidate images (75%) will not train" in w for w in r["warnings"]), r["warnings"]
    assert any("skipped_unannotated" in w for w in r["warnings"])


def test_preflight_config_no_coverage_warning_when_everything_trains(tmp_path):
    pytest.importorskip("torch")
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images"
    lbls = tmp_path / "annotations"
    imgs.mkdir()
    lbls.mkdir()
    Image.new("RGB", (20, 20)).save(imgs / "ann.jpg")
    json_io.write_annotations(lbls / "ann.json",
                              [Annotation(subject="catkin", geometry=BBox(2, 2, 10, 10))], 20, 20)

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls), "subject": "catkin"},
        "training": {"batch_size": 2},
    }
    assert preflight_config(cfg)["warnings"] == []


# --------------------------------------------------------------------------
# preflight_config — training_source seam (K9): a bare "module:function" string
# --------------------------------------------------------------------------

def test_preflight_config_training_source_shape_and_importability(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images"
    lbls = tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    base_cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls)},
        "training": {"batch_size": 2},
    }

    # A dict (the old, wrong shape) is rejected.
    cfg = dict(base_cfg, training_source={"train": "tests.bespoke_models:build_bespoke_detection"})
    r = preflight_config(cfg)
    assert any("training_source must be a non-empty" in i for i in r["issues"])

    # A bare string that doesn't import is rejected with the import error surfaced.
    cfg = dict(base_cfg, training_source="nonexistent_module:train")
    r = preflight_config(cfg)
    assert any("training_source not importable" in i for i in r["issues"])

    # A bare, importable string passes.
    cfg = dict(base_cfg, training_source="tests.bespoke_models:build_bespoke_detection")
    assert preflight_config(cfg)["valid"] is True

    # Absent training_source is fine (optional seam).
    assert preflight_config(base_cfg)["valid"] is True


# --------------------------------------------------------------------------
# preflight_config — selection_metric coherence (K9): reject a comparability-only
# metric for a center-match trait at validation time, not mid-run.
# --------------------------------------------------------------------------

def test_preflight_config_rejects_incoherent_selection_metric(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images"
    lbls = tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    base_cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls)},
        "training": {"batch_size": 2},
    }

    # A comparability-only metric for a center-match trait is rejected.
    cfg = dict(base_cfg)
    cfg["training"] = dict(cfg["training"], evaluation={"trait": "catkin", "selection_metric": "map50"})
    r = preflight_config(cfg)
    assert any("comparability-only" in i for i in r["issues"])

    # A governing metric for the same trait is fine.
    cfg["training"] = dict(cfg["training"], evaluation={"trait": "catkin", "selection_metric": "f1"})
    assert preflight_config(cfg)["valid"] is True

    # No trait -> no coherence gate, even for a comparability metric.
    cfg["training"] = dict(cfg["training"], evaluation={"selection_metric": "map50"})
    assert preflight_config(cfg)["valid"] is True


# --------------------------------------------------------------------------
# _apply_hpo_params — lr/weight_decay reach what the trainer actually reads
# --------------------------------------------------------------------------

def test_apply_hpo_params_lr_reaches_optimizer_param_groups():
    """Suggested lr/weight_decay must survive the trainer's exact config reads
    (top-level optimizer) all the way into optimizer.param_groups."""
    torch = pytest.importorskip("torch")
    from tcip_mcp.pipelines.training.optimizer_factory import build_optimizer
    from tcip_mcp.tools.training_tools import _apply_hpo_params

    base = {"model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                             "builder_kwargs": {"num_classes": 1}, "task": "detection"}}
    out = _apply_hpo_params(base, {"lr": 3e-3, "weight_decay": 2e-4})

    # Mirror generic_trainer.train()'s reads exactly (top-level keys + defaults).
    opt_cfg = out.get("optimizer", {"name": "adamw", "backbone_lr": 1e-4,
                                    "head_lr": 1e-3, "weight_decay": 1e-4})
    model = torch.nn.Linear(4, 2)
    optimizer = build_optimizer(
        opt_cfg.get("name", "adamw"), model,
        backbone_lr=opt_cfg.get("backbone_lr", 1e-4),
        head_lr=opt_cfg.get("head_lr", 1e-3),
        weight_decay=opt_cfg.get("weight_decay", 1e-4),
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-3)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(2e-4)


def test_apply_hpo_params_preserves_base_config_stages():
    """K11: sweeping lr must NOT overwrite the agent's own progressive-unfreeze schedule with a
    hardcoded recipe — base_config's stages (however it expressed them) survive unchanged."""
    from tcip_mcp.tools.training_tools import _apply_hpo_params

    custom_stages = [{"freeze_to": -1, "epochs": 2}, {"freeze_to": 0, "epochs": 8}]
    base = {"model_source": {"builder": "x:y", "task": "detection"},
            "training": {"stages": custom_stages}}
    out = _apply_hpo_params(base, {"lr": 3e-3})
    assert out["stages"] == custom_stages

    # No stages configured at all -> still nothing invented here; generic_trainer.train()'s own
    # single-stage fallback covers it.
    base_no_stages = {"model_source": {"builder": "x:y", "task": "detection"}}
    out2 = _apply_hpo_params(base_no_stages, {"lr": 3e-3})
    assert "stages" not in out2


def test_apply_hpo_params_derives_backbone_ratio_not_frozen():
    """K11: backbone_lr must scale by whatever ratio the agent's own base_config expressed, not
    a frozen *0.1 — a pinned constant here discards a deliberate agent choice (derive, don't pin)."""
    from tcip_mcp.tools.training_tools import _apply_hpo_params

    base = {"model_source": {"builder": "x:y", "task": "detection"},
            "optimizer": {"backbone_lr": 2e-5, "head_lr": 1e-4}}  # ratio 0.2, not 0.1
    out = _apply_hpo_params(base, {"lr": 0.02})
    assert out["optimizer"]["head_lr"] == pytest.approx(0.02)
    assert out["optimizer"]["backbone_lr"] == pytest.approx(0.004)  # 0.02 * 0.2, not 0.002

    # No explicit ratio expressed at all -> default 1.0, not the old frozen 0.1.
    base_no_ratio = {"model_source": {"builder": "x:y", "task": "detection"}}
    out2 = _apply_hpo_params(base_no_ratio, {"lr": 0.02})
    assert out2["optimizer"]["backbone_lr"] == pytest.approx(0.02)


def test_apply_hpo_params_unrecognized_key_reaches_top_level():
    """K11 (F1): a swept key outside the known optimizer/batch/weight_decay set must land at the
    TOP LEVEL of the resolved config (where train() reads it), not nested under "training" after
    normalize_train_config's hoist already ran — the bug that made it silently unreachable."""
    from tcip_mcp.tools.training_tools import _apply_hpo_params

    base = {"model_source": {"builder": "x:y", "task": "detection"}}
    out = _apply_hpo_params(base, {"momentum": 0.9})
    assert out["momentum"] == 0.9
    assert "momentum" not in out.get("training", {})


# --------------------------------------------------------------------------
# _run_hpo_trial — reports the composite (lower=better) each epoch + final, with
# failed / empty trials reporting +inf so a dead trial can never win a min sweep.
# The trial runs directly (no Ray) so the training machinery can be stubbed.
# --------------------------------------------------------------------------

class _FakeDataset:
    def __len__(self):
        return 4

    def __getitem__(self, i):
        return i


def _detection_base() -> dict:
    return {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": "imgs", "labels_dir": "lbls"},
        "training": {"batch_size": 2},
    }


def _patch_hpo_trial_machinery(monkeypatch, fake_train, captured=None):
    """Stub dataset building + training + loaders so a trial runs instantly, no Ray."""
    import torch.utils.data as tud
    from tcip_mcp.pipelines.data import samplers
    from tcip_mcp.pipelines.training import generic_trainer as gt
    from tcip_mcp.tools import training_tools as tt

    ds = _FakeDataset()

    def fake_auto_train_val(task, data_cfg, transforms):
        if captured is not None:
            captured["transforms"] = transforms
        return ds, ds

    monkeypatch.setattr(tt, "_auto_train_val", fake_auto_train_val)
    monkeypatch.setattr(gt, "train", fake_train)
    monkeypatch.setattr(samplers, "build_sampler", lambda *a, **k: None)
    monkeypatch.setattr(tud, "DataLoader", lambda *a, **k: object())


def test_run_hpo_trial_reports_each_epoch_then_final_composite(monkeypatch, tmp_path):
    """Every epoch's composite plus the final best_metric are reported (lower=better),
    so a min-mode scheduler sees the improving trace."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import _run_hpo_trial

    def fake_train(run, train_loader, val_loader, task="detection",
                   epoch_callback=None, resume_from=""):
        for epoch, value in enumerate([50.0, 40.0, 30.0]):
            if epoch_callback:
                epoch_callback(epoch, {"val_objective": value})
        run.best_metric = 30.0
        run.status = "completed"
        return run

    _patch_hpo_trial_machinery(monkeypatch, fake_train)
    reported: list = []
    _run_hpo_trial({"lr": 3e-4}, reported.append, _detection_base(), str(tmp_path / "trial_0"))
    assert reported == [50.0, 40.0, 30.0, 30.0]  # per-epoch trace + final composite


def test_run_hpo_trial_epoch_cb_prefers_selection_over_val_objective(monkeypatch, tmp_path):
    """K11/K9: once a center-match trait sets which key governs checkpoint choice ('selection'),
    HPO pruning must rank trials on that key, not the raw composite it can diverge from."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import _run_hpo_trial

    def fake_train(run, train_loader, val_loader, task="detection",
                   epoch_callback=None, resume_from=""):
        if epoch_callback:
            epoch_callback(0, {"val_objective": 99.0, "selection": 12.0})
        run.best_metric = 12.0
        run.status = "completed"
        return run

    _patch_hpo_trial_machinery(monkeypatch, fake_train)
    reported: list = []
    _run_hpo_trial({"lr": 3e-4}, reported.append, _detection_base(), str(tmp_path / "trial_0"))
    assert reported == [12.0, 12.0]  # the "selection" value, not 99.0


def test_run_hpo_trial_failed_or_empty_reports_inf(monkeypatch, tmp_path):
    """A crashed trial (or one with no model_source) reports +inf — the worst value under
    mode='min' — so it can never become the sweep's best."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import _run_hpo_trial

    def fake_train(run, train_loader, val_loader, task="detection",
                   epoch_callback=None, resume_from=""):
        raise RuntimeError("CUDA out of memory")

    _patch_hpo_trial_machinery(monkeypatch, fake_train)
    reported: list = []
    _run_hpo_trial({"lr": 3e-4}, reported.append, _detection_base(), str(tmp_path / "trial_0"))
    assert reported == [float("inf")]

    empty: list = []
    _run_hpo_trial({"lr": 3e-4}, empty.append, {"data": {}}, str(tmp_path / "trial_1"))  # no model_source
    assert empty == [float("inf")]


def test_run_hpo_trial_uses_base_augmentation_and_model(monkeypatch, tmp_path):
    """Trials train under the final run's regime: base_config augmentation reaches the train
    dataset, and the bespoke model_source is carried through. (Loss is owned by the builder.)"""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import _run_hpo_trial

    captured: dict = {}

    def fake_train(run, train_loader, val_loader, task="classification",
                   epoch_callback=None, resume_from=""):
        captured["model_source"] = run.config["model_source"]
        run.best_metric = 1.0
        run.status = "completed"
        return run

    _patch_hpo_trial_machinery(monkeypatch, fake_train, captured=captured)
    base = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_classifier",
                         "builder_kwargs": {"num_classes": 2}, "task": "classification"},
        "data": {"images_dir": "imgs"},
        "training": {"batch_size": 2},
        "augmentation": {"horizontal_flip": 0.5},
    }
    _run_hpo_trial({"lr": 3e-4}, [].append, base, str(tmp_path / "trial_0"))
    assert captured["transforms"] is not None       # augmentation was built + passed
    assert captured["model_source"]["builder"].endswith(":build_bespoke_classifier")


def test_run_hpo_trial_writes_resolved_config_with_unconsumed_params(monkeypatch, tmp_path):
    """K11 (F1): a swept key the training body never reads is surfaced by OBSERVATION, not
    gated by a whitelist. resolved_config.json records which swept keys went unconsumed."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import _run_hpo_trial

    def fake_train(run, train_loader, val_loader, task="detection",
                   epoch_callback=None, resume_from=""):
        run.config.get("lr")  # a known key, consumed -- but "totally_bogus_key" never read
        run.best_metric = 1.0
        run.status = "completed"
        return run

    _patch_hpo_trial_machinery(monkeypatch, fake_train)
    trial_dir = tmp_path / "trial_0"
    _run_hpo_trial({"lr": 3e-4, "totally_bogus_key": 5}, [].append, _detection_base(), str(trial_dir))

    resolved = json.loads((trial_dir / "resolved_config.json").read_text())
    assert resolved["unconsumed_params"] == ["totally_bogus_key"]


def _bespoke_hpo_agent_train(ctx):
    """Module-level (dotted-import-able) bespoke train(ctx) that reads its own swept key."""
    ctx.config.get("custom_axis")  # the bespoke loop reads its own swept key
    ctx.run.status = "completed"
    ctx.run.best_metric = 1.0


def test_run_hpo_trial_bespoke_custom_key_not_falsely_flagged_unconsumed(monkeypatch, tmp_path):
    """K11 (F1): a bespoke training_source reading its OWN swept custom key must not be
    falsely flagged unconsumed merely because generic_trainer.train() doesn't know it —
    tracking is genuine runtime access, not a static comparison against train()'s key list."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import _run_hpo_trial

    import tcip_mcp.tools.training_tools as tt
    monkeypatch.setattr(tt, "_auto_train_val",
                        lambda task, data_cfg, transforms: (_FakeDataset(), _FakeDataset()))
    import torch.utils.data as tud
    monkeypatch.setattr(tud, "DataLoader", lambda *a, **k: object())
    from tcip_mcp.pipelines.data import samplers
    monkeypatch.setattr(samplers, "build_sampler", lambda *a, **k: None)

    base = {"model_source": {"builder": "x:y", "task": "detection"},
            "training_source": f"{__name__}:_bespoke_hpo_agent_train"}
    trial_dir = tmp_path / "trial_0"
    _run_hpo_trial({"custom_axis": 42}, [].append, base, str(trial_dir))

    resolved = json.loads((trial_dir / "resolved_config.json").read_text())
    assert resolved["unconsumed_params"] == []


# --------------------------------------------------------------------------
# get_worst_predictions — confidence comes from the canonical prediction format
# --------------------------------------------------------------------------

def test_get_worst_predictions_reads_canonical_confidence(tmp_path, monkeypatch):
    """Prediction files are per-image JSON with a native ``score`` (json_io); confidence
    reads from that field, not from box geometry. The old parser read a normalized box
    HEIGHT as confidence, so the (1 - avg_conf) ranking term was ~1.0 for every image with
    small boxes (e.g. catkins)."""
    pytest.importorskip("torch")
    monkeypatch.chdir(tmp_path)
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.training_tools import get_worst_predictions

    preds = tmp_path / "preds"
    gts = tmp_path / "labels"
    preds.mkdir()
    gts.mkdir()

    def write_image(stem: str, scores: list[float]) -> None:
        # Confidence lives in the JSON `score`; box geometry is irrelevant to this
        # count + confidence heuristic (no IoU matching), so the boxes can be anything.
        pred_anns = [Annotation(subject="catkin", geometry=BBox(10.0, 10.0, 40.0, 22.0), score=s)
                     for s in scores]
        json_io.write_annotations(str(preds / f"{stem}.json"), pred_anns, 100, 100)
        # Matching GT count → missed = extra = 0, error is exactly (1 - avg_conf).
        gt_anns = [Annotation(subject="catkin", geometry=BBox(20.0, 11.0, 40.0, 31.0)) for _ in scores]
        json_io.write_annotations(str(gts / f"{stem}.json"), gt_anns, 100, 100)

    write_image("confident", [0.9, 0.9])
    write_image("shaky", [0.1, 0.1])

    out = get_worst_predictions(str(preds), str(gts), top_k=2)
    by_stem = {w["stem"]: w["error_score"] for w in out["worst_images"]}
    assert by_stem["confident"] == pytest.approx(0.1, abs=1e-3)
    assert by_stem["shaky"] == pytest.approx(0.9, abs=1e-3)
    assert out["worst_images"][0]["stem"] == "shaky"  # low confidence ranks worst


# --------------------------------------------------------------------------
# _ensure_experiment — experiments are immutable on relaunch
# --------------------------------------------------------------------------

def test_ensure_experiment_mints_fresh_id_instead_of_mutating(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # .tcip/experiments lives under cwd
    import json

    from tcip_mcp.experiments import create_experiment, log_metrics, update_status
    from tcip_mcp.tools.training_tools import _ensure_experiment

    # A completed experiment with recorded history.
    create_experiment("exp1", {"a": 1}, data_source="imgs_v1")
    update_status("exp1", "running")
    log_metrics("exp1", 1, {"map50": 0.7})
    update_status("exp1", "completed")
    exp_dir = tmp_path / ".tcip" / "experiments" / "exp1"
    status_before = (exp_dir / "status.json").read_text()
    metrics_before = (exp_dir / "metrics.jsonl").read_text()

    # Relaunching with the same experiment_id must not reuse it.
    eid = _ensure_experiment("exp1", {"a": 2}, "imgs_v2", resume_from="", run_id="run_9_0",
                             output_dir="out")
    assert eid == "exp1_run_9_0"
    assert (exp_dir / "status.json").read_text() == status_before      # untouched
    assert (exp_dir / "metrics.jsonl").read_text() == metrics_before   # untouched
    assert json.loads((exp_dir / "config.json").read_text()) == {"a": 1}

    # The fresh experiment exists and points back at the original.
    fresh_dir = tmp_path / ".tcip" / "experiments" / "exp1_run_9_0"
    lineage = json.loads((fresh_dir / "lineage.json").read_text())
    assert lineage["parent_experiment"] == "exp1"
    assert lineage["data_source"] == "imgs_v2"


def test_ensure_experiment_attaches_to_precreated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.training_tools import _ensure_experiment

    # Agent pre-created the experiment (state 'created', no metrics): attach.
    create_experiment("pre", {"a": 1})
    assert _ensure_experiment("pre", {"a": 1}, None, resume_from="", run_id="r1",
                              output_dir="out") == "pre"

    # A brand-new id is simply created.
    assert _ensure_experiment("new", {"a": 1}, None, resume_from="", run_id="r3",
                              output_dir="out") == "new"


def test_ensure_experiment_attaches_to_precreated_and_rewrites_config(tmp_path, monkeypatch):
    """K12 finding 3: a pristine pre-created experiment's config.json is refreshed with the config
    actually launched (tiling/seed resolved after create_experiment ran), not left describing the
    config as it stood before those were resolved."""
    monkeypatch.chdir(tmp_path)
    import json

    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.training_tools import _ensure_experiment

    create_experiment("pre", {"a": 1})
    effective_config = {"a": 1, "data": {"tiling": {"tile_size": 512}}, "seed": 99}
    assert _ensure_experiment("pre", effective_config, None, resume_from="", run_id="r1",
                              output_dir="out") == "pre"

    config = json.loads((tmp_path / ".tcip" / "experiments" / "pre" / "config.json").read_text())
    assert config == effective_config


def test_ensure_experiment_resume_into_populated_id_mints_fresh_parented_id(tmp_path, monkeypatch):
    """K12 finding 4: resume_from no longer reuses an id that already has recorded history — it
    mints a fresh parented id instead, matching the non-resume collision behavior. Silently reusing
    it discarded the resumed run's own metrics/lineage writes (refused by the terminal-state lock)
    and let ModelRegistry.register_model replace the original's registry entry by name with no
    record of what was superseded."""
    monkeypatch.chdir(tmp_path)
    import json

    from tcip_mcp.experiments import create_experiment, log_metrics, update_status
    from tcip_mcp.tools.training_tools import _ensure_experiment

    create_experiment("res", {"a": 1})
    update_status("res", "running")
    log_metrics("res", 1, {"loss": 0.5})
    eid = _ensure_experiment("res", {"a": 1}, None, resume_from="ckpt/checkpoint_epoch_5.pt",
                             run_id="r2", output_dir="out")
    assert eid == "res_r2"

    fresh_dir = tmp_path / ".tcip" / "experiments" / "res_r2"
    assert fresh_dir.is_dir()
    lineage = json.loads((fresh_dir / "lineage.json").read_text())
    assert lineage["parent_experiment"] == "res"


def test_ensure_experiment_resume_into_pristine_id_still_attaches(tmp_path, monkeypatch):
    """A resume_from target that is itself still pristine (never actually run) is unaffected —
    pristine reuse doesn't depend on resume_from at all."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.training_tools import _ensure_experiment

    create_experiment("pre2", {"a": 1})
    eid = _ensure_experiment("pre2", {"a": 1}, None,
                             resume_from="ckpt/checkpoint_epoch_5.pt", run_id="r9",
                             output_dir="out")
    assert eid == "pre2"
