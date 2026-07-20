"""training_tools fixes — validator/StageSpec alignment, HPO param plumbing,
HPO trial reporting (per-epoch trace + failed-trial sentinels), HPO/train regime parity,
experiment immutability on relaunch, and canonical-format confidence parsing
in get_worst_predictions."""

from __future__ import annotations

import pytest


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


# --------------------------------------------------------------------------
# _apply_hpo_params — lr/weight_decay reach what the trainer actually reads
# --------------------------------------------------------------------------

def test_apply_hpo_params_lr_reaches_optimizer_param_groups():
    """Suggested lr/weight_decay must survive the trainer's exact config reads
    (top-level optimizer/stages) all the way into optimizer.param_groups."""
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

    # The unfreeze schedule sits where train() reads it: top-level config["stages"].
    stages = out.get("stages", [{"freeze_to": 0, "epochs": 10}])
    assert [s["freeze_to"] for s in stages] == [-1, 2, 0]
    assert sum(s["epochs"] for s in stages) == 10  # per-trial budget unchanged


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


def test_run_hpo_trial_reports_each_epoch_then_final_composite(monkeypatch):
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
    _run_hpo_trial({"lr": 3e-4}, reported.append, _detection_base(), "trial_0")
    assert reported == [50.0, 40.0, 30.0, 30.0]  # per-epoch trace + final composite


def test_run_hpo_trial_failed_or_empty_reports_inf(monkeypatch):
    """A crashed trial (or one with no model_source) reports +inf — the worst value under
    mode='min' — so it can never become the sweep's best."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import _run_hpo_trial

    def fake_train(run, train_loader, val_loader, task="detection",
                   epoch_callback=None, resume_from=""):
        raise RuntimeError("CUDA out of memory")

    _patch_hpo_trial_machinery(monkeypatch, fake_train)
    reported: list = []
    _run_hpo_trial({"lr": 3e-4}, reported.append, _detection_base(), "trial_0")
    assert reported == [float("inf")]

    empty: list = []
    _run_hpo_trial({"lr": 3e-4}, empty.append, {"data": {}}, "trial_1")  # no model_source
    assert empty == [float("inf")]


def test_run_hpo_trial_uses_base_augmentation_and_model(monkeypatch):
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
    _run_hpo_trial({"lr": 3e-4}, [].append, base, "trial_0")
    assert captured["transforms"] is not None       # augmentation was built + passed
    assert captured["model_source"]["builder"].endswith(":build_bespoke_classifier")


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
    from tcip_annotation.state import BBox, PredBBox
    from tcip_mcp.tools.training_tools import get_worst_predictions

    preds = tmp_path / "preds"
    gts = tmp_path / "labels"
    preds.mkdir()
    gts.mkdir()

    def write_image(stem: str, scores: list[float]) -> None:
        # Confidence lives in the JSON `score`; box geometry is irrelevant to this
        # count + confidence heuristic (no IoU matching), so the boxes can be anything.
        pred_boxes = [PredBBox(10.0, 10.0, 40.0, 22.0, 1, confidence=s) for s in scores]
        json_io.write_detect(str(preds / f"{stem}.json"), pred_boxes, 100, 100)
        # Matching GT count → missed = extra = 0, error is exactly (1 - avg_conf).
        gt_boxes = [BBox(20.0, 11.0, 40.0, 31.0, 0) for _ in scores]
        json_io.write_detect(str(gts / f"{stem}.json"), gt_boxes, 100, 100)

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
    eid = _ensure_experiment("exp1", {"a": 2}, "imgs_v2", resume_from="", run_id="run_9_0")
    assert eid == "exp1_run_9_0"
    assert (exp_dir / "status.json").read_text() == status_before      # untouched
    assert (exp_dir / "metrics.jsonl").read_text() == metrics_before   # untouched
    assert json.loads((exp_dir / "config.json").read_text()) == {"a": 1}

    # The fresh experiment exists and points back at the original.
    fresh_dir = tmp_path / ".tcip" / "experiments" / "exp1_run_9_0"
    lineage = json.loads((fresh_dir / "lineage.json").read_text())
    assert lineage["parent_experiment"] == "exp1"
    assert lineage["data_source"] == "imgs_v2"


def test_ensure_experiment_attaches_to_precreated_and_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, log_metrics, update_status
    from tcip_mcp.tools.training_tools import _ensure_experiment

    # Agent pre-created the experiment (state 'created', no metrics): attach.
    create_experiment("pre", {"a": 1})
    assert _ensure_experiment("pre", {"a": 1}, None, resume_from="", run_id="r1") == "pre"

    # resume_from continues its own experiment even after it has history.
    create_experiment("res", {"a": 1})
    update_status("res", "running")
    log_metrics("res", 1, {"loss": 0.5})
    eid = _ensure_experiment("res", {"a": 1}, None, resume_from="ckpt/checkpoint_epoch_5.pt",
                             run_id="r2")
    assert eid == "res"

    # A brand-new id is simply created.
    assert _ensure_experiment("new", {"a": 1}, None, resume_from="", run_id="r3") == "new"
