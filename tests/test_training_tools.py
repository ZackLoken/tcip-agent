"""training_tools fixes — validator/StageSpec alignment, HPO param plumbing,
ASHA pruning direction, failed-trial sentinels, HPO/train regime parity,
experiment immutability on relaunch, and canonical-format confidence parsing
in get_worst_predictions."""

from __future__ import annotations

from pathlib import Path

import pytest


# --------------------------------------------------------------------------
# validate_config — per-stage 'lr' is optional (trainer never reads it)
# --------------------------------------------------------------------------

def test_validate_config_accepts_trainer_canonical_stages(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import validate_config

    imgs = tmp_path / "images"
    lbls = tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    cfg = {
        "model_spec": {"backbone": {"name": "tv_resnet50"}, "neck": {"name": "fpn"},
                       "heads": [{"name": "anchor_detection", "num_classes": 1}]},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls)},
        # launch_training's own default stage shape: freeze_to + epochs, no lr.
        "training": {"batch_size": 2,
                     "stages": [{"freeze_to": -1, "epochs": 5}, {"freeze_to": 2, "epochs": 10}]},
    }
    r = validate_config(cfg)
    assert r["valid"] is True, r["issues"]

    # 'epochs' is still required per provided stage.
    cfg["training"]["stages"] = [{"freeze_to": 0}]
    r2 = validate_config(cfg)
    assert any("Stage 0 missing 'epochs'" in i for i in r2["issues"])

    # No stages at all is fine — launch_training supplies its own default schedule.
    del cfg["training"]["stages"]
    assert validate_config(cfg)["valid"] is True


# --------------------------------------------------------------------------
# _apply_hpo_params — lr/weight_decay reach what the trainer actually reads
# --------------------------------------------------------------------------

def test_apply_hpo_params_lr_reaches_optimizer_param_groups():
    """Suggested lr/weight_decay must survive the trainer's exact config reads
    (top-level optimizer/stages) all the way into optimizer.param_groups."""
    torch = pytest.importorskip("torch")
    from tcip_mcp.pipelines.training.optimizer_factory import build_optimizer
    from tcip_mcp.tools.training_tools import _apply_hpo_params

    base = {"model_spec": {"backbone": {"name": "tv_resnet50"},
                           "heads": [{"name": "anchor_detection"}]}}
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
# run_hpo — ASHA pruning direction, failed-trial sentinels, regime parity
# --------------------------------------------------------------------------

class _FakeDataset:
    def __len__(self):
        return 4

    def __getitem__(self, i):
        return i


def _detection_base() -> dict:
    return {
        "model_spec": {"backbone": {"name": "tv_resnet50"},
                       "heads": [{"name": "anchor_detection", "task": "detection"}]},
        "data": {"images_dir": "imgs", "labels_dir": "lbls"},
        "training": {"batch_size": 2},
    }


def _patch_hpo_trial_machinery(monkeypatch, fake_train, captured=None):
    """Stub dataset building + training so run_hpo trials run instantly."""
    from tcip_mcp.pipelines.training import generic_trainer as gt
    from tcip_mcp.tools import training_tools as tt

    ds = _FakeDataset()

    def fake_auto_train_val(task, data_cfg, transforms):
        if captured is not None:
            captured["transforms"] = transforms
        return ds, ds

    monkeypatch.setattr(tt, "_auto_train_val", fake_auto_train_val)
    monkeypatch.setattr(gt, "train", fake_train)


def test_run_hpo_asha_maximize_never_prunes_best_trial(tmp_path, monkeypatch):
    """The composite objective is lower=better; under direction='maximize' + ASHA a
    trial that beats every prior trial at every epoch must complete, not be pruned
    (raw un-negated reports made ASHA kill exactly the improving trials)."""
    pytest.importorskip("torch")
    pytest.importorskip("optuna")
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import run_hpo

    n_epochs = 5

    def fake_train(run, train_loader, val_loader, task="detection",
                   epoch_callback=None, resume_from=""):
        t = int(Path(run.output_dir).name.split("_")[-1])  # trial number
        # Strictly improving within a trial; each later trial dominates all
        # earlier ones at every epoch (a converging TPE sweep).
        values = [100.0 - 10.0 * t - e for e in range(n_epochs)]
        for epoch, value in enumerate(values):
            if epoch_callback:
                epoch_callback(epoch, {"val_objective": value})
        run.best_metric = min(values)
        run.status = "completed"
        return run

    _patch_hpo_trial_machinery(monkeypatch, fake_train)
    result = run_hpo(
        _detection_base(),
        param_space={"lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2}},
        n_trials=6, output_dir=str(tmp_path),
        direction="maximize", pruner="asha", grace_period=1, reduction_factor=2,
    )
    states = {t["number"]: t["state"] for t in result["all_trials"]}
    assert states[5] == "TrialState.COMPLETE"  # the best trial survived ASHA
    # best_value is the negated best composite of the best trial (lowest composite).
    assert result["best_value"] == pytest.approx(-(100.0 - 50.0 - (n_epochs - 1)))


def test_run_hpo_failed_trial_never_wins(tmp_path, monkeypatch):
    """A crashed trial must rank below every real trial: it used to return 0.0,
    which beat all successful trials' negative -composite values under maximize."""
    pytest.importorskip("torch")
    pytest.importorskip("optuna")
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import run_hpo

    def fake_train(run, train_loader, val_loader, task="detection",
                   epoch_callback=None, resume_from=""):
        t = int(Path(run.output_dir).name.split("_")[-1])
        if t == 0:
            raise RuntimeError("CUDA out of memory")
        run.best_metric = 10.0 + t  # trial 1 is the best real trial
        run.status = "completed"
        return run

    _patch_hpo_trial_machinery(monkeypatch, fake_train)
    result = run_hpo(
        _detection_base(),
        param_space={"lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2}},
        n_trials=3, output_dir=str(tmp_path),
        direction="maximize", pruner="none",
    )
    values = {t["number"]: t["value"] for t in result["all_trials"]}
    assert values[0] == float("-inf")               # crashed trial = worst possible
    assert result["best_value"] == pytest.approx(-11.0)  # best REAL trial wins


def test_run_hpo_trials_use_base_augmentation_and_loss(tmp_path, monkeypatch):
    """Trials must train under the final run's regime: base_config augmentation
    reaches the train dataset and the W8 imbalance loss reaches the head spec."""
    pytest.importorskip("torch")
    pytest.importorskip("optuna")
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import run_hpo

    captured: dict = {}

    def fake_train(run, train_loader, val_loader, task="classification",
                   epoch_callback=None, resume_from=""):
        captured["head"] = run.config["model_spec"]["heads"][0]
        run.best_metric = 1.0
        run.status = "completed"
        return run

    _patch_hpo_trial_machinery(monkeypatch, fake_train, captured=captured)
    base = {
        "model_spec": {"backbone": {"name": "tv_resnet50"},
                       "heads": [{"name": "classification", "task": "classification",
                                  "num_classes": 2}]},
        "data": {"images_dir": "imgs"},
        "training": {"batch_size": 2},
        "augmentation": {"horizontal_flip": 0.5},
        "loss": {"name": "focal", "class_weights": [1.0, 2.0]},
    }
    run_hpo(
        base, param_space={"lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2}},
        n_trials=1, output_dir=str(tmp_path),
        direction="maximize", pruner="none",
    )
    assert captured["transforms"] is not None       # augmentation was built + passed
    assert captured["head"]["loss"] == "focal"      # W8 loss injected into the trial
    assert captured["head"]["class_weights"] == [1.0, 2.0]


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

    out = get_worst_predictions(str(preds), str(gts), n=2)
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

    # Relaunching with the same experiment_id must NOT reuse it.
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
