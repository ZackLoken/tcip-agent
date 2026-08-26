"""A trait's first localization derivation, driven through the real training subprocess.

The derived-localization write in ``evaluation.resolve_match_criterion`` persists the trait's
``localization`` and then appends an audit line through ``record_event_or_raise``, which raises
rather than warns when the append fails, and ``generic_trainer`` marks such a run failed. Both
halves are proved at unit level elsewhere (``test_evaluation_metrics.py``,
``test_audit_entry_failure.py``); this drives the healthy half through the process a training run
actually executes in, ``subprocess_worker.run``, so the boundary between the two is covered by
the entry point itself and not only by in-process calls.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest


def _seed_dataset(root: Path) -> tuple[Path, Path, Path, Path]:
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir, labels_dir = root / "images", root / "labels"
    val_images, val_labels = root / "val_images", root / "val_labels"
    for d in (images_dir, labels_dir, val_images, val_labels):
        d.mkdir(parents=True)
    # 20 px boxes: under a 15 px jitter their achievable IoU is below 0.5, so the derivation
    # lands on center_match, the same geometry the unit test derives from.
    box = BBox(10, 10, 30, 30)
    for i in range(2):
        Image.new("RGB", (128, 128)).save(images_dir / f"t{i}.png")
        json_io.write_annotations(str(labels_dir / f"t{i}.json"),
                                  [Annotation(subject="leaf", geometry=box)], 128, 128)
    Image.new("RGB", (128, 128)).save(val_images / "v0.png")
    json_io.write_annotations(str(val_labels / "v0.json"),
                              [Annotation(subject="leaf", geometry=box)], 128, 128)
    return images_dir, labels_dir, val_images, val_labels


def _seed_bare_trait(name: str) -> None:
    import tcip_store as ts
    from tcip_mcp.project_paths import resolve_state
    from tcip_mcp.traits import _TRAIT_SPECS_RELPATH, trait_spec_key

    specs_dir = resolve_state(_TRAIT_SPECS_RELPATH)
    ts.replace(trait_spec_key(specs_dir, name), {"name": name, "delivers": ["leaf_length"]},
               expect=ts.Version.ABSENT)


def _wait_terminal(run_id: str, seconds: float) -> str:
    from tcip_mcp.tools import training_tools

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        status = training_tools.check_training_status(run_id)
        if status.get("status") in ("completed", "failed", "cancelled"):
            return str(status.get("status"))
        time.sleep(0.5)
    pytest.fail("timed out waiting for the training subprocess to reach a terminal state")


def test_first_derivation_through_the_real_subprocess_records_the_kind_and_its_audit_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The child derives the unrecorded kind from the validation GT, persists it on the spec
    under the run's own root, appends the audit line naming the derivation, and completes."""
    pytest.importorskip("torchvision")
    monkeypatch.chdir(tmp_path)
    import tcip_store as ts
    from tcip_mcp import audit as audit_module
    from tcip_mcp.tools import training_tools
    from tcip_mcp.traits import get_trait

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})
    images_dir, labels_dir, val_images, val_labels = _seed_dataset(tmp_path / "ds")
    _seed_bare_trait("leaf")
    assert get_trait("leaf").localization == ""

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1, "min_size": 64, "max_size": 128},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "leaf",
                 "val_images_dir": str(val_images), "val_labels_dir": str(val_labels)},
        "evaluation": {"trait": "leaf"},
        "training": {"batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
                     "mixed_precision": False, "device": "cpu"},
    }
    res = training_tools.launch_training(cfg, str(tmp_path / "out"))
    assert "error" not in res, res
    assert res["pid"] != os.getpid()

    assert _wait_terminal(res["run_id"], 180) == "completed"

    assert get_trait("leaf").localization == "center_match"
    key = audit_module.audit_log_key(audit_module.platform_audit_scope())
    rows = [r for r in ts.read_log(key).records if r["tool"] == "trait_spec_field_derived"]
    assert len(rows) == 1
    assert rows[0]["arguments"]["trait"] == "leaf"
    assert rows[0]["arguments"]["field"] == "localization"
    assert rows[0]["arguments"]["value"] == "center_match"
