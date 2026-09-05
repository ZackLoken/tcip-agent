"""Tests for training_tools: validator/StageSpec alignment, HPO param plumbing, HPO trial
reporting (per-epoch trace + failed-trial sentinels), HPO/train regime parity, experiment
immutability on relaunch, and canonical-format confidence parsing in get_worst_predictions."""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store as ts

# No built-in traits: seed_bud_trait_spec (conftest.py) writes a real bud.yml into this
# test's pinned platform state root so trait="bud_opening" call sites keep resolving.
pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")


# --------------------------------------------------------------------------
# preflight_config: per-stage 'lr' is optional (trainer never reads it)
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

    # No stages at all is fine: launch_training supplies its own default schedule.
    del cfg["training"]["stages"]
    assert preflight_config(cfg)["valid"] is True


def test_preflight_config_refuses_a_misspelled_model_source_key_nested_under_training():
    """preflight_config reads model_source off normalize_train_config's hoisted view, so the
    schema it validates must be the same view: a model_source nested under training (TrainingSection
    is extra="allow") is otherwise never typed by ModelSourceSchema (extra="forbid") and a
    misspelling like bulider_kwargs reaches the trainer instead of being refused by name."""
    from tcip_mcp.tools.training_tools import preflight_config

    cfg = {"training": {"model_source": {
        "builder": "tests.bespoke_models:build_bespoke_detection",
        "bulider_kwargs": {"num_classes": 1}, "task": "detection",
    }}}
    r = preflight_config(cfg)
    assert r["valid"] is False
    assert any("bulider_kwargs" in i for i in r["issues"]), r["issues"]


def test_preflight_config_types_a_non_dict_data_section_instead_of_raising():
    from tcip_mcp.tools.training_tools import preflight_config

    r = preflight_config({"training": {"data": "x"}})
    assert r["valid"] is False
    assert any("'data' must be a dict" in i for i in r["issues"]), r["issues"]


# preflight_config's overfit branch: reseed-run-restore, never gating

def _frozen_detection_builder(**kwargs):
    """An importable builder whose model has nothing to optimize (a legitimate stage-0 shape)."""
    from tests import bespoke_models

    model = bespoke_models.build_bespoke_detection(num_classes=1, min_size=64, max_size=96)
    for p in model.parameters():
        p.requires_grad = False
    return model


def _detection_smoke_cfg(builder: str, tmp_path: Path) -> dict:
    """A smoke config over the bespoke detector at a small resize target: the detector's own
    transform resizes the 224 px contract input to ``min_size``, so 64 keeps two smoke builds
    plus twenty overfit steps to seconds where the 800 px default took minutes. ``fcos`` builds
    in a fraction of ``faster_rcnn``'s time over the same resnet18 backbone (single-stage, no
    region-proposal network), and nothing either smoke test asserts is faster-rcnn-specific.
    """
    imgs = tmp_path / "images"
    lbls = tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    return {
        "model_source": {"builder": builder,
                         "builder_kwargs": {"num_classes": 1, "min_size": 64, "max_size": 96,
                                            "detector": "fcos"},
                         "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls)},
        "training": {"batch_size": 2},
    }


def test_preflight_config_overfit_restores_rng_state(tmp_path, monkeypatch):
    """The overfit branch's own reseed-run-restore leaves no net trace on the streams: a plain
    smoke build (overfit=False) and the same build with overfit=True, run from the same seed,
    must consume the streams identically once the overfit call returns. Comparing against a
    fresh reseed with nothing run in between would be wrong: build_model + check_model_contract
    draw real entropy too, on both paths alike, so the only valid baseline is the sibling call
    that skips just the overfit branch."""
    pytest.importorskip("torch")
    import functools
    import random

    import numpy as np
    import torch

    from tcip_mcp.pipelines import model_contract
    from tcip_mcp.tools.training_tools import preflight_config

    # RNG-stream identity holds after any positive step count; four steps exercise the same
    # reseed-run-restore path as the default twenty for a fraction of the wall time.
    monkeypatch.setattr(model_contract, "overfit_check",
                        functools.partial(model_contract.overfit_check, steps=4))

    cfg = _detection_smoke_cfg("tests.bespoke_models:build_bespoke_detection", tmp_path)

    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    preflight_config(cfg, smoke=True, overfit=False)
    without_overfit = (random.random(), np.random.rand(), torch.rand(1))

    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    r = preflight_config(cfg, smoke=True, overfit=True)
    assert r["overfit_check"] is not None
    with_overfit = (random.random(), np.random.rand(), torch.rand(1))

    assert with_overfit[0] == without_overfit[0]
    assert with_overfit[1] == without_overfit[1]
    assert torch.equal(with_overfit[2], without_overfit[2])


def test_preflight_config_overfit_on_all_frozen_model_reports_without_raising(tmp_path):
    """check_model_contract's own gradient-presence check independently requires a real
    learnable path, so a literally all-frozen model already fails smoke on its own terms
    (valid stays False for that reason); what this proves is narrower and still real: the
    overfit branch's own empty-parameter case never raises out of preflight_config, and its
    report still names the empty parameter list rather than being swallowed."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    cfg = _detection_smoke_cfg(f"{__name__}:_frozen_detection_builder", tmp_path)
    r = preflight_config(cfg, smoke=True, overfit=True)
    assert any("no parameter received a gradient" in i or "does not require grad" in i
              for i in r["issues"])
    assert r["overfit_check"]["passed"] is False
    assert "empty parameter list" in r["overfit_check"]["issue"]


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
    """trainable_stems' own partition is computed by DetectionDataset/InstanceSegDataset and
    thrown away: a run whose label store admits only a fraction of its candidate images must not
    report "valid, no warnings" with no visibility into what would silently train on far fewer
    images than the operator expects."""
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
                              [Annotation(subject="bud", geometry=BBox(2, 2, 10, 10))], 20, 20)
    for stem in ("a", "b", "c"):
        Image.new("RGB", (20, 20)).save(imgs / f"{stem}.jpg")

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls), "subject": "bud"},
        "training": {"batch_size": 2},
    }
    r = preflight_config(cfg)
    assert r["valid"] is True  # informational only, never gating
    assert any("3/4 candidate images (75%) will not train" in w for w in r["warnings"]), r["warnings"]
    assert any("skipped_unannotated" in w for w in r["warnings"])


def test_preflight_config_warns_of_a_negative_the_label_file_now_contradicts(tmp_path):
    """A stored negative whose label file now holds the subject is a stale confirmation: excluded
    from the negative count (it trains on its real content instead) but named for the agent at the
    validation door rather than swallowed."""
    pytest.importorskip("torch")
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.dataset_layout import CONFIRMED_NEGATIVE, record_image_statuses, status_bucket
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images"
    lbls = tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    Image.new("RGB", (20, 20)).save(imgs / "bush.jpg")
    json_io.write_annotations(lbls / "bush.json", [], 20, 20, keep_empty=True)
    record_image_statuses(tmp_path, status_bucket("bud", None), {"bush.jpg": CONFIRMED_NEGATIVE},
                          recorded_by="user:breeder")
    json_io.write_annotations(
        lbls / "bush.json", [Annotation(subject="bud", geometry=BBox(2, 2, 10, 10))], 20, 20)

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls), "subject": "bud"},
        "training": {"batch_size": 2},
    }
    r = preflight_config(cfg)
    assert r["valid"] is True  # informational only, never gating
    assert any("bush.jpg" in w and "stale" in w for w in r["warnings"]), r["warnings"]


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
                              [Annotation(subject="bud", geometry=BBox(2, 2, 10, 10))], 20, 20)

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls), "subject": "bud"},
        "training": {"batch_size": 2},
    }
    assert preflight_config(cfg)["warnings"] == []


# --------------------------------------------------------------------------
# preflight_config's training_source seam: a bare "module:function" string
# --------------------------------------------------------------------------

def test_preflight_config_blocks_rather_than_swallows_an_unreadable_label(tmp_path):
    """The coverage check's own trainable_stems scan must not fold an unreadable label into a
    generic build failure it silently drops: a run over this labels_dir would fail on the same
    file, so preflight reports it as a blocking issue, naming the file, not a warning."""
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
    Image.new("RGB", (20, 20)).save(imgs / "bad.jpg")
    json_io.write_annotations(lbls / "ann.json",
                              [Annotation(subject="bud", geometry=BBox(2, 2, 10, 10))], 20, 20)
    (lbls / "bad.json").write_bytes(b"{not json")

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls), "subject": "bud"},
        "training": {"batch_size": 2},
    }
    r = preflight_config(cfg)
    assert r["valid"] is False
    assert any("bad.json" in i for i in r["issues"]), r["issues"]


@pytest.mark.parametrize("bad_document", [
    '{"image": "bad", "width": 20, "height": 20, "annotations": 5}',
    '{"image": "bad", "width": 20, "height": 20, "annotations": [7]}',
])
def test_preflight_config_blocks_a_document_only_the_admission_reader_refuses(tmp_path, bad_document):
    """A document that decodes to a dict but whose annotations field is not a list, or whose
    record cannot be coerced, is exactly what the run's own admission (read_annotations) refuses
    at launch: preflight reads through the same call so it blocks here too, rather than passing a
    document the launch then aborts on."""
    pytest.importorskip("torch")
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images"
    lbls = tmp_path / "annotations"
    val_imgs = tmp_path / "val_images"
    val_lbls = tmp_path / "val_annotations"
    for d in (imgs, lbls, val_imgs, val_lbls):
        d.mkdir()
    Image.new("RGB", (20, 20)).save(imgs / "ann.jpg")
    Image.new("RGB", (20, 20)).save(val_imgs / "bad.jpg")
    json_io.write_annotations(lbls / "ann.json",
                              [Annotation(subject="bud", geometry=BBox(2, 2, 10, 10))], 20, 20)
    (val_lbls / "bad.json").write_text(bad_document, encoding="utf-8")

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls), "subject": "bud",
                 "val_images_dir": str(val_imgs), "val_labels_dir": str(val_lbls)},
        "training": {"batch_size": 2},
    }
    r = preflight_config(cfg)
    assert r["valid"] is False
    assert any(i.startswith("data.val_labels_dir:") for i in r["issues"]), r["issues"]


def test_preflight_config_blocks_an_unreadable_label_in_val_labels_dir(tmp_path):
    """An unreadable label under an explicit validation source aborts the launch (the explicit
    validation build re-raises rather than degrading to no validation), so it blocks here too,
    even though the training labels_dir it sits beside is entirely readable."""
    pytest.importorskip("torch")
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images"
    lbls = tmp_path / "annotations"
    val_imgs = tmp_path / "val_images"
    val_lbls = tmp_path / "val_annotations"
    for d in (imgs, lbls, val_imgs, val_lbls):
        d.mkdir()
    Image.new("RGB", (20, 20)).save(imgs / "ann.jpg")
    Image.new("RGB", (20, 20)).save(val_imgs / "bad.jpg")
    json_io.write_annotations(lbls / "ann.json",
                              [Annotation(subject="bud", geometry=BBox(2, 2, 10, 10))], 20, 20)
    (val_lbls / "bad.json").write_bytes(b"{not json")

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls), "subject": "bud",
                 "val_images_dir": str(val_imgs), "val_labels_dir": str(val_lbls)},
        "training": {"batch_size": 2},
    }
    r = preflight_config(cfg)
    assert r["valid"] is False
    assert any("bad.json" in i for i in r["issues"]), r["issues"]


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


def test_preflight_config_catches_a_training_source_nested_only_under_training(tmp_path):
    """training_source can be nested under training (TrainingSection allows extra keys there),
    and normalize_train_config hoists it onto the top level for the trainer to read; preflight
    must validate the same hoisted view, not only a top-level training_source, or an
    unimportable nested one would pass structural validation and only fail once the run starts."""
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
        "training": {"batch_size": 2, "training_source": "nonexistent_module:train"},
    }
    r = preflight_config(cfg)
    assert any("training_source not importable" in i for i in r["issues"])


# --------------------------------------------------------------------------
# preflight_config's selection_metric coherence: reject a comparability-only
# metric for a center-match trait at validation time, not mid-run.
# --------------------------------------------------------------------------

def test_preflight_config_rejects_incoherent_selection_metric(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images"
    lbls = tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    base_cfg: dict[str, dict[str, object]] = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls)},
        "training": {"batch_size": 2},
    }

    # A comparability-only metric for a center-match trait is rejected.
    cfg = dict(base_cfg)
    cfg["training"] = dict(cfg["training"], evaluation={"trait": "bud_opening", "selection_metric": "map50"})
    r = preflight_config(cfg)
    assert any("comparability-only" in i for i in r["issues"])

    # A governing metric for the same trait is fine.
    cfg["training"] = dict(cfg["training"], evaluation={"trait": "bud_opening", "selection_metric": "f1"})
    assert preflight_config(cfg)["valid"] is True

    # No trait -> no coherence gate, even for a comparability metric.
    cfg["training"] = dict(cfg["training"], evaluation={"selection_metric": "map50"})
    assert preflight_config(cfg)["valid"] is True

    # An undeclared direction is caught here even with no trait at all: it would otherwise
    # surface only as a failed run once resolve_selection_metric runs mid-training.
    cfg["training"] = dict(cfg["training"], evaluation={"selection_metric": "not_a_real_metric"})
    r = preflight_config(cfg)
    assert any("no declared ranking direction" in i for i in r["issues"])


def test_preflight_config_reads_the_top_level_evaluation_block(tmp_path):
    """A config carrying both a top-level and a nested ``training.evaluation`` block with
    different ``selection_metric`` values must validate against the top-level one, since that
    is the block ``normalize_train_config`` hoists to the top and the trainer actually trains
    on."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images"
    lbls = tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    cfg: dict[str, object] = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls)},
        "training": {"batch_size": 2, "evaluation": {"selection_metric": "not_a_real_metric"}},
        "evaluation": {"selection_metric": "f1"},
    }
    r = preflight_config(cfg)
    assert r["valid"] is True
    assert not any("no declared ranking direction" in i for i in r["issues"])


def test_preflight_config_precedence_also_refuses_on_the_top_level_block(tmp_path):
    """Coverage of the chosen precedence's other side: with the placements swapped, the
    top-level block's undeclared metric is what preflight refuses on, not the nested one."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images"
    lbls = tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    cfg: dict[str, object] = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls)},
        "training": {"batch_size": 2, "evaluation": {"selection_metric": "f1"}},
        "evaluation": {"selection_metric": "not_a_real_metric"},
    }
    r = preflight_config(cfg)
    assert any("not_a_real_metric" in i and "no declared ranking direction" in i
                for i in r["issues"])


def test_preflight_config_refuses_a_top_level_stages_entry_missing_epochs(tmp_path):
    """A top-level ``stages`` entry wins over ``training.stages``, the same precedence
    ``train()`` reads under; a top-level stage missing 'epochs' must be refused here, not only
    mid-run, even beside an otherwise-valid nested ``training.stages``."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images"
    lbls = tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    cfg: dict[str, object] = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls)},
        "training": {"batch_size": 2, "stages": [{"epochs": 5}]},
        "stages": [{"freeze_to": 0}],
    }
    r = preflight_config(cfg)
    assert any("Stage 0 missing 'epochs'" in i for i in r["issues"])


def test_preflight_config_accepts_a_nested_only_stages_config(tmp_path):
    """Coverage of the precedence's other side: no top-level ``stages`` entry, so the
    normalized config falls back to ``training.stages``, and a valid nested schedule still
    passes."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images"
    lbls = tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    cfg: dict[str, object] = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls)},
        "training": {"batch_size": 2, "stages": [{"freeze_to": 0, "epochs": 5}]},
    }
    r = preflight_config(cfg)
    assert r["valid"] is True


def test_preflight_config_names_a_non_mapping_evaluation_block_as_an_issue(tmp_path):
    """``TrainingSection``/``TrainConfigSchema`` both allow extra keys of any type, so a
    non-mapping ``evaluation`` block reaches ``eval_cfg.get(...)`` unchecked; it must become a
    named issue, not an ``AttributeError`` that crashes the whole preflight call."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images"
    lbls = tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    cfg: dict[str, object] = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls)},
        "training": {"batch_size": 2},
        "evaluation": "not_a_mapping",
    }
    r = preflight_config(cfg)
    assert any("evaluation" in i and "mapping" in i for i in r["issues"])


def test_a_config_naming_an_unregistered_trait_still_lists(tmp_path, monkeypatch):
    """A run's own row never touches the trait registry: naming a trait this platform's
    registry does not carry (an evaluation.trait config field with no matching spec) must not
    take down the whole run listing, the way a per-row trait lookup used to."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.pipelines.training.run_registry import create_run, list_runs

    run = create_run(
        {"model_source": {"builder": "x:y", "task": "detection"}, "data": {},
         "training": {"evaluation": {"trait": "no_such_trait_here"}}},
        str(tmp_path / "out"),
    )

    # The registry is process-wide, so the listing may hold runs other tests created.
    rows = [row for row in list_runs() if row["run_id"] == run.run_id]
    assert len(rows) == 1
    assert rows[0]["best_metric_name"] is None  # train() never ran, so nothing was stamped yet


# preflight_config's reserve_calibration_fraction feasibility check (N7): a training-launch-time
# refusal through this module's own validation surface, never review_calibration._FAILURE_MESSAGES.

def _reserve_cal_big_single_source(root, width=4000, height=3000, tile_size=128):
    """One large single-image detection source with real width/height, real GT scattered evenly
    across it: enough for a feasible 4-way spatial-strip split."""
    import torch
    from torchvision.utils import save_image

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir, labels_dir = root / "images", root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    stem = "mosaic"
    save_image(torch.rand(3, height, width) * 0.3, str(images_dir / f"{stem}.png"))
    boxes = [Annotation(subject="bud", geometry=BBox(x, y, x + 20, y + 20))
            for x in range(20, width - 20, 200) for y in range(20, height - 20, 200)]
    json_io.write_annotations(str(labels_dir / f"{stem}.json"), boxes, width, height, keep_empty=True)
    return images_dir, labels_dir


def test_preflight_reserve_calibration_fraction_wrong_task_flags_issue(tmp_path):
    from tcip_mcp.tools.training_tools import preflight_config

    images_dir, labels_dir = _reserve_cal_big_single_source(tmp_path / "ds")
    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "classification"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                 "split": {"reserve_calibration_fraction": 0.15}},
        "training": {"batch_size": 2},
    }
    r = preflight_config(cfg)
    assert any("reserve_calibration_fraction" in i and "has no effect" in i for i in r["issues"])


def test_preflight_reserve_calibration_fraction_multi_stem_flags_issue(tmp_path):
    from tcip_mcp.tools.training_tools import preflight_config

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    from PIL import Image
    Image.new("RGB", (32, 32)).save(images_dir / "a.png")
    Image.new("RGB", (32, 32)).save(images_dir / "b.png")
    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                 "tiling": {"enabled": True, "tile_size": 32},
                 "split": {"reserve_calibration_fraction": 0.15}},
        "training": {"batch_size": 2},
    }
    r = preflight_config(cfg)
    assert any("reserve_calibration_fraction" in i and "multi-stem" in i for i in r["issues"])


def test_preflight_reserve_calibration_fraction_infeasible_layout_refuses_under_smoke(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    images_dir, labels_dir = _reserve_cal_big_single_source(tmp_path / "ds")
    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
                 "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
                 # Nothing left for a real train fraction at this mosaic size.
                 "split": {"val_ratio": 0.45, "test_ratio": 0.45, "seed": 1,
                          "reserve_calibration_fraction": 0.3}},
        "training": {"batch_size": 2},
    }
    r = preflight_config(cfg, smoke=True)
    assert any("reserve_calibration_fraction" in i for i in r["issues"]), r["issues"]

    # Without smoke, this specific geometry check doesn't run (needs a real dataset build).
    r_no_smoke = preflight_config(cfg, smoke=False)
    assert not any("reserve_calibration_fraction" in i for i in r_no_smoke["issues"])


def test_preflight_reserve_calibration_fraction_reports_an_unreadable_label_by_name(tmp_path):
    """This probe's own generic except Exception must not swallow an unreadable label into a
    silently-logged build failure: the breeder needs to see which file is broken."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    images_dir, labels_dir = _reserve_cal_big_single_source(tmp_path / "ds")
    bad = labels_dir / "mosaic.json"
    bad.write_bytes(b"{not json")
    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
                 "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
                 "split": {"val_ratio": 0.2, "test_ratio": 0.1, "seed": 1,
                          "reserve_calibration_fraction": 0.15}},
        "training": {"batch_size": 2},
    }
    r = preflight_config(cfg, smoke=True)
    assert any(str(bad) in i for i in r["issues"]), r["issues"]


def test_preflight_reserve_calibration_fraction_admits_a_feasible_layout(tmp_path):
    """The rail-admits-valid-work paired test: a real, feasible reserve_calibration_fraction
    config produces no reserve_calibration_fraction issue under smoke=True."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    images_dir, labels_dir = _reserve_cal_big_single_source(tmp_path / "ds")
    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1, "min_size": 128, "max_size": 256},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
                 "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
                 "split": {"val_ratio": 0.2, "test_ratio": 0.1, "seed": 1,
                          "reserve_calibration_fraction": 0.15}},
        "training": {"batch_size": 2},
    }
    r = preflight_config(cfg, smoke=True)
    assert not any("reserve_calibration_fraction" in i for i in r["issues"]), r["issues"]


# --------------------------------------------------------------------------
# _apply_hpo_params: lr/weight_decay reach what the trainer actually reads
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
    """Sweeping lr must not overwrite the agent's own progressive-unfreeze schedule with a
    hardcoded recipe: base_config's stages (however it expressed them) survive unchanged."""
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
    """backbone_lr must scale by whatever ratio the agent's own base_config expressed, not
    a frozen *0.1: a pinned constant here discards a deliberate agent choice (derive, don't pin)."""
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
    """A swept key outside the known optimizer/batch/weight_decay set must land at the top level
    of the resolved config (where train() reads it), not nested under "training" after
    normalize_train_config's hoist already ran, since nesting there would leave it silently
    unreachable."""
    from tcip_mcp.tools.training_tools import _apply_hpo_params

    base = {"model_source": {"builder": "x:y", "task": "detection"}}
    out = _apply_hpo_params(base, {"momentum": 0.9})
    assert out["momentum"] == 0.9
    assert "momentum" not in out.get("training", {})


def test_apply_hpo_params_dotted_key_reaches_nested_field():
    """A dotted param (e.g. a swept builder) reaches the nested field it names."""
    from tcip_mcp.tools.training_tools import _apply_hpo_params

    base = {"model_source": {"builder": "old:builder", "task": "detection"}}
    out = _apply_hpo_params(base, {"model_source.builder": "new:builder"})
    assert out["model_source"]["builder"] == "new:builder"
    assert out["model_source"]["task"] == "detection"  # the rest of the mapping survives


def test_apply_hpo_params_refuses_a_dotted_key_through_a_non_mapping_intermediate():
    """A dotted key whose path walks through a value that is not a mapping is refused by name,
    naming the key and what was found there, rather than raising an opaque AttributeError."""
    from tcip_mcp.tools.training_tools import _apply_hpo_params

    base = {"model_source": "not-a-mapping"}
    with pytest.raises(ValueError, match="model_source.builder"):
        _apply_hpo_params(base, {"model_source.builder": "x:y"})


def test_preflight_points_covers_every_categorical_choice_and_both_numeric_bounds():
    """The preflight must check the whole search space, not only the first sampled corner: one
    point per categorical choice, and one point per numeric bound (low and high)."""
    from tcip_mcp.tools.training_tools import _preflight_points

    space = {
        "model_source.builder": {"type": "categorical", "choices": ["a:b", "c:d", "e:f"]},
        "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
    }
    points = _preflight_points(space)

    builder_values = {p["model_source.builder"] for _, p in points if "model_source.builder" in p}
    assert builder_values == {"a:b", "c:d", "e:f"}
    lr_values = {p["lr"] for label, p in points if "lr" in p and "lr" in label}
    assert lr_values == {1e-5, 1e-2}


# --------------------------------------------------------------------------
# _run_hpo_trial: reports the composite (lower=better) each epoch + final, with
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
    from tcip_mcp.pipelines.data import split_construction as sc
    from tcip_mcp.pipelines.training import generic_trainer as gt

    ds = _FakeDataset()

    def fake_auto_train_val(task, data_cfg, transforms):
        if captured is not None:
            captured["transforms"] = transforms
        return ds, ds, None

    monkeypatch.setattr(sc, "auto_train_val", fake_auto_train_val)
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
    """Once a center-match trait sets which key governs checkpoint choice ('selection'),
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
    """A crashed trial (or one with no model_source) reports +inf, the worst value under
    mode='min', so it can never become the sweep's best."""
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


def test_run_hpo_trial_reports_the_highest_value_for_a_higher_is_better_metric(monkeypatch, tmp_path):
    """A higher-is-better selection metric (accuracy) must report the trial's best epoch as the
    highest reported value, not a minimize convention that would instead prefer the lowest."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import _run_hpo_trial

    def fake_train(run, train_loader, val_loader, task="classification",
                   epoch_callback=None, resume_from=""):
        for epoch, value in enumerate([0.5, 0.9, 0.6]):
            if epoch_callback:
                epoch_callback(epoch, {"selection": value})
        # run.best_metric is left at its dataclass default (+inf), as a bespoke loop that never
        # sets it would leave it; the trial's own tracking must supply the real final value.
        run.status = "completed"
        return run

    _patch_hpo_trial_machinery(monkeypatch, fake_train)
    base = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_classifier",
                         "builder_kwargs": {"num_classes": 2}, "task": "classification"},
        "data": {"images_dir": "imgs"},
        "training": {"batch_size": 2},
        "evaluation": {"selection_metric": "accuracy"},
    }
    reported: list = []
    _run_hpo_trial({"lr": 3e-4}, reported.append, base, str(tmp_path / "trial_0"))
    assert reported == [0.5, 0.9, 0.6, 0.9]  # per-epoch trace, then the highest, not +inf


def test_a_trial_with_no_metric_never_outranks_a_real_one_under_a_maximize_direction(
    monkeypatch, tmp_path,
):
    """A trial that never reports a real value must report the losing side of the metric's own
    direction (here, -inf, since accuracy is higher-is-better), so it can never be mistaken for
    the sweep's best trial even under a maximize (mode='max') sweep."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import _run_hpo_trial

    base = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_classifier",
                         "builder_kwargs": {"num_classes": 2}, "task": "classification"},
        "data": {"images_dir": "imgs"},
        "training": {"batch_size": 2},
        "evaluation": {"selection_metric": "accuracy"},
    }

    def fake_train_ok(run, train_loader, val_loader, task="classification",
                      epoch_callback=None, resume_from=""):
        if epoch_callback:
            epoch_callback(0, {"selection": 0.7})
        run.status = "completed"
        return run

    _patch_hpo_trial_machinery(monkeypatch, fake_train_ok)
    real: list = []
    _run_hpo_trial({"lr": 3e-4}, real.append, base, str(tmp_path / "trial_real"))

    def fake_train_fails(run, train_loader, val_loader, task="classification",
                         epoch_callback=None, resume_from=""):
        raise RuntimeError("boom")

    _patch_hpo_trial_machinery(monkeypatch, fake_train_fails)
    failed: list = []
    _run_hpo_trial({"lr": 3e-4}, failed.append, base, str(tmp_path / "trial_failed"))

    assert failed == [float("-inf")]
    assert failed[-1] < real[-1]  # strictly loses under mode='max' too


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
    """A swept key the training body never reads is surfaced by observation, not
    gated by a whitelist. resolved_config.json records which swept keys went unconsumed."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import _run_hpo_trial, trial_config_key

    def fake_train(run, train_loader, val_loader, task="detection",
                   epoch_callback=None, resume_from=""):
        run.config.get("lr")  # a known key, consumed -- but "totally_bogus_key" never read
        run.best_metric = 1.0
        run.status = "completed"
        return run

    _patch_hpo_trial_machinery(monkeypatch, fake_train)
    trial_dir = tmp_path / "trial_0"
    _run_hpo_trial({"lr": 3e-4, "totally_bogus_key": 5}, [].append, _detection_base(), str(trial_dir))

    resolved = ts.read(trial_config_key(trial_dir.parent, trial_dir.name))
    assert resolved["unconsumed_params"] == ["totally_bogus_key"]


def test_run_hpo_trial_resolved_config_records_seed_actually_trained_under(monkeypatch, tmp_path):
    """resolved_config.json's seed must match the seed the run actually trained under.
    An unset seed is drawn fresh onto the run's own config, never copied back onto the
    pre-run merged config the persisted record is built from."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import _run_hpo_trial, trial_config_key

    captured: dict = {}

    def fake_train(run, train_loader, val_loader, task="detection",
                   epoch_callback=None, resume_from=""):
        captured["seed"] = run.config.get("seed")
        run.best_metric = 1.0
        run.status = "completed"
        return run

    _patch_hpo_trial_machinery(monkeypatch, fake_train)
    trial_dir = tmp_path / "trial_0"
    _run_hpo_trial({"lr": 3e-4}, [].append, _detection_base(), str(trial_dir))

    resolved = ts.read(trial_config_key(trial_dir.parent, trial_dir.name))
    assert resolved["seed"] is not None
    assert resolved["seed"] == captured["seed"]


def _bespoke_hpo_agent_train(ctx):
    """Module-level (dotted-import-able) bespoke train(ctx) that reads its own swept key."""
    ctx.config.get("custom_axis")  # the bespoke loop reads its own swept key
    ctx.run.status = "completed"
    ctx.run.best_metric = 1.0


def test_run_hpo_trial_bespoke_custom_key_not_falsely_flagged_unconsumed(monkeypatch, tmp_path):
    """A bespoke training_source reading its own swept custom key must not be
    falsely flagged unconsumed merely because generic_trainer.train() doesn't know it:
    tracking is genuine runtime access, not a static comparison against train()'s key list."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import _run_hpo_trial, trial_config_key

    from tcip_mcp.pipelines.data import split_construction as sc
    monkeypatch.setattr(sc, "auto_train_val",
                        lambda task, data_cfg, transforms: (_FakeDataset(), _FakeDataset(), None))
    import torch.utils.data as tud
    monkeypatch.setattr(tud, "DataLoader", lambda *a, **k: object())
    from tcip_mcp.pipelines.data import samplers
    monkeypatch.setattr(samplers, "build_sampler", lambda *a, **k: None)

    base = {"model_source": {"builder": "x:y", "task": "detection"},
            "training_source": f"{__name__}:_bespoke_hpo_agent_train"}
    trial_dir = tmp_path / "trial_0"
    _run_hpo_trial({"custom_axis": 42}, [].append, base, str(trial_dir))

    resolved = ts.read(trial_config_key(trial_dir.parent, trial_dir.name))
    assert resolved["unconsumed_params"] == []


def test_run_hpo_trial_swept_top_level_evaluation_is_not_reported_unconsumed(monkeypatch, tmp_path):
    """A swept top-level ``evaluation`` axis must be seen as read: the fake training body reads
    it the same way ``generic_trainer.train()`` does, through ``evaluation_section``, which must
    read ``run.config`` (the access-tracking wrapper) directly rather than through a
    ``dict(config)`` copy that would bypass the wrapper's own ``get`` override."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import _run_hpo_trial, trial_config_key
    from tcip_mcp.pipelines.schemas import evaluation_section

    def fake_train(run, train_loader, val_loader, task="detection",
                   epoch_callback=None, resume_from=""):
        evaluation_section(run.config)  # the same read generic_trainer.train() performs
        run.best_metric = 1.0
        run.status = "completed"
        return run

    _patch_hpo_trial_machinery(monkeypatch, fake_train)
    trial_dir = tmp_path / "trial_0"
    _run_hpo_trial({"lr": 3e-4, "evaluation": {"selection_metric": "f1"}}, [].append,
                   _detection_base(), str(trial_dir))

    resolved = ts.read(trial_config_key(trial_dir.parent, trial_dir.name))
    assert "evaluation" not in resolved["unconsumed_params"]


def test_run_hpo_trial_swept_dotted_selection_metric_is_not_reported_unconsumed(monkeypatch, tmp_path):
    """A dotted swept key (``evaluation.selection_metric``) is applied by ``_apply_hpo_params``
    into the nested ``evaluation`` field it names, never as a literal top-level key, so the
    dotted string itself is never read; it must count as consumed when the top-level segment
    it lands under (``evaluation``) was read."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import _run_hpo_trial, trial_config_key
    from tcip_mcp.pipelines.schemas import evaluation_section

    def fake_train(run, train_loader, val_loader, task="detection",
                   epoch_callback=None, resume_from=""):
        evaluation_section(run.config)  # the same read generic_trainer.train() performs
        run.best_metric = 1.0
        run.status = "completed"
        return run

    _patch_hpo_trial_machinery(monkeypatch, fake_train)
    trial_dir = tmp_path / "trial_0"
    _run_hpo_trial({"lr": 3e-4, "evaluation.selection_metric": "f1"}, [].append,
                   _detection_base(), str(trial_dir))

    resolved = ts.read(trial_config_key(trial_dir.parent, trial_dir.name))
    assert "evaluation.selection_metric" not in resolved["unconsumed_params"]


def test_run_hpo_trial_swept_dotted_key_whose_segment_is_unread_is_reported_unconsumed(
    monkeypatch, tmp_path,
):
    """A dotted swept key still counts as unconsumed when nothing reads the top-level segment
    it lands under, the same as a plain top-level key nothing reads. Coverage: this already
    passes before and after the axis-conflict fix, since a leaf mismatch under a read block is
    a separate rule from consumption (see ``_AccessTrackingConfig``'s own docstring)."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import _run_hpo_trial, trial_config_key

    def fake_train(run, train_loader, val_loader, task="detection",
                   epoch_callback=None, resume_from=""):
        run.config.get("lr")  # a known key, consumed; "model_source" itself is never read
        run.best_metric = 1.0
        run.status = "completed"
        return run

    _patch_hpo_trial_machinery(monkeypatch, fake_train)
    trial_dir = tmp_path / "trial_0"
    _run_hpo_trial({"lr": 3e-4, "model_source.builder_kwargs.width": 32}, [].append,
                   _detection_base(), str(trial_dir))

    resolved = ts.read(trial_config_key(trial_dir.parent, trial_dir.name))
    assert resolved["unconsumed_params"] == ["model_source.builder_kwargs.width"]


def test_run_hpo_trial_diverged_run_never_outranks_a_worse_but_alive_config(tmp_path):
    """A trial that trains one real epoch and then diverges must report the losing side as its
    final value, not that epoch's real score, so it can never outrank a config that only scored
    worse. Drives the real training body (nothing mocked)."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import _run_hpo_trial
    from tests.tiny_trainer_fixtures import write_regression_dataset

    images_dir, csv_path = write_regression_dataset(
        tmp_path, intensities=[0.1, 0.3, 0.5, 0.7], values=[0.2, 0.6, 1.0, 1.4])

    base_config = {
        "model_source": {"builder": "tests.tiny_trainer_fixtures:build_diverges_after_model",
                         "builder_kwargs": {"good_calls": 1}, "task": "regression", "in_chans": 3},
        "data": {"images_dir": str(images_dir), "csv_path": str(csv_path), "task": "regression"},
        "device": "cpu",
        "mixed_precision": False,
        "stages": [{"freeze_to": 0, "epochs": 5}],
        "optimizer": {"name": "adamw", "backbone_lr": 0.05, "head_lr": 0.05, "weight_decay": 0.0},
        "checkpoint_every_n_epochs": 0,
        "early_stopping": {"enabled": False},
    }
    reported: list = []
    _run_hpo_trial({}, reported.append, base_config, str(tmp_path / "trial_0"))

    import math
    assert math.isfinite(reported[0])  # epoch 1's real score, reported before the run died
    assert reported[-1] == float("inf")  # the losing side of 'loss' (lower=better)


# --------------------------------------------------------------------------
# get_worst_predictions: confidence comes from the canonical prediction format
# --------------------------------------------------------------------------

def test_get_worst_predictions_reads_canonical_confidence(tmp_path, monkeypatch):
    """Prediction files are per-image JSON with a native ``score`` (json_io); confidence
    reads from that field, not from box geometry. Reading a normalized box height as confidence
    instead would make the (1 - avg_conf) ranking term ~1.0 for every image with small boxes
    (e.g. buds)."""
    pytest.importorskip("torch")
    monkeypatch.chdir(tmp_path)
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.vision_tools import get_worst_predictions

    preds = tmp_path / "preds"
    gts = tmp_path / "labels"
    preds.mkdir()
    gts.mkdir()

    def write_image(stem: str, scores: list[float]) -> None:
        # Confidence lives in the JSON `score`; box geometry is irrelevant to this
        # count + confidence heuristic (no IoU matching), so the boxes can be anything.
        pred_anns = [Annotation(subject="bud", geometry=BBox(10.0, 10.0, 40.0, 22.0), score=s)
                     for s in scores]
        json_io.write_annotations(str(preds / f"{stem}.json"), pred_anns, 100, 100)
        # Matching GT count → missed = extra = 0, error is exactly (1 - avg_conf).
        gt_anns = [Annotation(subject="bud", geometry=BBox(20.0, 11.0, 40.0, 31.0)) for _ in scores]
        json_io.write_annotations(str(gts / f"{stem}.json"), gt_anns, 100, 100)

    write_image("confident", [0.9, 0.9])
    write_image("shaky", [0.1, 0.1])

    out = get_worst_predictions(str(preds), str(gts), top_k=2)
    by_stem = {w["stem"]: w["error_score"] for w in out["worst_images"]}
    assert by_stem["confident"] == pytest.approx(0.1, abs=1e-3)
    assert by_stem["shaky"] == pytest.approx(0.9, abs=1e-3)
    assert out["worst_images"][0]["stem"] == "shaky"  # low confidence ranks worst


# --------------------------------------------------------------------------
# _ensure_experiment: experiments are immutable on relaunch
# --------------------------------------------------------------------------

def test_ensure_experiment_mints_fresh_id_instead_of_mutating(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # .tcip/experiments lives under cwd
    from tcip_mcp.audit import audit_log_key
    from tcip_mcp.experiments import (
        config_key, create_experiment, lineage_key, log_metrics, read_metrics, status_key,
        update_status,
    )
    from tcip_mcp.tools.training_tools import _ensure_experiment

    # A completed experiment with recorded history.
    create_experiment("exp1", {"a": 1}, data_source="imgs_v1")
    update_status("exp1", "running")
    log_metrics("exp1", 1, {"map50": 0.7})
    update_status("exp1", "completed")
    status_before = ts.read(status_key("exp1"))
    metrics_before = read_metrics("exp1")

    # Relaunching with the same experiment_id must not reuse it.
    eid = _ensure_experiment("exp1", {"a": 2}, "imgs_v2", resume_from="", run_id="run_9_0",
                             output_dir="out")
    assert eid == "exp1_run_9_0"
    assert ts.read(status_key("exp1")) == status_before      # untouched
    assert read_metrics("exp1") == metrics_before             # untouched
    assert ts.read(config_key("exp1")) == {"a": 1}

    # The fresh experiment exists and points back at the original.
    lineage = ts.read(lineage_key("exp1_run_9_0"))
    assert lineage["parent_experiment"] == "exp1"
    assert lineage["data_source"] == "imgs_v2"

    # An ordinary relaunch refuses nothing: overwrite_config_if_pristine is never even attempted
    # against a non-pristine id, so no experiment_mutation_refused line is appended for it.
    events = ts.read_log(audit_log_key(tmp_path)).records
    assert not [e for e in events if e.get("tool") == "experiment_mutation_refused"]


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
    """A pristine pre-created experiment's config.json is refreshed with the config actually
    launched (tiling/seed resolved after create_experiment ran), not left describing the config
    as it stood before those were resolved."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import config_key, create_experiment
    from tcip_mcp.tools.training_tools import _ensure_experiment

    create_experiment("pre", {"a": 1})
    effective_config = {"a": 1, "data": {"tiling": {"tile_size": 512}}, "seed": 99}
    assert _ensure_experiment("pre", effective_config, None, resume_from="", run_id="r1",
                              output_dir="out") == "pre"

    config = ts.read(config_key("pre"))
    assert config == effective_config


def test_ensure_experiment_resume_into_populated_id_mints_fresh_parented_id(tmp_path, monkeypatch):
    """resume_from must not reuse an id that already has recorded history: it mints a fresh
    parented id instead, matching the non-resume collision behavior. Reusing it would discard the
    resumed run's own metrics/lineage writes (refused by the terminal-state lock) and let
    ModelRegistry.register_model replace the original's registry entry by name with no record of
    what was superseded."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, lineage_key, log_metrics, update_status
    from tcip_mcp.tools.training_tools import _ensure_experiment

    create_experiment("res", {"a": 1})
    update_status("res", "running")
    log_metrics("res", 1, {"loss": 0.5})
    eid = _ensure_experiment("res", {"a": 1}, None, resume_from="ckpt/checkpoint_epoch_5.pt",
                             run_id="r2", output_dir="out")
    assert eid == "res_r2"

    lineage = ts.read(lineage_key("res_r2"))
    assert lineage["parent_experiment"] == "res"


def test_ensure_experiment_resume_into_pristine_id_still_attaches(tmp_path, monkeypatch):
    """A resume_from target that is itself still pristine (never actually run) is unaffected:
    pristine reuse doesn't depend on resume_from at all."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.training_tools import _ensure_experiment

    create_experiment("pre2", {"a": 1})
    eid = _ensure_experiment("pre2", {"a": 1}, None,
                             resume_from="ckpt/checkpoint_epoch_5.pt", run_id="r9",
                             output_dir="out")
    assert eid == "pre2"


def test_a_launch_config_that_json_cannot_hold_is_refused_before_the_run_starts(
    tmp_path, monkeypatch
):
    """The caller's config is stored twice, as the launch config and as the experiment's
    snapshot, so the field that will not encode is named before either write and before a
    subprocess is spawned for a run whose provenance could not be recorded."""
    from tcip_mcp.tools import training_tools

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(training_tools, "preflight_config",
                        lambda config, smoke=False: {"valid": False, "issues": ["stub"]})

    with pytest.raises(TypeError) as refused:
        training_tools.launch_training({"model_source": {"builder": Path("m.py")}})
    assert "config.model_source.builder" in str(refused.value)


def test_an_ordinary_launch_config_passes_the_boundary_to_preflight(tmp_path, monkeypatch):
    """The refusal above must not stop a legitimate config: it reaches preflight and comes
    back with preflight's own verdict rather than a refusal from the boundary check."""
    from tcip_mcp.tools import training_tools

    monkeypatch.chdir(tmp_path)
    seen = []

    def stub_preflight(config, smoke=False, overfit=False):
        seen.append(config)
        return {"valid": False, "issues": ["stub"]}

    monkeypatch.setattr(training_tools, "preflight_config", stub_preflight)

    result = training_tools.launch_training({"model_source": {"builder": "m:f"}})

    assert result == {"error": "Invalid config", "issues": ["stub"]}
    assert seen == [{"model_source": {"builder": "m:f"}}]


def test_a_sweep_payload_that_json_cannot_hold_is_refused_before_any_trial_runs(
    tmp_path, monkeypatch
):
    """The space reaches the sweep manifest and the base config reaches every trial's resolved
    config, so both are checked before a single trial is trained against them."""
    from tcip_mcp.pipelines.training import hpo
    from tcip_mcp.tools import training_tools

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hpo, "tune_search", lambda *args, **kwargs: {"best_params": {},
                                                                    "best_value": 0.1,
                                                                    "n_trials": 1})

    with pytest.raises(TypeError) as space_refused:
        training_tools.run_hyperparameter_search({"model_source": {"builder": "m:f"}},
                               param_space={"lr": Path("lr.txt")})
    assert "param_space.lr" in str(space_refused.value)

    with pytest.raises(TypeError) as config_refused:
        training_tools.run_hyperparameter_search({"model_source": {"builder": Path("m.py")}},
                               param_space={"lr": [0.1, 0.01]})
    assert "base_config.model_source.builder" in str(config_refused.value)


def test_an_ordinary_sweep_payload_still_runs_its_search(tmp_path, monkeypatch):
    """The refusal above must not cost a legitimate sweep its search: admits valid work through
    the sweep door's structural preflight (an importable builder, a real data section)."""
    from tcip_mcp.pipelines.training import hpo
    from tcip_mcp.tools import training_tools

    monkeypatch.chdir(tmp_path)
    seen = []

    def fake_search(*args, **kwargs):
        seen.append(kwargs.get("param_space", args[1] if len(args) > 1 else None))
        return {"best_params": {"lr": 0.01}, "best_value": 0.1, "n_trials": 1}

    monkeypatch.setattr(hpo, "tune_search", fake_search)

    imgs, lbls = tmp_path / "images", tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    base_config = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls)},
    }
    result = training_tools.run_hyperparameter_search(base_config, param_space={"lr": [0.1, 0.01]}, n_trials=1)

    assert result["best_params"] == {"lr": 0.01}
    assert seen == [{"lr": [0.1, 0.01]}]


def test_run_hyperparameter_search_refuses_a_param_space_axis_naming_the_dotted_selection_metric(
    tmp_path, monkeypatch,
):
    """run_hyperparameter_search fixes the sweep's selection metric and direction once from base_config and
    reuses it for every trial; a param_space axis naming ``evaluation.selection_metric``
    directly would let a trial's own resolution disagree, so it is refused by name before
    the sweep is minted."""
    from tcip_mcp.tools import training_tools

    monkeypatch.chdir(tmp_path)
    result = training_tools.run_hyperparameter_search(
        {"model_source": {"builder": "m:f"}},
        param_space={"evaluation.selection_metric": {
            "type": "categorical", "choices": ["map", "iou_mean"],
        }},
    )
    assert "error" in result
    assert "evaluation.selection_metric" in result["error"]


def test_run_hyperparameter_search_refuses_a_param_space_axis_whose_choices_carry_selection_metric(
    tmp_path, monkeypatch,
):
    """The same refusal catches an ``evaluation`` axis whose sampled value is itself a dict
    naming ``selection_metric``, not only the dotted-key form."""
    from tcip_mcp.tools import training_tools

    monkeypatch.chdir(tmp_path)
    result = training_tools.run_hyperparameter_search(
        {"model_source": {"builder": "m:f"}},
        param_space={"evaluation": {
            "type": "categorical",
            "choices": [{"selection_metric": "map"}, {"selection_metric": "iou_mean"}],
        }},
    )
    assert "error" in result
    assert "evaluation" in result["error"]


def test_run_hyperparameter_search_admits_an_lr_sweep_beside_a_base_config_selection_metric(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """The selection-metric refusal targets param_space, never base_config: a config that
    states its own selection metric still runs an ordinary lr sweep."""
    from tcip_mcp.pipelines.training import hpo
    from tcip_mcp.tools import training_tools

    monkeypatch.chdir(tmp_path)
    seen = []

    def fake_search(*args, **kwargs):
        seen.append(kwargs.get("param_space", args[1] if len(args) > 1 else None))
        return {"best_params": {"lr": 0.01}, "best_value": 0.1, "n_trials": 1}

    monkeypatch.setattr(hpo, "tune_search", fake_search)

    base_config = {**real_hpo_base_config, "evaluation": {"selection_metric": "map"}}
    result = training_tools.run_hyperparameter_search(base_config, param_space={"lr": [0.1, 0.01]}, n_trials=1)

    assert result["best_params"] == {"lr": 0.01}
    assert seen == [{"lr": [0.1, 0.01]}]


def test_run_hyperparameter_search_refuses_a_param_space_axis_that_changes_the_metrics_own_task_default(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """A param_space axis with no selection_metric key anywhere can still split the sweep's
    fixed metric from a trial's own: model_source.task changes resolve_selection_metric's
    task-derived default (objective for detection, loss otherwise), so a categorical task axis
    is refused exactly like a dotted or nested-dict selection_metric axis."""
    from tcip_mcp.tools import training_tools

    monkeypatch.chdir(tmp_path)
    result = training_tools.run_hyperparameter_search(
        real_hpo_base_config,
        param_space={"model_source.task": {
            "type": "categorical", "choices": ["detection", "classification"],
        }},
    )
    assert "error" in result
    assert "model_source.task" in result["error"]


def test_run_hyperparameter_search_admits_a_categorical_evaluation_axis_naming_the_same_metric_at_every_choice(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """The axis-conflict check resolves each point's own selection metric rather than refusing
    any evaluation-shaped axis outright: a categorical evaluation axis whose every choice
    resolves to the same metric as base_config's own is admitted."""
    from tcip_mcp.pipelines.training import hpo
    from tcip_mcp.tools import training_tools

    monkeypatch.chdir(tmp_path)
    seen = []

    def fake_search(*args, **kwargs):
        seen.append(kwargs.get("param_space", args[1] if len(args) > 1 else None))
        return {"best_params": {}, "best_value": 0.1, "n_trials": 1}

    monkeypatch.setattr(hpo, "tune_search", fake_search)

    base_config = {**real_hpo_base_config, "evaluation": {"selection_metric": "map"}}
    param_space = {"evaluation": {
        "type": "categorical",
        "choices": [{"selection_metric": "map"}, {"selection_metric": "map"}],
    }}
    result = training_tools.run_hyperparameter_search(base_config, param_space=param_space, n_trials=1)

    assert "error" not in result, result
    assert seen == [param_space]


# dataset_identity: a version-refused identity propagates rather than reading as unregistered.

def test_dataset_identity_propagates_a_version_refused_identity(tmp_path, monkeypatch):
    """``except ValueError: ds_id = None`` must not swallow a version refusal identically to
    not-registered: the two are different facts, and this call's own caller already wraps it in a
    best-effort ``except Exception`` that logs and continues the run, so propagating here
    surfaces the fact rather than silently recording ``(None, fp)``."""
    import tcip_store as ts
    from tcip_store import SchemaVersionRefused

    from tcip_mcp.dataset_layout import dataset_identity_key
    from tcip_mcp.pipelines.data.split_construction import dataset_identity

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    key = dataset_identity_key(tmp_path)
    document = {"crop": "chestnut", "id": "abc123", "fingerprint": "v1:deadbeef",
                "schema_version": 2}
    ts.put_blob(key, ts.RECORD_JSON.encode(document))

    with pytest.raises(SchemaVersionRefused):
        dataset_identity({"images_dir": str(images_dir)})


def test_dataset_identity_tolerates_a_genuinely_unregistered_dataset(tmp_path, monkeypatch):
    """The admitting half: no identity document at all still reads as (None, fp), not a refusal."""
    from tcip_mcp.pipelines.data.split_construction import dataset_identity

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    ds_id, fp = dataset_identity({"images_dir": str(images_dir)})
    assert ds_id is None


def test_list_launchable_configs_state_agrees_with_the_runs_list_for_a_crashed_record(
    tmp_path, monkeypatch
) -> None:
    """A launched-but-heartbeat-stale record reads 'interrupted' here the identical way
    list_experiments(launched_only=True)'s own derivation reads it (the runs list beside this
    picker); a
    never-launched pristine record reads its recorded 'created', not a heartbeat-derived
    guess implying a crash that never happened."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    import tcip_store
    from tcip_mcp.experiments import create_experiment, status_key
    from tcip_mcp.tools.training_tools import list_launchable_configs

    create_experiment("exp-crashed", {"model_source": {"builder": "m:f", "task": "detection"},
                                      "data": {"images_dir": "/d", "subject": "bud"}})
    with tcip_store.transaction(status_key("exp-crashed")) as txn:
        txn.write(status_key("exp-crashed"), {"state": "running", "started": None, "ended": None})

    create_experiment("exp-pristine", {"model_source": {"builder": "m:f", "task": "detection"},
                                       "data": {"images_dir": "/d"}})

    rows = {r["experiment_id"]: r for r in list_launchable_configs()}
    assert rows["exp-crashed"]["state"] == "interrupted"
    assert rows["exp-pristine"]["state"] == "created"


def test_cancel_end_to_end_through_the_real_trainer_ends_cancelled_with_records_and_losing_side(
    tmp_path
) -> None:
    """A trial that trains one real epoch and is then cancelled mid-training (the run-level
    sentinel written only once a genuine score is already on record) ends cancelled, still
    writes its resolved-config record, and reports the losing side as its final value, not
    that epoch's real score, so a cancelled trial can never outrank one that merely scored
    worse."""
    pytest.importorskip("torch")
    import math

    import tcip_store
    from tcip_mcp.pipelines.training.run_registry import CANCEL_SENTINEL
    from tcip_mcp.tools.training_tools import _run_hpo_trial, trial_config_key
    from tests.tiny_trainer_fixtures import write_regression_dataset

    images_dir, csv_path = write_regression_dataset(
        tmp_path, intensities=[0.1, 0.3, 0.5, 0.7], values=[0.2, 0.6, 1.0, 1.4])
    labels_dir = tmp_path / "unused_labels"
    labels_dir.mkdir()
    base_config = {
        "model_source": {"builder": "tests.tiny_trainer_fixtures:build_mean_intensity_regressor",
                         "task": "regression", "in_chans": 3},
        "data": {"images_dir": str(images_dir), "csv_path": str(csv_path), "labels_dir": str(labels_dir)},
        "training": {"batch_size": 2, "stages": [{"freeze_to": 0, "epochs": 5}],
                     "mixed_precision": False, "device": "cpu",
                     "checkpoint_every_n_epochs": 0, "early_stopping": {"enabled": False}},
    }
    trial_dir = tmp_path / "sweep" / "trial_cancel01"
    trial_dir.mkdir(parents=True)

    reported: list = []

    def report(value: float) -> None:
        # The sentinel is written only after the first real report, so a genuine score is
        # already on record by the time the run-level cancel takes effect mid-training.
        reported.append(value)
        if len(reported) == 1:
            (trial_dir / CANCEL_SENTINEL).touch()

    _run_hpo_trial({}, report, base_config, str(trial_dir))

    assert math.isfinite(reported[0])  # epoch 1's real score, reported before the cancel
    assert reported[-1] == float("inf")  # the losing side, not that real score
    resolved = tcip_store.read(trial_config_key(trial_dir.parent, trial_dir.name))
    assert resolved["trial_params"] == {}
