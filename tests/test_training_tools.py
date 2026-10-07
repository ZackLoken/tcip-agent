"""Tests for training_tools: validator/StageSpec alignment, HPO param plumbing, HPO trial
reporting (per-epoch trace + failed-trial sentinels), HPO/train regime parity, experiment
immutability on relaunch, and canonical-format confidence parsing in get_worst_predictions."""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

import json
from pathlib import Path

import pytest

# No built-in traits: seed_bud_trait_spec (conftest.py) writes a real bud.yml into this
# test's pinned platform state root so trait="bud_opening" call sites keep resolving.
pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")


def _labeled(tmp_path: Path) -> dict:
    """A data section over two labeled frames of ``bud`` under ``tmp_path``
    (``_verified_checkpoint_fixtures.detection_images``), drawn at seed 0."""
    from tests._verified_checkpoint_fixtures import detection_images

    scope = {"subject": "bud"}
    return {**detection_images(tmp_path / "labeled", scope), "scope": scope,
            "split": {"seed": 0, "val_ratio": 0.15}}


def _bud_image(image: Path, *, labeled: bool = True) -> None:
    """A 20x20 frame at ``image`` and, unless ``labeled`` is false, its label document holding
    one ``bud`` box."""
    from PIL import Image
    from tcip_annotation.state import Annotation, BBox

    from tests._producer_fixtures import label_image

    image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20)).save(image)
    if labeled:
        label_image(image, [Annotation(subject="bud", geometry=BBox(2, 2, 10, 10))], 20, 20)


def _damaged(image: Path, stored: bytes) -> None:
    """``image``'s label document replaced in the store by ``stored``."""
    from tests._producer_fixtures import image_label_key
    from tests._record_damage_fixtures import damage_record

    damage_record(image_label_key(image), stored)


# --------------------------------------------------------------------------
# preflight_config: per-stage 'lr' is optional (trainer never reads it)
# --------------------------------------------------------------------------

def test_preflight_config_accepts_trainer_canonical_stages(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "task": "detection"},
        "data": _labeled(tmp_path),
        # launch_training's own default stage shape: freeze_to + epochs, no lr.
        "batch_size": 2,
        "stages": [{"freeze_to": -1, "epochs": 5}, {"freeze_to": 2, "epochs": 10}],
    }
    r = preflight_config(tmp_path, cfg)
    assert r["valid"] is True, r["issues"]

    # 'epochs' is still required per provided stage.
    cfg["stages"] = [{"freeze_to": 0}]
    r2 = preflight_config(tmp_path, cfg)
    assert any(i.startswith("stages.0.epochs:") for i in r2["issues"]), r2["issues"]

    # No stages at all is fine: launch_training supplies its own default schedule.
    del cfg["stages"]
    assert preflight_config(tmp_path, cfg)["valid"] is True


def test_preflight_config_refuses_a_nested_training_section_by_name(tmp_path):
    """The config has one placement: a ``training`` section is refused naming the move, since
    every key under it would be read by nothing and the run would train at the trainer's own
    defaults in silence."""
    from tcip_mcp.tools.training_tools import preflight_config

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "task": "detection"},
        "data": _labeled(tmp_path),
    }
    nested_keys = {"batch_size": 2, "stages": [{"freeze_to": -1, "epochs": 5}]}
    r = preflight_config(tmp_path, {**cfg, "training": nested_keys})
    assert r["valid"] is False
    assert any("'training' is not a config section" in i for i in r["issues"]), r["issues"]

    assert preflight_config(tmp_path, {**cfg, **nested_keys})["valid"] is True


def test_a_smoke_preflight_validates_its_config_once(tmp_path, monkeypatch):
    """A preflight that admits a config, from its structural check through its resolution and
    its smoke build, reads it off one validation and never validates its model source again."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config
    from tests.tiny_trainer_fixtures import count_validations, write_regression_dataset

    intensities = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]
    images_dir, csv_path = write_regression_dataset(
        tmp_path / "ds", intensities, [2.0 * c for c in intensities])
    config = {"model_source": {"builder": "tests.tiny_trainer_fixtures:build_two_rate_regressor",
                               "task": "regression"},
              "data": {"num_channels": 1, "scope": {}, "images_dir": str(images_dir),
                       "labels_dir": str(csv_path), "split": {"seed": 1, "val_ratio": 0.15}},
              "device": "cpu", "mixed_precision": False}

    validations = count_validations(monkeypatch)
    report = preflight_config(tmp_path, config, smoke=True)
    assert report["valid"] is True, report["issues"]
    assert validations == ["TrainConfigSchema"]


def test_preflight_config_types_a_non_dict_data_section_instead_of_raising(tmp_path):
    from tcip_mcp.tools.training_tools import preflight_config

    r = preflight_config(tmp_path, {"data": "x"})
    assert r["valid"] is False
    assert any(i.startswith("data:") and "dictionary" in i for i in r["issues"]), r["issues"]


# preflight_config's overfit branch: reseed-run-restore, never gating

def _frozen_detection_builder(**kwargs):
    """An importable builder whose model has nothing to optimize (a legitimate stage-0 shape).
    Forwards its kwargs so a caller's ``detector`` choice (``fcos``, say) actually lands."""
    from tests import bespoke_models

    model = bespoke_models.build_bespoke_detection(**kwargs)
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
    return {
        "model_source": {"builder": builder,
                         "builder_kwargs": {"min_size": 64, "max_size": 96, "detector": "fcos"},
                         "task": "detection"},
        "data": {**_labeled(tmp_path), "num_channels": 3},
        "batch_size": 2,
    }


def test_preflight_config_overfit_restores_rng_state(tmp_path, monkeypatch):
    """The overfit branch's own reseed-run-restore leaves no net trace on the streams: a plain
    smoke build (overfit=False) and the same build with overfit=True, run from the same seed,
    must consume the streams identically once the overfit call returns."""
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
    preflight_config(tmp_path, cfg, smoke=True, overfit=False)
    without_overfit = (random.random(), np.random.rand(), torch.rand(1))

    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    r = preflight_config(tmp_path, cfg, smoke=True, overfit=True)
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
    r = preflight_config(tmp_path, cfg, smoke=True, overfit=True)
    assert any("no parameter received a gradient" in i or "does not require grad" in i
              for i in r["issues"])
    assert r["overfit_check"]["passed"] is False
    assert "empty parameter list" in r["overfit_check"]["issue"]


def test_preflight_config_refuses_a_per_stage_lr_by_name(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "task": "detection"},
        "data": _labeled(tmp_path),
        "batch_size": 2,
        "stages": [{"freeze_to": -1, "epochs": 5, "lr": 1e-3}],
    }
    r = preflight_config(tmp_path, cfg)
    assert r["valid"] is False
    assert any(i.startswith("stages.0.lr:") for i in r["issues"]), r["issues"]

    cfg["stages"] = [{"freeze_to": -1, "epochs": 5}]
    assert preflight_config(tmp_path, cfg)["valid"] is True


def test_preflight_config_warns_when_most_candidates_wont_train(tmp_path):
    """The admission's own partition is otherwise thrown away: a run whose label store admits
    only a fraction of its candidate images must not report "valid, no warnings" with no
    visibility into what would silently train on far fewer images than the operator expects."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images" / UNDATED_BUCKET
    # 1 annotated, 3 unannotated (no label document at all) -> 75% of candidates won't train.
    _bud_image(imgs / "ann.jpg")
    for stem in ("a", "b", "c"):
        _bud_image(imgs / f"{stem}.jpg", labeled=False)

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "task": "detection"},
        "data": {"images_dir": str(imgs), "scope": {"subject": "bud"},
                 "auto_val": False},
        "batch_size": 2,
        # One admitted image holds nothing out, so the run selects on its training loss.
        "evaluation": {"selection_metric": "loss"},
    }
    r = preflight_config(tmp_path, cfg)
    assert r["valid"] is True, r  # informational only, never gating
    assert any("3/4 candidate images (75%) will not train" in w
               for w in r["warnings"]), r["warnings"]
    assert any("{'absent': 3}" in w for w in r["warnings"])


def test_preflight_config_no_coverage_warning_when_everything_trains(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images" / UNDATED_BUCKET
    _bud_image(imgs / "ann.jpg")

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "task": "detection"},
        "data": {"images_dir": str(imgs), "scope": {"subject": "bud"},
                 "split": {"seed": 0, "val_ratio": 0.15}},
        "batch_size": 2,
    }
    assert preflight_config(tmp_path, cfg)["warnings"] == []


# --------------------------------------------------------------------------
# preflight_config's training_source seam: a bare "module:function" string
# --------------------------------------------------------------------------

def test_preflight_config_blocks_rather_than_swallows_an_unreadable_label(tmp_path):
    """The coverage check's own admission must not fold an unreadable label into a generic build
    failure it silently drops: a run over these images would fail on the same document, so
    preflight reports it as a blocking issue, naming the document, not a warning; the launch's
    own admission refuses it in the same words."""
    pytest.importorskip("torch")
    from tcip_annotation import json_io
    from tcip_mcp.pipelines.data.split_construction import auto_train_val
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images" / UNDATED_BUCKET
    _bud_image(imgs / "ann.jpg")
    _bud_image(imgs / "bad.jpg")
    _damaged(imgs / "bad.jpg", b"{not json")

    data_cfg = {"images_dir": str(imgs), "scope": {"subject": "bud"},
                "split": {"seed": 0, "val_ratio": 0.15}}
    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "task": "detection"},
        "data": dict(data_cfg),
        "batch_size": 2,
    }
    r = preflight_config(tmp_path, cfg)
    with pytest.raises(json_io.UnreadableLabelDocumentError, match="bad") as raised:
        auto_train_val(tmp_path, "detection", dict(data_cfg), None)
    assert r["valid"] is False
    assert any(str(raised.value) in i for i in r["issues"]), r["issues"]


def test_preflight_admits_the_run_once(tmp_path):
    """Every preflight leg reads one population, so the producer is asked once however many legs
    need a source, a member name or a count: a second admission is a second answer to the one
    question of what this run trains on."""
    pytest.importorskip("torch")
    from PIL import Image
    from tcip_annotation.state import Annotation, BBox

    import tcip_mcp.pipelines.data.label_queries as label_queries
    from tcip_mcp.tools.training_tools import preflight_config
    from tests._producer_fixtures import label_image

    imgs = tmp_path / "images" / UNDATED_BUCKET
    imgs.mkdir(parents=True)
    Image.new("RGB", (256, 256)).save(imgs / "mosaic.jpg")
    label_image(imgs / "mosaic.jpg",
                [Annotation(subject="bud", geometry=BBox(20, 20, 60, 60))], 256, 256)

    calls = []
    real_admit = label_queries.admit

    def counting_admit(*args, **kwargs):
        calls.append(args)
        return real_admit(*args, **kwargs)

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "task": "detection"},
        "data": {"images_dir": str(imgs), "scope": {"subject": "bud"},
                 "tiling": {"enabled": True, "tile_size": 64, "overlap": 0.2},
                 "split": {"calibration_ratio": 0.15, "val_ratio": 0.2,
                           "holdout_ratio": 0.1, "seed": 1}},
        "batch_size": 2,
    }
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(label_queries, "admit", counting_admit)
        preflight_config(tmp_path, cfg, smoke=True)

    assert len(calls) == 1, calls


@pytest.mark.parametrize(("bad_document", "refusal"), [
    ('{"image": "bad", "width": 20, "height": 20, "annotations": 5}',
     "not the list a label document holds"),
    ('{"image": "bad", "width": 20, "height": 20, "annotations": [7]}',
     "not an annotation object"),
])
def test_preflight_config_blocks_a_document_only_the_admission_reader_refuses(
        tmp_path, bad_document, refusal):
    """A document that decodes to a dict but whose annotations field is not a list, or whose
    record cannot be coerced, is exactly what the run's own admission refuses at launch:
    preflight reads through the same call so it blocks here too, rather than passing a document
    the launch then aborts on."""
    pytest.importorskip("torch")
    from tcip_store import encode_record

    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images" / UNDATED_BUCKET
    _bud_image(imgs / "ann.jpg")
    _bud_image(imgs / "bad.jpg")
    _damaged(imgs / "bad.jpg", encode_record(json.loads(bad_document)))

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "task": "detection"},
        "data": {"images_dir": str(imgs), "scope": {"subject": "bud"},
                 "split": {"seed": 0, "val_ratio": 0.15}},
        "batch_size": 2,
    }
    r = preflight_config(tmp_path, cfg)
    assert r["valid"] is False
    assert any(refusal in i for i in r["issues"]), r["issues"]


def test_preflight_config_training_source_shape_and_importability(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    base_cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "task": "detection"},
        "data": _labeled(tmp_path),
        "batch_size": 2,
    }

    # A dict is rejected.
    cfg = dict(base_cfg, training_source={"train": "tests.bespoke_models:build_bespoke_detection"})
    r = preflight_config(tmp_path, cfg)
    assert any("training_source must be a non-empty" in i for i in r["issues"])

    # A bare string that doesn't import is rejected with the import error surfaced.
    cfg = dict(base_cfg, training_source="nonexistent_module:train")
    r = preflight_config(tmp_path, cfg)
    assert any("training_source not importable" in i for i in r["issues"])

    # A bare, importable string passes.
    cfg = dict(base_cfg, training_source="tests.bespoke_models:build_bespoke_detection")
    assert preflight_config(tmp_path, cfg)["valid"] is True

    # Absent training_source is fine (optional seam).
    assert preflight_config(tmp_path, base_cfg)["valid"] is True


# --------------------------------------------------------------------------
# preflight_config's selection_metric coherence: reject a comparability-only
# metric for a center-match trait at validation time, not mid-run.
# --------------------------------------------------------------------------

def test_preflight_config_rejects_incoherent_selection_metric(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    base_cfg: dict[str, object] = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "task": "detection"},
        "data": _labeled(tmp_path),
        "batch_size": 2,
    }

    # A comparability-only metric for a center-match trait is rejected.
    cfg = dict(base_cfg)
    cfg["evaluation"] = {"trait": "bud_opening", "selection_metric": "map50"}
    r = preflight_config(tmp_path, cfg)
    assert any("comparability-only" in i for i in r["issues"])

    # A governing metric for the same trait is fine.
    cfg["evaluation"] = {"trait": "bud_opening", "selection_metric": "f1"}
    assert preflight_config(tmp_path, cfg)["valid"] is True

    # No trait -> no coherence gate, even for a comparability metric.
    cfg["evaluation"] = {"selection_metric": "map50"}
    assert preflight_config(tmp_path, cfg)["valid"] is True

    # An undeclared direction is caught here even with no trait at all: it would otherwise
    # surface only as a failed run once resolve_selection_metric runs mid-training.
    cfg["evaluation"] = {"selection_metric": "not_a_real_metric"}
    r = preflight_config(tmp_path, cfg)
    assert any("no declared ranking direction" in i for i in r["issues"])


def test_preflight_config_names_a_non_mapping_evaluation_block_as_an_issue(tmp_path):
    """``TrainingSection``/``TrainConfigSchema`` both allow extra keys of any type, so a
    non-mapping ``evaluation`` block reaches ``eval_cfg.get(...)`` unchecked; it must become a
    named issue, not an ``AttributeError`` that crashes the whole preflight call."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images" / UNDATED_BUCKET
    imgs.mkdir(parents=True)
    cfg: dict[str, object] = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "task": "detection"},
        "data": {"images_dir": str(imgs), "scope": {"subject": "bud"},
                 "split": {"seed": 0, "val_ratio": 0.15}},
        "batch_size": 2,
        "evaluation": "not_a_mapping",
    }
    r = preflight_config(tmp_path, cfg)
    assert any(i.startswith("evaluation:") for i in r["issues"]), r["issues"]


# preflight_config's reserved-region feasibility check: a training-launch-time refusal through
# this module's own validation surface, never review_calibration._FAILURE_MESSAGES.

def _reserve_cal_big_single_source(root, width=4000, height=3000, tile_size=128):
    """One large single-image detection source with real width/height, real GT scattered evenly
    across it: enough for a feasible 4-way spatial-strip split; its images directory."""
    import torch
    from torchvision.utils import save_image

    from tcip_annotation.state import Annotation, BBox
    from tests._producer_fixtures import label_image

    images_dir = root / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True, exist_ok=True)
    image = images_dir / "mosaic.png"
    save_image(torch.rand(3, height, width) * 0.3, str(image))
    boxes = [Annotation(subject="bud", geometry=BBox(x, y, x + 20, y + 20))
            for x in range(20, width - 20, 200) for y in range(20, height - 20, 200)]
    label_image(image, boxes, width, height, keep_empty=True)
    return images_dir


def test_preflight_calibration_ratio_wrong_task_flags_issue(tmp_path):
    """A classification run never resolves to the within-image split a calibration region
    reserves from, so the fraction is named as having no effect."""
    from PIL import Image

    from tcip_mcp.tools.training_tools import preflight_config

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    rows = ["stem,label"]
    for i in range(4):
        Image.new("RGB", (32, 32), (20 * i, 30, 40)).save(images_dir / f"img{i}.png")
        rows.append(f"img{i},{i % 2}")
    (tmp_path / "labels.csv").write_text("\n".join(rows) + "\n")
    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_classifier",
                         "task": "classification"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(tmp_path / "labels.csv"),
                 "split": {"calibration_ratio": 0.15, "val_ratio": 0.15, "seed": 1}},
        "batch_size": 2,
    }
    r = preflight_config(tmp_path, cfg)
    assert any("calibration_ratio" in i and "no effect" in i for i in r["issues"]), r["issues"]


def test_preflight_calibration_ratio_multi_member_flags_issue(tmp_path):
    """Two admitted members resolve to the group-balanced split, which reserves no calibration
    region, so the fraction is named as having no effect."""
    from tcip_mcp.tools.training_tools import preflight_config

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    for stem in ("a", "b"):
        _bud_image(images_dir / f"{stem}.png")
    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "scope": {"subject": "bud"},
                 # sliver_frac stated: two boxes derive no size spread.
                 "tiling": {"enabled": True, "tile_size": 32, "sliver_frac": 0.5},
                 "split": {"calibration_ratio": 0.15, "val_ratio": 0.15, "seed": 1}},
        "batch_size": 2,
    }
    r = preflight_config(tmp_path, cfg)
    assert any("calibration_ratio" in i and "no effect" in i for i in r["issues"]), r["issues"]


def test_preflight_reserved_regions_infeasible_layout_refuses_under_smoke(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    images_dir = _reserve_cal_big_single_source(tmp_path / "ds")
    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "scope": {"subject": "bud"},
                 "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
                 # Nothing left for a real train fraction at this mosaic size.
                 "split": {"val_ratio": 0.33, "holdout_ratio": 0.33, "seed": 1,
                          "calibration_ratio": 0.33}},
        "batch_size": 2,
    }
    r = preflight_config(tmp_path, cfg, smoke=True)
    assert any("is infeasible" in i for i in r["issues"]), r["issues"]

    # The run's resolution builds its datasets whether or not the model is smoked, so the same
    # geometry refuses without smoke too.
    r_no_smoke = preflight_config(tmp_path, cfg, smoke=False)
    assert any("is infeasible" in i for i in r_no_smoke["issues"])


def test_preflight_reserved_regions_report_an_unreadable_label_by_name(tmp_path):
    """This probe's own generic except Exception must not swallow an unreadable label into a
    silently-logged build failure: the breeder needs to see which document is broken."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    images_dir = _reserve_cal_big_single_source(tmp_path / "ds")
    _damaged(images_dir / "mosaic.png", b"{not json")
    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "scope": {"subject": "bud"},
                 "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
                 "split": {"val_ratio": 0.2, "holdout_ratio": 0.1, "seed": 1,
                          "calibration_ratio": 0.15}},
        "batch_size": 2,
    }
    r = preflight_config(tmp_path, cfg, smoke=True)
    assert any("mosaic" in i and "decode" in i for i in r["issues"]), r["issues"]


def test_preflight_reserved_regions_admit_a_feasible_layout(tmp_path):
    """A real, feasible reserved-region config produces no spatial-split issue under
    smoke=True."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    images_dir = _reserve_cal_big_single_source(tmp_path / "ds")
    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"min_size": 128, "max_size": 256},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "scope": {"subject": "bud"},
                 "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
                 "split": {"val_ratio": 0.2, "holdout_ratio": 0.1, "seed": 1,
                          "calibration_ratio": 0.15}},
        "batch_size": 2,
    }
    r = preflight_config(tmp_path, cfg, smoke=True)
    assert not any("spatial split" in i or "calibration_ratio" in i for i in r["issues"]), \
        r["issues"]


# --------------------------------------------------------------------------
# _apply_hpo_params: lr/weight_decay reach what the trainer actually reads
# --------------------------------------------------------------------------

def _built_groups(config: dict) -> list[tuple[float, float]]:
    """``(lr, weight_decay)`` of each param group the trainer's optimizer read of ``config``
    (``schemas.train_config``'s ``optimizer`` into ``optimizer_factory.build_optimizer``)
    builds over a backbone-and-head model, the backbone group first."""
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.schemas import train_config
    from tcip_mcp.pipelines.training.optimizer_factory import build_optimizer
    from tests.tiny_trainer_fixtures import build_two_rate_regressor

    spec = train_config(config).optimizer
    optimizer = build_optimizer(spec.name, build_two_rate_regressor(),
                                backbone_lr=spec.backbone_lr, head_lr=spec.head_lr,
                                weight_decay=spec.weight_decay)
    return [(g["lr"], g["weight_decay"]) for g in optimizer.param_groups]


def test_apply_hpo_params_lr_reaches_optimizer_param_groups():
    """Suggested lr/weight_decay survive the trainer's own optimizer read into the head group's
    rate and every group's weight decay."""
    from tcip_mcp.tools.training_tools import _apply_hpo_params

    base = {"model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                             "task": "detection"}}
    (_, backbone_decay), (head_lr, head_decay) = _built_groups(
        _apply_hpo_params(base, {"lr": 3e-3, "weight_decay": 2e-4}))
    assert head_lr == pytest.approx(3e-3)
    assert backbone_decay == head_decay == pytest.approx(2e-4)


def test_apply_hpo_params_preserves_base_config_stages():
    """Sweeping lr must not overwrite the agent's own progressive-unfreeze schedule with a
    hardcoded recipe: base_config's stages (however it expressed them) survive unchanged."""
    from tcip_mcp.tools.training_tools import _apply_hpo_params

    custom_stages = [{"freeze_to": -1, "epochs": 2}, {"freeze_to": 0, "epochs": 8}]
    base = {"model_source": {"builder": "x:y", "task": "detection"},
            "stages": custom_stages}
    out = _apply_hpo_params(base, {"lr": 3e-3})
    assert out["stages"] == custom_stages

    # No stages configured at all -> still nothing invented here; generic_trainer.train()'s own
    # single-stage fallback covers it.
    base_no_stages = {"model_source": {"builder": "x:y", "task": "detection"}}
    out2 = _apply_hpo_params(base_no_stages, {"lr": 3e-3})
    assert "stages" not in out2


@pytest.mark.parametrize("optimizer", [
    {}, {"head_lr": 0.002}, {"backbone_lr": 0.0002}, {"backbone_lr": 2e-5, "head_lr": 1e-4},
])
def test_apply_hpo_params_keeps_the_backbone_ratio_the_base_config_trains_at(optimizer):
    """The swept ``lr`` sets the head group's rate, and the backbone group keeps the ratio to
    it that the base config's own optimizer, built the way the trainer builds it, trains at,
    defaults standing for any rate left unstated."""
    from tcip_mcp.tools.training_tools import _apply_hpo_params

    base = {"model_source": {"builder": "x:y", "task": "detection"}, "optimizer": optimizer}
    (base_backbone, _), (base_head, _) = _built_groups(base)
    (swept_backbone, _), (swept_head, _) = _built_groups(_apply_hpo_params(base, {"lr": 0.02}))
    assert swept_head == pytest.approx(0.02)
    assert swept_backbone / swept_head == pytest.approx(base_backbone / base_head)


def test_apply_hpo_params_unrecognized_key_reaches_top_level():
    """A swept key outside the known optimizer/batch/weight_decay set must land at the top level
    of the resolved config, the one placement train() reads, never nested under "training",
    which the config schema refuses."""
    from tcip_mcp.tools.training_tools import _apply_hpo_params

    base = {"model_source": {"builder": "x:y", "task": "detection"}}
    out = _apply_hpo_params(base, {"momentum": 0.9})
    assert out["momentum"] == 0.9
    assert "training" not in out


def test_apply_hpo_params_dotted_key_reaches_nested_field():
    """A dotted param (e.g. a swept builder) reaches the nested field it names."""
    from tcip_mcp.tools.training_tools import _apply_hpo_params

    base = {"model_source": {"builder": "old:builder", "task": "detection"}}
    out = _apply_hpo_params(base, {"model_source.builder": "new:builder"})
    assert out["model_source"]["builder"] == "new:builder"
    assert out["model_source"]["task"] == "detection"  # the rest of the mapping survives


def test_apply_hpo_params_dotted_seed_key_reaches_the_split_config():
    """split_draws's own paired grid axis (data.split.seed) reaches the nested field
    run_hyperparameter_search's drawn path reads, the same dotted-key mechanism a swept
    builder uses above."""
    from tcip_mcp.tools.training_tools import _apply_hpo_params

    base = {"model_source": {"builder": "x:y", "task": "detection"}}
    out = _apply_hpo_params(base, {"data.split.seed": 7})
    assert out["data"]["split"]["seed"] == 7


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


class _TiledFakeDataset(_FakeDataset):
    """A stand-in dataset carrying tile geometry, for stamp_effective_data_geometry to record."""
    tile_size = 224
    overlap = 0.2


def _detection_base() -> dict:
    return {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "task": "detection"},
        "data": {"images_dir": "imgs"},
        "batch_size": 2,
    }


def _trial(point: dict, report, base: dict, project: Path, name: str = "t0", *,
           metric: str = "loss", higher_is_better: bool = False) -> Path:
    """One HPO trial ``name`` run as its sweep runs it (``_run_hpo_trial``), under one sweep of
    ``project`` over ``base`` optimizing ``metric`` in its direction, the sweep's record written
    by the writer its opening uses (``experiments.write_record``) on its first trial. Returns
    the trial's run directory."""
    from tcip_mcp import experiments
    from tcip_mcp.audit import now_iso
    from tcip_mcp.tools.training_tools import _run_hpo_trial

    sweep = experiments.experiment_dir("hpo_trials", project=project)
    if not sweep.is_dir():
        experiments.create_run_directory(sweep)
        experiments.write_record(sweep / experiments.SWEEP_FILE, {
            "created": now_iso(), "input": {"base_config": base, "split_draws": 1},
            "objective": {"selection_metric": metric, "higher_is_better": higher_is_better}})
    _run_hpo_trial(point, report, sweep, name)
    return sweep / f"{sweep.name}_{name}"


def _row(value: float, metric: str = "loss") -> dict:
    """An epoch row stamping ``metric`` as the run's selection, the way the trainer stamps it."""
    return {"selection_metric": metric, "selection": value}


def _complete(run):
    """End a stand-in training body the way a real one does: its final weights saved under
    ``model_final``."""
    from tests._verified_checkpoint_fixtures import checkpoint_file

    run.saved["model_final"] = checkpoint_file(Path(run.output_dir) / "model_final.pt", "weights")
    run.status = "completed"
    return run


def _sweep_outcome(project: Path) -> dict:
    """What the trials of :func:`_trial`'s sweep amount to, as the project's listing reads it."""
    from tcip_mcp.experiments import training_listing

    (sweep,) = training_listing(project).sweeps
    return sweep.outcome


def _patch_hpo_trial_machinery(monkeypatch, fake_train, captured=None):
    """Stub dataset building + training + loaders so a trial runs instantly, no Ray: the
    resolution builds stand-in datasets and records no samples, and the child builds the same
    stand-ins from that record."""
    import torch.utils.data as tud
    from tcip_mcp.pipelines.data import samplers
    from tcip_mcp.pipelines.data import split_construction as sc
    from tcip_mcp.pipelines.training import generic_trainer as gt

    ds = _FakeDataset()

    def fake_auto_train_val(project, task, data_cfg, transforms, **_):
        if captured is not None:
            captured["transforms"] = transforms
            captured["data_cfg"] = data_cfg
        return ds, ds, {"seed": None, "group_by": "stem", "samples": [], "selection": None}

    monkeypatch.setattr(sc, "auto_train_val", fake_auto_train_val)
    monkeypatch.setattr(sc, "recorded_datasets", lambda *a, **k: (ds, ds))
    monkeypatch.setattr(gt, "train", fake_train)
    monkeypatch.setattr(samplers, "build_sampler", lambda *a, **k: None)
    monkeypatch.setattr(tud, "DataLoader", lambda *a, **k: object())


def _spaces_searched(monkeypatch) -> list:
    """Replace ``tune_search`` with a search that answers its study directory, and the list it
    records each param space it ran over into."""
    from tcip_mcp.pipelines.training import hpo

    seen: list = []

    def fake_search(*args, **kwargs):
        seen.append(kwargs.get("param_space", args[1] if len(args) > 1 else None))

    monkeypatch.setattr(hpo, "tune_search", fake_search)
    return seen


def _completed_train(run, train_loader, val_loader, epoch_callback=None, batch_callback=None,
                     resume_from=""):
    """A training call that completes the run at once, reporting no epoch."""
    run.status = "completed"
    return run


def test_run_hpo_trial_reports_each_epoch_and_its_result_is_the_best_of_them(
        monkeypatch, tmp_path):
    """Every epoch's selection value is reported as the trial logs it, and the trial's result is
    the best of those same rows (lower=better)."""
    pytest.importorskip("torch")

    def fake_train(run, train_loader, val_loader,
                   epoch_callback=None, batch_callback=None, resume_from=""):
        for epoch, value in enumerate([50.0, 40.0, 30.0]):
            if epoch_callback:
                epoch_callback(epoch, _row(value))
        return _complete(run)

    _patch_hpo_trial_machinery(monkeypatch, fake_train)
    reported: list = []
    _trial({"lr": 3e-4}, reported.append, _detection_base(), tmp_path)
    assert reported == [50.0, 40.0, 30.0]
    assert _sweep_outcome(tmp_path)["best_value"] == 30.0


def test_run_hpo_trial_that_fails_has_no_result(monkeypatch, tmp_path):
    """A crashed trial ends failed and carries no result, so it can never become the sweep's
    best."""
    pytest.importorskip("torch")
    from tcip_mcp.experiments import observe

    def fake_train(run, train_loader, val_loader,
                   epoch_callback=None, batch_callback=None, resume_from=""):
        raise RuntimeError("CUDA out of memory")

    _patch_hpo_trial_machinery(monkeypatch, fake_train)
    reported: list = []
    trial_dir = _trial({"lr": 3e-4}, reported.append, _detection_base(), tmp_path)
    assert reported == []
    assert observe(trial_dir).state == "failed"
    assert _sweep_outcome(tmp_path)["best_value"] is None


def test_run_hpo_trial_result_is_the_highest_value_for_a_higher_is_better_metric(
        monkeypatch, tmp_path):
    """A higher-is-better selection metric (accuracy) makes the trial's result its highest
    reported value, not a minimize convention that would instead prefer the lowest."""
    pytest.importorskip("torch")

    def fake_train(run, train_loader, val_loader,
                   epoch_callback=None, batch_callback=None, resume_from=""):
        for epoch, value in enumerate([0.5, 0.9, 0.6]):
            if epoch_callback:
                epoch_callback(epoch, _row(value, metric="accuracy"))
        return _complete(run)

    _patch_hpo_trial_machinery(monkeypatch, fake_train)
    base = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_classifier",
                         "task": "classification"},
        "data": {"images_dir": "imgs"},
        "batch_size": 2,
        "evaluation": {"selection_metric": "accuracy"},
    }
    reported: list = []
    _trial({"lr": 3e-4}, reported.append, base, tmp_path, metric="accuracy",
           higher_is_better=True)
    assert reported == [0.5, 0.9, 0.6]
    assert _sweep_outcome(tmp_path)["best_value"] == 0.9


def test_a_failed_trial_never_outranks_a_real_one_under_a_maximize_direction(
    monkeypatch, tmp_path,
):
    """A trial that fails carries no result, so the sweep's outcome under a maximize direction
    is the one completed trial's, never the failed one's."""
    pytest.importorskip("torch")

    base = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_classifier",
                         "task": "classification"},
        "data": {"images_dir": "imgs"},
        "batch_size": 2,
        "evaluation": {"selection_metric": "accuracy"},
    }

    def fake_train_ok(run, train_loader, val_loader, task="classification",
                      epoch_callback=None, batch_callback=None, resume_from=""):
        if epoch_callback:
            epoch_callback(0, _row(0.7, metric="accuracy"))
        return _complete(run)

    _patch_hpo_trial_machinery(monkeypatch, fake_train_ok)
    real: list = []
    _trial({"lr": 3e-4}, real.append, base, tmp_path, "real", metric="accuracy",
           higher_is_better=True)

    def fake_train_fails(run, train_loader, val_loader, task="classification",
                         epoch_callback=None, batch_callback=None, resume_from=""):
        raise RuntimeError("boom")

    _patch_hpo_trial_machinery(monkeypatch, fake_train_fails)
    failed: list = []
    _trial({"lr": 1e-2}, failed.append, base, tmp_path, "failed", metric="accuracy",
           higher_is_better=True)

    outcome = _sweep_outcome(tmp_path)
    assert (outcome["best_params"], outcome["best_value"]) == ({"lr": 3e-4}, 0.7)


def test_run_hpo_trial_uses_base_augmentation_and_model(monkeypatch, tmp_path):
    """Trials train under the final run's regime: base_config augmentation reaches the train
    dataset, and the bespoke model_source is carried through. (Loss is owned by the builder.)"""
    pytest.importorskip("torch")

    captured: dict = {}

    def fake_train(run, train_loader, val_loader,
                   epoch_callback=None, batch_callback=None, resume_from=""):
        captured["model_source"] = run.config["model_source"]
        run.status = "completed"
        return run

    _patch_hpo_trial_machinery(monkeypatch, fake_train, captured=captured)
    base = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_classifier",
                         "task": "classification"},
        "data": {"images_dir": "imgs"},
        "batch_size": 2,
        "augmentation": {"horizontal_flip": 0.5},
    }
    _trial({"lr": 3e-4}, [].append, base, tmp_path)
    assert captured["transforms"] is not None       # augmentation was built + passed
    assert captured["model_source"]["builder"].endswith(":build_bespoke_classifier")


def test_run_hpo_trial_dotted_seed_axis_reaches_the_data_cfg_handed_to_auto_train_val(
    monkeypatch, tmp_path,
):
    """The split_draws grid axis (data.split.seed) resolves onto the config before the trial's
    data section is built, so auto_train_val (and the drawn split behind it) actually reads the
    draw's own seed rather than the base config's unswept one."""
    pytest.importorskip("torch")

    captured: dict = {}
    _patch_hpo_trial_machinery(monkeypatch, _completed_train, captured=captured)
    _trial({"data.split.seed": 7}, [].append, _detection_base(), tmp_path)
    assert captured["data_cfg"]["split"]["seed"] == 7


def _fake_auto_train_val_reading_seed_like_split_construction(
        project, task, data_cfg, transforms, **_):
    """The reads ``auto_train_val`` performs on its multi-stem drawn path: setdefault the split
    block, then get its seed off that block."""
    split_cfg = data_cfg.setdefault("split", {})
    split_cfg.get("seed", 42)
    ds = _TiledFakeDataset()
    return ds, ds, {"samples": []}


def test_run_hpo_trial_dotted_seed_axis_reaches_the_trials_own_records(monkeypatch, tmp_path):
    """data.split.seed, the split_draws grid axis, lands in the trial's launch record and in the
    data section it resolved, at the nested field auto_train_val reads it from, beside the
    sampled point itself."""
    pytest.importorskip("torch")
    from tcip_mcp.experiments import RUN_FILE, read_record

    _patch_hpo_trial_machinery(monkeypatch, _completed_train)
    from tcip_mcp.pipelines.data import split_construction as sc
    monkeypatch.setattr(
        sc, "auto_train_val", _fake_auto_train_val_reading_seed_like_split_construction)

    trial_dir = _trial({"data.split.seed": 7}, [].append, _detection_base(), tmp_path)

    run = read_record(trial_dir / RUN_FILE)
    assert run["config"]["data"]["split"]["seed"] == 7
    assert run["trial_params"] == {"data.split.seed": 7}
    assert run["resolved"]["data"]["split"]["seed"] == 7


def test_run_hpo_trial_geometry_stamp_from_a_tiled_dataset_reaches_the_resolved_record(
    monkeypatch, tmp_path,
):
    """The tile geometry stamp_effective_data_geometry records off the tiled dataset a trial's
    auto_train_val returns is in the trial's resolved data section, the record a caller reads
    back to know what the trial actually trained on."""
    pytest.importorskip("torch")
    from tcip_mcp.experiments import RUN_FILE, read_record

    _patch_hpo_trial_machinery(monkeypatch, _completed_train)
    from tcip_mcp.pipelines.data import split_construction as sc
    monkeypatch.setattr(
        sc, "auto_train_val", _fake_auto_train_val_reading_seed_like_split_construction)

    trial_dir = _trial({"data.split.seed": 7}, [].append, _detection_base(), tmp_path)

    assert read_record(trial_dir / RUN_FILE)["resolved"]["data"]["tiling"]["tile_size"] == 224


def test_run_hpo_trial_producer_fed_data_split_seed_over_the_single_source_spatial_path(
    monkeypatch, tmp_path,
):
    """The producer path: a real one-source tiled dataset through the real, unstubbed
    auto_train_val. Its single-source spatial-strip branch places every strip by declared order
    alone, and the trial's launch record carries the real spatial_manifest and tiling
    auto_train_val wrote, with no seed inside the manifest it never read one for."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from tcip_mcp.experiments import RUN_FILE, read_record
    from tests.test_training_autoval import _big_single_source

    images_dir, _stem = _big_single_source(tmp_path / "ds", 4000, 3000)
    base = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "scope": {"subject": "bud"},
                 "auto_val": True, "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
                 "split": {"val_ratio": 0.25, "holdout_ratio": 0.1, "calibration_ratio": 0}},
        "batch_size": 2,
    }

    import torch.utils.data as tud
    from tcip_mcp.pipelines.data import samplers
    from tcip_mcp.pipelines.training import generic_trainer as gt
    monkeypatch.setattr(gt, "train", _completed_train)
    monkeypatch.setattr(samplers, "build_sampler", lambda *a, **k: None)
    monkeypatch.setattr(tud, "DataLoader", lambda *a, **k: object())

    trial_dir = _trial({"data.split.seed": 3}, [].append, base, tmp_path)

    resolved = read_record(trial_dir / RUN_FILE)["resolved"]
    assert resolved["partition"]["seed"] is None
    assert resolved["data"]["split"]["spatial_manifest"]
    assert "seed" not in resolved["data"]["split"]["spatial_manifest"]
    assert resolved["data"]["tiling"]["tile_size"] == 128


def test_a_trials_launch_record_carries_the_seed_it_trained_under(monkeypatch, tmp_path):
    """An unset seed is drawn onto the trial's launch record once, and the body trains under
    that recorded seed."""
    pytest.importorskip("torch")
    from tcip_mcp.experiments import RUN_FILE, read_record

    captured: dict = {}

    def fake_train(run, train_loader, val_loader,
                   epoch_callback=None, batch_callback=None, resume_from=""):
        captured["seed"] = run.config.get("seed")
        run.status = "completed"
        return run

    _patch_hpo_trial_machinery(monkeypatch, fake_train)
    trial_dir = _trial({"lr": 3e-4}, [].append, _detection_base(), tmp_path)

    recorded = read_record(trial_dir / RUN_FILE)["config"]["seed"]
    assert recorded is not None and recorded == captured["seed"]


def test_run_hpo_trial_diverged_run_never_outranks_a_worse_but_alive_config(tmp_path):
    """A trial that trains one real epoch and then diverges ends failed and carries no result,
    not that epoch's real score, so it can never outrank a config that only scored worse. Drives
    the real training body (nothing mocked).

    ``auto_val`` off, so the score ranked is the training loss over every admitted row: what is
    under test is how a diverged run ranks, and a drawn validation side would spend this model's
    one good forward on measuring rather than training.
    """
    pytest.importorskip("torch")
    from tests.tiny_trainer_fixtures import write_regression_dataset

    images_dir, csv_path = write_regression_dataset(
        tmp_path, intensities=[0.1, 0.3, 0.5, 0.7], values=[0.2, 0.6, 1.0, 1.4])

    base_config = {
        "model_source": {"builder": "tests.tiny_trainer_fixtures:build_diverges_after_model",
                         "builder_kwargs": {"good_calls": 1}, "task": "regression"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(csv_path), "auto_val": False},
        "device": "cpu",
        "mixed_precision": False,
        "batch_size": 4,
        "stages": [{"freeze_to": 0, "epochs": 5}],
        "optimizer": {"name": "adamw", "backbone_lr": 0.05, "head_lr": 0.05, "weight_decay": 0.0},
        "checkpoint_every_n_epochs": 0,
        "early_stopping": {"enabled": False},
    }
    reported: list = []
    _trial({}, reported.append, base_config, tmp_path, metric="loss")

    import math
    assert math.isfinite(reported[0])  # epoch 1's real score, reported before the run died
    assert _sweep_outcome(tmp_path)["best_value"] is None


# --------------------------------------------------------------------------
# get_worst_predictions: confidence comes from the canonical prediction format
# --------------------------------------------------------------------------

def test_get_worst_predictions_reads_canonical_confidence(tmp_path):
    """Prediction documents carry a native ``score``; confidence reads from that field, not from
    box geometry. Reading a normalized box height as confidence instead would make the
    (1 - avg_conf) ranking term ~1.0 for every image with small boxes (e.g. buds)."""
    pytest.importorskip("torch")
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.vision_tools import get_worst_predictions
    from tests._chain_fixtures import published
    from tests._producer_fixtures import label_image

    from PIL import Image

    images = tmp_path / "images" / UNDATED_BUCKET
    images.mkdir(parents=True)
    scored = {"confident": [0.9, 0.9], "shaky": [0.1, 0.1]}
    for stem, scores in scored.items():
        Image.new("RGB", (100, 100)).save(images / f"{stem}.png")
        # Matching GT count → missed = extra = 0, error is exactly (1 - avg_conf).
        gt_anns = [Annotation(subject="bud", geometry=BBox(20.0, 11.0, 40.0, 31.0)) for _ in scores]
        label_image(images / f"{stem}.png", gt_anns, 100, 100)
    # Confidence lives in the document's `score`; box geometry is irrelevant to this count +
    # confidence heuristic (no IoU matching), so the boxes can be anything.
    bucket = published(tmp_path, "preds", [
        {"image": str(images / f"{stem}.png"), "width": 100, "height": 100,
         "boxes": [[10.0, 10.0, 40.0, 22.0]] * len(scores), "scores": scores,
         "labels": [1] * len(scores)} for stem, scores in scored.items()],
        scope={"subject": "bud"})

    out = get_worst_predictions(bucket, top_k=2)
    by_stem = {w["stem"]: w["error_score"] for w in out["worst_images"]}
    assert by_stem["confident"] == pytest.approx(0.1, abs=1e-3)
    assert by_stem["shaky"] == pytest.approx(0.9, abs=1e-3)
    assert out["worst_images"][0]["stem"] == "shaky"  # low confidence ranks worst


def test_a_launch_config_that_json_cannot_hold_is_refused_before_the_run_starts(
    tmp_path, monkeypatch
):
    """The caller's config is stored twice, as the launch config and as the experiment's
    snapshot, so the field that will not encode is named before either write and before a
    subprocess is spawned for a run whose provenance could not be recorded."""
    from tcip_mcp.tools import training_tools

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(training_tools, "_preflight",
                        lambda project, config, *, smoke, overfit: (
                            {"valid": False, "issues": ["stub"]}, None))

    with pytest.raises(TypeError) as refused:
        training_tools.launch_training(tmp_path, {"model_source": {"builder": Path("m.py")}},
                                       actor=None)
    assert "config.model_source.builder" in str(refused.value)


def test_an_ordinary_launch_config_passes_the_boundary_to_preflight(tmp_path, monkeypatch):
    """The refusal above must not stop a legitimate config: it reaches preflight and comes
    back with preflight's own verdict rather than a refusal from the boundary check."""
    from tcip_mcp.tools import training_tools

    monkeypatch.chdir(tmp_path)
    seen = []

    def stub_preflight(project, config, *, smoke, overfit):
        seen.append(config)
        return {"valid": False, "issues": ["stub"]}, None

    monkeypatch.setattr(training_tools, "_preflight", stub_preflight)

    result = training_tools.launch_training(tmp_path, {"model_source": {"builder": "m:f"}},
                                            actor=None)

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
        training_tools.run_hyperparameter_search(tmp_path, {"model_source": {"builder": "m:f"}},
                                                 param_space={"lr": Path("lr.txt")}, search_seed=0)
    assert "param_space.lr" in str(space_refused.value)

    with pytest.raises(TypeError) as config_refused:
        training_tools.run_hyperparameter_search(
            tmp_path, {"model_source": {"builder": Path("m.py")}},
            param_space={"lr": [0.1, 0.01]}, search_seed=0)
    assert "base_config.model_source.builder" in str(config_refused.value)


def test_an_ordinary_sweep_payload_still_runs_its_search(tmp_path, monkeypatch):
    """The refusal above must not cost a legitimate sweep its search: admits valid work through
    the sweep door's structural preflight (an importable builder, a real data section)."""
    from tcip_mcp.tools import training_tools

    monkeypatch.chdir(tmp_path)
    seen = _spaces_searched(monkeypatch)

    base_config = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "task": "detection"},
        "data": _labeled(tmp_path),
    }
    result = training_tools.run_hyperparameter_search(
        tmp_path, base_config, param_space={"lr": [0.1, 0.01]}, n_trials=1, search_seed=0)

    assert result["sweep"]["state"] == "completed", result
    assert seen == [{"lr": [0.1, 0.01]}]


def test_run_hyperparameter_search_admits_an_lr_sweep_beside_a_base_config_selection_metric(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """The selection-metric refusal targets param_space, never base_config: a config that
    states its own selection metric still runs an ordinary lr sweep."""
    from tcip_mcp.tools import training_tools

    monkeypatch.chdir(tmp_path)
    seen = _spaces_searched(monkeypatch)

    base_config = {**real_hpo_base_config, "evaluation": {"selection_metric": "map"}}
    result = training_tools.run_hyperparameter_search(
        tmp_path, base_config, param_space={"lr": [0.1, 0.01]}, n_trials=1, search_seed=0)

    assert result["sweep"]["state"] == "completed", result
    assert seen == [{"lr": [0.1, 0.01]}]


def test_hpo_admits_a_categorical_evaluation_axis_naming_the_same_metric_at_every_choice(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """A categorical evaluation axis is admitted: every trial records the sweep's one
    objective."""
    from tcip_mcp.tools import training_tools

    monkeypatch.chdir(tmp_path)
    seen = _spaces_searched(monkeypatch)

    base_config = {**real_hpo_base_config, "evaluation": {"selection_metric": "map"}}
    param_space = {"evaluation": {
        "type": "categorical",
        "choices": [{"selection_metric": "map"}, {"selection_metric": "map"}],
    }}
    result = training_tools.run_hyperparameter_search(
        tmp_path, base_config, param_space=param_space, n_trials=1, search_seed=0)

    assert "error" not in result, result
    assert seen == [param_space]


def test_dataset_identity_tolerates_a_genuinely_unregistered_dataset(tmp_path):
    """The admitting half: no identity document at all still reads as (None, fp), not a refusal."""
    from tcip_mcp.pipelines.data.split_construction import dataset_identity

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)

    ds_id, fp = dataset_identity({"images_dir": str(images_dir)})
    assert ds_id is None


def test_cancel_end_to_end_through_the_real_trainer_ends_canceled_with_records_and_no_result(
    tmp_path
) -> None:
    """A trial that trains one real epoch and is then canceled mid-training (the sweep's
    cancellation requested only once a genuine score is already on record) ends canceled with
    its launch record and carries no result, not that epoch's real score, so a canceled trial can
    never outrank one that merely scored worse."""
    pytest.importorskip("torch")
    import math

    from tcip_mcp.experiments import (
        RUN_FILE, experiment_dir, observe, read_record, request_cancel,
    )
    from tests.tiny_trainer_fixtures import write_regression_dataset

    images_dir, csv_path = write_regression_dataset(
        tmp_path, intensities=[0.1, 0.3, 0.5, 0.7], values=[0.2, 0.6, 1.0, 1.4])
    base_config = {
        "model_source": {"builder": "tests.tiny_trainer_fixtures:build_mean_intensity_regressor",
                         "task": "regression"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(csv_path),
                 "split": {"seed": 0, "val_ratio": 0.15}},
        "batch_size": 2, "stages": [{"freeze_to": 0, "epochs": 5}],
                     "mixed_precision": False, "device": "cpu",
                     "checkpoint_every_n_epochs": 0, "early_stopping": {"enabled": False},
    }
    reported: list = []

    def report(value: float) -> None:
        # The cancel is requested only after the first real report, so a genuine score is
        # already on record by the time the sweep's cancel takes effect mid-training.
        reported.append(value)
        if len(reported) == 1:
            request_cancel(experiment_dir("hpo_trials", project=tmp_path))

    trial_dir = _trial({}, report, base_config, tmp_path, metric="loss")

    assert math.isfinite(reported[0])  # epoch 1's real score, reported before the cancel
    assert observe(trial_dir).state == "canceled"
    assert _sweep_outcome(tmp_path)["best_value"] is None
    assert read_record(trial_dir / RUN_FILE)["trial_params"] == {}
