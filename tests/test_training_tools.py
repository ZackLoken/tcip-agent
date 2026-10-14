"""Tests for training_tools: validator/StageSpec alignment, HPO param plumbing, HPO trial
reporting (each epoch as it completes; only a completed trial can be the sweep's best),
HPO/train regime parity, experiment immutability on relaunch, and canonical-format confidence
parsing in get_worst_predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tests._chain_fixtures import (
    BESPOKE_CLASSIFIER,
    BESPOKE_DETECTION,
    BESPOKE_MODELS,
    CLASSIFIER_SOURCE,
    DETECTION_SOURCE,
    FRAME_STATISTICS,
    TRAIN_BESPOKE,
    training_config,
    weights_statistics,
)
from tests._producer_fixtures import big_single_source
from tests._training_values import (
    TRIAGE_WEIGHTS,
    augmentation,
    drawn_split,
    evaluation_block,
    fifo_search,
    stage,
    tiling_block,
)
from tests._web_fixtures import INVARIANCES, new_project

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")


def _labeled(tmp_path: Path) -> dict:
    """A data section over two labeled frames of ``bud`` under ``tmp_path``
    (``_verified_checkpoint_fixtures.detection_images``), drawn at seed 0.
    """
    from tests._verified_checkpoint_fixtures import detection_images

    scope = {"subject": "bud"}
    return {
        **detection_images(tmp_path / "labeled", scope),
        "scope": scope,
        "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
    }


def _bud_image(image: Path, *, labeled: bool = True) -> None:
    """A 20x20 frame at ``image``, its bud's box bright on a dim ground, and, unless ``labeled``
    is false, its label document holding that ``bud`` box.
    """
    from tcip_annotation.state import Annotation, BBox
    from tests._producer_fixtures import BRIGHT, label_image, painted_frame

    image.parent.mkdir(parents=True, exist_ok=True)
    painted_frame(20, 20, (0, 0, 0), [((2, 2, 10, 10), BRIGHT)]).save(image)
    if labeled:
        label_image(
            image,
            [Annotation(subject="bud", geometry=BBox(2, 2, 10, 10))],
            20,
            20,
        )


def _damaged(image: Path, stored: bytes) -> None:
    """``image``'s label document replaced in the store by ``stored``."""
    from tests._producer_fixtures import image_label_key
    from tests._record_damage_fixtures import damage_record

    damage_record(image_label_key(image), stored)


# --------------------------------------------------------------------------
# preflight_config: stages
# --------------------------------------------------------------------------


def test_preflight_config_accepts_trainer_canonical_stages(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    cfg = training_config(
        tmp_path,
        DETECTION_SOURCE,
        _labeled(tmp_path),
        stages=[stage(-1), stage(2)],
    )
    r = preflight_config(tmp_path, cfg)
    assert r["valid"] is True, r["issues"]

    # A stage carries no epoch count, and the stop rule that ends it is stated.
    r2 = preflight_config(tmp_path, {**cfg, "stages": [stage(0, epochs=5)]})
    assert any(i.startswith("stages.0.epochs:") for i in r2["issues"]), r2[
        "issues"
    ]
    unstopped = {k: v for k, v in cfg.items() if k != "early_stopping"}
    r4 = preflight_config(tmp_path, unstopped)
    assert any(
        "'early_stopping'" in i and "unstated" in i for i in r4["issues"]
    ), r4["issues"]

    # No stages at all is refused naming them: no schedule ships as a default.
    del cfg["stages"]
    r3 = preflight_config(tmp_path, cfg)
    assert any("['stages'] unstated" in i for i in r3["issues"]), r3["issues"]


def test_preflight_config_refuses_a_nested_training_section_by_name(tmp_path):
    """The config has one placement: a ``training`` section is refused naming the move."""
    from tcip_mcp.tools.training_tools import preflight_config

    flat = training_config(tmp_path, DETECTION_SOURCE, _labeled(tmp_path))
    placed = ("model_source", "data")
    nested = {
        **{k: flat[k] for k in placed},
        "training": {k: v for k, v in flat.items() if k not in placed},
    }
    r = preflight_config(tmp_path, nested)
    assert r["valid"] is False
    assert any(
        "'training' is not a config section" in i for i in r["issues"]
    ), r["issues"]

    assert preflight_config(tmp_path, flat)["valid"] is True


def test_a_smoke_preflight_validates_its_config_once(tmp_path, monkeypatch):
    """A preflight that admits a config, from its structural check through its resolution and
    its smoke build, reads it off one validation and never validates its model source again.
    """
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config
    from tests.tiny_trainer_fixtures import (
        TWO_RATE_REGRESSOR,
        count_validations,
        regressor_config,
        write_regression_dataset,
    )

    intensities = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]
    images_dir, csv_path = write_regression_dataset(
        tmp_path / "ds", intensities, [2.0 * c for c in intensities]
    )
    config = regressor_config(
        builder=TWO_RATE_REGRESSOR,
        data={
            "num_channels": 1,
            "scope": {},
            "images_dir": str(images_dir),
            "labels_dir": str(csv_path),
            "split": drawn_split(
                val_ratio=0.15, seed=1, grouping="tile_prefix"
            ),
        },
        project=tmp_path,
    )

    validations = count_validations(monkeypatch)
    report = preflight_config(tmp_path, config, smoke=True)
    assert report["valid"] is True, report["issues"]
    assert validations == ["TrainConfigSchema"]


def test_preflight_config_types_a_non_dict_data_section_instead_of_raising(
    tmp_path,
):
    from tcip_mcp.tools.training_tools import preflight_config

    r = preflight_config(tmp_path, {"data": "x"})
    assert r["valid"] is False
    assert any(
        i.startswith("data:") and "dictionary" in i for i in r["issues"]
    ), r["issues"]


# preflight_config's overfit branch: reseed-run-restore, never gating


def _frozen_detection_builder(**kwargs):
    """An importable builder whose model has nothing to optimize (a legitimate stage-0 shape).
    Forwards its kwargs so a caller's ``detector`` choice (``fcos``, say) actually lands.
    """
    from tests import bespoke_models

    model = bespoke_models.build_bespoke_detection(**kwargs)
    for p in model.parameters():
        p.requires_grad = False
    return model


def _detection_smoke_cfg(builder: str, tmp_path: Path) -> dict:
    """A smoke config over the bespoke detector at the statistics and geometry its own small
    frames derive, which keeps two smoke builds plus twenty overfit steps to seconds. ``fcos``
    builds in a fraction of ``faster_rcnn``'s time over the same resnet18 backbone (single-stage,
    no region-proposal network), and nothing either smoke test asserts is
    faster-rcnn-specific.
    """
    return training_config(
        tmp_path,
        {
            "builder": builder,
            "builder_kwargs": {"detector": "fcos"},
            "source_files": [__file__, BESPOKE_MODELS],
            "task": "detection",
            "normalization": FRAME_STATISTICS,
        },
        {**_labeled(tmp_path), "num_channels": 3},
    )


def test_preflight_config_overfit_restores_rng_state(tmp_path, monkeypatch):
    """The overfit branch's own reseed-run-restore leaves no net trace on the streams: a plain
    smoke build (overfit=False) and the same build with overfit=True, run from the same seed,
    must consume the streams identically once the overfit call returns.
    """
    pytest.importorskip("torch")
    import functools
    import random

    import numpy as np
    import torch

    from tcip_mcp.pipelines import model_contract
    from tcip_mcp.tools.training_tools import preflight_config

    # RNG-stream identity holds after any positive step count; four steps exercise the same
    # reseed-run-restore path as the default twenty for a fraction of the wall time.
    monkeypatch.setattr(
        model_contract,
        "overfit_check",
        functools.partial(model_contract.overfit_check, steps=4),
    )

    cfg = _detection_smoke_cfg(BESPOKE_DETECTION, tmp_path)

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


def test_preflight_config_overfit_on_all_frozen_model_reports_without_raising(
    tmp_path,
):
    """check_model_contract's own gradient-presence check independently requires a real
    learnable path, so a literally all-frozen model already fails smoke on its own terms
    (valid stays False for that reason); what this proves is narrower and still real: the
    overfit branch's own empty-parameter case never raises out of preflight_config, and its
    report still names the empty parameter list rather than being swallowed.
    """
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    cfg = _detection_smoke_cfg(
        f"{Path(__file__).stem}:_frozen_detection_builder", tmp_path
    )
    r = preflight_config(tmp_path, cfg, smoke=True, overfit=True)
    assert any(
        "no parameter received a gradient" in i or "does not require grad" in i
        for i in r["issues"]
    )
    assert r["overfit_check"]["passed"] is False
    assert "empty parameter list" in r["overfit_check"]["issue"]


def test_preflight_config_refuses_a_per_stage_lr_by_name(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    cfg = training_config(
        tmp_path,
        DETECTION_SOURCE,
        _labeled(tmp_path),
        stages=[stage(-1, lr=1e-3)],
    )
    r = preflight_config(tmp_path, cfg)
    assert r["valid"] is False
    assert any(i.startswith("stages.0.lr:") for i in r["issues"]), r["issues"]

    cfg["stages"] = [stage(-1)]
    assert preflight_config(tmp_path, cfg)["valid"] is True


def test_preflight_config_warns_when_most_candidates_wont_train(tmp_path):
    """The admission's own partition is otherwise thrown away: a run whose label store admits
    only a fraction of its candidate images must not report "valid, no warnings" with no
    visibility into what would silently train on far fewer images than the operator expects.
    """
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images" / UNDATED_BUCKET
    # 1 annotated, 3 unannotated (no label document at all) -> 75% of candidates won't train.
    _bud_image(imgs / "ann.jpg")
    for stem in ("a", "b", "c"):
        _bud_image(imgs / f"{stem}.jpg", labeled=False)

    # One admitted image holds nothing out, so the run selects on its training loss.
    cfg = training_config(
        tmp_path,
        DETECTION_SOURCE,
        {
            "images_dir": str(imgs),
            "scope": {"subject": "bud"},
            "split": drawn_split(val_ratio=0.0),
        },
        evaluation=evaluation_block(selection_metric="loss"),
    )
    r = preflight_config(tmp_path, cfg)
    assert r["valid"] is True, r  # informational only, never gating
    assert any(
        "3/4 candidate images (75%) will not train" in w for w in r["warnings"]
    ), r["warnings"]
    assert any("{'absent': 3}" in w for w in r["warnings"])


def test_preflight_config_no_coverage_warning_when_everything_trains(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images" / UNDATED_BUCKET
    _bud_image(imgs / "ann.jpg")

    cfg = training_config(
        tmp_path,
        weights_statistics(DETECTION_SOURCE),
        {
            "images_dir": str(imgs),
            "scope": {"subject": "bud"},
            "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
        },
    )
    assert preflight_config(tmp_path, cfg)["warnings"] == []


# --------------------------------------------------------------------------
# preflight_config's training_source seam: a bare "module:function" string
# --------------------------------------------------------------------------


def test_preflight_config_blocks_rather_than_swallows_an_unreadable_label(
    tmp_path,
):
    """The coverage check's own admission must not fold an unreadable label into a generic build
    failure it silently drops: a run over these images would fail on the same document, so
    preflight reports it as a blocking issue, naming the document, not a warning; the launch's
    own admission refuses it in the same words.
    """
    pytest.importorskip("torch")
    from tcip_annotation import json_io
    from tcip_mcp.pipelines.data.split_construction import auto_train_val
    from tcip_mcp.pipelines.schemas import DataSpec
    from tcip_mcp.tools.training_tools import preflight_config
    from tests._producer_fixtures import staged_layout

    imgs = tmp_path / "images" / UNDATED_BUCKET
    _bud_image(imgs / "ann.jpg")
    _bud_image(imgs / "bad.jpg")
    _damaged(imgs / "bad.jpg", b"{not json")

    data_cfg = {
        "images_dir": str(imgs),
        "scope": {"subject": "bud"},
        "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
    }
    cfg = training_config(
        tmp_path, weights_statistics(DETECTION_SOURCE), dict(data_cfg)
    )
    r = preflight_config(tmp_path, cfg)
    with pytest.raises(
        json_io.UnreadableLabelDocumentError, match="bad"
    ) as raised:
        auto_train_val(
            tmp_path,
            "detection",
            DataSpec.model_validate(data_cfg),
            None,
            staged_layout(tmp_path, {}),
        )
    assert r["valid"] is False
    assert any(str(raised.value) in i for i in r["issues"]), r["issues"]


def test_preflight_admits_the_run_once(tmp_path):
    """Every preflight leg reads one population, so the producer is asked once however many legs
    need a source, a member name or a count: a second admission is a second answer to the one
    question of what this run trains on.
    """
    pytest.importorskip("torch")
    import tcip_mcp.pipelines.data.label_queries as label_queries
    from tcip_mcp.tools.training_tools import preflight_config

    images_dir, _stem = big_single_source(tmp_path / "ds", 4000, 3000)
    cfg = training_config(
        tmp_path,
        DETECTION_SOURCE,
        {
            "images_dir": str(images_dir),
            "scope": {"subject": "bud"},
            "tiling": tiling_block(tile_size=128, overlap=0.2),
            "split": drawn_split(
                calibration_ratio=0.15,
                val_ratio=0.2,
                holdout_ratio=0.1,
                seed=1,
            ),
        },
    )

    calls = []
    real_admit = label_queries.admit

    def counting_admit(*args, **kwargs):
        calls.append(args)
        return real_admit(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(label_queries, "admit", counting_admit)
        report = preflight_config(tmp_path, cfg, smoke=True)

    assert report["valid"] is True, report["issues"]
    assert len(calls) == 1, calls


@pytest.mark.parametrize(
    ("bad_document", "refusal"),
    [
        (
            '{"image": "bad", "width": 20, "height": 20, "annotations": 5}',
            "not the list a label document holds",
        ),
        (
            '{"image": "bad", "width": 20, "height": 20, "annotations": [7]}',
            "not an annotation object",
        ),
    ],
)
def test_preflight_config_blocks_a_document_only_the_admission_reader_refuses(
    tmp_path, bad_document, refusal
):
    """A document that decodes to a dict but whose annotations field is not a list, or whose
    record cannot be coerced, is exactly what the run's own admission refuses at launch:
    preflight reads through the same call so it blocks here too, rather than passing a document
    the launch then aborts on.
    """
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config
    from tcip_store import encode_record

    imgs = tmp_path / "images" / UNDATED_BUCKET
    _bud_image(imgs / "ann.jpg")
    _bud_image(imgs / "bad.jpg")
    _damaged(imgs / "bad.jpg", encode_record(json.loads(bad_document)))

    cfg = training_config(
        tmp_path,
        weights_statistics(DETECTION_SOURCE),
        {
            "images_dir": str(imgs),
            "scope": {"subject": "bud"},
            "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
        },
    )
    r = preflight_config(tmp_path, cfg)
    assert r["valid"] is False
    assert any(refusal in i for i in r["issues"]), r["issues"]


def test_preflight_config_training_source_shape_and_importability(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    base_cfg = training_config(tmp_path, DETECTION_SOURCE, _labeled(tmp_path))

    # A dict and an empty string are refused by the config's own schema.
    for stated in ({"train": BESPOKE_DETECTION}, ""):
        r = preflight_config(tmp_path, dict(base_cfg, training_source=stated))
        assert any(i.startswith("training_source: ") for i in r["issues"]), r[
            "issues"
        ]

    # A bare string naming a module no declared file holds is refused naming source_files.
    cfg = dict(base_cfg, training_source="nonexistent_module:train")
    r = preflight_config(tmp_path, cfg)
    assert any(
        i.startswith("source_files: ") and "nonexistent_module" in i
        for i in r["issues"]
    ), r["issues"]

    # A bare, importable string passes.
    cfg = dict(base_cfg, training_source=BESPOKE_DETECTION)
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

    base_cfg = training_config(tmp_path, DETECTION_SOURCE, _labeled(tmp_path))

    # A comparability-only metric for a center-match trait is rejected.
    cfg = dict(base_cfg)
    cfg["evaluation"] = evaluation_block(
        trait="bud_opening", selection_metric="map50"
    )
    r = preflight_config(tmp_path, cfg)
    assert any("comparability-only" in i for i in r["issues"])

    # A governing metric for the same trait is fine.
    cfg["evaluation"] = evaluation_block(
        trait="bud_opening", selection_metric="f1"
    )
    assert preflight_config(tmp_path, cfg)["valid"] is True

    # No trait -> no coherence gate, even for a comparability metric.
    cfg["evaluation"] = evaluation_block(selection_metric="map50")
    assert preflight_config(tmp_path, cfg)["valid"] is True

    # A metric with no declared direction is refused here by the field's type, before any run.
    cfg["evaluation"] = evaluation_block(selection_metric="not_a_real_metric")
    r = preflight_config(tmp_path, cfg)
    assert any("evaluation.selection_metric" in i for i in r["issues"]), r


def test_preflight_config_names_a_non_mapping_evaluation_block_as_an_issue(
    tmp_path,
):
    """A non-mapping ``evaluation`` block is a named issue, never an exception that crashes the
    preflight call.
    """
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images" / UNDATED_BUCKET
    imgs.mkdir(parents=True)
    cfg = training_config(
        tmp_path,
        weights_statistics(DETECTION_SOURCE),
        {
            "images_dir": str(imgs),
            "scope": {"subject": "bud"},
            "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
        },
        evaluation="not_a_mapping",
    )
    r = preflight_config(tmp_path, cfg)
    assert any(i.startswith("evaluation:") for i in r["issues"]), r["issues"]


# preflight_config's reserved-region feasibility check: a training-launch-time refusal through
# this module's own validation surface, never review_calibration._FAILURE_MESSAGES.


def test_preflight_calibration_ratio_wrong_task_flags_issue(tmp_path):
    """A classification run never resolves to the within-image split a calibration region
    reserves from, so its draw refuses the fraction by name.
    """
    from tcip_mcp.tools.training_tools import preflight_config
    from tests.tiny_trainer_fixtures import write_regression_dataset

    images_dir, csv_path = write_regression_dataset(
        tmp_path, [0.1, 0.3, 0.5, 0.7], [0, 1, 0, 1], height=32, width=32
    )
    cfg = training_config(
        tmp_path,
        weights_statistics(CLASSIFIER_SOURCE),
        {
            "images_dir": str(images_dir),
            "labels_dir": str(csv_path),
            "split": drawn_split(
                calibration_ratio=0.15, val_ratio=0.15, seed=1
            ),
        },
    )
    r = preflight_config(tmp_path, cfg)
    assert any(
        "calibration_ratio" in i and "reserved strips" in i
        for i in r["issues"]
    ), r["issues"]


def test_preflight_calibration_ratio_multi_member_flags_issue(tmp_path):
    """Two admitted members resolve to the group-balanced split, which reserves no calibration
    region, so its draw refuses the fraction by name.
    """
    from tcip_mcp.tools.training_tools import preflight_config

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    for stem in ("a", "b"):
        _bud_image(images_dir / f"{stem}.png")
    cfg = training_config(
        tmp_path,
        weights_statistics(DETECTION_SOURCE),
        {
            "images_dir": str(images_dir),
            "scope": {"subject": "bud"},
            # sliver_frac stated: two boxes derive no size spread.
            "tiling": tiling_block(tile_size=32, sliver_frac=0.5),
            "split": drawn_split(
                calibration_ratio=0.15, val_ratio=0.15, seed=1
            ),
        },
    )
    r = preflight_config(tmp_path, cfg)
    assert any(
        "calibration_ratio" in i and "reserved strips" in i
        for i in r["issues"]
    ), r["issues"]


def test_preflight_reserved_regions_infeasible_layout_refuses_under_smoke(
    tmp_path,
):
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    images_dir, _stem = big_single_source(tmp_path / "ds", 4000, 3000)
    cfg = training_config(
        tmp_path,
        weights_statistics(DETECTION_SOURCE),
        {
            "images_dir": str(images_dir),
            "scope": {"subject": "bud"},
            "tiling": tiling_block(tile_size=128, overlap=0.2),
            # Nothing left for a real train fraction at this mosaic size.
            "split": drawn_split(
                val_ratio=0.33,
                holdout_ratio=0.33,
                seed=1,
                calibration_ratio=0.33,
            ),
        },
    )
    r = preflight_config(tmp_path, cfg, smoke=True)
    assert any("is infeasible" in i for i in r["issues"]), r["issues"]

    # The run's resolution builds its datasets whether or not the model is smoked, so the same
    # geometry refuses without smoke too.
    r_no_smoke = preflight_config(tmp_path, cfg, smoke=False)
    assert any("is infeasible" in i for i in r_no_smoke["issues"])


def test_preflight_reserved_regions_report_an_unreadable_label_by_name(
    tmp_path,
):
    """This probe's own generic except Exception must not swallow an unreadable label into a
    silently-logged build failure: the breeder needs to see which document is broken.
    """
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    images_dir, _stem = big_single_source(tmp_path / "ds", 4000, 3000)
    _damaged(images_dir / "mosaic.png", b"{not json")
    cfg = training_config(
        tmp_path,
        weights_statistics(DETECTION_SOURCE),
        {
            "images_dir": str(images_dir),
            "scope": {"subject": "bud"},
            "tiling": tiling_block(tile_size=128, overlap=0.2),
            "split": drawn_split(
                val_ratio=0.2,
                holdout_ratio=0.1,
                seed=1,
                calibration_ratio=0.15,
            ),
        },
    )
    r = preflight_config(tmp_path, cfg, smoke=True)
    assert any("mosaic" in i and "decode" in i for i in r["issues"]), r[
        "issues"
    ]


def test_preflight_reserved_regions_admit_a_feasible_layout(tmp_path):
    """A real, feasible reserved-region config produces no spatial-split issue under
    smoke=True.
    """
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    images_dir, _stem = big_single_source(tmp_path / "ds", 4000, 3000)
    cfg = training_config(
        tmp_path,
        {
            "builder": BESPOKE_DETECTION,
            "source_files": [BESPOKE_MODELS],
            "task": "detection",
            "normalization": FRAME_STATISTICS,
        },
        {
            "images_dir": str(images_dir),
            "scope": {"subject": "bud"},
            "tiling": tiling_block(tile_size=128, overlap=0.2),
            "split": drawn_split(
                val_ratio=0.2,
                holdout_ratio=0.1,
                seed=1,
                calibration_ratio=0.15,
            ),
        },
    )
    r = preflight_config(tmp_path, cfg, smoke=True)
    assert not any(
        "spatial split" in i or "calibration_ratio" in i for i in r["issues"]
    ), r["issues"]


# --------------------------------------------------------------------------
# _apply_hpo_params: a swept value reaches what the trainer actually reads
# --------------------------------------------------------------------------


def _built_groups(config: dict) -> list[tuple[float, float]]:
    """``(lr, weight_decay)`` of each param group the trainer's optimizer read of ``config``
    (``schemas.train_config``'s ``optimizer`` into ``optimizer_factory.build_optimizer``)
    builds over a backbone-and-head model, the backbone group first.
    """
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.schemas import train_config
    from tcip_mcp.pipelines.training.optimizer_factory import build_optimizer
    from tests.tiny_trainer_fixtures import build_two_rate_regressor

    regime = train_config(config).trainer_reads().regime
    assert regime is not None
    spec = regime.optimizer
    optimizer = build_optimizer(
        spec,
        build_two_rate_regressor(),
        backbone_lr=spec.backbone_lr,
        head_lr=spec.head_lr,
    )
    return [(g["lr"], g["weight_decay"]) for g in optimizer.param_groups]


@pytest.mark.parametrize("training_source", [None, TRAIN_BESPOKE])
@pytest.mark.parametrize(
    "unstated, named",
    [
        ("conf_threshold", "evaluation.conf_threshold unstated"),
        ("batch_size", "batch_size unstated"),
        ("stages", "['stages'] unstated"),
    ],
)
def test_the_launch_door_refuses_what_a_trainer_reads_unstated_and_the_schema_admits_it(
    tmp_path, training_source, unstated, named
):
    """What a trainer reads is asked at the launch door, never by the schema: a detector config
    stating no ``evaluation.conf_threshold`` or no ``batch_size``, under the default trainer or a
    loop of its own, and a default-trainer config stating no ``stages``, are each refused by
    ``preflight_config`` naming the key, while ``train_config`` admits each as a checkpoint's
    config; a config naming its own loop and no ``stages`` is admitted by the door. A head that
    counts no boxes needs no confidence.
    """
    from tcip_mcp.pipelines.schemas import train_config
    from tcip_mcp.tools.training_tools import preflight_config

    loop = (
        {} if training_source is None else {"training_source": training_source}
    )
    config = training_config(
        tmp_path, DETECTION_SOURCE, _labeled(tmp_path), **loop
    )
    (config["evaluation"] if unstated == "conf_threshold" else config).pop(
        unstated
    )
    report = preflight_config(tmp_path, config)
    if unstated == "stages" and training_source is not None:
        assert report["valid"] is True, report["issues"]
    else:
        assert any(named in issue for issue in report["issues"]), report[
            "issues"
        ]
    train_config(config)

    classifier = training_config(
        tmp_path,
        weights_statistics(CLASSIFIER_SOURCE),
        {},
        evaluation=evaluation_block(
            selection_metric="loss", conf_threshold=None
        ),
        **loop,
    )
    assert train_config(classifier).trainer_reads().conf_threshold is None


def _applied(base: dict, params: dict) -> dict:
    """``params`` applied to ``base`` (``training_tools._apply_hpo_params``)."""
    from tcip_mcp.tools.training_tools import _apply_hpo_params

    return _apply_hpo_params(base, params)


def _base(builder: str = BESPOKE_DETECTION, **overrides) -> dict:
    """A detection base config of this repository of ``builder`` over no data, normalizing by
    its named weights' statistics (``_chain_fixtures.weights_statistics``), ``overrides`` in
    place of theirs.
    """
    from tests import REPO_ROOT

    return training_config(
        REPO_ROOT,
        weights_statistics(
            {
                "builder": builder,
                "source_files": [BESPOKE_MODELS],
                "task": "detection",
            }
        ),
        {},
        **overrides,
    )


def test_apply_hpo_params_optimizer_axes_reach_optimizer_param_groups():
    """Swept optimizer values survive the trainer's own optimizer read into each group's rate
    and every group's weight decay.
    """
    swept = {
        "optimizer.backbone_lr": 3e-4,
        "optimizer.head_lr": 3e-3,
        "optimizer.weight_decay": 2e-4,
    }
    (backbone_lr, backbone_decay), (head_lr, head_decay) = _built_groups(
        _applied(_base(), swept)
    )
    assert backbone_lr == pytest.approx(3e-4)
    assert head_lr == pytest.approx(3e-3)
    assert backbone_decay == head_decay == pytest.approx(2e-4)


def test_apply_hpo_params_preserves_base_config_stages():
    """Sweeping a rate must not overwrite the agent's own progressive-unfreeze schedule with a
    hardcoded recipe: base_config's stages survive unchanged.
    """
    custom_stages = [stage(-1), stage(2), stage(0)]
    out = _applied(
        _base("x:y", stages=custom_stages), {"optimizer.head_lr": 3e-3}
    )
    assert out["stages"] == custom_stages


def test_apply_hpo_params_undotted_key_reaches_top_level():
    """An undotted swept key must land at the top level of the resolved config, the one
    placement train() reads, never nested under "training", which the config schema refuses.
    """
    out = _applied(_base("x:y"), {"batch_size": 4})
    assert out["batch_size"] == 4
    assert "training" not in out


def test_apply_hpo_params_dotted_key_reaches_nested_field():
    """A dotted param (e.g. a swept builder) reaches the nested field it names."""
    out = _applied(
        _base("old:builder"), {"model_source.builder": "new:builder"}
    )
    assert out["model_source"]["builder"] == "new:builder"
    assert (
        out["model_source"]["task"] == "detection"
    )  # the rest of the mapping survives


def test_apply_hpo_params_dotted_seed_key_reaches_the_split_config():
    """split_draws's own paired grid axis (data.split.seed) reaches the nested field
    run_hyperparameter_search's drawn path reads, the same dotted-key mechanism a swept
    builder uses above.
    """
    out = _applied(_base("x:y"), {"data.split.seed": 7})
    assert out["data"]["split"]["seed"] == 7


def test_apply_hpo_params_refuses_a_dotted_key_through_a_non_mapping_intermediate():
    """A dotted key whose path walks through a value that is not a mapping is refused by name,
    naming the key and what was found there, rather than raising an opaque AttributeError.
    """
    with pytest.raises(ValueError, match="model_source.builder.x"):
        _applied(_base("x:y"), {"model_source.builder.x": "y"})


def test_preflight_points_covers_every_categorical_choice_and_both_numeric_bounds():
    """The preflight checks more than the first corner: every categorical choice and both numeric
    bounds (low and high) are held by some point, and no point is listed twice.
    """
    from tcip_mcp.tools.training_tools import _preflight_points

    space = {
        "model_source.builder": {
            "type": "categorical",
            "choices": ["a:b", "c:d", "e:f"],
        },
        "optimizer.head_lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
    }
    points = _preflight_points(space, "random")

    assert {p["model_source.builder"] for _, p in points} == {
        "a:b",
        "c:d",
        "e:f",
    }
    assert {p["optimizer.head_lr"] for _, p in points} == {1e-5, 1e-2}
    held = [sorted(p.items()) for _, p in points]
    assert len(held) == len({repr(p) for p in held}) == 4, points

    repeated = _preflight_points(
        {"batch_size": {"type": "categorical", "choices": [2, 4, 4]}}, "random"
    )
    assert [p for _, p in repeated] == [
        {"batch_size": 2},
        {"batch_size": 4},
    ], repeated


def test_a_trials_combined_point_is_admitted_as_the_launch_door_admits_it(
    tmp_path,
):
    """Each point a sweep checks before it opens can pass while a combination of two axes away
    from the corner does not: a default-trainer config with no ``stages``. Its trial ends failed
    naming why, resolving nothing; a combination the door admits opens its trial resolved.
    """
    from tcip_mcp.experiments import RUN_FILE, observe, read_record
    from tcip_mcp.tools.training_tools import open_trial
    from tests._verified_checkpoint_fixtures import opened_sweep

    base = training_config(tmp_path, DETECTION_SOURCE, _labeled(tmp_path))
    sweep = opened_sweep(
        tmp_path,
        base,
        param_space={
            "training_source": {
                "type": "categorical",
                "choices": [TRAIN_BESPOKE, None],
            },
            "stages": {
                "type": "categorical",
                "choices": [base["stages"], None],
            },
        },
    )

    refused = open_trial(
        sweep, "refused", {"training_source": None, "stages": None}
    )
    final = observe(refused).final
    assert final is not None and final["state"] == "failed"
    assert "['stages'] unstated" in final["status_error"]
    assert read_record(refused / RUN_FILE)["resolved"] is None
    admitted = open_trial(
        sweep, "admitted", {"training_source": TRAIN_BESPOKE, "stages": None}
    )
    assert observe(admitted).final is None
    assert read_record(admitted / RUN_FILE)["resolved"] is not None


# --------------------------------------------------------------------------
# _run_hpo_trial: each epoch's selection value is reported as it completes; only a completed
# trial is eligible for the sweep's best. Run directly (no Ray) so training can be stubbed.
# --------------------------------------------------------------------------


class _FakeDataset:
    def __len__(self):
        return 4

    def __getitem__(self, i):
        return i


def _detection_base(project: Path) -> dict:
    """A detection base config of a model no trial here builds
    (:func:`~tests._verified_checkpoint_fixtures.unbuilt_source`) over :func:`_labeled`'s frames
    under ``project``, selecting on the loss, its trials' partitions stubbed empty
    (:func:`_patch_hpo_trial_machinery`; ``_chain_fixtures.weights_statistics``).
    """
    from tests._verified_checkpoint_fixtures import unbuilt_source

    return training_config(
        project,
        weights_statistics(unbuilt_source("detection")),
        _labeled(project),
        evaluation=evaluation_block(selection_metric="loss"),
    )


def _classifier_base(project: Path, **overrides) -> dict:
    """A classification base config over :func:`_labeled`'s frames under ``project``, selecting
    on accuracy, ``overrides`` in place of their keys, its trials' partitions stubbed empty
    (:func:`_patch_hpo_trial_machinery`; ``_chain_fixtures.weights_statistics``).
    """
    return training_config(
        project,
        weights_statistics(CLASSIFIER_SOURCE),
        _labeled(project),
        evaluation=evaluation_block(
            selection_metric="accuracy",
            conf_threshold=None,
            score_weights=None,
        ),
        **overrides,
    )


def _the_sweep(project: Path) -> Path:
    """The one sweep directory of ``project``."""
    from tcip_mcp import experiments

    (sweep,) = experiments.sweep_dirs(project)
    return sweep


def _trial(
    point: dict, report, base: dict, project: Path, name: str = "t0"
) -> Path:
    """One HPO trial ``name`` run as its sweep runs it (``_run_hpo_trial``), under the one sweep
    of ``project`` over ``base`` the sweep door opens (``opened_sweep``) on its first trial.
    Returns the trial's run directory.
    """
    from tcip_mcp import experiments
    from tcip_mcp.tools.training_tools import _run_hpo_trial
    from tests._verified_checkpoint_fixtures import opened_sweep

    sweep = (
        _the_sweep(project)
        if experiments.sweep_dirs(project)
        else opened_sweep(project, base)
    )
    _run_hpo_trial(point, report, sweep, name)
    return sweep / f"{sweep.name}_{name}"


def _row(value: float, metric: str = "loss") -> dict:
    """An epoch row stamping ``metric`` as the run's selection, the way the trainer stamps it."""
    return {"selection_metric": metric, "selection": value}


def _complete(run):
    """End a stand-in training body the way a real one does: final weights, here a stand-in
    tensor since no model was built, saved under ``model_final`` through the run's own writer
    (``TrainContext.save_checkpoint``).
    """
    import torch

    from tcip_mcp.pipelines.model_build import STATE_DICT_KEY
    from tcip_mcp.pipelines.training.envelope import TrainContext

    TrainContext(run=run, train_loader=None).save_checkpoint(
        {STATE_DICT_KEY: {"weight": torch.zeros(1)}}, "model_final"
    )
    run.status = "completed"
    return run


def _sweep_outcome(project: Path) -> dict:
    """What the trials of :func:`_trial`'s sweep amount to, as the project's listing reads it."""
    from tcip_mcp.experiments import training_listing

    (sweep,) = training_listing(project).sweeps
    return sweep.outcome


def _patch_hpo_trial_machinery(monkeypatch, fake_train, captured=None):
    """Stub dataset building + training so a trial runs instantly, no Ray: the resolution builds
    stand-in datasets, records no samples and stamps nothing they serve, and the child builds the
    same stand-ins from that record.
    """
    from tcip_mcp.pipelines.data import samplers, split_construction as sc
    from tcip_mcp.pipelines.schemas import TrainGeometry
    from tcip_mcp.pipelines.training import generic_trainer as gt

    ds = _FakeDataset()

    def fake_auto_train_val(project, task, data, transforms, plan, **_):
        if captured is not None:
            captured["transforms"] = transforms
            captured["data"] = data
        return (
            ds,
            ds,
            sc._partition_record(
                [], seed=None, group_by="stem", selection=None, spatial=None
            ),
            data,
        )

    monkeypatch.setattr(sc, "auto_train_val", fake_auto_train_val)
    monkeypatch.setattr(sc, "recorded_datasets", lambda *a, **k: (ds, ds))
    monkeypatch.setattr(
        gt,
        "effective_data_geometry",
        lambda task, data, train_ds, stated: TrainGeometry(
            native_size=None, object_density=None, detector=None
        ),
    )
    monkeypatch.setattr(gt, "train", fake_train)
    monkeypatch.setattr(samplers, "build_sampler", lambda *a, **k: None)


def _spaces_searched(monkeypatch) -> list:
    """Replace ``tune_search`` with a search that answers its study directory, and the list it
    records each param space it ran over into.
    """
    from tcip_mcp.pipelines.training import hpo

    seen: list = []

    def fake_search(*args, **kwargs):
        seen.append(
            kwargs.get("param_space", args[1] if len(args) > 1 else None)
        )

    monkeypatch.setattr(hpo, "tune_search", fake_search)
    return seen


def _completed_train(
    run,
    train_loader,
    val_loader,
    epoch_callback=None,
    batch_callback=None,
    resume_from="",
):
    """A training call that completes the run at once, reporting no epoch."""
    run.status = "completed"
    return run


def test_run_hpo_trial_reports_each_epoch_and_its_result_is_the_best_of_them(
    monkeypatch, tmp_path
):
    """Every epoch's selection value is reported as the trial logs it, and the trial's result is
    the best of those same rows (lower=better).
    """
    pytest.importorskip("torch")

    def fake_train(
        run,
        train_loader,
        val_loader,
        epoch_callback=None,
        batch_callback=None,
        resume_from="",
    ):
        for epoch, value in enumerate([50.0, 40.0, 30.0]):
            if epoch_callback:
                epoch_callback(epoch, _row(value))
        return _complete(run)

    _patch_hpo_trial_machinery(monkeypatch, fake_train)
    reported: list = []
    _trial(
        {"optimizer.head_lr": 3e-4},
        reported.append,
        _detection_base(tmp_path),
        tmp_path,
    )
    assert reported == [50.0, 40.0, 30.0]
    assert _sweep_outcome(tmp_path)["best_value"] == 30.0


@pytest.mark.parametrize(
    ("head_lr", "at_end"),
    [(1e-2, {"optimizer.head_lr": "high"}), (3e-4, {})],
)
def test_a_sweeps_best_at_an_end_of_its_stated_range_is_reported(
    monkeypatch, tmp_path, head_lr, at_end
):
    """A best whose value sits at the stated ``low`` or ``high`` of its axis is named with that
    end, a range the search could not look past; a best inside the range names none.
    """
    pytest.importorskip("torch")

    def fake_train(
        run,
        train_loader,
        val_loader,
        epoch_callback=None,
        batch_callback=None,
        resume_from="",
    ):
        if epoch_callback:
            epoch_callback(0, _row(30.0))
        return _complete(run)

    _patch_hpo_trial_machinery(monkeypatch, fake_train)
    _trial(
        {"optimizer.head_lr": head_lr},
        lambda _value: None,
        _detection_base(tmp_path),
        tmp_path,
    )

    assert _sweep_outcome(tmp_path)["best_at_range_end"] == at_end


def test_run_hpo_trial_that_fails_has_no_result(monkeypatch, tmp_path):
    """A crashed trial ends failed and carries no result, so it can never become the sweep's
    best.
    """
    pytest.importorskip("torch")
    from tcip_mcp.experiments import observe

    def fake_train(
        run,
        train_loader,
        val_loader,
        epoch_callback=None,
        batch_callback=None,
        resume_from="",
    ):
        raise RuntimeError("CUDA out of memory")

    _patch_hpo_trial_machinery(monkeypatch, fake_train)
    reported: list = []
    trial_dir = _trial(
        {"optimizer.head_lr": 3e-4},
        reported.append,
        _detection_base(tmp_path),
        tmp_path,
    )
    assert reported == []
    assert observe(trial_dir).state == "failed"
    assert _sweep_outcome(tmp_path)["best_value"] is None


def test_run_hpo_trial_result_is_the_highest_value_for_a_higher_is_better_metric(
    monkeypatch, tmp_path
):
    """A higher-is-better selection metric (accuracy) makes the trial's result its highest
    reported value, not a minimize convention that would instead prefer the lowest.
    """
    pytest.importorskip("torch")

    def fake_train(
        run,
        train_loader,
        val_loader,
        epoch_callback=None,
        batch_callback=None,
        resume_from="",
    ):
        for epoch, value in enumerate([0.5, 0.9, 0.6]):
            if epoch_callback:
                epoch_callback(epoch, _row(value, metric="accuracy"))
        return _complete(run)

    _patch_hpo_trial_machinery(monkeypatch, fake_train)
    reported: list = []
    _trial(
        {"optimizer.head_lr": 3e-4},
        reported.append,
        _classifier_base(tmp_path),
        tmp_path,
    )
    assert reported == [0.5, 0.9, 0.6]
    assert _sweep_outcome(tmp_path)["best_value"] == 0.9


def test_a_failed_trial_never_outranks_a_real_one_under_a_maximize_direction(
    monkeypatch,
    tmp_path,
):
    """A trial that fails carries no result, so the sweep's outcome under a maximize direction
    is the one completed trial's, never the failed one's.
    """
    pytest.importorskip("torch")

    base = _classifier_base(tmp_path)

    def fake_train_ok(
        run,
        train_loader,
        val_loader,
        epoch_callback=None,
        batch_callback=None,
        resume_from="",
    ):
        if epoch_callback:
            epoch_callback(0, _row(0.7, metric="accuracy"))
        return _complete(run)

    _patch_hpo_trial_machinery(monkeypatch, fake_train_ok)
    real: list = []
    _trial({"optimizer.head_lr": 3e-4}, real.append, base, tmp_path, "real")

    def fake_train_fails(
        run,
        train_loader,
        val_loader,
        epoch_callback=None,
        batch_callback=None,
        resume_from="",
    ):
        raise RuntimeError("boom")

    _patch_hpo_trial_machinery(monkeypatch, fake_train_fails)
    failed: list = []
    _trial(
        {"optimizer.head_lr": 1e-2}, failed.append, base, tmp_path, "failed"
    )

    outcome = _sweep_outcome(tmp_path)
    assert (outcome["best_params"], outcome["best_value"]) == (
        {"optimizer.head_lr": 3e-4},
        0.7,
    )


def test_run_hpo_trial_uses_base_augmentation_and_model(monkeypatch, tmp_path):
    """Trials train under the final run's regime: base_config augmentation reaches the train
    dataset, and the bespoke model_source is carried through. (Loss is owned by the builder.)
    """
    pytest.importorskip("torch")

    captured: dict = {}

    def fake_train(
        run,
        train_loader,
        val_loader,
        epoch_callback=None,
        batch_callback=None,
        resume_from="",
    ):
        captured["model_source"] = run.spec.model_source.model_dump(
            exclude_unset=True
        )
        run.status = "completed"
        return run

    _patch_hpo_trial_machinery(monkeypatch, fake_train, captured=captured)
    new_project(tmp_path)
    base = _classifier_base(
        tmp_path, augmentation=augmentation(horizontal_flip={"p": 0.5})
    )
    _trial({"optimizer.head_lr": 3e-4}, [].append, base, tmp_path)
    assert (
        captured["transforms"] is not None
    )  # augmentation was built + passed
    assert captured["model_source"]["builder"] == BESPOKE_CLASSIFIER


def test_run_hpo_trial_dotted_seed_axis_reaches_the_data_cfg_handed_to_auto_train_val(
    monkeypatch,
    tmp_path,
):
    """The split_draws grid axis (data.split.seed) resolves onto the config before the trial's
    data section is built, so auto_train_val (and the drawn split behind it) actually reads the
    draw's own seed rather than the base config's unswept one.
    """
    pytest.importorskip("torch")

    captured: dict = {}
    _patch_hpo_trial_machinery(
        monkeypatch, _completed_train, captured=captured
    )
    _trial(
        {"data.split.seed": 7}, [].append, _detection_base(tmp_path), tmp_path
    )
    assert captured["data"].split.seed == 7


def test_run_hpo_trial_dotted_seed_axis_reaches_the_trials_own_records(
    monkeypatch, tmp_path
):
    """data.split.seed, the split_draws grid axis, lands in the trial's launch record and in the
    data section it resolved, at the nested field auto_train_val reads it from, beside the
    sampled point itself.
    """
    pytest.importorskip("torch")
    from tcip_mcp.experiments import RUN_FILE, read_record

    _patch_hpo_trial_machinery(monkeypatch, _completed_train)

    trial_dir = _trial(
        {"data.split.seed": 7}, [].append, _detection_base(tmp_path), tmp_path
    )

    run = read_record(trial_dir / RUN_FILE)
    assert run["config"]["data"]["split"]["seed"] == 7
    assert run["trial_params"] == {"data.split.seed": 7}
    assert run["resolved"]["data"]["split"]["seed"] == 7


def test_run_hpo_trial_producer_fed_data_split_seed_over_the_single_source_spatial_path(
    monkeypatch,
    tmp_path,
):
    """The producer path: a real one-source tiled dataset through the real, unstubbed
    auto_train_val. Its single-source spatial-strip branch places every strip by declared order
    alone, and the trial's launch record carries the real spatial split and tiling
    auto_train_val wrote, with no seed inside the split it never read one for.
    """
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from tcip_mcp.experiments import RUN_FILE, read_record

    images_dir, _stem = big_single_source(tmp_path / "ds", 4000, 3000)
    base = training_config(
        tmp_path,
        DETECTION_SOURCE,
        {
            "images_dir": str(images_dir),
            "scope": {"subject": "bud"},
            "tiling": tiling_block(tile_size=128, overlap=0.2),
            "split": drawn_split(
                val_ratio=0.25, holdout_ratio=0.1, grouping="tile_prefix"
            ),
        },
    )

    from tcip_mcp.pipelines.data import samplers
    from tcip_mcp.pipelines.training import generic_trainer as gt

    monkeypatch.setattr(gt, "train", _completed_train)
    monkeypatch.setattr(samplers, "build_sampler", lambda *a, **k: None)

    trial_dir = _trial({"data.split.seed": 3}, [].append, base, tmp_path)

    resolved = read_record(trial_dir / RUN_FILE)["resolved"]
    assert resolved["partition"]["seed"] is None
    assert resolved["partition"]["spatial"]
    assert "seed" not in resolved["partition"]["spatial"]
    assert resolved["data"]["tiling"]["tile_size"] == 128


def test_a_trials_launch_record_carries_the_seed_it_trained_under(
    monkeypatch, tmp_path
):
    """A trial's launch record carries the seed its sweep's base config states, the one every
    trial of the sweep trains under, and the body trains under that recorded seed.
    """
    pytest.importorskip("torch")
    from tcip_mcp.experiments import RUN_FILE, read_record

    captured: dict = {}

    def fake_train(
        run,
        train_loader,
        val_loader,
        epoch_callback=None,
        batch_callback=None,
        resume_from="",
    ):
        captured["seed"] = run.spec.seed
        run.status = "completed"
        return run

    _patch_hpo_trial_machinery(monkeypatch, fake_train)
    base = {**_detection_base(tmp_path), "seed": 11}
    trial_dir = _trial({"optimizer.head_lr": 3e-4}, [].append, base, tmp_path)

    recorded = read_record(trial_dir / RUN_FILE)["config"]["seed"]
    assert recorded == captured["seed"] == 11


def test_run_hpo_trial_diverged_run_never_outranks_a_worse_but_alive_config(
    tmp_path,
):
    """A trial that trains one real epoch and then diverges ends failed and carries no result,
    not that epoch's real score, so it can never outrank a config that only scored worse. Drives
    the real training body (nothing mocked).

    Every share zero, so the score ranked is the training loss over every admitted row: what is
    under test is how a diverged run ranks, and a drawn validation side would spend this model's
    one good forward on measuring rather than training.
    """
    pytest.importorskip("torch")
    from tests.tiny_trainer_fixtures import (
        regressor_config,
        write_regression_dataset,
    )

    images_dir, csv_path = write_regression_dataset(
        tmp_path, intensities=[0.1, 0.3, 0.5, 0.7], values=[0.2, 0.6, 1.0, 1.4]
    )

    base_config = regressor_config(
        5,
        builder="tiny_trainer_fixtures:build_diverges_after_model",
        builder_kwargs={"good_calls": 1},
        batch_size=4,
        data={
            "images_dir": str(images_dir),
            "labels_dir": str(csv_path),
            "split": drawn_split(val_ratio=0.0),
        },
        project=tmp_path,
    )
    reported: list = []
    _trial({}, reported.append, base_config, tmp_path)

    import math

    assert math.isfinite(
        reported[0]
    )  # epoch 1's real score, reported before the run died
    assert _sweep_outcome(tmp_path)["best_value"] is None


# --------------------------------------------------------------------------
# get_worst_predictions: confidence comes from the canonical prediction format
# --------------------------------------------------------------------------


def test_get_worst_predictions_reads_canonical_confidence(tmp_path):
    """Prediction documents carry a native ``score``; confidence reads from that field, not from
    box geometry. Reading a normalized box height as confidence instead would make the
    (1 - avg_conf) ranking term ~1.0 for every image with small boxes (e.g. buds).
    """
    pytest.importorskip("torch")
    from PIL import Image

    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.vision_tools import get_worst_predictions
    from tests._chain_fixtures import published
    from tests._producer_fixtures import label_image

    images = tmp_path / "images" / UNDATED_BUCKET
    images.mkdir(parents=True)
    scored = {"confident": [0.9, 0.9], "shaky": [0.1, 0.1]}
    for stem, scores in scored.items():
        Image.new("RGB", (100, 100)).save(images / f"{stem}.png")
        # Matching GT count → missed = extra = 0, error is exactly (1 - avg_conf).
        gt_anns = [
            Annotation(subject="bud", geometry=BBox(20.0, 11.0, 40.0, 31.0))
            for _ in scores
        ]
        label_image(images / f"{stem}.png", gt_anns, 100, 100)
    # Confidence lives in the document's `score`; box geometry is irrelevant to this count +
    # confidence heuristic (no IoU matching), so the boxes can be anything.
    bucket = published(
        tmp_path,
        "preds",
        [
            {
                "image": str(images / f"{stem}.png"),
                "width": 100,
                "height": 100,
                "boxes": [[10.0, 10.0, 40.0, 22.0]] * len(scores),
                "scores": scores,
                "labels": [1] * len(scores),
                "cap": len(scores) + 1,
            }
            for stem, scores in scored.items()
        ],
        scope={"subject": "bud"},
    )

    out = get_worst_predictions(bucket, top_k=2, **TRIAGE_WEIGHTS)
    by_stem = {w["stem"]: w["error_score"] for w in out["worst_images"]}
    assert by_stem["confident"] == pytest.approx(0.1, abs=1e-3)
    assert by_stem["shaky"] == pytest.approx(0.9, abs=1e-3)
    assert (
        out["worst_images"][0]["stem"] == "shaky"
    )  # low confidence ranks worst


def test_a_launch_config_that_json_cannot_hold_is_refused_before_the_run_starts(
    tmp_path, monkeypatch
):
    """The caller's config reaches the run's record, so the field that will not encode is named
    before any write and before a subprocess is spawned.
    """
    from tcip_mcp.tools import training_tools

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        training_tools,
        "_preflight",
        lambda project, spec, issues, *, smoke, overfit: (
            {"valid": False, "issues": ["stub"]},
            None,
        ),
    )

    with pytest.raises(TypeError) as refused:
        training_tools.launch_training(
            tmp_path, {"model_source": {"builder": Path("m.py")}}, actor=None
        )
    assert "config.model_source.builder" in str(refused.value)


def test_an_ordinary_launch_config_passes_the_boundary_to_preflight(
    tmp_path, monkeypatch
):
    """The refusal above must not stop a legitimate config: it reaches preflight admitted, with
    no issue.
    """
    from tcip_mcp.tools import training_tools

    monkeypatch.chdir(tmp_path)
    seen = []

    def stub_preflight(project, spec, plan, issues, *, smoke, overfit):
        seen.append((spec, issues))
        return {"valid": False, "issues": ["stub"]}, None

    monkeypatch.setattr(training_tools, "_preflight", stub_preflight)
    config = training_config(tmp_path, DETECTION_SOURCE, _labeled(tmp_path))

    result = training_tools.launch_training(tmp_path, config, actor=None)

    assert result == {"error": "Invalid config", "issues": ["stub"]}
    [(spec, issues)] = seen
    assert spec is not None and issues == []


def test_a_sweep_payload_that_json_cannot_hold_is_refused_before_any_trial_runs(
    tmp_path, monkeypatch
):
    """The space reaches the sweep manifest and the base config reaches every trial's resolved
    config, so both are checked before a single trial is trained against them.
    """
    from tcip_mcp.pipelines.training import hpo
    from tcip_mcp.tools import training_tools

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        hpo,
        "tune_search",
        lambda *args, **kwargs: {
            "best_params": {},
            "best_value": 0.1,
            "n_trials": 1,
        },
    )

    with pytest.raises(TypeError) as space_refused:
        training_tools.run_hyperparameter_search(
            tmp_path,
            {"model_source": {"builder": "m:f"}},
            param_space={"lr": Path("lr.txt")},
            n_trials=1,
            **fifo_search(),
            search_seed=0,
        )
    assert "param_space.lr" in str(space_refused.value)

    with pytest.raises(TypeError) as config_refused:
        training_tools.run_hyperparameter_search(
            tmp_path,
            {"model_source": {"builder": Path("m.py")}},
            param_space=_CHOICE_SPACE,
            n_trials=1,
            **fifo_search(),
            search_seed=0,
        )
    assert "base_config.model_source.builder" in str(config_refused.value)


_CHOICE_SPACE = {
    "optimizer.head_lr": {"type": "categorical", "choices": [0.1, 0.01]}
}
"""A sample space of two head learning rates."""


def test_a_space_not_in_the_search_space_shape_is_refused_before_any_trial(
    tmp_path, monkeypatch
):
    """The sweep door reads the space through the one reader of its shape: a list where an axis
    spec belongs is refused by name, never taken as a value.
    """
    from tcip_mcp.tools import training_tools

    monkeypatch.chdir(tmp_path)
    seen = _spaces_searched(monkeypatch)

    result = training_tools.run_hyperparameter_search(
        tmp_path,
        training_config(tmp_path, DETECTION_SOURCE, _labeled(tmp_path)),
        param_space={"optimizer.head_lr": [0.1, 0.01]},
        n_trials=1,
        **fifo_search(),
        search_seed=0,
    )

    assert "param_space is not a search space" in result["error"], result
    assert seen == []


def test_an_ordinary_sweep_payload_still_runs_its_search(
    tmp_path, monkeypatch
):
    """The refusal above must not cost a legitimate sweep its search: admits valid work through
    the sweep door's structural preflight (an importable builder, a real data section).
    """
    from tcip_mcp.tools import training_tools

    monkeypatch.chdir(tmp_path)
    seen = _spaces_searched(monkeypatch)

    base_config = training_config(
        tmp_path, DETECTION_SOURCE, _labeled(tmp_path)
    )
    result = training_tools.run_hyperparameter_search(
        tmp_path,
        base_config,
        param_space=_CHOICE_SPACE,
        n_trials=1,
        **fifo_search(),
        search_seed=0,
    )

    assert result["sweep"]["state"] == "completed", result
    assert seen == [_CHOICE_SPACE]


@pytest.mark.parametrize(
    "choices, label",
    [
        ([-1.0, 1e-3], "the first sampled point"),
        ([1e-3, -1.0], "optimizer.head_lr=-1.0"),
    ],
    ids=["the-first-corner", "a-later-corner"],
)
def test_a_sweep_opens_over_a_base_omitting_what_its_space_sweeps_and_refuses_an_invalid_point(
    tmp_path,
    monkeypatch,
    choices,
    label,
):
    """A base config may leave unstated what its search space names: the sweep validates each
    point applied to it, so a base omitting ``optimizer.head_lr`` opens over a space naming it,
    and a corner whose applied config the schema refuses is named by its path and its point's
    label with nothing opened, whether it is the first corner or a later one.
    """
    from tcip_mcp import experiments
    from tcip_mcp.tools import training_tools
    from tests._verified_checkpoint_fixtures import SWEEP_ARGUMENTS

    monkeypatch.chdir(tmp_path)
    base = training_config(tmp_path, DETECTION_SOURCE, _labeled(tmp_path))
    del base["optimizer"]["head_lr"]
    arguments = {
        k: v for k, v in SWEEP_ARGUMENTS.items() if k != "param_space"
    }

    refused = training_tools.open_sweep(
        tmp_path,
        base,
        {"optimizer.head_lr": {"type": "categorical", "choices": choices}},
        actor=None,
        **arguments,
    )

    assert isinstance(refused, dict), refused
    assert (
        refused["error"]
        == f"the sweep's base config fails preflight at {label}"
    ), refused
    assert any(
        issue.startswith("optimizer.head_lr") for issue in refused["issues"]
    ), refused
    assert experiments.sweep_dirs(tmp_path) == []

    opened = training_tools.open_sweep(
        tmp_path,
        base,
        {
            "optimizer.head_lr": {
                "type": "categorical",
                "choices": [1e-3, 2e-3],
            }
        },
        actor=None,
        **arguments,
    )

    assert isinstance(opened, Path), opened
    assert experiments.sweep_dirs(tmp_path) == [opened]


def _project_answering(project: Path, **answers: bool) -> None:
    """``project`` made a project through ``initialize_project``, its capture invariances the
    sample answer (``_web_fixtures.INVARIANCES``) with ``answers`` over it.
    """
    from tcip_mcp.tools.project_tools import initialize_project

    result = initialize_project(
        str(project),
        "Test project",
        "north orchard",
        INVARIANCES.model_copy(update=answers),
    )
    assert "error" not in result, result


def _door_issues(door: str, project: Path, config: dict, monkeypatch) -> list:
    """The issues ``door`` answers ``config`` with over ``project``: ``preflight_config``'s
    report, or the issues ``launch_training`` hands its preflight (stubbed so nothing runs).
    """
    from tcip_mcp.tools import training_tools

    if door == "preflight_config":
        return training_tools.preflight_config(project, config)["issues"]
    seen: list = []

    def stub_preflight(project, spec, sources, issues, *, smoke, overfit):
        seen.append(issues)
        return {"valid": False, "issues": ["stub"]}, None

    monkeypatch.setattr(training_tools, "_preflight", stub_preflight)
    training_tools.launch_training(project, config, actor=None)
    (issues,) = seen
    return issues


_EVERY_INVARIANCE_FALSE = {
    "horizontal_flip": False,
    "vertical_flip": False,
    "rotation": False,
    "color": False,
    "intensity": False,
    "scale": False,
}
"""A record stating no capture transform leaves the trait's measurement unchanged."""

_BRIGHTNESS_ALONE = {
    "color_jitter": {"brightness": 0.2, "contrast": 0.0, "saturation": 0.0}
}
"""A color jitter shifting brightness and leaving contrast and saturation unchanged."""


@pytest.mark.parametrize("door", ["preflight_config", "launch_training"])
@pytest.mark.parametrize(
    "answers, transforms, refused",
    [
        (
            {"horizontal_flip": False},
            {"horizontal_flip": {"p": 0.5}},
            ["horizontal_flip"],
        ),
        ({}, {"horizontal_flip": {"p": 0.5}}, []),
        ({"horizontal_flip": False}, {"horizontal_flip": {"p": 0.0}}, []),
        ({"rotation": False}, {"rotation": {"degrees": 30, "p": 0.0}}, []),
        (
            {"rotation": False},
            {"rotation": {"degrees": 30, "p": 0.5}},
            ["rotation"],
        ),
        (
            {"scale": False},
            {
                "random_crop": {
                    "size": [20, 20],
                    "min_scale": 0.5,
                    "max_scale": 0.5,
                }
            },
            ["scale"],
        ),
        (
            _EVERY_INVARIANCE_FALSE,
            {
                "gaussian_blur": {"p": 0.5, "radius": 1.0},
                "resize": {"size": [20, 20]},
            },
            [],
        ),
        ({"color": False}, _BRIGHTNESS_ALONE, []),
        ({"intensity": False}, _BRIGHTNESS_ALONE, ["intensity"]),
    ],
    ids=[
        "a-flip-stated-invalid",
        "a-flip-stated-valid",
        "a-flip-never-taken-under-no-flip",
        "a-rotation-never-taken-under-no-rotation",
        "a-rotation-taken-under-no-rotation",
        "a-crop-under-no-scale",
        "blur-and-resize-under-every-record",
        "brightness-alone-under-no-color",
        "brightness-alone-under-no-intensity",
    ],
)
def test_the_launch_doors_judge_each_augmentation_by_the_invariance_the_project_records(
    tmp_path, monkeypatch, door, answers, transforms, refused
):
    """A transform relying on an invariance the record states false is refused naming the
    transform, the invariance and the door that states it; the same transform under a record
    stating it true is admitted, a flip or a rotation stated at probability zero relies on
    nothing, a crop relies on scale whatever its bounds, a blur and a resize rely on none, and a
    color jitter whose saturation is zero relies on intensity alone.
    """
    pytest.importorskip("torch")

    _project_answering(tmp_path, **answers)
    config = training_config(
        tmp_path,
        DETECTION_SOURCE,
        _labeled(tmp_path),
        augmentation=augmentation(**transforms),
    )

    issues = _door_issues(door, tmp_path, config, monkeypatch)

    if not refused:
        assert issues == [], issues
    named = [issue for issue in issues if "capture invariance" in issue]
    assert len(named) == len(refused), issues
    for fact, issue in zip(refused, named, strict=True):
        assert repr(next(iter(transforms))) in issue, issue
        assert repr(fact) in issue and "initialize_project" in issue, issue


@pytest.mark.parametrize(
    "transforms, field, refusal",
    [
        ({"horizontal_flip": {"p": -0.5}}, "p", "greater than or equal to 0"),
        (
            {
                "color_jitter": {
                    "brightness": -0.2,
                    "contrast": 0.0,
                    "saturation": 0.0,
                }
            },
            "brightness",
            "greater than or equal to 0",
        ),
        (
            {"rotation": {"degrees": "inf", "p": 1.0}},
            "degrees",
            "finite number",
        ),
        (
            {"rotation": {"degrees": 181, "p": 1.0}},
            "degrees",
            "less than or equal to 180",
        ),
        (
            {
                "color_jitter": {
                    "brightness": 1.5,
                    "contrast": 0.0,
                    "saturation": 0.0,
                }
            },
            "brightness",
            "less than or equal to 1",
        ),
    ],
    ids=[
        "a-negative-flip-probability",
        "a-negative-brightness",
        "an-infinite-rotation",
        "a-rotation-past-a-half-turn",
        "a-brightness-past-1",
    ],
)
def test_preflight_refuses_an_augmentation_value_outside_its_range_naming_the_field(
    tmp_path, transforms, field, refusal
):
    """A stated value outside its transform's range is refused by the transform's type, naming
    the entry and the field, under a record admitting every transform.
    """
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    _project_answering(tmp_path)
    config = training_config(
        tmp_path,
        DETECTION_SOURCE,
        _labeled(tmp_path),
        augmentation=augmentation(**transforms),
    )

    report = preflight_config(tmp_path, config)

    assert report["valid"] is False
    (name,) = transforms
    assert any(
        repr(name) in issue and f"\n{field}\n" in issue and refusal in issue
        for issue in report["issues"]
    ), report["issues"]


def test_a_sweep_refuses_the_point_its_augmentation_axis_samples_against_the_record(
    tmp_path, monkeypatch
):
    """An axis over an augmentation value is judged at each point it samples: a saturation of
    zero relies on no invariance and a nonzero one on color, so a record stating color false
    refuses the nonzero corner by its label with nothing opened, and one stating it true opens.
    """
    from tcip_mcp import experiments
    from tcip_mcp.tools import training_tools
    from tests._verified_checkpoint_fixtures import SWEEP_ARGUMENTS

    monkeypatch.chdir(tmp_path)
    base = training_config(
        tmp_path,
        DETECTION_SOURCE,
        _labeled(tmp_path),
        augmentation=augmentation(
            color_jitter={
                "brightness": 0.0,
                "contrast": 0.0,
                "saturation": 0.0,
            }
        ),
    )
    arguments = {
        k: v for k, v in SWEEP_ARGUMENTS.items() if k != "param_space"
    }
    space = {
        "augmentation.color_jitter.saturation": {
            "type": "categorical",
            "choices": [0.0, 0.3],
        }
    }

    forbidding = tmp_path / "forbidding"
    _project_answering(forbidding, color=False)
    refused = training_tools.open_sweep(
        forbidding, base, space, actor=None, **arguments
    )

    assert isinstance(refused, dict), refused
    assert refused["error"] == (
        "the sweep's base config fails preflight at "
        "augmentation.color_jitter.saturation=0.3"
    ), refused
    assert any("'color'" in issue for issue in refused["issues"]), refused
    assert experiments.sweep_dirs(forbidding) == []

    admitting = tmp_path / "admitting"
    _project_answering(admitting)
    opened = training_tools.open_sweep(
        admitting, base, space, actor=None, **arguments
    )

    assert isinstance(opened, Path), opened


def test_preflight_names_a_missing_task_where_the_config_is_validated(
    tmp_path,
):
    """A config stating no ``model_source.task`` is refused by name where preflight validates it,
    before any resolver reads the task; the same config stating its task is admitted.
    """
    from tcip_mcp.tools.training_tools import preflight_config

    data = _labeled(tmp_path)
    config = training_config(
        tmp_path,
        {"builder": BESPOKE_DETECTION},
        data,
        batch_size=1,
    )

    refused = preflight_config(tmp_path, config)

    assert refused["valid"] is False
    assert [
        issue
        for issue in refused["issues"]
        if issue.startswith("model_source.task")
    ]

    stated = training_config(tmp_path, DETECTION_SOURCE, data, batch_size=1)
    admitted = preflight_config(tmp_path, stated)
    assert admitted["valid"] is True, admitted["issues"]


def test_run_hyperparameter_search_admits_an_lr_sweep_beside_a_base_config_selection_metric(
    tmp_path,
    real_hpo_base_config,
    monkeypatch,
):
    """The selection-metric refusal targets param_space, never base_config: a config that
    states its own selection metric still runs an ordinary lr sweep.
    """
    from tcip_mcp.tools import training_tools

    monkeypatch.chdir(tmp_path)
    seen = _spaces_searched(monkeypatch)

    base_config = {
        **real_hpo_base_config,
        "evaluation": evaluation_block(selection_metric="map"),
    }
    result = training_tools.run_hyperparameter_search(
        tmp_path,
        base_config,
        param_space=_CHOICE_SPACE,
        n_trials=1,
        **fifo_search(),
        search_seed=0,
    )

    assert result["sweep"]["state"] == "completed", result
    assert seen == [_CHOICE_SPACE]


def test_hpo_admits_a_categorical_evaluation_axis_naming_the_same_metric_at_every_choice(
    tmp_path,
    real_hpo_base_config,
    monkeypatch,
):
    """A categorical evaluation axis naming the base config's metric at every choice is
    admitted, every trial selecting on the sweep's one metric; a choice naming another is
    refused naming the base config's.
    """
    from tcip_mcp.tools import training_tools

    monkeypatch.chdir(tmp_path)
    seen = _spaces_searched(monkeypatch)

    base_config = {
        **real_hpo_base_config,
        "evaluation": evaluation_block(selection_metric="map"),
    }
    param_space = {
        "evaluation": {
            "type": "categorical",
            "choices": [evaluation_block(selection_metric="map")] * 2,
        }
    }
    result = training_tools.run_hyperparameter_search(
        tmp_path,
        base_config,
        param_space=param_space,
        n_trials=1,
        **fifo_search(),
        search_seed=0,
    )

    assert "error" not in result, result
    assert seen == [param_space]


@pytest.mark.parametrize(
    "param_space",
    [
        {
            "evaluation": {
                "type": "categorical",
                "choices": [
                    evaluation_block(selection_metric="map"),
                    evaluation_block(selection_metric="loss"),
                ],
            }
        },
        {
            "evaluation.selection_metric": {
                "type": "categorical",
                "choices": ["map", "loss"],
            }
        },
    ],
    ids=["a-block-axis", "a-metric-axis"],
)
def test_a_sweep_whose_space_samples_another_selection_metric_is_refused_naming_the_base_configs(
    tmp_path, real_hpo_base_config, param_space
):
    """Guard. A sweep ranks every trial by its base config's one selection metric: a space axis
    sampling another, whether the whole evaluation block or the metric itself, refuses before
    anything opens, naming the base config's metric.
    """
    from tcip_mcp.experiments import sweep_dirs
    from tcip_mcp.tools import training_tools

    base_config = {
        **real_hpo_base_config,
        "evaluation": evaluation_block(selection_metric="map"),
    }
    refused = training_tools.run_hyperparameter_search(
        tmp_path,
        base_config,
        param_space=param_space,
        n_trials=1,
        **fifo_search(),
        search_seed=0,
    )

    assert refused["error"] == (
        "a sweep ranks every trial by the one evaluation.selection_metric its base_config "
        "states ('map'); state it there and sample no other."
    )
    assert sweep_dirs(tmp_path) == []


def test_dataset_identity_tolerates_a_genuinely_unregistered_dataset(tmp_path):
    """The admitting half: no identity document at all still reads as (None, fp), not a refusal."""
    from tcip_mcp.pipelines.data.split_construction import dataset_identity
    from tcip_mcp.pipelines.schemas import DataSpec

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)

    ds_id, fp = dataset_identity(
        DataSpec.model_validate(
            {"images_dir": str(images_dir), "split": drawn_split()}
        )
    )
    assert ds_id is None


def test_cancel_end_to_end_through_the_real_trainer_ends_canceled_with_records_and_no_result(
    tmp_path,
) -> None:
    """A trial that trains one real epoch and is then canceled mid-training (the sweep's
    cancellation requested only once a genuine score is already on record) ends canceled with
    its launch record and carries no result, not that epoch's real score, so a canceled trial can
    never outrank one that merely scored worse.
    """
    pytest.importorskip("torch")
    import math

    from tcip_mcp.experiments import (
        RUN_FILE,
        observe,
        read_record,
        request_cancel,
    )
    from tests.tiny_trainer_fixtures import (
        regressor_config,
        write_regression_dataset,
    )

    images_dir, csv_path = write_regression_dataset(
        tmp_path, intensities=[0.1, 0.3, 0.5, 0.7], values=[0.2, 0.6, 1.0, 1.4]
    )
    base_config = regressor_config(
        5,
        data={
            "images_dir": str(images_dir),
            "labels_dir": str(csv_path),
            "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
        },
        project=tmp_path,
    )
    reported: list = []

    def report(value: float) -> None:
        # The cancel is requested only after the first real report, so a genuine score is
        # already on record by the time the sweep's cancel takes effect mid-training.
        reported.append(value)
        if len(reported) == 1:
            request_cancel(_the_sweep(tmp_path))

    trial_dir = _trial({}, report, base_config, tmp_path)

    assert math.isfinite(
        reported[0]
    )  # epoch 1's real score, reported before the cancel
    assert observe(trial_dir).state == "canceled"
    assert _sweep_outcome(tmp_path)["best_value"] is None
    assert read_record(trial_dir / RUN_FILE)["trial_params"] == {}
