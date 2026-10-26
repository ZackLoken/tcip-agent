"""Tests for training_tools.

The tests cover validator/StageSpec alignment, HPO param plumbing, HPO trial reporting (each
epoch as it completes; no result for a trial that fails or is canceled), a trial's forwarding
of the base config's augmentation and model source, experiment immutability on relaunch, and
canonical-format confidence parsing in get_worst_predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
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

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import NoReturn, Required, TypedDict, Unpack

    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.data.label_queries import Admission
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.data.split_construction import (
        Partition,
        ResolvedRun,
    )
    from tcip_mcp.pipelines.model_build import SourceLayout, StagedSources
    from tcip_mcp.pipelines.schemas import DataSpec, TrainConfigSchema
    from tcip_mcp.pipelines.training.run_registry import TrainRun
    from tests.bespoke_models import BespokeDetection

    class _DetectionBuilderKeywords(TypedDict, total=False):
        """The keywords ``build_bespoke_detection`` takes."""

        image_mean: Required[list[float]]
        image_std: Required[list[float]]
        num_classes: int
        in_chans: int
        detector: str

    class _AdmitKeywords(TypedDict, total=False):
        """The keywords ``label_queries.admit`` takes beside its directories."""

        scope: ClassScope | None
        members: list[str] | None


pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")


def _labeled(tmp_path: Path) -> dict:
    """Return a data section over two labeled frames of ``bud`` under ``tmp_path``.

    The frames come from ``_verified_checkpoint_fixtures.detection_images``, drawn at seed 0.
    """
    from tests._verified_checkpoint_fixtures import detection_images

    scope = {"subject": "bud"}
    return {
        **detection_images(tmp_path / "labeled", scope),
        "scope": scope,
        "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
    }


def _bud_image(image: Path, *, labeled: bool = True) -> None:
    """Write a 20x20 black ``painted_frame`` at ``image``, painted ``BRIGHT`` over (2, 2, 10, 10).

    Unless ``labeled`` is false, it also writes the label document holding that ``bud`` box.
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
    """Replace ``image``'s label document in the store by ``stored``."""
    from tests._producer_fixtures import image_label_key
    from tests._record_damage_fixtures import damage_record

    damage_record(image_label_key(image), stored)


# --------------------------------------------------------------------------
# preflight_config: stages
# --------------------------------------------------------------------------


def test_preflight_config_accepts_trainer_canonical_stages(
    tmp_path: Path,
) -> None:
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


def test_preflight_config_refuses_a_nested_training_section_by_name(
    tmp_path: Path,
) -> None:
    """A preflight refuses a nested ``training`` section by name and admits its keys flat."""
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


def test_a_smoke_preflight_validates_its_config_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A smoke preflight validates its config once.

    A smoke preflight that admits a config records one validation, of ``TrainConfigSchema``.
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
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.training_tools import preflight_config

    r = preflight_config(tmp_path, {"data": "x"})
    assert r["valid"] is False
    assert any(
        i.startswith("data:") and "dictionary" in i for i in r["issues"]
    ), r["issues"]


# preflight_config's overfit branch: reseed-run-restore, never gating


def _frozen_detection_builder(
    **kwargs: Unpack[_DetectionBuilderKeywords],
) -> BespokeDetection:
    """Build a detector through ``build_bespoke_detection`` with its parameters frozen.

    The kwargs pass through to ``build_bespoke_detection``.
    """
    from tests import bespoke_models

    model = bespoke_models.build_bespoke_detection(**kwargs)
    for p in model.parameters():
        p.requires_grad = False
    return model


def _detection_smoke_cfg(builder: str, tmp_path: Path) -> dict:
    """Return a smoke config over the bespoke detector built by ``builder``.

    The config states an ``fcos`` detector normalized by ``FRAME_STATISTICS``, over
    :func:`_labeled`'s frames at three channels.
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


def test_preflight_config_overfit_restores_rng_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preflight's overfit check restores the random streams it reseeded.

    A smoke build with overfit=False and the same build with overfit=True, each run from seed
    11, leave the next draw of the Python, numpy and torch streams equal.
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
    tmp_path: Path,
) -> None:
    """An overfit preflight on an all-frozen model reports its empty parameters without raising.

    check_model_contract's own gradient-presence check requires a real learnable path, so the
    report carries that issue for a literally all-frozen model; the overfit branch's
    empty-parameter case does not raise out of preflight_config, and its report names the empty
    parameter list.
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


def test_preflight_config_refuses_a_per_stage_lr_by_name(
    tmp_path: Path,
) -> None:
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


def test_preflight_config_warns_when_most_candidates_wont_train(
    tmp_path: Path,
) -> None:
    """A preflight warns when most candidate images will not train.

    A run whose label store admits one of its four candidate images stays valid and warns that
    3/4 will not train, counting them by reason.
    """
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = image_dir(tmp_path, UNDATED_BUCKET)
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


def test_preflight_config_no_coverage_warning_when_everything_trains(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = image_dir(tmp_path, UNDATED_BUCKET)
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
    tmp_path: Path,
) -> None:
    """A preflight over an unreadable label is invalid.

    Its issues carry the refusal the admission raises naming the document.
    """
    pytest.importorskip("torch")
    from tcip_annotation import json_io
    from tcip_mcp.pipelines.data.split_construction import auto_train_val
    from tcip_mcp.pipelines.schemas import DataSpec
    from tcip_mcp.tools.training_tools import preflight_config
    from tests._producer_fixtures import staged_layout

    imgs = image_dir(tmp_path, UNDATED_BUCKET)
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


def test_preflight_admits_the_run_once(tmp_path: Path) -> None:
    """A preflight admits the run once.

    A valid preflight of a single tiled source with reserved shares calls the admission once.
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

    def counting_admit(
        *args: Unpack[tuple[str | Path, str | Path | None]],
        **kwargs: Unpack[_AdmitKeywords],
    ) -> Admission:
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
    tmp_path: Path, bad_document: str, refusal: str
) -> None:
    """A preflight is invalid over a document whose annotations the admission refuses.

    A document whose annotations field is not a list, or holds a non-object, makes preflight
    invalid, and its issues carry the admission's refusal.
    """
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config
    from tcip_store import encode_record

    imgs = image_dir(tmp_path, UNDATED_BUCKET)
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


def test_preflight_config_training_source_shape_and_importability(
    tmp_path: Path,
) -> None:
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


def test_preflight_config_rejects_incoherent_selection_metric(
    tmp_path: Path,
) -> None:
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
    tmp_path: Path,
) -> None:
    """A non-mapping ``evaluation`` block is a named preflight issue."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = image_dir(tmp_path, UNDATED_BUCKET)
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


def test_preflight_calibration_ratio_wrong_task_flags_issue(
    tmp_path: Path,
) -> None:
    """A preflight flags a calibration_ratio on a classification run by name.

    The issue names calibration_ratio and the reserved strips it needs.
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


def test_preflight_calibration_ratio_multi_member_flags_issue(
    tmp_path: Path,
) -> None:
    """A preflight flags a calibration_ratio over two admitted members by name.

    The issue names calibration_ratio and the reserved strips it needs.
    """
    from tcip_mcp.tools.training_tools import preflight_config

    images_dir = image_dir(tmp_path, UNDATED_BUCKET)
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
    tmp_path: Path,
) -> None:
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
    tmp_path: Path,
) -> None:
    """The reserved-region preflight probe reports an unreadable label by name.

    An issue names the document and its decode failure.
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


def test_preflight_reserved_regions_admit_a_feasible_layout(
    tmp_path: Path,
) -> None:
    """A real, feasible reserved-region config produces no spatial-split issue.

    The preflight runs under smoke=True.
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
    """Return ``(lr, weight_decay)`` of each param group the optimizer of ``config`` builds.

    ``optimizer_factory.build_optimizer`` builds over ``build_two_rate_regressor()`` from the
    optimizer spec ``schemas.train_config`` reads off ``config``, at the spec's backbone and
    head rates.
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
    tmp_path: Path, training_source: str | None, unstated: str, named: str
) -> None:
    """The launch door refuses a key a trainer reads unstated, and the schema admits it.

    What a trainer reads is asked at the launch door, not by the schema: a detector config
    stating no ``evaluation.conf_threshold`` or no ``batch_size``, under the default trainer or a
    loop of its own, and a default-trainer config stating no ``stages``, are each refused by
    ``preflight_config`` naming the key, while ``train_config`` admits each as a checkpoint's
    config; a config naming its own loop and no ``stages`` is admitted by the door. A
    classifier's trainer reads no confidence threshold.
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
    """Apply ``params`` to ``base`` through ``training_tools._apply_hpo_params``."""
    from tcip_mcp.tools.training_tools import _apply_hpo_params

    return _apply_hpo_params(base, params)


def _base(builder: str = BESPOKE_DETECTION, **overrides: object) -> dict:
    """Return a detection base config of this repository of ``builder`` over no data.

    It normalizes by its named weights' statistics (``_chain_fixtures.weights_statistics``),
    ``overrides`` in place of theirs.
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


def test_apply_hpo_params_optimizer_axes_reach_optimizer_param_groups() -> (
    None
):
    """Swept optimizer values reach the optimizer's param groups.

    They reach each group's rate and both groups' weight decay through the trainer's own
    optimizer read.
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


def test_apply_hpo_params_preserves_base_config_stages() -> None:
    """Sweeping a rate leaves base_config's stages unchanged."""
    custom_stages = [stage(-1), stage(2), stage(0)]
    out = _applied(
        _base("x:y", stages=custom_stages), {"optimizer.head_lr": 3e-3}
    )
    assert out["stages"] == custom_stages


def test_apply_hpo_params_undotted_key_reaches_top_level() -> None:
    """An undotted swept key lands at the top level of the resolved config.

    The resolved config holds no ``training`` section.
    """
    out = _applied(_base("x:y"), {"batch_size": 4})
    assert out["batch_size"] == 4
    assert "training" not in out


def test_apply_hpo_params_dotted_key_reaches_nested_field() -> None:
    """A dotted param (e.g. a swept builder) reaches the nested field it names."""
    out = _applied(
        _base("old:builder"), {"model_source.builder": "new:builder"}
    )
    assert out["model_source"]["builder"] == "new:builder"
    assert (
        out["model_source"]["task"] == "detection"
    )  # the rest of the mapping survives


def test_apply_hpo_params_dotted_seed_key_reaches_the_split_config() -> None:
    """A swept ``data.split.seed`` reaches the nested split field.

    split_draws's own paired grid axis reaches the nested field run_hyperparameter_search's
    drawn path reads, through the dotted-key mechanism a swept builder uses.
    """
    out = _applied(_base("x:y"), {"data.split.seed": 7})
    assert out["data"]["split"]["seed"] == 7


def test_apply_hpo_params_refuses_a_dotted_key_through_a_non_mapping_intermediate() -> (
    None
):
    """A dotted key whose path walks through a non-mapping value is refused by name."""
    with pytest.raises(ValueError, match="model_source.builder.x"):
        _applied(_base("x:y"), {"model_source.builder.x": "y"})


def test_preflight_points_covers_every_categorical_choice_and_both_numeric_bounds() -> (
    None
):
    """The preflight points cover every categorical choice and both numeric bounds.

    Every categorical choice and both numeric bounds (low and high) are held by some point, no
    point is listed twice, and a choice repeated in the space yields one point.
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
    tmp_path: Path,
) -> None:
    """A trial's combined point is admitted as the launch door admits it.

    A trial whose combined point is a default-trainer config with no ``stages`` ends failed
    naming why, with nothing resolved; a combination the door admits opens its trial resolved.
    """
    from tcip_mcp.experiments import observe, read_run
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
    assert final is not None and final.state == "failed"
    assert final.status_error is not None
    assert "['stages'] unstated" in final.status_error
    assert read_run(refused).resolved is None
    admitted = open_trial(
        sweep, "admitted", {"training_source": TRAIN_BESPOKE, "stages": None}
    )
    assert observe(admitted).final is None
    assert read_run(admitted).resolved is not None


# --------------------------------------------------------------------------
# _run_hpo_trial: each epoch's selection value is reported as it completes; only a completed
# trial is eligible for the sweep's best. Run directly (no Ray) so training can be stubbed.
# --------------------------------------------------------------------------


class _FakeDataset:
    def __len__(self) -> int:
        return 4

    def __getitem__(self, i: int) -> int:
        return i


def _detection_base(project: Path) -> dict:
    """Return a detection base config over :func:`_labeled`'s frames, selecting on the loss.

    Its model is :func:`~tests._verified_checkpoint_fixtures.unbuilt_source`'s detector,
    normalized by its frames' own statistics, and its frames lie under ``project``.
    """
    from tests._verified_checkpoint_fixtures import unbuilt_source

    return training_config(
        project,
        unbuilt_source("detection"),
        _labeled(project),
        evaluation=evaluation_block(selection_metric="loss"),
    )


def _classifier_base(project: Path, **overrides: object) -> dict:
    """Return a classification base config over four frames labeled 0, 1, 0, 1, on accuracy.

    The frames and their table lie under ``project``, the model normalizes by the frames' own
    statistics, and ``overrides`` stand in place of their keys.
    """
    from tests.tiny_trainer_fixtures import write_regression_dataset

    images_dir, csv_path = write_regression_dataset(
        project / "ds", [0.1, 0.3, 0.5, 0.7], [0, 1, 0, 1], height=32, width=32
    )
    return training_config(
        project,
        CLASSIFIER_SOURCE,
        {"images_dir": str(images_dir), "labels_dir": str(csv_path)},
        evaluation=evaluation_block(
            selection_metric="accuracy",
            conf_threshold=None,
            score_weights=None,
        ),
        **overrides,
    )


def _the_sweep(project: Path) -> Path:
    """Return the one sweep directory of ``project``."""
    from tcip_mcp import experiments

    (sweep,) = experiments.sweep_dirs(project)
    return sweep


def _sweep_of(project: Path, base: dict) -> Path:
    """Return the one sweep of ``project``, opened over ``base`` by the sweep door when none is."""
    from tcip_mcp import experiments
    from tests._verified_checkpoint_fixtures import opened_sweep

    return (
        _the_sweep(project)
        if experiments.sweep_dirs(project)
        else opened_sweep(project, base)
    )


def _trial(
    point: dict,
    report: Callable[[float], None],
    base: dict,
    project: Path,
    name: str = "t0",
) -> Path:
    """Run one HPO trial ``name`` through ``_run_hpo_trial``.

    The trial runs under :func:`_sweep_of` ``project`` over ``base``. Returns the trial's run
    directory.
    """
    from tcip_mcp.tools.training_tools import _run_hpo_trial

    sweep = _sweep_of(project, base)
    _run_hpo_trial(point, report, sweep, name)
    return sweep / f"{sweep.name}_{name}"


def _row(value: float, metric: str = "loss") -> dict:
    """Return an epoch row naming ``metric`` as its selection metric, at ``value``."""
    return {"selection_metric": metric, "selection": value}


def _complete(run: TrainRun) -> TrainRun:
    """Save a one-zero weight under ``model_final`` and mark ``run`` completed.

    The save goes through the run's own writer, ``TrainContext.save_checkpoint``.
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
    """Return the outcome of :func:`_trial`'s sweep as the project's listing reads it."""
    from tcip_mcp.experiments import training_listing

    (sweep,) = training_listing(project).sweeps
    return sweep.outcome


def _patch_hpo_trial_machinery(
    monkeypatch: pytest.MonkeyPatch,
    fake_train: Callable[
        [
            TrainRun,
            DataLoader,
            DataLoader | None,
            Callable[[int, dict], None] | None,
            Callable[[int, int, dict], None] | None,
            str,
        ],
        TrainRun,
    ],
    base: dict,
    project: Path,
    captured: dict | None = None,
) -> None:
    """Stub a trial's dataset building, and its training with ``fake_train``.

    The sweep of ``project`` over ``base`` is opened first (:func:`_sweep_of`), its base
    preflight smoking the run's own data. Then ``auto_train_val`` answers one stand-in dataset on
    both sides with an empty partition record, recording its ``transforms`` and ``data`` into
    ``captured`` when one is given; ``recorded_datasets`` answers the same stand-in, the data
    geometry is empty, and no sampler is built.
    """
    from tcip_mcp.pipelines.data import samplers, split_construction as sc
    from tcip_mcp.pipelines.schemas import TrainGeometry
    from tcip_mcp.pipelines.training import generic_trainer as gt

    _sweep_of(project, base)
    ds = _FakeDataset()

    def fake_auto_train_val(
        project: Path,
        task: str,
        data: DataSpec,
        transforms: object,
        plan: SourceLayout,
        **_: object,
    ) -> tuple[_FakeDataset, _FakeDataset, Partition, DataSpec]:
        if captured is not None:
            captured["transforms"] = transforms
            captured["data"] = data
        return (
            ds,
            ds,
            sc.Partition(
                seed=None,
                group_by="stem",
                samples=[],
                selection=None,
                spatial=None,
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


def _spaces_searched(monkeypatch: pytest.MonkeyPatch) -> list:
    """Replace ``tune_search`` with a stub that records the param space of each call.

    The stub returns nothing; the return is the list it records into.
    """
    from tcip_mcp.pipelines.training import hpo

    seen: list = []

    def fake_search(*args: object, **kwargs: object) -> None:
        seen.append(
            kwargs.get("param_space", args[1] if len(args) > 1 else None)
        )

    monkeypatch.setattr(hpo, "tune_search", fake_search)
    return seen


def _completed_train(
    run: TrainRun,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epoch_callback: Callable[[int, dict], None] | None = None,
    batch_callback: Callable[[int, int, dict], None] | None = None,
    resume_from: str = "",
) -> TrainRun:
    """Mark the run completed, reporting no epoch."""
    run.status = "completed"
    return run


def test_run_hpo_trial_reports_each_epoch_and_its_result_is_the_best_of_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An HPO trial reports each epoch, and its result is the best of them.

    Every epoch's selection value is reported as the trial logs it, and the trial's result is
    the best of those same rows (lower=better).
    """
    pytest.importorskip("torch")

    def fake_train(
        run: TrainRun,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        epoch_callback: Callable[[int, dict], None] | None = None,
        batch_callback: Callable[[int, int, dict], None] | None = None,
        resume_from: str = "",
    ) -> TrainRun:
        for epoch, value in enumerate([50.0, 40.0, 30.0]):
            if epoch_callback:
                epoch_callback(epoch, _row(value))
        return _complete(run)

    base = _detection_base(tmp_path)
    _patch_hpo_trial_machinery(monkeypatch, fake_train, base, tmp_path)
    reported: list = []
    _trial({"optimizer.head_lr": 3e-4}, reported.append, base, tmp_path)
    assert reported == [50.0, 40.0, 30.0]
    assert _sweep_outcome(tmp_path)["best_value"] == 30.0


@pytest.mark.parametrize(
    ("head_lr", "at_end"),
    [(1e-2, {"optimizer.head_lr": "high"}), (3e-4, {})],
)
def test_a_sweeps_best_at_an_end_of_its_stated_range_is_reported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    head_lr: float,
    at_end: dict,
) -> None:
    """A sweep's best at an end of its stated range is reported with that end.

    A best whose value sits at the stated ``low`` or ``high`` of its axis is named with that
    end; a best inside the range names none.
    """
    pytest.importorskip("torch")

    def fake_train(
        run: TrainRun,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        epoch_callback: Callable[[int, dict], None] | None = None,
        batch_callback: Callable[[int, int, dict], None] | None = None,
        resume_from: str = "",
    ) -> TrainRun:
        if epoch_callback:
            epoch_callback(0, _row(30.0))
        return _complete(run)

    base = _detection_base(tmp_path)
    _patch_hpo_trial_machinery(monkeypatch, fake_train, base, tmp_path)
    _trial({"optimizer.head_lr": head_lr}, lambda _value: None, base, tmp_path)

    assert _sweep_outcome(tmp_path)["best_at_range_end"] == at_end


def test_run_hpo_trial_that_fails_has_no_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A crashed HPO trial ends failed and carries no result.

    It reports nothing, and the sweep has no best value.
    """
    pytest.importorskip("torch")
    from tcip_mcp.experiments import observe

    def fake_train(
        run: TrainRun,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        epoch_callback: Callable[[int, dict], None] | None = None,
        batch_callback: Callable[[int, int, dict], None] | None = None,
        resume_from: str = "",
    ) -> NoReturn:
        raise RuntimeError("CUDA out of memory")

    base = _detection_base(tmp_path)
    _patch_hpo_trial_machinery(monkeypatch, fake_train, base, tmp_path)
    reported: list = []
    trial_dir = _trial(
        {"optimizer.head_lr": 3e-4}, reported.append, base, tmp_path
    )
    assert reported == []
    assert observe(trial_dir).state == "failed"
    assert _sweep_outcome(tmp_path)["best_value"] is None


def test_run_hpo_trial_result_is_the_highest_value_for_a_higher_is_better_metric(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An HPO trial's result is its highest value under a higher-is-better metric.

    Under accuracy, a trial reporting 0.5, 0.9 and 0.6 has 0.9 as its result.
    """
    pytest.importorskip("torch")

    def fake_train(
        run: TrainRun,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        epoch_callback: Callable[[int, dict], None] | None = None,
        batch_callback: Callable[[int, int, dict], None] | None = None,
        resume_from: str = "",
    ) -> TrainRun:
        for epoch, value in enumerate([0.5, 0.9, 0.6]):
            if epoch_callback:
                epoch_callback(epoch, _row(value, metric="accuracy"))
        return _complete(run)

    base = _classifier_base(tmp_path)
    _patch_hpo_trial_machinery(monkeypatch, fake_train, base, tmp_path)
    reported: list = []
    _trial({"optimizer.head_lr": 3e-4}, reported.append, base, tmp_path)
    assert reported == [0.5, 0.9, 0.6]
    assert _sweep_outcome(tmp_path)["best_value"] == 0.9


def test_a_failed_trial_never_outranks_a_real_one_under_a_maximize_direction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Beside a failed trial, a maximize sweep's best is the completed trial's params and value."""
    pytest.importorskip("torch")

    base = _classifier_base(tmp_path)

    def fake_train_ok(
        run: TrainRun,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        epoch_callback: Callable[[int, dict], None] | None = None,
        batch_callback: Callable[[int, int, dict], None] | None = None,
        resume_from: str = "",
    ) -> TrainRun:
        if epoch_callback:
            epoch_callback(0, _row(0.7, metric="accuracy"))
        return _complete(run)

    _patch_hpo_trial_machinery(monkeypatch, fake_train_ok, base, tmp_path)
    real: list = []
    _trial({"optimizer.head_lr": 3e-4}, real.append, base, tmp_path, "real")

    def fake_train_fails(
        run: TrainRun,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        epoch_callback: Callable[[int, dict], None] | None = None,
        batch_callback: Callable[[int, int, dict], None] | None = None,
        resume_from: str = "",
    ) -> NoReturn:
        raise RuntimeError("boom")

    _patch_hpo_trial_machinery(monkeypatch, fake_train_fails, base, tmp_path)
    failed: list = []
    _trial(
        {"optimizer.head_lr": 1e-2}, failed.append, base, tmp_path, "failed"
    )

    outcome = _sweep_outcome(tmp_path)
    assert (outcome["best_params"], outcome["best_value"]) == (
        {"optimizer.head_lr": 3e-4},
        0.7,
    )


def test_run_hpo_trial_uses_base_augmentation_and_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An HPO trial hands base_config's augmentation and bespoke model_source to its training.

    The dataset build receives built transforms, and the body receives the bespoke builder.
    """
    pytest.importorskip("torch")

    captured: dict = {}

    def fake_train(
        run: TrainRun,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        epoch_callback: Callable[[int, dict], None] | None = None,
        batch_callback: Callable[[int, int, dict], None] | None = None,
        resume_from: str = "",
    ) -> TrainRun:
        captured["model_source"] = run.spec.model_source.model_dump(
            exclude_unset=True
        )
        run.status = "completed"
        return run

    new_project(tmp_path)
    base = _classifier_base(
        tmp_path, augmentation=augmentation(horizontal_flip={"p": 0.5})
    )
    _patch_hpo_trial_machinery(
        monkeypatch, fake_train, base, tmp_path, captured=captured
    )
    _trial({"optimizer.head_lr": 3e-4}, [].append, base, tmp_path)
    assert (
        captured["transforms"] is not None
    )  # augmentation was built + passed
    assert captured["model_source"]["builder"] == BESPOKE_CLASSIFIER


def test_run_hpo_trial_dotted_seed_axis_reaches_the_data_cfg_handed_to_auto_train_val(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A trial's swept ``data.split.seed`` reaches the data section auto_train_val is handed.

    The data section auto_train_val is handed carries the swept seed 7.
    """
    pytest.importorskip("torch")

    captured: dict = {}
    base = _detection_base(tmp_path)
    _patch_hpo_trial_machinery(
        monkeypatch, _completed_train, base, tmp_path, captured=captured
    )
    _trial({"data.split.seed": 7}, [].append, base, tmp_path)
    assert captured["data"].split.seed == 7


def test_run_hpo_trial_dotted_seed_axis_reaches_the_trials_own_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A trial's swept ``data.split.seed`` reaches the trial's own records.

    The split_draws grid axis lands in the trial's launch record and in the data section it
    resolved, at the nested field auto_train_val reads it from, beside the sampled point itself.
    """
    pytest.importorskip("torch")
    from tcip_mcp.experiments import read_run

    base = _detection_base(tmp_path)
    _patch_hpo_trial_machinery(monkeypatch, _completed_train, base, tmp_path)

    trial_dir = _trial({"data.split.seed": 7}, [].append, base, tmp_path)

    run = read_run(trial_dir)
    assert run.spec.data.split.seed == 7
    assert run.trial_params == {"data.split.seed": 7}
    assert run.resolution.data.split.seed == 7


def test_run_hpo_trial_producer_fed_data_split_seed_over_the_single_source_spatial_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A trial over a single-source spatial split records the split and tiling it resolved.

    The producer path: a real one-source tiled dataset through the real, unstubbed
    auto_train_val. The trial's launch record carries the spatial split and tiling
    auto_train_val wrote, with no seed in the partition or its spatial split.
    """
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from tcip_mcp.experiments import read_run

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

    resolved = read_run(trial_dir).resolution
    assert resolved.partition.seed is None
    assert resolved.partition.spatial is not None
    assert "seed" not in resolved.partition.spatial.model_dump()
    assert resolved.data.tiling is not None
    assert resolved.data.tiling.tile_size == 128


def test_a_trials_launch_record_carries_the_seed_it_trained_under(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A trial's launch record and its body's received spec both carry the base config's seed."""
    pytest.importorskip("torch")
    from tcip_mcp.experiments import read_run

    captured: dict = {}

    def fake_train(
        run: TrainRun,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        epoch_callback: Callable[[int, dict], None] | None = None,
        batch_callback: Callable[[int, int, dict], None] | None = None,
        resume_from: str = "",
    ) -> TrainRun:
        captured["seed"] = run.spec.seed
        run.status = "completed"
        return run

    base = {**_detection_base(tmp_path), "seed": 11}
    _patch_hpo_trial_machinery(monkeypatch, fake_train, base, tmp_path)
    trial_dir = _trial({"optimizer.head_lr": 3e-4}, [].append, base, tmp_path)

    recorded = read_run(trial_dir).spec.seed
    assert recorded == captured["seed"] == 11


def test_run_hpo_trial_diverged_run_never_outranks_a_worse_but_alive_config(
    tmp_path: Path,
) -> None:
    """A trial over a model diverging after one good call reports a finite first score.

    The sweep is left no best value. The model is ``build_diverges_after_model`` with one good
    call, and the real training body runs, nothing mocked, with a val_ratio of zero.
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


def test_get_worst_predictions_reads_canonical_confidence(
    tmp_path: Path,
) -> None:
    """The worst-predictions ranking reads confidence from a document's native ``score``.

    Prediction documents carry a native ``score``; confidence reads from that field, not from
    box geometry.
    """
    pytest.importorskip("torch")
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.vision_tools import get_worst_predictions
    from tests._chain_fixtures import published
    from tests._producer_fixtures import label_image, write_image

    images = image_dir(tmp_path, UNDATED_BUCKET)
    scored = {"confident": [0.9, 0.9], "shaky": [0.1, 0.1]}
    for stem, scores in scored.items():
        write_image(images / f"{stem}.png", (100, 100))
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


def test_an_ordinary_launch_config_reaches_preflight_admitted_with_no_issue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ordinary launch config reaches preflight admitted, with no issue."""
    from tcip_mcp.tools import training_tools

    monkeypatch.chdir(tmp_path)
    seen = []

    def stub_preflight(
        project: Path,
        spec: TrainConfigSchema | None,
        plan: StagedSources | None,
        issues: list[str],
        *,
        smoke: bool,
        overfit: bool,
    ) -> tuple[dict, ResolvedRun | None]:
        seen.append((spec, issues))
        return {"valid": False, "issues": ["stub"]}, None

    monkeypatch.setattr(training_tools, "_preflight", stub_preflight)
    config = training_config(tmp_path, DETECTION_SOURCE, _labeled(tmp_path))

    result = training_tools.launch_training(tmp_path, config, actor=None)

    assert result == {"error": "Invalid config", "issues": ["stub"]}
    [(spec, issues)] = seen
    assert spec is not None and issues == []


_CHOICE_SPACE = {
    "optimizer.head_lr": {"type": "categorical", "choices": [0.1, 0.01]}
}
"""A sample space of two head learning rates."""


def test_a_space_not_in_the_search_space_shape_is_refused_before_any_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A space not in the search-space shape is refused, and no search runs.

    A list where an axis spec belongs is refused with an error saying param_space is not a
    search space.
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
        actor=None,
    )

    assert "param_space is not a search space" in result["error"], result
    assert seen == []


def test_an_ordinary_sweep_payload_still_runs_its_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ordinary sweep payload reaches its search and completes.

    The sweep passes the sweep door's structural preflight (an importable builder, a real data
    section).
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
        actor=None,
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    choices: list[float],
    label: str,
) -> None:
    """A sweep opens over a base omitting what its space sweeps and refuses an invalid point.

    A base config may leave unstated what its search space names: the sweep validates each
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

    assert isinstance(opened, training_tools.OpenedSweep), opened
    assert experiments.sweep_dirs(tmp_path) == [opened.directory]


def _project_answering(project: Path, **answers: bool | float) -> None:
    """Make ``project`` a project through ``initialize_project`` with ``answers`` as invariances.

    Its capture invariances are the sample answer (``_web_fixtures.INVARIANCES``) with
    ``answers`` over it, validated by the record's model.
    """
    from tcip_mcp.project_record import CaptureInvariances
    from tcip_mcp.tools.project_tools import initialize_project

    stated = INVARIANCES.model_dump() | answers
    result = initialize_project(
        str(project),
        "Test project",
        "north orchard",
        CaptureInvariances.model_validate(stated),
        actor=None,
    )
    assert "error" not in result, result


def _door_issues(
    door: str, project: Path, config: dict, monkeypatch: pytest.MonkeyPatch
) -> list:
    """Return the issues ``door`` answers ``config`` with over ``project``.

    The issues are ``preflight_config``'s report, or the issues ``launch_training`` hands its
    preflight, which is stubbed to answer invalid.
    """
    from tcip_mcp.tools import training_tools

    if door == "preflight_config":
        return training_tools.preflight_config(project, config)["issues"]
    seen: list = []

    def stub_preflight(
        project: Path,
        spec: TrainConfigSchema | None,
        sources: StagedSources | None,
        issues: list[str],
        *,
        smoke: bool,
        overfit: bool,
    ) -> tuple[dict, ResolvedRun | None]:
        seen.append(issues)
        return {"valid": False, "issues": ["stub"]}, None

    monkeypatch.setattr(training_tools, "_preflight", stub_preflight)
    training_tools.launch_training(project, config, actor=None)
    (issues,) = seen
    return issues


_EVERY_INVARIANCE_FALSE = {
    "horizontal_flip": False,
    "vertical_flip": False,
    "rotation": 0,
    "color": False,
    "intensity": False,
    "scale": False,
}
"""A record stating no capture transform leaves the trait's measurement unchanged."""


def _rotation(degrees: float, p: float) -> dict:
    """Return a rotation entry stating ``degrees`` and ``p``."""
    return {"rotation": {"degrees": degrees, "p": p}}


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
        ({"rotation": 5}, _rotation(5, 0.5), []),
        ({"rotation": 5}, _rotation(5.5, 0.5), ["rotation"]),
        ({"rotation": 5}, _rotation(30, 0.0), []),
        ({"rotation": 0}, _rotation(0, 1.0), []),
        ({"rotation": 0}, _rotation(1, 1.0), ["rotation"]),
        ({"rotation": 180}, _rotation(180, 1.0), []),
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
        "a-rotation-at-the-stated-angle",
        "a-rotation-past-the-stated-angle",
        "a-rotation-past-the-stated-angle-never-taken",
        "no-rotation-under-no-rotation",
        "a-rotation-under-no-rotation",
        "any-rotation-under-any-rotation",
        "a-crop-under-no-scale",
        "blur-and-resize-under-every-record",
        "brightness-alone-under-no-color",
        "brightness-alone-under-no-intensity",
    ],
)
def test_the_launch_doors_judge_each_augmentation_by_the_invariance_the_project_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    door: str,
    answers: dict,
    transforms: dict,
    refused: list[str],
) -> None:
    """The launch doors judge each augmentation by the invariance the project records.

    A transform relying on an invariance the record states false, or a rotation past the
    record's angle, is refused naming the transform, the invariance, the record's stated value
    and the door that states it; the same transform under a record stating it true is admitted,
    a rotation at or under the record's angle is admitted, a flip or a rotation stated at
    probability zero relies on nothing, a crop fixed at half scale relies on scale, a blur and a
    resize rely on none, and a color jitter whose saturation is zero relies on intensity alone.
    """
    pytest.importorskip("torch")
    from tcip_mcp.project_record import read_record

    _project_answering(tmp_path, **answers)
    stated = read_record(tmp_path).invariances
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
        assert f"({getattr(stated, fact)!r})" in issue, issue


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
            {"rotation": {"degrees": float("inf"), "p": 1.0}},
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
    tmp_path: Path, transforms: dict, field: str, refusal: str
) -> None:
    """An augmentation value outside its range is refused by preflight, naming the field.

    The transform's type refuses the stated value, naming the entry and the field, under the
    sample invariances record.
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep refuses the point its augmentation axis samples against the project record.

    An axis over an augmentation value is judged at each point it samples: a saturation of
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

    assert isinstance(opened, training_tools.OpenedSweep), opened


def test_preflight_names_a_missing_task_where_the_config_is_validated(
    tmp_path: Path,
) -> None:
    """Preflight names a missing ``model_source.task`` where it validates the config.

    The config stating its task is admitted.
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
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A base config stating its own selection metric still completes an lr sweep.

    The sweep's search is called with the stated space.
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
        actor=None,
    )

    assert result["sweep"]["state"] == "completed", result
    assert seen == [_CHOICE_SPACE]


def test_hpo_admits_a_categorical_evaluation_axis_naming_the_same_metric_at_every_choice(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A categorical evaluation axis naming the base config's metric at every choice is admitted.

    The sweep's search is called with that space.
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
        actor=None,
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
    tmp_path: Path, real_hpo_base_config: dict, param_space: dict
) -> None:
    """A sweep whose space samples another selection metric is refused, naming the base's.

    A space axis sampling another metric, whether the whole evaluation block or the metric
    itself, refuses naming the base config's metric, and no sweep directory is opened.
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
        actor=None,
    )

    assert refused["error"] == (
        "a sweep ranks every trial by the one evaluation.selection_metric its base_config "
        "states ('map'); state it there and sample no other."
    )
    assert sweep_dirs(tmp_path) == []


def test_dataset_identity_tolerates_a_genuinely_unregistered_dataset(
    tmp_path: Path,
) -> None:
    """A dataset with no identity document reads as no dataset id, not a refusal."""
    from tcip_mcp.pipelines.data.split_construction import dataset_identity
    from tcip_mcp.pipelines.schemas import DataSpec

    images_dir = image_dir(tmp_path, UNDATED_BUCKET)
    images_dir.mkdir(parents=True)

    ds_id, fp = dataset_identity(
        DataSpec.model_validate(
            {"images_dir": str(images_dir), "split": drawn_split()}
        )
    )
    assert ds_id is None


def test_cancel_end_to_end_through_the_real_trainer_ends_canceled_with_records_and_no_result(
    tmp_path: Path,
) -> None:
    """A trial canceled through the real trainer ends canceled with records and no result.

    A trial that reports one finite epoch and is then canceled mid-training ends canceled, keeps
    its launch record, and leaves the sweep no best value.
    """
    pytest.importorskip("torch")
    import math

    from tcip_mcp.experiments import observe, read_run, request_cancel
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
    assert read_run(trial_dir).trial_params == {}
