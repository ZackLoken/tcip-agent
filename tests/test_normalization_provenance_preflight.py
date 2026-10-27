"""Tests of the normalization record a model source requires and preflight checks.

The statistics a model normalizes by are one required record, model_source.normalization: the
per-band mean and std beside their one source, the band statistics' own sampling when no weights
load, the pretrained weights whose own statistics they are, or a statement with its reason
naming the weights a model reading more bands than they carry loads. The record's type refuses
statistics with no source and a statement naming no weights, and preflight_config refuses a
sampling whose window paths do not resolve to one of the run's own admitted sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir, parse_image_path
from tests._chain_fixtures import (
    BESPOKE_CLASSIFIER,
    BESPOKE_DETECTION,
    BESPOKE_MODELS,
    FRAME_STATISTICS,
    run_config,
    training_config,
)
from tests._training_values import RATIONALE, drawn_split, evaluation_block

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tcip_mcp.pipelines.data.band_groups import BandGroupRef

SUBJECT = "leaf"
TIMM_WEIGHTS = {"library": "timm", "name": "resnet18.a1_in1k"}
"""A sample pretrained weights record a normalization source names."""


def _cfg(
    project: Path,
    images_dir: Path,
    *,
    normalization: dict | str = FRAME_STATISTICS,
) -> dict:
    """Return ``training_config`` of ``project`` for ``BESPOKE_DETECTION`` over ``images_dir``.

    The model source normalizes by ``normalization``; the data section scopes to the subject under
    a drawn split of validation ratio 0.0, and the evaluation selects on loss.
    """
    model_source = {
        "builder": BESPOKE_DETECTION,
        "source_files": [BESPOKE_MODELS],
        "task": "detection",
        "normalization": normalization,
    }
    return training_config(
        project,
        model_source,
        {
            "images_dir": str(images_dir),
            "scope": {"subject": SUBJECT},
            "split": drawn_split(val_ratio=0.0),
        },
        evaluation=evaluation_block(selection_metric="loss"),
    )


def _label(
    images_dir: Path, *stems: str, width: int = 16, height: int = 16
) -> None:
    """Write one label document, one box of the subject, per stem's image in ``images_dir``."""
    from tcip_annotation.state import Annotation, BBox
    from tests._producer_fixtures import label_image

    for stem in stems:
        label_image(
            images_dir / stem,
            [Annotation(subject=SUBJECT, geometry=BBox(1, 1, 5, 5))],
            width,
            height,
        )


def _frame(path: Path) -> None:
    """Write a 16 px RGB frame of seeded uniform uint8 noise at ``path``."""
    import numpy as np
    from PIL import Image

    Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    ).save(path)


def _derived(
    paths: Sequence[str | Path | BandGroupRef], num_channels: int = 3
) -> dict:
    """Return ``band_normalization_stats`` over ``paths`` as a dumped record.

    It is called at ``num_channels`` with ``max_images=len(paths)`` and ``seed=0``.
    """
    from tcip_mcp.pipelines.derivations import band_normalization_stats

    result = band_normalization_stats(
        paths, num_channels, max_images=len(paths), seed=0
    )
    assert result is not None
    assert len(getattr(result.source, "windows", ())) == len(paths), result
    return result.model_dump()


def test_preflight_refuses_statistics_with_no_source(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = image_dir(tmp_path, UNDATED_BUCKET)
    imgs.mkdir(parents=True)
    cfg = _cfg(
        tmp_path,
        imgs,
        normalization={"image_mean": [0.1, 0.2], "image_std": [0.1, 0.1]},
    )

    r = preflight_config(tmp_path, cfg)
    assert r["valid"] is False
    assert any("model_source.normalization.source" in i for i in r["issues"])


def test_the_weights_own_statistics_are_admitted_naming_the_weights(
    tmp_path: Path,
) -> None:
    """The weights' producer's record, its source the weights themselves, validates cleanly."""
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.derivations import pretrained_normalization
    from tcip_mcp.pipelines.schemas import (
        PretrainedWeights,
        checked_train_config,
    )

    cfg = _cfg(
        tmp_path,
        tmp_path,
        normalization=pretrained_normalization(
            PretrainedWeights(**TIMM_WEIGHTS)
        ).model_dump(),
    )

    assert checked_train_config(cfg)[1] == []


def test_a_model_reading_more_bands_than_its_weights_states_every_band_with_its_reason(
    tmp_path: Path,
) -> None:
    """A five-band model under three-band weights builds from a stated record with its reason.

    The weights' own three statistics refuse at the build's band count, a stated record lacking
    its reason or the weights it loads refuses through its type, and a stated record of five bands
    with both is admitted and builds.
    """
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.derivations import pretrained_normalization
    from tcip_mcp.pipelines.schemas import (
        PretrainedWeights,
        checked_train_config,
    )
    from tests._chain_fixtures import detector_geometry
    from tests.bespoke_models import build_bespoke_detection

    weights = PretrainedWeights(**TIMM_WEIGHTS)
    geometry = detector_geometry().handed()
    three = pretrained_normalization(weights)
    with pytest.raises(ValueError, match="StatedBeyondWeights"):
        build_bespoke_detection(
            in_chans=5,
            image_mean=three.image_mean,
            image_std=three.image_std,
            **geometry,
        )

    stated = {"image_mean": [0.4] * 5, "image_std": [0.2] * 5}
    for lacking, source in (
        ("rationale", {"pretrained_weights": TIMM_WEIGHTS}),
        ("pretrained_weights", {"rationale": RATIONALE}),
    ):
        refused = _cfg(
            tmp_path, tmp_path, normalization={**stated, "source": source}
        )
        assert any(lacking in i for i in checked_train_config(refused)[1])

    reasoned = _cfg(
        tmp_path,
        tmp_path,
        normalization={
            **stated,
            "source": {
                "rationale": RATIONALE,
                "pretrained_weights": TIMM_WEIGHTS,
            },
        },
    )
    spec, issues = checked_train_config(reasoned)
    assert issues == [] and spec is not None
    normalization = spec.model_source.normalization
    assert normalization.pretrained_weights == weights
    model = build_bespoke_detection(
        in_chans=5,
        image_mean=normalization.image_mean,
        image_std=normalization.image_std,
        **geometry,
    )
    assert model.detector.transform.image_mean == stated["image_mean"]


def test_preflight_refuses_a_window_path_outside_images_dir(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = image_dir(tmp_path, UNDATED_BUCKET)
    imgs.mkdir(parents=True)
    _frame(imgs / "a.jpg")
    _label(imgs, "a")
    outside = tmp_path / "elsewhere.jpg"
    _frame(outside)

    cfg = _cfg(tmp_path, imgs, normalization=_derived([outside]))

    r = preflight_config(tmp_path, cfg)
    assert r["valid"] is False
    assert any("outside" in i for i in r["issues"])
    assert r["image_stats_containment"] == "checked"


def test_preflight_records_not_checked_when_no_membership_resolved(
    tmp_path: Path,
) -> None:
    """Preflight records the windows as not checked when no membership resolved.

    A config naming no locations resolves no membership, so there are no sources to check the
    window paths against; preflight says so explicitly and names no window outside the run.
    """
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    image = tmp_path / "a.png"
    _frame(image)
    cfg = training_config(
        tmp_path,
        {
            "builder": BESPOKE_DETECTION,
            "task": "detection",
            "source_files": [BESPOKE_MODELS],
            "normalization": _derived([image]),
        },
        {
            "dataset_source": {
                "builder": BESPOKE_CLASSIFIER,
                "source_files": [BESPOKE_MODELS],
            }
        },
    )

    r = preflight_config(tmp_path, cfg)
    assert r["image_stats_containment"] == "not_checked"
    assert not any("outside" in i for i in r["issues"])


def test_preflight_admits_the_exact_derivations_own_record(
    tmp_path: Path,
) -> None:
    """Preflight admits the exact derivation's sampling over an admitted image as contained."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = image_dir(tmp_path, UNDATED_BUCKET)
    imgs.mkdir(parents=True)
    a = imgs / "a.jpg"
    _frame(a)
    _label(imgs, "a")

    cfg = _cfg(tmp_path, imgs, normalization=_derived([a]))

    r = preflight_config(tmp_path, cfg)
    assert not any(
        "normalization" in i or "outside" in i for i in r["issues"]
    ), r["issues"]
    assert r["image_stats_containment"] == "checked"


def test_preflight_admits_the_sampled_derivations_own_record(
    tmp_path: Path,
) -> None:
    """Preflight admits the windowed derivation's own sampling record.

    The sampling, its window size and windows per raster beside the seed and cap, is the same
    record shape the exact sibling's is.
    """
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.derivations import band_normalization_stats_sampled
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = image_dir(tmp_path, UNDATED_BUCKET)
    imgs.mkdir(parents=True)
    a = imgs / "a.jpg"
    _frame(a)
    _label(imgs, "a")

    result = band_normalization_stats_sampled(
        [a], 3, seed=1, window_size=8, max_windows_per_image=4, max_images=1
    )
    assert result is not None
    assert (
        getattr(result.source, "window_size", None),
        getattr(result.source, "max_windows_per_image", None),
    ) == (8, 4)
    cfg = _cfg(tmp_path, imgs, normalization=result.model_dump())

    r = preflight_config(tmp_path, cfg)
    assert not any(
        "normalization" in i or "outside" in i for i in r["issues"]
    ), r["issues"]
    assert r["image_stats_containment"] == "checked"


def test_preflight_admits_the_exact_derivations_record_over_a_band_group_dataset(
    tmp_path: Path,
) -> None:
    """Preflight admits the exact derivation's record over a band-group dataset.

    The derivation labels a band-group raster by its manifest path, the label preflight's
    containment check resolves data.images_dir's own band groups to, so the window is contained.
    """
    pytest.importorskip("torch")
    import numpy as np
    import tifffile

    from tcip_mcp.pipelines.image_utils import list_logical_images
    from tcip_mcp.tools.training_tools import preflight_config
    from tests._producer_fixtures import write_band_group

    imgs = image_dir(tmp_path, UNDATED_BUCKET)
    imgs.mkdir(parents=True)
    band_a, band_b = imgs / "cap_G.tif", imgs / "cap_R.tif"
    ramp = np.arange(64, dtype=np.uint16).reshape(8, 8)
    tifffile.imwrite(str(band_a), ramp + 111)
    tifffile.imwrite(str(band_b), ramp + 222)
    write_band_group(
        imgs,
        "cap",
        {"Green": band_a, "Red": band_b},
        central_wavelength_nm=None,
    )
    _label(imgs, "cap", width=8, height=8)

    cfg = _cfg(
        tmp_path,
        imgs,
        normalization=_derived([list_logical_images(imgs)["cap"]], 2),
    )

    r = preflight_config(tmp_path, cfg)
    assert not any("outside" in i for i in r["issues"]), r["issues"]
    assert r["image_stats_containment"] == "checked"


def test_preflight_keeps_every_sample_of_a_two_date_selection(
    tmp_path: Path,
) -> None:
    """Preflight finds a window over a two-date selection's sources contained.

    Such a selection holds two samples of one name; preflight over a window of the bound run's
    sources reports the window contained, names nothing outside and validates the config.
    """
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.tools.training_tools import preflight_config
    from tests.test_selection_binding import (
        DATES,
        SUBJECT,
        _two_subject_two_date_dataset,
    )

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    from tcip_mcp.tools.data_tools import draw_splits

    assert "error" not in draw_splits(
        tmp_path,
        str(root),
        output_path=str(out),
        subject=SUBJECT,
        **drawn_split(
            seed=7,
            val_ratio=0.25,
            calibration_ratio=0.125,
            holdout_ratio=0.125,
            grouping="tile_prefix",
        ),
        actor=None,
    ), "the draw is the fixture"
    drawn = read_selection(out, project=tmp_path)
    bound = [s for s in drawn.samples if s.side in ("train", "val")]
    assert len({s.member for s in bound}) < len(bound), (
        "the fixture must hold one member name on two dates for this to bite"
    )
    assert {parse_image_path(s.source)[1] for s in bound} == set(DATES)

    model_source = {
        "builder": BESPOKE_DETECTION,
        "source_files": [BESPOKE_MODELS],
        "task": "detection",
        "normalization": _derived([s.source for s in bound]),
    }
    r = preflight_config(tmp_path, run_config(tmp_path, out, model_source))

    assert r["image_stats_containment"] == "checked"
    assert not any("outside" in i for i in r["issues"]), r["issues"]
    assert r["valid"] is True, r["issues"]


def test_preflight_refuses_a_hand_written_source_naming_the_record_keys(
    tmp_path: Path,
) -> None:
    """Preflight refuses a source shaped by neither derivation, naming both record keys.

    The source carries neither 'windows' nor 'pixel_fraction', and the refusal names both keys.
    """
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = image_dir(tmp_path, UNDATED_BUCKET)
    imgs.mkdir(parents=True)
    cfg = _cfg(
        tmp_path,
        imgs,
        normalization={
            "image_mean": [0.1, 0.2],
            "image_std": [0.1, 0.1],
            "source": {
                "note": "derived by hand, not through either derivation"
            },
        },
    )

    r = preflight_config(tmp_path, cfg)
    assert r["valid"] is False
    for key in ("windows", "pixel_fraction"):
        assert any(key in i for i in r["issues"]), r["issues"]


def test_preflight_refuses_a_sampling_naming_no_window(tmp_path: Path) -> None:
    """A sampling naming no window is refused by its type.

    Such a sampling says no image produced the statistics.
    """
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = image_dir(tmp_path, UNDATED_BUCKET)
    imgs.mkdir(parents=True)
    cfg = _cfg(
        tmp_path,
        imgs,
        normalization={
            "image_mean": [0.1, 0.2],
            "image_std": [0.1, 0.1],
            "source": {
                "windows": [],
                "seed": 0,
                "max_images": 1,
                "pixel_fraction": 1.0,
            },
        },
    )

    r = preflight_config(tmp_path, cfg)
    assert r["valid"] is False
    assert any("windows" in i for i in r["issues"]), r["issues"]


def test_preflight_reads_the_band_count_over_every_source_the_run_admits(
    tmp_path: Path,
) -> None:
    """Preflight refuses a run whose two admitted sources differ in band count, saying so."""
    pytest.importorskip("torch")
    import numpy as np
    import tifffile

    from tcip_mcp.tools.training_tools import preflight_config

    imgs = image_dir(tmp_path, UNDATED_BUCKET)
    imgs.mkdir(parents=True)
    _frame(imgs / "a.png")
    tifffile.imwrite(
        str(imgs / "b.tif"), np.zeros((16, 16, 5), dtype=np.uint8)
    )
    _label(imgs, "a", "b")

    from tests._chain_fixtures import weights_normalization

    r = preflight_config(
        tmp_path, _cfg(tmp_path, imgs, normalization=weights_normalization())
    )

    assert r["valid"] is False
    assert any("different band counts" in i for i in r["issues"]), r["issues"]


def test_a_copied_projects_run_reads_its_statistics_windows_from_the_copy(
    tmp_path: Path,
) -> None:
    """A copied project's run resolves its recorded statistics-window paths under the copy."""
    pytest.importorskip("torch")
    import shutil

    import tcip_store as ts
    from tcip_mcp.experiments import read_run
    from tests._verified_checkpoint_fixtures import opened_run

    project = tmp_path / "project"
    imgs = image_dir(project, UNDATED_BUCKET)
    imgs.mkdir(parents=True)
    a = imgs / "a.jpg"
    _frame(a)
    _label(imgs, "a")
    run = opened_run(project, _cfg(project, imgs, normalization=_derived([a])))

    copy = tmp_path / "copy"
    ts.release_root(project)
    shutil.copytree(project, copy)
    sampling = read_run(
        copy / run.relative_to(project)
    ).spec.model_source.normalization.source

    assert [window[0] for window in sampling.windows] == [
        str((copy / a.relative_to(project)).resolve())
    ]


def test_preflight_refuses_a_model_source_stating_no_statistics(
    tmp_path: Path,
) -> None:
    """Preflight refuses a model source stating no normalization record, naming the field.

    The refusal is the field's type, where preflight validates the config.
    """
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = image_dir(tmp_path, UNDATED_BUCKET)
    imgs.mkdir(parents=True)
    cfg = _cfg(tmp_path, imgs, normalization={})
    del cfg["model_source"]["normalization"]

    r = preflight_config(tmp_path, cfg)
    assert r["valid"] is False
    assert any("model_source.normalization" in i for i in r["issues"])


def test_a_selection_bound_runs_checkpoint_records_statistics_preflight_admits(
    tmp_path: Path,
) -> None:
    """A selection-bound run's checkpoint records statistics that preflight admits.

    A run trained over a drawn selection (``_chain_fixtures.train_on``) registers a checkpoint
    whose statistics name exactly the sources its resolved train side holds, so its recorded model
    source, relaunched over that selection, passes preflight with no containment issue.
    """
    pytest.importorskip("torch")
    from tcip_mcp.experiments import CONFIG_PATHS
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.registry_paths import runtime_paths
    from tcip_mcp.tools.training_tools import preflight_config
    from tests._chain_fixtures import (
        draw_reference_selection,
        synthetic_capture,
        train_on,
    )

    root = tmp_path / "ds"
    synthetic_capture(root)
    selection_dir = tmp_path / "selection"
    draw_reference_selection(tmp_path, root, selection_dir)
    checkpoint = load_registered_checkpoint(
        train_on(selection_dir, tmp_path, "exp-own-statistics"),
        project=tmp_path,
    )

    windows = checkpoint.spec.model_source.normalization.source.windows
    assert {label for label, _ in windows} == _train_side(
        tmp_path, checkpoint.experiment_id
    )
    recorded = runtime_paths(checkpoint.stamp.config, CONFIG_PATHS, tmp_path)
    assert isinstance(recorded, dict)
    relaunch = {
        **run_config(tmp_path, selection_dir),
        "model_source": recorded["model_source"],
    }
    r = preflight_config(tmp_path, relaunch)
    assert r["issues"] == [], r["issues"]
    assert r["image_stats_containment"] == "checked"


def _train_side(project: Path, experiment_id: str) -> set:
    """Return the train-side sources of the partition run ``experiment_id`` recorded.

    The partition is the one ``experiment_id`` of ``project`` resolved and recorded, read off the
    record by the platform's own reader. It asserts the validation side holds a source.
    """
    from tcip_mcp.experiments import run_resolution

    samples = run_resolution(experiment_id, project=project).partition.samples
    assert any(s.side == "val" for s in samples), experiment_id
    return {str(s.source) for s in samples if s.side == "train"}


def test_a_configs_statistics_are_its_resolved_train_side_whatever_its_route(
    tmp_path: Path,
) -> None:
    """A config's statistics name exactly its resolved partition's train-side sources.

    Both configs are ``_chain_fixtures.training_config``'s: one over a selection redrawn at a
    stated seed, one over a directory drawing its own split.
    """
    pytest.importorskip("torch")
    from tests._chain_fixtures import (
        DETECTION_SOURCE,
        draw_reference_selection,
        synthetic_capture,
        training_config,
    )
    from tests._verified_checkpoint_fixtures import opened_run

    root = tmp_path / "ds"
    images_dir = synthetic_capture(root)
    selection_dir = tmp_path / "selection"
    draw_reference_selection(tmp_path, root, selection_dir)
    configs = {
        "exp-redrawn": training_config(
            tmp_path,
            DETECTION_SOURCE,
            {"split": {"selection_dir": str(selection_dir), "seed": 7}},
        ),
        "exp-drawn": training_config(
            tmp_path,
            DETECTION_SOURCE,
            {
                "images_dir": str(images_dir),
                "scope": {"subject": "bud"},
                "split": drawn_split(val_ratio=0.25, grouping="tile_prefix"),
            },
        ),
    }
    for experiment_id, config in configs.items():
        opened_run(tmp_path, config, experiment_id=experiment_id)
        windows = config["model_source"]["normalization"]["source"]["windows"]
        assert {label for label, _ in windows} == _train_side(
            tmp_path, experiment_id
        ), experiment_id


def test_own_statistics_refuses_a_config_whose_frames_carry_none(
    tmp_path: Path,
) -> None:
    """The config fixture's own statistics refuse a config whose frames carry none.

    The fixture's statistics have no fallback: a config whose data section names no frames
    refuses naming ``images_dir``, and frames whose bands each hold one value propagate the
    derivation's ``image_std`` refusal.
    """
    pytest.importorskip("torch")
    from pydantic import ValidationError

    from tcip_annotation.state import Annotation, BBox
    from tests._chain_fixtures import DETECTION_SOURCE, training_config
    from tests._producer_fixtures import label_image, write_image

    with pytest.raises(ValueError, match="images_dir"):
        training_config(
            tmp_path, DETECTION_SOURCE, {"scope": {"subject": "bud"}}
        )

    imgs = image_dir(tmp_path, UNDATED_BUCKET)
    for stem in ("a", "b"):
        label_image(
            write_image(imgs / f"{stem}.png", (16, 16), (90, 90, 90)),
            [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))],
            16,
            16,
        )
    with pytest.raises(ValidationError, match="image_std"):
        training_config(
            tmp_path,
            DETECTION_SOURCE,
            {
                "images_dir": str(imgs),
                "scope": {"subject": "bud"},
                "split": drawn_split(val_ratio=0.0),
            },
        )
