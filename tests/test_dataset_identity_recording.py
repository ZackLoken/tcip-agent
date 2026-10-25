"""A run records the dataset identity it launched against in its launch record.

The id and fingerprint are written into the run's launch record when it is opened, and
compare_experiments reports whether two runs share a dataset fingerprint.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import pytest

from tcip_mcp.experiments import compare_experiments
from tests._chain_fixtures import FRAME_STATISTICS, training_config
from tests._training_values import drawn_split
from tests._verified_checkpoint_fixtures import BUILT_DETECTOR, opened_run

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tcip_mcp.pipelines.data.selection import Sample

SUBJECT = "bud"


def _make_dataset(root: Path, *, shade: int = 0) -> Path:
    """Write two labeled images under one capture date and return their images directory.

    Image ``i`` is a ``labeled_frame`` on the ground ``(shade, 10 * i, 0)``.
    """
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import Subject, SubjectRegistry
    from tests._producer_fixtures import labeled_frame, registry_over

    images_dir = root / "images" / "2-11-26"
    images_dir.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        labeled_frame(
            images_dir / f"img_{i:03d}.jpg",
            [Annotation(subject=SUBJECT, geometry=BBox(1, 1, 9, 9))],
            64,
            64,
            (shade, 10 * i, 0),
        )
    registry_over(
        root,
        SubjectRegistry(subjects=(Subject(name=SUBJECT, description=""),)),
    )
    return images_dir


def _config(project: Path, images_dir: Path) -> dict:
    """Return a :data:`BUILT_DETECTOR` run's config of ``project`` over ``images_dir``."""
    return training_config(
        project,
        BUILT_DETECTOR,
        {
            "images_dir": str(images_dir),
            "scope": {"subject": SUBJECT},
            "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
        },
    )


def test_compare_experiments_surfaces_shared_fingerprint(
    tmp_path: Path,
) -> None:
    first = _config(tmp_path, _make_dataset(tmp_path / "first"))
    opened_run(tmp_path, first, experiment_id="a")
    opened_run(tmp_path, first, experiment_id="b")
    assert (
        compare_experiments(["a", "b"], project=tmp_path)[
            "same_dataset_fingerprint"
        ]
        is True
    )
    opened_run(
        tmp_path,
        _config(tmp_path, _make_dataset(tmp_path / "second", shade=200)),
        experiment_id="c",
    )
    assert (
        compare_experiments(["a", "c"], project=tmp_path)[
            "same_dataset_fingerprint"
        ]
        is False
    )


def test_compare_experiments_mixed_none_fingerprint_is_unknown_not_same(
    tmp_path: Path,
) -> None:
    """Comparing a run of known fingerprint with one of none reports unknown identity, not same.

    The second run is over a dataset holding no label document, and the comparison's
    ``same_dataset_fingerprint`` is ``None``.
    """
    from tests._verified_checkpoint_fixtures import table_images
    from tests.tiny_trainer_fixtures import (
        MEAN_INTENSITY_CLASSIFIER,
        TINY_TRAINER_FILE,
    )

    opened_run(
        tmp_path,
        _config(tmp_path, _make_dataset(tmp_path / "first")),
        experiment_id="a",
    )
    opened_run(
        tmp_path,
        training_config(
            tmp_path,
            {
                "builder": MEAN_INTENSITY_CLASSIFIER,
                "source_files": [TINY_TRAINER_FILE],
                "task": "classification",
                "normalization": FRAME_STATISTICS,
            },
            {
                **table_images(tmp_path / "loose"),
                "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
            },
        ),
        experiment_id="b",
    )
    assert (
        compare_experiments(["a", "b"], project=tmp_path)[
            "same_dataset_fingerprint"
        ]
        is None
    )


def test_dataset_identity_helper_registered_vs_bespoke(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.data.split_construction import dataset_identity
    from tcip_mcp.pipelines.schemas import DataSpec
    from tcip_mcp.tools.project_tools import register_dataset

    _make_dataset(tmp_path)
    reg = register_dataset(tmp_path, str(tmp_path), crop="currant", actor=None)
    ds_id, fp = dataset_identity(
        DataSpec.model_validate(
            {
                "images_dir": str(tmp_path / "images" / "2-11-26"),
                "split": drawn_split(),
            }
        )
    )
    assert ds_id == reg["id"] and fp == reg["fingerprint"]
    # bespoke / imageless run -> no fabricated identity
    assert dataset_identity(
        DataSpec.model_validate({"split": drawn_split()})
    ) == (None, None)


def test_dataset_identity_raises_a_fingerprint_read_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fingerprint read failure raises out of ``dataset_identity`` by name.

    A fingerprint read raising ``OSError`` mid-scan raises that error out of the call.
    """
    import tcip_mcp.pipelines.data.dataset_fingerprint as dataset_fingerprint_mod
    from tcip_mcp.pipelines.data.split_construction import dataset_identity
    from tcip_mcp.pipelines.schemas import DataSpec
    from tcip_mcp.tools.project_tools import register_dataset

    _make_dataset(tmp_path)
    register_dataset(tmp_path, str(tmp_path), crop="currant", actor=None)

    def _raise(_root: str | Path, _admitted: Iterable[Sample]) -> NoReturn:
        raise OSError("simulated I/O error mid-scan")

    # dataset_identity reads the fingerprint through its own module at call time, so it is
    # patched at the source module.
    monkeypatch.setattr(dataset_fingerprint_mod, "dataset_fingerprint", _raise)
    with pytest.raises(OSError, match="simulated I/O error"):
        dataset_identity(
            DataSpec.model_validate(
                {
                    "images_dir": str(tmp_path / "images" / "2-11-26"),
                    "split": drawn_split(),
                }
            )
        )


def test_a_trial_over_a_data_axis_records_the_dataset_its_own_input_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep trial over a data axis records the identity of the dataset its own input names.

    A sweep trial whose sampled point names another dataset records that dataset's id and
    fingerprint.
    """
    pytest.importorskip("torch")
    from tcip_mcp.experiments import read_run
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.tools.training_tools import open_trial
    from tests._verified_checkpoint_fixtures import opened_sweep

    base_images = _make_dataset(tmp_path / "base")
    register_dataset(
        tmp_path / "base", str(tmp_path / "base"), crop="currant", actor=None
    )
    other_images = _make_dataset(tmp_path / "other", shade=200)
    other = register_dataset(
        tmp_path / "other", str(tmp_path / "other"), crop="currant", actor=None
    )

    trial_dir = open_trial(
        opened_sweep(tmp_path, _config(tmp_path, base_images)),
        "a",
        {"data.images_dir": str(other_images)},
    )

    dataset = read_run(trial_dir).dataset
    assert (dataset.id, dataset.fingerprint) == (
        other["id"],
        other["fingerprint"],
    )


def test_a_launch_records_the_identity_of_the_dataset_it_trains_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launch records the identity of the dataset its config names.

    The run's launch record carries the registered dataset's id and fingerprint.
    """
    pytest.importorskip("torch")
    from tcip_mcp.experiments import read_run
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.tools.training_tools import launch_training
    from tests._producer_fixtures import fake_popen

    fake_popen(monkeypatch, [])

    images_dir = _make_dataset(tmp_path)
    registered = register_dataset(
        tmp_path, str(tmp_path), crop="currant", actor=None
    )
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR

    launched = launch_training(
        tmp_path,
        training_config(
            tmp_path,
            BUILT_DETECTOR,
            {
                "images_dir": str(images_dir),
                "scope": {"subject": SUBJECT},
                "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
            },
        ),
        actor=None,
    )
    assert "error" not in launched, launched

    dataset = read_run(Path(launched["output_dir"])).dataset
    assert (dataset.id, dataset.fingerprint) == (
        registered["id"],
        registered["fingerprint"],
    )
