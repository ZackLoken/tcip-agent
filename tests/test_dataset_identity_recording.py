"""A run records the dataset identity it launched against in its launch record.

The id and fingerprint are written into the run's launch record when it is opened, and
compare_experiments reports whether two runs share a dataset fingerprint.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import pytest

from tcip_mcp.dataset_layout import image_dir
from tcip_mcp.experiments import compare_experiments
from tests._chain_fixtures import (
    BUILT_DETECTOR,
    FRAME_STATISTICS,
    training_config,
)
from tests._training_values import drawn_split
from tests._verified_checkpoint_fixtures import opened_run

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tcip_mcp.pipelines.data.selection import Sample

SUBJECT = "bud"


def _make_dataset(root: Path, *, shade: int = 0) -> Path:
    """Write two labeled images under one capture date and return their images directory.

    Image ``i`` is a ``labeled_frame`` on the ground ``(shade, 10 * i, 0)``.
    """
    from tcip_annotation.state import Annotation, BBox
    from tests._producer_fixtures import (
        labeled_frame,
        registry_of,
        registry_over,
    )

    images_dir = image_dir(root, "2-11-26")
    images_dir.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        labeled_frame(
            images_dir / f"img_{i:03d}.jpg",
            [Annotation(subject=SUBJECT, geometry=BBox(1, 1, 9, 9))],
            64,
            64,
            (shade, 10 * i, 0),
        )
    registry_over(root, registry_of(SUBJECT))
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


def _fingerprints(ids: list[str], project: Path) -> list[str | None]:
    """Return the dataset fingerprint each compared run of ``ids`` carries, in order."""
    fingerprints = []
    for entry in compare_experiments(ids, project=project).experiments:
        assert entry.kind == "compared", entry
        fingerprints.append(entry.dataset_fingerprint)
    return fingerprints


def test_compare_experiments_surfaces_shared_fingerprint(
    tmp_path: Path,
) -> None:
    first = _config(tmp_path, _make_dataset(tmp_path / "first"))
    opened_run(tmp_path, first, experiment_id="a")
    opened_run(tmp_path, first, experiment_id="b")
    a, b = _fingerprints(["a", "b"], tmp_path)
    assert a is not None and a == b
    opened_run(
        tmp_path,
        _config(tmp_path, _make_dataset(tmp_path / "second", shade=200)),
        experiment_id="c",
    )
    a, c = _fingerprints(["a", "c"], tmp_path)
    assert None not in (a, c) and a != c


def test_compare_experiments_mixed_none_fingerprint_is_unknown_not_same(
    tmp_path: Path,
) -> None:
    """Comparing a run of known fingerprint with one of none carries the none, not a match.

    The second run is over a dataset holding no label document, and its column carries no
    fingerprint.
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
    a, b = _fingerprints(["a", "b"], tmp_path)
    assert a is not None and b is None


def test_dataset_identity_helper_registered_vs_bespoke(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.data.split_construction import dataset_identity
    from tcip_mcp.pipelines.schemas import DataSpec
    from tcip_mcp.tools.project_tools import register_dataset

    _make_dataset(tmp_path)
    reg = register_dataset(tmp_path, str(tmp_path), crop="currant", actor=None)
    ds_id, fp = dataset_identity(
        DataSpec.model_validate(
            {
                "images_dir": str(image_dir(tmp_path, "2-11-26")),
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
                    "images_dir": str(image_dir(tmp_path, "2-11-26")),
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
    from tcip_mcp.tools.training_tools import RunLaunched, launch_training
    from tests._producer_fixtures import fake_popen

    fake_popen(monkeypatch, [])

    images_dir = _make_dataset(tmp_path)
    registered = register_dataset(
        tmp_path, str(tmp_path), crop="currant", actor=None
    )

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
    assert isinstance(launched, RunLaunched), launched

    dataset = read_run(Path(launched.output_dir)).dataset
    assert (dataset.id, dataset.fingerprint) == (
        registered["id"],
        registered["fingerprint"],
    )
