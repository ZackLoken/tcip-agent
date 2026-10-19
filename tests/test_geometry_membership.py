"""A geometry route's loaders answer for their own membership.

These drive the drawn, zero-share and within-image routes under each geometry task, with the
geometry that task's loader reads, and read each loader's membership back off the loader
itself: its keys name samples whose sources are on disk, and the drawn route's training loader
holds the training side its partition records.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET

if TYPE_CHECKING:
    from torch.utils.data import Dataset

    from tcip_mcp.pipelines.data.datasets import BaseImageDataset
    from tcip_mcp.pipelines.data.split_construction import Partition

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_annotation.state import Annotation, BBox, Polygon  # noqa: E402
from tcip_mcp.pipelines.model_contract import (  # noqa: E402
    DETECTOR_TASKS,
    TILING_TASKS,
)
from tests._producer_fixtures import (  # noqa: E402
    big_single_source,
    labeled_frame,
    train_val,
)
from tests._training_values import (  # noqa: E402
    drawn_split,
    evaluation_block,
    tiling_block,
)
from tests._verified_checkpoint_fixtures import partition_side as recorded_side  # noqa: E402

IMG = 64
SUBJECT = "bud"


def _target_geometry(task: str) -> Polygon | BBox:
    """Return a four-point square ``Polygon`` for instance_seg and a ``BBox`` for any other task."""
    if task == "instance_seg":
        return Polygon(
            rings=[[(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)]]
        )
    return BBox(10.0, 10.0, 30.0, 30.0)


def _labeled(root: Path, stems: list[str], *, task: str = "detection") -> Path:
    """Write one labeled directory and return it.

    Each stem gets a ``labeled_frame`` under ``root``'s undated bucket, labeled with one
    ``SUBJECT`` annotation of ``_target_geometry(task)``.
    """
    images_dir = root / "images" / UNDATED_BUCKET
    for stem in stems:
        labeled_frame(
            images_dir / f"{stem}.png",
            [Annotation(subject=SUBJECT, geometry=_target_geometry(task))],
            IMG,
            IMG,
            (70, 90, 60),
        )
    return images_dir


def _one_target(ds: Dataset, task: str) -> None:
    """Read one example off a loader and assert the task's own target shape came back."""
    _image_tensor, target = ds[0]
    assert target["boxes"].tolist() == [[10.0, 10.0, 30.0, 30.0]]
    if task == "instance_seg":
        assert target["masks"].shape[0] == 1
        assert int(target["masks"].sum()) > 0


def _membership(ds: BaseImageDataset) -> set[str]:
    """Return what one loader says its own members are, read off the samples it was built from.

    Assert each key the loader indexes names a sample whose source is on disk.
    """
    keys = set(ds.stems)
    for key in keys:
        assert Path(ds.sample_of(key).source).is_file()
    return {ds.sample_of(key).member for key in keys}


def _recorded(
    project: Path, task: str, data_cfg: dict, **config: object
) -> Partition:
    """Return the partition the launcher's own producer resolves for a ``task`` run.

    The run is of ``project`` over ``data_cfg``, with ``config``'s keys laid beside it, and the
    partition is read back from the launch record it writes.
    """
    import copy

    from tcip_mcp.experiments import run_resolution
    from tests._chain_fixtures import training_config
    from tests._verified_checkpoint_fixtures import opened_run, unbuilt_source

    run_dir = opened_run(
        project,
        training_config(
            project, unbuilt_source(task), copy.deepcopy(data_cfg), **config
        ),
    )
    return run_resolution(run_dir.name, project=project).partition


@pytest.mark.parametrize("task", sorted(DETECTOR_TASKS))
def test_the_drawn_route_loaders_name_their_own_samples(
    tmp_path: Path, task: str
) -> None:
    stems = [f"src{i}_0_0" for i in range(4)]
    images_dir = _labeled(tmp_path / "ds", stems, task=task)
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": SUBJECT},
        "split": drawn_split(val_ratio=0.5, seed=1, grouping="tile_prefix"),
    }

    train_ds, val_ds, partition, _data = train_val(tmp_path, task, data_cfg)

    assert val_ds is not None
    assert _membership(train_ds).isdisjoint(_membership(val_ds))
    assert _membership(train_ds) | _membership(val_ds) == set(stems)
    assert recorded_side(partition, "train") == sorted(_membership(train_ds))
    _one_target(train_ds, task)


@pytest.mark.parametrize("task", sorted(DETECTOR_TASKS))
def test_a_zero_validation_share_trains_on_every_admitted_sample_and_records_them(
    tmp_path: Path, task: str
) -> None:
    """A zero validation share loads and records the three admitted samples on the training side.

    With all shares at zero and selection on the training loss, the training loader holds the
    three admitted samples, no validation loader is built, and the partition records them on the
    training side; that split selecting on a validation metric refuses, naming the validation
    loader it lacks.
    """
    from tests._chain_fixtures import training_config
    from tests._verified_checkpoint_fixtures import opened_run, unbuilt_source

    stems = ["a_0_0", "b_0_0", "c_0_0"]
    images_dir = _labeled(tmp_path / "ds", stems, task=task)
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": SUBJECT},
        "split": drawn_split(val_ratio=0.0, grouping="tile_prefix"),
    }
    record = _recorded(
        tmp_path,
        task,
        data_cfg,
        evaluation=evaluation_block(selection_metric="loss"),
    )

    train_ds, val_ds, partition, _data = train_val(tmp_path, task, data_cfg)

    assert val_ds is None
    assert _membership(train_ds) == set(stems)
    assert recorded_side(partition, "train") == sorted(stems)
    assert recorded_side(partition, "val") == []
    _one_target(train_ds, task)
    assert recorded_side(record, "train") == sorted(stems)
    with pytest.raises(ValueError, match="needs a validation loader"):
        opened_run(
            tmp_path,
            training_config(
                tmp_path,
                unbuilt_source(task),
                data_cfg,
                evaluation=evaluation_block(selection_metric="objective"),
            ),
        )


@pytest.mark.parametrize("task", sorted(DETECTOR_TASKS))
def test_a_starved_draw_refuses_by_the_draws_own_floor(
    tmp_path: Path, task: str
) -> None:
    """A draw starved to one group refuses naming its own floor.

    A grouping map putting both samples in one group refuses, naming the two groups the
    requested sides need.
    """
    stems = ["a_0_0", "b_0_0"]
    images_dir = _labeled(tmp_path / "ds", stems, task=task)
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": SUBJECT},
        "split": drawn_split(
            val_ratio=0.5,
            seed=1,
            grouping={f"{UNDATED_BUCKET}/{s}": "one_group" for s in stems},
        ),
    }

    with pytest.raises(
        ValueError, match="fewer than the 2 the requested sides need"
    ):
        train_val(tmp_path, task, data_cfg)


@pytest.mark.parametrize("task", sorted(TILING_TASKS))
def test_one_tiled_source_still_splits_spatially_over_its_own_samples(
    tmp_path: Path, task: str
) -> None:
    """One tiled source splits spatially over its own sample.

    For each task in ``TILING_TASKS``, the within-image route records the one sample as its
    partition, both loaders hold it on disjoint tiles, and its training region identities name
    the bare stem.
    """
    images_dir, stem = big_single_source(
        tmp_path / "ds", 4000, 3000, polygons=task == "instance_seg"
    )
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": SUBJECT},
        "tiling": tiling_block(tile_size=128, overlap=0.2),
        "split": drawn_split(
            val_ratio=0.25, holdout_ratio=0.1, seed=1, grouping="tile_prefix"
        ),
    }

    train_ds, val_ds, partition, _resolved = train_val(
        tmp_path, task, data_cfg
    )

    assert val_ds is not None
    assert [s.member for s in partition.samples] == [stem]
    assert _membership(train_ds) == _membership(val_ds) == {stem}
    assert set(train_ds.tile_entries).isdisjoint(set(val_ds.tile_entries))
    manifest = partition.spatial
    assert manifest is not None
    assert manifest.stem == stem
    assert all(
        i.startswith(f"{stem}::strip_") for i in manifest.train_identities
    )
    assert set(manifest.train_identities).isdisjoint(manifest.val_identities)


def test_the_train_only_and_drawn_routes_record_one_directory_the_same_way(
    tmp_path: Path,
) -> None:
    """A run holding nothing out and a drawn run record one directory's members alike.

    One training directory of four samples is admitted by a run holding nothing out and by a
    run drawing its own split. The two record the same four members, sources, ground truths and
    digests, and the drawn run alone records a validation side.
    """
    stems = ["a_0_0", "b_0_0", "c_0_0", "d_0_0"]
    images_dir = _labeled(tmp_path / "train_ds", stems)

    whole_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": SUBJECT},
        "split": drawn_split(val_ratio=0.0, seed=1, grouping="tile_prefix"),
    }
    whole = _recorded(
        tmp_path,
        "detection",
        whole_cfg,
        evaluation=evaluation_block(selection_metric="loss"),
    )

    drawn_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": SUBJECT},
        "split": drawn_split(val_ratio=0.5, seed=1, grouping="tile_prefix"),
    }
    drawn = _recorded(tmp_path, "detection", drawn_cfg)

    def _held(record: Partition) -> list[tuple]:
        return sorted(
            (s.member, s.source, s.ground_truth, s.ground_truth_digest)
            for s in record.samples
        )

    assert _held(whole) == _held(drawn)
    assert len(_held(whole)) == len(stems)
    assert all(digest for *_, digest in _held(whole))
    assert (
        recorded_side(whole, "val") == [] and recorded_side(drawn, "val") != []
    )
