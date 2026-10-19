"""Membership for the tasks a per-image label document does not answer for.

A semantic segmentation run's ground truth is a mask raster beside its image; a classification,
ordinal or regression run's is one row of a table. Both are drawn into selections by
``draw_splits``, and their membership is read back off the loaders the run builds.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.data import Dataset

    from tcip_mcp.experiments import ResolvedInput
    from tcip_mcp.pipelines.data.selection import Selection

pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from tcip_mcp.dataset_layout import UNDATED_BUCKET  # noqa: E402
from tcip_mcp.pipelines.data.selection import ClassScope, read_selection  # noqa: E402
from tcip_mcp.pipelines.data.split_construction import (  # noqa: E402
    Partition,
    auto_train_val,
)
from tcip_mcp.pipelines.schemas import DataSpec  # noqa: E402
from tcip_mcp.tools.data_tools import draw_splits  # noqa: E402
from tests._chain_fixtures import training_config  # noqa: E402
from tests._producer_fixtures import painted_frame, staged_layout  # noqa: E402
from tests._training_values import IOU_THRESHOLD, drawn_split  # noqa: E402
from tests._verified_checkpoint_fixtures import partition_side, unbuilt_source  # noqa: E402


def _train_val(
    project: Path, task: str, data: DataSpec
) -> tuple[Dataset, Dataset | None, Partition, DataSpec]:
    """Return ``split_construction.auto_train_val`` of a ``task`` run of ``project`` over ``data``.

    The run declares no dataset builder and is untransformed; the answer is
    ``(train_ds, val_ds, partition, resolved)``.
    """
    return auto_train_val(
        project, task, data, None, staged_layout(project, {})
    )


STEMS = ("a", "b", "c", "d", "e", "f", "g", "h")


def _resolved(
    project: Path, experiment_id: str, task: str, data_cfg: dict
) -> tuple[ResolvedInput, Dataset, Dataset | None]:
    """Resolve a run over ``data_cfg`` and return its record and its loaders' datasets.

    The run is under ``project``, resolved by the launcher's own producer: what its launch record
    says it resolved, and the train and val datasets the child's own context producer builds its
    loaders from out of that record.
    """
    from tcip_mcp.experiments import observe, run_resolution
    from tcip_mcp.pipelines.training.subprocess_worker import (
        prepare_run_context,
    )
    from tests._verified_checkpoint_fixtures import opened_run

    run_dir = opened_run(
        project,
        training_config(project, unbuilt_source(task), data_cfg),
        experiment_id=experiment_id,
    )
    ctx = prepare_run_context(observe(run_dir))
    val = ctx.val_loader.dataset if ctx.val_loader is not None else None
    return (
        run_resolution(experiment_id, project=project),
        ctx.train_loader.dataset,
        val,
    )


def _mask_dataset(root: Path) -> tuple[Path, Path]:
    """Write a dataset whose ground truth is one ``<stem>.png`` mask per image.

    For the nth of :data:`STEMS`, the 16 px image goes under ``root``'s undated bucket and the
    mask, painted 1 over a square of side n + 1 at its origin, under ``root / "masks"``.
    """
    images_dir, masks_dir = root / "images" / UNDATED_BUCKET, root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(STEMS):
        Image.new("RGB", (16, 16), (10 * index, 20 + index, 30 + index)).save(
            images_dir / f"{stem}.png"
        )
        painted_frame(
            16, 16, 0, [((0, 0, index + 1, index + 1), 1)], mode="L"
        ).save(masks_dir / f"{stem}.png")
    return images_dir, masks_dir


def _table_dataset(root: Path) -> tuple[Path, Path]:
    """Write a dataset whose ground truth is one table row per image, labels cycling 0, 1, 2."""
    images_dir, csv_path = (
        root / "images" / UNDATED_BUCKET,
        root / "labels.csv",
    )
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, stem in enumerate(STEMS):
        Image.new("RGB", (16, 16), (10 * index, 20 + index, 30 + index)).save(
            images_dir / f"{stem}.png"
        )
        rows.append((stem, index % 3))
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stem", "label"))
        writer.writerows(rows)
    return images_dir, csv_path


def _drawn(
    root: Path, ground_truth: Path, out: Path, *, seed: int = 4
) -> Selection:
    result = draw_splits(
        out.parent,
        str(root),
        output_path=str(out),
        ground_truth=str(ground_truth),
        **drawn_split(
            seed=seed,
            val_ratio=0.25,
            calibration_ratio=0.125,
            holdout_ratio=0.125,
        ),
    )
    assert "error" not in result, result
    return read_selection(out, project=out.parent)


# -- a bound run trains over exactly the selection's own samples ---------------


def test_a_bound_semantic_seg_run_trains_over_exactly_its_selections_samples(
    tmp_path: Path,
) -> None:
    """A bound semantic_seg run's loaders hold exactly its selection's samples.

    The training and validation loaders hold the samples the selection recorded on each side,
    none from the calibration side, and the partition records the training side.
    """
    root = tmp_path / "ds"
    _images_dir, masks_dir = _mask_dataset(root)
    out = tmp_path / "m"
    drawn = _drawn(root, masks_dir, out)

    data = DataSpec.model_validate({"split": {"selection_dir": str(out)}})
    train_ds, val_ds, partition, _data = _train_val(
        tmp_path, "semantic_seg", data
    )

    assert sorted(train_ds.stems) == sorted(
        s.location for s in drawn.on("train")
    )
    assert sorted(val_ds.stems) == sorted(s.location for s in drawn.on("val"))
    held_out = {s.location for s in drawn.on("calibration")}
    assert held_out and not held_out & set(train_ds.stems + val_ds.stems)
    assert partition_side(partition, "train") == sorted(
        s.member for s in drawn.on("train")
    )


def test_a_bound_classification_run_trains_over_exactly_its_selections_samples(
    tmp_path: Path,
) -> None:
    """A bound classification run's sides hold the table rows its selection put on them."""
    root = tmp_path / "ds"
    _images_dir, csv_path = _table_dataset(root)
    out = tmp_path / "m"
    drawn = _drawn(root, csv_path, out)

    data = DataSpec.model_validate({"split": {"selection_dir": str(out)}})
    train_ds, val_ds, partition, _data = _train_val(
        tmp_path, "classification", data
    )

    assert sorted(train_ds.stems) == sorted(
        s.location for s in drawn.on("train")
    )
    assert sorted(val_ds.stems) == sorted(s.location for s in drawn.on("val"))
    held_out = {s.location for s in drawn.on("calibration")}
    assert held_out and not held_out & set(train_ds.stems + val_ds.stems)
    assert {s.ground_truth for s in partition.samples} == {str(csv_path)}


def _document_dataset(root: Path) -> tuple[Path, None]:
    """Write a dataset whose ground truth is one label document per image, each holding a ``bud``.

    Return its images directory and ``None`` for the ground-truth place.
    """
    from tcip_annotation.state import Annotation, BBox
    from tests._producer_fixtures import label_image

    images_dir = root / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True, exist_ok=True)
    for stem in STEMS:
        Image.new("RGB", (16, 16), (10, 20, 30)).save(
            images_dir / f"{stem}.png"
        )
        label_image(
            images_dir / f"{stem}.png",
            [Annotation(subject="bud", geometry=BBox(1, 1, 8, 8))],
            16,
            16,
        )
    return images_dir, None


def _drop_ground_truth_of_a(
    images_dir: Path, ground_truth: Path | None
) -> None:
    """Remove member ``a``'s label document, table row or mask, by ``ground_truth``'s form."""
    import tcip_store
    from tests._producer_fixtures import image_label_key

    if ground_truth is None:
        tcip_store.delete(image_label_key(images_dir / "a.png"))
    elif ground_truth.suffix == ".csv":
        rows = [
            r for r in csv.reader(ground_truth.open(newline="")) if r[0] != "a"
        ]
        with open(ground_truth, "w", newline="") as handle:
            csv.writer(handle).writerows(rows)
    else:
        next(p for p in ground_truth.iterdir() if p.stem == "a").unlink()


def _drop_image_of_a(images_dir: Path, ground_truth: Path | None) -> None:
    """Remove member ``a``'s image, and empty its label document when it has one."""
    from tests._producer_fixtures import label_image

    (images_dir / "a.png").unlink()
    if ground_truth is None:
        label_image(images_dir / "a.png", [], 16, 16, keep_empty=True)


@pytest.mark.parametrize(
    "dataset",
    [_document_dataset, _mask_dataset, _table_dataset],
    ids=["document", "mask", "table"],
)
@pytest.mark.parametrize("drop", ["ground_truth", "image"])
def test_a_recorded_sample_is_tallied_as_a_fresh_admission_tallies_it(
    tmp_path: Path,
    dataset: Callable[[Path], tuple[Path, Path | None]],
    drop: str,
) -> None:
    """A recorded sample is tallied as a fresh admission tallies it.

    With one member's ground truth or image gone, the refusal of the recorded samples names the
    refused tallies a fresh admission over the same members answers.
    """
    from tcip_mcp.pipelines.data.label_queries import (
        admit,
        admits,
        readmitted_samples,
    )

    images_dir, ground_truth = dataset(tmp_path / "ds")
    scope = (
        ClassScope(subject="bud", attributes=())
        if dataset is _document_dataset
        else None
    )
    admission = admit(images_dir, ground_truth, scope=scope)
    samples = admission.every_sample()
    if drop == "image":
        _drop_image_of_a(images_dir, ground_truth)
    else:
        _drop_ground_truth_of_a(images_dir, ground_truth)

    fresh = admit(
        images_dir,
        ground_truth,
        scope=admission.scope,
        members=[s.member for s in samples],
    )
    refused = ", ".join(
        f"{name}={count}"
        for name, count in sorted(fresh.tallies.items())
        if not admits(name)
    )
    assert refused == "absent=1"
    with pytest.raises(ValueError, match=f"a\\.png'\\]\\): {refused}\\."):
        readmitted_samples(samples, admission.scope)


def test_a_directory_named_like_a_mask_is_admitted_by_neither_read_of_it(
    tmp_path: Path,
) -> None:
    """A directory named like a mask is admitted by neither read of it.

    A directory named ``a.1.png`` among the masks is not admitted, and the samples that are
    admitted pass re-admission.
    """
    from tcip_mcp.pipelines.data.label_queries import admit, readmitted_samples

    images_dir, masks_dir = (
        tmp_path / "images" / UNDATED_BUCKET,
        tmp_path / "masks",
    )
    images_dir.mkdir(parents=True)
    masks_dir.mkdir()
    for stem in ("a.1", "b"):
        Image.new("RGB", (16, 16), (10, 20, 30)).save(
            images_dir / f"{stem}.png"
        )
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8), mode="L").save(
        masks_dir / "b.png"
    )
    (masks_dir / "a.1.png").mkdir()

    admitted = admit(images_dir, masks_dir)
    assert [record.member for record in admitted.records] == ["b"]
    # Admits valid work: what it did admit still admits when it is checked again.
    readmitted_samples(
        admitted.samples(
            {record.member: "train" for record in admitted.records},
            lambda m: m,
        ),
        admitted.scope,
    )


# -- each sample reaches its own ground truth ----------------------------------


def test_a_mask_sample_reaches_its_own_mask(tmp_path: Path) -> None:
    """A mask sample reaches its own mask.

    Each training sample's mask holds the distinct foreground area of the file its sample
    names.
    """
    root = tmp_path / "ds"
    _images_dir, masks_dir = _mask_dataset(root)
    out = tmp_path / "m"
    drawn = _drawn(root, masks_dir, out)

    train_ds, _val_ds, _partition, _data = _train_val(
        tmp_path,
        "semantic_seg",
        DataSpec.model_validate({"split": {"selection_dir": str(out)}}),
    )

    by_identity = {s.location: s for s in drawn.samples}
    for index, key in enumerate(train_ds.stems):
        sample = by_identity[key]
        expected = STEMS.index(sample.member) + 1
        _image, target = train_ds[index]
        assert int(target["masks"].sum()) == expected * expected
        assert Path(sample.ground_truth) == masks_dir / f"{sample.member}.png"


def test_a_row_key_sample_reaches_the_row_it_names(tmp_path: Path) -> None:
    """A row-key sample reaches the row it names.

    Each training sample's label is the one its ``row_key`` names in the table it names.
    """
    root = tmp_path / "ds"
    _images_dir, csv_path = _table_dataset(root)
    out = tmp_path / "m"
    drawn = _drawn(root, csv_path, out)

    train_ds, _val_ds, _partition, _data = _train_val(
        tmp_path,
        "classification",
        DataSpec.model_validate({"split": {"selection_dir": str(out)}}),
    )

    by_identity = {s.location: s for s in drawn.samples}
    for index, key in enumerate(train_ds.stems):
        sample = by_identity[key]
        assert sample.row_key == sample.member
        assert Path(sample.ground_truth) == csv_path
        _image, target = train_ds[index]
        assert target["labels"] == STEMS.index(sample.row_key) % 3


# -- an unbound run's loaders answer for their own membership -------------------


def test_an_unbound_semantic_seg_run_reads_its_membership_off_its_own_samples(
    tmp_path: Path,
) -> None:
    """An unbound semantic_seg run reads its membership off its own samples.

    With no selection, the loaders' members split the eight stems as the recorded partition does,
    and each sample holds its own source and mask.
    """
    root = tmp_path / "ds"
    images_dir, masks_dir = _mask_dataset(root)
    data = DataSpec.model_validate(
        {
            "images_dir": str(images_dir),
            "labels_dir": str(masks_dir),
            "split": drawn_split(
                val_ratio=0.25, seed=7, grouping="tile_prefix"
            ),
        }
    )

    train_ds, val_ds, partition, _data = _train_val(
        tmp_path, "semantic_seg", data
    )

    assert val_ds is not None
    members = {
        ds: {ds.sample_of(key).member for key in ds.stems}
        for ds in (train_ds, val_ds)
    }
    assert members[train_ds].isdisjoint(members[val_ds])
    assert members[train_ds] | members[val_ds] == set(STEMS)
    assert partition_side(partition, "train") == sorted(members[train_ds])
    assert partition_side(partition, "val") == sorted(members[val_ds])
    # Each loader indexes by its samples' own source identities and holds that sample's own
    # source, ground truth and member name.
    for ds in (train_ds, val_ds):
        for key in ds.stems:
            sample = ds.sample_of(key)
            assert sample.source == key, (
                "a sample is indexed by its own source"
            )
            assert Path(sample.source) == images_dir / f"{sample.member}.png"
            assert (
                Path(sample.ground_truth) == masks_dir / f"{sample.member}.png"
            )


def test_an_unbound_regression_run_reads_its_membership_off_its_own_samples(
    tmp_path: Path,
) -> None:
    """An unbound regression run reads its membership off its own samples.

    The two loaders' members split the eight stems as the recorded partition does, and each
    sample is indexed by its own image path and names the run's table as its ground truth.
    """
    root = tmp_path / "ds"
    images_dir, csv_path = _table_dataset(root)
    data = DataSpec.model_validate(
        {
            "images_dir": str(images_dir),
            "labels_dir": str(csv_path),
            "split": drawn_split(
                val_ratio=0.25, seed=7, grouping="tile_prefix"
            ),
        }
    )

    train_ds, val_ds, partition, _data = _train_val(
        tmp_path, "regression", data
    )

    assert val_ds is not None
    members = {
        ds: {ds.sample_of(key).member for key in ds.stems}
        for ds in (train_ds, val_ds)
    }
    assert members[train_ds].isdisjoint(members[val_ds])
    assert members[train_ds] | members[val_ds] == set(STEMS)
    assert partition_side(partition, "train") == sorted(members[train_ds])
    assert partition_side(partition, "val") == sorted(members[val_ds])
    for ds in (train_ds, val_ds):
        for key in ds.stems:
            sample = ds.sample_of(key)
            assert sample.source == key, (
                "a sample is indexed by its own source"
            )
            assert Path(sample.ground_truth) == csv_path
            assert Path(key) == images_dir / f"{sample.member}.png"


# -- a member's name is its row key, dots and all ------------------------------


def test_a_dotted_row_key_names_one_member_end_to_end(tmp_path: Path) -> None:
    """A dotted row key names one member end to end.

    A row key is the member's own name: ``a.0`` through ``a.7`` are eight members in the
    loaders' own membership and in the resolved partition, each its own group.
    """
    root = tmp_path / "ds"
    images_dir, csv_path = (
        root / "images" / UNDATED_BUCKET,
        root / "labels.csv",
    )
    images_dir.mkdir(parents=True, exist_ok=True)
    dotted = [f"a.{index}" for index in range(8)]
    rows = []
    for index, key in enumerate(dotted):
        Image.new("RGB", (16, 16), (10 * index, 20 + index, 30 + index)).save(
            images_dir / f"{key}.png"
        )
        rows.append((key, index % 3))
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stem", "label"))
        writer.writerows(rows)

    data_cfg = {
        "images_dir": str(images_dir),
        "labels_dir": str(csv_path),
        "split": drawn_split(val_ratio=0.5, seed=3),
    }
    resolved, train_ds, val_ds = _resolved(
        tmp_path, "exp-dotted", "classification", data_cfg
    )

    # The loaders name every row apart, not one collapsed member.
    named = {
        ds.sample_of(k).member for ds in (train_ds, val_ds) for k in ds.stems
    }
    assert named == set(dotted)
    assert len(train_ds.stems) == train_ds.num_samples

    samples = resolved.partition.samples
    assert sorted(s.member for s in samples) == sorted(dotted)
    assert len({s.group for s in samples}) == len(dotted)


# -- freezing a run whose ground truth is not a label document -----------------


def _frozen(
    project: Path, experiment_id: str, task: str, data_cfg: dict
) -> dict:
    """Resolve a run under ``project`` through the child's own producer and freeze it."""
    from tcip_mcp.tools.data_tools import freeze_selection

    _resolved(project, experiment_id, task, data_cfg)
    return freeze_selection(project, experiment_id)


def test_a_mask_run_freezes_into_a_selection_its_bind_accepts(
    tmp_path: Path,
) -> None:
    """A mask run freezes into a selection its bind accepts.

    Each frozen sample names an existing ``.png`` under the run's mask directory, and a later
    run bound to the selection builds a training dataset holding exactly the frozen training
    side.
    """
    root = tmp_path / "ds"
    images_dir, masks_dir = _mask_dataset(root)
    data_cfg = {
        "images_dir": str(images_dir),
        "labels_dir": str(masks_dir),
        "split": drawn_split(val_ratio=0.25, seed=7, grouping="tile_prefix"),
    }

    result = _frozen(tmp_path, "exp-mask-freeze", "semantic_seg", data_cfg)

    assert "error" not in result, result
    frozen = read_selection(result["selection_dir"], project=tmp_path)
    assert frozen.scope == ClassScope()
    assert {Path(s.ground_truth).suffix for s in frozen.samples} == {".png"}
    for sample in frozen.samples:
        assert Path(sample.ground_truth).parent == masks_dir
        assert Path(sample.ground_truth).is_file()

    train_ds, val_ds, partition, _data = _train_val(
        tmp_path,
        "semantic_seg",
        DataSpec.model_validate(
            {"split": {"selection_dir": result["selection_dir"]}}
        ),
    )
    assert sorted(train_ds.stems) == sorted(
        s.location for s in frozen.on("train")
    )
    assert val_ds is not None
    assert result["train"] == len(frozen.on("train"))
    assert partition_side(partition, "train") == sorted(
        s.member for s in frozen.on("train")
    )


def test_a_table_run_freezes_into_a_selection_its_bind_accepts(
    tmp_path: Path,
) -> None:
    """A table run freezes into a selection its bind accepts.

    Each frozen sample names the run's table and carries a row key, and a later run bound to the
    selection builds a training dataset holding the frozen training side and a validation one.
    """
    root = tmp_path / "ds"
    images_dir, csv_path = _table_dataset(root)
    data_cfg = {
        "images_dir": str(images_dir),
        "labels_dir": str(csv_path),
        "split": drawn_split(val_ratio=0.25, seed=7, grouping="tile_prefix"),
    }

    result = _frozen(tmp_path, "exp-table-freeze", "classification", data_cfg)

    assert "error" not in result, result
    frozen = read_selection(result["selection_dir"], project=tmp_path)
    assert {s.ground_truth for s in frozen.samples} == {str(csv_path)}
    assert all(s.row_key is not None for s in frozen.samples)

    train_ds, val_ds, _partition, _data = _train_val(
        tmp_path,
        "classification",
        DataSpec.model_validate(
            {"split": {"selection_dir": result["selection_dir"]}}
        ),
    )
    assert sorted(train_ds.stems) == sorted(
        s.location for s in frozen.on("train")
    )
    assert val_ds is not None


def test_a_mask_edited_after_the_run_refuses_the_freeze_by_name(
    tmp_path: Path,
) -> None:
    """A mask edited after the run refuses the freeze by name.

    A member's mask replaced after the run refuses the freeze, naming the member.
    """
    root = tmp_path / "ds"
    images_dir, masks_dir = _mask_dataset(root)
    data_cfg = {
        "images_dir": str(images_dir),
        "labels_dir": str(masks_dir),
        "split": drawn_split(val_ratio=0.25, seed=7, grouping="tile_prefix"),
    }
    from tcip_mcp.tools.data_tools import freeze_selection

    _resolved(tmp_path, "exp-mask-moved", "semantic_seg", data_cfg)

    painted_frame(16, 16, 0, [((0, 0, 12, 12), 1)], mode="L").save(
        masks_dir / f"{STEMS[0]}.png"
    )

    result = freeze_selection(tmp_path, "exp-mask-moved")

    assert "error" in result and "changed since" in result["error"]
    assert STEMS[0] in result["error"]


def _three_class_masks(root: Path) -> tuple[Path, Path]:
    """Write four images and masks painted 1 over (0, 0, 8, 8), the first 2 over (8, 8, 16, 16)."""
    images_dir, masks_dir = root / "images" / UNDATED_BUCKET, root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(STEMS[:4]):
        Image.new("RGB", (16, 16), (10 * index, 20 + index, 30 + index)).save(
            images_dir / f"{stem}.png"
        )
        extents = [((0, 0, 8, 8), 1)] + (
            [((8, 8, 16, 16), 2)] if index == 0 else []
        )
        painted_frame(16, 16, 0, extents, mode="L").save(
            masks_dir / f"{stem}.png"
        )
    return images_dir, masks_dir


def test_a_stated_class_count_refuses_on_the_run_path_whatever_its_value(
    tmp_path: Path,
) -> None:
    """A stated class count refuses on the run path, below, at or above the derived count.

    A config stating a class count of 2, 3 or 9 refuses naming ``num_classes``, and a config
    stating none records the derived count of three.
    """
    images_dir, masks_dir = _three_class_masks(tmp_path / "ds")
    base = {
        "images_dir": str(images_dir),
        "labels_dir": str(masks_dir),
        "split": drawn_split(val_ratio=0.25, seed=3),
    }

    for stated in (2, 3, 9):
        with pytest.raises(ValueError, match=r"states \['num_classes'\]"):
            _train_val(
                tmp_path,
                "semantic_seg",
                DataSpec.model_validate({**base, "num_classes": stated}),
            )

    _train_ds, val_ds, _partition, resolved = _train_val(
        tmp_path, "semantic_seg", DataSpec.model_validate(base)
    )
    assert val_ds is not None
    assert resolved.num_classes == 3 and resolved.num_ranks is None


def test_a_table_run_stating_a_count_refuses(tmp_path: Path) -> None:
    """A table run stating a class count refuses by name.

    Table values ``0`` and ``1`` derive two classes; a count stated beside them refuses by
    name, and the unstated run records two.
    """
    from tests._producer_fixtures import run_over

    images_dir, csv_path = _table_dataset(tmp_path / "table")
    with open(csv_path, "w", newline="") as handle:
        csv.writer(handle).writerows(
            [("stem", "label"), *((s, i % 2) for i, s in enumerate(STEMS))]
        )
    with pytest.raises(ValueError, match=r"states \['num_classes'\]"):
        run_over(
            "classification",
            str(images_dir),
            str(csv_path),
            stated={"num_classes": 9},
        )

    _dataset, data = run_over("classification", str(images_dir), str(csv_path))
    assert data["num_classes"] == 2


def test_a_runs_unstated_class_count_is_read_once_over_every_sample(
    tmp_path: Path,
) -> None:
    """A run's unstated class count is read over the samples of both sides.

    With the one mask reaching class 2 on the validation side, the run records three classes.
    """
    images_dir, masks_dir = _three_class_masks(tmp_path / "ds")
    data = DataSpec.model_validate(
        {
            "images_dir": str(images_dir),
            "labels_dir": str(masks_dir),
            "split": drawn_split(val_ratio=0.25, seed=3),
        }
    )

    train_ds, val_ds, partition, resolved = _train_val(
        tmp_path, "semantic_seg", data
    )

    # The one mask reaching class 2 is on the validation side, which is the disagreement a
    # per-loader count would produce here.
    assert partition_side(partition, "val") == [STEMS[0]]
    assert val_ds is not None
    assert resolved.num_classes == 3
    assert resolved.num_channels == train_ds.expected_channels


def test_a_runs_metrics_are_reported_over_the_class_space_it_trains_in(
    tmp_path: Path,
) -> None:
    """Evaluation reports metrics over the run's resolved class space.

    ``evaluate`` over a training half whose masks reach class 1 alone reports per-class IoU for
    the three classes the run resolved.
    """
    import torch
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.training.collation import task_collate
    from tcip_mcp.pipelines.training.evaluation import evaluate
    from tests import bespoke_models

    images_dir, masks_dir = _three_class_masks(tmp_path / "ds")
    data = DataSpec.model_validate(
        {
            "images_dir": str(images_dir),
            "labels_dir": str(masks_dir),
            "split": drawn_split(val_ratio=0.25, seed=3),
        }
    )

    train_ds, _val_ds, partition, resolved = _train_val(
        tmp_path, "semantic_seg", data
    )
    held = {
        int(np.array(Image.open(masks_dir / f"{stem}.png")).max())
        for stem in STEMS[1:4]
    }
    assert held == {1}, (
        "the training half reaches only class 1, which is what this measures"
    )

    count = resolved.num_classes
    assert count is not None
    from tests._chain_fixtures import frame_statistics, train_sources

    stats = frame_statistics(train_sources(partition), 3)
    model = bespoke_models.build_bespoke_semantic_seg(
        num_classes=count,
        image_mean=stats["image_mean"],
        image_std=stats["image_std"],
    )
    loader = DataLoader(
        train_ds, batch_size=1, collate_fn=task_collate("semantic_seg")
    )
    result = evaluate(
        model,
        loader,
        torch.device("cpu"),
        "semantic_seg",
        dims={"in_chans": 3, "num_classes": count},
        conf_threshold=None,
        iou_threshold=IOU_THRESHOLD,
        score_weights=None,
        trait=None,
        density=None,
        mask_cut=None,
    )

    assert sorted(result["per_class_iou"]) == [0, 1, 2]


def test_the_place_and_the_record_read_one_ground_truth_shape(
    tmp_path: Path,
) -> None:
    """A configured place and a recorded sample read one ground-truth shape.

    A table, a directory of masks, and no place at all for label documents each read as the
    shape of a sample's recorded ground truth there.
    """
    from tcip_mcp.pipelines.data.label_queries import ground_truth_shape
    from tcip_mcp.pipelines.data.selection import shape_of
    from tests._producer_fixtures import image_label_key

    root = tmp_path / "ds"
    images_dir, masks_dir = _mask_dataset(root)
    _table_images, csv_path = _table_dataset(tmp_path / "table")

    for place, member in (
        (str(csv_path), str(csv_path)),
        (str(masks_dir), str(masks_dir / f"{STEMS[0]}.png")),
        (None, image_label_key(images_dir / f"{STEMS[0]}.png")),
    ):
        assert ground_truth_shape(place) == shape_of(member, None), place


def test_a_mask_run_refuses_a_stated_class_space_and_admits_the_empty_one(
    tmp_path: Path,
) -> None:
    """A mask run refuses a stated class space and admits the empty one.

    A mask raster carries its own classes, so a scope naming a subject over it refuses by name;
    the explicit empty scope admits and is what the run records.
    """
    images_dir, masks_dir = _mask_dataset(tmp_path / "ds")

    def data(scope: dict) -> DataSpec:
        return DataSpec.model_validate(
            {
                "images_dir": str(images_dir),
                "labels_dir": str(masks_dir),
                "scope": scope,
                "split": drawn_split(val_ratio=0.25, seed=5),
            }
        )

    with pytest.raises(ValueError, match="carries its own classes"):
        _train_val(tmp_path, "semantic_seg", data({"subject": "leaf"}))

    *_loaders, resolved = _train_val(tmp_path, "semantic_seg", data({}))
    assert resolved.scope == ClassScope()


# -- two producers of one record, compared against each other ------------------


def test_the_bound_and_drawn_mask_routes_record_one_directory_the_same_way(
    tmp_path: Path,
) -> None:
    """The bound and drawn mask routes record one directory's members the same way.

    One mask directory is admitted by a run bound to a selection drawn over it and by a run
    drawing its own split. The bound run's training and validation members are a subset of
    the drawn run's, each carrying the same ground truth, digest and source in both, under the
    mask directory.
    """
    root = tmp_path / "ds"
    images_dir, masks_dir = _mask_dataset(root)
    out = tmp_path / "m"
    _drawn(root, masks_dir, out)

    def trained(experiment_id: str, data_cfg: dict) -> dict:
        """Return the recorded training and validation members with their recorded facts.

        Each member carries the ground truth, its digest at the run and the source the record
        names for it.
        """
        partition = _resolved(
            tmp_path, experiment_id, "semantic_seg", data_cfg
        )[0].partition
        return {
            s.member: (s.ground_truth, s.ground_truth_digest, s.source)
            for s in partition.samples
            if s.side in ("train", "val")
        }

    bound = trained("exp-mask-bound", {"split": {"selection_dir": str(out)}})
    drawn = trained(
        "exp-mask-drawn",
        {
            "images_dir": str(images_dir),
            "labels_dir": str(masks_dir),
            "split": drawn_split(
                val_ratio=0.25, seed=7, grouping="tile_prefix"
            ),
        },
    )

    assert {
        Path(truth).parent
        for truth, _digest, _source in (*bound.values(), *drawn.values())
    } == {masks_dir}
    # The bound run holds out a calibration side the drawn one trains on, so the two agree on
    # every member they both name rather than on the union.
    assert bound and drawn and set(bound) <= set(drawn)
    assert {member: bound[member] for member in bound} == {
        member: drawn[member] for member in bound
    }


def test_a_document_run_and_a_mask_run_record_the_same_images_the_same_way(
    tmp_path: Path,
) -> None:
    """A document run and a mask run record the same images the same way.

    One set of images is resolved under a per-image label document each and under a
    ``<stem>.png`` mask each, at one seed. The two partitions carry the same keys, sides, groups
    and sources, and each records the ground truth and source its loaders served.
    """
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import Subject, SubjectRegistry
    from tests._producer_fixtures import (
        image_label_key,
        label_image,
        registry_over,
    )

    root = tmp_path / "ds"
    images_dir, masks_dir = _mask_dataset(root)
    registry_over(
        root, SubjectRegistry(subjects=(Subject(name="leaf", description=""),))
    )
    for stem in STEMS:
        label_image(
            images_dir / f"{stem}.png",
            [Annotation(subject="leaf", geometry=BBox(2, 2, 8, 8))],
            16,
            16,
        )

    split = drawn_split(val_ratio=0.25, seed=11)

    def resolved(
        experiment_id: str, task: str, data_cfg: dict
    ) -> tuple[Partition, dict, dict]:
        """Return the resolved partition and each member's ground truth and source as served."""
        record, train_ds, val_ds = _resolved(
            tmp_path, experiment_id, task, data_cfg
        )
        served = [
            ds.sample_of(key) for ds in (train_ds, val_ds) for key in ds.stems
        ]
        served_truth = {s.member: s.ground_truth for s in served}
        served_sources = {s.member: s.source for s in served}
        return record.partition, served_truth, served_sources

    document_run, document_served, document_sources = resolved(
        "exp-shape-document",
        "detection",
        {
            "images_dir": str(images_dir),
            "scope": {"subject": "leaf"},
            "split": dict(split),
        },
    )
    mask_run, mask_served, mask_sources = resolved(
        "exp-shape-mask",
        "semantic_seg",
        {
            "images_dir": str(images_dir),
            "labels_dir": str(masks_dir),
            "split": dict(split),
        },
    )

    assert set(document_run.model_dump()) == set(mask_run.model_dump()), (
        "one record shape, whatever the ground truth is"
    )
    assert document_run.group_by == mask_run.group_by == "stem"

    def by_member(partition: Partition) -> dict:
        return {s.member: s for s in partition.samples}

    document_samples, mask_samples = (
        by_member(document_run),
        by_member(mask_run),
    )
    assert {m: s.ground_truth for m, s in document_samples.items()} == {
        m: image_label_key(images_dir / f"{m}.png") for m in STEMS
    }
    assert {Path(s.ground_truth).parent for s in mask_samples.values()} == {
        masks_dir
    }
    assert {m: (s.side, s.group) for m, s in document_samples.items()} == {
        m: (s.side, s.group) for m, s in mask_samples.items()
    }

    # One assertion over both shapes: each path the record names for a member is the path the
    # loader that trained on it read, so a record naming any other file fails here.
    for samples, served, sources in (
        (document_samples, document_served, document_sources),
        (mask_samples, mask_served, mask_sources),
    ):
        assert {m: s.ground_truth for m, s in samples.items()} == served
        assert {m: s.source for m, s in samples.items()} == sources

    # Two runs over one set of images name the same pixels per member: a source wrong the same
    # way in a producer and in the record it writes agrees with itself, and fails only here.
    assert document_sources == mask_sources
