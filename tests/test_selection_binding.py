"""A run's and a sweep's binding to a named selection (``data.split.selection_dir``).

Covers ``auto_train_val``'s branch that binds a run to one, the redraw inside a bound selection,
the refusals a launch raises, and what the run's own partition record carries.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from torch import Tensor

    from tcip_mcp.pipelines.data.selection import Sample, Selection
    from tcip_mcp.pipelines.data.split_construction import Partition

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.dataset_layout import image_dir, parse_image_path
from tcip_mcp.pipelines.data.datasets import BaseDataset
from tcip_mcp.pipelines.data.selection import ClassScope, read_selection
from tcip_mcp.subject_registry import Attribute, Subject, SubjectRegistry
from tcip_mcp.tools.data_tools import draw_splits
from tests._chain_fixtures import (
    BLOB_BUILDER,
    DETECTION_SOURCE,
    run_config,
    stated_detector_geometry,
    training_config,
    weights_statistics,
)
from tests._producer_fixtures import (
    attribute_scoped_dataset,
    label_image,
    labeled_frame,
    registry_of,
    registry_over,
    scope_over,
    train_val as _train_val,
    write_image,
)
from tests._training_values import RATIONALE, drawn_split
from tests._verified_checkpoint_fixtures import partition_side

SUBJECT = "leaf"
OTHER_SUBJECT = "bud"
DATES = ("2-11-26", "2-12-01")


def _write_stem(
    images_dir: Path, stem: str, annotations: list[Annotation]
) -> Path:
    """Write the 64 px labeled frame ``<stem>.jpg`` into ``images_dir``; return its path."""
    return labeled_frame(
        images_dir / f"{stem}.jpg",
        annotations,
        64,
        64,
        (100, 120, 90),
        keep_empty=True,
    )


def _two_subject_two_date_dataset(root: Path) -> Path:
    """Write a dataset of two capture dates, six stems each, over two subjects, and return it.

    In each date, stems ``a`` to ``f`` are written with a ``leaf`` box, and ``c`` to ``f`` also
    with a box of the unrelated ``bud``.
    """
    registry_over(
        root,
        SubjectRegistry(
            subjects=(
                Subject(name=SUBJECT, description=""),
                Subject(name=OTHER_SUBJECT, description=""),
            )
        ),
    )
    for date in DATES:
        images_dir = image_dir(root, date)
        for stem in ("a", "b"):
            _write_stem(
                images_dir,
                stem,
                [Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20))],
            )
        for stem in ("c", "d", "e", "f"):
            _write_stem(
                images_dir,
                stem,
                [
                    Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20)),
                    Annotation(
                        subject=OTHER_SUBJECT, geometry=BBox(30, 30, 44, 44)
                    ),
                ],
            )
    return root


def test_every_sample_groups_each_member_the_way_the_stem_policy_records_it(
    tmp_path: Path,
) -> None:
    """Every sample groups each member the way the ``stem`` policy records it.

    On a dated tree the ``stem`` policy's group key names the capture date, so each member's
    group is the key ``recorded_group_key_fn`` gives it, never its bare member name.
    """
    from tcip_mcp.pipelines.data.splits import recorded_group_key_fn
    from tests._producer_fixtures import admit_over

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    admitted = admit_over(image_dir(root, DATES[0]), subject=SUBJECT)
    key = recorded_group_key_fn("stem", date=admitted.date)
    samples = admitted.every_sample()

    assert admitted.date == DATES[0] and samples
    assert {s.member: s.group for s in samples} == {
        s.member: key(s.member) for s in samples
    }
    assert all(s.group != s.member for s in samples)


def _dataset_with_a_confirmed_negative(root: Path) -> Path:
    """Write a dataset of four annotated stems and one empty one, and return it.

    The first date holds stems ``a`` to ``d``, each with one box of the subject, and ``n``,
    whose label document is empty.
    """
    registry_over(root, registry_of(SUBJECT))
    images_dir = image_dir(root, DATES[0])
    for stem in ("a", "b", "c", "d"):
        _write_stem(
            images_dir,
            stem,
            [Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20))],
        )
    _write_stem(images_dir, "n", [])
    return root


def _tiled_dataset(root: Path) -> Path:
    """Write a dataset of four parents, three crops each, and return it.

    The crops of parents ``srcA`` to ``srcD`` are named ``<parent>_<x>_0`` for x of 0, 1 and 2,
    each with one box of the subject, in the first date.
    """
    registry_over(root, registry_of(SUBJECT))
    images_dir = image_dir(root, DATES[0])
    for parent in ("srcA", "srcB", "srcC", "srcD"):
        for x in range(3):
            _write_stem(
                images_dir,
                f"{parent}_{x}_0",
                [Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20))],
            )
    return root


class _RecordingDataset(BaseDataset):
    """A bespoke dataset standing in for an agent's own builder.

    It keeps the samples, class space and data locations it is constructed with as
    ``seen_samples``, ``seen_scope`` and ``seen_directories``; its length is the samples' count,
    and ``__getitem__`` returns ``torch.zeros(3, 8, 8)`` with a target of box ``(0, 0, 1, 1)``,
    label 1 and ``iscrowd`` 0 for every index.
    """

    def __init__(
        self,
        samples: list[Sample],
        scope: ClassScope | None,
        directories: dict[str, object],
    ) -> None:
        self.seen_samples = list(samples)
        self.seen_scope = scope
        self.seen_directories = directories

    @property
    def num_samples(self) -> int:
        return len(self.seen_samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Tensor]]:
        return torch.zeros(3, 8, 8), {
            "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            "labels": torch.tensor([1]),
            "iscrowd": torch.tensor([0]),
        }

    @property
    def regions(self) -> list:
        """The regions ``served_regions`` reads off each item this dataset serves."""
        from tests._producer_fixtures import served_regions

        return served_regions(self)


def build_recording_dataset(
    samples: list[Sample] | None = None,
    scope: ClassScope | None = None,
    **kwargs: object,
) -> _RecordingDataset:
    """Build a recording dataset, the ``dataset_source`` builder registered by dotted name.

    It records whichever of ``images_dir``, ``labels_dir``, ``csv_path``, ``coco_data`` and
    ``stems`` it was handed.
    """
    located = {
        k: v
        for k, v in kwargs.items()
        if k in ("images_dir", "labels_dir", "csv_path", "coco_data", "stems")
    }
    return _RecordingDataset(samples or [], scope, located)


RECORDING_DATASET = f"{Path(__file__).stem}:build_recording_dataset"
"""The ``dataset_source`` builder of :func:`build_recording_dataset`."""


def _draw(
    project: Path,
    root: Path,
    out: Path,
    *,
    subject: str = SUBJECT,
    seed: int = 2,
) -> Selection:
    result = draw_splits(
        project,
        str(root),
        output_path=str(out),
        subject=subject,
        **drawn_split(
            seed=seed,
            val_ratio=0.25,
            calibration_ratio=0.125,
            holdout_ratio=0.125,
            grouping="tile_prefix",
        ),
        actor=None,
    )
    assert "error" not in result, result
    return read_selection(out, project=project)


def _run_data_cfg(
    root: Path, selection_dir: Path, **overrides: object
) -> dict:
    data_cfg: dict = {"split": {"selection_dir": str(selection_dir)}}
    data_cfg.update(overrides)
    return data_cfg


# -- binding a run to a selection ----------------------------------------------


def test_auto_train_val_binds_the_selections_own_partition(
    tmp_path: Path,
) -> None:
    """``auto_train_val`` binds the loaders to the selection's own recorded partition.

    The loaders hold exactly the samples the selection recorded on each side, both capture
    dates reach the sources the loaders read, and no held-out sample reaches either loader.
    """
    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(tmp_path, root, out)
    data_cfg = _run_data_cfg(root, out)

    train_ds, val_ds, partition, resolved = _train_val(
        tmp_path, "detection", data_cfg
    )

    assert sorted(train_ds.stems) == sorted(
        s.location for s in drawn.on("train")
    )
    assert sorted(val_ds.stems) == sorted(s.location for s in drawn.on("val"))
    held_out = {
        s.location for s in drawn.on("calibration") + drawn.on("holdout")
    }
    assert held_out and not held_out & set(train_ds.stems + val_ds.stems)
    # The sources the two loaders will open, not a record written beside them.
    read_from = {
        parse_image_path(ds.sample_of(key).source)[1]
        for ds in (train_ds, val_ds)
        for key in ds.stems
    }
    assert read_from == set(DATES)

    binding = partition.selection
    assert binding is not None
    assert binding.selection_dir == str(out)
    assert binding.redraw is False
    assert "date" not in binding.model_dump()
    # The class space is the run's own, recorded once on the data block, never restated here.
    assert resolved.scope == drawn.scope
    assert drawn.scope.subject == SUBJECT
    assert "scope" not in binding.model_dump()
    assert {s.ground_truth.parts[0] for s in partition.samples} == {
        s.ground_truth.parts[0] for s in drawn.samples
    }


def _pixels_and_ground_truth(under: Path) -> dict[Path, int]:
    """Return each file under ``under`` outside a ``.tcip`` tree, mapped to its size in bytes.

    The files kept are those suffixed ``.jpg``, ``.png``, ``.tif``, ``.tiff``, ``.npy``,
    ``.json`` or ``.bandgroup``, in any case.
    """
    suffixes = {".jpg", ".png", ".tif", ".tiff", ".npy", ".json", ".bandgroup"}
    return {
        p: p.stat().st_size
        for p in under.rglob("*")
        if p.is_file()
        and p.suffix.lower() in suffixes
        and ".tcip" not in p.parts
    }


def test_a_multi_date_selection_trains_without_copying_anything(
    tmp_path: Path,
) -> None:
    """A selection spanning two capture dates trains with its selected file paths and sizes kept.

    One selection spanning two capture dates trains through the worker: the run's resolved
    partition names members of both captures, the checkpoint carries the selection's class map,
    and the image and document paths outside a ``.tcip`` state tree, with their sizes, are as
    they were before the run.
    """
    from tcip_mcp.experiments import run_resolution
    from tests._verified_checkpoint_fixtures import opened_run

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(tmp_path, root, out)

    run_dir = opened_run(
        tmp_path,
        run_config(tmp_path, out, BLOB_BUILDER),
        experiment_id="exp-multi-date",
    )

    from tcip_mcp.pipelines.training.subprocess_worker import run_directory

    before = _pixels_and_ground_truth(tmp_path)
    run_directory(run_dir)

    assert _pixels_and_ground_truth(tmp_path) == before, (
        "a bound run reads the dataset's own imagery and ground truth; nothing is copied"
    )

    # Both dates reach the members the run actually recorded, from two captures.
    samples = run_resolution(run_dir.name, project=tmp_path).partition.samples
    consumed = {s for s in samples if s.side in ("train", "val")}
    assert {s.ground_truth.parts[0] for s in consumed} == set(DATES)
    assert {s.ground_truth.parts[-1] for s in consumed} == {
        s.ground_truth.parts[-1] for s in drawn.on("train") + drawn.on("val")
    }

    # The checkpoint speaks the selection's own vocabulary, not one re-read from the registry.
    checkpoint = torch.load(
        run_dir / "model_final.pt", map_location="cpu", weights_only=False
    )
    stamped = (checkpoint.get("config") or {}).get("data") or {}
    assert ClassScope.of(stamped) == drawn.scope
    assert drawn.scope.subject == SUBJECT


def test_a_bound_run_keeps_its_selections_attributes_when_the_registry_is_reordered(
    tmp_path: Path,
) -> None:
    """A bound run keeps its selection's attributes when the registry is reordered.

    The registry's declared value order is reversed between the draw and the run, and the
    checkpoint stamps the scope the selection recorded at the draw.
    """
    from tcip_mcp.pipelines.training.subprocess_worker import run_directory
    from tests._verified_checkpoint_fixtures import opened_run

    root = attribute_scoped_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(tmp_path, root, out, seed=1)
    assert [a.values for a in drawn.scope.attributes] == [
        ("healthy", "damaged")
    ]

    # The same attribute, its values declared the other way round: an order re-read here would
    # give each value the other's id, a different class space than the samples were admitted in.
    registry_over(
        root,
        registry_of(
            SUBJECT,
            Attribute(
                name="condition",
                type="categorical",
                values=("damaged", "healthy"),
            ),
        ),
    )

    # The directory a relaunched config still carries beside its binding: exactly what a
    # registry re-read would resolve the wrong map from.
    config = training_config(
        tmp_path,
        BLOB_BUILDER,
        {
            "images_dir": str(image_dir(root, DATES[0])),
            "split": {"selection_dir": str(out)},
        },
    )
    run_dir = opened_run(tmp_path, config, experiment_id="exp-reordered")

    run_directory(run_dir)

    checkpoint = torch.load(
        run_dir / "model_final.pt", map_location="cpu", weights_only=False
    )
    stamped = (checkpoint.get("config") or {}).get("data") or {}
    assert ClassScope.of(stamped) == drawn.scope


def test_a_scope_stated_beside_a_selection_refuses_the_bound_run(
    tmp_path: Path,
) -> None:
    """A scope stated beside a selection refuses the bound run by name.

    The selection records the class space its samples were admitted under; a data section
    stating one beside it refuses by name, the selection's own included, and one stating none
    resolves its loaders in the selection's.
    """
    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(tmp_path, root, out)

    for stated in ({"subject": OTHER_SUBJECT}, asdict(drawn.scope)):
        with pytest.raises(ValueError, match="Drop data.scope"):
            _train_val(
                tmp_path, "detection", _run_data_cfg(root, out, scope=stated)
            )

    train_ds, val_ds, _partition, resolved = _train_val(
        tmp_path, "detection", _run_data_cfg(root, out)
    )
    assert resolved.scope == drawn.scope
    assert train_ds.scope == drawn.scope and val_ds.scope == drawn.scope


def test_a_ground_truth_place_stated_beside_a_selection_refuses_the_bound_run(
    tmp_path: Path,
) -> None:
    """A ``labels_dir`` stated beside a selection refuses the bound run by name.

    A data section naming ``labels_dir`` beside the selection refuses by name, an existing
    directory or not, at the run and at the preflight.
    """
    from tcip_mcp.tools.training_tools import preflight_config

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(tmp_path, root, out)
    (tmp_path / "empty").mkdir()

    for place in (tmp_path / "empty", tmp_path / "gone"):
        data_cfg = _run_data_cfg(root, out, labels_dir=str(place))
        with pytest.raises(ValueError, match="Drop data.labels_dir"):
            _train_val(tmp_path, "detection", data_cfg)
        issues = preflight_config(
            tmp_path,
            training_config(
                tmp_path, weights_statistics(DETECTION_SOURCE), data_cfg
            ),
        )["issues"]
        assert any("Drop data.labels_dir" in issue for issue in issues), issues


def test_a_selected_label_emptied_since_the_draw_refuses_the_run(
    tmp_path: Path,
) -> None:
    """A selected label emptied since the draw with no confirmed negative refuses the run.

    A label that held the subject at draw time is emptied with nobody confirming that image
    negative, and the bind refuses, naming it no longer admissible.
    """
    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(tmp_path, root, out)
    emptied = drawn.on("train")[0]
    json_io.write_label_document(
        emptied.ground_truth, [], 64, 64, keep_empty=True
    )

    with pytest.raises(ValueError, match="no longer admissible"):
        _train_val(tmp_path, "detection", _run_data_cfg(root, out))


def test_a_selected_label_a_human_confirmed_negative_still_trains(
    tmp_path: Path,
) -> None:
    """A selected label emptied since the draw binds after a human confirms it negative.

    An emptied label whose image a human marked negative for this subject is a real negative,
    and the run binds.
    """
    from tests._producer_fixtures import mark_complete

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(tmp_path, root, out)
    emptied = drawn.on("train")[0]
    json_io.write_label_document(
        emptied.ground_truth, [], 64, 64, keep_empty=True
    )
    mark_complete(emptied.source, SUBJECT, project=root)

    train_ds, val_ds, partition, _data = _train_val(
        tmp_path, "detection", _run_data_cfg(root, out)
    )

    assert len(train_ds) == len(drawn.on("train"))
    assert len(val_ds) == len(drawn.on("val"))
    assert emptied.member in set(partition_side(partition, "train"))


def test_a_bound_run_admits_when_an_unselected_images_stem_turns_ambiguous(
    tmp_path: Path,
) -> None:
    """A bound run admits when an unselected image's stem turns ambiguous.

    An image nobody selected picks up a second file under its stem after the draw, and the bound
    run's loaders hold as many samples as the selection recorded on each side.
    """
    root = _two_subject_two_date_dataset(tmp_path / "ds")
    images_dir = image_dir(root, DATES[0])
    write_image(images_dir / "spare.jpg", (64, 64), (10, 10, 10))
    out = tmp_path / "m"
    drawn = _draw(tmp_path, root, out)
    assert "spare" not in {Path(s.source).stem for s in drawn.samples}

    # The same logical image now resolves to two files: the directory can no longer be listed.
    write_image(images_dir / "spare.png", (64, 64), (20, 20, 20))

    train_ds, val_ds, _partition, _data = _train_val(
        tmp_path, "detection", _run_data_cfg(root, out)
    )

    assert len(train_ds) == len(drawn.on("train"))
    assert len(val_ds) == len(drawn.on("val"))


def test_a_sample_naming_a_row_of_its_ground_truth_refuses_the_geometry_loaders(
    tmp_path: Path,
) -> None:
    """A sample naming a row of its ground truth is refused by the geometry loaders.

    ``row_key`` names one row inside a ground truth that answers for many samples. A detection
    loader over a sample naming a row refuses, naming the ground truth a detection loader does
    not read, while the same image's own document builds. The selection comes back
    through ``read_selection``, the platform's own reader.
    """
    from tcip_mcp.pipelines.data.datasets import build_dataset, resolve_sizes
    from tcip_mcp.pipelines.data.selection import (
        ClassScope,
        Sample,
        Selection,
        read_selection,
        write_selection,
    )
    from tests._producer_fixtures import image_label_key

    root = tmp_path / "ds"
    images_dir = image_dir(root, DATES[0])
    image = write_image(images_dir / "a.jpg", (64, 64), (100, 120, 90))
    scope = scope_over(root, registry_of(SUBJECT), SUBJECT)
    label_image(
        image,
        [Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20))],
        64,
        64,
        keep_empty=True,
    )
    table = root / "values.csv"
    table.write_text("image,value\na.jpg,1\n", encoding="utf-8")

    def _selection(row_key: str | None) -> Selection:
        out = tmp_path / ("rows" if row_key else "whole")
        write_selection(
            out,
            Selection(
                samples=(
                    Sample(
                        member=row_key or "a",
                        source=str(image),
                        ground_truth=str(table)
                        if row_key
                        else image_label_key(image),
                        group="g",
                        side="train",
                        row_key=row_key,
                    ),
                ),
                scope=ClassScope() if row_key else scope,
                seed=0,
                group_by="stem",
                ratios={"train": 1.0},
                stratify_foreground=True,
                rationale=RATIONALE,
            ),
            project=tmp_path,
        )
        return read_selection(out, project=tmp_path)

    with pytest.raises(ValueError, match="a detection loader does not read"):
        resolve_sizes("detection", {}, _selection("a.jpg").samples)

    whole = _selection(None).samples
    admitted = build_dataset(
        "detection",
        samples=whole,
        scope=scope,
        sizes=resolve_sizes("detection", {}, whole),
    )
    assert list(admitted.stems) == [str(image)]


def test_crops_of_one_parent_cannot_cross_sides(tmp_path: Path) -> None:
    """Crops of one parent image never land on two sides of a bound run.

    The draw places every crop of one parent on one side, the bound run's train and val loaders
    share no parent, and the reader refuses a selection whose group keys straddle two sides.
    """
    from tcip_mcp.pipelines.data.selection import (
        as_selection,
        stored_selection_document,
    )

    root = _tiled_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(tmp_path, root, out, seed=1)

    side_of_parent: dict[str, str] = {}
    for sample in drawn.samples:
        parent = sample.member.rsplit("_", 2)[0]
        assert side_of_parent.setdefault(parent, sample.side) == sample.side
    assert len(side_of_parent) == 4

    train_ds, val_ds, _, _data = _train_val(
        tmp_path, "detection", _run_data_cfg(root, out)
    )
    parents_on = {
        side: {Path(stem).stem.rsplit("_", 2)[0] for stem in ds.stems}
        for side, ds in (("train", train_ds), ("val", val_ds))
    }
    assert not parents_on["train"] & parents_on["val"]

    # And the record itself refuses a hand-made partition that splits one parent's crops.
    document = stored_selection_document(drawn, tmp_path)
    assert isinstance(document, dict) and isinstance(document["samples"], list)
    samples = [s for s in document["samples"] if isinstance(s, dict)]
    a_train = next(s for s in samples if s["side"] == "train")
    moved = dict(a_train, side="val")
    document["samples"] = [s for s in samples if s is not a_train] + [moved]
    with pytest.raises(ValueError, match="group"):
        as_selection(document, where="hand-made")


def test_auto_train_val_admits_a_confirmed_negative_the_draw_admitted(
    tmp_path: Path,
) -> None:
    """A bound run's loader holds a confirmed negative the draw admitted, as a zero-object sample.

    A stem whose label document is empty and whose image a human marked negative is admitted
    at the draw, and the bound run's loader holding it serves it with no boxes.
    """
    from tests._producer_fixtures import mark_complete

    root = _dataset_with_a_confirmed_negative(tmp_path / "ds")
    mark_complete(image_dir(root, DATES[0]) / "n.jpg", SUBJECT, project=root)
    out = tmp_path / "m"
    drawn = _draw(tmp_path, root, out, seed=1)
    assert "n" in {s.member for s in drawn.samples}

    train_ds, val_ds, _, _data = _train_val(
        tmp_path, "detection", _run_data_cfg(root, out)
    )

    negatives = [
        s for s in drawn.on("train") + drawn.on("val") if s.member == "n"
    ]
    assert negatives, "the negative landed on neither loader's side"
    loader = train_ds if negatives[0].location in train_ds.stems else val_ds
    _image, target = loader[loader.stems.index(negatives[0].location)]
    assert target["boxes"].shape[0] == 0


def test_an_unconfirmed_empty_label_never_reaches_a_bound_run(
    tmp_path: Path,
) -> None:
    """An empty label document nobody confirmed appears in no selection.

    The draw admits no empty label document nobody confirmed, so none is a selection's sample.
    """
    root = _dataset_with_a_confirmed_negative(tmp_path / "ds")
    out = tmp_path / "m"

    drawn = _draw(tmp_path, root, out, seed=1)

    assert "n" not in {s.member for s in drawn.samples}


def test_a_bound_run_reads_attribute_ids_from_the_selections_own_scope(
    tmp_path: Path,
) -> None:
    """A bound run's loaders read the attributes the selection's own scope records.

    The selection records the attributes its admission read, the bound train loader carries that
    scope, and the stem not assessed for the attribute stays among the selection's samples.
    """
    root = attribute_scoped_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(tmp_path, root, out)
    assert [a.values for a in drawn.scope.attributes] == [
        ("healthy", "damaged")
    ]
    assert "unassessed" in {s.member for s in drawn.samples}

    train_ds, _val_ds, _, resolved = _train_val(
        tmp_path, "detection", _run_data_cfg(root, out)
    )

    assert train_ds.scope == drawn.scope
    assert resolved.num_classes is None  # the scope's subject is the one class
    assert drawn.scope.subject == SUBJECT


def test_a_bound_run_threads_a_bespoke_dataset_source(tmp_path: Path) -> None:
    """A selection-bound run threads its samples and scope to a bespoke dataset source.

    The bespoke seam stays reachable from a selection-bound run: the agent's own builder
    receives the selection's samples and the scope it was drawn under.
    """
    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(tmp_path, root, out)
    data_cfg = _run_data_cfg(root, out)
    data_cfg["dataset_source"] = {
        "builder": RECORDING_DATASET,
        "source_files": [__file__],
    }

    train_ds, _val_ds, _, _data = _train_val(tmp_path, "detection", data_cfg)

    assert sorted(s.location for s in train_ds.seen_samples) == sorted(
        s.location for s in drawn.on("train")
    )
    assert train_ds.seen_scope == drawn.scope


@pytest.mark.parametrize("task", ["detection", "canopy_extent"])
def test_an_unbound_bespoke_run_is_handed_the_same_samples_a_bound_one_is(
    tmp_path: Path, task: str
) -> None:
    """A bound and an unbound bespoke run agree on the samples of the date they share.

    The same tree is resolved twice through a bespoke builder, bound to a selection drawn over it
    and unbound over its first date's directory, for a task with a built-in loader and for one
    without; the builder records what it was handed each time. Each route draws both sides, and
    the bound route's first-date samples hold no repeat and equal the unbound route's samples
    outside the selection's calibration and holdout locations, compared by source, ground truth,
    row key, rect, group and member; both train builders hold the selection's class space, and
    no builder is handed a directory.
    """
    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(tmp_path, root, out)
    source = {"builder": RECORDING_DATASET, "source_files": [__file__]}

    bound_cfg = _run_data_cfg(root, out)
    bound_cfg["dataset_source"] = source
    bound_train, bound_val, _partition, _bound = _train_val(
        tmp_path, task, bound_cfg
    )

    unbound_cfg = {
        "images_dir": str(image_dir(root, DATES[0])),
        "scope": {"subject": SUBJECT},
        "dataset_source": source,
        "split": drawn_split(val_ratio=0.5, seed=3, grouping="tile_prefix"),
    }
    unbound_train, unbound_val, _unbound_partition, _unbound = _train_val(
        tmp_path, task, unbound_cfg
    )

    assert bound_val is not None and unbound_val is not None, (
        "each route draws both sides over the samples its producer named"
    )

    def handed(*datasets: _RecordingDataset) -> list[tuple]:
        """Return the samples the builders were handed, with multiplicity, sorted.

        Each sample is the tuple of its source, ground truth, row key, rect, group and member.
        """
        seen: list[tuple] = []
        for dataset in datasets:
            for sample in dataset.seen_samples:
                seen.append(
                    (
                        sample.source,
                        sample.ground_truth,
                        sample.row_key,
                        sample.rect,
                        sample.group,
                        sample.member,
                    )
                )
        return sorted(seen)

    # Comparable universes: the unbound run admits one date, so the bound run's members from the
    # other one are not members the unbound run could have held.
    bound_here = [
        s
        for s in handed(bound_train, bound_val)
        if parse_image_path(s[0])[1] == DATES[0]
    ]
    unbound_here = handed(unbound_train, unbound_val)
    held_out = {
        s.location for s in drawn.on("calibration") + drawn.on("holdout")
    }
    unbound_trained = [s for s in unbound_here if s[0] not in held_out]

    assert bound_here and unbound_trained
    assert len(bound_here) == len(set(bound_here)), (
        "each sample is handed once, not repeated"
    )
    assert bound_here == unbound_trained
    assert unbound_train.seen_scope == bound_train.seen_scope == drawn.scope
    # Nothing the builder can go looking in: a directory would let it read its own membership.
    for dataset in (bound_train, bound_val, unbound_train, unbound_val):
        assert dataset.seen_directories == {}


def test_the_preflight_smoke_batch_is_the_batch_the_bound_run_trains(
    tmp_path: Path,
) -> None:
    """The batch a preflight smokes comes off the bound run's training loader.

    The smoked loader, resolved as the launch resolves it, holds the selection's train side and
    no validation or calibration member, one real batch comes off it, and the ``data.split`` of
    the config handed in is still ``{"selection_dir": str(out)}``.
    """
    from tcip_mcp.pipelines.data.split_construction import resolve_run
    from tcip_mcp.pipelines.model_build import staged_sources
    from tcip_mcp.pipelines.schemas import train_config
    from tcip_mcp.tools.training_tools import _one_real_batch

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(tmp_path, root, out)
    config = training_config(
        tmp_path,
        DETECTION_SOURCE,
        {
            **_run_data_cfg(root, out),
            "dataset_source": {
                "builder": RECORDING_DATASET,
                "source_files": [__file__],
            },
        },
    )
    config["model_source"]["detector_geometry"] = stated_detector_geometry()
    spec = train_config(config)

    smoked = resolve_run(
        spec, staged_sources(spec, tmp_path).layout, project=tmp_path
    ).train_ds
    batch = _one_real_batch("detection", smoked)

    assert not isinstance(batch, str), batch
    assert sorted(s.location for s in smoked.seen_samples) == sorted(
        s.location for s in drawn.on("train")
    )
    held_out = {s.location for s in drawn.on("val") + drawn.on("calibration")}
    assert not held_out & {s.location for s in smoked.seen_samples}
    # The caller's own config is left exactly as it was found: resolution reads a copy.
    assert config["data"]["split"] == {"selection_dir": str(out)}


def test_auto_train_val_refuses_a_selection_with_an_empty_side(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.data.selection import Selection, write_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(tmp_path, root, out)
    write_selection(
        out,
        Selection(
            samples=tuple(s for s in drawn.samples if s.side != "val"),
            scope=drawn.scope,
            seed=drawn.seed,
            group_by=drawn.group_by,
            ratios=drawn.ratios,
            stratify_foreground=drawn.stratify_foreground,
            rationale=drawn.rationale,
        ),
        project=tmp_path,
    )

    with pytest.raises(ValueError, match="empty side"):
        _train_val(tmp_path, "detection", _run_data_cfg(root, out))


def _strip_the_recorded_subject(out: Path) -> None:
    """Rewrite a drawn selection's record with no subject, past the writer."""
    import tcip_store as ts
    from tcip_mcp.pipelines.data.selection import selection_key

    document = ts.read(selection_key(out))
    assert isinstance(document, dict)
    scope = document["scope"]
    assert isinstance(scope, dict)
    ts.replace(
        selection_key(out), {**document, "scope": {**scope, "subject": None}}
    )


def test_the_selection_writer_refuses_a_document_selection_with_no_subject(
    tmp_path: Path,
) -> None:
    import dataclasses

    from tcip_mcp.pipelines.data.selection import write_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    drawn = _draw(tmp_path, root, tmp_path / "m")
    unscoped = dataclasses.replace(
        drawn, scope=dataclasses.replace(drawn.scope, subject=None)
    )

    with pytest.raises(ValueError, match="no subject"):
        write_selection(tmp_path / "unscoped", unscoped, project=tmp_path)
    assert not (tmp_path / "unscoped").exists()


def test_auto_train_val_refuses_a_selection_with_no_subject(
    tmp_path: Path,
) -> None:
    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(tmp_path, root, out)
    _strip_the_recorded_subject(out)

    with pytest.raises(ValueError, match="no subject"):
        _train_val(tmp_path, "detection", _run_data_cfg(root, out))


def test_a_split_naming_a_selection_refuses_a_drawn_splits_own_parameter_by_name(
    tmp_path: Path,
) -> None:
    """A split naming a selection refuses a drawn split's own parameter by name.

    A split naming a selection is a bound split, which carries no share: the schema refuses
    one stated beside it by name.
    """
    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(tmp_path, root, out)
    data_cfg = _run_data_cfg(root, out)
    data_cfg["split"]["val_ratio"] = 0.3

    with pytest.raises(ValueError, match="val_ratio"):
        _train_val(tmp_path, "detection", data_cfg)


def test_auto_train_val_selection_refuses_a_task_reading_another_ground_truth(
    tmp_path: Path,
) -> None:
    """A selection refuses a task whose loader reads another ground truth.

    A selection of per-image label documents refuses a semantic segmentation run, naming them as
    ground truth a semantic segmentation loader does not read.
    """
    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(tmp_path, root, out)

    with pytest.raises(
        ValueError, match="a semantic_seg loader does not read"
    ):
        _train_val(tmp_path, "semantic_seg", _run_data_cfg(root, out))


def test_auto_train_val_binding_failure_raises_rather_than_degrading(
    tmp_path: Path,
) -> None:
    """A bind whose sample's image moved since the draw raises, naming it no longer admissible."""
    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(tmp_path, root, out)
    gone = drawn.on("train")[0]
    Path(gone.source).unlink()

    with pytest.raises(ValueError, match="no longer admissible"):
        _train_val(tmp_path, "detection", _run_data_cfg(root, out))


# -- a redraw inside the selection, at a stated seed ---------------------------


def _redraw_cfg(root: Path, out: Path, seed: int) -> dict:
    data_cfg = _run_data_cfg(root, out)
    data_cfg["split"]["seed"] = seed
    return data_cfg


def test_a_redraw_repartitions_the_selections_own_members_and_leaves_calibration_alone(
    tmp_path: Path,
) -> None:
    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(tmp_path, root, out)
    pool = {s.location for s in drawn.on("train") + drawn.on("val")}
    held_out = {s.location for s in drawn.on("calibration")}

    train_ds, val_ds, partition, _data = _train_val(
        tmp_path, "detection", _redraw_cfg(root, out, seed=1)
    )

    assert set(train_ds.stems) | set(val_ds.stems) == pool
    assert not (set(train_ds.stems) | set(val_ds.stems)) & held_out
    assert partition.selection is not None
    assert partition.selection.redraw is True
    assert partition.seed == 1


def test_two_redraw_seeds_differ_and_the_same_seed_repeats(
    tmp_path: Path,
) -> None:
    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(tmp_path, root, out)

    def _train(seed: int) -> list[str]:
        train_ds, _val, _, _data = _train_val(
            tmp_path, "detection", _redraw_cfg(root, out, seed)
        )
        return sorted(train_ds.stems)

    first = _train(1)
    assert _train(1) == first
    assert any(_train(seed) != first for seed in range(2, 8))


def test_a_bound_split_stating_a_seed_redraws_and_one_stating_none_keeps_the_recorded_sides(
    tmp_path: Path,
) -> None:
    """A bound split stating a seed redraws; one stating none keeps the recorded sides.

    Through the child's own context (``run_context``) over a bound selection: a split stating no
    seed builds a training loader holding the selection's recorded train side and records its
    recorded seed, and one stating a seed redraws inside the selection at it, recorded as a
    redraw.
    """
    from tcip_mcp.experiments import read_run
    from tests._chain_fixtures import training_config
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR
    from tests.tiny_trainer_fixtures import run_context

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(tmp_path, root, out)

    def _opened(
        experiment_id: str, **split: int
    ) -> tuple[set[str], Partition]:
        ctx = run_context(
            tmp_path,
            training_config(
                tmp_path,
                BUILT_DETECTOR,
                {"split": {"selection_dir": str(out), **split}},
            ),
            experiment_id=experiment_id,
        )
        partition = read_run(ctx.run_dir).resolution.partition
        return set(ctx.train_loader.dataset.stems), partition

    kept, bound = _opened("exp-bound")
    assert kept == {s.location for s in drawn.on("train")}
    assert bound.selection is not None
    assert (bound.selection.redraw, bound.seed) == (False, drawn.seed)

    redrawn = [
        _opened(f"exp-redrawn-{seed}", seed=seed) for seed in range(3, 8)
    ]
    for seed, (_train, partition) in zip(range(3, 8), redrawn):
        assert partition.selection is not None
        assert (partition.selection.redraw, partition.seed) == (True, seed)
    assert any(train != kept for train, _ in redrawn)


def test_a_redraw_refuses_a_starved_pool_through_the_draws_own_floor(
    tmp_path: Path,
) -> None:
    """A redraw over a starved pool is refused through the draw's own floor.

    A redraw is the one draw over a selection's own train-plus-val members: fewer than two
    foreground groups among them refuses with the draw's own floor message, naming the selection
    and the redraw's remedy. A selection whose members do hold two is admitted.
    """
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.pipelines.data.split_construction import redrawn_selection

    drawn_dir = tmp_path / "drawn"
    _draw(tmp_path, _two_subject_two_date_dataset(tmp_path / "ds"), drawn_dir)
    redrawn = redrawn_selection(
        read_selection(drawn_dir, project=tmp_path), str(drawn_dir), 3
    )
    assert redrawn.counts()["train"] and redrawn.counts()["val"]

    _root, one_group = one_foreground_group_selection(tmp_path / "one")
    with pytest.raises(ValueError) as refused:
        redrawn_selection(
            read_selection(one_group, project=tmp_path / "one"),
            str(one_group),
            3,
        )
    starved = str(refused.value)
    assert f"{str(one_group)!r}" in starved
    assert (
        "1 foreground group(s), fewer than the 2 the requested sides need"
        in starved
    )
    assert "Drop data.split.seed" in starved


def test_the_explicit_draw_the_runs_own_draw_and_the_redraw_agree_member_for_member(
    tmp_path: Path,
) -> None:
    """The explicit draw, a run's own draw and a redraw agree member for member.

    One membership and one draw: ``draw_splits`` over a dated tree, a run drawing its own
    train and val over that date's directories and a redraw of the written selection at the same
    seed place every member on the same side, though the stems sort differently by identity than
    by location.
    """
    from tcip_mcp.pipelines.data.split_construction import redrawn_selection

    root = tmp_path / "ds"
    images_dir = image_dir(root, DATES[0])
    for stem in ("a", "a-", "b", "b-", "c", "c-", "d", "d-"):
        _write_stem(
            images_dir,
            stem,
            [Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20))],
        )
    registry_over(root, registry_of(SUBJECT))
    out = tmp_path / "m"
    explicit = draw_splits(
        tmp_path,
        str(root),
        output_path=str(out),
        subject=SUBJECT,
        **drawn_split(seed=2, val_ratio=0.25, grouping="tile_prefix"),
        actor=None,
    )
    assert explicit["splits"]["val"] == 2, explicit
    selection = read_selection(out, project=tmp_path)

    _train, _val, partition, _data = _train_val(
        tmp_path,
        "detection",
        {
            "images_dir": str(images_dir),
            "scope": {"subject": SUBJECT},
            "split": drawn_split(
                val_ratio=0.25, seed=2, grouping="tile_prefix"
            ),
        },
    )
    redrawn = redrawn_selection(selection, str(out), 2)

    drawn = {s.member: s.side for s in selection.samples}
    assert {s.member: s.side for s in partition.samples} == drawn
    assert {s.member: s.side for s in redrawn.samples} == drawn


def one_foreground_group_selection(tmp_path: Path) -> tuple[Path, Path]:
    """Write a dataset under ``tmp_path / "ds"`` and a selection over it; return both paths.

    The dataset's first date holds ``fg``, ``held_a`` and ``held_b`` with one subject box each
    and ``neg`` with none, marked complete for the subject. The selection, written at
    ``tmp_path / "m"``, places ``fg`` on train, ``neg`` on val and ``held_a`` and ``held_b`` on
    calibration, each its own group. Returns ``(root, selection_dir)``.
    """
    from tcip_mcp.pipelines.data.selection import (
        Sample,
        Selection,
        write_selection,
    )
    from tests._producer_fixtures import image_label_key, mark_complete

    root = tmp_path / "ds"
    scope = scope_over(root, registry_of(SUBJECT), SUBJECT)
    images_dir = image_dir(root, DATES[0])
    frames = {
        stem: _write_stem(
            images_dir,
            stem,
            []
            if stem == "neg"
            else [Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20))],
        )
        for stem in ("fg", "neg", "held_a", "held_b")
    }
    mark_complete(frames["neg"], SUBJECT, project=root)

    def _sample(stem: str, side: str) -> Sample:
        return Sample(
            member=stem,
            source=str(frames[stem]),
            ground_truth=image_label_key(frames[stem]),
            group=stem,
            side=side,
        )

    out = tmp_path / "m"
    write_selection(
        out,
        Selection(
            samples=(
                _sample("fg", "train"),
                _sample("neg", "val"),
                _sample("held_a", "calibration"),
                _sample("held_b", "calibration"),
            ),
            scope=scope,
            seed=1,
            group_by="explicit_map",
            ratios={"train": 0.25, "val": 0.25, "calibration": 0.5},
            stratify_foreground=True,
            rationale=RATIONALE,
        ),
        project=tmp_path,
    )
    return root, out


def test_preflight_flags_a_redraw_whose_members_hold_one_foreground_group(
    tmp_path: Path,
) -> None:
    """Preflight flags a redraw whose members hold one foreground group.

    The selection's val side holds one group, a confirmed negative carrying no foreground, and
    preflight reports the foreground-group shortfall as an issue.
    """
    from tcip_mcp.tools.training_tools import preflight_config

    root, out = one_foreground_group_selection(tmp_path)
    config = training_config(
        tmp_path,
        weights_statistics(DETECTION_SOURCE),
        {"split": {"selection_dir": str(out), "seed": 1}},
    )

    result = preflight_config(tmp_path, config)

    assert any("foreground group" in issue for issue in result["issues"]), (
        result["issues"]
    )


# -- preflight_config's selection-level checks ---------------------------------


def test_preflight_config_admits_a_bound_selection_with_no_issues(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.training_tools import preflight_config

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(tmp_path, root, out)

    result = preflight_config(
        tmp_path,
        training_config(tmp_path, DETECTION_SOURCE, _run_data_cfg(root, out)),
    )

    # A bound config names no images_dir: the selection's samples carry their own sources and
    # ground truth, so the directory checks must not raise an objection against it.
    assert result["issues"] == []


def test_preflight_config_flags_a_selection_with_no_subject(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.training_tools import preflight_config

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(tmp_path, root, out)
    config = training_config(
        tmp_path, DETECTION_SOURCE, _run_data_cfg(root, out)
    )
    _strip_the_recorded_subject(out)

    result = preflight_config(tmp_path, config)

    assert any("records no subject" in i for i in result["issues"])


def test_preflight_config_admits_a_redraw_seed_with_a_warning(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.training_tools import preflight_config

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(tmp_path, root, out)
    config = training_config(
        tmp_path,
        DETECTION_SOURCE,
        {"split": {"selection_dir": str(out), "seed": 1}},
    )

    result = preflight_config(tmp_path, config)

    assert result["issues"] == []
    assert any("this run redraws" in w for w in result["warnings"])


# -- the run's own resolved record ---------------------------------------------


def test_the_resolved_partition_carries_the_selection_it_bound(
    tmp_path: Path,
) -> None:
    from tcip_mcp.experiments import read_run
    from tests._verified_checkpoint_fixtures import resolved_run

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(tmp_path, root, out)

    run_dir = resolved_run(
        tmp_path, _run_data_cfg(root, out), experiment_id="exp-bound"
    )
    partition = read_run(run_dir).resolution.partition

    assert partition.selection is not None
    assert partition.selection.selection_dir == str(out)
    assert partition_side(partition, "train") == sorted(
        {s.member for s in drawn.on("train")}
    )
    assert partition_side(partition, "val") == sorted(
        {s.member for s in drawn.on("val")}
    )
    # The selection's own named grouping policy, carried rather than collapsed into the finite
    # per-sample groups beside it: a stem the groups do not cover is what the policy answers for.
    assert partition.group_by == drawn.group_by == "tile_prefix"
    samples = partition.samples
    assert all(s.group for s in samples)
    assert all(s.ground_truth_digest for s in samples)


def test_a_bound_detection_run_stating_the_composite_objective_selects_by_it(
    tmp_path: Path,
) -> None:
    """A bound detection run stating the composite objective selects by it.

    The chain's detector trained over a drawn selection stating the composite objective as
    its selection metric: the completed run resolves no trait revision, and every epoch it
    logged selects by its validation objective, finite and above zero.
    """
    import math

    from tcip_mcp.experiments import epoch_rows, observe_run, read_rows
    from tests import _chain_fixtures as chain
    from tests._verified_checkpoint_fixtures import worker_run

    root = tmp_path / "ds"
    chain.synthetic_capture(root)
    chain.draw_reference_selection(tmp_path, root, tmp_path / "selection")
    config = chain.run_config(tmp_path, tmp_path / "selection")
    assert config["evaluation"]["selection_metric"] == "objective"

    observation = observe_run(
        worker_run(tmp_path, config, experiment_id="exp-default-objective")
    )

    assert observation.state == "completed", observation.status_error
    assert observation.record.resolution.trait_revision is None
    rows = epoch_rows(read_rows(observation.metrics_log)[0])
    assert rows
    for row in rows:
        assert row["selection_metric"] == "objective"
        assert row["selection"] == row["val_objective"]
        assert math.isfinite(row["selection"]) and row["selection"] > 0


def test_a_run_that_draws_its_own_split_records_no_binding(
    tmp_path: Path,
) -> None:
    """A run that draws its own split records no selection binding.

    A run over the same tree that names no selection draws a partition of its own, and its
    resolved partition names no selection it bound.
    """
    from tcip_mcp.experiments import read_run
    from tests._verified_checkpoint_fixtures import resolved_run

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    run_dir = resolved_run(
        tmp_path,
        {
            "images_dir": str(image_dir(root, DATES[0])),
            "scope": {"subject": SUBJECT},
            "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
        },
        experiment_id="exp-drawn",
    )
    partition = read_run(run_dir).resolution.partition

    assert partition.selection is None
    assert partition_side(partition, "train")
    assert partition_side(partition, "val")
