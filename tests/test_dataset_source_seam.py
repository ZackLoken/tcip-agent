"""The ``dataset_source`` bespoke seam.

An agent-supplied importable builder produces a ``BaseDataset`` for a task the built-in loaders
do not cover, built over the producer's own samples (``build_from_dataset_source``), the known
loaders stay the factory's, and the builder source is snapshotted for provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tests import REPO_ROOT
from tests._chain_fixtures import (
    BESPOKE_MODELS,
    FRAME_STATISTICS,
    GT_ANCHOR_DETECTOR,
)
from tests._training_values import drawn_split, tiling_block

torch = pytest.importorskip("torch")

from torch.utils.data import Dataset  # noqa: E402

from tcip_mcp.pipelines.data.datasets import BaseDataset

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from PIL import Image
    from torch import Tensor

    from tcip_mcp.pipelines.data.datasets import BespokeSource
    from tcip_mcp.pipelines.data.selection import ClassScope, Sample
    from tcip_mcp.pipelines.schemas import DatasetSourceSchema
    from tcip_mcp.pipelines.training.envelope import TrainContext


class _CountingDataset(BaseDataset):
    """A trivially-bespoke dataset of one item per sample the producer named.

    It echoes the context it was built with, each item a black 8 px frame holding one object
    named by its stem and valued zero. ``rows`` is the builder's own configuration, the items of
    a construction given no samples.
    """

    def __init__(
        self,
        *,
        samples: Sequence[Sample] | None = None,
        scope: ClassScope | None = None,
        rows: list[str] | None = None,
        marker: str = "",
        **_ignored: object,
    ) -> None:
        self.samples = list(samples or [])
        self.scope = scope
        self.rows = list(rows or [])
        self.marker = marker

    @property
    def num_samples(self) -> int:
        return len(self.samples) or len(self.rows)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, str | Tensor]]:
        named = self.samples or self.rows
        return torch.zeros(3, 8, 8), {
            "stem": str(named[idx]),
            "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            "labels": torch.tensor([1]),
            "iscrowd": torch.tensor([0]),
            "values": torch.tensor(0.0),
        }

    @property
    def regions(self) -> list:
        """The dataset's regions, one per frame, holding the object it serves."""
        from tests._producer_fixtures import served_regions

        return served_regions(self)


def build_bespoke_ds(
    **kwargs: str
    | Sequence[Sample]
    | ClassScope
    | list[str]
    | Callable[[Image.Image, dict], tuple[Tensor, dict]]
    | None,
) -> _CountingDataset:
    """Build a :class:`_CountingDataset`, as an importable agent-authored builder does."""
    return _CountingDataset(**kwargs)


class _PlainDataset(Dataset):
    """A torch dataset that is no ``BaseDataset``, holding one item per sample it was handed."""

    def __init__(
        self, *, samples: Sequence[Sample] | None = None, **_ignored: object
    ) -> None:
        self.samples = list(samples or [])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict]:
        return torch.zeros(3, 8, 8), {}


def build_plain_ds(**kwargs: object) -> _PlainDataset:
    """Build a :class:`_PlainDataset`, a builder returning what the seam refuses."""
    return _PlainDataset(**kwargs)


BESPOKE_DS = "test_dataset_source_seam:build_bespoke_ds"
"""The ``dataset_source`` builder of :func:`build_bespoke_ds`."""
BESPOKE_DS_FILE = str(REPO_ROOT / "tests" / "test_dataset_source_seam.py")
"""The source file defining :data:`BESPOKE_DS`."""


class _PointDataset(BaseDataset):
    """A bespoke keypoint dataset over the samples the producer named.

    Each item is one sample's own image and the first point its own label document records.
    """

    def __init__(
        self,
        *,
        samples: Sequence[Sample] | None = None,
        transforms: Callable[[Image.Image, dict], tuple[Tensor, dict]]
        | None = None,
        **_ignored: object,
    ) -> None:
        from tcip_annotation import json_io
        from tcip_annotation.state import Point

        self.samples = list(samples or [])
        self.transforms = transforms
        self.points = [
            next(
                (float(a.geometry.x), float(a.geometry.y))
                for a in json_io.read_label_document(
                    sample.ground_truth
                ).annotations
                if isinstance(a.geometry, Point)
            )
            for sample in self.samples
        ]

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict]:
        from tcip_mcp.pipelines.image_utils import load_image, pil_to_tensor

        image = pil_to_tensor(load_image(self.samples[idx].image, 3))
        return image, {
            "point": torch.tensor(self.points[idx], dtype=torch.float32)
        }


def build_point_ds(
    **kwargs: str
    | Sequence[Sample]
    | ClassScope
    | Callable[[Image.Image, dict], tuple[Tensor, dict]]
    | None,
) -> _PointDataset:
    """Build a :class:`_PointDataset` for a task whose ground truth is a placed point."""
    return _PointDataset(**kwargs)


DATASET_SOURCE = {
    "builder": BESPOKE_DS,
    "builder_kwargs": {"marker": "bespoke", "rows": ["s0", "s1"]},
    "source_files": [BESPOKE_DS_FILE],
}


def _source(stated: dict) -> DatasetSourceSchema:
    """Return ``stated`` validated as a run's ``data.dataset_source``."""
    from tcip_mcp.pipelines.schemas import DatasetSourceSchema

    return DatasetSourceSchema.model_validate(stated)


def _bespoke(stated: dict) -> BespokeSource:
    """Return ``stated`` validated as a run's ``data.dataset_source``, and its layout.

    The layout is the one a run declaring it imports from, this repository as its project
    (``datasets.BespokeSource``).
    """
    from tests._producer_fixtures import staged_layout

    return _source(stated), staged_layout(
        REPO_ROOT, {"dataset_source": stated}
    )


def _admitted_samples(root: Path) -> list[Sample]:
    """Return two labeled samples, admitted through ``samples_over``.

    Each frame holds its object bright on a dim ground.
    """
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import Subject, SubjectRegistry
    from tests._producer_fixtures import (
        BRIGHT,
        label_image,
        painted_frame,
        registry_over,
        samples_over,
    )

    images_dir = root / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True, exist_ok=True)
    registry_over(
        root, SubjectRegistry(subjects=(Subject(name="leaf", description=""),))
    )
    for stem in ("s0", "s1"):
        painted_frame(16, 16, (40, 60, 80), [((2, 2, 8, 8), BRIGHT)]).save(
            images_dir / f"{stem}.png"
        )
        label_image(
            images_dir / f"{stem}.png",
            [Annotation(subject="leaf", geometry=BBox(2, 2, 8, 8))],
            16,
            16,
            keep_empty=True,
        )
    return samples_over(images_dir, subject="leaf")


def _scope() -> ClassScope:
    """Return the class space :func:`_admitted_samples` admits its samples under.

    It is a registry's, declaring no attribute on its subject.
    """
    from tcip_mcp.pipelines.data.label_queries import registry_scope

    return registry_scope(Path(__file__).parent, "leaf")


def _ctx(
    root: Path, task: str, data: dict
) -> tuple[TrainContext, list[Sample]]:
    """Return a training body's context over a run of ``root`` of ``task``, and its samples.

    The run is opened by the launcher's own producer, its data block stating ``data`` beside the
    place :func:`_admitted_samples` admits there and the class space it admits under, with no
    validation side.
    """
    from dataclasses import asdict

    from tests._chain_fixtures import training_config
    from tests._training_values import evaluation_block
    from tests.tiny_trainer_fixtures import run_context

    samples = _admitted_samples(root / "ds")
    config = training_config(
        root,
        {
            "builder": GT_ANCHOR_DETECTOR,
            "source_files": [BESPOKE_MODELS],
            "task": task,
            "normalization": FRAME_STATISTICS,
        },
        {
            **data,
            "scope": asdict(_scope()),
            "split": drawn_split(val_ratio=0.0),
            "images_dir": str(root / "ds" / "images" / UNDATED_BUCKET),
        },
        evaluation=evaluation_block(selection_metric="loss"),
    )
    return run_context(root, config), samples


def _bespoke_ctx(root: Path) -> tuple[TrainContext, list[Sample]]:
    """Return :func:`_ctx` over a run whose loaders :data:`DATASET_SOURCE` builds."""
    return _ctx(root, "grape_bunch_count", {"dataset_source": DATASET_SOURCE})


def test_a_bodys_platform_loader_reads_at_the_runs_sizes_and_lattice(
    tmp_path: Path,
) -> None:
    """A body's platform loader reads at the run's recorded band count and lattice.

    On a run whose data block names no ``dataset_source``, a training body's
    ``ctx.build_dataset`` builds the platform's loader over the subset of samples it was handed,
    at the band count and the lattice the run's data block records.
    """
    lattice = tiling_block(tile_size=8, overlap=0.25, sliver_frac=0.5)
    ctx, samples = _ctx(
        tmp_path, "detection", {"num_channels": 3, "tiling": lattice}
    )

    tiler = ctx.build_dataset(samples=samples[:1])

    assert (tiler.tile_size, tiler.overlap) == (8, 0.25)
    assert tiler.expected_channels == 3
    assert {tiler.sample_of(k).member for k in tiler.stems} == {
        samples[0].member
    }


def test_a_bodys_loader_on_a_bespoke_run_builds_through_its_builder_over_the_subset(
    tmp_path: Path,
) -> None:
    """A body's loader on a bespoke run builds through its builder over the subset handed it.

    On a run whose data block names a ``dataset_source``, a training body's
    ``ctx.build_dataset`` imports that builder from the run's layout and hands it the subset of
    samples it was given and the run's recorded class space; the builder's own configuration
    rides in ``builder_kwargs``.
    """
    ctx, samples = _bespoke_ctx(tmp_path)
    ds = ctx.build_dataset(samples=samples[:1])

    assert type(ds).__qualname__ == "_CountingDataset"
    assert ds.samples == list(
        samples[:1]
    )  # the subset handed through the door, unchanged
    assert (
        ds.scope == ctx.spec.data.recorded_scope
    )  # the class space they were admitted under
    assert ds.rows == [
        "s0",
        "s1",
    ]  # the builder's own configuration, from its own kwargs
    assert ds.marker == "bespoke"  # builder_kwargs applied
    # The platform states nothing about a dataset it did not build, and reads nothing off it.
    assert not hasattr(ds, "expected_channels")


def test_a_bodys_door_takes_the_samples_and_the_augmentation_and_nothing_else(
    tmp_path: Path,
) -> None:
    """``ctx.build_dataset`` takes the samples and the augmentation and refuses other keywords.

    Each keyword tried, a builder, a class space, a tiling and sizes among them, is refused by
    name, and a call with the samples and the augmentation builds the run's own dataset.
    """
    from tcip_mcp.pipelines.schemas import TilingSpec

    ctx, samples = _bespoke_ctx(tmp_path)
    for unowned in (
        {"dataset_source": ctx.spec.data.dataset_source},
        {"scope": _scope()},
        {"tiling": TilingSpec.model_validate(tiling_block())},
        {"sizes": {"num_channels": 3}},
        {"labels_dir": "X:/elsewhere"},
        {"stems": ["s0"]},
        {"label_format": "coco"},
        {"subject": "leaf"},
    ):
        with pytest.raises(TypeError) as refused:
            ctx.build_dataset(samples=samples, **unowned)
        assert all(name in str(refused.value) for name in unowned)

    built = ctx.build_dataset(samples=samples, transforms=None)
    assert type(built).__qualname__ == "_CountingDataset"
    assert built.samples == list(samples) and built.scope == _scope()


def test_a_builder_returning_no_base_dataset_is_refused_naming_it(
    tmp_path: Path,
) -> None:
    """A run whose builder returns a dataset that is no ``BaseDataset`` refuses naming the builder.

    The launcher's own producer builds the run's loaders through the builder, and the seam
    refuses the plain torch dataset it returns, naming the builder and the class it must return.
    """
    plain = {
        "builder": "test_dataset_source_seam:build_plain_ds",
        "source_files": [BESPOKE_DS_FILE],
    }
    with pytest.raises(ValueError, match="build_plain_ds.*_PlainDataset"):
        _ctx(tmp_path, "grape_bunch_count", {"dataset_source": plain})


def test_builder_kwargs_may_not_restate_what_the_producer_named() -> None:
    """``builder_kwargs`` restating a value the producer or the run names is refused by name.

    The context's fields are ``task``, ``samples``, ``scope`` and ``transforms``, and a
    ``builder_kwargs`` entry for any of them is refused naming it.
    """
    from tcip_mcp.pipelines.schemas import DatasetContext

    assert DatasetContext._fields == ("task", "samples", "scope", "transforms")
    for reserved in DatasetContext._fields:
        stated = {
            "builder": BESPOKE_DS,
            "builder_kwargs": {reserved: "foreign"},
            "source_files": [BESPOKE_DS_FILE],
        }
        with pytest.raises(ValueError, match=f"restates \\['{reserved}'\\]"):
            _source(stated)


def test_known_task_registry_stays_the_default(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.data.datasets import build_dataset

    # No dataset_source -> the closed-registry refusal stays the honest signal for a bad name.
    with pytest.raises(ValueError, match="Unknown task"):
        build_dataset(
            "grape_bunch_count",
            samples=_admitted_samples(tmp_path / "ds"),
            sizes={},
            scope=_scope(),
        )


def test_builder_kwargs_configure_the_builder(tmp_path: Path) -> None:
    """The builder's own configuration is its ``builder_kwargs`` and nothing else.

    The lowest boundary requires the dataset source with the layout it imports from, the task,
    the samples, the scope and the transforms.
    """
    import inspect

    from tcip_mcp.pipelines.data.datasets import build_from_dataset_source

    required = {
        name
        for name, p in inspect.signature(
            build_from_dataset_source
        ).parameters.items()
        if p.default is inspect.Parameter.empty
    }
    assert required == {"bespoke", "task", "samples", "scope", "transforms"}

    samples = _admitted_samples(tmp_path / "ds")
    stated = {
        "builder": BESPOKE_DS,
        "builder_kwargs": {"marker": "pinned"},
        "source_files": [BESPOKE_DS_FILE],
    }
    ds = build_from_dataset_source(
        _bespoke(stated),
        task="grape_bunch_count",
        samples=samples,
        scope=_scope(),
        transforms=None,
    )
    assert ds.marker == "pinned" and ds.samples == list(samples)

    with pytest.raises(ValueError, match="builder_kwargs"):
        _source({"builder": BESPOKE_DS, "builder_kwargs": [1, 2]})


def test_preflight_requires_the_data_a_bespoke_run_is_still_admitted_from(
    tmp_path: Path,
) -> None:
    """Preflight requires the data place a bespoke run is admitted from.

    A bespoke config naming no place is refused naming ``data.images_dir``, the same config over
    a labeled place passes, and a builder no declared file holds is refused naming
    ``source_files``.
    """
    from tcip_mcp.tools.training_tools import preflight_config
    from tests._chain_fixtures import training_config

    data: dict = {
        "dataset_source": DATASET_SOURCE,
        "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
    }
    root = tmp_path / "ds"
    _admitted_samples(root)
    located_data = {
        **data,
        "images_dir": str(root / "images" / UNDATED_BUCKET),
        "scope": {"subject": "leaf"},
    }
    config = training_config(
        tmp_path,
        {
            "builder": GT_ANCHOR_DETECTOR,
            "source_files": [BESPOKE_MODELS],
            "task": "grape_bunch_count",
            "normalization": FRAME_STATISTICS,
        },
        located_data,
        batch_size=1,
    )
    result = preflight_config(tmp_path, {**config, "data": data}, smoke=False)
    assert not result["valid"]
    assert result["issues"] == ["Missing 'data.images_dir'"]

    # Admits valid work: the same bespoke source over a real place, whose samples the producer
    # actually admits, passes with no issue.
    admitted = preflight_config(
        tmp_path, {**config, "data": located_data}, smoke=False
    )
    assert admitted["issues"] == [], admitted["issues"]

    # A builder no declared file holds is refused naming source_files, over that same place.
    bad_data = {**located_data, "dataset_source": {"builder": "no.such:fn"}}
    result = preflight_config(
        tmp_path, {**config, "data": bad_data}, smoke=False
    )
    assert not result["valid"]
    assert any(
        i.startswith("source_files: ") and "no.such" in i
        for i in result["issues"]
    )


def test_snapshot_copies_the_dataset_builders_module_as_that_module(
    tmp_path: Path,
) -> None:
    """The snapshot copies the dataset builder's module as that module.

    The module, declared under its ``source_files``, is copied at its path under its import
    root; a dataset builder none of them holds refuses the snapshot by name.
    """
    from tcip_mcp.pipelines.model_build import snapshot_model_source
    from tcip_mcp.pipelines.schemas import train_config
    from tests._chain_fixtures import training_config, weights_statistics

    model_source = weights_statistics(
        {
            "builder": GT_ANCHOR_DETECTOR,
            "source_files": [BESPOKE_MODELS],
            "task": "detection",
        }
    )
    spec = train_config(
        training_config(
            tmp_path,
            model_source,
            {"dataset_source": DATASET_SOURCE, "split": drawn_split()},
        )
    )
    record, copies = snapshot_model_source(spec, tmp_path)

    entry = record["files"][str(Path(BESPOKE_DS_FILE).resolve())]
    assert entry["file"] == "model_src/test_dataset_source_seam.py"
    assert copies[entry["file"]] == Path(__file__).read_bytes()
    undeclared = train_config(
        training_config(
            tmp_path,
            model_source,
            {
                "dataset_source": {"builder": BESPOKE_DS},
                "split": drawn_split(),
            },
        )
    )
    with pytest.raises(ValueError, match="test_dataset_source_seam"):
        snapshot_model_source(undeclared, tmp_path)
