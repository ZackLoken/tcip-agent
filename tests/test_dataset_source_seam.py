"""The ``dataset_source`` bespoke seam, mirroring ``model_source``.

An agent-supplied importable builder produces a torch ``Dataset`` for a task the built-in loaders
do not cover, built over the producer's own samples (``build_from_dataset_source``, which a run's
loaders and ``ctx.build_dataset`` route to), the known loaders stay the factory's, and the builder
source is snapshotted for provenance.
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

from pathlib import Path

import pytest

from tests import REPO_ROOT
from tests._chain_fixtures import BESPOKE_MODELS, GT_ANCHOR_DETECTOR

torch = pytest.importorskip("torch")

from torch.utils.data import Dataset  # noqa: E402


class _CountingDataset(Dataset):
    """A trivially-bespoke dataset: one item per sample the producer named, echoing the context it
    was built with, each an 8 px frame holding one object named by its stem. ``rows`` is the
    builder's own configuration, which a direct construction with no producer behind it stands on
    instead."""

    def __init__(self, *, samples=None, scope=None, rows=None, marker: str = "", **_ignored):
        self.samples = list(samples or [])
        self.scope = scope
        self.rows = list(rows or [])
        self.marker = marker

    def __len__(self):
        return len(self.samples) or len(self.rows)

    def __getitem__(self, idx):
        named = self.samples or self.rows
        return torch.zeros(3, 8, 8), {"stem": str(named[idx]),
                                      "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                                      "labels": torch.tensor([1]), "iscrowd": torch.tensor([0])}

    @property
    def regions(self):
        """Each frame's served object."""
        from tests._producer_fixtures import served_regions

        return served_regions(self)


def build_bespoke_ds(**kwargs) -> _CountingDataset:
    """Agent-authored dataset builder (importable, no exec)."""
    return _CountingDataset(**kwargs)


BESPOKE_DS = "test_dataset_source_seam:build_bespoke_ds"
"""The ``dataset_source`` builder of :func:`build_bespoke_ds`."""
BESPOKE_DS_FILE = str(REPO_ROOT / "tests" / "test_dataset_source_seam.py")
"""The file :data:`BESPOKE_DS` is defined in, which its ``source_files`` declare; never this
module's own ``__file__``, which names a run's snapshot copy once a run has bound the module
there."""


class _PointDataset(Dataset):
    """A bespoke keypoint dataset over the samples the producer named: each item is one sample's
    own image and the point coordinates its own label document records, which is ground truth no
    built-in loader reads."""

    def __init__(self, *, samples=None, transforms=None, **_ignored):
        from tcip_annotation import json_io
        from tcip_annotation.state import Point

        self.samples = list(samples or [])
        self.transforms = transforms
        self.points = [
            next((float(a.geometry.x), float(a.geometry.y))
                 for a in json_io.read_label_document(sample.ground_truth).annotations
                 if isinstance(a.geometry, Point))
            for sample in self.samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from tcip_mcp.pipelines.image_utils import load_image, pil_to_tensor

        image = pil_to_tensor(load_image(self.samples[idx].image, 3))
        return image, torch.tensor(self.points[idx], dtype=torch.float32)


def build_point_ds(**kwargs) -> _PointDataset:
    """Agent-authored builder for a task whose ground truth is a placed point."""
    return _PointDataset(**kwargs)


DATASET_SOURCE = {
    "builder": BESPOKE_DS,
    "builder_kwargs": {"marker": "bespoke", "rows": ["s0", "s1"]},
    "source_files": [BESPOKE_DS_FILE],
}


def _source(stated: dict):
    """``stated`` validated as a run's ``data.dataset_source``."""
    from tcip_mcp.pipelines.schemas import DatasetSourceSchema

    return DatasetSourceSchema.model_validate(stated)


def _bespoke(stated: dict):
    """``stated`` validated as a run's ``data.dataset_source`` with the layout a run declaring it
    imports from, this repository as its project (``datasets.BespokeSource``)."""
    from tests._producer_fixtures import staged_layout

    return _source(stated), staged_layout(REPO_ROOT, {"dataset_source": stated})


def _admitted_samples(root: Path):
    """Two samples, admitted the way every run's own membership is."""
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import Subject, SubjectRegistry
    from PIL import Image

    from tests._producer_fixtures import label_image, registry_over, samples_over

    images_dir = root / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True, exist_ok=True)
    registry_over(root, SubjectRegistry(subjects=(Subject(name="leaf"),)))
    for stem in ("s0", "s1"):
        Image.new("RGB", (16, 16), (40, 60, 80)).save(images_dir / f"{stem}.png")
        label_image(images_dir / f"{stem}.png",
                    [Annotation(subject="leaf", geometry=BBox(2, 2, 8, 8))], 16, 16,
                    keep_empty=True)
    return samples_over(images_dir, subject="leaf")


def _scope():
    """The class space :func:`_admitted_samples` admits its samples under, a registry declaring
    no attribute on its subject."""
    from tcip_mcp.pipelines.data.label_queries import registry_scope

    return registry_scope(Path(__file__).parent, "leaf")


def _ctx(root: Path, task: str, data: dict):
    """A training body's context over a run of ``root`` of ``task`` whose data block states
    ``data`` beside the place :func:`_admitted_samples` admits there and the class space it
    admits under, and those samples."""
    from dataclasses import asdict

    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tests._chain_fixtures import training_config
    from tests._training_values import evaluation_block
    from tests.tiny_trainer_fixtures import trainer_run

    samples = _admitted_samples(root / "ds")
    config = training_config(
        {"builder": GT_ANCHOR_DETECTOR, "builder_kwargs": {"gt_boxes_wh": [(10, 10)]},
         "source_files": [BESPOKE_MODELS], "task": task},
        {**data, "scope": asdict(_scope()),
         "images_dir": str(root / "ds" / "images" / UNDATED_BUCKET)},
        evaluation=evaluation_block(selection_metric="loss"))
    run = trainer_run(config, root / "out", project=root, has_val_loader=False)
    return TrainContext(run=run, train_loader=None, val_loader=None), samples


def _bespoke_ctx(root: Path):
    """:func:`_ctx` over a run whose loaders :data:`DATASET_SOURCE` builds."""
    return _ctx(root, "grape_bunch_count", {"dataset_source": DATASET_SOURCE})


def test_a_bodys_platform_loader_reads_at_the_runs_sizes_and_lattice(tmp_path: Path):
    """On a run whose data block names no ``dataset_source``, a training body's
    ``ctx.build_dataset`` builds the platform's loader over the subset of samples it was handed,
    at the band count and the lattice the run's data block records."""
    lattice = {"enabled": True, "tile_size": 8, "overlap": 0.25, "sliver_frac": 0.5}
    ctx, samples = _ctx(tmp_path, "detection", {"num_channels": 3, "tiling": lattice})

    tiler = ctx.build_dataset(samples=samples[:1])

    assert (tiler.tile_size, tiler.overlap) == (8, 0.25)
    assert tiler.expected_channels == 3
    assert {tiler.sample_of(k).member for k in tiler.stems} == {samples[0].member}


def test_a_bodys_loader_on_a_bespoke_run_builds_through_its_builder_over_the_subset(
        tmp_path: Path):
    """On a run whose data block names a ``dataset_source``, a training body's
    ``ctx.build_dataset`` imports that builder from the run's layout and hands it the subset of
    samples it was given and the run's recorded class space; the builder's own configuration
    rides in ``builder_kwargs``."""
    ctx, samples = _bespoke_ctx(tmp_path)
    ds = ctx.build_dataset(samples=samples[:1])

    assert type(ds).__qualname__ == "_CountingDataset"
    assert ds.samples == list(samples[:1])   # the subset handed through the door, unchanged
    assert ds.scope == ctx.spec.data.recorded_scope  # the class space they were admitted under
    assert ds.rows == ["s0", "s1"]           # the builder's own configuration, from its own kwargs
    assert ds.marker == "bespoke"            # builder_kwargs applied
    # The platform states nothing about a dataset it did not build, and reads nothing off it.
    assert not hasattr(ds, "expected_channels")


def test_a_bodys_door_takes_the_samples_and_the_augmentation_and_nothing_else(tmp_path: Path):
    """``ctx.build_dataset`` takes the samples and the augmentation and refuses every other
    keyword by name, a builder, a class space, a tiling and sizes included: the run's own block
    states each, and a loader given another could answer for a membership, a class space, a
    format or a lattice this run's record does not state."""
    from tcip_mcp.pipelines.schemas import TilingSpec

    ctx, samples = _bespoke_ctx(tmp_path)
    for unowned in ({"dataset_source": ctx.spec.data.dataset_source}, {"scope": _scope()},
                    {"tiling": TilingSpec.model_validate({"enabled": True})},
                    {"sizes": {"num_channels": 3}}, {"labels_dir": "X:/elsewhere"},
                    {"stems": ["s0"]}, {"label_format": "coco"}, {"subject": "leaf"}):
        with pytest.raises(TypeError) as refused:
            ctx.build_dataset(samples=samples, **unowned)
        assert all(name in str(refused.value) for name in unowned)

    built = ctx.build_dataset(samples=samples, transforms=None)
    assert type(built).__qualname__ == "_CountingDataset"
    assert built.samples == list(samples) and built.scope == _scope()


def test_builder_kwargs_may_not_restate_what_the_producer_named():
    """``samples`` and ``scope`` are the producer's, and ``task``/``transforms`` the run's: a
    builder given a second value for one of them would build over membership or a class space the
    run's own record does not describe, so the seam refuses it by name."""
    from tcip_mcp.pipelines.data.datasets import build_from_dataset_source

    for reserved in ("samples", "scope", "task", "transforms"):
        stated = {"builder": BESPOKE_DS, "builder_kwargs": {reserved: "foreign"},
                  "source_files": [BESPOKE_DS_FILE]}
        with pytest.raises(ValueError, match="restates"):
            build_from_dataset_source(_bespoke(stated), samples=[], scope=_scope(),
                                      task="detection", transforms=None)


def test_known_task_registry_stays_the_default(tmp_path: Path):
    from tcip_mcp.pipelines.data.datasets import build_dataset

    # No dataset_source -> the closed-registry refusal stays the honest signal for a bad name.
    with pytest.raises(ValueError, match="Unknown task"):
        build_dataset("grape_bunch_count", samples=_admitted_samples(tmp_path / "ds"), sizes={},
                      scope=_scope())


def test_builder_kwargs_configure_the_builder(tmp_path: Path):
    """The builder's own configuration is its ``builder_kwargs`` and nothing else: the lowest
    boundary requires the dataset source with the layout it imports from, the task, the samples,
    the scope and the transforms."""
    import inspect

    from tcip_mcp.pipelines.data.datasets import build_from_dataset_source

    required = {name for name, p in
                inspect.signature(build_from_dataset_source).parameters.items()
                if p.default is inspect.Parameter.empty}
    assert required == {"bespoke", "task", "samples", "scope", "transforms"}

    samples = _admitted_samples(tmp_path / "ds")
    stated = {"builder": BESPOKE_DS, "builder_kwargs": {"marker": "pinned"},
              "source_files": [BESPOKE_DS_FILE]}
    ds = build_from_dataset_source(_bespoke(stated), task="grape_bunch_count",
                                   samples=samples, scope=_scope(), transforms=None)
    assert ds.marker == "pinned" and ds.samples == list(samples)

    with pytest.raises(ValueError, match="builder_kwargs"):
        _source({"builder": BESPOKE_DS, "builder_kwargs": [1, 2]})


def test_preflight_requires_the_data_a_bespoke_run_is_still_admitted_from(tmp_path: Path):
    """A bespoke builder does not exempt a run from naming its data: the platform's own producer
    admits its samples whatever loads them, so a config naming no place refuses here by naming the
    missing key, the one refusal every route shares."""
    from tcip_mcp.tools.training_tools import preflight_config
    from tests._chain_fixtures import training_config

    data: dict = {"dataset_source": DATASET_SOURCE}
    config = training_config({"builder": GT_ANCHOR_DETECTOR,
                              "builder_kwargs": {"gt_boxes_wh": [(10, 10)]},
                              "source_files": [BESPOKE_MODELS],
                              "task": "grape_bunch_count"}, data, batch_size=1)
    result = preflight_config(tmp_path, config, smoke=False)
    assert not result["valid"]
    assert result["issues"] == ["Missing 'data.images_dir'"]

    # Admits valid work: the same bespoke source over a real place, whose samples the producer
    # actually admits, passes with no issue.
    root = tmp_path / "ds"
    _admitted_samples(root)
    located_data = {**data, "images_dir": str(root / "images" / UNDATED_BUCKET),
                    "scope": {"subject": "leaf"}, "split": {"seed": 0, "val_ratio": 0.15}}
    admitted = preflight_config(tmp_path, {**config, "data": located_data}, smoke=False)
    assert admitted["issues"] == [], admitted["issues"]

    # A builder no declared file holds is refused naming source_files, over that same place.
    bad_data = {**located_data, "dataset_source": {"builder": "no.such:fn"}}
    result = preflight_config(tmp_path, {**config, "data": bad_data}, smoke=False)
    assert not result["valid"]
    assert any(i.startswith("source_files: ") and "no.such" in i for i in result["issues"])


def test_snapshot_copies_the_dataset_builders_module_as_that_module(tmp_path: Path):
    """The dataset builder's module, declared under its ``source_files``, is copied at its path
    under its import root; a dataset builder none of them holds refuses the snapshot by name."""
    from tcip_mcp.pipelines.model_build import snapshot_model_source
    from tcip_mcp.pipelines.schemas import train_config
    from tests._chain_fixtures import training_config

    model_source = {"builder": GT_ANCHOR_DETECTOR, "source_files": [BESPOKE_MODELS],
                    "task": "detection"}
    spec = train_config(training_config(model_source, {"dataset_source": DATASET_SOURCE}))
    record, copies = snapshot_model_source(spec, tmp_path)

    entry = record["files"][str(Path(BESPOKE_DS_FILE).resolve())]
    assert entry["file"] == "model_src/test_dataset_source_seam.py"
    assert copies[entry["file"]] == Path(__file__).read_bytes()
    undeclared = train_config(training_config(model_source,
                                              {"dataset_source": {"builder": BESPOKE_DS}}))
    with pytest.raises(ValueError, match="test_dataset_source_seam"):
        snapshot_model_source(undeclared, tmp_path)
