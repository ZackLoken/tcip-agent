"""The ``dataset_source`` bespoke seam, mirroring ``model_source``.

An agent-supplied importable builder produces a torch ``Dataset`` for a task the built-in loaders
do not cover, ``build_dataset`` routes to it over the producer's own samples, the known loaders
stay the default, and the builder source is snapshotted for provenance.
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

from pathlib import Path

import pytest

from tests._chain_fixtures import GT_ANCHOR_DETECTOR

torch = pytest.importorskip("torch")

from torch.utils.data import Dataset  # noqa: E402


class _CountingDataset(Dataset):
    """A trivially-bespoke dataset: one item per sample the producer named, echoing the context it
    was built with. ``rows`` is the builder's own configuration, which a direct construction with
    no producer behind it stands on instead."""

    def __init__(self, *, samples=None, scope=None, rows=None, marker: str = "", **_ignored):
        self.samples = list(samples or [])
        self.scope = scope
        self.rows = list(rows or [])
        self.marker = marker

    def __len__(self):
        return len(self.samples) or len(self.rows)

    def __getitem__(self, idx):
        named = self.samples or self.rows
        return torch.zeros(3, 8, 8), {"stem": str(named[idx])}


def build_bespoke_ds(**kwargs) -> _CountingDataset:
    """Agent-authored dataset builder (importable, no exec)."""
    return _CountingDataset(**kwargs)


BESPOKE_DS = "tests.test_dataset_source_seam:build_bespoke_ds"
"""The ``dataset_source`` builder of :func:`build_bespoke_ds`."""


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
    "source_files": [__file__],
}


def _source(stated: dict):
    """``stated`` validated as a run's ``data.dataset_source``."""
    from tcip_mcp.pipelines.schemas import DatasetSourceSchema

    return DatasetSourceSchema.model_validate(stated)


def _admitted_samples(root: Path):
    """Two samples, admitted the way every run's own membership is: a builder is handed the
    producer's records, never a list a test wrote by hand."""
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


def test_build_dataset_routes_to_dataset_source(tmp_path: Path):
    """The factory routes a task no built-in loader covers to the agent's own builder, handing it
    the run's samples and nothing else; the builder's own configuration rides in
    ``builder_kwargs``."""
    from tcip_mcp.pipelines.data.datasets import build_dataset

    samples = _admitted_samples(tmp_path / "ds")
    ds = build_dataset("grape_bunch_count", dataset_source=_source(DATASET_SOURCE),
                       samples=samples, sizes={}, scope=_scope(), transforms=None)

    assert isinstance(ds, _CountingDataset)
    assert ds.samples == list(samples)       # the producer's own membership, unchanged
    assert ds.scope == _scope()              # the class space they were admitted under, whole
    assert ds.rows == ["s0", "s1"]           # the builder's own configuration, from its own kwargs
    assert ds.marker == "bespoke"            # builder_kwargs applied
    # The platform states nothing about a dataset it did not build, and reads nothing off it.
    assert not hasattr(ds, "expected_channels")


def test_a_bespoke_builder_is_handed_only_what_the_producer_named(tmp_path: Path):
    """The one factory boundary hands a bespoke builder the producer's four names and refuses
    every other kwarg it was given, so the promise holds on every route into it, an agent's own
    ``ctx.build_dataset`` call included: a builder given anything else could answer for a
    membership, a class space or a format this run's own record does not state. The check runs
    before anything is read off those kwargs, so a name the factory itself owns is refused too."""
    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.schemas import TilingSpec

    samples = _admitted_samples(tmp_path / "ds")
    source = _source(DATASET_SOURCE)
    for unowned in ({"labels_dir": "X:/elsewhere"}, {"images_dir": "X:/elsewhere"},
                    {"csv_path": "X:/elsewhere/table.csv"}, {"stems": ["s0"]},
                    {"val_images_dir": "X:/elsewhere/val"},
                    {"label_format": "coco"}, {"coco_path": "X:/elsewhere/instances.json"},
                    {"subject": "leaf"}):
        with pytest.raises(ValueError, match="was given"):
            build_dataset("grape_bunch_count", dataset_source=source, samples=samples,
                          sizes={}, scope=_scope(), **unowned)

    # A tiling is refused beside a builder that composes its own.
    with pytest.raises(ValueError, match="beside a dataset_source"):
        build_dataset("grape_bunch_count", dataset_source=source, samples=samples,
                      sizes={"num_channels": 3}, scope=_scope(),
                      tiling=TilingSpec.model_validate({"enabled": True}))

    # Admits valid work: the producer's own samples and class space reach the builder unchanged.
    built = build_dataset("grape_bunch_count", dataset_source=source, samples=samples,
                          sizes={}, scope=_scope(), transforms=None)
    assert isinstance(built, _CountingDataset)
    assert built.samples == list(samples) and built.scope == _scope()


def test_builder_kwargs_may_not_restate_what_the_producer_named():
    """``samples`` and ``scope`` are the producer's, and ``task``/``transforms`` the run's: a
    builder given a second value for one of them would build over membership or a class space the
    run's own record does not describe, so the seam refuses it by name."""
    from tcip_mcp.pipelines.data.datasets import build_from_dataset_source

    for reserved in ("samples", "scope", "task", "transforms"):
        with pytest.raises(ValueError, match="restates"):
            build_from_dataset_source(
                _source({"builder": BESPOKE_DS, "builder_kwargs": {reserved: "foreign"}}),
                samples=[], scope=_scope(), task="detection", transforms=None)


def test_known_task_registry_stays_the_default(tmp_path: Path):
    from tcip_mcp.pipelines.data.datasets import build_dataset

    # No dataset_source -> the closed-registry refusal stays the honest signal for a bad name.
    with pytest.raises(ValueError, match="Unknown task"):
        build_dataset("grape_bunch_count", samples=_admitted_samples(tmp_path / "ds"), sizes={},
                      scope=_scope())


def test_builder_kwargs_configure_the_builder(tmp_path: Path):
    """The builder's own configuration is its ``builder_kwargs`` and nothing else: the lowest
    boundary takes the producer's four names as required arguments, so there is no second,
    looser entrance a caller could hand other context through."""
    import inspect

    from tcip_mcp.pipelines.data.datasets import build_from_dataset_source

    required = {name for name, p in
                inspect.signature(build_from_dataset_source).parameters.items()
                if p.default is inspect.Parameter.empty}
    assert required == {"dataset_source", "task", "samples", "scope", "transforms"}

    samples = _admitted_samples(tmp_path / "ds")
    ds = build_from_dataset_source(
        _source({"builder": BESPOKE_DS, "builder_kwargs": {"marker": "pinned"}}),
        task="grape_bunch_count", samples=samples, scope=_scope(), transforms=None)
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

    # A non-importable builder is caught honestly, over that same real place.
    bad_data = {**located_data, "dataset_source": {"builder": "no.such:fn"}}
    result = preflight_config(tmp_path, {**config, "data": bad_data}, smoke=False)
    assert not result["valid"]
    assert any("dataset_source.builder not importable" in i for i in result["issues"])


def test_snapshot_records_dataset_builder(tmp_path: Path):
    from tcip_mcp.pipelines.model_build import SNAPSHOT_DIR, snapshot_model_source
    from tcip_mcp.pipelines.schemas import train_config
    from tests._chain_fixtures import training_config

    spec = train_config(training_config({"builder": GT_ANCHOR_DETECTOR, "task": "detection"},
                                        {"dataset_source": DATASET_SOURCE}))
    manifest = snapshot_model_source(spec, tmp_path)
    assert manifest["dataset_builder"] == BESPOKE_DS
    [entry] = [e for e in manifest["files"] if e["src"] == __file__]
    assert len(entry["sha256"]) == 64
    assert (tmp_path / SNAPSHOT_DIR / entry["file"]).read_bytes() == Path(__file__).read_bytes()
