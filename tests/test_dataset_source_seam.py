"""The ``dataset_source`` bespoke seam, mirroring ``model_source``.

An agent-supplied importable builder produces a torch ``Dataset`` for a task the built-in loaders
do not cover, ``build_dataset`` routes to it over the producer's own samples, the known loaders
stay the default, and the builder source is snapshotted for provenance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import Dataset  # noqa: E402


class _CountingDataset(Dataset):
    """A trivially-bespoke dataset: one item per sample the producer named, echoing the context it
    was built with. ``rows`` is the builder's own configuration, which a direct construction with
    no producer behind it stands on instead."""

    def __init__(self, *, samples=None, id_map=None, rows=None, marker: str = "", **_ignored):
        self.samples = list(samples or [])
        self.id_map = id_map
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
                 for a in json_io.read_annotations(sample.ground_truth)
                 if isinstance(a.geometry, Point))
            for sample in self.samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from tcip_mcp.pipelines.image_utils import load_image, pil_to_tensor

        image = pil_to_tensor(load_image(Path(self.samples[idx].source), 3))
        return image, torch.tensor(self.points[idx], dtype=torch.float32)


def build_point_ds(**kwargs) -> _PointDataset:
    """Agent-authored builder for a task whose ground truth is a placed point."""
    return _PointDataset(**kwargs)


DATASET_SOURCE = {
    "builder": "tests.test_dataset_source_seam:build_bespoke_ds",
    "builder_kwargs": {"marker": "bespoke", "rows": ["s0", "s1"]},
    "source_files": [__file__],
    "task": "grape_bunch_count",
}


def _admitted_samples(root: Path):
    """Two samples, admitted the way every run's own membership is: a builder is handed the
    producer's records, never a list a test wrote by hand."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import Subject, SubjectRegistry, write_registry
    from PIL import Image

    from tests._producer_fixtures import samples_over

    images_dir, labels_dir = root / "images", root / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    write_registry(root / "subjects.json", SubjectRegistry(subjects=(Subject(name="leaf"),)))
    for stem in ("s0", "s1"):
        Image.new("RGB", (16, 16), (40, 60, 80)).save(images_dir / f"{stem}.png")
        json_io.write_annotations(labels_dir / f"{stem}.json",
                                  [Annotation(subject="leaf", geometry=BBox(2, 2, 8, 8))],
                                  16, 16, keep_empty=True)
    return samples_over(images_dir, labels_dir, subject="leaf")


def test_build_dataset_routes_to_dataset_source(tmp_path: Path):
    """The factory routes a task no built-in loader covers to the agent's own builder, handing it
    the run's samples and nothing else; the builder's own configuration rides in
    ``builder_kwargs``."""
    from tcip_mcp.pipelines.data.datasets import build_dataset

    samples = _admitted_samples(tmp_path / "ds")
    ds = build_dataset("grape_bunch_count", dataset_source=DATASET_SOURCE, samples=samples,
                       sizes={}, transforms=None)

    assert isinstance(ds, _CountingDataset)
    assert ds.samples == list(samples)       # the producer's own membership, unchanged
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

    samples = _admitted_samples(tmp_path / "ds")
    for unowned in ({"labels_dir": "X:/elsewhere"}, {"images_dir": "X:/elsewhere"},
                    {"csv_path": "X:/elsewhere/table.csv"}, {"stems": ["s0"]},
                    {"val_images_dir": "X:/elsewhere/val"},
                    {"label_format": "coco"}, {"coco_path": "X:/elsewhere/instances.json"},
                    {"subject": "leaf"}):
        with pytest.raises(ValueError, match="was given"):
            build_dataset("grape_bunch_count", dataset_source=DATASET_SOURCE, samples=samples,
                          sizes={}, **unowned)

    # A size the platform resolved is refused beside a builder that sizes its own dataset.
    with pytest.raises(ValueError, match="beside a dataset_source"):
        build_dataset("grape_bunch_count", dataset_source=DATASET_SOURCE, samples=samples,
                      sizes={"num_channels": 3})

    # Admits valid work: the producer's own samples and class space reach the builder unchanged.
    from tcip_mcp.pipelines.data.selection import ClassScope

    built = build_dataset("grape_bunch_count", dataset_source=DATASET_SOURCE, samples=samples,
                          sizes={}, scope=ClassScope(id_map={"leaf": 0}), transforms=None)
    assert isinstance(built, _CountingDataset)
    assert built.samples == list(samples) and built.id_map == {"leaf": 0}


def test_builder_kwargs_may_not_restate_what_the_producer_named():
    """``samples`` and ``id_map`` are the producer's, and ``task``/``transforms`` the run's: a
    builder given a second value for one of them would build over membership or a class space the
    run's own record does not describe, so the seam refuses it by name."""
    from tcip_mcp.pipelines.data.datasets import build_from_dataset_source

    for reserved in ("samples", "id_map", "task", "transforms"):
        with pytest.raises(ValueError, match="restates"):
            build_from_dataset_source(
                {"builder": "tests.test_dataset_source_seam:build_bespoke_ds",
                 "builder_kwargs": {reserved: "foreign"}},
                samples=[], id_map={"admitted": 0}, task="detection", transforms=None)


def test_known_task_registry_stays_the_default(tmp_path: Path):
    from tcip_mcp.pipelines.data.datasets import build_dataset

    # No dataset_source -> the closed-registry refusal stays the honest signal for a bad name.
    with pytest.raises(ValueError, match="Unknown task"):
        build_dataset("grape_bunch_count", samples=_admitted_samples(tmp_path / "ds"), sizes={})


def test_builder_kwargs_configure_the_builder(tmp_path: Path):
    """The builder's own configuration is its ``builder_kwargs`` and nothing else: the lowest
    boundary takes the producer's four names as required arguments, so there is no second,
    looser entrance a caller could hand other context through."""
    import inspect

    from tcip_mcp.pipelines.data.datasets import build_from_dataset_source

    required = {name for name, p in
                inspect.signature(build_from_dataset_source).parameters.items()
                if p.default is inspect.Parameter.empty}
    assert required == {"dataset_source", "task", "samples", "id_map", "transforms"}

    samples = _admitted_samples(tmp_path / "ds")
    ds = build_from_dataset_source(
        {"builder": "tests.test_dataset_source_seam:build_bespoke_ds",
         "builder_kwargs": {"marker": "pinned"}},
        task="grape_bunch_count", samples=samples, id_map=None, transforms=None)
    assert ds.marker == "pinned" and ds.samples == list(samples)

    with pytest.raises(ValueError, match="builder_kwargs must be a dict"):
        build_from_dataset_source(
            {"builder": "tests.test_dataset_source_seam:build_bespoke_ds",
             "builder_kwargs": [1, 2]},
            task="grape_bunch_count", samples=samples, id_map=None, transforms=None)


def test_preflight_requires_the_data_a_bespoke_run_is_still_admitted_from(tmp_path: Path):
    """A bespoke builder does not exempt a run from naming its data: the platform's own producer
    admits its samples whatever loads them, so a config naming no place refuses here by naming the
    missing key, the one refusal every route shares."""
    from tcip_mcp.tools.training_tools import preflight_config

    data: dict = {"dataset_source": DATASET_SOURCE, "task": "grape_bunch_count"}
    config = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detector",
                         "builder_kwargs": {"gt_boxes_wh": [(10, 10)], "num_classes": 1},
                         "task": "grape_bunch_count"},
        "data": data,
        "batch_size": 1,
    }
    result = preflight_config(config, smoke=False)
    assert not result["valid"]
    assert sorted(result["issues"]) == ["Missing 'data.images_dir'", "Missing 'data.labels_dir'"]

    # Admits valid work: the same bespoke source over a real place, whose samples the producer
    # actually admits, passes with no issue.
    root = tmp_path / "ds"
    _admitted_samples(root)
    located_data = {**data, "images_dir": str(root / "images"),
                    "labels_dir": str(root / "annotations"), "subject": "leaf"}
    admitted = preflight_config({**config, "data": located_data}, smoke=False)
    assert admitted["issues"] == [], admitted["issues"]

    # A non-importable builder is caught honestly, over that same real place.
    bad_data = {**located_data, "dataset_source": {"builder": "no.such:fn"}}
    result = preflight_config({**config, "data": bad_data}, smoke=False)
    assert not result["valid"]
    assert any("dataset_source.builder not importable" in i for i in result["issues"])


def test_snapshot_records_dataset_builder(tmp_path: Path):
    import tcip_store as ts
    from tcip_mcp.pipelines.model_build import snapshot_manifest_key, snapshot_model_source

    config = {"data": {"dataset_source": DATASET_SOURCE}}
    manifest = snapshot_model_source(config, tmp_path)
    assert manifest is not None
    assert manifest["dataset_builder"] == "tests.test_dataset_source_seam:build_bespoke_ds"
    assert any(e["src"] == __file__ and len(e["sha256"]) == 64
               for e in manifest["files"])
    saved = ts.read(snapshot_manifest_key(tmp_path))
    assert saved["dataset_builder"] == manifest["dataset_builder"]


def test_dataset_source_key_has_one_home():
    """Structural (AST-only, no import of the module under test): the modules that once spelled
    the ``dataset_source`` config key as a bare literal now read it only through the one
    constant, ``model_build.DATASET_SOURCE_KEY``, or through a shared predicate built over it.

    Checked in Load context only, so the seventh site (``training_tools.py``'s
    ``kw["dataset_source"] = ...``, whose left side names ``build_dataset``'s parameter, not a
    config key) is untouched by design. A bare ``grep`` is not this test: the literal
    legitimately survives in docstrings, parameter names and one function name across these
    modules. The literal scan covers every load shape the key could still hide behind: a
    ``.get(``/``.pop(``/``.setdefault(`` call, a subscript, or an ``in``/``not in`` membership
    test. A reader importing the constant under a local re-spelling instead of the real one
    would still pass the absence half alone, so the second half requires a genuine
    ``from tcip_mcp.pipelines.model_build import DATASET_SOURCE_KEY``, not just a same-named
    local variable or an import from anywhere else. ``datasets.py``, ``subprocess_worker.py`` and
    ``label_queries.py`` are checked for absence only: whether a run's targets came from the
    platform's own admission does not turn on the presence of a bespoke builder, since the
    producer admits a bespoke run's samples too.
    """
    import ast
    from pathlib import Path

    import tcip_mcp

    src_root = Path(tcip_mcp.__file__).resolve().parent
    files = {
        "pipelines/data/datasets.py": src_root / "pipelines" / "data" / "datasets.py",
        "pipelines/data/label_queries.py": src_root / "pipelines" / "data" / "label_queries.py",
        "pipelines/model_build.py": src_root / "pipelines" / "model_build.py",
        "pipelines/training/subprocess_worker.py":
            src_root / "pipelines" / "training" / "subprocess_worker.py",
        "tools/training_tools.py": src_root / "tools" / "training_tools.py",
    }
    imports_it = {"tools/training_tools.py"}
    uses_it = imports_it | {"pipelines/model_build.py"}

    for name, path in files.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        literal_loads = [
            node for node in ast.walk(tree)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("get", "pop", "setdefault") and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "dataset_source")
            or (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load)
                and isinstance(node.slice, ast.Constant) and node.slice.value == "dataset_source")
            or (isinstance(node, ast.Compare)
                and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops)
                and isinstance(node.left, ast.Constant) and node.left.value == "dataset_source")
        ]
        assert not literal_loads, f"{name} still reads a raw 'dataset_source' literal"

        if name in uses_it:
            loaded = any(isinstance(n, ast.Name) and n.id == "DATASET_SOURCE_KEY"
                        and isinstance(n.ctx, ast.Load) for n in ast.walk(tree))
            assert loaded, f"{name} never loads DATASET_SOURCE_KEY"
        if name in imports_it:
            imported = any(isinstance(n, ast.ImportFrom)
                           and n.module == "tcip_mcp.pipelines.model_build"
                           and any(a.name == "DATASET_SOURCE_KEY" for a in n.names)
                           for n in ast.walk(tree))
            assert imported, f"{name} reads DATASET_SOURCE_KEY without importing it from model_build"
