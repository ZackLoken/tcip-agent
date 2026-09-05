"""The ``dataset_source`` bespoke seam, mirroring ``model_source``.

Proves the dataset layer is no longer a closed task registry: an agent-supplied importable
builder produces a torch ``Dataset`` for a new task, ``build_dataset`` routes to it, the known
loaders stay the default, and the builder source is snapshotted for provenance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import Dataset  # noqa: E402


class _CountingDataset(Dataset):
    """A trivially-bespoke dataset: one item per stem, echoing the context it was built with."""

    def __init__(self, *, stems=None, marker: str = "", **_ignored):
        self.stems = list(stems or ["a", "b", "c"])
        self.marker = marker

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        return torch.zeros(3, 8, 8), {"stem": self.stems[idx]}


def build_bespoke_ds(**kwargs) -> _CountingDataset:
    """Agent-authored dataset builder (importable, no exec)."""
    return _CountingDataset(**kwargs)


DATASET_SOURCE = {
    "builder": "tests.test_dataset_source_seam:build_bespoke_ds",
    "builder_kwargs": {"marker": "bespoke"},
    "source_files": [__file__],
    "task": "grape_bunch_count",
}


def test_build_dataset_routes_to_dataset_source():
    from tcip_mcp.pipelines.data.datasets import build_dataset

    # A new task the known loaders don't cover: routes to the bespoke builder, not the registry.
    ds = build_dataset("grape_bunch_count", dataset_source=DATASET_SOURCE,
                        stems=["s0", "s1"])
    assert isinstance(ds, _CountingDataset)
    assert ds.stems == ["s0", "s1"]          # runtime context threaded through
    assert ds.marker == "bespoke"            # builder_kwargs applied
    assert ds.expected_channels == 3         # channel default stamped when the builder set none


def test_known_task_registry_stays_the_default():
    from tcip_mcp.pipelines.data.datasets import build_dataset

    # No dataset_source -> the closed-registry KeyError stays the honest signal for a bad name.
    with pytest.raises(ValueError, match="Unknown task"):
        build_dataset("grape_bunch_count")


def test_builder_kwargs_win_over_context():
    from tcip_mcp.pipelines.data.datasets import build_from_dataset_source

    ds = build_from_dataset_source(
        {"builder": "tests.test_dataset_source_seam:build_bespoke_ds",
         "builder_kwargs": {"marker": "pinned"}},
        marker="context")
    assert ds.marker == "pinned"

    with pytest.raises(ValueError, match="builder_kwargs must be a dict"):
        build_from_dataset_source(
            {"builder": "tests.test_dataset_source_seam:build_bespoke_ds",
             "builder_kwargs": [1, 2]})


def test_preflight_accepts_dataset_source_without_image_dirs(tmp_path: Path):
    from tcip_mcp.tools.training_tools import preflight_config

    config = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detector",
                         "builder_kwargs": {"gt_boxes_wh": [(10, 10)], "num_classes": 1},
                         "task": "grape_bunch_count"},
        "data": {"dataset_source": DATASET_SOURCE, "task": "grape_bunch_count"},
        "training": {"batch_size": 1},
    }
    # No images_dir/labels_dir: the bespoke builder owns loading, so those are not required.
    result = preflight_config(config, smoke=False)
    assert result["valid"], result["issues"]

    # A non-importable builder is caught honestly.
    bad = {**config, "data": {"dataset_source": {"builder": "no.such:fn"}}}
    result = preflight_config(bad, smoke=False)
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
    local variable or an import from anywhere else. ``datasets.py`` no longer reads the key at
    all now that the constant has moved out of it, so it is checked for absence only.
    ``subprocess_worker.py`` no longer reads the key itself either: its registry-derived check
    now goes through ``pipelines.data.label_queries.targets_registry_derived``, the one predicate
    the inference-side door's own remedy calls too, so it is checked here for the absence of a
    raw literal only, not for loading the constant; ``label_queries.py``, where that predicate
    lives, is the module that imports and loads it now.
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
    imports_it = {"pipelines/data/label_queries.py", "tools/training_tools.py"}
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
