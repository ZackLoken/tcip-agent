"""D4 — the ``dataset_source`` bespoke seam, mirroring ``model_source``.

Proves the dataset layer is no longer a closed task registry: an agent-supplied importable
builder produces a torch ``Dataset`` for a NEW task, ``build_dataset`` routes to it, the known
loaders stay the default, and the builder source is snapshotted for provenance.
"""

from __future__ import annotations

import json
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

    # A NEW task the known loaders don't cover — routes to the bespoke builder, not the registry.
    ds = build_dataset("grape_bunch_count", dataset_source=DATASET_SOURCE,
                        stems=["s0", "s1"])
    assert isinstance(ds, _CountingDataset)
    assert ds.stems == ["s0", "s1"]          # runtime context threaded through
    assert ds.marker == "bespoke"            # builder_kwargs applied
    assert ds.expected_channels == 3         # channel default stamped when the builder set none


def test_known_task_registry_stays_the_default():
    from tcip_mcp.pipelines.data.datasets import build_dataset

    # No dataset_source -> the closed-registry KeyError stays the honest signal for a bad NAME.
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
    # No images_dir/labels_dir — the bespoke builder owns loading, so those are not required.
    result = preflight_config(config, smoke=False)
    assert result["valid"], result["issues"]

    # A non-importable builder is caught honestly.
    bad = {**config, "data": {"dataset_source": {"builder": "no.such:fn"}}}
    result = preflight_config(bad, smoke=False)
    assert not result["valid"]
    assert any("dataset_source.builder not importable" in i for i in result["issues"])


def test_snapshot_records_dataset_builder(tmp_path: Path):
    from tcip_mcp.pipelines.model_build import snapshot_model_source

    config = {"data": {"dataset_source": DATASET_SOURCE}}
    manifest = snapshot_model_source(config, tmp_path)
    assert manifest is not None
    assert manifest["dataset_builder"] == "tests.test_dataset_source_seam:build_bespoke_ds"
    assert any(e["src"] == __file__ and len(e["sha256"]) == 64
               for e in manifest["files"])
    saved = json.loads((tmp_path / "model_src" / "manifest.json").read_text())
    assert saved["dataset_builder"] == manifest["dataset_builder"]
