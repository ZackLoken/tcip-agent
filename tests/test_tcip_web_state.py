"""Tests for tcip_web.state.StateStore: versioning, flush targeting, rehydrate."""

from __future__ import annotations

import asyncio
from pathlib import Path

from tcip_web.state import DatasetSelection, GuiState, StateStore


def test_version_increments_on_mutate() -> None:
    store = StateStore()
    assert store.version == 0
    asyncio.run(store.mutate({"active_tab": "review"}))
    assert store.version == 1
    asyncio.run(store.mutate({"mode": "polygon"}))
    assert store.version == 2


def test_flush_targets_current_project_not_stale(tmp_path: Path) -> None:
    # Regression: the debounced save used to capture the destination dir at schedule
    # time, so a project switch during the debounce window wrote the new project's
    # snapshot into the old project's gui.json. The flush must resolve the dir from
    # the *current* state.
    store = StateStore()
    proj_a = tmp_path / "A"
    proj_b = tmp_path / "B"

    store._state = GuiState(dataset=DatasetSelection(project_root=str(proj_a)))
    # ...project switches before the pending flush runs...
    store._state = GuiState(dataset=DatasetSelection(project_root=str(proj_b)))
    store._flush_sync()

    assert (proj_b / ".tcip" / "state" / "gui.json").exists()
    assert not (proj_a / ".tcip" / "state" / "gui.json").exists()


def test_load_from_disk_roundtrip(tmp_path: Path) -> None:
    store = StateStore()
    store._state = GuiState(
        active_tab="results",
        dataset=DatasetSelection(project_root=str(tmp_path), dataset_root=str(tmp_path / "ds")),
    )
    store._flush_sync()

    # A fresh store simulates a backend restart.
    restarted = StateStore()
    assert restarted.load_from_disk(tmp_path) is True
    assert restarted.state.active_tab == "results"
    assert restarted.state.dataset.dataset_root == str(tmp_path / "ds")


def test_load_from_disk_missing_returns_false(tmp_path: Path) -> None:
    assert StateStore().load_from_disk(tmp_path) is False
