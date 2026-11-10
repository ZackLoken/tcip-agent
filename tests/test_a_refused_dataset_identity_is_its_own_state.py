"""A dataset identity its model refuses reads as refused, apart from one that is absent.

The identity is damaged through the store's own writer after ``register_dataset`` wrote it. The
check answers ``refused`` for the root and for a registered path, ``unregistered`` and ``gone``
staying the absent document; ``register_dataset`` repairs the document at a root the project's
registry holds, from the registry's id, and refuses at one nothing cites; ``launch_training`` over
a refused identity answers the error naming the document.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import tcip_store
from tcip_mcp.dataset_layout import (
    UNDATED_BUCKET,
    dataset_identity_path,
    image_dir,
)
from tcip_mcp.tools.project_tools import (
    check_dataset_identity,
    read_datasets,
    register_dataset,
)
from tests._web_fixtures import new_project
from tests.test_a_refused_record_reads_as_its_readers_sentence import (
    _rewritten,
)
from tests.test_check_dataset_identity_script import (
    _real_dataset,
    _registered,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_the_check_of_a_refused_identity_answers_refused_and_names_the_repair(
    tmp_path: Path,
) -> None:
    """The check over a damaged identity answers ``refused``, naming the path and the repair.

    The root is registered, so the reason names ``register_dataset`` as the repair.
    """
    project, root, _ = _registered(tmp_path)
    path = dataset_identity_path(root)
    _rewritten(path, {"crop": 5})

    checked = check_dataset_identity(project, str(root))

    assert checked.state == "refused"
    assert checked.reason is not None
    assert str(path) in checked.reason
    assert "register_dataset on this root" in checked.reason
    assert "backup" not in checked.reason


def test_the_check_of_a_refused_identity_at_an_unregistered_root_names_the_restore(
    tmp_path: Path,
) -> None:
    """A damaged identity at a root the project's registry does not hold reads as a restore.

    The root was registered to another project, so this project's registry holds no entry for it.
    """
    _, root, _ = _registered(tmp_path)
    other = new_project(tmp_path / "other").root
    _rewritten(dataset_identity_path(root), {"crop": 5})

    checked = check_dataset_identity(other, str(root))

    assert checked.state == "refused"
    assert checked.reason is not None
    assert "backup" in checked.reason
    assert "register_dataset on this root" not in checked.reason


def test_a_registered_path_holding_a_refused_identity_reads_refused_not_gone(
    tmp_path: Path,
) -> None:
    """A registered path reads ``refused`` over a damaged document, ``gone`` over an absent one."""
    project, root, _ = _registered(tmp_path)
    moved = tmp_path / "moved"
    moved.mkdir()
    _real_dataset(moved)
    tcip_store.put_blob(
        dataset_identity_path(moved),
        tcip_store.read_blob_versioned(dataset_identity_path(root)).value,
        expect=tcip_store.read_blob_versioned(
            dataset_identity_path(moved), default=None
        ).version,
    )
    _rewritten(dataset_identity_path(root), {"crop": 5})

    (row,) = check_dataset_identity(project, str(moved)).registered

    assert row.path == str(root)
    assert row.state == "refused"
    assert row.unreadable is not None
    assert str(dataset_identity_path(root)) in row.unreadable

    dataset_identity_path(root).unlink()
    (absent,) = check_dataset_identity(project, str(moved)).registered
    assert absent.state == "gone"


def test_register_dataset_repairs_a_damaged_identity_at_a_registered_root(
    tmp_path: Path,
) -> None:
    """Registering a registered root over a damaged document rewrites it with the registry's id.

    The check then answers ``unchanged`` and the registry holds the one entry it held.
    """
    project, root, registered = _registered(tmp_path)
    entries = read_datasets(project)
    _rewritten(dataset_identity_path(root), {"crop": 5})

    repaired = register_dataset(project, str(root), "chestnut", actor=None)

    assert "error" not in repaired, repaired
    assert repaired["id"] == registered["id"]
    assert str(dataset_identity_path(root)) in repaired["repaired"]
    assert check_dataset_identity(project, str(root)).state == "unchanged"
    assert read_datasets(project) == entries


def test_register_dataset_refuses_a_damaged_identity_at_a_root_the_registry_does_not_hold(
    tmp_path: Path,
) -> None:
    """A damaged document at a root this registry lacks stays; the refusal names the restore."""
    _, root, _ = _registered(tmp_path)
    other = new_project(tmp_path / "other").root
    path = dataset_identity_path(root)
    _rewritten(path, {"crop": 5})
    damaged = path.read_bytes()

    refused = register_dataset(other, str(root), "chestnut", actor=None)

    assert str(path) in refused["error"]
    assert "backup" in refused["error"]
    assert path.read_bytes() == damaged
    assert read_datasets(other) == []


def test_the_check_answers_an_os_error_at_the_roots_identity_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ``OSError`` reading the root's identity is answered, as the per-path read answers it."""
    project, root, _ = _registered(tmp_path)
    path = dataset_identity_path(root)
    read = tcip_store.read_blob_versioned

    def unreadable(
        target: Path, **kwargs: bytes | None
    ) -> tcip_store.Versioned:
        if target == path:
            raise OSError("device not ready")
        return read(target, **kwargs)

    monkeypatch.setattr(tcip_store, "read_blob_versioned", unreadable)

    checked = check_dataset_identity(project, str(root))

    assert checked.state == "unregistered"
    assert checked.reason is not None
    assert "device not ready" in checked.reason


def test_a_launch_over_a_refused_dataset_identity_answers_the_error_naming_its_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launch over a dataset identity its model refuses answers the error dict naming its path.

    No run directory is opened.
    """
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from tcip_mcp.experiments import run_dirs
    from tcip_mcp.tools import training_tools
    from tests._producer_fixtures import (
        fake_popen,
        register_dataset_at,
        seed_bud_images,
        small_detection_config,
    )

    images_dir = image_dir(tmp_path, UNDATED_BUCKET)
    seed_bud_images(images_dir, n=2, size=64, box=(1, 1, 9, 9))
    fake_popen(monkeypatch, [])
    register_dataset_at(tmp_path, tmp_path, crop="currant")
    path = dataset_identity_path(tmp_path)
    _rewritten(path, {"crop": 5})

    result = training_tools.launch_training(
        tmp_path,
        small_detection_config(images_dir, project=tmp_path),
        actor=None,
    )

    assert isinstance(result, dict)
    assert result["error"].startswith("launch_training: ")
    assert str(path) in result["error"]
    assert run_dirs(tmp_path) == []
