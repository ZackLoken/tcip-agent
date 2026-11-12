"""A registration's audit line says which identity it gave the root and how it chose it.

The ``register_dataset`` line carries the id the registration wrote (``dataset_id``), whether it
minted that id (``minted``), and, when it restored one, its ``repaired`` sentence, each the
registration's own answer; a line with an id not minted and no ``repaired`` is a kept id. The
line's arguments stay the call's ``dataset_root`` and ``crop``.

The steps made by hand are fault injection: removing or damaging a root's ``dataset.json``, a
change to the project's files outside the platform, and a competing registration landing inside
the store's identity write.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Any

import pytest

import tcip_store
from tcip_mcp.dataset_layout import (
    dataset_identity_path,
    image_dir,
    read_dataset_identity,
)
from tcip_mcp.tools.project_tools import register_dataset
from tests._audit_fixtures import audit_rows
from tests._mapping_fixtures import CROP
from tests._producer_fixtures import register_dataset_at, write_image
from tests._web_fixtures import new_project
from tests.test_a_refused_record_reads_as_its_readers_sentence import (
    _rewritten,
)
from tests.test_registering_a_dataset_restores_or_refuses import _published

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path


def _imaged(tmp_path: Path) -> Path:
    """Return a root under ``tmp_path`` holding one image and no identity document."""
    root = tmp_path / "ds"
    write_image(image_dir(root, "2026-01-01") / "a.png")
    return root


def _never_registered(tmp_path: Path) -> tuple[Path, Path]:
    """Return the project ``tmp_path`` and a root it never registered."""
    return tmp_path, _imaged(tmp_path)


def _registered(tmp_path: Path) -> tuple[Path, Path]:
    """Return the project ``tmp_path`` and a root it registered."""
    root = _imaged(tmp_path)
    register_dataset_at(tmp_path, root, crop=CROP)
    return tmp_path, root


def _gone_from_a_registered_root(tmp_path: Path) -> tuple[Path, Path]:
    """Return the project ``tmp_path`` and a root it registered, its identity document removed."""
    project, root = _registered(tmp_path)
    dataset_identity_path(root).unlink()
    return project, root


def _copied_without_its_identity(tmp_path: Path) -> tuple[Path, Path]:
    """Return the project ``tmp_path`` and a copy of a published root without ``dataset.json``."""
    root = tmp_path / "ds"
    _published(tmp_path, root, "first/2026-01-01")
    tcip_store.release_root(root)
    copy = tmp_path / "ds-copy"
    shutil.copytree(root, copy, ignore=shutil.ignore_patterns("dataset.json"))
    return tmp_path, copy


def _refused_at_a_root_another_project_registered(
    tmp_path: Path,
) -> tuple[Path, Path]:
    """Return a project whose registry holds nothing and a published root, its document damaged."""
    root = tmp_path / "ds"
    _published(tmp_path, root, "first/2026-01-01")
    other = new_project(tmp_path.parent / "other").root
    _rewritten(dataset_identity_path(root), {"crop": 5})
    return other, root


def _registration_and_its_line(
    project: Path, root: Path
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Register ``root`` in ``project`` at the sample crop; return the answer and its one line."""
    before = len(audit_rows(root, "register_dataset"))
    answer = register_dataset(project, str(root), CROP, actor=None)
    assert "error" not in answer, answer
    (line,) = audit_rows(root, "register_dataset")[before:]
    return answer, line


@pytest.mark.parametrize(
    ("made", "minted", "restored"),
    [
        (_never_registered, True, False),
        (_registered, False, False),
        (_gone_from_a_registered_root, False, True),
        (_copied_without_its_identity, False, True),
        (_refused_at_a_root_another_project_registered, False, True),
    ],
    ids=[
        "minted",
        "kept",
        "restored_from_the_registry",
        "restored_from_a_bucket",
        "refused_document_repaired",
    ],
)
def test_the_line_carries_the_id_written_whether_it_was_minted_and_the_repair(
    tmp_path: Path,
    made: Callable[[Path], tuple[Path, Path]],
    minted: bool,
    restored: bool,
) -> None:
    """The line's facts are the answer's, its id the one the identity document reads.

    Its arguments are the call's ``dataset_root`` and ``crop`` and nothing else.
    """
    project, root = made(tmp_path)
    identity_before = read_dataset_identity(root) if not restored else None

    answer, line = _registration_and_its_line(project, root)

    identity = read_dataset_identity(root)
    assert identity is not None
    assert line.get("minted") is minted
    assert answer["minted"] is minted
    assert line["dataset_id"] == answer["id"] == identity.id
    assert ("repaired" in answer) is restored
    assert ("repaired" in line) is restored
    assert line.get("repaired") == answer.get("repaired")
    if identity_before is not None:
        assert identity.id == identity_before.id
    assert line["arguments"] == {"dataset_root": str(root), "crop": CROP}


def test_a_registration_whose_write_loses_to_a_competing_one_records_the_winners_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The landed pass's answer and line: the winner's id, not minted.

    Over an absent document, a competing registration of the same root lands between the read
    and the identity write; the write is refused as a lost compare-and-set and the next pass keeps
    the id that committed. The competing registration's line, minted, lands first.
    """
    root = _imaged(tmp_path)
    real_put = tcip_store.put_blob
    raced: list[Path] = []
    competing: list[dict[str, Any]] = []

    def put_after_a_competitor(
        path: Path, data: bytes, **kwargs: tcip_store.Version | None
    ) -> tcip_store.Version:
        if path.name == dataset_identity_path(root).name and not raced:
            raced.append(path)
            competing.append(
                register_dataset(tmp_path, str(root), CROP, actor=None)
            )
        return real_put(path, data, **kwargs)

    monkeypatch.setattr(tcip_store, "put_blob", put_after_a_competitor)

    answer = register_dataset(tmp_path, str(root), CROP, actor=None)

    (winner,) = competing
    assert winner.get("minted") is True, winner
    assert answer.get("minted") is False, answer
    assert answer["id"] == winner["id"]
    first, landed = audit_rows(root, "register_dataset")
    assert first.get("dataset_id") == winner["id"]
    assert first.get("minted") is True
    assert landed.get("dataset_id") == winner["id"]
    assert landed.get("minted") is False
    assert "repaired" not in landed
