"""Path confinement while another workspace project's dataset registry does not read.

The open project is ``<workspace>/project``, its dataset registered outside the workspace, so
only a registry read admits a path under it. Another project, ``<workspace>/another``, is listed
ahead of it, its own dataset registered outside the workspace too, and its registry is damaged by
fault injection: bytes that will not decode, a value its model refuses, a database that will not
open, or a store another connection holds.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pytest

import tcip_store
from tcip_mcp.dataset_layout import image_dir
from tcip_mcp.tools.project_tools import (
    dataset_entry_path,
    dataset_registry_key,
    read_datasets,
)
from tcip_mcp.workspace import project_dirs
from tcip_web.state import store
from tests._busy_store_fixtures import (
    SHORT_WAIT_S,
    held_by_another_connection,
    waiting,
)
from tests._mapping_fixtures import CROP
from tests._producer_fixtures import register_dataset_at, write_image
from tests._record_damage_fixtures import damage_record, unopenable_database
from tests._web_fixtures import new_project, open_new_project

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from fastapi.testclient import TestClient
    from starlette.testclient import httpx

DATE = "2026-02-11"
"""The capture every dataset here holds its one image under."""

NO_CAPTURE = "2026-03-01"
"""A capture no dataset here holds, so a launch admitted past confinement answers 404."""

DISPLAY_PIXELS = 3840 * 2160
"""A sample display's device-pixel count, a 4K screen's, far above any image here."""

UNDER_NO_ROOT = "under no root that read"
"""The phrase a refusal of a path under no root that read carries."""


class Arranged(NamedTuple):
    """The damaged project, the open project's external root and the damaged project's own."""

    damaged: Path
    admitted: Path
    unreached: Path


@pytest.fixture
def outside(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a fresh temporary directory ``tmp_path_factory.mktemp("outside")`` creates."""
    return tmp_path_factory.mktemp("outside")


@pytest.fixture
def arranged(tmp_path: Path, outside: Path) -> Arranged:
    """Open ``tmp_path`` and make ``another`` beside it, each with a dataset outside the workspace.

    Each dataset root holds one image, ``images/<DATE>/a.jpg``, and is registered through
    ``register_dataset``.
    """
    open_new_project(tmp_path)
    admitted = _dataset(outside / "open-data")
    register_dataset_at(tmp_path, admitted, crop=CROP)
    damaged = new_project(tmp_path.parent / "another").root
    unreached = _dataset(outside / "their-data")
    register_dataset_at(damaged, unreached, crop=CROP)
    assert project_dirs(store.workspace)[0] == damaged
    return Arranged(damaged, admitted, unreached)


def _dataset(root: Path) -> Path:
    """Write ``root``'s one image (:func:`_image`) and return ``root``."""
    write_image(_image(root), (8, 8))
    return root


def _image(root: Path) -> Path:
    """Return the one image a dataset root here holds."""
    return image_dir(root, DATE) / "a.jpg"


def _refusal(project: Path) -> str:
    """Return the sentence reading ``project``'s registry and resolving its entries raises."""
    with pytest.raises((tcip_store.StoreError, ValueError)) as refused:
        [dataset_entry_path(project, e) for e in read_datasets(project)]
    return str(refused.value)


@contextmanager
def _undecodable(project: Path) -> Iterator[str]:
    """Put bytes that will not decode in ``project``'s registry; yield the reader's refusal."""
    damage_record(dataset_registry_key(project), b"{not json")
    yield _refusal(project)


@contextmanager
def _refused_by_its_model(project: Path) -> Iterator[str]:
    """Replace ``project``'s registry with a value that is no list; yield the reader's refusal."""
    tcip_store.replace(dataset_registry_key(project), {"datasets": "none"})
    yield _refusal(project)


@contextmanager
def _unopenable(project: Path) -> Iterator[str]:
    """Put bytes that are no database over ``project``'s; yield the reader's refusal inside."""
    with unopenable_database(project):
        yield _refusal(project)


@contextmanager
def _busy(project: Path) -> Iterator[str]:
    """Hold ``project``'s store past a short wait; yield the store's busy phrase."""
    tcip_store.release_root(project)
    with waiting(SHORT_WAIT_S), held_by_another_connection(project):
        yield "was busy"


DAMAGE = {
    "undecodable": _undecodable,
    "refused_by_its_model": _refused_by_its_model,
    "unopenable": _unopenable,
    "busy": _busy,
}


def _label_read(client: TestClient, root: Path) -> httpx.Response:
    """Read the label document of ``root``'s one image."""
    return client.get(
        "/api/annotate/labels", params={"image_path": str(_image(root))}
    )


def _image_read(client: TestClient, root: Path) -> httpx.Response:
    """Read ``root``'s one image for a display of :data:`DISPLAY_PIXELS`."""
    return client.get(
        "/api/images",
        params={"path": str(_image(root)), "display_pixels": DISPLAY_PIXELS},
    )


def _launch(client: TestClient, root: Path) -> httpx.Response:
    """Launch inference over ``root`` from a checkpoint under it, for a capture it lacks."""
    return client.post(
        "/api/inference/launch",
        json={
            "checkpoint_path": str(root / "model.pt"),
            "dataset_root": str(root),
            "date": NO_CAPTURE,
            "bucket_name": "baseline",
            "user": "breeder",
            "stated": {},
        },
    )


ROUTES: dict[str, tuple[Callable[[TestClient, Path], httpx.Response], int]] = {
    "label_read": (_label_read, 200),
    "image_read": (_image_read, 200),
    "launch": (_launch, 404),
}
"""Each route that confines a path, beside what it answers a path confinement admits."""


def _assert_refused_as_unread(
    resp: httpx.Response, *named: tuple[Path, str]
) -> None:
    """Assert ``resp`` is the 409 of a path under no root that read, naming each refusal given."""
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail.startswith("registry unreadable: "), detail
    assert UNDER_NO_ROOT in detail, detail
    assert "outside the allowed roots" not in detail, detail
    for project, refusal in named:
        assert str(project) in detail, detail
        assert refusal in detail, detail


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("damage", DAMAGE)
def test_a_path_under_a_root_that_reads_is_answered_as_before_the_damage(
    client: TestClient, arranged: Arranged, damage: str, route: str
) -> None:
    request, admitted_status = ROUTES[route]
    before = request(client, arranged.admitted)
    assert before.status_code == admitted_status, before.text

    with DAMAGE[damage](arranged.damaged):
        after = request(client, arranged.admitted)

    assert (after.status_code, after.content) == (
        before.status_code,
        before.content,
    )


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("damage", DAMAGE)
def test_a_path_under_no_root_that_reads_is_answered_409_naming_the_unread_registry(
    client: TestClient, arranged: Arranged, damage: str, route: str
) -> None:
    request, _admitted_status = ROUTES[route]
    with DAMAGE[damage](arranged.damaged) as refusal:
        resp = request(client, arranged.unreached)

    _assert_refused_as_unread(resp, (arranged.damaged, refusal))


def test_a_refused_entry_after_a_valid_one_leaves_none_of_its_projects_roots(
    client: TestClient, arranged: Arranged
) -> None:
    """A registry whose second entry is refused admits nothing under its valid first entry.

    The second entry's path carries a ``..`` segment, replaced through the store (fault
    injection); the open project's own external root is still admitted.
    """
    key = dataset_registry_key(arranged.damaged)
    stored = tcip_store.read(key)
    assert isinstance(stored, list) and len(stored) == 1, stored
    tcip_store.replace(
        key, [*stored, {"id": "zz-escape", "path": "../escape"}]
    )
    refusal = _refusal(arranged.damaged)

    _assert_refused_as_unread(
        _label_read(client, arranged.unreached), (arranged.damaged, refusal)
    )
    assert _label_read(client, arranged.admitted).status_code == 200


def test_two_unread_registries_are_each_named(
    client: TestClient, arranged: Arranged
) -> None:
    second = new_project(arranged.damaged.parent / "beside").root
    register_dataset_at(second, second, crop=CROP)

    with _undecodable(arranged.damaged) as first_refusal:
        with _undecodable(second) as second_refusal:
            resp = _label_read(client, arranged.unreached)

    _assert_refused_as_unread(
        resp,
        (arranged.damaged, first_refusal),
        (second, second_refusal),
    )


def test_a_path_under_a_configured_image_root_is_admitted_beside_an_unread_registry(
    client: TestClient, arranged: Arranged, outside: Path
) -> None:
    archive = _dataset(outside / "archive")
    store.configure(store.workspace, (archive,))

    with _undecodable(arranged.damaged):
        resp = _image_read(client, archive)

    assert resp.status_code == 200, resp.text


def test_a_staging_tree_is_refused_403_as_outside_beside_an_unread_registry(
    client: TestClient, arranged: Arranged
) -> None:
    staged = _dataset(store.workspace / ".imports" / "run-1")

    with _undecodable(arranged.damaged):
        resp = _label_read(client, staged)

    assert resp.status_code == 403, resp.text
    assert "outside the allowed roots" in resp.json()["detail"]


def test_an_unread_registry_still_contributes_its_project_and_is_named_once(
    arranged: Arranged,
) -> None:
    from tcip_web.paths import allowed_roots

    with _undecodable(arranged.damaged) as refusal:
        roots, unread = allowed_roots()

    assert arranged.damaged in roots
    assert tcip_store.canonical_path(arranged.unreached) not in {
        tcip_store.canonical_path(root) for root in roots
    }
    (sentence,) = unread
    assert str(arranged.damaged) in sentence
    assert refusal in sentence
