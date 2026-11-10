"""Registering a root whose identity document is gone restores the one id its records cite.

What cites a root is each entry of the project's registry at the root's path and each bucket
published under it. ``register_dataset`` over a root with no ``dataset.json`` restores the one id
they cite, saying so and from what; refuses, naming each id and what cites it, when they cite
more than one, writing nothing; and mints only for a root nothing cites. A refused document takes
its id by the same rule, and ``check_dataset_identity`` names the repair the registration makes.
A bucket record that will not read raises its reader's refusal at both doors.

The steps made by hand are fault injection: removing or damaging a root's ``dataset.json``, a
change to the project's files outside the platform, and rewriting or damaging a published bucket's
record, which no producer does.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pytest

import tcip_store
from tcip_mcp.dataset_layout import (
    dataset_identity_path,
    image_dir,
    read_dataset_identity,
)
from tcip_mcp.registry_paths import stored_path
from tcip_mcp.tools.project_tools import (
    DatasetEntry,
    check_dataset_identity,
    read_datasets,
    register_dataset,
    upsert_dataset,
)
from tests._mapping_fixtures import CROP
from tests._producer_fixtures import (
    box_annotation,
    bucket_of_another_dataset,
    label_image,
    left_written,
    published_document,
    register_dataset_at,
    write_image,
)
from tests._web_fixtures import new_project
from tests.test_a_refused_record_reads_as_its_readers_sentence import (
    _rewritten,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from fastapi.testclient import TestClient

    type Conflict = tuple[Path, Path, dict[str, list[str]]]
    """A project, a root with its identity in place, and each id cited beside what cites it."""

REGISTRY = "this project's registry at this path"
"""How a refusal names a registry entry at the root's path as what cites an id."""

REMEDY = "to one identity outside the platform"
"""The words ending the remedy a root cited by more than one id is answered with."""


def _bucket_witness(name: str) -> str:
    """Return how a refusal names the bucket ``name`` under the root as what cites an id."""
    return f"this root's bucket {name!r}"


def _cited(dataset_id: str, witnesses: list[str]) -> str:
    """Return the clause naming ``dataset_id`` beside each of ``witnesses``, as a refusal does."""
    return f"id {dataset_id!r} is cited by {', '.join(witnesses)}"


def _names_each_citation_and_the_remedy(
    answer: str, cited: dict[str, list[str]]
) -> None:
    """Assert ``answer`` names each id with its own witnesses, and the remedy over every witness.

    The remedy names no registration.
    """
    for dataset_id, witnesses in cited.items():
        assert _cited(dataset_id, witnesses) in answer, answer
    assert REMEDY in answer, answer
    remedy = answer.rsplit(", so bring ", 1)[1]
    for witnesses in cited.values():
        for witness in witnesses:
            assert witness in remedy, answer
    assert "register_dataset" not in remedy, answer


def _published(project: Path, root: Path, *names: str) -> list[str]:
    """Publish one bucket of ``root``'s one image per entry of ``names`` for ``project``.

    The image carries an empty label document, so the root yields a fingerprint, and the root is
    registered at :data:`~tests._mapping_fixtures.CROP` by the first publication. Returns
    ``names``.
    """
    image = write_image(image_dir(root, "2026-01-01") / "a.png")
    label_image(image, [], 100, 80)
    for name in names:
        published_document(
            project, name, image, [box_annotation(1, 1, 9, 9)], 100, 80
        )
    return list(names)


def _register(project: Path, root: Path) -> dict:
    """Return ``register_dataset``'s answer for ``root`` in ``project`` at the sample crop."""
    return register_dataset(project, str(root), CROP, actor=None)


def _footprint(project: Path, root: Path) -> dict[str, object]:
    """Return what a registration of ``root`` in ``project`` can write, beside ``left_written``."""
    return {
        **left_written(project, root),
        "identity": tcip_store.read_blob_versioned(
            dataset_identity_path(root), default=None
        ).value,
        "registry": read_datasets(project),
    }


def test_a_root_whose_identity_is_gone_takes_back_the_id_its_registry_entry_holds(
    tmp_path: Path,
) -> None:
    """The registration answers the registered id and says it restored it from the registry.

    The registry still holds one entry at the root's path, and the document written carries the id.
    """
    root = tmp_path / "ds"
    write_image(image_dir(root, "2026-01-01") / "a.png")
    registered = register_dataset_at(tmp_path, root, crop=CROP)
    dataset_identity_path(root).unlink()

    restored = _register(tmp_path, root)

    assert "error" not in restored, restored
    assert restored["id"] == registered
    assert str(dataset_identity_path(root)) in restored["repaired"]
    assert _cited(registered, [REGISTRY]) in restored["repaired"]
    assert read_datasets(tmp_path) == [
        DatasetEntry(id=registered, path=stored_path(root, tmp_path))
    ]
    identity = read_dataset_identity(root)
    assert identity is not None and identity.id == registered


def test_a_copy_made_without_its_identity_takes_the_id_its_buckets_record(
    tmp_path: Path,
) -> None:
    """A root copied with its store and without ``dataset.json`` is the dataset its buckets name.

    No registry entry is at the copy's path; the answer names the bucket it read the id from, and
    the registry's entry of that id moves to the copy.
    """
    root = tmp_path / "ds"
    (name,) = _published(tmp_path, root, "first/2026-01-01")
    recorded = read_dataset_identity(root)
    assert recorded is not None
    copy = tmp_path / "ds-copy"
    tcip_store.release_root(root)
    shutil.copytree(root, copy, ignore=shutil.ignore_patterns("dataset.json"))

    restored = _register(tmp_path, copy)

    assert "error" not in restored, restored
    assert restored["id"] == recorded.id
    assert _cited(recorded.id, [_bucket_witness(name)]) in restored["repaired"]
    assert read_datasets(tmp_path) == [
        DatasetEntry(id=recorded.id, path=stored_path(copy, tmp_path))
    ]


def _two_buckets_of_two_ids(project: Path) -> Conflict:
    """Return a root whose two buckets record two ids, registered in a project holding neither.

    The returned project is a second one beside ``project``, whose registry holds nothing.
    """
    root = project / "ds"
    first, second = _published(
        project, root, "first/2026-01-01", "second/2026-01-01"
    )
    identity = read_dataset_identity(root)
    assert identity is not None
    other = bucket_of_another_dataset(root, second)
    return (
        new_project(project.parent / "other").root,
        root,
        {
            identity.id: [_bucket_witness(first)],
            other: [_bucket_witness(second)],
        },
    )


def _registry_and_bucket_of_two_ids(project: Path) -> Conflict:
    """Return a root the project's registry names by one id and whose one bucket records another."""
    root = project / "ds"
    (name,) = _published(project, root, "first/2026-01-01")
    identity = read_dataset_identity(root)
    assert identity is not None
    other = bucket_of_another_dataset(root, name)
    return (
        project,
        root,
        {identity.id: [REGISTRY], other: [_bucket_witness(name)]},
    )


def _two_registry_entries_at_one_path(
    project: Path,
) -> Conflict:
    """Return a root the project's registry holds under two ids, the second written by upsert."""
    from tcip_mcp.project_record import mint_id

    root = project / "ds"
    write_image(image_dir(root, "2026-01-01") / "a.png")
    first = register_dataset_at(project, root, crop=CROP)
    second = mint_id()
    upsert_dataset(
        project, DatasetEntry(id=second, path=stored_path(root, project))
    )
    return project, root, {first: [REGISTRY], second: [REGISTRY]}


@pytest.mark.parametrize(
    "conflicting",
    [
        _two_buckets_of_two_ids,
        _registry_and_bucket_of_two_ids,
        _two_registry_entries_at_one_path,
    ],
    ids=["two_buckets", "registry_and_bucket", "two_registry_entries"],
)
@pytest.mark.parametrize("damaged", [False, True], ids=["absent", "damaged"])
def test_a_root_whose_identity_is_gone_and_whose_records_cite_two_ids_refuses(
    tmp_path: Path, conflicting: Callable[[Path], Conflict], damaged: bool
) -> None:
    """The refusal names each id beside what cites it, and the remedy over those records.

    The document is removed, or damaged so its model refuses it. The refusal writes no identity
    document, changes no registry and appends no audit line, and its remedy names no
    registration.
    """
    project, root, cited = conflicting(tmp_path)
    if damaged:
        _rewritten(dataset_identity_path(root), {"crop": 5})
    else:
        dataset_identity_path(root).unlink()
    before = _footprint(project, root)

    refused = _register(project, root)

    assert set(refused) == {"error"}, refused
    _names_each_citation_and_the_remedy(refused["error"], cited)
    assert _footprint(project, root) == before


@pytest.mark.parametrize(
    "conflicting",
    [_two_buckets_of_two_ids, _registry_and_bucket_of_two_ids],
    ids=["two_buckets", "registry_and_bucket"],
)
def test_the_check_of_a_refused_identity_at_a_root_cited_by_two_ids_names_them_and_the_remedy(
    tmp_path: Path, conflicting: Callable[[Path], Conflict]
) -> None:
    """``check_dataset_identity`` answers ``refused``, naming the ids, their records and the remedy.

    The reason is the one the registration refuses with, over a damaged document.
    """
    project, root, cited = conflicting(tmp_path)
    _rewritten(dataset_identity_path(root), {"crop": 5})

    checked = check_dataset_identity(project, str(root))

    assert checked.state == "refused", checked
    assert checked.reason is not None
    _names_each_citation_and_the_remedy(checked.reason, cited)


def _damaged_bucket(project: Path) -> tuple[Path, str]:
    """Return a registered root and the name of its one bucket, whose record will not decode."""
    from tcip_mcp.dataset_layout import bucket_key
    from tests._record_damage_fixtures import damage_record

    root = project / "ds"
    (name,) = _published(project, root, "first/2026-01-01")
    damage_record(bucket_key(root, name), b"{not json")
    return root, name


@pytest.mark.parametrize("damaged", [False, True], ids=["absent", "damaged"])
def test_a_registration_over_an_unreadable_bucket_raises_its_readers_refusal(
    tmp_path: Path, damaged: bool
) -> None:
    """A bucket record that will not read is refused naming the bucket, and nothing is written.

    The identity document is removed, or damaged so its model refuses it; no document is written
    and the registry is unchanged.
    """
    from tcip_mcp.buckets import NotABucketError

    root, name = _damaged_bucket(tmp_path)
    if damaged:
        _rewritten(dataset_identity_path(root), {"crop": 5})
    else:
        dataset_identity_path(root).unlink()
    identity = tcip_store.read_blob_versioned(
        dataset_identity_path(root), default=None
    ).value
    registry = read_datasets(tmp_path)

    with pytest.raises(NotABucketError) as refused:
        _register(tmp_path, root)

    assert repr(name) in str(refused.value), refused.value
    assert (
        tcip_store.read_blob_versioned(
            dataset_identity_path(root), default=None
        ).value
        == identity
    )
    assert read_datasets(tmp_path) == registry


def test_the_check_of_a_refused_identity_over_an_unreadable_bucket_raises_its_readers_refusal(
    tmp_path: Path,
) -> None:
    """The check over a damaged document raises the bucket reader's refusal, naming the bucket."""
    from tcip_mcp.buckets import NotABucketError

    root, name = _damaged_bucket(tmp_path)
    _rewritten(dataset_identity_path(root), {"crop": 5})

    with pytest.raises(NotABucketError) as refused:
        check_dataset_identity(tmp_path, str(root))

    assert repr(name) in str(refused.value), refused.value


def test_a_refused_identity_at_a_root_another_project_registered_is_repaired_from_its_buckets(
    tmp_path: Path,
) -> None:
    """The check names the repair, and the registration in a project not holding the root makes it.

    The root's bucket records its id; the document is damaged, and a second project whose registry
    holds nothing checks and registers it.
    """
    root = tmp_path / "ds"
    (name,) = _published(tmp_path, root, "first/2026-01-01")
    identity = read_dataset_identity(root)
    assert identity is not None
    other = new_project(tmp_path.parent / "other").root
    _rewritten(dataset_identity_path(root), {"crop": 5})

    checked = check_dataset_identity(other, str(root))
    repaired = _register(other, root)

    assert checked.state == "refused"
    assert checked.reason is not None
    assert "register_dataset on this root" in checked.reason, checked.reason
    assert "backup" not in checked.reason, checked.reason
    assert "error" not in repaired, repaired
    assert repaired["id"] == identity.id
    assert name in repaired["repaired"]
    assert read_datasets(other) == [
        DatasetEntry(id=identity.id, path=stored_path(root, other))
    ]


def test_an_assessed_delivery_refused_for_a_missing_identity_delivers_validated_once_restored(
    client: TestClient, tmp_path: Path
) -> None:
    """The per-image count tool and route refuse the missing identity, then deliver validated.

    The bucket was published under a passed assessment; with ``dataset.json`` removed both doors
    answer the missing identity, and after the registration restores the id each delivers
    validated under the assessment that covered it before.
    """
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from tcip_mcp.delivery import DeliveryRefusal, read_delivery_events
    from tcip_mcp.pipelines.delivery_events_schema import ShippedValidated
    from tcip_mcp.tools.inference_tools import deliver_per_image_counts
    from tcip_web.routes.results import EXPORT_HEADER, CountExport
    from tests import _trait_fixtures as fx
    from tests._chain_fixtures import run_the_chain
    from tests._mapping_fixtures import COUNT_CROP
    from tests._producer_fixtures import unregistered_refusal
    from tests._web_fixtures import open_new_project
    from tests.test_tcip_web_results_routes import _export_count, _per_image

    chain = run_the_chain(tmp_path, experiment_id="exp-restored")
    open_new_project(tmp_path)
    bucket = chain.read()

    def tool() -> dict:
        return deliver_per_image_counts(
            tmp_path,
            str(chain.root),
            chain.bucket,
            str(tmp_path / "counts.csv"),
            trait=fx.COUNT_TRAIT,
            actor=None,
        )

    dataset_identity_path(chain.root).unlink()
    missing = unregistered_refusal(chain.root)
    refused_route = _export_count(client, _per_image(bucket))

    assert tool() == {"error": missing}
    assert refused_route.status_code == 400, refused_route.text[:300]
    assert (
        DeliveryRefusal.model_validate(refused_route.json()["detail"]).message
        == missing
    )

    assert (
        register_dataset_at(tmp_path, chain.root, crop=COUNT_CROP)
        == bucket.dataset_id
    )
    delivered = tool()
    route = _export_count(client, _per_image(bucket))

    assert "error" not in delivered, delivered
    assert delivered["shipped"] == {"kind": "validated"}
    assert route.status_code == 200, route.text[:300]
    assert (
        CountExport.model_validate_json(route.headers[EXPORT_HEADER]).shipped
        == ShippedValidated()
    )
    events = read_delivery_events(tmp_path)
    assert [e.buckets[0].assessment_id for e in events] == [
        chain.assessment["assessment_id"]
    ] * 2
