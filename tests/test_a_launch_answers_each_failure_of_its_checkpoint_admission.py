"""Each launch answers every failure of its checkpoint's admission at the request, starting no job.

The inference launch and the review-queue launch admit a checkpoint through the registry's own
read (``registered_bytes``). A checkpoint path that is a directory, or that lies beneath a regular
file, answers 404 with that read's sentence; a run record that will not read answers 400 with the
refused read's sentence; a registry index that will not decode answers 409 as an unreadable
registry with the store's sentence; and a checkpoint the process may not read answers 400 with the
refused read's sentence, naming the checkpoint and its path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests._record_damage_fixtures import damage_record
from tests._verified_checkpoint_fixtures import (
    produced_checkpoint,
    read_denied,
    register_checkpoint,
)
from tests.test_review_priority_queue import QUEUE, select
from tests.test_web_inference_launch_root import (
    DATE,
    _jobs,
    _launch,
    _registered_root,
    run_record_refused,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from fastapi.testclient import TestClient
    from starlette.testclient import httpx

DOORS = ("inference", "queue")
"""The launches that admit a checkpoint: a pass's, and a review queue's."""


def _door(
    door: str, client: TestClient, project: Path
) -> tuple[Callable[[Path], httpx.Response], Callable[[], list]]:
    """Ready ``door``'s launch in ``project``; return how it is posted and how its jobs are listed.

    The inference launch runs over the capture :data:`DATE` of a registered root, and the queue
    launch over the capture the select route selects.
    """
    from tcip_web.routes.annotate import _pq_registry

    if door == "inference":
        root = _registered_root(project)
        return (
            lambda checkpoint: _launch(client, root, DATE, checkpoint),
            lambda: _jobs(client),
        )
    select(client, project, "bud")
    return (
        lambda checkpoint: client.post(
            "/api/annotate/queue/launch",
            json={"checkpoint_path": str(checkpoint), **QUEUE},
        ),
        lambda: list(_pq_registry.jobs),
    )


def _registered(project: Path) -> Path:
    """Write a checkpoint into ``project`` and register it through ``register_model``."""
    checkpoint = produced_checkpoint(project / "m.pt", "launch root")
    register_checkpoint(project, str(checkpoint), name="admission")
    return checkpoint


@pytest.mark.parametrize("door", DOORS)
def test_a_checkpoint_path_that_is_a_directory_answers_404_with_the_registry_s_sentence(
    opened_client: TestClient, tmp_path: Path, door: str
) -> None:
    """A directory named as the checkpoint answers 404 with ``registered_bytes``'s sentence."""
    from tcip_mcp.model_registry import (
        CheckpointNotFoundError,
        registered_bytes,
    )

    launch, jobs = _door(door, opened_client, tmp_path)
    directory = tmp_path / "weights.pt"
    directory.mkdir()
    with pytest.raises(CheckpointNotFoundError) as missing:
        registered_bytes(directory, project=tmp_path)
    before = jobs()

    answer = launch(directory)

    assert answer.status_code == 404, answer.text
    assert answer.json()["detail"] == str(missing.value)
    assert jobs() == before


@pytest.mark.parametrize("door", DOORS)
def test_a_checkpoint_path_beneath_a_regular_file_answers_404_with_the_registry_s_sentence(
    opened_client: TestClient, tmp_path: Path, door: str
) -> None:
    """A checkpoint named beneath a regular file answers 404 with ``registered_bytes``'s sentence.

    The regular file is a checkpoint a run completed, so nothing can be at a path beneath it.
    """
    from tcip_mcp.model_registry import (
        CheckpointNotFoundError,
        registered_bytes,
    )

    launch, jobs = _door(door, opened_client, tmp_path)
    beneath = (
        produced_checkpoint(tmp_path / "m.pt", "launch root") / "weights.pt"
    )
    with pytest.raises(CheckpointNotFoundError) as missing:
        registered_bytes(beneath, project=tmp_path)
    before = jobs()

    answer = launch(beneath)

    assert answer.status_code == 404, answer.text
    assert answer.json()["detail"] == str(missing.value)
    assert jobs() == before


@pytest.mark.parametrize("door", DOORS)
def test_a_run_record_that_will_not_read_answers_400_with_the_refused_read(
    opened_client: TestClient, tmp_path: Path, door: str
) -> None:
    """A run of the project whose record will not decode answers 400 with its refusal, no job."""
    launch, jobs = _door(door, opened_client, tmp_path)
    checkpoint = _registered(tmp_path)
    refusal = run_record_refused(tmp_path)
    before = jobs()

    answer = launch(checkpoint)

    assert answer.status_code == 400, answer.text
    assert answer.json()["detail"] == refusal
    assert jobs() == before


@pytest.mark.parametrize("door", DOORS)
def test_a_registry_index_that_will_not_decode_answers_409_as_an_unreadable_registry(
    opened_client: TestClient, tmp_path: Path, door: str
) -> None:
    """A registry index given bytes that will not decode answers 409 with the store's sentence.

    The index is written by a legitimate registration, then given bytes that are no JSON in its
    root's database: fault injection, since the registry writes only an index that decodes.
    """
    import tcip_store
    from tcip_mcp.model_registry import (
        read_registry_index,
        registry_index_key,
    )

    launch, jobs = _door(door, opened_client, tmp_path)
    checkpoint = _registered(tmp_path)
    damage_record(registry_index_key(tmp_path), b"{ not json")
    with pytest.raises(tcip_store.DecodeError) as undecoded:
        read_registry_index(tmp_path)
    before = jobs()

    answer = launch(checkpoint)

    assert answer.status_code == 409, answer.text
    assert answer.json()["detail"] == f"registry unreadable: {undecoded.value}"
    assert jobs() == before


@pytest.mark.parametrize("door", DOORS)
def test_a_checkpoint_the_process_may_not_read_answers_400_naming_the_checkpoint(
    opened_client: TestClient, tmp_path: Path, door: str
) -> None:
    """A registered checkpoint the operating system denies a read of answers 400, no job.

    The answer carries the refusal ``registered_bytes`` raises under the same denial, naming the
    checkpoint and its path. Skipped where the platform lets this process read the file anyway (a
    POSIX superuser).
    """
    from tcip_mcp.model_registry import registered_bytes
    from tcip_store import ReadRefusedError

    launch, jobs = _door(door, opened_client, tmp_path)
    checkpoint = _registered(tmp_path)
    before = jobs()
    with read_denied(checkpoint):
        answer = launch(checkpoint)
        with pytest.raises(ReadRefusedError) as refused:
            registered_bytes(checkpoint, project=tmp_path)

    assert answer.status_code == 400, answer.text
    assert answer.json()["detail"] == str(refused.value)
    assert str(checkpoint) in answer.json()["detail"]
    assert jobs() == before
