"""The inference launch answers a dataset root or checkpoint its pass refuses at the request.

The launch and the tool admit a pass through one admission, the root and then the checkpoint, and
answer each of its refusals with one sentence. A launch over a capture whose root was never
registered, whose identity document does not decode, or whose directory is no capture answers 400
with the sentence the tool's pass refuses with, and starts no job; so does a launch over a
checkpoint the open project's registry does not name, the root's refusal answering first when both
are refused. A checkpoint that is no file answers 404 after the root's admission; a capture
directory that does not exist answers 404 before it. A registered root and checkpoint still launch
a pass that publishes its bucket.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import tcip_store
from tcip_mcp.dataset_layout import (
    UNDATED_BUCKET,
    bucket_key,
    dataset_identity_path,
    image_dir,
    parse_capture_dir,
    require_dataset_identity,
)
from tcip_web.jobstore import Completed
from tests._mapping_fixtures import CROP
from tests._producer_fixtures import (
    gray_frame,
    ingested_unregistered,
    left_written,
    register_dataset_at,
    unregistered_refusal,
    write_image,
)
from tests._server_calls import called_through_the_server
from tests._verified_checkpoint_fixtures import (
    SAMPLE_DETECTOR_PASS,
    produced_checkpoint,
    read_denied,
    register_checkpoint,
    registered_checkpoint,
    unsearchable,
)
from tests._web_fixtures import bound_to
from tests.test_a_refused_record_reads_as_its_readers_sentence import (
    _rewritten,
)
from tools.fixture_session import ended

if TYPE_CHECKING:
    from fastapi.testclient import TestClient
    from starlette.testclient import httpx

    from tcip_web.state import OpenProject

DATE = "2026-02-11"
"""The capture every registered root here holds."""

STATED = {"tile": False, **SAMPLE_DETECTOR_PASS}
"""What every launch states of its execution record."""


@pytest.fixture
def deferred_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the inference worker with one that does nothing, so a launch's job stays as made."""
    from tcip_web.routes import inference

    monkeypatch.setattr(inference, "_worker", lambda job: None)


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    """Write a checkpoint a run completed under another root into the project; return its path.

    The project's registry does not name it until a test registers it.
    """
    return produced_checkpoint(tmp_path / "m.pt", "launch root")


def _launch(
    client: TestClient, root: Path, capture: str, checkpoint: Path | str
) -> httpx.Response:
    """Post a launch of the capture ``capture`` of ``root`` under ``checkpoint``."""
    return client.post(
        "/api/inference/launch",
        json={
            "user": "tester",
            "checkpoint_path": str(checkpoint),
            "dataset_root": str(root),
            "date": capture,
            "bucket_name": "run",
            "stated": STATED,
        },
    )


def _jobs(client: TestClient) -> list[dict]:
    """Return the jobs the launch route lists for the open project."""
    listed = client.get("/api/inference/jobs")
    assert listed.status_code == 200, listed.text
    return listed.json()["jobs"]


def _unregistered_root(tmp_path: Path) -> Path:
    """Ingest one image into ``tmp_path / "field"`` and register nothing; return the root."""
    raw = tmp_path / "raw"
    gray_frame(raw, 100)
    ingested_unregistered(tmp_path / "field", raw)
    return tmp_path / "field"


def _registered_root(tmp_path: Path) -> Path:
    """Write one image into the capture :data:`DATE` of ``tmp_path / "orchard"``, registered."""
    root = tmp_path / "orchard"
    write_image(image_dir(root, DATE) / "a.jpg", (100, 100), (110, 110, 110))
    register_dataset_at(tmp_path, root, crop=CROP)
    return root


def test_a_launch_over_a_root_never_registered_refuses_naming_it_and_starts_no_job(
    opened_client: TestClient,
    tmp_path: Path,
    checkpoint: Path,
    deferred_worker: None,
) -> None:
    """A launch over a root never registered answers 400 with the root's refusal and no job.

    The answer is the sentence ``require_dataset_identity`` refuses the root with, over a
    checkpoint the registry does not name either, no job is listed and nothing is written; once
    the root and the checkpoint are registered, the same launch is admitted.
    """
    root = _unregistered_root(tmp_path)
    before = left_written(tmp_path, root)

    refused = _launch(opened_client, root, UNDATED_BUCKET, checkpoint)

    assert refused.status_code == 400, refused.text
    assert refused.json()["detail"] == unregistered_refusal(root)
    assert _jobs(opened_client) == []
    assert left_written(tmp_path, root) == before
    register_dataset_at(tmp_path, root, crop=CROP)
    register_checkpoint(tmp_path, str(checkpoint), name="launch-root")
    admitted = _launch(opened_client, root, UNDATED_BUCKET, checkpoint)
    assert admitted.status_code == 200, admitted.text
    assert [j["job_id"] for j in _jobs(opened_client)] == [
        admitted.json()["job_id"]
    ]


def run_record_refused(project: Path) -> str:
    """Open a run in ``project`` whose record will not decode; return the registry's refusal.

    The run is opened by the launcher's own producer (``opened_run``), and its ``run.json`` is
    then rewritten through the store's blob interface with bytes that are no JSON: fault
    injection, since the producer writes only a record that decodes. Returns the sentence listing
    the project's registry (``registered_entries``) refuses with.
    """
    from tcip_mcp.experiments import RUN_FILE
    from tcip_mcp.model_registry import registered_entries
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        opened_run,
    )

    record = (
        opened_run(project, detection_config(project / "ds", project=project))
        / RUN_FILE
    )
    tcip_store.put_blob(
        record,
        b"{ not json",
        expect=tcip_store.read_blob_versioned(record).version,
    )
    with pytest.raises(tcip_store.ReadRefusedError) as refused:
        registered_entries(project)
    return str(refused.value)


ADMISSION_CASES = {
    "root_never_registered": (False, "unregistered", None, 400),
    "unregistered_checkpoint": (True, "unregistered", None, 400),
    "missing_checkpoint": (True, "missing", None, 404),
    "missing_checkpoint_under_a_root_never_registered": (
        False,
        "missing",
        None,
        400,
    ),
    "checkpoint_is_a_directory": (True, "directory", None, 404),
    "checkpoint_beneath_a_regular_file": (True, "beneath a file", None, 404),
    "identity_does_not_decode": (True, "registered", "identity", 400),
    "run_record_refused": (True, "registered", "run record", 400),
    "checkpoint_read_denied": (True, "registered", "checkpoint read", 400),
    "admitted": (True, "registered", None, 200),
}
"""Each admission case by id: whether its root is registered, what its checkpoint is (one the
registry does not name, one a run of the project completed, no file, a directory, a path beneath
a regular file), the fault applied after a legitimate write (a record damaged, or the checkpoint's
read held denied), and the launch's status."""


@pytest.mark.parametrize(
    ("root_registered", "checkpoint_is", "damaged", "status"),
    list(ADMISSION_CASES.values()),
    ids=list(ADMISSION_CASES),
)
def test_the_launch_and_the_tool_answer_each_admission_case_alike(
    opened: OpenProject,
    opened_client: TestClient,
    tmp_path: Path,
    checkpoint: Path,
    deferred_worker: None,
    root_registered: bool,
    checkpoint_is: str,
    damaged: str | None,
    status: int,
) -> None:
    """The route's answer and ``run_inference``'s through the server agree on each case.

    A refused launch's detail is the tool's error, sentence for sentence; an admitted launch's
    job and the tool's dry run name one dataset root. The route's status is the case's.
    """
    root, capture = (
        (_registered_root(tmp_path), DATE)
        if root_registered
        else (_unregistered_root(tmp_path), UNDATED_BUCKET)
    )
    named = {
        "unregistered": lambda: checkpoint,
        "registered": lambda: Path(registered_checkpoint(tmp_path)),
        "missing": lambda: tmp_path / "absent.pt",
        "directory": lambda: tmp_path / "weights.pt",
        "beneath a file": lambda: checkpoint / "weights.pt",
    }[checkpoint_is]()
    if checkpoint_is == "directory":
        named.mkdir()
    if damaged == "identity":
        _rewritten(dataset_identity_path(root), {"crop": 5})
    if damaged == "run record":
        run_record_refused(tmp_path)

    with read_denied(named) if damaged == "checkpoint read" else nullcontext():
        through_the_route = _launch(opened_client, root, capture, named)
        through_the_tool = called_through_the_server(
            "run_inference",
            {
                "checkpoint_path": str(named),
                "images_dir": str(image_dir(root, capture)),
                "bucket": f"run/{capture}",
                "stated": STATED,
                "dry_run": True,
            },
            bound=bound_to(opened),
        )

    assert through_the_route.status_code == status, through_the_route.text
    assert isinstance(through_the_tool, dict), through_the_tool
    if status == 200:
        assert (
            through_the_tool.get("dataset_root")
            == through_the_route.json()["dataset_root"]
        ), through_the_tool
    else:
        assert through_the_tool == {
            "error": through_the_route.json()["detail"]
        }


def test_the_tool_answers_a_checkpoint_in_a_directory_the_process_may_not_search_naming_it(
    opened: OpenProject, tmp_path: Path
) -> None:
    """A registered checkpoint whose directory the operating system denies a search of is named.

    ``registered_bytes`` refuses an inspection the operating system denies, naming the checkpoint
    and its path, and ``run_inference`` answers that sentence as its error. The checkpoint lies in
    a directory of its own, since the project's own directory cannot be made unsearchable.
    Skipped where the platform lets this process inspect the checkpoint anyway (a POSIX
    superuser, or a platform that gives a directory no search permission to remove).
    """
    from tcip_mcp.model_registry import registered_bytes

    root = _registered_root(tmp_path)
    checkpoint = produced_checkpoint(
        tmp_path / "weights" / "m.pt", "launch root"
    )
    register_checkpoint(tmp_path, str(checkpoint), name="inspection")
    with unsearchable(checkpoint):
        with pytest.raises(tcip_store.ReadRefusedError) as refused:
            registered_bytes(checkpoint, project=tmp_path)
        through_the_tool = called_through_the_server(
            "run_inference",
            {
                "checkpoint_path": str(checkpoint),
                "images_dir": str(image_dir(root, DATE)),
                "bucket": f"run/{DATE}",
                "stated": STATED,
                "dry_run": True,
            },
            bound=bound_to(opened),
        )

    assert through_the_tool == {"error": str(refused.value)}
    assert str(checkpoint) in str(refused.value)


def test_a_launch_names_a_missing_capture_before_the_root_and_the_root_before_the_checkpoint(
    opened_client: TestClient, tmp_path: Path, checkpoint: Path
) -> None:
    """A capture with no directory keeps its 404 over any root; a missing checkpoint does not.

    The root here was never registered. A launch over a capture that does not exist is told what
    is missing and not to register it; a launch over a checkpoint with no file is told the root's
    refusal, the order ``run_inference`` admits in.
    """
    root = _unregistered_root(tmp_path)

    no_checkpoint = _launch(
        opened_client, root, UNDATED_BUCKET, tmp_path / "absent.pt"
    )
    no_capture = _launch(opened_client, root, "2030-01-01", checkpoint)

    assert no_checkpoint.status_code == 400, no_checkpoint.text
    assert no_checkpoint.json()["detail"] == unregistered_refusal(root)
    assert no_capture.status_code == 404, no_capture.text
    assert "images_dir not found" in no_capture.json()["detail"]
    assert _jobs(opened_client) == []


def test_a_launch_over_an_identity_document_that_does_not_decode_refuses_with_its_sentence(
    opened_client: TestClient,
    tmp_path: Path,
    checkpoint: Path,
    deferred_worker: None,
) -> None:
    """A registered root whose identity document no longer decodes answers 400, no job.

    The damage is fault injection after a legitimate registration, since registration writes
    only a document that decodes: the stored document's crop is replaced through the store by a
    value the model refuses. The answer carries the sentence the read refuses with.
    """
    root = _registered_root(tmp_path)
    _rewritten(dataset_identity_path(root), {"crop": 5})
    with pytest.raises(ValueError) as read:
        require_dataset_identity(root)
    assert str(dataset_identity_path(root)) in str(read.value), read.value
    before = left_written(tmp_path, root)

    refused = _launch(opened_client, root, DATE, checkpoint)

    assert refused.status_code == 400, refused.text
    assert refused.json()["detail"] == str(read.value)
    assert _jobs(opened_client) == []
    assert left_written(tmp_path, root) == before


def test_a_launch_over_a_checkpoint_the_registry_does_not_name_refuses_and_starts_no_job(
    opened: OpenProject,
    opened_client: TestClient,
    tmp_path: Path,
    checkpoint: Path,
    deferred_worker: None,
) -> None:
    """A launch over a checkpoint the open project never registered answers 400, no job.

    The answer is the sentence the registry's load refuses the checkpoint with, naming its path,
    its digest and ``register_model``; no job is listed and nothing is written.
    """
    from tcip_mcp.model_registry import (
        UnregisteredCheckpointError,
        load_registered_checkpoint,
    )

    root = _registered_root(tmp_path)
    with pytest.raises(UnregisteredCheckpointError) as unregistered:
        load_registered_checkpoint(checkpoint, project=opened.root)
    before = left_written(tmp_path, root)

    refused = _launch(opened_client, root, DATE, checkpoint)

    assert refused.status_code == 400, refused.text
    assert refused.json()["detail"] == str(unregistered.value)
    assert str(checkpoint) in refused.json()["detail"]
    assert "register_model" in refused.json()["detail"]
    assert _jobs(opened_client) == []
    assert left_written(tmp_path, root) == before


def _capture_named_images(
    tmp_path: Path, *, registered: tuple[str, ...]
) -> Path:
    """Write an image into the capture ``images`` of ``tmp_path / "grove"``; return the root.

    Each of ``registered`` ("root", "tree") is registered at :data:`~tests._mapping_fixtures.CROP`:
    the root, and the directory ``images`` inside it, which a reading of the capture ``images``
    as the image tree of a root named ``images`` would make a root.
    """
    root = tmp_path / "grove"
    write_image(
        image_dir(root, "images") / "a.jpg", (100, 100), (110, 110, 110)
    )
    for place in registered:
        register_dataset_at(
            tmp_path, root if place == "root" else root / "images", crop=CROP
        )
    return root


def test_a_capture_named_images_under_a_root_carrying_state_alone_launches(
    opened_client: TestClient,
    tmp_path: Path,
    checkpoint: Path,
    deferred_worker: None,
) -> None:
    """A capture the name ``images`` is admitted when its root alone carries state of its own."""
    root = _capture_named_images(tmp_path, registered=("root",))
    register_checkpoint(tmp_path, str(checkpoint), name="launch-root")

    launched = _launch(opened_client, root, "images", checkpoint)

    assert launched.status_code == 200, launched.text
    assert parse_capture_dir(launched.json()["images_dir"])[0] == root


@pytest.mark.parametrize(
    ("registered", "sentence"),
    [
        (("tree",), "is not a capture"),
        ((), "neither carries"),
        (("root", "tree"), "both carry"),
    ],
    ids=["inner_tree_alone", "neither", "both"],
)
def test_a_capture_named_images_the_directory_layout_refuses_launches_no_job(
    opened_client: TestClient,
    tmp_path: Path,
    checkpoint: Path,
    deferred_worker: None,
    registered: tuple[str, ...],
    sentence: str,
) -> None:
    """A directory named ``images`` that is no capture or reads two ways answers 400 and no job.

    The answer carries the sentence ``parse_capture_dir`` refuses the directory with.
    """
    root = _capture_named_images(tmp_path, registered=registered)
    with pytest.raises(ValueError, match=sentence) as layout:
        parse_capture_dir(image_dir(root, "images"))

    refused = _launch(opened_client, root, "images", checkpoint)

    assert refused.status_code == 400, refused.text
    assert refused.json()["detail"] == str(layout.value)
    assert _jobs(opened_client) == []


def test_a_launch_over_a_registered_root_runs_its_pass_to_a_published_bucket(
    opened_client: TestClient, tmp_path: Path
) -> None:
    """A launch over a registered root answers its job, which the real worker ends succeeded.

    The checkpoint is a trained detector's, registered in the project by its completion, and no
    model is stood in for: the pass publishes the capture's bucket.
    """
    from tcip_web.routes.inference import _registry
    from tests._chain_fixtures import DATE as TRAINED_DATE, trained_detector

    root, _images, _selection, trained = trained_detector(
        tmp_path, experiment_id="exp-launch-root"
    )

    launched = _launch(opened_client, root, TRAINED_DATE, trained)

    assert launched.status_code == 200, launched.text
    job = ended(_registry, launched.json()["job_id"])
    assert job.state == Completed(), job.state
    assert tcip_store.exists(bucket_key(root, job.bucket))
