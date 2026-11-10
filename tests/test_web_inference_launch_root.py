"""The inference launch answers a dataset root or checkpoint its pass refuses at the request.

A launch over a capture whose root was never registered, whose identity document does not decode,
or whose directory is no capture answers 400 with the sentence the tool's pass refuses with, and
starts no job; so does a launch over a checkpoint the open project's registry does not name, the
root's refusal answering first when both are refused. A missing checkpoint file or capture
directory keeps its 404 and comes first. A registered root and checkpoint still launch a pass that
publishes its bucket.
"""

from __future__ import annotations

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
    register_checkpoint,
)
from tests._web_fixtures import bound_to
from tests.test_a_refused_record_reads_as_its_readers_sentence import (
    _rewritten,
)
from tools.fixture_session import ended

if TYPE_CHECKING:
    from pathlib import Path

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


def test_the_launch_and_the_tool_refuse_a_root_never_registered_with_one_sentence(
    opened: OpenProject,
    opened_client: TestClient,
    tmp_path: Path,
    checkpoint: Path,
    deferred_worker: None,
) -> None:
    """The route's detail and the tool's error, each its own door's answer, are one sentence."""
    root = _unregistered_root(tmp_path)
    capture = image_dir(root, UNDATED_BUCKET)

    through_the_route = _launch(
        opened_client, root, UNDATED_BUCKET, checkpoint
    )
    through_the_tool = called_through_the_server(
        "run_inference",
        {
            "checkpoint_path": str(checkpoint),
            "images_dir": str(capture),
            "bucket": f"run/{UNDATED_BUCKET}",
            "stated": STATED,
        },
        bound=bound_to(opened),
    )

    assert through_the_route.status_code == 400
    assert through_the_route.json()["detail"] == unregistered_refusal(root)
    assert through_the_tool == {"error": through_the_route.json()["detail"]}


def test_a_launch_names_the_missing_checkpoint_or_capture_before_the_root(
    opened_client: TestClient, tmp_path: Path, checkpoint: Path
) -> None:
    """A checkpoint with no file and a capture with no directory keep their 404 over any root.

    The root here was never registered, and the 404 comes first so a mistyped location is told
    what is missing and not to register it.
    """
    root = _unregistered_root(tmp_path)

    no_checkpoint = _launch(
        opened_client, root, UNDATED_BUCKET, tmp_path / "absent.pt"
    )
    no_capture = _launch(opened_client, root, "2030-01-01", checkpoint)

    assert no_checkpoint.status_code == 404, no_checkpoint.text
    assert "checkpoint not found" in no_checkpoint.json()["detail"]
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


def test_the_launch_and_the_tool_refuse_an_unregistered_checkpoint_with_one_sentence(
    opened: OpenProject,
    opened_client: TestClient,
    tmp_path: Path,
    checkpoint: Path,
    deferred_worker: None,
) -> None:
    """The route's detail and the tool's error over one unregistered checkpoint are one sentence.

    The capture is under a registered root, so the checkpoint is what each door refuses.
    """
    root = _registered_root(tmp_path)

    through_the_route = _launch(opened_client, root, DATE, checkpoint)
    through_the_tool = called_through_the_server(
        "run_inference",
        {
            "checkpoint_path": str(checkpoint),
            "images_dir": str(image_dir(root, DATE)),
            "bucket": f"run/{DATE}",
            "stated": STATED,
        },
        bound=bound_to(opened),
    )

    assert through_the_route.status_code == 400, through_the_route.text
    assert through_the_tool == {"error": through_the_route.json()["detail"]}
    assert "register_model" in through_the_route.json()["detail"]


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
