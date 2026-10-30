"""Write the inference routes' answers as a fixture file, produced by the platform itself.

In a scratch session (``tools/fixture_session.py``), a project created through
``tests._web_fixtures.named_project`` and opened through the picker's route holds one capture of
:data:`IMAGES` images (``tests._producer_fixtures.write_image``), a registered checkpoint
(``tests._verified_checkpoint_fixtures.project_checkpoint``) and a file that is no checkpoint.
Each pass is launched through ``POST /api/inference/launch``, built through the route's request
model, and runs on the route's own worker, which holds at each pause of a
``fixture_session.Gate`` until the scene steps it on; the pass's engine
(``inference_tools.infer``) is the platform's own for every pass but the one whose bucket the
launch under :data:`HELD` answers, whose engine reports one image done and ends once canceled.

A pass over the file that is no checkpoint is listed through ``GET /api/inference/jobs`` running
before its engine starts (``failing_started``) and fails on the platform's own engine, its stream
ending on the failure (``failing_final``). The held pass is launched (``launched``), listed
running before its engine reports (``held_started``) and after (``running``) as its stream's
first frame sends it (``progress``); a second launch naming its bucket is refused
(``bucket_exists``); it is canceled through ``POST .../cancel`` while it still runs (``cancel``)
and, once it ends, listed (``canceled``) and streamed to its final frame (``final``). A launch
naming a checkpoint that does not exist is refused (``missing_checkpoint``), and a stream to a job
no one launched answers so (``unknown_job``). The answers, each refusal with its status, are
written, each minted id read as what it names (``fixture_session.normalized``), to
:data:`GENERATED_PATH`.

    python -m tools.generate_inference_fixture
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests import FRONTEND_SRC
from tools.fixture_session import (
    ACTOR,
    Gate,
    answered,
    ended,
    normalized,
    opened,
    scratch_session,
    socket_url,
    write_fixture,
)
from tools.generate_open_fixture import SUBJECT

if TYPE_CHECKING:
    from fastapi.testclient import TestClient
    from pydantic import JsonValue

GENERATED_PATH = FRONTEND_SRC / "test" / "inferenceAnswers.json"

CAPTURE = "2026-02-11"
"""The capture every pass runs over."""

IMAGES = 9
"""The images of the capture, more than one so a held pass stands part way."""

HELD = "baseline"
"""The named bucket of the pass that reports one image done and ends once it is canceled."""

FAILING = "broken"
"""The named bucket of the pass over the file that is no checkpoint."""


def _frames(client: TestClient, job_id: str, count: int) -> list[JsonValue]:
    """Return the first ``count`` frames the stream of the job ``job_id`` names sends."""
    with client.websocket_connect(
        socket_url(client, "stream_job", job_id=job_id)
    ) as ws:
        return [ws.receive_json() for _ in range(count)]


def produce() -> dict[str, JsonValue]:
    """Launch, list, stream, cancel and refuse passes in a scratch session; return the answers.

    A route refusing what it should admit, or admitting what it should refuse, raises.
    """
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools import inference_tools
    from tcip_web.routes import inference
    from tcip_web.routes._body_common import PersonPayload
    from tests._producer_fixtures import write_image
    from tests._verified_checkpoint_fixtures import project_checkpoint
    from tests._web_fixtures import named_project

    engine = inference_tools.infer
    gate = Gate()
    served: dict[str, JsonValue] = {}

    def staged(project: Path, **run: object) -> dict:
        gate.pause()
        launched = served.get("launched")
        if not isinstance(launched, dict) or run["bucket"] != launched["bucket"]:
            return engine(project, **run)  # type: ignore[arg-type]
        progress, canceled = run["progress"], run["canceled"]
        assert callable(progress) and callable(canceled), run
        progress(1, IMAGES)
        gate.pause()
        assert canceled(), "the held pass was stepped on uncanceled"
        return {}

    with (
        scratch_session() as (workspace, client),
        pytest.MonkeyPatch.context() as patch,
    ):
        root = workspace / "inference"
        opened(client, named_project(root, "Inference"))
        for i in range(IMAGES):
            write_image(image_dir(root, CAPTURE) / f"img{i}.jpg")
        checkpoint = project_checkpoint(
            root, data={"num_channels": 3, "scope": {"subject": SUBJECT}}
        )
        not_a_checkpoint = root / "model.pt"
        not_a_checkpoint.write_bytes(b"weights")
        patch.setattr(inference_tools, "infer", staged)

        def launch(bucket_name: str, status: int, path: str) -> JsonValue:
            return answered(
                client,
                "launch_inference",
                status,
                inference.LaunchInferencePayload(
                    checkpoint_path=path,
                    dataset_root=str(root),
                    date=CAPTURE,
                    bucket_name=bucket_name,
                    stated=Stated(),
                    user=ACTOR,
                ),
            )

        failing = inference.InferenceJob.model_validate(
            launch(FAILING, 200, str(not_a_checkpoint))
        ).job_id
        gate.reached()
        served["failing_started"] = answered(client, "list_jobs", 200)
        gate.step()
        ended(inference._registry, failing)
        _, served["failing_final"] = _frames(client, failing, 2)
        served["launched"] = launch(HELD, 200, checkpoint)
        held = inference.InferenceJob.model_validate(served["launched"]).job_id
        gate.reached()
        served["held_started"] = answered(client, "list_jobs", 200)
        gate.step()
        gate.reached()
        served["running"] = answered(client, "list_jobs", 200)
        (served["progress"],) = _frames(client, held, 1)
        served["bucket_exists"] = launch(HELD, 409, checkpoint)
        served["cancel"] = answered(
            client,
            "cancel_job",
            200,
            PersonPayload(user=ACTOR),
            path={"job_id": held},
        )
        gate.step()
        ended(inference._registry, held)
        served["canceled"] = answered(client, "list_jobs", 200)
        _, served["final"] = _frames(client, held, 2)
        served["missing_checkpoint"] = launch(
            FAILING, 404, str(root / "absent.pt")
        )
        (served["unknown_job"],) = _frames(client, "inf-unknown", 1)
        return normalized(
            served,
            workspace,
            minted={
                held: "<held job>",
                failing: "<failed job>",
                Path(checkpoint).parent.name: "<registered run>",
            },
        )


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))
