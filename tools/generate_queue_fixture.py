"""Write the review queue routes' answers over the open fixture's photographs, made by the platform.

In a scratch session (``tools/fixture_session.py``), the project ``tools.generate_open_fixture``
opens is laid down again by its own layout (``generate_open_fixture.captures``), a checkpoint is
registered in it (``tests._verified_checkpoint_fixtures.project_checkpoint``), and it is opened
through the picker's route. The second photograph holds a box marked complete for
:data:`~tools.generate_open_fixture.FOCUSED_SUBJECT` and the third a box not marked complete
(``tests._producer_fixtures``). ``GET /api/annotate/scorers`` answers the scorer catalog
(``scorers``), and again with :func:`typed_scorer` registered beside the platform's
(``typed_scorers``), when a launch stating :data:`REFUSED` is refused (``refused_launch``). The
photographs' capture is selected for that subject (``selection``) and a
queue launched through ``POST /api/annotate/queue/launch`` under :data:`STATED` (``launched``); its
worker runs the platform's own ranking, whose scorer holds at a ``fixture_session.Gate`` before it
scores, while ``GET /api/annotate/queue/{job_id}`` answers it running (``running``), and once
stepped on and ended it answers it completed (``completed``). The capture is selected again for
:data:`~tools.generate_open_fixture.SUBJECT`, of which no photograph is finished, and a
``diversity`` queue launched there ends failed on the scorer's own refusal (``failed``). Last, the
dataset's later capture, holding one photograph named as the first is, is selected
(``later_selection``). The answers, each minted job id read as what it names
(``fixture_session.normalized``), are written to :data:`GENERATED_PATH`.

    python -m tools.generate_queue_fixture
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests import FRONTEND_SRC
from tools.fixture_session import (
    Gate,
    answered,
    ended,
    normalized,
    opened,
    scratch_session,
    write_fixture,
)
from tools.generate_open_fixture import (
    FOCUSED_SUBJECT,
    FRAMES,
    RAW_IMAGES,
    SUBJECT,
    captures,
)

if TYPE_CHECKING:
    from pydantic import JsonValue

    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

GENERATED_PATH = FRONTEND_SRC / "test" / "queueAnswers.json"

STATED = {
    "scorer": {"name": "uncertainty"},
    "budget": 4,
    "rationale": "a sample reason the scene states",
}
"""The scorer block, budget and reason the completed queue states, sample values: the budget
holds every photograph not finished, the first among them."""

LATER = "2026-01-08"
"""The photographs' dataset's later capture (``generate_open_fixture.captures``)."""

TYPED = "typed"
"""The name :func:`typed_scorer` is registered under for the typed catalog."""

REFUSED: dict[str, JsonValue] = {
    "scorer": {"name": TYPED, "cap": "half", "unstated": 1},
    "budget": 4,
    "rationale": "a sample reason the scene states",
}
"""A launch :func:`typed_scorer`'s parameters refuse: a value its type refuses (``cap``), one it
does not take (``unstated``) and its required ones left unstated. Sample values."""


def typed_scorer(
    task: str, flag: bool, weights: list[float], cap: float | None = None
) -> object:
    """Build an uncertainty scorer for ``task``: a sample factory typing its values.

    It takes a boolean, a list of floats and an optional float, none of which it reads.
    """
    from tcip_mcp.pipelines.active_learning.scorer import UncertaintyScorer

    return UncertaintyScorer(task)


def produce() -> dict[str, JsonValue]:
    """Launch, poll and refuse review queues in a scratch session; return the answers.

    A route refusing what it should admit, or admitting what it should refuse, raises.
    """
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.pipelines.active_learning import (
        ScorerSpec,
        scorer as al_scorer,
    )
    from tcip_mcp.web_client import CaptureChoice
    from tcip_web.routes import annotate
    from tests._producer_fixtures import (
        box_annotation,
        label_image,
        mark_complete,
    )
    from tests._verified_checkpoint_fixtures import project_checkpoint
    from tests._web_fixtures import named_project

    resolve = al_scorer.resolve_scorer
    gate = Gate()

    class Held(al_scorer.BaseScorer):
        """The scorer a block resolves to, holding at the gate before it scores."""

        def __init__(self, scorer: al_scorer.BaseScorer) -> None:
            self.scorer = scorer

        def score(
            self,
            image_paths: list,
            labeled: list,
            predictor: GenericPredictor,
        ) -> list:
            gate.pause()
            return self.scorer.score(image_paths, labeled, predictor)

    def held(block: ScorerSpec, task: str) -> al_scorer.BaseScorer:
        return Held(resolve(block, task))

    with scratch_session() as (workspace, client):
        valley = named_project(workspace / "valley", "valley")
        root, date = captures(valley.root)["raw"]
        checkpoint = project_checkpoint(
            valley.root,
            data={"num_channels": 3, "scope": {"subject": FOCUSED_SUBJECT}},
        )
        opened(client, valley)

        def select(subject: str, capture: str = date) -> JsonValue:
            return answered(
                client,
                "select_dataset",
                200,
                CaptureChoice(
                    dataset_root=str(root), subject=subject, date=capture
                ),
            )

        def launch(**queue: object) -> JsonValue:
            return answered(
                client,
                "launch_priority_queue",
                200,
                annotate.LaunchPriorityQueuePayload.model_validate(
                    {"checkpoint_path": checkpoint, **queue}
                ),
            )

        def status(job_id: str) -> JsonValue:
            return answered(
                client,
                "get_priority_queue_job",
                200,
                path={"job_id": job_id},
            )

        served: dict[str, JsonValue] = {
            "scorers": answered(client, "list_scorers", 200)
        }
        with pytest.MonkeyPatch.context() as patch:
            patch.setitem(al_scorer.SCORER_REGISTRY, TYPED, typed_scorer)
            served["typed_scorers"] = answered(client, "list_scorers", 200)
            served["refused_launch"] = answered(
                client,
                "launch_priority_queue",
                422,
                {"checkpoint_path": checkpoint, **REFUSED},
            )
        served["selection"] = select(FOCUSED_SUBJECT)
        finished, in_progress = (
            image_dir(root, date) / name for name in RAW_IMAGES[1:3]
        )
        for path, frame in ((finished, FRAMES[1]), (in_progress, FRAMES[2])):
            label_image(
                path,
                [box_annotation(10, 10, 50, 50, subject=FOCUSED_SUBJECT)],
                *frame,
            )
        mark_complete(finished, FOCUSED_SUBJECT, project=valley.root)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(al_scorer, "resolve_scorer", held)
            served["launched"] = launch(**STATED)
            job_id = annotate.PriorityQueueJob.model_validate(
                served["launched"]
            ).job_id
            gate.reached()
            served["running"] = status(job_id)
            gate.step()
            ended(annotate._pq_registry, job_id)
        served["completed"] = status(job_id)

        select(SUBJECT)
        failing = annotate.PriorityQueueJob.model_validate(
            launch(**{**STATED, "scorer": {"name": "diversity"}})
        ).job_id
        ended(annotate._pq_registry, failing)
        served["failed"] = status(failing)
        served["later_selection"] = select(FOCUSED_SUBJECT, LATER)
        return normalized(
            served,
            workspace,
            minted={
                valley.id: "<valley id>",
                job_id: "<completed job>",
                failing: "<failed job>",
                Path(checkpoint).parent.name: "<registered run>",
            },
            rounded=True,
        )


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))
