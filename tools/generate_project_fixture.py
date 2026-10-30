"""Write served project listings, corrections and removals as a fixture file.

All are produced by the platform itself.

In a scratch session (``tools/fixture_session.py``), two projects are created through
``tests._web_fixtures.named_project`` and left unopened; the listing ``GET /api/projects`` serves
for them, the answer ``POST /api/projects/invariances`` gives for one correction of the second,
that route's refusal of a block whose rotation is out of range, the second's open answer, and,
with the second open, the refusal ``POST /api/canvas/state`` gives a push naming the first are
written. A second session
lays down the project picker's workspace (:func:`_picker`). Every answer, each refusal with its
status, is written normalized (``fixture_session.normalized``) to :data:`GENERATED_PATH`.

    python -m tools.generate_project_fixture
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests import FRONTEND_SRC
from tools.fixture_session import (
    ACTOR,
    answered,
    normalized,
    opened,
    scratch_session,
    write_fixture,
)
from tools.generate_open_fixture import SUBJECT

if TYPE_CHECKING:
    from pydantic import JsonValue

GENERATED_PATH = FRONTEND_SRC / "test" / "projectFixtures.json"

INSTANTS = frozenset({"modified"})
"""The listed time a run stamps, the project directory's filesystem stat in epoch seconds, which no
served model types as an instant."""


CAPTURES = ("2026-02-11", "2026-03-01", "2026-03-24")
"""The picker scene's capture dates: the first labeled with two subjects, the second with one of
them and published over, the newest holding no label."""


def produce() -> dict[str, JsonValue]:
    """Return the answers of :func:`_corrections` and of :func:`_picker`, each its own session."""
    return {**_corrections(), **_picker()}


def _picker() -> dict[str, JsonValue]:
    """Lay down the picker's workspace in a scratch session; return what its routes answered.

    The workspace is listed empty first (``empty_listing``). Then ``Valley farm``, its registry
    declaring ``bush`` and :data:`SUBJECT`, holds an image
    of :data:`CAPTURES`' first date labeled ``bush`` and :data:`SUBJECT`, two of the second, one
    labeled :data:`SUBJECT` with the bucket ``baseline`` published over it, and one of the newest,
    unlabeled; ``Ridge farm`` one unlabeled image of each of the first two dates; ``Empty block``
    none; ``stray`` is a directory holding no record; and ``Retired block`` is removed through
    ``POST /api/projects/remove`` (``removed``). The
    workspace is listed (``picker``), ``Valley farm`` opened and the workspace listed again
    (``picker_open``), ``Valley farm`` renamed through ``POST /api/projects/rename``
    (``renamed``), and ``Ridge farm`` then ``Empty block`` opened, the three open answers kept in
    that order (``opened``). The answers
    are normalized by :data:`INSTANTS`, each minted project id and the removal's archive read as
    its name.
    """
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.pipelines.postprocessing.export import DetectionResult
    from tcip_mcp.subject_registry import Subject, SubjectRegistry
    from tcip_mcp.workspace import ProjectRemoval
    from tcip_web.routes.projects import RemovalRequest, RenameRequest
    from tests._chain_fixtures import published
    from tests._producer_fixtures import (
        box_annotation,
        label_image,
        registry_over,
        write_image,
    )
    from tests._web_fixtures import named_project

    both_labeled, one_labeled, newest = CAPTURES
    with scratch_session() as (workspace, client):
        empty_listing = answered(client, "list_projects", 200)
        valley = named_project(workspace / "valley", "Valley farm")
        registry_over(
            valley.root,
            SubjectRegistry(
                subjects=tuple(
                    Subject(name=name, description="")
                    for name in ("bush", SUBJECT)
                )
            ),
        )
        both = write_image(image_dir(valley.root, both_labeled) / "a.jpg")
        label_image(
            both,
            [
                box_annotation(10, 10, 40, 30, subject="bush"),
                box_annotation(50, 10, 90, 30, subject=SUBJECT),
            ],
            100,
            80,
        )
        one = write_image(image_dir(valley.root, one_labeled) / "b.jpg")
        label_image(
            one, [box_annotation(10, 10, 40, 30, subject=SUBJECT)], 100, 80
        )
        published(
            valley.root,
            f"baseline/{one_labeled}",
            [
                DetectionResult(
                    image=str(one),
                    width=100,
                    height=80,
                    boxes=[(10.0, 10.0, 40.0, 30.0)],
                    scores=[0.9],
                    cap=2,
                ).model_dump()
            ],
            scope={"subject": SUBJECT},
        )
        write_image(image_dir(valley.root, one_labeled) / "c.jpg")
        write_image(image_dir(valley.root, newest) / "d.jpg")
        ridge = named_project(workspace / "ridge", "Ridge farm")
        for date in (both_labeled, one_labeled):
            write_image(image_dir(ridge.root, date) / "a.jpg")
        empty = named_project(workspace / "empty", "Empty block")
        (workspace / "stray" / ".tcip").mkdir(parents=True)
        retired = named_project(workspace / "retired", "Retired block")
        removed = answered(
            client,
            "remove_project",
            200,
            RemovalRequest(
                id=retired.id, confirm_name="Retired block", user=ACTOR
            ),
        )
        archive = ProjectRemoval.model_validate(removed)
        picker = answered(client, "list_projects", 200)
        opened_valley = opened(client, valley)
        picker_open = answered(client, "list_projects", 200)
        renamed = answered(
            client,
            "rename_project_route",
            200,
            RenameRequest(
                id=valley.id, display_name="Valley farm east", user=ACTOR
            ),
        )
        opened_projects = [
            opened_valley,
            *(opened(client, project) for project in (ridge, empty)),
        ]
        return normalized(
            {
                "empty_listing": empty_listing,
                "picker": picker,
                "opened": opened_projects,
                "picker_open": picker_open,
                "removed": removed,
                "renamed": renamed,
            },
            workspace,
            instants=INSTANTS,
            minted={
                valley.id: "<valley farm id>",
                ridge.id: "<ridge farm id>",
                empty.id: "<empty block id>",
                retired.id: "<retired block id>",
                archive.archive_name: "<retired block archive>",
            },
        )


def _corrections() -> dict[str, JsonValue]:
    """Create, list and correct the fixture projects in a scratch session; return the answers.

    The second project's invariances are corrected, the route refuses an out-of-range block, and
    the second project is opened (``corrected_open``). The answers are normalized by
    :data:`INSTANTS`, each minted project id read as its name.
    """
    from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
    from tcip_mcp.project_record import CaptureInvariances
    from tcip_web.routes.canvas import CanvasStatePayload
    from tcip_web.routes.projects import InvariancesRequest
    from tests._web_fixtures import INVARIANCES, named_project

    def invariances_with(answers: dict) -> CaptureInvariances:
        return CaptureInvariances.model_validate(
            INVARIANCES.model_dump() | answers
        )

    with scratch_session() as (workspace, client):
        other = named_project(workspace / "other", "other")
        valley = named_project(
            workspace / "valley",
            "valley",
            invariances=invariances_with(
                {"vertical_flip": False, "rotation": 5}
            ),
        )
        listing = answered(client, "list_projects", 200)

        def corrected(answers: dict) -> dict:
            return InvariancesRequest(
                id=valley.id,
                invariances=invariances_with(answers),
                user=ACTOR,
            ).model_dump(mode="json")

        correction = answered(
            client,
            "replace_invariances_route",
            200,
            corrected({"rotation": 10}),
        )
        request = corrected({})
        request["invariances"]["rotation"] = 400
        refusal = answered(client, "replace_invariances_route", 422, request)
        corrected_open = opened(client, valley)
        push = CanvasStatePayload(
            project_id=other.id,
            tab="annotate",
            image_path=str(image_dir(other.root, UNDATED_BUCKET) / "a.jpg"),
            viewport=None,
            mode="box",
            active_subject=None,
            cut_armed=False,
            dirty=False,
            user=ACTOR,
            classes=[],
            counts={},
            shapes=None,
        )
        elsewhere = answered(client, "push_canvas_state", 409, push)
        return normalized(
            {
                "listing": listing,
                "correction": correction,
                "refusal": refusal,
                "corrected_open": corrected_open,
                "project_not_open": elsewhere,
            },
            workspace,
            instants=INSTANTS,
            minted={other.id: "<other id>", valley.id: "<valley id>"},
        )


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))
