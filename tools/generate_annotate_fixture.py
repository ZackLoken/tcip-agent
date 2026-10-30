"""Write the annotate routes' answers over the open fixture's photographs, produced by the platform.

In a scratch session (``tools/fixture_session.py``), the project ``tools.generate_open_fixture``
opens is laid down again by its own layout (``generate_open_fixture.captures``) and opened through
the picker's route, and its capture of photographs is selected. Saves through
``POST /api/annotate/labels``, each built through the route's request model and echoing the token
the last answer for its image named, write on each photograph a document holding no annotation,
which ``GET /api/annotate/labels`` then loads (``loads``, each its own token), then write on the
first photograph one :data:`SUBJECT` box (``box``), no annotation (``cleared``) and one
:data:`POINT_SUBJECT` point (``point``); once the second photograph holds a box,
``GET /api/annotate/states``
reads every photograph's state of :data:`SUBJECT` (``states``); the first photograph is marked
complete over its box and loaded (``complete``), then marked complete holding no annotation
(``negative``), and once the fifth is marked complete over a box every photograph's state is read
again (``marked_states``).

Proposals are staged through ``stage_proposals`` and served by ``GET /api/annotate/proposals``:
one over the third photograph (``single_proposals``), then :data:`PROPOSED` over the first, the
first accepted and :data:`REVIEWED` with :data:`REVIEWED_RING` saved (``review_saved``), loaded
(``review``) and served (``review_proposals``). The first box is flagged (``review_flagged``),
the paired proposal and the unpaired one are accepted (``accept_paired``, ``accept_unpaired``),
and a second bucket of a box and :data:`PROPOSED_RING` is served (``shaped_proposals``) and its
polygon accepted (``accept_polygon``); the unpaired proposal is accepted again by a save that
draws a box over it first (``accept_drawn``). A document whose box at :data:`TOOL_INDEX` a tool
wrote is loaded (``tool_two``), that box confirmed (``confirmed``) and the boxes saved as a
person's (``person_boxes``); the second photograph's document holding :data:`AUTHORED` (a tool's,
a person's, a tool's a person accepted, an unattributed box) is loaded (``authored``), then one
holding a tool's polygon (``tool_polygon``); the flag scenes follow (``_flag_scenes``), and the
prompt scenes over the fourth photograph (``_prompt_scenes``), and the photographs' registry
declaring the scenes' subjects (``registry``); last, the fifth photograph's document is damaged
in the store and its load refused (``unreadable``). Each staging's minted
bucket name is published under ``buckets``, and each save's answer under ``saves``. The answers
are written, each instant an annotation,
a mark or a flag stamps read as an instant (``fixture_session.LABEL_DOCUMENT_STAMPS`` beside the
served models' instants), each version token and each flag
id numbered (``fixture_session.normalized``), to :data:`GENERATED_PATH`.

    python -m tools.generate_annotate_fixture
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest

from tests import FRONTEND_SRC
from tools.fixture_session import (
    ACTOR,
    LABEL_DOCUMENT_STAMPS,
    answered,
    normalized,
    opened,
    scratch_session,
    write_fixture,
)
from tools.generate_open_fixture import FRAMES, SUBJECT, captures

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi.testclient import TestClient
    from pydantic import JsonValue

    from tcip_web.routes.annotate import AnnotationPayload

GENERATED_PATH = FRONTEND_SRC / "test" / "annotateAnswers.json"

BOX = [10.0, 10.0, 50.0, 50.0]
"""The corners of each box a scene saves, inside every photograph's frame."""

POINT_SUBJECT = "tip"
"""The subject of the point a scene saves, a sample keypoint subject."""

POINT = [200.0, 200.0]
"""Where the saved point sits, inside every photograph's frame."""

SECOND_SUBJECT = "subject_b"
"""A second subject the registry declares beside :data:`SUBJECT`, a sample name."""

MODEL = "detector"
"""The model the review scene's proposals are staged as, a sample name."""

MODEL_AUTHOR = f"model:{MODEL}"
"""Who a tool-authored box names as its producer."""

SIDE = 40.0
"""The side of each staged proposal's square box, in pixels."""

PROPOSED = (
    (10.0, 10.0, 0.9),
    (600.0, 600.0, 0.9),
    (700.0, 100.0, 0.2),
    (300.0, 300.0, 0.8),
)
"""Each staged proposal's top-left corner and score: one the review accepts, one no annotation
pairs with, one scored below the rest, and one a person's box pairs with. Sample values."""

REVIEWED = (
    [10.0, 10.0, 50.0, 50.0],
    [300.0, 300.0, 340.0, 340.0],
    [500.0, 50.0, 540.0, 90.0],
)
"""The review document's boxes: the first proposal's, accepted, then two of a person's, the first
over the last proposal and the second clear of every one."""

REVIEWED_RING = [[800.0, 600.0], [900.0, 600.0], [900.0, 700.0]]
"""The review document's polygon, clear of every proposal."""

PROPOSED_RING = [[100.0, 500.0], [200.0, 500.0], [200.0, 600.0]]
"""A polygon a second bucket proposes, clear of the review document's shapes."""

TOOL_INDEX = 2
"""The index of the box a tool authored in the authorship scene's document."""

AUTHORED = (
    (10.0, 10.0, 50.0, 50.0),
    (60.0, 10.0, 90.0, 50.0),
    (10.0, 60.0, 50.0, 90.0),
    (60.0, 60.0, 90.0, 90.0),
)
"""The corners of the boxes of the document each authorship stands in: a tool's, a person's, a
tool's a person accepted, and one no one authored. Sample values."""

FLAGGER = "first"
"""The person who flags the review document's first box and the flag scenes' boxes."""

SECOND = "second"
"""The person who resolves a flag and removes a flagged box."""

FLAG_TEXT = "open?"
"""What each flag asks, a sample question."""

FLAG_POINT = (30.0, 30.0)
"""Where each flag points, inside the first box."""

FOUND = ([10.0, 10.0, 40.0, 40.0], [100.0, 100.0, 300.0, 300.0])
"""The boxes the stub detector finds for a phrase, sample values."""

OTHER_FOUND = ([200.0, 200.0, 260.0, 260.0],)
"""What a second find answers, a sample box apart from the first answer's."""

PHRASE = "open bud"
"""The phrase each find asks for, the one the frontend's mocked toolbar sends."""

PROMPT_POINTS = [[120.0, 340.0], [200.0, 100.0]]
"""The points a propose sends, sample values inside the frame."""

ENGINE_REFUSAL = "a sample refusal the stub engine raises as it loads"
"""What a propose whose engine will not load answers."""


NUMBERED = frozenset({"base_mtime", "id"})
"""The served values that differ between runs and no scene step holds: each document's version
token and the ids of the flags a save minted that the scene never reads back."""


def produce() -> dict[str, JsonValue]:
    """Load and save the photographs' label documents in a scratch session; return the answers.

    A route refusing what it should admit raises.
    """
    from tcip_annotation.flags import FlagRequest, PointTarget
    from tcip_annotation.state import Annotation, Polygon
    from tcip_mcp.dataset_layout import Decisions, Gestures, label_key_of
    from tcip_mcp.identity import actor
    from tcip_mcp.tools.proposal_tools import StagedShapes, stage_proposals
    from tcip_mcp.web_client import CaptureChoice
    from tcip_web.routes.annotate import (
        AnnotationPayload,
        LabelDocumentAnswer,
        SavedLabels,
        SavePayload,
    )
    from tests._producer_fixtures import box_annotation, label_image
    from tests._record_damage_fixtures import damage_record
    from tests._web_fixtures import named_project

    with scratch_session() as (workspace, client):
        valley = named_project(workspace / "valley", "valley")
        opened(client, valley)
        root, date = captures(valley.root)["raw"]
        selected = answered(
            client,
            "select_dataset",
            200,
            CaptureChoice(dataset_root=str(root), date=date),
        )
        assert isinstance(selected, dict), selected
        # A served selection carries each image's computed name, which no model reads back.
        selection = cast(
            "dict[str, list[dict[str, str]]]", selected["selection"]
        )
        paths = [image["path"] for image in selection["images"]]
        first, second, *_ = paths
        tokens: dict[str, str | None] = dict.fromkeys(paths)

        box = AnnotationPayload(subject=SUBJECT, bbox=BOX)

        def save(
            path: str,
            annotations: list[AnnotationPayload],
            user: str = ACTOR,
            **gestures: object,
        ) -> JsonValue:
            answer = answered(
                client,
                "save_labels",
                200,
                SavePayload(
                    image_path=path,
                    annotations=annotations,
                    base_mtime=tokens[path],
                    user=user,
                    gestures=Gestures(**gestures),  # type: ignore[arg-type]
                ),
            )
            tokens[path] = SavedLabels.model_validate(
                answer
            ).document.base_mtime
            return answer

        def load(path: str) -> JsonValue:
            answer = answered(
                client, "load_labels", 200, params={"image_path": path}
            )
            tokens[path] = LabelDocumentAnswer.model_validate(
                answer
            ).base_mtime
            return answer

        for path in paths:
            save(path, [])
        served: dict[str, JsonValue] = {
            "loads": [load(path) for path in paths]
        }
        saves: dict[str, JsonValue] = {
            "box": save(first, [box]),
            "cleared": save(first, []),
            "point": save(
                first, [AnnotationPayload(subject=POINT_SUBJECT, point=POINT)]
            ),
        }
        save(second, [box])
        served["states"] = answered(
            client, "image_states", 200, params={"subject": SUBJECT}
        )
        save(first, [box], complete={SUBJECT: True})
        served["complete"] = load(first)
        saves["negative"] = save(first, [], complete={SUBJECT: True})
        save(paths[4], [box], complete={SUBJECT: True})
        served["marked_states"] = answered(
            client, "image_states", 200, params={"subject": SUBJECT}
        )
        width, height = FRAMES[0]

        def staged_bucket(
            path: str,
            corners: tuple[tuple[float, float, float], ...],
            polygons: list[dict] | None = None,
        ) -> str:
            frame_w, frame_h = FRAMES[paths.index(path)]
            return StagedShapes.model_validate(
                stage_proposals(
                    valley.root,
                    path,
                    model_name=MODEL,
                    boxes=[
                        {
                            "subject": SUBJECT,
                            "conf": conf,
                            "cx": (x + SIDE / 2) / frame_w,
                            "cy": (y + SIDE / 2) / frame_h,
                            "w": SIDE / frame_w,
                            "h": SIDE / frame_h,
                        }
                        for x, y, conf in corners
                    ],
                    polygons=polygons,
                    actor=None,
                )
            ).bucket

        single = staged_bucket(paths[2], (PROPOSED[0],))
        served["single_proposals"] = answered(
            client,
            "load_proposals",
            200,
            params={"image_path": paths[2], "bucket": single},
        )
        bucket = staged_bucket(first, PROPOSED)
        reviewed = [
            *(AnnotationPayload(subject=SUBJECT, bbox=b) for b in REVIEWED),
            AnnotationPayload(subject=SUBJECT, rings=[REVIEWED_RING]),
        ]
        save(
            first,
            [],
            decide=Decisions(bucket=bucket, verdicts={0: "accepted"}),
        )
        saves["review_saved"] = save(first, reviewed)
        served["review"] = load(first)
        served["review_proposals"] = answered(
            client,
            "load_proposals",
            200,
            params={"image_path": first, "bucket": bucket},
        )
        save(
            first,
            reviewed,
            user=FLAGGER,
            flag=(
                FlagRequest(
                    text=FLAG_TEXT,
                    target=PointTarget(point=FLAG_POINT, subject=SUBJECT),
                ),
            ),
        )
        served["review_flagged"] = load(first)
        saves["accept_paired"] = save(
            first,
            reviewed,
            decide=Decisions(bucket=bucket, verdicts={3: "accepted"}),
        )
        saves["accept_unpaired"] = save(
            first,
            reviewed,
            decide=Decisions(bucket=bucket, verdicts={1: "accepted"}),
        )
        shaped = staged_bucket(
            first,
            (PROPOSED[1],),
            polygons=[
                {
                    "subject": SUBJECT,
                    "conf": 0.9,
                    "points": [
                        [x / width, y / height] for x, y in PROPOSED_RING
                    ],
                }
            ],
        )
        served["shaped_proposals"] = answered(
            client,
            "load_proposals",
            200,
            params={"image_path": first, "bucket": shaped},
        )
        saves["accept_polygon"] = save(
            first,
            reviewed,
            decide=Decisions(bucket=shaped, verdicts={1: "accepted"}),
        )
        x, y, _ = PROPOSED[1]
        over_unpaired = AnnotationPayload(
            subject=SUBJECT, bbox=[x, y, x + SIDE, y + SIDE]
        )
        saves["accept_drawn"] = save(
            first,
            [over_unpaired, *reviewed],
            decide=Decisions(bucket=bucket, verdicts={1: "accepted"}),
        )
        served["buckets"] = {
            "single": single,
            "review": bucket,
            "shaped": shaped,
        }
        three = [AnnotationPayload(subject=SUBJECT, bbox=b) for b in REVIEWED]
        label_image(
            first,
            [
                replace(
                    box_annotation(*b, subject=SUBJECT),
                    created_by=MODEL_AUTHOR
                    if i == TOOL_INDEX
                    else actor(ACTOR),
                )
                for i, b in enumerate(REVIEWED)
            ],
            width,
            height,
        )
        served["tool_two"] = load(first)
        person = actor(ACTOR)
        label_image(
            second,
            [
                replace(
                    box_annotation(*b, subject=SUBJECT),
                    created_by=by,
                    accepted_by=accepted,
                )
                for b, (by, accepted) in zip(
                    AUTHORED,
                    (
                        (MODEL_AUTHOR, None),
                        (person, None),
                        (MODEL_AUTHOR, person),
                        (None, None),
                    ),
                    strict=True,
                )
            ],
            *FRAMES[1],
        )
        served["authored"] = load(second)
        label_image(
            second,
            [
                Annotation(
                    subject=SUBJECT,
                    geometry=Polygon([[(x, y) for x, y in REVIEWED_RING]]),
                    created_by=MODEL_AUTHOR,
                )
            ],
            *FRAMES[1],
        )
        served["tool_polygon"] = load(second)
        saves["confirmed"] = save(
            first, three, confirm=frozenset({TOOL_INDEX})
        )
        save(first, [])
        saves["person_boxes"] = save(first, three)
        flag_saves, flag_loads, held_flag = _flag_scenes(save, load, first, box)
        saves.update(flag_saves)
        served.update(flag_loads)
        served["saves"] = saves
        prompted, minted = _prompt_scenes(client, paths[3])
        served.update(prompted)
        served["registry"] = _registry(client, str(root), date)
        damage_record(label_key_of(paths[4]), b"[]")
        served["unreadable"] = answered(
            client, "load_labels", 400, params={"image_path": paths[4]}
        )
        return normalized(
            served,
            workspace,
            instants=LABEL_DOCUMENT_STAMPS,
            numbered=NUMBERED,
            minted={
                valley.id: "<valley id>",
                single: "<single bucket>",
                bucket: "<review bucket>",
                shaped: "<shaped bucket>",
                **minted,
                **held_flag,
            },
        )


def _registry(client: TestClient, dataset_root: str, date: str) -> JsonValue:
    """Declare the scenes' subjects in the photographs' registry; return what the load serves.

    The subjects, :data:`SUBJECT`, :data:`POINT_SUBJECT` and :data:`SECOND_SUBJECT`, are saved
    through the subjects save route over the version the load route answers.
    """
    from tcip_mcp.subject_registry import Subject
    from tcip_web.routes.subjects import SaveSubjectsPayload, SubjectsLoad

    def load() -> JsonValue:
        return answered(
            client,
            "load_subjects",
            200,
            params={"dataset_root": dataset_root, "date": date},
        )

    answered(
        client,
        "save_subjects",
        200,
        SaveSubjectsPayload(
            subjects=tuple(
                Subject(name=name, description=f"one {name}")
                for name in (SUBJECT, POINT_SUBJECT, SECOND_SUBJECT)
            ),
            dataset_root=dataset_root,
            version=SubjectsLoad.model_validate(load()).version,
            user=ACTOR,
        ),
    )
    return load()


def _flag_scenes(
    save: Callable[..., JsonValue],
    load: Callable[[str], JsonValue],
    path: str,
    box: AnnotationPayload,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue], dict[str, str]]:
    """Flag the box of the photograph at ``path``, remove it, flag a new one twice and resolve one.

    Returns the saves, then the loads: the save of :data:`FLAGGER` flagging its one box
    (``flag_raised``) and the document it wrote (``flagged``), the save of :data:`SECOND` removing
    that box (``flag_removed``), and the document once a new box is flagged twice and
    :data:`SECOND` resolved the second flag (``flags_mixed``); then the resolved flag's id, which
    the scene holds, by its name.
    """
    from tcip_annotation.flags import FlagRequest, PointTarget
    from tcip_web.routes.annotate import LabelDocumentAnswer

    on_box = FlagRequest(
        text=FLAG_TEXT, target=PointTarget(point=FLAG_POINT, subject=SUBJECT)
    )
    save(path, [box])
    saves = {"flag_raised": save(path, [box], user=FLAGGER, flag=(on_box,))}
    loads = {"flagged": load(path)}
    saves["flag_removed"] = save(path, [], user=SECOND)
    save(path, [box], user=FLAGGER, flag=(on_box,))
    save(path, [box], user=FLAGGER, flag=(on_box,))
    latest = LabelDocumentAnswer.model_validate(load(path)).flags[-1]
    save(path, [box], user=SECOND, resolve={latest.id: "seen"})
    loads["flags_mixed"] = load(path)
    return saves, loads, {latest.id: "<resolved flag id>"}


def _prompt_scenes(
    client: TestClient, path: str
) -> tuple[dict[str, JsonValue], dict[str, str]]:
    """Propose and find over the photograph at ``path`` through a stub engine.

    Returns the answers, and what each id the staging minted reads as: its bucket's name and its
    render's place.

    The image's stagings before any propose list nothing (``no_stagings``). The engine
    (``tests._producer_fixtures.install_stub_proposer``) answers :data:`FOUND` as its
    candidates: a propose of :data:`PROMPT_POINTS` stages them (``staged``), the image's stagings
    then list its bucket (``stagings``) and the proposals route serves it (``staged_proposals``);
    a find answers their boxes (``found``), and one over :data:`OTHER_FOUND` alone answers that
    (``found_other``). An engine answering nothing proposes nothing (``nothing_proposed``) and
    finds nothing (``found_none``), and one that will not load, raising :data:`ENGINE_REFUSAL`,
    refuses the propose (``engine_refusal``).
    """
    from tcip_mcp.pipelines import proposal
    from tcip_mcp.tools.proposal_tools import Staged
    from tcip_web.routes.annotate import FindPayload, ProposePayload
    from tests._producer_fixtures import install_stub_proposer

    def candidates(boxes: tuple[list[float], ...]) -> list[dict]:
        return [
            {
                "candidate_id": i,
                "bbox": b,
                "area": (b[2] - b[0]) * (b[3] - b[1]),
                "score": 0.9,
                "engine_meta": {},
                "rings": [
                    [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]
                ],
            }
            for i, b in enumerate(boxes)
        ]

    def propose(status: int = 200) -> JsonValue:
        return answered(
            client,
            "propose",
            status,
            ProposePayload.model_validate(
                {
                    "image_path": path,
                    "subject": SUBJECT,
                    "user": ACTOR,
                    "prompt": {"points": PROMPT_POINTS, "boxes": []},
                }
            ),
        )

    def find() -> JsonValue:
        return answered(
            client,
            "find",
            200,
            FindPayload(image_path=path, phrase=PHRASE, user=ACTOR),
        )

    scenes: dict[str, JsonValue] = {
        "no_stagings": answered(
            client, "image_stagings", 200, params={"image_path": path}
        )
    }
    with pytest.MonkeyPatch.context() as patch:
        install_stub_proposer(patch, candidates(FOUND))
        scenes["staged"] = propose()
        staged = Staged.model_validate(scenes["staged"])
        bucket = staged.bucket
        scenes["stagings"] = answered(
            client, "image_stagings", 200, params={"image_path": path}
        )
        scenes["staged_proposals"] = answered(
            client,
            "load_proposals",
            200,
            params={"image_path": path, "bucket": bucket},
        )
        scenes["found"] = find()
    with pytest.MonkeyPatch.context() as patch:
        install_stub_proposer(patch, candidates(OTHER_FOUND))
        scenes["found_other"] = find()
    with pytest.MonkeyPatch.context() as patch:
        install_stub_proposer(patch, [])
        scenes["nothing_proposed"] = propose()
        scenes["found_none"] = find()

    def refuse(_engine: str) -> object:
        raise ValueError(ENGINE_REFUSAL)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(proposal, "resolve_proposer", refuse)
        scenes["engine_refusal"] = propose(400)
    return scenes, {
        bucket: "<staged bucket>",
        staged.image_path: "<staged render>",
    }


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))
