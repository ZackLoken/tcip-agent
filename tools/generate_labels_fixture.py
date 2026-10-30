"""Write the label routes' answers as a fixture file, produced by the platform itself.

In a scratch session (``tools/fixture_session.py``), a project created through
``tests._web_fixtures.named_project`` and opened through the picker's route holds one image
(``tests._producer_fixtures.write_image``). Its label document is saved through the save route,
built through the route's request model, with a box, a polygon, a point and a geometry-less
rating of :data:`SUBJECT` and a flag on the box; it is then loaded through the load route, and
one box staged clear of them
(``tests._producer_fixtures.staged_box``) is served through the proposals route; a save naming a
stale version token is refused, and a second save then resolves the flag. The dataset's subject
registry is then grown twice through the subjects save route, the second time while every audit
append fails. The answers are written, normalized by the label document's stamps
(``fixture_session.LABEL_DOCUMENT_STAMPS``) and :data:`NUMBERED`, the
staged bucket and the flag's id read under the scene's names for them
(``fixture_session.normalized``), to :data:`GENERATED_PATH`, keyed ``saved``, ``loaded``,
``proposals``, ``resolved``, ``version_conflict``, the stale save's refusal,
``registry_saved``, the first registry save, ``registry_stale``, the refusal of it repeated over
its stale version, ``registry_audit_gap``, the second's refusal, and
``registry_loaded``, the registry the load route then serves; each refusal is kept with its
status.

    python -m tools.generate_labels_fixture
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
from tools.generate_open_fixture import FOCUSED_SUBJECT as SUBJECT

if TYPE_CHECKING:
    from fastapi.testclient import TestClient
    from pydantic import JsonValue

GENERATED_PATH = FRONTEND_SRC / "test" / "labelAnswers.json"

CAPTURE = "2026-02-11"
"""The capture the image is under."""

FRAME = (100, 80)
"""The image's width and height in pixels."""

BOX = [10.0, 10.0, 40.0, 30.0]
"""The saved box's corners, which the flag's point sits inside."""

STAGED = [75.0, 45.0, 95.0, 75.0]
"""The staged box's corners, clear of every saved annotation, so the proposal pairs with none."""


NUMBERED = frozenset({"base_mtime", "version"})
"""The served values that differ between runs and no scene step holds: the document's and the
registry's version tokens."""


def produce() -> dict[str, JsonValue]:
    """Save, load and stage over one image in a scratch session; return the routes' answers.

    A route refusing what it should admit raises.
    """
    from tcip_annotation.flags import FlagRequest, PointTarget
    from tcip_mcp.dataset_layout import Gestures, image_dir
    from tcip_web.routes.annotate import (
        AnnotationPayload,
        SavedLabels,
        SavePayload,
    )
    from tests._producer_fixtures import staged_box, write_image
    from tests._web_fixtures import named_project

    with scratch_session() as (workspace, client):
        root = workspace / "labels"
        opened(client, named_project(root, "Labels"))
        image = write_image(image_dir(root, CAPTURE) / "a.jpg", FRAME)
        request = SavePayload(
            image_path=str(image),
            user=ACTOR,
            annotations=[
                AnnotationPayload(subject=SUBJECT, bbox=BOX),
                AnnotationPayload(
                    subject=SUBJECT,
                    points=[[50.0, 10.0], [70.0, 10.0], [70.0, 30.0]],
                ),
                AnnotationPayload(subject=SUBJECT, point=[20.0, 60.0]),
                AnnotationPayload(subject=SUBJECT),
            ],
            gestures=Gestures(
                flag=(
                    FlagRequest(
                        text="a second look",
                        target=PointTarget(
                            point=(20.0, 20.0), subject=SUBJECT
                        ),
                    ),
                )
            ),
        )
        saved = answered(client, "save_labels", 200, request)
        loaded = answered(
            client, "load_labels", 200, params={"image_path": str(image)}
        )
        bucket = staged_box(root, image, STAGED)
        proposals = answered(
            client,
            "load_proposals",
            200,
            params={"image_path": str(image), "bucket": bucket},
        )
        stale_request = request.model_copy(update={"base_mtime": "0"})
        stale = answered(client, "save_labels", 409, stale_request)
        (flag,) = SavedLabels.model_validate(saved).document.flags
        resolve = request.model_copy(
            update={"gestures": Gestures(resolve={flag.id: "seen"})}
        )
        resolved = answered(client, "save_labels", 200, resolve)
        registry = _registry_saves(client, str(root))
        return normalized(
            {
                "saved": saved,
                "loaded": loaded,
                "proposals": proposals,
                "resolved": resolved,
                "version_conflict": stale,
                **registry,
            },
            workspace,
            instants=LABEL_DOCUMENT_STAMPS,
            numbered=NUMBERED,
            minted={bucket: "<staged bucket>", flag.id: "<flag id>"},
        )


def _registry_saves(
    client: TestClient, dataset_root: str
) -> dict[str, JsonValue]:
    """Grow ``dataset_root``'s subject registry through the subjects save route.

    :data:`SUBJECT` is declared with no attribute first, and the load route then serves it
    (``registry_bare``). A save gives it one attribute over the version that load answers; the
    same save over that version again is refused, the version having moved; the next adds a value
    while every audit append fails (``tests._audit_fixtures.refuse_audit_appends``); the load
    route then serves what was saved. Returns, besides the bare load, the attribute's save
    (``registry_saved``), the stale save's refusal (``registry_stale``), the next save's refusal
    (``registry_audit_gap``) and the last load (``registry_loaded``); a route refusing what it
    should admit, or admitting what it should refuse, raises.
    """
    from tcip_annotation.state import Attribute
    from tcip_mcp.subject_registry import RegistrySave, Subject
    from tcip_web.routes.subjects import SaveSubjectsPayload, SubjectsLoad
    from tests._audit_fixtures import refuse_audit_appends

    def load() -> JsonValue:
        return answered(
            client,
            "load_subjects",
            200,
            params={"dataset_root": dataset_root, "date": CAPTURE},
        )

    def grown(values: list[str], version: str | None) -> SaveSubjectsPayload:
        return SaveSubjectsPayload(
            subjects=(
                Subject(
                    name=SUBJECT,
                    description=f"one {SUBJECT}",
                    attributes=(
                        Attribute(
                            name="opening",
                            type="categorical",
                            values=tuple(values),
                        ),
                    )
                    if values
                    else (),
                ),
            ),
            dataset_root=dataset_root,
            version=version,
            user=ACTOR,
        )

    answered(
        client,
        "save_subjects",
        200,
        grown([], SubjectsLoad.model_validate(load()).version),
    )
    bare = load()
    first = SubjectsLoad.model_validate(bare).version
    saved = answered(client, "save_subjects", 200, grown(["closed"], first))
    second = RegistrySave.model_validate(saved).version
    stale = answered(client, "save_subjects", 409, grown(["closed"], first))
    with pytest.MonkeyPatch.context() as patch:
        refuse_audit_appends(patch)
        gap = answered(
            client, "save_subjects", 409, grown(["closed", "open"], second)
        )
    return {
        "registry_bare": bare,
        "registry_saved": saved,
        "registry_stale": stale,
        "registry_audit_gap": gap,
        "registry_loaded": load(),
    }


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))
