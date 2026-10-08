"""Every GUI door whose act writes a project record names the person making it: a body that omits
the person is refused by its model, and one naming no one is refused with the refusal
``identity.actor`` raises, before anything is read or written. The names are the one list the
browser's field is tested on, and the browser's own rule admits exactly what ``actor`` admits."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_annotation.json_io import PERSON_IDENTITY_PREFIX, is_person
from tcip_mcp.identity import NoActorError, actor
from tests import ADMITTED_NAMES, FRONTEND_SRC, REFUSED_NAMES
from tests._audit_fixtures import audit_rows

NO_PROJECT = "0" * 12
"""A project id the workspace holds no project under."""

DOORS = [
    ("/api/training/runs", {"relaunched_from": "exp-1"}),
    ("/api/training/runs/exp-1/cancel", {}),
    ("/api/inference/launch", {"checkpoint_path": "m.pt", "dataset_root": "d",
                               "date": "2026-01-01", "bucket": "o"}),
    ("/api/inference/jobs/inf-1/cancel", {}),
    ("/api/subjects/save", {"subjects": {"bud": {}}, "dataset_root": "d", "version": None}),
    ("/api/results/plant_mapping/build", {"name": "valley", "images_root": "i",
                                          "plant_registry": "r"}),
    ("/api/results/export_csv", {"mapping_name": "valley", "dataset_root": "d", "buckets": [],
                                 "trait": "t", "plants": ["p"], "payload": "milestones"}),
    ("/api/results/export_count_csv", {"delivery": {"kind": "per_image_count",
                                                    "dataset_root": "d", "bucket": "b",
                                                    "trait": "t"},
                                       "filename": "x.csv"}),
    ("/api/results/traits/confirm", {"trait": "t", "revision": 1, "entry_sha256": "0" * 64,
                                     "confirmed": True}),
    ("/api/terminal/sessions", {"provider": "claude"}),
    ("/api/terminal/sessions/term_1/restart", {"provider": "claude"}),
    ("/api/sessions/image_event", {"image_name": "a.jpg", "seconds": 1.0, "project_id": "p",
                                   "annotations_added": 0, "activity": "review"}),
    ("/api/annotate/labels", {"image_path": "a.jpg"}),
    ("/api/canvas/state", {"project_id": "p", "tab": "annotate", "image_path": "a.jpg",
                           "image": "a.jpg"}),
    ("/api/projects/open", {"id": NO_PROJECT}),
    ("/api/projects/remove", {"id": NO_PROJECT, "confirm_name": "p"}),
    ("/api/projects/rename", {"id": NO_PROJECT, "display_name": "p"}),
]

_RULE_EXPORT = re.compile(r"^export const PERSON_NAME_RULE = (\".*\");$", re.M)


def _actor_admits(name: str) -> bool:
    try:
        actor(name)
    except NoActorError:
        return False
    return True


def _browser_admits(names: list[str]) -> list[bool]:
    """Whether the browser's rule, as the generated module exports it, admits each of ``names``,
    run under Node's own ``RegExp``."""
    generated = (FRONTEND_SRC / "api" / "types.generated.ts").read_text(encoding="utf-8")
    exported = _RULE_EXPORT.search(generated)
    assert exported is not None, "the generated module exports no PERSON_NAME_RULE"
    script = ("const {rule, names} = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
              "process.stdout.write(JSON.stringify(names.map((n) => new RegExp(rule).test(n))));")
    ran = subprocess.run(["node", "-e", script], capture_output=True, encoding="utf-8",
                         check=True, input=json.dumps({"rule": json.loads(exported.group(1)),
                                                       "names": names}))
    return json.loads(ran.stdout)


def test_actor_records_each_admitted_name_and_refuses_each_refused_one() -> None:
    for stated, recorded in ADMITTED_NAMES:
        assert actor(stated) == PERSON_IDENTITY_PREFIX + recorded, stated
    for stated in REFUSED_NAMES:
        with pytest.raises(NoActorError):
            actor(stated)


def test_a_recorded_identity_is_exactly_the_spelling_actor_records() -> None:
    names = [stated for stated, _ in ADMITTED_NAMES] + REFUSED_NAMES
    assert {n: is_person(n) for n in names} == {
        n: _actor_admits(n) and actor(n) == n for n in names}
    assert all(is_person(actor(stated)) for stated, _ in ADMITTED_NAMES)


def test_the_browsers_rule_admits_exactly_what_actor_admits() -> None:
    names = [stated for stated, _ in ADMITTED_NAMES] + REFUSED_NAMES
    assert dict(zip(names, _browser_admits(names))) == {n: _actor_admits(n) for n in names}


@pytest.mark.parametrize(("route", "body"), DOORS, ids=[route for route, _ in DOORS])
def test_a_door_refuses_a_body_naming_no_one_and_records_nothing(
    opened_project: Path, client: TestClient, route: str, body: dict,
) -> None:
    before = audit_rows(opened_project)

    omitted = client.post(route, json=body)
    assert omitted.status_code == 422, omitted.text
    for name in REFUSED_NAMES:
        resp = client.post(route, json={**body, "user": name})
        assert resp.status_code == 400, (name, resp.text)
        assert "names no one" in resp.json()["detail"]
    assert audit_rows(opened_project) == before
