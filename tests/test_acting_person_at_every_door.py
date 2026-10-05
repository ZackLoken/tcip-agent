"""Every GUI door whose act writes a project record names the person making it: a body that omits
the person is refused by its model, and one naming no one (blank, or the prefix alone) is refused
with the refusal ``identity.actor`` raises, before anything is read or written."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_mcp.identity import NoActor, actor
from tests._audit_fixtures import audit_rows

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
    ("/api/terminal/sessions", {"provider": "claude"}),
    ("/api/terminal/sessions/term_1/restart", {"provider": "claude"}),
    ("/api/sessions/image_event", {"image_name": "a.jpg", "seconds": 1.0, "project_id": "p",
                                   "annotations_added": 0, "activity": "review"}),
]

NO_ONE = ["  ", "user:", "user:   "]
"""Names that name no person: blank, and the prefix with no name after it."""


@pytest.mark.parametrize("name", NO_ONE)
def test_actor_refuses_a_name_that_names_no_one(name: str) -> None:
    with pytest.raises(NoActor):
        actor(name)


def test_actor_spells_a_name_once_with_or_without_its_prefix() -> None:
    assert actor(" Alice ") == actor("user: Alice") == "user:Alice"


@pytest.mark.parametrize(("route", "body"), DOORS, ids=[route for route, _ in DOORS])
def test_a_door_refuses_a_body_naming_no_one_and_records_nothing(
    opened_project: Path, client: TestClient, route: str, body: dict,
) -> None:
    before = audit_rows(opened_project)

    omitted = client.post(route, json=body)
    assert omitted.status_code == 422, omitted.text
    for name in NO_ONE:
        resp = client.post(route, json={**body, "user": name})
        assert resp.status_code == 400, (name, resp.text)
        assert "names no one" in resp.json()["detail"]
    assert audit_rows(opened_project) == before
