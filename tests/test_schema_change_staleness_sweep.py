"""The staleness sweep both registry writers run before a schema change lands.

A confirmation and its digest stamp are two transactions, status first, so confirmations with no
stamp legitimately exist and are admitted on read. Once the attribute vocabulary changes, an
unstamped confirmation can no longer be told from one made under the new vocabulary, so it would
train against a definition its human never saw. Each writer therefore stamps the outgoing digest
onto the still-unstamped confirmations of every subject whose schema is changing, while the outgoing
registry is still readable. Both writers here run their real implementations, and the training-side
reader does the reading, so the two sides cannot disagree about what a stamp means.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import tcip_store as ts
from tcip_annotation.json_io import write_annotations
from tcip_mcp.class_registry import attribute_schema_digest, registry_from_dict
from tcip_mcp.dataset_layout import (
    classes_path, image_status_digest_key, image_status_digest_path, record_image_statuses,
    status_bucket,
)
from tcip_store.file_backend import FileBackend
from tcip_mcp.pipelines.data.datasets import confirmed_negative_names
from tcip_mcp.tools.annotation_tools import write_class_map
from tcip_web.app import app

CATKIN_TWO_STATES = {"elongation": {"type": "categorical", "values": ["dormant", "elongated"]}}
CATKIN_THREE_STATES = {
    "elongation": {"type": "categorical", "values": ["dormant", "elongating", "elongated"]}
}
CATKIN_FOUR_STATES = {
    "elongation": {"type": "categorical",
                   "values": ["dormant", "swelling", "elongating", "elongated"]}
}


def _subjects(catkin_attributes: dict, bush_attributes: dict | None = None) -> dict:
    """The registry mapping both writers take: catkin carries an attribute vocabulary, bush is
    detection-only unless a case gives it one, so the two subjects have different schemas."""
    bush: dict = {"description": "one plant crown"}
    if bush_attributes is not None:
        bush["attributes"] = bush_attributes
    return {"catkin": {"description": "the male flower", "attributes": catkin_attributes},
            "bush": bush}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """A dataset root whose label files are present but empty, ready to be confirmed negative."""
    labels = tmp_path / "annotations"
    labels.mkdir()
    for stem in ("img_alpha", "img_beta", "img_gamma"):
        write_annotations(str(labels / f"{stem}.json"), [], 800, 450, keep_empty=True)
    return tmp_path


def _save_via_route(client: TestClient, root: Path, catkin_attributes: dict,
                    bush_attributes: dict | None = None) -> dict:
    resp = client.post(
        "/api/classes/save",
        json={"project_root": str(root), "dataset_root": str(root),
              "subjects": _subjects(catkin_attributes, bush_attributes)},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _save_via_tool(root: Path, catkin_attributes: dict,
                   bush_attributes: dict | None = None) -> dict:
    res = write_class_map(str(root), subjects=_subjects(catkin_attributes, bush_attributes))
    assert "error" not in res, res
    return res


def _confirm_negative_stamped(client: TestClient, root: Path, image_name: str,
                              subject: str) -> None:
    """A confirmation through the GUI route, which stamps the schema in effect at confirm time."""
    resp = client.post(
        "/api/classes/image_status",
        json={"project_root": str(root), "dataset_root": str(root), "image_name": image_name,
              "status": "negative", "subject": subject},
    )
    assert resp.status_code == 200, resp.text


def _confirm_negative_unstamped(root: Path, image_name: str, subject: str) -> None:
    """A confirmation recorded through the status writer alone, the state a dataset is left in
    whenever the stamp transaction did not follow the status one."""
    record_image_statuses(root, status_bucket(subject, None), {image_name: "negative"},
                          recorded_by="user:breeder")
    assert not ts.exists(image_status_digest_key(root))


def _stamps(root: Path, subject: str) -> dict:
    return ts.read(image_status_digest_key(root), default={}).get(status_bucket(subject, None), {})


def _read_negatives(root: Path, subject: str) -> tuple[set[str], set[str]]:
    quarantined: set[str] = set()
    admitted = confirmed_negative_names(
        root / "annotations", subject=subject, date=None, quarantined_out=quarantined)
    return admitted, quarantined


def test_the_save_route_marks_an_unstamped_confirmation_as_predating_the_schema_change(
    client: TestClient, dataset: Path
) -> None:
    """Saving a grown attribute vocabulary stamps the confirmations that carry no stamp with the
    schema they were made under, so they quarantine instead of training under a vocabulary their
    human never saw."""
    _save_via_route(client, dataset, CATKIN_TWO_STATES)
    _confirm_negative_unstamped(dataset, "img_alpha.jpg", "catkin")
    assert _read_negatives(dataset, "catkin") == ({"img_alpha.jpg"}, set())

    response = _save_via_route(client, dataset, CATKIN_THREE_STATES)

    assert _read_negatives(dataset, "catkin") == (set(), {"img_alpha.jpg"})
    assert response["schema_change_sweep"]["newly_stamped"] == {"catkin": 1}
    assert response["schema_change_sweep"]["warning"] is None


def test_the_class_map_tool_marks_an_unstamped_confirmation_as_predating_the_schema_change(
    dataset: Path
) -> None:
    """The agent's own registry writer sweeps identically: one implementation, so the GUI and the
    tool cannot disagree about which confirmations predate a vocabulary change."""
    _save_via_tool(dataset, CATKIN_TWO_STATES)
    _confirm_negative_unstamped(dataset, "img_alpha.jpg", "catkin")
    assert _read_negatives(dataset, "catkin") == ({"img_alpha.jpg"}, set())

    result = _save_via_tool(dataset, CATKIN_THREE_STATES)

    assert _read_negatives(dataset, "catkin") == (set(), {"img_alpha.jpg"})
    assert result["schema_change_sweep"]["newly_stamped"] == {"catkin": 1}
    assert result["schema_change_sweep"]["warning"] is None


def test_a_confirmation_that_carries_a_stamp_keeps_it_across_later_schema_changes(
    client: TestClient, dataset: Path
) -> None:
    """The stamp set at confirmation time is the direct evidence of what a human saw, so the sweep
    records only what is missing: a confirmation stamped two vocabularies ago still carries that
    digest, and one made under the vocabulary now current still admits."""
    _save_via_route(client, dataset, CATKIN_TWO_STATES)
    _confirm_negative_stamped(client, dataset, "img_alpha.jpg", "catkin")
    two_state_digest = attribute_schema_digest(
        registry_from_dict(_subjects(CATKIN_TWO_STATES)), "catkin")
    assert _stamps(dataset, "catkin") == {"img_alpha.jpg": two_state_digest}

    first = _save_via_route(client, dataset, CATKIN_THREE_STATES)
    second = _save_via_route(client, dataset, CATKIN_FOUR_STATES)
    _confirm_negative_stamped(client, dataset, "img_gamma.jpg", "catkin")

    assert first["schema_change_sweep"]["newly_stamped"] == {}
    assert second["schema_change_sweep"]["newly_stamped"] == {}
    assert _stamps(dataset, "catkin")["img_alpha.jpg"] == two_state_digest
    assert _read_negatives(dataset, "catkin") == ({"img_gamma.jpg"}, {"img_alpha.jpg"})


def test_a_subject_whose_schema_is_unchanged_keeps_its_unstamped_confirmations(
    client: TestClient, dataset: Path
) -> None:
    """The sweep is scoped to the subjects whose digest actually changes: another subject's
    confirmations are neither stamped nor quarantined by a change they had no part in."""
    _save_via_route(client, dataset, CATKIN_TWO_STATES)
    _confirm_negative_unstamped(dataset, "img_beta.jpg", "bush")

    response = _save_via_route(client, dataset, CATKIN_THREE_STATES)

    assert "bush" not in response["schema_change_sweep"]["newly_stamped"]
    assert _stamps(dataset, "bush") == {}
    assert _read_negatives(dataset, "bush") == ({"img_beta.jpg"}, set())


def test_a_schema_change_with_nothing_to_stamp_writes_the_registry_and_no_digest_store(
    client: TestClient, dataset: Path
) -> None:
    """A dataset with no confirmations has nothing to record, so the sweep leaves no digest store
    behind and the registry write is unaffected."""
    _save_via_route(client, dataset, CATKIN_TWO_STATES)

    response = _save_via_route(client, dataset, CATKIN_THREE_STATES)

    assert response["schema_change_sweep"] == {"newly_stamped": {}, "warning": None}
    assert not image_status_digest_path(dataset).exists()
    saved = json.loads(classes_path(dataset).read_text(encoding="utf-8"))
    assert saved["catkin"]["attributes"]["elongation"]["values"] == \
        ["dormant", "elongating", "elongated"]


def _block_the_digest_store(root: Path) -> None:
    """Occupy the digest store's own path with a directory, so the sweep's write fails while the
    registry write stays fine."""
    path = image_status_digest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()


def test_the_save_route_writes_the_registry_and_reports_a_sweep_it_could_not_complete(
    client: TestClient, dataset: Path
) -> None:
    """A stamp failure never rejects the human's confirmation, and a sweep failure never rejects the
    expert's schema change: it comes back as a warning naming what is left unstamped.

    Bound to the file backend: the failure is simulated by occupying the digest store's own file
    path with a directory, which is a file-backend write obstruction, not a backend-general one.
    """
    ts.bind(FileBackend())
    _save_via_route(client, dataset, CATKIN_TWO_STATES)
    _confirm_negative_unstamped(dataset, "img_alpha.jpg", "catkin")
    _block_the_digest_store(dataset)

    response = _save_via_route(client, dataset, CATKIN_THREE_STATES)

    assert response["status"] == "ok"
    assert response["schema_change_sweep"]["newly_stamped"] == {}
    assert "re-review them before they train" in response["schema_change_sweep"]["warning"]
    saved = json.loads(classes_path(dataset).read_text(encoding="utf-8"))
    assert saved["catkin"]["attributes"]["elongation"]["values"] == \
        ["dormant", "elongating", "elongated"]


def test_the_class_map_tool_writes_the_registry_and_reports_a_sweep_it_could_not_complete(
    dataset: Path
) -> None:
    """The tool warns and proceeds on the same terms as the route, through the same sweep.

    Bound to the file backend: the failure is simulated by occupying the digest store's own file
    path with a directory, which is a file-backend write obstruction, not a backend-general one.
    """
    ts.bind(FileBackend())
    _save_via_tool(dataset, CATKIN_TWO_STATES)
    _confirm_negative_unstamped(dataset, "img_alpha.jpg", "catkin")
    _block_the_digest_store(dataset)

    result = _save_via_tool(dataset, CATKIN_THREE_STATES)

    assert result["schema_change_sweep"]["newly_stamped"] == {}
    assert "re-review them before they train" in result["schema_change_sweep"]["warning"]
    saved = json.loads(classes_path(dataset).read_text(encoding="utf-8"))
    assert saved["catkin"]["attributes"]["elongation"]["values"] == \
        ["dormant", "elongating", "elongated"]
