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
    stamp_image_status_digests, status_bucket,
)
from tcip_store.file_backend import FileBackend
from tcip_mcp.pipelines.data.label_queries import confirmed_negative_names
from tcip_mcp.tools.annotation_tools import write_class_map
from tcip_web.app import app

BUD_TWO_STATES = {"opening": {"type": "categorical", "values": ["closed", "open"]}}
BUD_THREE_STATES = {
    "opening": {"type": "categorical", "values": ["closed", "partial", "open"]}
}
BUD_FOUR_STATES = {
    "opening": {"type": "categorical",
                   "values": ["closed", "swelling", "partial", "open"]}
}


def _subjects(bud_attributes: dict, bush_attributes: dict | None = None) -> dict:
    """The registry mapping both writers take: bud carries an attribute vocabulary, bush is
    detection-only unless a case gives it one, so the two subjects have different schemas."""
    bush: dict = {"description": "one plant crown"}
    if bush_attributes is not None:
        bush["attributes"] = bush_attributes
    return {"bud": {"description": "the male flower", "attributes": bud_attributes},
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


def _save_via_route(client: TestClient, root: Path, bud_attributes: dict,
                    bush_attributes: dict | None = None) -> dict:
    """Save through the route, carrying forward the version the route itself last reported, the
    way the toolbar carries what it loaded rather than defaulting to an unconditional write."""
    version = client.get(
        "/api/classes/load",
        params={"project_root": str(root), "dataset_root": str(root)},
    ).json()["version"]
    resp = client.post(
        "/api/classes/save",
        json={"project_root": str(root), "dataset_root": str(root),
              "subjects": _subjects(bud_attributes, bush_attributes), "version": version},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _save_via_tool(root: Path, bud_attributes: dict,
                   bush_attributes: dict | None = None) -> dict:
    res = write_class_map(str(root), subjects=_subjects(bud_attributes, bush_attributes))
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


def _confirm_complete_stamped(client: TestClient, root: Path, image_name: str,
                              subject: str) -> None:
    """A Complete confirmation through the GUI route: the sweep's predating-vocabulary count
    covers every status a bucket holds, not the negatives alone."""
    resp = client.post(
        "/api/classes/image_status",
        json={"project_root": str(root), "dataset_root": str(root), "image_name": image_name,
              "status": "complete", "subject": subject},
    )
    assert resp.status_code == 200, resp.text


def _audit_entries(root: Path) -> list[dict]:
    from tcip_mcp.audit import audit_log_key

    return list(ts.read_log(audit_log_key(root)).records)


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
    _save_via_route(client, dataset, BUD_TWO_STATES)
    _confirm_negative_unstamped(dataset, "img_alpha.jpg", "bud")
    assert _read_negatives(dataset, "bud") == ({"img_alpha.jpg"}, set())

    response = _save_via_route(client, dataset, BUD_THREE_STATES)

    assert _read_negatives(dataset, "bud") == (set(), {"img_alpha.jpg"})
    assert response["schema_change_sweep"]["newly_stamped"] == {"bud": 1}
    assert response["schema_change_sweep"]["warning"] is None


def test_the_class_map_tool_marks_an_unstamped_confirmation_as_predating_the_schema_change(
    dataset: Path
) -> None:
    """The agent's own registry writer sweeps identically: one implementation, so the GUI and the
    tool cannot disagree about which confirmations predate a vocabulary change."""
    _save_via_tool(dataset, BUD_TWO_STATES)
    _confirm_negative_unstamped(dataset, "img_alpha.jpg", "bud")
    assert _read_negatives(dataset, "bud") == ({"img_alpha.jpg"}, set())

    result = _save_via_tool(dataset, BUD_THREE_STATES)

    assert _read_negatives(dataset, "bud") == (set(), {"img_alpha.jpg"})
    assert result["schema_change_sweep"]["newly_stamped"] == {"bud": 1}
    assert result["schema_change_sweep"]["warning"] is None


def test_a_confirmation_that_carries_a_stamp_keeps_it_across_later_schema_changes(
    client: TestClient, dataset: Path
) -> None:
    """The stamp set at confirmation time is the direct evidence of what a human saw, so the sweep
    records only what is missing: a confirmation stamped two vocabularies ago still carries that
    digest, and one made under the vocabulary now current still admits."""
    _save_via_route(client, dataset, BUD_TWO_STATES)
    _confirm_negative_stamped(client, dataset, "img_alpha.jpg", "bud")
    two_state_digest = attribute_schema_digest(
        registry_from_dict(_subjects(BUD_TWO_STATES)), "bud")
    assert _stamps(dataset, "bud") == {"img_alpha.jpg": two_state_digest}

    first = _save_via_route(client, dataset, BUD_THREE_STATES)
    second = _save_via_route(client, dataset, BUD_FOUR_STATES)
    _confirm_negative_stamped(client, dataset, "img_gamma.jpg", "bud")

    assert first["schema_change_sweep"]["newly_stamped"] == {}
    assert second["schema_change_sweep"]["newly_stamped"] == {}
    assert _stamps(dataset, "bud")["img_alpha.jpg"] == two_state_digest
    assert _read_negatives(dataset, "bud") == ({"img_gamma.jpg"}, {"img_alpha.jpg"})


def test_a_subject_whose_schema_is_unchanged_keeps_its_unstamped_confirmations(
    client: TestClient, dataset: Path
) -> None:
    """The sweep is scoped to the subjects whose digest actually changes: another subject's
    confirmations are neither stamped nor quarantined by a change they had no part in."""
    _save_via_route(client, dataset, BUD_TWO_STATES)
    _confirm_negative_unstamped(dataset, "img_beta.jpg", "bush")

    response = _save_via_route(client, dataset, BUD_THREE_STATES)

    assert "bush" not in response["schema_change_sweep"]["newly_stamped"]
    assert _stamps(dataset, "bush") == {}
    assert _read_negatives(dataset, "bush") == ({"img_beta.jpg"}, set())


def test_a_schema_change_with_nothing_to_stamp_writes_the_registry_and_no_digest_store(
    client: TestClient, dataset: Path
) -> None:
    """A dataset with no confirmations has nothing to record, so the sweep leaves no digest store
    behind and the registry write is unaffected."""
    _save_via_route(client, dataset, BUD_TWO_STATES)

    response = _save_via_route(client, dataset, BUD_THREE_STATES)

    assert response["schema_change_sweep"] == {
        "newly_stamped": {}, "predating_vocabulary": {}, "warning": None}
    assert not image_status_digest_path(dataset).exists()
    saved = json.loads(classes_path(dataset).read_text(encoding="utf-8"))
    assert saved["bud"]["attributes"]["opening"]["values"] == \
        ["closed", "partial", "open"]


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
    _save_via_route(client, dataset, BUD_TWO_STATES)
    _confirm_negative_unstamped(dataset, "img_alpha.jpg", "bud")
    _block_the_digest_store(dataset)

    response = _save_via_route(client, dataset, BUD_THREE_STATES)

    assert response["status"] == "ok"
    assert response["schema_change_sweep"]["newly_stamped"] == {}
    assert "re-review them before they train" in response["schema_change_sweep"]["warning"]
    saved = json.loads(classes_path(dataset).read_text(encoding="utf-8"))
    assert saved["bud"]["attributes"]["opening"]["values"] == \
        ["closed", "partial", "open"]


def test_the_class_map_tool_writes_the_registry_and_reports_a_sweep_it_could_not_complete(
    dataset: Path
) -> None:
    """The tool warns and proceeds on the same terms as the route, through the same sweep.

    Bound to the file backend: the failure is simulated by occupying the digest store's own file
    path with a directory, which is a file-backend write obstruction, not a backend-general one.
    """
    ts.bind(FileBackend())
    _save_via_tool(dataset, BUD_TWO_STATES)
    _confirm_negative_unstamped(dataset, "img_alpha.jpg", "bud")
    _block_the_digest_store(dataset)

    result = _save_via_tool(dataset, BUD_THREE_STATES)

    assert result["schema_change_sweep"]["newly_stamped"] == {}
    assert "re-review them before they train" in result["schema_change_sweep"]["warning"]
    saved = json.loads(classes_path(dataset).read_text(encoding="utf-8"))
    assert saved["bud"]["attributes"]["opening"]["values"] == \
        ["closed", "partial", "open"]


def test_a_confirmation_stamped_under_current_code_reads_as_predating_the_next_change(
    client: TestClient, dataset: Path
) -> None:
    """The bug the predating_vocabulary count exists to fix: a confirmation made through the GUI
    is already stamped with the schema in effect at the time (never unstamped), so a later change
    left it silently uncounted by newly_stamped. It has to surface here instead."""
    _save_via_route(client, dataset, BUD_TWO_STATES)
    _confirm_negative_stamped(client, dataset, "img_alpha.jpg", "bud")

    response = _save_via_route(client, dataset, BUD_THREE_STATES)

    assert response["schema_change_sweep"]["newly_stamped"] == {}
    assert response["schema_change_sweep"]["predating_vocabulary"] == {"bud": 1}


def test_the_class_map_tool_reports_predating_vocabulary_the_same_way(dataset: Path) -> None:
    """One implementation: the tool's own writer reports the same count for the same case."""
    _save_via_tool(dataset, BUD_TWO_STATES)
    from tcip_mcp.class_registry import attribute_schema_digest, registry_from_dict
    from tcip_mcp.dataset_layout import stamp_image_status_digests

    record_image_statuses(dataset, status_bucket("bud", None), {"img_alpha.jpg": "negative"},
                          recorded_by="user:breeder")
    two_state_digest = attribute_schema_digest(
        registry_from_dict(_subjects(BUD_TWO_STATES)), "bud")
    stamp_image_status_digests(
        dataset, status_bucket("bud", None), ["img_alpha.jpg"], two_state_digest)

    result = _save_via_tool(dataset, BUD_THREE_STATES)

    assert result["schema_change_sweep"]["newly_stamped"] == {}
    assert result["schema_change_sweep"]["predating_vocabulary"] == {"bud": 1}


def test_predating_vocabulary_counts_a_complete_confirmation_too_not_only_negatives(
    client: TestClient, dataset: Path
) -> None:
    """The sweep stamps every status in a subject's buckets, and the predating count follows the
    same scope over the finished statuses: a stale complete confirmation counts exactly like a
    stale negative."""
    _save_via_route(client, dataset, BUD_TWO_STATES)
    _confirm_complete_stamped(client, dataset, "img_beta.jpg", "bud")

    response = _save_via_route(client, dataset, BUD_THREE_STATES)

    assert response["schema_change_sweep"]["predating_vocabulary"] == {"bud": 1}


def test_the_save_route_marks_an_unstamped_complete_as_predating_the_schema_change_too(
    client: TestClient, dataset: Path
) -> None:
    """newly_stamped follows the same finished-status scope as predating_vocabulary: an unstamped
    complete confirmation is stamped, and counted, exactly like an unstamped negative."""
    _save_via_route(client, dataset, BUD_TWO_STATES)
    record_image_statuses(dataset, status_bucket("bud", None), {"img_alpha.jpg": "complete"},
                          recorded_by="user:breeder")
    assert not ts.exists(image_status_digest_key(dataset))

    response = _save_via_route(client, dataset, BUD_THREE_STATES)

    assert response["schema_change_sweep"]["newly_stamped"] == {"bud": 1}


def test_neither_sweep_count_includes_a_stamped_partial(
    client: TestClient, dataset: Path
) -> None:
    """A partial is not a human's assertion, so a stale stamp on one is harmless and neither
    count, which exist to track finished confirmations, ever reports it."""
    _save_via_route(client, dataset, BUD_TWO_STATES)
    record_image_statuses(dataset, status_bucket("bud", None), {"img_alpha.jpg": "partial"},
                          recorded_by="user:breeder")
    two_state_digest = attribute_schema_digest(
        registry_from_dict(_subjects(BUD_TWO_STATES)), "bud")
    stamp_image_status_digests(dataset, status_bucket("bud", None), ["img_alpha.jpg"],
                               two_state_digest)

    response = _save_via_route(client, dataset, BUD_THREE_STATES)

    assert "bud" not in response["schema_change_sweep"]["newly_stamped"]
    assert "bud" not in response["schema_change_sweep"]["predating_vocabulary"]


def test_predating_vocabulary_counts_only_the_subject_whose_schema_actually_changed(
    client: TestClient, dataset: Path
) -> None:
    """Another subject's confirmations are neither stamped nor counted by a change they had no
    part in, the same scoping newly_stamped already applies."""
    _save_via_route(client, dataset, BUD_TWO_STATES)
    _confirm_negative_stamped(client, dataset, "img_beta.jpg", "bush")

    response = _save_via_route(client, dataset, BUD_THREE_STATES)

    assert "bush" not in response["schema_change_sweep"]["predating_vocabulary"]


def test_the_save_route_records_predating_vocabulary_in_its_audit_line(
    client: TestClient, dataset: Path
) -> None:
    """The audit line for the change carries the same fact the response and the toast do, beside
    the existing newly_stamped mapping, rather than recording a change that reads as harmless."""
    _save_via_route(client, dataset, BUD_TWO_STATES)
    _confirm_negative_stamped(client, dataset, "img_alpha.jpg", "bud")

    _save_via_route(client, dataset, BUD_THREE_STATES)

    entries = _audit_entries(dataset)
    save_entries = [e for e in entries if e["tool"] == "gui_save_classes"]
    assert save_entries[-1]["arguments"]["confirmations_predating_vocabulary"] == {"bud": 1}
    assert save_entries[-1]["arguments"]["confirmations_stamped_with_outgoing_schema"] == {}
