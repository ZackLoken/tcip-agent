"""The breeder's own surface for an operationalization record: read it, confirm it, withdraw it.

The confirmation is the one act no agent may perform, so these cases pin the seam from the HTTP
side: the route refuses a confirmation of something nobody stated, refuses a click aimed at text
that moved, records whether the request named the person at all, and leaves an audit entry naming
who it was recorded as. They also pin that the surface can enumerate the project's records, that
the project root a request names is confined and then resolved once, and that no MCP tool reaches
the writer behind any of it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import tcip_store as ts
from tcip_mcp import operationalization as op
from tcip_mcp.audit import audit_log_key
from tcip_web.app import app
from tcip_web.state import store
from tests import _operationalization_fixtures as fx

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "packages" / "tcip-mcp" / "src" / "tcip_mcp" / "tools"
ROUTES_MODULE = (
    REPO_ROOT / "packages" / "tcip-web" / "src" / "tcip_web" / "routes" / "results.py"
)

GET_ROUTE = "/api/results/operationalization"
LIST_ROUTE = "/api/results/operationalizations"
CONFIRM_ROUTE = "/api/results/operationalization/confirm"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project holding both fixture traits, deliberately not the process-pinned root.

    The pinned root is this test's ``tmp_path``; the project sits under it, so a route that read
    the record from the pin rather than from the root the request names finds nothing.
    """
    return fx.seed_project(tmp_path / "project")


def _read(client: TestClient, project: Path, trait: str, delivery_kind: str):
    return client.get(
        GET_ROUTE,
        params={
            "project_root": str(project),
            "trait": trait,
            "delivery_kind": delivery_kind,
        },
    )


def _confirm(client: TestClient, project: Path, trait: str, kind: str, seen: str, **extra):
    body = {
        "project_root": str(project),
        "trait": trait,
        "delivery_kind": kind,
        "record_seen": seen,
    }
    body.update(extra)
    return client.post(CONFIRM_ROUTE, json=body)


def _audit_entries(project: Path) -> list[dict]:
    return list(ts.read_log(audit_log_key(project)).records)


# ── reading the record ───────────────────────────────────────────────────────


def test_the_read_route_shows_the_statement_the_definitions_and_the_confirmation_state(
    client: TestClient, project: Path
) -> None:
    """The panel's whole content in one answer, including the vocabulary's own wording."""
    record = fx.state_crossing(project)

    body = _read(client, project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES).json()

    assert body["statement"] == record["statement"]
    assert body["mechanism"] == record["mechanism"]
    assert body["measured_subject"] == record["measured_subject"]
    assert body["stated_by"] == op.STATEMENT_SURFACE
    assert body["confirmed_by"] is None
    assert body["confirmed_current"] is False
    assert body["record_seen"] == op.record_seen_hash(record)
    assert [entry["name"] for entry in body["delivers"]] == list(fx.CROSSING_SPEC.delivers)
    assert all(isinstance(entry["definition"], str) for entry in body["delivers"])


def test_the_read_route_answers_for_a_trait_nothing_is_stated_for(
    client: TestClient, project: Path
) -> None:
    """An unstated pair is a real answer, not an error: the panel needs to say nothing is on file."""
    body = _read(client, project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES).json()

    assert body["statement"] is None
    assert body["confirmed_current"] is False
    assert body["superseded"] == []


def test_the_record_routes_refuse_an_unregistered_trait(
    client: TestClient, project: Path
) -> None:
    """A trait this project's registry does not carry is refused in the registry's own words."""
    read = _read(client, project, "not_a_registered_trait", op.STATE_CROSSING_DATES)
    confirmed = _confirm(
        client, project, "not_a_registered_trait", op.STATE_CROSSING_DATES, "any hash"
    )

    assert read.status_code == 400
    assert "not_a_registered_trait" in read.json()["detail"]
    assert confirmed.status_code == 400
    assert "not_a_registered_trait" in confirmed.json()["detail"]


def test_the_read_route_refuses_a_delivery_kind_the_platform_does_not_declare(
    client: TestClient, project: Path
) -> None:
    resp = _read(client, project, fx.CROSSING_TRAIT, "per_orchard_vibe")

    assert resp.status_code == 400
    assert "unknown delivery kind" in resp.json()["detail"]


# ── confirming ───────────────────────────────────────────────────────────────


def test_confirm_route_refuses_when_nothing_is_stated(
    client: TestClient, project: Path
) -> None:
    """A confirmation of nothing would record a breeder confirming a meaning no one wrote down."""
    resp = _confirm(
        client, project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, "a hash of nothing"
    )

    assert resp.status_code == 400
    assert "state_trait_operationalization" in resp.json()["detail"]
    _, stored, _ = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)
    assert stored.value is None


def test_confirm_route_returns_409_when_the_seen_text_moved(
    client: TestClient, project: Path
) -> None:
    """A record rewritten while the panel was open re-renders instead of harvesting the click."""
    seen = fx.state_crossing(project)
    restated = fx.state_crossing(project, statement="what the breeder actually meant")

    resp = _confirm(
        client,
        project,
        fx.CROSSING_TRAIT,
        op.STATE_CROSSING_DATES,
        op.record_seen_hash(seen),
        user="rosalind",
    )

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["record"]["statement"] == restated["statement"]
    assert detail["record"]["record_seen"] == op.record_seen_hash(restated)
    _, stored, _ = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)
    assert stored.value["confirmed_by"] is None

    assert _confirm(
        client,
        project,
        fx.CROSSING_TRAIT,
        op.STATE_CROSSING_DATES,
        detail["record"]["record_seen"],
        user="rosalind",
    ).status_code == 200


def test_identity_from_request_reflects_what_the_route_observed(
    client: TestClient, project: Path
) -> None:
    """The recorded bit is whether the request carried a name, which only the route can see.

    It is not a claim that the name is true: nothing here authenticates anybody. A request with no
    name falls back to the backend's own identity and records that it did.
    """
    named = _confirm(
        client,
        project,
        fx.CROSSING_TRAIT,
        op.STATE_CROSSING_DATES,
        op.record_seen_hash(fx.state_crossing(project)),
        user="rosalind",
    ).json()

    assert named["identity_from_request"] is True
    assert named["confirmed_by"] == "user:rosalind"

    fx.state_count(project)
    nameless = _confirm(
        client,
        project,
        fx.COUNT_TRAIT,
        op.PER_IMAGE_COUNT,
        _read(client, project, fx.COUNT_TRAIT, op.PER_IMAGE_COUNT).json()["record_seen"],
    ).json()

    assert nameless["identity_from_request"] is False
    assert nameless["confirmed_by"].startswith("user:")


def test_the_confirmation_is_recorded_under_the_name_once_not_twice(
    client: TestClient, project: Path
) -> None:
    """The record's writer owns the ``user:`` convention, so the route hands it a bare name."""
    fx.state_crossing(project)
    seen = _read(client, project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES).json()["record_seen"]

    confirmed = _confirm(
        client, project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, seen, user="user:rosalind"
    ).json()

    assert confirmed["confirmed_by"] == "user:rosalind"


def test_withdrawing_clears_the_confirmation_without_touching_the_statement(
    client: TestClient, project: Path
) -> None:
    """Withdrawal is the same door with ``confirmed`` false: nothing is deleted and nothing else asked."""
    record = fx.state_crossing(project)
    _confirm(
        client,
        project,
        fx.CROSSING_TRAIT,
        op.STATE_CROSSING_DATES,
        op.record_seen_hash(record),
        user="rosalind",
    )
    seen = _read(client, project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES).json()["record_seen"]

    withdrawn = _confirm(
        client,
        project,
        fx.CROSSING_TRAIT,
        op.STATE_CROSSING_DATES,
        seen,
        user="rosalind",
        confirmed=False,
    )

    assert withdrawn.status_code == 200
    assert all(withdrawn.json()[field] is None for field in op.CONFIRMATION_FIELDS)
    after = _read(client, project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES).json()
    assert after["confirmed_current"] is False
    assert after["statement"] == record["statement"]


def test_restating_clears_the_confirmation_and_the_routes_reflect_it(
    client: TestClient, project: Path
) -> None:
    """A changed definition is unconfirmed, and the surface must not keep showing it as confirmed."""
    record = fx.state_crossing(project)
    _confirm(
        client,
        project,
        fx.CROSSING_TRAIT,
        op.STATE_CROSSING_DATES,
        op.record_seen_hash(record),
        user="rosalind",
    )
    assert _read(client, project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES).json()[
        "confirmed_current"
    ] is True

    fx.state_crossing(project, mechanism="a different call entirely")

    after = _read(client, project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES).json()
    assert after["confirmed_current"] is False
    assert after["confirmed_by"] is None
    assert after["mechanism"] == "a different call entirely"
    listed = client.get(LIST_ROUTE, params={"project_root": str(project)}).json()["records"]
    assert [row["confirmed_current"] for row in listed] == [False]


def test_both_events_land_in_the_project_log_with_the_actor(
    client: TestClient, project: Path
) -> None:
    """Confirming and withdrawing are the project's own decisions, and both name who made them.

    The name is on the entry itself rather than inside the arguments, because the entry shape has
    no human-identity field and a reader must not have to guess which argument was the person.
    """
    record = fx.state_crossing(project)
    _confirm(
        client,
        project,
        fx.CROSSING_TRAIT,
        op.STATE_CROSSING_DATES,
        op.record_seen_hash(record),
        user="rosalind",
    )
    seen = _read(client, project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES).json()["record_seen"]
    _confirm(
        client,
        project,
        fx.CROSSING_TRAIT,
        op.STATE_CROSSING_DATES,
        seen,
        user="rosalind",
        confirmed=False,
    )

    entries = [
        e for e in _audit_entries(project)
        if e["tool"] == "results.confirm_trait_operationalization"
    ]
    assert len(entries) == 2
    assert [e["arguments"]["confirmed"] for e in entries] == [True, False]
    assert all(e["user"] == "user:rosalind" for e in entries)
    assert all(e["arguments"]["trait"] == fx.CROSSING_TRAIT for e in entries)


# ── the list ─────────────────────────────────────────────────────────────────


def test_the_list_route_returns_every_kinds_record_and_confirming_one_leaves_the_others(
    client: TestClient, project: Path
) -> None:
    """The panel enumerates the pairs that exist, and a confirmation is scoped to its own pair.

    A count record is listed and confirmable the same way a crossing record is, regardless of
    which delivery kind's door serves it, and confirming it must not read as a confirmation of
    the crossing record beside it.
    """
    fx.state_crossing(project)
    fx.state_count(project)
    fx.state_aggregate(project, op.PER_PLANT_COUNT_AGGREGATE)

    listed = client.get(LIST_ROUTE, params={"project_root": str(project)}).json()
    assert listed["unresolved"] == []
    by_pair = {(row["trait"], row["delivery_kind"]): row for row in listed["records"]}
    assert set(by_pair) == {
        (fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES),
        (fx.COUNT_TRAIT, op.PER_IMAGE_COUNT),
        (fx.COUNT_TRAIT, op.PER_PLANT_COUNT_AGGREGATE),
    }
    assert not any(row["confirmed_current"] for row in listed["records"])

    target = by_pair[(fx.COUNT_TRAIT, op.PER_IMAGE_COUNT)]
    assert _confirm(
        client, project, fx.COUNT_TRAIT, op.PER_IMAGE_COUNT, target["record_seen"], user="rosalind"
    ).status_code == 200

    after = client.get(LIST_ROUTE, params={"project_root": str(project)}).json()["records"]
    confirmed = {(row["trait"], row["delivery_kind"]) for row in after if row["confirmed_current"]}
    assert confirmed == {(fx.COUNT_TRAIT, op.PER_IMAGE_COUNT)}


def test_the_list_route_names_the_fields_a_confirmation_covers(
    client: TestClient, project: Path
) -> None:
    """The surface renders the hashed set the record module owns, not a list it keeps itself.

    Served whether or not anything is stated, because the panel needs the field order before it has
    a record to lay out.
    """
    empty = client.get(LIST_ROUTE, params={"project_root": str(project)}).json()
    assert empty["statement_fields"] == list(op.STATEMENT_FIELDS)

    fx.state_crossing(project)
    listed = client.get(LIST_ROUTE, params={"project_root": str(project)}).json()

    assert listed["statement_fields"] == list(op.STATEMENT_FIELDS)
    assert all(field in listed["records"][0] for field in listed["statement_fields"])


def test_the_list_route_names_a_record_whose_trait_is_no_longer_registered(
    client: TestClient, project: Path
) -> None:
    """A record nothing can resolve is named rather than dropped, since silence reads as none."""
    from tcip_mcp import traits

    fx.state_crossing(project)
    key = traits.trait_spec_key(traits.trait_specs_dir(project), fx.CROSSING_TRAIT)
    ts.delete(key, expect=ts.read_versioned(key).version)

    listed = client.get(LIST_ROUTE, params={"project_root": str(project)}).json()

    assert listed["records"] == []
    assert listed["unresolved"] == [
        {
            "trait": fx.CROSSING_TRAIT,
            "delivery_kind": op.STATE_CROSSING_DATES,
            "reason": listed["unresolved"][0]["reason"],
        }
    ]
    assert fx.CROSSING_TRAIT in listed["unresolved"][0]["reason"]


# ── the project root a request names ─────────────────────────────────────────


def test_a_project_root_under_an_allowed_root_still_answers(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The containment guard must admit the ordinary case, not only refuse the escaping one.

    The record is written and read back through a path carrying a dot segment, so a door resolving
    from the guard's returned path and one resolving from the raw string both have to reach it.
    """
    project = fx.seed_project(tmp_path / "workspace" / "project")
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(tmp_path / "workspace"))
    record = fx.state_crossing(project)
    dotted = str(project / ".." / "project")

    resp = client.get(
        GET_ROUTE,
        params={
            "project_root": dotted,
            "trait": fx.CROSSING_TRAIT,
            "delivery_kind": op.STATE_CROSSING_DATES,
        },
    )

    assert resp.status_code == 200
    assert resp.json()["statement"] == record["statement"]
    assert client.get(LIST_ROUTE, params={"project_root": dotted}).json()["records"]
    assert _confirm(
        client,
        Path(dotted),
        fx.CROSSING_TRAIT,
        op.STATE_CROSSING_DATES,
        op.record_seen_hash(record),
        user="rosalind",
    ).status_code == 200
    _, stored, _ = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)
    assert stored.value["confirmed_by"] == "user:rosalind"


def test_a_project_root_outside_the_allowed_roots_is_refused_at_every_results_door(
    client: TestClient, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every door taking a project root confines it, the record routes and the delivery doors alike."""
    allowed = tmp_path / "workspace"
    allowed.mkdir()
    outside = tmp_path_factory.mktemp("outside")
    fx.seed_project(outside)
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(allowed))
    store.open_project(tmp_path.resolve())

    phenology_body = {
        "project_root": str(outside),
        "mapping_name": "mapping",
        "predictions_by_date": {},
        "trait": fx.CROSSING_TRAIT,
    }
    assert client.post(
        "/api/results/phenology_measurement", json=phenology_body).status_code == 403
    assert client.post(
        "/api/results/export_csv", json={**phenology_body, "payload": "milestones"}
    ).status_code == 403
    assert _read(client, outside, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES).status_code == 403
    assert client.get(LIST_ROUTE, params={"project_root": str(outside)}).status_code == 403
    assert _confirm(
        client, outside, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, "any hash"
    ).status_code == 403


def test_a_delivery_door_still_runs_for_a_project_root_the_guard_admits(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same guard on the delivery doors must leave a legitimate request reaching the door.

    A missing plant mapping under an admitted root reports the missing mapping, which is the door
    answering rather than the guard refusing. The trait is stated and confirmed here so the answer
    comes from the door itself rather than from the precondition standing in front of it.
    """
    fx.write_spec(tmp_path, fx.CROSSING_SPEC)
    fx.seed_confirmed_crossing(tmp_path, fx.CROSSING_TRAIT)
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(tmp_path))
    store.open_project(tmp_path.resolve())

    resp = client.post(
        "/api/results/phenology_measurement",
        json={
            "project_root": str(tmp_path),
            "mapping_name": "mapping",
            "predictions_by_date": {},
            "trait": fx.CROSSING_TRAIT,
        },
    )

    assert resp.status_code == 404
    assert "mapping not found" in resp.json()["detail"]


# ── the seal between the two writers ─────────────────────────────────────────


def test_no_mcp_tool_reaches_the_confirmation_writer() -> None:
    """The agent has a statement tool and no confirmation tool, checked against the live registry.

    The registry comes from ``tools/list_tools.py`` rather than a hand-kept list, so a tool added
    later is covered without this test being edited.
    """
    writer = "confirm_trait_operationalization"
    listing = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "list_tools.py")],
        capture_output=True, text=True, cwd=REPO_ROOT, check=True,
    ).stdout
    registered = {line.strip() for line in listing.splitlines() if line.startswith("  ")}

    assert "state_trait_operationalization" in registered
    assert not [name for name in registered if writer in name]
    assert writer in ROUTES_MODULE.read_text(encoding="utf-8"), (
        "the writer's name moved, so this test would be searching for nothing"
    )
    assert not [
        module.name
        for module in TOOLS_DIR.glob("*.py")
        if writer in module.read_text(encoding="utf-8")
    ]


# ── the operationalization confirm route's audit line must not land silently (A8 retrofit) ───


class _AppendRefused(RuntimeError):
    """Stands in for whatever stops a real append: a busy lock, a refused root, a bad key."""


def _refuse_append(*args: object, **kwargs: object) -> None:
    raise _AppendRefused("the audit log could not be appended to")


def test_confirm_operationalization_still_returns_200_with_a_warning_when_the_audit_append_fails(
    client: TestClient, project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A8: the confirmation already committed by the time the audit append can fail, so a lost
    line must not read as a refused confirmation. It rides back as ``audit_warning`` on an
    ordinary 200 whose confirmation fields are populated exactly as a healthy append's would be."""
    import tcip_mcp.audit as audit_module

    record = fx.state_crossing(project)
    monkeypatch.setattr(audit_module, "append", _refuse_append)

    resp = _confirm(
        client, project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES,
        op.record_seen_hash(record), user="rosalind",
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["confirmed_by"] == "user:rosalind"
    assert body["audit_warning"] is not None
    assert "do not retry it blind" in body["audit_warning"]
    _, stored, _ = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)
    assert stored.value["confirmed_by"] == "user:rosalind"


def test_confirm_operationalization_carries_no_warning_against_a_healthy_log(
    client: TestClient, project: Path,
) -> None:
    """The rail must admit the work it exists beside: a healthy append leaves the field null."""
    record = fx.state_crossing(project)

    resp = _confirm(
        client, project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES,
        op.record_seen_hash(record), user="rosalind",
    )

    assert resp.status_code == 200
    assert resp.json()["audit_warning"] is None


# ── trait-spec authoring statements: the same confirmation surface, generalized ──────────────


GET_STATEMENT_ROUTE = "/api/results/trait-spec-statement"
LIST_STATEMENT_ROUTE = "/api/results/trait-spec-statements"
CONFIRM_STATEMENT_ROUTE = "/api/results/trait-spec-statement/confirm"
DELIVERY_EVENTS_ROUTE = "/api/results/delivery-events"

STATEMENT_TRAIT = "astringency"


def _author_statement(root: Path, trait: str = STATEMENT_TRAIT, **overrides: object) -> dict:
    from tcip_mcp import traits

    fields: dict[str, object] = dict(
        delivers=(trait,),
        rationale="the breeder described the state directly, in their own field-scoring terms",
    )
    fields.update(overrides)
    return traits.author_trait_spec(str(root), trait, **fields)


def _read_statement(client: TestClient, root: Path, trait: str):
    return client.get(GET_STATEMENT_ROUTE, params={"project_root": str(root), "trait": trait})


def _confirm_statement(client: TestClient, root: Path, trait: str, seen: str, **extra: object):
    body = {"project_root": str(root), "trait": trait, "record_seen": seen}
    body.update(extra)
    return client.post(CONFIRM_STATEMENT_ROUTE, json=body)


def test_the_statement_read_route_shows_the_authored_fields_and_the_confirmation_state(
    client: TestClient, tmp_path: Path,
) -> None:
    from tcip_mcp import traits

    statement = _author_statement(tmp_path)

    body = _read_statement(client, tmp_path, STATEMENT_TRAIT).json()

    assert body["trait"] == STATEMENT_TRAIT
    assert body["statement_fields"] == statement["statement_fields"]
    assert body["rationale"] == statement["rationale"]
    assert body["stated_by"] == traits.TRAIT_SPEC_STATEMENT_SURFACE
    assert body["confirmed_by"] is None
    assert body["confirmed_current"] is False
    assert body["record_seen"] == traits.trait_spec_statement_seen_hash(statement)


def test_the_statement_read_route_answers_for_a_trait_nothing_is_stated_for(
    client: TestClient, tmp_path: Path,
) -> None:
    """A registered spec with no statement behind it (S4/A1's own recovery state) is a real
    answer, not an error: the panel needs to say nothing is on file."""
    import tcip_store as ts
    from tcip_mcp import traits

    directory = traits.trait_specs_dir(str(tmp_path))
    key = traits.trait_spec_key(directory, "unstated_trait")
    ts.replace(key, {"name": "unstated_trait", "delivers": ["astringency"]},
               expect=ts.Version.ABSENT)

    resp = _read_statement(client, tmp_path, "unstated_trait")
    body = resp.json()

    assert resp.status_code == 200
    assert body["statement_fields"] is None
    assert body["rationale"] is None
    assert body["confirmed_by"] is None
    assert body["confirmed_current"] is False


def test_the_statement_read_route_refuses_an_unregistered_trait(
    client: TestClient, tmp_path: Path,
) -> None:
    resp = _read_statement(client, tmp_path, "not_a_registered_trait")

    assert resp.status_code == 400
    assert "not_a_registered_trait" in resp.json()["detail"]


def test_confirm_statement_route_refuses_when_nothing_is_stated(
    client: TestClient, tmp_path: Path,
) -> None:
    resp = _confirm_statement(client, tmp_path, "no_such_trait", "a hash of nothing")

    assert resp.status_code == 400
    assert "author_trait_spec" in resp.json()["detail"]


def test_confirm_statement_route_returns_409_when_the_record_moved(
    client: TestClient, tmp_path: Path,
) -> None:
    """A statement rewritten while the panel was open re-renders instead of harvesting the click."""
    import tcip_store as ts
    from tcip_mcp import traits

    statement = _author_statement(tmp_path)
    seen = traits.trait_spec_statement_seen_hash(statement)

    # A second write, simulated directly since the subject here is the confirm route, not the
    # writer that would otherwise produce this state.
    scope = traits.trait_spec_statements_scope(tmp_path)
    key = traits.trait_spec_statement_key(scope, STATEMENT_TRAIT)
    moved = dict(statement)
    moved["rationale"] = "a different rationale entirely"
    ts.replace(key, moved, expect=ts.read_versioned(key).version)

    resp = _confirm_statement(client, tmp_path, STATEMENT_TRAIT, seen, user="rosalind")

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["kind"] == "trait_spec_authoring"
    assert detail["record"]["rationale"] == "a different rationale entirely"
    assert detail["record"]["record_seen"] == traits.trait_spec_statement_seen_hash(moved)

    assert _confirm_statement(
        client, tmp_path, STATEMENT_TRAIT, detail["record"]["record_seen"], user="rosalind",
    ).status_code == 200


def test_confirm_statement_route_confirms_and_withdraws(
    client: TestClient, tmp_path: Path,
) -> None:
    from tcip_mcp import traits

    statement = _author_statement(tmp_path)
    seen = traits.trait_spec_statement_seen_hash(statement)

    confirmed = _confirm_statement(client, tmp_path, STATEMENT_TRAIT, seen, user="rosalind")
    assert confirmed.status_code == 200
    body = confirmed.json()
    assert body["confirmed_by"] == "user:rosalind"
    assert body["audit_warning"] is None

    after_read = _read_statement(client, tmp_path, STATEMENT_TRAIT).json()
    assert after_read["confirmed_current"] is True

    withdraw_seen = after_read["record_seen"]
    withdrawn = _confirm_statement(
        client, tmp_path, STATEMENT_TRAIT, withdraw_seen, user="rosalind", confirmed=False,
    )
    assert withdrawn.status_code == 200
    assert all(
        withdrawn.json()[f] is None for f in traits.TRAIT_SPEC_CONFIRMATION_FIELDS
    )


def test_confirm_statement_route_still_returns_200_with_a_warning_when_the_audit_append_fails(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A8, proven for the new route: the write to ``trait_spec_statements`` already committed by
    the time the audit append can fail, so a lost line must not read as a refused confirmation."""
    import tcip_mcp.audit as audit_module
    from tcip_mcp import traits

    statement = _author_statement(tmp_path)
    seen = traits.trait_spec_statement_seen_hash(statement)
    monkeypatch.setattr(audit_module, "append", _refuse_append)

    resp = _confirm_statement(client, tmp_path, STATEMENT_TRAIT, seen, user="rosalind")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["confirmed_by"] == "user:rosalind"
    assert body["audit_warning"] is not None
    assert "do not retry it blind" in body["audit_warning"]

    scope = traits.trait_spec_statements_scope(tmp_path)
    key = traits.trait_spec_statement_key(scope, STATEMENT_TRAIT)
    import tcip_store as ts

    assert ts.read(key)["confirmed_by"] == "user:rosalind"


def test_the_statement_list_route_returns_every_traits_row(
    client: TestClient, tmp_path: Path,
) -> None:
    from tcip_mcp import traits

    _author_statement(tmp_path, trait="astringency", delivers=("astringency",))
    _author_statement(tmp_path, trait="fruit_diameter", delivers=("fruit_diameter",))

    listed = client.get(LIST_STATEMENT_ROUTE, params={"project_root": str(tmp_path)}).json()

    assert listed["unresolved"] == []
    by_trait = {row["trait"] for row in listed["records"]}
    assert by_trait == {"astringency", "fruit_diameter"}
    assert listed["statement_fields"] == list(traits.TRAIT_SPEC_STATEMENT_FIELDS)


def test_the_statement_list_route_names_a_record_whose_trait_is_no_longer_registered(
    client: TestClient, tmp_path: Path,
) -> None:
    from tcip_mcp import traits

    _author_statement(tmp_path)
    key = traits.trait_spec_key(traits.trait_specs_dir(tmp_path), STATEMENT_TRAIT)
    import tcip_store as ts

    ts.delete(key, expect=ts.read_versioned(key).version)

    listed = client.get(LIST_STATEMENT_ROUTE, params={"project_root": str(tmp_path)}).json()

    assert listed["records"] == []
    assert listed["unresolved"][0]["trait"] == STATEMENT_TRAIT


def test_no_mcp_tool_reaches_the_trait_spec_confirmation_writer() -> None:
    """The agent has an authoring tool and no confirmation tool, checked against the live
    registry rather than a hand-kept list, so a tool added later is covered automatically."""
    writer = "confirm_trait_spec"
    listing = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "list_tools.py")],
        capture_output=True, text=True, cwd=REPO_ROOT, check=True,
    ).stdout
    registered = {line.strip() for line in listing.splitlines() if line.startswith("  ")}

    assert "author_trait_spec" in registered
    assert not [name for name in registered if writer in name]
    assert writer in ROUTES_MODULE.read_text(encoding="utf-8"), (
        "the writer's name moved, so this test would be searching for nothing"
    )
    assert not [
        module.name
        for module in TOOLS_DIR.glob("*.py")
        if writer in module.read_text(encoding="utf-8")
    ]


# ── delivery events: a read-only fact, not a statement ───────────────────────────────────────


def test_delivery_events_route_lists_a_recorded_event(client: TestClient, tmp_path: Path) -> None:
    from tcip_mcp.operationalization import PER_IMAGE_COUNT
    from tcip_mcp.pipelines.resolution import record_delivery_binding_event

    record_delivery_binding_event(
        "test_door", None, [], {}, measurement_documents=["operating_point"],
        scale_document=None, acknowledgement=None, trait=STATEMENT_TRAIT,
        delivery_kind=PER_IMAGE_COUNT, project_root=tmp_path,
    )

    resp = client.get(DELIVERY_EVENTS_ROUTE, params={"project_root": str(tmp_path)})

    assert resp.status_code == 200
    records = resp.json()["records"]
    assert len(records) == 1
    assert records[0]["door"] == "test_door"
    assert records[0]["trait"] == STATEMENT_TRAIT
    assert records[0]["delivery_kind"] == PER_IMAGE_COUNT


def test_delivery_events_route_confines_project_root_to_allowed_roots(
    client: TestClient, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path_factory.mktemp("outside")
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(allowed))

    resp = client.get(DELIVERY_EVENTS_ROUTE, params={"project_root": str(outside)})

    assert resp.status_code == 403


def test_delivery_events_route_refuses_a_stored_event_whose_plant_mapping_predates_the_disclosure(
    client: TestClient, tmp_path: Path,
) -> None:
    """A record written before ``plant_mapping`` carried ``dates_delivered``,
    ``images_unattributed`` and ``plant_attribution`` cannot be read back as if those keys meant
    nothing: none of the three is reconstructable from the rest of the record, so the route
    refuses the whole listing by event_id rather than serving it with the gap silently absent."""
    from tcip_mcp.pipelines import resolution

    event_id = "old-shaped"
    key = resolution.delivery_event_key(resolution.delivery_events_scope(tmp_path), event_id)
    ts.replace(
        key,
        {
            "event_id": event_id,
            "trait": STATEMENT_TRAIT,
            "delivery_kind": "state_crossing_dates",
            "door": "compute_phenology",  # a stored value written before the rename
            "output_path": None,
            "measurement_documents": ["operating_point"],
            "scale_document": None,
            "plant_mapping": {
                "name": "valley",
                "project_root": str(tmp_path),
                "dataset_id": "ds-1",
                "dataset_root": "C:/data",
                "built_at": "2026-02-01T00:00:00+00:00",
                "record_sha256": "0" * 64,
                "nn_tolerance_m": {"value": 3, "source": "stated"},
                "capture_identity": {},
                "captures_unverified": [],
                "plant_csvs_unverified": [],
                "images_unattributed_scope": "delivered_dates",
            },
            "documents": {},
            "produced_at": "2026-02-03T12:00:00+00:00",
        },
        expect=ts.Version.ABSENT,
    )

    resp = client.get(DELIVERY_EVENTS_ROUTE, params={"project_root": str(tmp_path)})

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert event_id in detail
    assert "no operator door rewrites an existing delivery_events record" in detail


def test_delivery_events_route_serves_a_real_plant_mapping_disclosure_with_all_three_keys(
    client: TestClient, tmp_path: Path,
) -> None:
    """A ``plant_mapping`` built the way a real delivery builds it (``MappingBuild.
    delivery_disclosure``) carries all three keys, and the route serves them through unchanged."""
    from datetime import datetime, timezone

    from tcip_mcp.pipelines.postprocessing.plant_mapping import MappingBuild
    from tcip_mcp.pipelines.resolution import record_delivery_binding_event

    dates = ["2026-01-01", "2026-01-08"]
    build = MappingBuild(
        name="valley", project_root=str(tmp_path), dataset_root=str(tmp_path / "data"),
        dataset_id="ds-1", built_by="build_plant_mapping",
        built_at=datetime.now(timezone.utc).isoformat(),
        dates_requested=None, dates=dates,
        nn_tolerance_m={"value": 3.0, "source": "stated"},
        plant_registry={"name": "unregistered", "digest": "0" * 64},
        capture_identity={d: "0" * 16 for d in dates},
        capture_digests={d: {} for d in dates}, unreadable={d: [] for d in dates},
        assignments={}, record_sha256="0" * 64,
    )
    disclosure = build.delivery_disclosure(
        {"captures_unverified": [], "plant_csvs_unverified": []}, dates)

    record_delivery_binding_event(
        "test_door", None, [], {}, measurement_documents=["operating_point"],
        scale_document=None, acknowledgement=None, trait=STATEMENT_TRAIT,
        delivery_kind="state_crossing_dates", project_root=tmp_path, plant_mapping=disclosure,
    )

    resp = client.get(DELIVERY_EVENTS_ROUTE, params={"project_root": str(tmp_path)})

    assert resp.status_code == 200
    records = resp.json()["records"]
    assert len(records) == 1
    mapping = records[0]["plant_mapping"]
    assert mapping["dates_delivered"] == dates
    assert mapping["images_unattributed"] == 0
    assert mapping["plant_attribution"] == "image"
