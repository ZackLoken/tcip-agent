"""One trait record with revisions: proposed through one tool, confirmed through one door, and
named by every delivery that ships under it.

A proposal appends an unconfirmed revision and nothing ships under it; the breeder's confirmation
of one revision, by number and the hash of the entry shown, covers its spec fields and every
operationalization it states together; a later proposal leaves deliveries under the earlier
confirmed revision until it is confirmed itself; the delivery event names the revision.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import tcip_store as ts
from tcip_mcp import traits
from tcip_mcp.audit import audit_log_key
from tcip_mcp.delivery import read_delivery_events
from tcip_mcp.operationalization import OperationalizationRefused
from tcip_web.app import app
from tests import _trait_fixtures as fx

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "packages" / "tcip-mcp" / "src" / "tcip_mcp" / "tools"
ROUTES_MODULE = REPO_ROOT / "packages" / "tcip-web" / "src" / "tcip_web" / "routes" / "results.py"
TRAITS_ROUTE = "/api/results/traits"
CONFIRM_ROUTE = "/api/results/traits/confirm"
_SCOPE = {"subject": fx.COUNT_SUBJECT}


@pytest.fixture
def client(opened_project: Path) -> TestClient:
    """A client of the backend with ``tmp_path`` open as its project."""
    return TestClient(app, base_url="http://127.0.0.1")


def _count_entry(**fields) -> traits.TraitEntry:
    return fx.with_operationalization(
        fx.COUNT_SPEC, traits.PER_IMAGE_COUNT, measured_subject=fx.COUNT_SUBJECT, **fields)


def _bucket(root: Path) -> Path:
    """``root``'s one unassessed bucket of three detections on one frame, published on first use
    (``_chain_fixtures.published``)."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import predicted, published

    bucket = root / "ds" / "predictions" / "counts" / "2026-02-11"
    if not (bucket / "bucket.json").exists():
        published(root, bucket, [predicted("a", [fx.COUNT_SUBJECT] * 3)], scope=_SCOPE)
    return bucket


def _deliver_counts(root: Path, name: str) -> dict:
    """One per-image count delivery of the count trait at ``root``, returning its event."""
    from tcip_mcp.pipelines.postprocessing.export import deliver_per_image_counts_csv

    from tests._chain_fixtures import acknowledged

    out = root / f"{name}.csv"
    bucket = _bucket(root)
    acknowledged(root, lambda ack: deliver_per_image_counts_csv(
        root, bucket, str(out), trait=fx.COUNT_TRAIT, acknowledgment_id=ack,
        door="test_trait_revisions"), by="user:tester", reason="no assessment backs these counts")
    (event,) = [e for e in read_delivery_events(root) if e.output_path == str(out)]
    return event.model_dump(mode="json")


def _confirm(client: TestClient, revision: traits.TraitRevision, **extra):
    body = {"trait": revision.entry.name,
            "revision": revision.number, "entry_sha256": revision.entry_sha256,
            "confirmed": True, "user": "breeder", **extra}
    return client.post(CONFIRM_ROUTE, json=body)


# ── the revision record ──────────────────────────────────────────────────────


def test_a_proposal_appends_an_unconfirmed_revision_and_a_delivery_refuses_by_name(
    tmp_path: Path,
) -> None:
    revision = fx.propose(tmp_path, _count_entry())

    assert revision.number == 1
    assert not revision.confirmed
    with pytest.raises(OperationalizationRefused) as excinfo:
        _deliver_counts(tmp_path, "unconfirmed")
    assert "none of its 1 revision(s) is confirmed" in str(excinfo.value)
    assert "Setup tab" in str(excinfo.value)
    assert not (tmp_path / "unconfirmed.csv").exists()


def test_a_confirmed_revision_delivers_and_the_event_names_it(tmp_path: Path) -> None:
    revision = fx.propose_and_confirm(tmp_path, _count_entry())

    event = _deliver_counts(tmp_path, "first")

    assert event["trait"] == fx.COUNT_TRAIT
    assert event["trait_revision"] == 1
    assert event["trait_revision_sha256"] == revision.entry_sha256
    assert event["delivery_kind"] == traits.PER_IMAGE_COUNT


def test_a_later_proposal_leaves_deliveries_under_the_confirmed_one_until_it_is_confirmed(
    tmp_path: Path,
) -> None:
    first = fx.propose_and_confirm(tmp_path, _count_entry())
    second = fx.propose(tmp_path, _count_entry(statement="stems per frame, broken tips excluded"))

    assert second.number == 2 and not second.confirmed
    assert second.entry_sha256 != first.entry_sha256
    assert traits.read_trait(fx.COUNT_TRAIT, tmp_path).revisions[0] == first
    under_first = _deliver_counts(tmp_path, "under_first")
    assert (under_first["trait_revision"], under_first["trait_revision_sha256"]) == (
        1, first.entry_sha256)

    fx.confirm(tmp_path, second)
    under_second = _deliver_counts(tmp_path, "under_second")

    assert (under_second["trait_revision"], under_second["trait_revision_sha256"]) == (
        2, second.entry_sha256)
    assert [e.trait_revision for e in read_delivery_events(tmp_path)
            if e.output_path == str(tmp_path / "under_first.csv")] == [1]


def test_one_confirmation_covers_every_delivery_kind_the_revision_states(tmp_path: Path) -> None:
    """Spec fields and the operationalization text of every kind are one entry and one
    confirmation: confirming revision 1 makes each kind it states deliverable."""
    from tcip_mcp.buckets import read_bucket
    from tests._chain_fixtures import deliver_acknowledged

    both = fx.with_operationalization(
        _count_entry(), traits.PER_PLANT_COUNT_AGGREGATE, delivered_phenotypes=("stem_count",),
        delivered_value_keys=("count",))
    revision = fx.propose_and_confirm(tmp_path, both)

    count_event = _deliver_counts(tmp_path, "per_image")
    deliver_acknowledged(
        tmp_path, [{"plant_id": "p1", "value": 5, "observations": 2, "value_key": "count",
                    "plant_attribution": "image"}],
        tmp_path / "per_plant.csv", "stem_count", delivery_kind=traits.PER_PLANT_COUNT_AGGREGATE,
        buckets=[read_bucket(_bucket(tmp_path))])

    kinds = {e.delivery_kind: e.trait_revision for e in read_delivery_events(tmp_path)}
    assert kinds == {traits.PER_IMAGE_COUNT: 1, traits.PER_PLANT_COUNT_AGGREGATE: 1}
    assert count_event["trait_revision_sha256"] == revision.entry_sha256


def test_a_withdrawn_confirmation_marks_the_revision_and_leaves_the_entry(tmp_path: Path) -> None:
    revision = fx.propose_and_confirm(tmp_path, _count_entry())

    withdrawn = traits.confirm_revision(
        tmp_path, fx.COUNT_TRAIT, 1, revision.entry_sha256, user="rosalind", confirmed=False)

    assert withdrawn.withdrawn_by == "user:rosalind" and not withdrawn.confirmed
    assert withdrawn.entry == revision.entry and withdrawn.confirmed_by == revision.confirmed_by
    with pytest.raises(OperationalizationRefused):
        _deliver_counts(tmp_path, "after_withdrawal")
    with pytest.raises(ValueError, match="already confirmed"):
        fx.confirm(tmp_path, withdrawn)


def test_a_withdrawn_later_revision_leaves_the_earlier_confirmed_one_answering(
    tmp_path: Path,
) -> None:
    fx.propose_and_confirm(tmp_path, _count_entry())
    second = fx.propose_and_confirm(tmp_path, _count_entry(statement="stems per frame, tips excluded"))
    traits.confirm_revision(tmp_path, fx.COUNT_TRAIT, 2, second.entry_sha256, user="rosalind",
                            confirmed=False)

    assert _deliver_counts(tmp_path, "back_to_first")["trait_revision"] == 1


def test_the_proposal_writes_one_audit_line_naming_the_agent(tmp_path: Path) -> None:
    from tcip_mcp import agent_identity

    agent_identity.begin("claude-code", "2.1.238")
    try:
        revision = fx.propose(tmp_path, _count_entry())
    finally:
        agent_identity.end()

    lines = [e for e in ts.read_log(audit_log_key(tmp_path)).records if e["tool"] == "propose_trait"]
    assert [line["arguments"] for line in lines] == [
        {"trait": fx.COUNT_TRAIT, "revision": 1, "entry_sha256": revision.entry_sha256}]
    assert lines[0]["agent_client_name"] == "claude-code"


# ── the schema, the one statement of what an entry holds ─────────────────────


def test_the_proposing_tool_takes_the_complete_entry_and_refuses_one_missing_a_field(
    tmp_path: Path,
) -> None:
    from pydantic import ValidationError

    from tcip_mcp.tools.trait_tools import propose_trait

    written = propose_trait(str(tmp_path), _count_entry(), rationale="the breeder's words")
    assert written["number"] == 1 and written["entry"]["name"] == fx.COUNT_TRAIT

    partial = fx.COUNT_SPEC.model_dump()
    del partial["holdout_match_quality_floor"]
    with pytest.raises(ValidationError, match="holdout_match_quality_floor"):
        traits.TraitEntry.model_validate(partial)


@pytest.mark.parametrize("fields, refusal", [
    ({"delivers": ("not_a_vocabulary_phenotype",)}, "off-vocabulary"),
    ({"holdout_match_quality_floor": None}, "holdout_match_quality_floor"),
])
def test_an_entry_a_proposal_refuses_never_reaches_the_record(
    tmp_path: Path, fields: dict, refusal: str,
) -> None:
    with pytest.raises(ValueError, match=refusal):
        fx.propose(tmp_path, fx.with_fields(_count_entry(), **fields))
    assert traits.trait_names(tmp_path) == []


def test_an_operationalization_binds_phenotypes_and_value_keys_by_its_kind(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="names no phenotype"):
        fx.propose(tmp_path, _count_entry(delivered_phenotypes=("stem_count",)))
    with pytest.raises(ValueError, match="value keys its rows carry"):
        fx.propose(tmp_path, fx.with_operationalization(
            fx.COUNT_SPEC, traits.PER_PLANT_COUNT_AGGREGATE, delivered_phenotypes=("stem_count",)))
    with pytest.raises(ValueError, match="at least 1 character"):
        _count_entry(statement="   ")

    admitted = fx.propose(tmp_path, fx.with_operationalization(
        fx.COUNT_SPEC, traits.PER_PLANT_COUNT_AGGREGATE, delivered_phenotypes=("stem_count",),
        delivered_value_keys=("count",)))
    assert admitted.entry.operationalizations[
        traits.PER_PLANT_COUNT_AGGREGATE].delivered_value_keys == ("count",)


def test_a_proposal_is_admitted_once_and_a_stored_record_decodes_without_the_vocabulary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The vocabulary is read once per proposal, by its admission; reading a stored record never
    reads it, so a crops.yml that no longer names a phenotype leaves a confirmed revision
    readable and deliverable."""
    confirmed = fx.propose_and_confirm(tmp_path, _count_entry())
    real = traits._crops_traits
    reads: list[int] = []

    def counted() -> list[dict]:
        reads.append(1)
        return real()

    monkeypatch.setattr(traits, "_crops_traits", counted)
    fx.propose(tmp_path, _count_entry(statement="a second reading"))
    assert len(reads) == 1

    monkeypatch.setattr(traits, "_crops_traits", lambda: [])
    assert traits.read_trait(fx.COUNT_TRAIT, tmp_path).latest_confirmed == confirmed
    assert _deliver_counts(tmp_path, "after_vocabulary_change")["trait_revision"] == 1


def test_the_entry_hash_covers_every_field_and_ignores_sequence_type() -> None:
    """A field the confirmation authorizes and the hash misses would let an entry change under a
    click the breeder gave to other text."""
    entry = _count_entry()
    baseline = traits.entry_sha256(entry)
    dumped = entry.model_dump()

    for field, value in dumped.items():
        if field in ("name", "delivers", "operationalizations"):
            continue
        changed = {True: False, False: True}.get(value) if isinstance(value, bool) else (
            "changed" if isinstance(value, str) else (0.25, 0.75) if isinstance(value, tuple)
            else 0.37)
        moved = traits.TraitEntry.model_construct(**{
            **dumped, "operationalizations": entry.operationalizations, field: changed})
        assert traits.entry_sha256(moved) != baseline, field
    restated = fx.with_operationalization(fx.COUNT_SPEC, traits.PER_IMAGE_COUNT,
                                          measured_subject="leaf")
    assert traits.entry_sha256(restated) != baseline
    as_lists = traits.TraitEntry.model_validate(
        {**entry.model_dump(mode="json"), "delivers": list(entry.delivers)})
    assert traits.entry_sha256(as_lists) == baseline


def test_an_entry_cannot_carry_a_confirmation_and_a_confirmation_needs_a_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="confirmed_by"):
        traits.TraitEntry.model_validate({**fx.COUNT_SPEC.model_dump(), "confirmed_by": "user:x"})

    revision = fx.propose(tmp_path, _count_entry())
    monkeypatch.setenv("TCIP_USER", "rosalind")
    with pytest.raises(ValueError, match="names no one"):
        fx.confirm(tmp_path, revision, user="  ")
    assert not traits.read_trait(fx.COUNT_TRAIT, tmp_path).latest.confirmed
    confirmed = fx.confirm(tmp_path, revision)
    assert confirmed.confirmed_by == "user:grüne" and confirmed.confirmed_at.endswith("+00:00")


def test_a_crossing_proposal_checks_its_positive_state_against_the_registry(tmp_path: Path) -> None:
    crossing = fx.with_operationalization(
        fx.CROSSING_SPEC, traits.STATE_CROSSING_DATES, measured_subject="flower",
        delivered_phenotypes=fx.CROSSING_SPEC.delivers)
    fx.seed_positive_class(tmp_path, "flower", traits.PositiveState(attribute="state",
                                                                     value="shed"))

    with pytest.raises(ValueError, match="'open'"):
        fx.propose(tmp_path, crossing)
    assert traits.trait_names(tmp_path) == []

    fx.seed_positive_class(tmp_path, "flower", fx.CROSSING_SPEC.positive_state)
    assert fx.propose(tmp_path, crossing).number == 1


def test_the_vocabulary_is_read_whole_or_the_read_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(traits, "crops_yml_path", lambda: tmp_path / "missing.yml")

    with pytest.raises(OSError):
        traits.crops_units()


# ── the one confirmation door ────────────────────────────────────────────────


def test_a_confirmation_with_another_hash_refuses_and_the_revisions_own_hash_confirms(
    client: TestClient, tmp_path: Path,
) -> None:
    revision = fx.propose(tmp_path, _count_entry())

    refused = _confirm(client, revision, entry_sha256="0" * 64, user="rosalind")

    assert refused.status_code == 409
    assert refused.json()["detail"]["record"]["revisions"][0]["entry_sha256"] == (
        revision.entry_sha256)
    assert not traits.read_trait(fx.COUNT_TRAIT, tmp_path).latest.confirmed

    confirmed = _confirm(client, revision, user="rosalind")

    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["confirmed_by"] == "user:rosalind"
    assert confirmed.json()["audit_warning"] is None
    assert traits.read_trait(fx.COUNT_TRAIT, tmp_path).latest_confirmed.number == 1


def test_the_traits_route_serves_every_revision_and_the_vocabulary_definitions(
    client: TestClient, tmp_path: Path,
) -> None:
    fx.propose_and_confirm(tmp_path, _count_entry())
    fx.propose(tmp_path, _count_entry(statement="a second reading"))

    body = client.get(TRAITS_ROUTE).json()

    (record,) = body["traits"]
    assert record["trait"] == fx.COUNT_TRAIT
    assert [r["number"] for r in record["revisions"]] == [1, 2]
    assert [r["confirmed"] for r in record["revisions"]] == [True, False]
    assert record["latest_confirmed"] == 1
    assert set(body["definitions"]) == {"stem_count"}
    assert body["unreadable"] == []


def test_a_nameless_confirmation_refuses_and_confirms_nothing(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TCIP_USER", "osuser")
    revision = fx.propose(tmp_path, _count_entry())

    resp = _confirm(client, revision, user=" ")

    assert resp.status_code == 400 and "names no one" in resp.text
    assert not traits.read_trait(fx.COUNT_TRAIT, tmp_path).latest.confirmed


def test_confirming_and_withdrawing_land_in_the_project_log_with_the_actor(
    client: TestClient, tmp_path: Path,
) -> None:
    revision = fx.propose(tmp_path, _count_entry())
    assert _confirm(client, revision, user="rosalind").status_code == 200
    assert _confirm(client, revision, user="rosalind", confirmed=False).status_code == 200

    entries = [e for e in ts.read_log(audit_log_key(tmp_path)).records
               if e["tool"] == "confirm_trait_revision"]

    assert [e["arguments"]["confirmed"] for e in entries] == [True, False]
    assert all(e["user"] == "user:rosalind" for e in entries)
    assert all(e["arguments"]["revision"] == 1 for e in entries)


def test_the_confirmation_writes_its_own_audit_line_after_its_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The library confirmation records itself, whoever calls it, and the record is already
    written when its audit line is appended."""
    import tcip_mcp.audit as audit_module

    revision = fx.propose(tmp_path, _count_entry())
    real_append = audit_module.append
    seen: list[bool] = []

    def append(key, entry):
        if entry["tool"] == "confirm_trait_revision":
            seen.append(traits.read_trait(fx.COUNT_TRAIT, tmp_path).latest.confirmed)
        return real_append(key, entry)

    monkeypatch.setattr(audit_module, "append", append)
    fx.confirm(tmp_path, revision)

    assert seen == [True]


def test_the_door_refuses_an_unknown_trait_and_a_revision_that_does_not_exist(
    client: TestClient, tmp_path: Path,
) -> None:
    revision = fx.propose(tmp_path, _count_entry())

    body = {"trait": fx.COUNT_TRAIT, "revision": 1,
            "entry_sha256": revision.entry_sha256, "confirmed": True, "user": "breeder"}
    unknown = client.post(CONFIRM_ROUTE, json={**body, "trait": "not_a_trait"})
    missing = client.post(CONFIRM_ROUTE, json={**body, "revision": 2})

    assert unknown.status_code == 400 and "not_a_trait" in unknown.json()["detail"]
    assert missing.status_code == 400 and "not 2" in missing.json()["detail"]


def test_a_committed_confirmation_returns_its_audit_failure_as_a_warning(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests._audit_fixtures import refuse_audit_appends

    revision = fx.propose(tmp_path, _count_entry())
    refuse_audit_appends(monkeypatch)
    resp = _confirm(client, revision, user="rosalind")

    assert resp.status_code == 200, resp.text
    assert "do not retry it blind" in resp.json()["audit_warning"]
    assert traits.read_trait(fx.COUNT_TRAIT, tmp_path).latest.confirmed


def test_both_trait_routes_refuse_while_no_project_is_open(tmp_path: Path) -> None:
    revision = fx.propose(tmp_path, _count_entry())
    client = TestClient(app, base_url="http://127.0.0.1")

    assert client.get(TRAITS_ROUTE).status_code == 409
    assert _confirm(client, revision).status_code == 409
    assert not traits.read_trait(fx.COUNT_TRAIT, tmp_path).latest.confirmed


def test_no_mcp_tool_reaches_the_confirmation_writer() -> None:
    """The agent has a proposing tool and no confirming one, checked against the live registry."""
    listing = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "list_tools.py")],
        capture_output=True, text=True, cwd=REPO_ROOT, check=True,
    ).stdout
    registered = {line.strip() for line in listing.splitlines() if line.startswith("  ")}

    assert "propose_trait" in registered
    assert not [name for name in registered if "confirm" in name and "trait" in name]
    assert "confirm_revision" in ROUTES_MODULE.read_text(encoding="utf-8")
    assert not [m.name for m in TOOLS_DIR.glob("*.py")
                if "confirm_revision" in m.read_text(encoding="utf-8")]


# ── the doctor ───────────────────────────────────────────────────────────────


def test_the_doctor_warns_on_an_unconfirmed_latest_revision_and_errors_on_an_unreadable_one(
    tmp_path: Path,
) -> None:
    from tcip_mcp.cli.doctor import check_traits

    fx.propose_and_confirm(tmp_path, _count_entry())
    clean: list = []
    check_traits(tmp_path, clean)
    assert clean == []

    fx.propose(tmp_path, _count_entry(statement="a second reading"))
    key = traits.trait_key(tmp_path, "broken")
    ts.replace(key, {"revisions": [{"number": 1}]}, expect=ts.Version.ABSENT)
    findings: list = []
    check_traits(tmp_path, findings)

    assert ("warn" in {level for level, _ in findings}
            and any("latest revision (2)" in text for _, text in findings))
    assert any(level == "error" and "'broken'" in text for level, text in findings)
