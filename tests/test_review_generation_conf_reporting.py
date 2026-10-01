"""The confidence a prediction bucket was published at, as the Review filter shelf reads it.

Raising the review's own confidence filter above this conf hides low-confidence detections from
the breeder and censors any reference built from the resulting verdicts, so the filter shelf warns
about it live. The warning is only as good as the number behind it: the route reports the bucket's
own recorded value, and reports its absence as absent rather than as a conf of zero. Beside it the
route answers the rule the bucket's assessment admits a prediction under, or the one sentence
naming why none does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_web.app import app

torch = pytest.importorskip("torch")

_SCOPE = {"subject": "bud", "id_map": {"bud": 0}}


@pytest.fixture
def client(opened_project) -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _generation(client: TestClient, bucket: Path) -> dict:
    resp = client.get("/api/review/generation_conf", params={"pred_dir": str(bucket)})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _published(project: Path, name: str) -> Path:
    from tests._chain_fixtures import published

    bucket = project / "ds" / "predictions" / name / "2-11-26"
    published(project, bucket, [], scope=_SCOPE)
    return bucket


def test_generation_conf_reports_the_conf_the_bucket_recorded(
    client: TestClient, tmp_path: Path,
) -> None:
    from tcip_mcp.buckets import read_bucket

    bucket = _published(tmp_path, "baseline")

    body = _generation(client, bucket)

    assert body["generation_conf"] == pytest.approx(
        read_bucket(bucket).execution.conf)
    assert body["admission_conf"] is None
    assert "no assessment answers for" in body["admission_reason"]


def test_a_bucket_of_staged_proposals_reports_an_absent_conf_and_names_why(
    client: TestClient, tmp_path: Path,
) -> None:
    """Proposals run under no execution record, so their conf is absent, never a zero, which
    would read as a bucket published with no confidence floor at all; a directory with no record
    is no bucket and refuses."""
    from PIL import Image

    from tcip_mcp.tools.proposal_tools import stage_proposals

    image = tmp_path / "ds" / "images" / "2-11-26" / "IMG_0001.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(image)
    staged = stage_proposals(tmp_path, str(image), model_name="sam", boxes=[
        {"subject": "bud", "conf": 0.8, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}])
    assert "error" not in staged, staged

    body = _generation(client, Path(staged["path"]).parent)

    assert body["generation_conf"] is None
    assert body["admission_conf"] is None
    assert body["admission_reason"]

    bare = tmp_path / "ds" / "predictions" / "bare" / "2-11-26"
    bare.mkdir(parents=True)
    resp = client.get("/api/review/generation_conf", params={"pred_dir": str(bare)})
    assert resp.status_code == 400
    assert "holds no bucket.json" in resp.json()["detail"]


def test_a_bucket_naming_an_assessment_nobody_ran_answers_no_rule_naming_it(
    client: TestClient, tmp_path: Path,
) -> None:
    bucket = _published(tmp_path, "forged")
    record_path = bucket / "bucket.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record_path.write_text(json.dumps({**record, "assessment_id": "assessment-never-run"}),
                           encoding="utf-8")

    body = _generation(client, bucket)

    assert body["admission_conf"] is None
    assert "assessment-never-run" in body["admission_reason"]


def test_an_undecodable_bucket_record_refuses_naming_its_decode_error(
    client: TestClient, tmp_path: Path,
) -> None:
    """A record that will not decode refuses as its own decode error, never reading as a
    directory of no bucket."""
    bucket = _published(tmp_path, "undecodable")
    (bucket / "bucket.json").write_bytes(b"{not json")

    resp = client.get("/api/review/generation_conf", params={"pred_dir": str(bucket)})

    assert resp.status_code == 400, resp.text
    assert "does not decode" in resp.json()["detail"]


def test_a_bucket_its_passing_assessment_answers_for_reports_the_rule(tmp_path: Path) -> None:
    """The admitting half: a bucket published under the assessment that passed for it answers
    the conf its record runs at."""
    from tcip_mcp.buckets import read_bucket
    from tests._chain_fixtures import run_the_chain
    from tests._web_fixtures import open_new_project

    chain = run_the_chain(tmp_path, experiment_id="exp-generation-rule")
    open_new_project(tmp_path)

    body = _generation(TestClient(app, base_url="http://127.0.0.1"), chain.bucket)

    conf = read_bucket(chain.bucket).execution.conf
    assert body["admission_conf"] == pytest.approx(conf)
    assert body["admission_reason"] == ""
