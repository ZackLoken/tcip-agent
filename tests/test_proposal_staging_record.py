"""The proposal staging record: an image's staged run is addressed by its place in the dataset
and carries the content identity of the pixels the engine ran on.

``propose_annotations`` writes the record; ``stage_proposals`` reads it back through the same
address and refuses when the image no longer matches the identity that run recorded. Every test
here drives both tools for real, through a stub engine installed at
``tcip_mcp.pipelines.proposal.resolve_proposer``, never a hand-written envelope.
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _install_stub(monkeypatch: pytest.MonkeyPatch, candidates: list[dict]) -> None:
    """A proposal engine that hands ``candidates`` back verbatim, installed through the
    platform's own engine-resolution seam."""
    from tcip_mcp.pipelines import proposal

    class StubProposer:
        def propose(self, image_path: str, **params: object) -> list[dict]:
            return candidates

    monkeypatch.setattr(proposal, "resolve_proposer", lambda engine: StubProposer())


def _candidate(candidate_id: int, x0: float) -> dict:
    """One candidate box at ``x0``, distinguishable from another by position alone so a staged
    annotation's geometry says which run it came from."""
    x1 = x0 + 20.0
    return {
        "candidate_id": candidate_id,
        "bbox": [x0, 10.0, x1, 30.0],
        "area": 400,
        "score": 0.9,
        "engine": "stub",
        "engine_meta": {},
        "rings": [[(x0, 10.0), (x1, 10.0), (x1, 30.0), (x0, 30.0)]],
    }


def _make_image(path: Path, fill: tuple[int, int, int] = (50, 50, 50)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=fill).save(path)


def _staged(root: Path, bucket: str, stem: str):
    """The key of ``stem``'s document in the staged bucket ``bucket`` under ``root``."""
    from tcip_mcp.dataset_layout import prediction_key

    return prediction_key(root, bucket, stem)


def _staged_annotations(root: Path, bucket: str, stem: str) -> list:
    from tcip_annotation import json_io

    return json_io.read_label_document(_staged(root, bucket, stem)).annotations


def test_two_dated_buckets_with_the_same_stem_stage_and_read_back_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two images sharing a stem in different capture-date buckets have their own record: the
    second run's candidates must never answer for the first."""
    from tcip_mcp.tools.proposal_tools import stage_proposals, propose_annotations

    first = tmp_path / "images" / "2026-01-01" / "leaf.jpg"
    second = tmp_path / "images" / "2026-02-01" / "leaf.jpg"
    _make_image(first)
    _make_image(second)

    _install_stub(monkeypatch, [_candidate(0, 5.0)])
    proposed_first = propose_annotations(tmp_path, image_path=str(first), engine="stub")
    assert "error" not in proposed_first, proposed_first
    assert proposed_first["staged"] is True

    _install_stub(monkeypatch, [_candidate(0, 300.0)])
    proposed_second = propose_annotations(tmp_path, image_path=str(second), engine="stub")
    assert "error" not in proposed_second, proposed_second
    assert proposed_second["staged"] is True

    accepted = stage_proposals(
        tmp_path, image_path=str(first), assignments=[{"candidate_id": 0, "subject": "bud"}])
    assert "error" not in accepted, accepted

    assert accepted["bucket"] == "stub/2026-01-01/leaf"
    anns = _staged_annotations(tmp_path, accepted["bucket"], "leaf")
    assert len(anns) == 1
    xs = [p[0] for ring in anns[0].geometry.rings for p in ring]
    assert min(xs) == pytest.approx(5.0)


def test_an_assignment_naming_no_subject_is_refused_never_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An assigned candidate is an annotation, refused by construction when it names no subject,
    the refusal naming the assignment; it is never dropped while the rest are staged."""
    from tcip_mcp.tools.proposal_tools import propose_annotations, stage_proposals

    img_path = tmp_path / "images" / "2026-01-01" / "bur.jpg"
    _make_image(img_path)
    _install_stub(monkeypatch, [_candidate(0, 5.0), _candidate(1, 30.0)])
    assert "error" not in propose_annotations(tmp_path, image_path=str(img_path), engine="stub")

    staged = stage_proposals(tmp_path, image_path=str(img_path), assignments=[
        {"candidate_id": 0, "subject": "bur"}, {"candidate_id": 1, "subject": ""}])
    import tcip_store

    from tcip_mcp.dataset_layout import bucket_key

    assert staged["error"].startswith("assignment 1: ")
    assert not tcip_store.exists(bucket_key(tmp_path, "stub/2026-01-01/bur"))


def test_accept_refuses_when_the_images_content_has_changed_since_the_proposal_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rewrite under the same name after propose_annotations ran means the staged candidates no
    longer describe what stage_proposals would be confirming."""
    from tcip_mcp.tools.proposal_tools import stage_proposals, propose_annotations

    img_path = tmp_path / "images" / UNDATED_BUCKET / "changed.jpg"
    _make_image(img_path, fill=(50, 50, 50))

    _install_stub(monkeypatch, [_candidate(0, 5.0)])
    proposed = propose_annotations(tmp_path, image_path=str(img_path), engine="stub")
    assert "error" not in proposed, proposed

    _make_image(img_path, fill=(200, 10, 10))

    accepted = stage_proposals(
        tmp_path, image_path=str(img_path), assignments=[{"candidate_id": 0, "subject": "bud"}])
    assert "error" in accepted
    assert str(img_path) in accepted["error"]


def test_the_envelope_carries_image_identity_at_the_dataset_rooted_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The record sits at a key naming the dataset root, capture date and stem, which name the
    image, and its envelope carries the identity of the pixels it was staged from."""
    import tcip_store as ts
    from tcip_mcp.tools.proposal_tools import PROPOSAL_STAGING_STORE, propose_annotations

    img_path = tmp_path / "images" / "2026-04-01" / "sample.jpg"
    _make_image(img_path)
    _install_stub(monkeypatch, [_candidate(0, 5.0)])

    result = propose_annotations(tmp_path, image_path=str(img_path), engine="stub")
    assert "error" not in result, result
    assert result["staged"] is True

    key = ts.Key(PROPOSAL_STAGING_STORE, str(tmp_path), ("2026-04-01", "sample"))
    envelope = ts.read(key)
    assert "image_path" not in envelope
    assert set(envelope["image_identity"]) >= {
        "width", "height", "num_channels", "pixel_checksum",
    }


def test_propose_outside_a_dataset_tree_runs_the_engine_and_stages_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A path with no dataset address to stage under still gets a real proposal run and a real
    render; it just can never be accepted, which was already true before it could be staged."""
    from tcip_mcp.tools.proposal_tools import propose_annotations

    img_path = tmp_path / "loose.jpg"
    _make_image(img_path)
    _install_stub(monkeypatch, [_candidate(0, 5.0)])

    result = propose_annotations(tmp_path, image_path=str(img_path), engine="stub")
    assert "error" not in result, result
    assert result["staged"] is False
    assert result["candidate_count"] == 1
    assert Path(result["image_path"]).is_file()
    assert not (tmp_path / ".tcip" / "state" / "proposals").exists()


@pytest.mark.parametrize("capture", [UNDATED_BUCKET, "2026-06-01"])
def test_propose_then_accept_stages_a_prediction_at_the_expected_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capture: str,
) -> None:
    """An undated and a dated capture both propose, stage, and accept the same way."""
    from tcip_mcp.tools.proposal_tools import stage_proposals, propose_annotations

    img_path = tmp_path / "images" / capture / "sample.jpg"
    _make_image(img_path)
    _install_stub(monkeypatch, [_candidate(0, 5.0)])

    proposed = propose_annotations(tmp_path, image_path=str(img_path), engine="stub")
    assert "error" not in proposed, proposed
    assert proposed["staged"] is True

    accepted = stage_proposals(
        tmp_path, image_path=str(img_path), assignments=[{"candidate_id": 0, "subject": "bud"}])
    assert "error" not in accepted, accepted

    assert accepted["bucket"] == f"stub/{capture}/sample"
    assert len(_staged_annotations(tmp_path, accepted["bucket"], "sample")) == 1


def test_propose_then_accept_through_a_band_groups_manifest_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A band-grouped capture is addressed by its manifest path on both sides, and resolves to
    the same BandGroupRef for the proposal run and for the accepted image dimensions."""
    import tifffile
    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest
    from tcip_mcp.tools.proposal_tools import stage_proposals, propose_annotations

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    bands = {}
    for name in ("Red", "Green", "Blue"):
        band_path = images_dir / f"capture_{name}.tif"
        tifffile.imwrite(str(band_path), np.full((48, 48), 30, dtype=np.uint8))
        bands[name] = band_path
    manifest = write_band_group_manifest(images_dir, "capture", bands)

    _install_stub(monkeypatch, [_candidate(0, 5.0)])
    proposed = propose_annotations(tmp_path, image_path=str(manifest), engine="stub")
    assert "error" not in proposed, proposed
    assert proposed["staged"] is True

    accepted = stage_proposals(
        tmp_path, image_path=str(manifest), assignments=[{"candidate_id": 0, "subject": "bud"}])
    assert "error" not in accepted, accepted

    assert accepted["bucket"] == f"stub/{UNDATED_BUCKET}/capture"
    assert len(_staged_annotations(tmp_path, accepted["bucket"], "capture")) == 1


def test_propose_on_a_band_groups_member_path_refuses_naming_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A band of a grouped capture is no logical image of its own: proposing on its path refuses
    naming the manifest, stages nothing, and the manifest's own path admits the proposal."""
    import tifffile
    import tcip_store as ts
    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest
    from tcip_mcp.tools.proposal_tools import _staging_key_for, propose_annotations

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    bands = {}
    for name in ("Red", "Green", "Blue"):
        band_path = images_dir / f"capture_{name}.tif"
        tifffile.imwrite(str(band_path), np.full((48, 48), 30, dtype=np.uint8))
        bands[name] = band_path
    manifest = write_band_group_manifest(images_dir, "capture", bands)

    _install_stub(monkeypatch, [_candidate(0, 5.0)])
    member_path = bands["Red"]
    proposed = propose_annotations(tmp_path, image_path=str(member_path), engine="stub")
    assert manifest.name in proposed["error"]
    assert ts.read(_staging_key_for(str(member_path)), default=None) is None

    admitted = propose_annotations(tmp_path, image_path=str(manifest), engine="stub")
    assert admitted["staged"] is True, admitted


def test_a_second_accept_of_the_same_staged_run_refuses_and_keeps_the_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first accept's document is written once, so a second accept for the same image refuses
    naming it and the first document is unchanged."""
    import tcip_store

    from tcip_mcp.tools.proposal_tools import stage_proposals, propose_annotations

    img_path = tmp_path / "images" / UNDATED_BUCKET / "twice.jpg"
    _make_image(img_path)
    _install_stub(monkeypatch, [_candidate(0, 5.0), _candidate(1, 40.0)])

    proposed = propose_annotations(tmp_path, image_path=str(img_path), engine="stub")
    assert "error" not in proposed, proposed

    first = stage_proposals(
        tmp_path, image_path=str(img_path), assignments=[{"candidate_id": 0, "subject": "bud"}])
    assert "error" not in first, first
    document = _staged(tmp_path, first["bucket"], "twice")
    written = tcip_store.read_versioned(document).version

    second = stage_proposals(
        tmp_path, image_path=str(img_path), assignments=[{"candidate_id": 1, "subject": "nut"}])
    assert "already" in second["error"]
    assert tcip_store.read_versioned(document).version == written


def test_a_second_proposal_run_replaces_the_first_and_accept_reads_the_newest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """last_writer_wins: a re-run overwrites the previous record rather than merging into it."""
    from tcip_mcp.tools.proposal_tools import stage_proposals, propose_annotations

    img_path = tmp_path / "images" / UNDATED_BUCKET / "rerun.jpg"
    _make_image(img_path)

    _install_stub(monkeypatch, [_candidate(0, 5.0)])
    first = propose_annotations(tmp_path, image_path=str(img_path), engine="stub")
    assert "error" not in first, first

    _install_stub(monkeypatch, [_candidate(0, 40.0)])
    second = propose_annotations(tmp_path, image_path=str(img_path), engine="stub")
    assert "error" not in second, second

    accepted = stage_proposals(
        tmp_path, image_path=str(img_path), assignments=[{"candidate_id": 0, "subject": "bud"}])
    assert "error" not in accepted, accepted

    anns = _staged_annotations(tmp_path, accepted["bucket"], "rerun")
    xs = [p[0] for ring in anns[0].geometry.rings for p in ring]
    assert min(xs) == pytest.approx(40.0)


def test_a_re_run_finding_nothing_clears_the_previous_runs_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run that proposes zero candidates must not leave a prior run's record readable: a later
    accept would otherwise stage that stale run's candidates as if this run had proposed them."""
    from tcip_mcp.tools.proposal_tools import stage_proposals, propose_annotations

    img_path = tmp_path / "images" / UNDATED_BUCKET / "goes_empty.jpg"
    _make_image(img_path)

    _install_stub(monkeypatch, [_candidate(0, 5.0)])
    first = propose_annotations(tmp_path, image_path=str(img_path), engine="stub")
    assert "error" not in first, first
    assert first["staged"] is True

    _install_stub(monkeypatch, [])
    second = propose_annotations(tmp_path, image_path=str(img_path), engine="stub")
    assert "error" not in second, second
    assert second["staged"] is False

    accepted = stage_proposals(
        tmp_path, image_path=str(img_path), assignments=[{"candidate_id": 0, "subject": "bud"}])
    assert "error" in accepted
    assert "propose_annotations" in accepted["error"]


def test_a_first_run_finding_nothing_leaves_no_audit_line_and_a_clearing_run_leaves_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a removal is an act: an image with no staged record that the engine finds nothing on
    records nothing, and a run that clears a prior run's record records that clearing."""
    from tcip_mcp.audit import acts_of
    from tcip_mcp.tools.proposal_tools import propose_annotations

    img_path = tmp_path / "images" / UNDATED_BUCKET / "fresh.jpg"
    _make_image(img_path)

    _install_stub(monkeypatch, [])
    result = propose_annotations(tmp_path, image_path=str(img_path), engine="stub")
    assert "error" not in result, result
    assert acts_of(tmp_path, ("propose_annotations",))[0] == []

    _install_stub(monkeypatch, [_candidate(0, 5.0)])
    propose_annotations(tmp_path, image_path=str(img_path), engine="stub")
    _install_stub(monkeypatch, [])
    propose_annotations(tmp_path, image_path=str(img_path), engine="stub")
    acts, _cursor = acts_of(tmp_path, ("propose_annotations",))
    assert [act["arguments"]["staged"] for act in acts][-1] == 0
    assert len(acts) == 2, acts


def test_accept_reports_an_unsampleable_image_as_an_error_dict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source that opens but cannot be sampled (``raster_content_identity``'s own refusal) must
    reach the caller the same way a mismatched or missing record does: a returned ``error``, not
    an uncaught exception out of the tool."""
    from tcip_mcp.pipelines import raster_source
    from tcip_mcp.tools.proposal_tools import stage_proposals, propose_annotations

    img_path = tmp_path / "images" / UNDATED_BUCKET / "unsampleable.jpg"
    _make_image(img_path)

    _install_stub(monkeypatch, [_candidate(0, 5.0)])
    proposed = propose_annotations(tmp_path, image_path=str(img_path), engine="stub")
    assert "error" not in proposed, proposed

    def _raises(*args: object, **kwargs: object) -> None:
        raise ValueError(f"cannot open raster {img_path!r} for a content identity: boom")

    monkeypatch.setattr(raster_source, "raster_content_identity", _raises)

    accepted = stage_proposals(
        tmp_path, image_path=str(img_path), assignments=[{"candidate_id": 0, "subject": "bud"}])
    assert "error" in accepted
    assert str(img_path) in accepted["error"]


def test_a_staging_is_one_publication_with_one_audit_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Staging publishes through the one publication, which owns every document's write-once and
    takes encoded documents, never a writer of the caller's; the publication's own line is the
    staging's only one."""
    import inspect

    import tcip_store
    from tcip_mcp.audit import audit_log_key
    from tcip_mcp.buckets import Document, publish
    from tcip_mcp.tools.proposal_tools import stage_proposals, propose_annotations

    assert Document._fields == ("source", "data", "dropped")
    assert "Callable" not in str(inspect.signature(publish))
    image = tmp_path / "images" / "2026-01-01" / "leaf.jpg"
    _make_image(image)
    _install_stub(monkeypatch, [_candidate(0, 5.0)])
    assert "error" not in propose_annotations(tmp_path, image_path=str(image), engine="stub")

    accepted = stage_proposals(
        tmp_path, image_path=str(image), assignments=[{"candidate_id": 0, "subject": "bud"}])

    assert "error" not in accepted, accepted
    lines = [e["tool"] for e in tcip_store.read_log(audit_log_key(tmp_path)).records
             if e["tool"] in ("stage_proposals", "prediction_bucket_published")]
    assert lines == ["prediction_bucket_published"]
