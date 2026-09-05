"""The proposal staging record: an image's staged run is addressed by its place in the dataset
and carries the content identity of the pixels the engine ran on.

``propose_annotations`` writes the record; ``stage_proposals`` reads it back through the same
address and refuses when the image no longer matches the identity that run recorded. Every test
here drives both tools for real, through a stub engine installed at
``tcip_mcp.pipelines.proposal.resolve_proposer``, never a hand-written envelope.
"""

from __future__ import annotations

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
        "engine": "sam",
        "engine_meta": {},
        "rings": [[(x0, 10.0), (x1, 10.0), (x1, 30.0), (x0, 30.0)]],
    }


def _make_image(path: Path, fill: tuple[int, int, int] = (50, 50, 50)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=fill).save(path)


def test_two_dated_buckets_with_the_same_stem_stage_and_read_back_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two images sharing a stem in different capture-date buckets have their own record: the
    second run's candidates must never answer for the first."""
    from tcip_annotation import json_io
    from tcip_mcp.tools.proposal_tools import stage_proposals, propose_annotations

    first = tmp_path / "images" / "2026-01-01" / "leaf.jpg"
    second = tmp_path / "images" / "2026-02-01" / "leaf.jpg"
    _make_image(first)
    _make_image(second)

    _install_stub(monkeypatch, [_candidate(0, 5.0)])
    proposed_first = propose_annotations(image_path=str(first), engine="sam")
    assert "error" not in proposed_first, proposed_first
    assert proposed_first["staged"] is True

    _install_stub(monkeypatch, [_candidate(0, 300.0)])
    proposed_second = propose_annotations(image_path=str(second), engine="sam")
    assert "error" not in proposed_second, proposed_second
    assert proposed_second["staged"] is True

    accepted = stage_proposals(
        image_path=str(first), assignments=[{"candidate_id": 0, "subject": "bud"}])
    assert "error" not in accepted, accepted

    anns = json_io.read_annotations(
        tmp_path / "predictions" / "sam" / "2026-01-01" / "leaf.json")
    assert len(anns) == 1
    xs = [p[0] for ring in anns[0].geometry.rings for p in ring]
    assert min(xs) == pytest.approx(5.0)


def test_accept_refuses_when_the_images_content_has_changed_since_the_proposal_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rewrite under the same name after propose_annotations ran means the staged candidates no
    longer describe what stage_proposals would be confirming."""
    from tcip_mcp.tools.proposal_tools import stage_proposals, propose_annotations

    img_path = tmp_path / "images" / "changed.jpg"
    _make_image(img_path, fill=(50, 50, 50))

    _install_stub(monkeypatch, [_candidate(0, 5.0)])
    proposed = propose_annotations(image_path=str(img_path), engine="sam")
    assert "error" not in proposed, proposed

    _make_image(img_path, fill=(200, 10, 10))

    accepted = stage_proposals(
        image_path=str(img_path), assignments=[{"candidate_id": 0, "subject": "bud"}])
    assert "error" in accepted
    assert str(img_path) in accepted["error"]


def test_the_envelope_carries_image_identity_and_path_at_the_dataset_rooted_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The record sits at a key naming the dataset root, capture date and stem, and its envelope
    names the pixels it was staged from."""
    import tcip_store as ts
    from tcip_mcp.tools.proposal_tools import PROPOSAL_STAGING_STORE, propose_annotations

    img_path = tmp_path / "images" / "2026-04-01" / "sample.jpg"
    _make_image(img_path)
    _install_stub(monkeypatch, [_candidate(0, 5.0)])

    result = propose_annotations(image_path=str(img_path), engine="sam")
    assert "error" not in result, result
    assert result["staged"] is True

    key = ts.Key(PROPOSAL_STAGING_STORE, str(tmp_path), ("2026-04-01", "sample"))
    envelope = ts.read(key)
    assert envelope["image_path"] == str(img_path)
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

    result = propose_annotations(image_path=str(img_path), engine="sam")
    assert "error" not in result, result
    assert result["staged"] is False
    assert result["candidate_count"] == 1
    assert Path(result["image_path"]).is_file()
    assert not (tmp_path / ".tcip" / "state" / "proposals").exists()


@pytest.mark.parametrize("dated", [False, True])
def test_propose_then_accept_stages_a_prediction_at_the_expected_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dated: bool,
) -> None:
    """The flat and the date-nested layout both propose, stage, and accept the same way."""
    from tcip_annotation import json_io
    from tcip_mcp.tools.proposal_tools import stage_proposals, propose_annotations

    images_dir = tmp_path / "images" / "2026-06-01" if dated else tmp_path / "images"
    img_path = images_dir / "sample.jpg"
    _make_image(img_path)
    _install_stub(monkeypatch, [_candidate(0, 5.0)])

    proposed = propose_annotations(image_path=str(img_path), engine="sam")
    assert "error" not in proposed, proposed
    assert proposed["staged"] is True

    accepted = stage_proposals(
        image_path=str(img_path), assignments=[{"candidate_id": 0, "subject": "bud"}])
    assert "error" not in accepted, accepted

    pred_dir = tmp_path / "predictions" / "sam"
    if dated:
        pred_dir = pred_dir / "2026-06-01"
    anns = json_io.read_annotations(pred_dir / "sample.json")
    assert len(anns) == 1


def test_propose_then_accept_through_a_band_groups_manifest_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A band-grouped capture is addressed by its manifest path on both sides, and resolves to
    the same BandGroupRef for the proposal run and for the accepted image dimensions."""
    import tifffile
    from tcip_annotation import json_io
    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest
    from tcip_mcp.tools.proposal_tools import stage_proposals, propose_annotations

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    bands = {}
    for name in ("Red", "Green", "Blue"):
        band_path = images_dir / f"capture_{name}.tif"
        tifffile.imwrite(str(band_path), np.full((48, 48), 30, dtype=np.uint8))
        bands[name] = band_path
    manifest = write_band_group_manifest(images_dir, "capture", bands)

    _install_stub(monkeypatch, [_candidate(0, 5.0)])
    proposed = propose_annotations(image_path=str(manifest), engine="sam")
    assert "error" not in proposed, proposed
    assert proposed["staged"] is True

    accepted = stage_proposals(
        image_path=str(manifest), assignments=[{"candidate_id": 0, "subject": "bud"}])
    assert "error" not in accepted, accepted

    anns = json_io.read_annotations(tmp_path / "predictions" / "sam" / "capture.json")
    assert len(anns) == 1


def test_propose_on_a_band_groups_member_path_stages_nothing_and_names_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proposing directly on a band-group member's own path (rather than its manifest) is a
    path ``stage_proposals`` could never resolve back to the same source, so it must not be
    staged: staging it anyway would leave a record accept can never confirm."""
    import tifffile
    import tcip_store as ts
    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest
    from tcip_mcp.tools.proposal_tools import _staging_key_for, propose_annotations

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    bands = {}
    for name in ("Red", "Green", "Blue"):
        band_path = images_dir / f"capture_{name}.tif"
        tifffile.imwrite(str(band_path), np.full((48, 48), 30, dtype=np.uint8))
        bands[name] = band_path
    manifest = write_band_group_manifest(images_dir, "capture", bands)

    _install_stub(monkeypatch, [_candidate(0, 5.0)])
    member_path = bands["Red"]
    proposed = propose_annotations(image_path=str(member_path), engine="sam")
    assert "error" not in proposed, proposed
    assert proposed["staged"] is False
    assert manifest.name in proposed["summary"]
    assert Path(proposed["image_path"]).is_file()

    address = _staging_key_for(str(member_path))
    assert ts.read(address.key, default=None) is None


def test_a_second_accept_of_the_same_staged_run_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepting a second subset of one run's proposals is a legitimate second call, since the
    record stays in place after accept."""
    from tcip_mcp.tools.proposal_tools import stage_proposals, propose_annotations

    img_path = tmp_path / "images" / "twice.jpg"
    _make_image(img_path)
    _install_stub(monkeypatch, [_candidate(0, 5.0), _candidate(1, 40.0)])

    proposed = propose_annotations(image_path=str(img_path), engine="sam")
    assert "error" not in proposed, proposed

    first = stage_proposals(
        image_path=str(img_path), assignments=[{"candidate_id": 0, "subject": "bud"}])
    assert "error" not in first, first

    second = stage_proposals(
        image_path=str(img_path), assignments=[{"candidate_id": 1, "subject": "nut"}])
    assert "error" not in second, second


def test_a_second_proposal_run_replaces_the_first_and_accept_reads_the_newest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """last_writer_wins: a re-run overwrites the previous record rather than merging into it."""
    from tcip_annotation import json_io
    from tcip_mcp.tools.proposal_tools import stage_proposals, propose_annotations

    img_path = tmp_path / "images" / "rerun.jpg"
    _make_image(img_path)

    _install_stub(monkeypatch, [_candidate(0, 5.0)])
    first = propose_annotations(image_path=str(img_path), engine="sam")
    assert "error" not in first, first

    _install_stub(monkeypatch, [_candidate(0, 40.0)])
    second = propose_annotations(image_path=str(img_path), engine="sam")
    assert "error" not in second, second

    accepted = stage_proposals(
        image_path=str(img_path), assignments=[{"candidate_id": 0, "subject": "bud"}])
    assert "error" not in accepted, accepted

    anns = json_io.read_annotations(tmp_path / "predictions" / "sam" / "rerun.json")
    xs = [p[0] for ring in anns[0].geometry.rings for p in ring]
    assert min(xs) == pytest.approx(40.0)


def test_a_re_run_finding_nothing_clears_the_previous_runs_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run that proposes zero candidates must not leave a prior run's record readable: a later
    accept would otherwise stage that stale run's candidates as if this run had proposed them."""
    from tcip_mcp.tools.proposal_tools import stage_proposals, propose_annotations

    img_path = tmp_path / "images" / "goes_empty.jpg"
    _make_image(img_path)

    _install_stub(monkeypatch, [_candidate(0, 5.0)])
    first = propose_annotations(image_path=str(img_path), engine="sam")
    assert "error" not in first, first
    assert first["staged"] is True

    _install_stub(monkeypatch, [])
    second = propose_annotations(image_path=str(img_path), engine="sam")
    assert "error" not in second, second
    assert second["staged"] is False

    accepted = stage_proposals(
        image_path=str(img_path), assignments=[{"candidate_id": 0, "subject": "bud"}])
    assert "error" in accepted
    assert "propose_annotations" in accepted["error"]


def test_accept_reports_an_unsampleable_image_as_an_error_dict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source that opens but cannot be sampled (``raster_content_identity``'s own refusal) must
    reach the caller the same way a mismatched or missing record does: a returned ``error``, not
    an uncaught exception out of the tool."""
    from tcip_mcp.pipelines import raster_source
    from tcip_mcp.tools.proposal_tools import stage_proposals, propose_annotations

    img_path = tmp_path / "images" / "unsampleable.jpg"
    _make_image(img_path)

    _install_stub(monkeypatch, [_candidate(0, 5.0)])
    proposed = propose_annotations(image_path=str(img_path), engine="sam")
    assert "error" not in proposed, proposed

    def _raises(*args: object, **kwargs: object) -> None:
        raise ValueError(f"cannot open raster {img_path!r} for a content identity: boom")

    monkeypatch.setattr(raster_source, "raster_content_identity", _raises)

    accepted = stage_proposals(
        image_path=str(img_path), assignments=[{"candidate_id": 0, "subject": "bud"}])
    assert "error" in accepted
    assert str(img_path) in accepted["error"]


def test_stage_proposals_refuses_a_reserved_stem_with_an_error_dict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepting proposals for an image whose stem is a bucket stamp name answers the staging
    writer's refusal as an error dict, never a raise through the audited door."""
    from tcip_mcp.tools.proposal_tools import stage_proposals, propose_annotations

    image = tmp_path / "images" / "2026-01-01" / "operating_point.jpg"
    _make_image(image)
    _install_stub(monkeypatch, [_candidate(0, 5.0)])
    proposed = propose_annotations(image_path=str(image), engine="sam")
    assert "error" not in proposed, proposed

    accepted = stage_proposals(
        image_path=str(image), assignments=[{"candidate_id": 0, "subject": "bud"}])
    assert "error" in accepted
    assert "operating_point" in accepted["error"]
    assert not (tmp_path / "predictions" / "sam" / "2026-01-01" / "operating_point.json").exists()
