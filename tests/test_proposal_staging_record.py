"""Tests of the proposal staging record.

Each staged run is addressed by its image's place in the dataset and the run's own id, and carries
the content identity of the pixels the engine ran on.

``propose_annotations`` writes the record and answers its ``staging_id``; ``stage_proposals``
reads back the run that id names and refuses when the image does not match the identity that run
recorded. Both tools run over a stub engine installed at
``tcip_mcp.pipelines.proposal.resolve_proposer``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
from tests._producer_fixtures import (
    install_stub_proposer,
    kept,
    write_image,
)

if TYPE_CHECKING:
    from tcip_mcp.pipelines.raster_source import RasterSource
    from tcip_mcp.tools.vision_tools import DisplayRead
    from tcip_store import Key


def _candidate(candidate_id: int, x0: float) -> dict:
    """Return one 20 px candidate box at ``x0``, with its ring, scored 0.9."""
    x1 = x0 + 20.0
    return {
        "candidate_id": candidate_id,
        "bbox": [x0, 10.0, x1, 30.0],
        "area": 400,
        "score": 0.9,
        "engine_meta": {},
        "rings": [[(x0, 10.0), (x1, 10.0), (x1, 30.0), (x0, 30.0)]],
    }


def _make_image(path: Path, fill: tuple[int, int, int] = (50, 50, 50)) -> None:
    write_image(path, (64, 64), fill)


def _staged(root: Path, bucket: str, stem: str) -> Key:
    """Return the key of ``stem``'s document in the staged bucket ``bucket`` under ``root``."""
    from tcip_mcp.dataset_layout import prediction_key

    return prediction_key(root, bucket, stem)


def _staged_capture(root: Path, bucket: str) -> str | None:
    """Return the capture the record of the staged bucket ``bucket`` under ``root`` states."""
    from tcip_mcp.buckets import Proposed, read_bucket

    record = read_bucket(root, bucket)
    assert record.producer == Proposed(proposed_by="stub"), record
    return record.date


def _staged_annotations(root: Path, bucket: str, stem: str) -> list:
    from tcip_annotation import json_io

    return json_io.read_label_document(_staged(root, bucket, stem)).annotations


def test_two_dated_buckets_with_the_same_stem_stage_and_read_back_independently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two dated buckets with the same stem stage and read back independently.

    Two images sharing a stem in different capture-date buckets have their own record: accepting
    the first stages the first run's candidate, not the second's.
    """
    from tcip_mcp.tools.proposal_tools import (
        propose_annotations,
        stage_proposals,
    )

    first = image_dir(tmp_path, "2026-01-01") / "leaf.jpg"
    second = image_dir(tmp_path, "2026-02-01") / "leaf.jpg"
    _make_image(first)
    _make_image(second)

    install_stub_proposer(monkeypatch, [_candidate(0, 5.0)])
    proposed_first = propose_annotations(
        tmp_path, image_path=str(first), engine="stub", actor=None
    )
    assert "error" not in proposed_first, proposed_first
    assert proposed_first["staged"] is True

    install_stub_proposer(monkeypatch, [_candidate(0, 300.0)])
    proposed_second = propose_annotations(
        tmp_path, image_path=str(second), engine="stub", actor=None
    )
    assert "error" not in proposed_second, proposed_second
    assert proposed_second["staged"] is True

    accepted = stage_proposals(
        tmp_path,
        image_path=str(first),
        actor=None,
        assignments=kept(proposed_first, (0, "bud")),
    )
    assert "error" not in accepted, accepted

    assert _staged_capture(tmp_path, accepted["bucket"]) == "2026-01-01"
    anns = _staged_annotations(tmp_path, accepted["bucket"], "leaf")
    assert len(anns) == 1
    xs = [p[0] for ring in anns[0].geometry.rings for p in ring]
    assert min(xs) == pytest.approx(5.0)


def test_assignments_naming_no_run_are_refused_by_their_type() -> None:
    from pydantic import ValidationError

    from tcip_mcp.tools.proposal_tools import StagedAssignments

    with pytest.raises(ValidationError, match="staging_id"):
        StagedAssignments.model_validate(
            {"candidates": [{"candidate_id": 0, "subject": "bud"}]}
        )


def test_an_assignment_naming_no_subject_is_refused_never_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An assignment naming no subject refuses by its index, and nothing is staged.

    An assigned candidate is an annotation, refused by construction when it names no subject.
    """
    from tcip_mcp.tools.proposal_tools import (
        propose_annotations,
        stage_proposals,
    )

    img_path = image_dir(tmp_path, "2026-01-01") / "bur.jpg"
    _make_image(img_path)
    install_stub_proposer(
        monkeypatch, [_candidate(0, 5.0), _candidate(1, 30.0)]
    )
    proposed = propose_annotations(
        tmp_path, image_path=str(img_path), engine="stub", actor=None
    )
    assert "error" not in proposed, proposed

    staged = stage_proposals(
        tmp_path,
        image_path=str(img_path),
        actor=None,
        assignments=kept(proposed, (0, "bur"), (1, "")),
    )
    from tcip_mcp.buckets import buckets_under

    assert staged["error"].startswith("assignment 1: ")
    assert buckets_under(tmp_path) == []


def test_accept_refuses_when_the_images_content_has_changed_since_the_proposal_ran(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept refuses when the image's content changed since the proposal ran.

    After a rewrite under the same name, the accept refuses naming the image, and no pixel of the
    changed image is read for display.
    """
    from tcip_mcp.tools import vision_tools
    from tcip_mcp.tools.proposal_tools import (
        propose_annotations,
        stage_proposals,
    )

    img_path = image_dir(tmp_path, UNDATED_BUCKET) / "changed.jpg"
    _make_image(img_path, fill=(50, 50, 50))

    install_stub_proposer(monkeypatch, [_candidate(0, 5.0)])
    proposed = propose_annotations(
        tmp_path, image_path=str(img_path), engine="stub", actor=None
    )
    assert "error" not in proposed, proposed

    _make_image(img_path, fill=(200, 10, 10))
    displayed: list = []
    real = vision_tools._read_for_display

    def counted(
        *args: RasterSource,
        **kwargs: int | tuple[float, float, float, float] | None,
    ) -> DisplayRead:
        displayed.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(vision_tools, "_read_for_display", counted)
    accepted = stage_proposals(
        tmp_path,
        image_path=str(img_path),
        actor=None,
        assignments=kept(proposed, (0, "bud")),
    )
    assert "error" in accepted
    assert str(img_path) in accepted["error"]
    assert displayed == []


def test_the_envelope_carries_image_identity_at_the_dataset_rooted_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The staging envelope carries image identity at the dataset-rooted location.

    The record sits at a key naming the dataset root, capture date and stem, which name the image,
    and the id the run answers, and its envelope carries the identity of the pixels it was staged
    from.
    """
    import tcip_store as ts
    from tcip_mcp.tools.proposal_tools import (
        PROPOSAL_STAGING_STORE,
        propose_annotations,
    )

    img_path = image_dir(tmp_path, "2026-04-01") / "sample.jpg"
    _make_image(img_path)
    install_stub_proposer(monkeypatch, [_candidate(0, 5.0)])

    result = propose_annotations(
        tmp_path, image_path=str(img_path), engine="stub", actor=None
    )
    assert "error" not in result, result
    assert result["staged"] is True

    key = ts.Key(
        PROPOSAL_STAGING_STORE,
        str(tmp_path),
        ("2026-04-01", "sample", result["staging_id"]),
    )
    envelope = ts.read(key)
    assert isinstance(envelope, dict) and "image_path" not in envelope
    identity = envelope["image_identity"]
    assert isinstance(identity, dict)
    assert set(identity) >= {
        "width",
        "height",
        "num_channels",
        "pixel_checksum",
    }


def test_propose_outside_a_dataset_tree_runs_the_engine_and_stages_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A proposal outside a dataset tree runs the engine and stages nothing.

    A path with no dataset address to stage under still gets a real proposal run and a real
    render, and no staging record is written.
    """
    from tcip_mcp.tools.proposal_tools import propose_annotations

    img_path = tmp_path / "loose.jpg"
    _make_image(img_path)
    install_stub_proposer(monkeypatch, [_candidate(0, 5.0)])

    result = propose_annotations(
        tmp_path, image_path=str(img_path), engine="stub", actor=None
    )
    assert "error" not in result, result
    assert (result["staged"], result["staging_id"]) == (False, None)
    assert result["candidate_count"] == 1
    assert Path(result["image_path"]).is_file()
    assert not (tmp_path / ".tcip" / "state" / "proposals").exists()


@pytest.mark.parametrize("capture", [UNDATED_BUCKET, "2026-06-01"])
def test_propose_then_accept_stages_a_prediction_at_the_expected_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capture: str,
) -> None:
    """An undated and a dated capture both propose, stage, and accept the same way."""
    from tcip_mcp.tools.proposal_tools import (
        propose_annotations,
        stage_proposals,
    )

    img_path = image_dir(tmp_path, capture) / "sample.jpg"
    _make_image(img_path)
    install_stub_proposer(monkeypatch, [_candidate(0, 5.0)])

    proposed = propose_annotations(
        tmp_path, image_path=str(img_path), engine="stub", actor=None
    )
    assert "error" not in proposed, proposed
    assert proposed["staged"] is True

    accepted = stage_proposals(
        tmp_path,
        image_path=str(img_path),
        actor=None,
        assignments=kept(proposed, (0, "bud")),
    )
    assert "error" not in accepted, accepted

    assert _staged_capture(tmp_path, accepted["bucket"]) == capture
    assert (
        len(_staged_annotations(tmp_path, accepted["bucket"], "sample")) == 1
    )


def test_propose_then_accept_through_a_band_groups_manifest_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propose then accept work through a band group's manifest path.

    A band-grouped capture is addressed by its manifest path on both sides: the proposal stages,
    and the accept writes one annotation into the capture's bucket.
    """
    import tifffile

    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest
    from tcip_mcp.tools.proposal_tools import (
        propose_annotations,
        stage_proposals,
    )

    images_dir = image_dir(tmp_path, UNDATED_BUCKET)
    images_dir.mkdir(parents=True)
    bands = {}
    for name in ("Red", "Green", "Blue"):
        band_path = images_dir / f"capture_{name}.tif"
        tifffile.imwrite(str(band_path), np.full((48, 48), 30, dtype=np.uint8))
        bands[name] = band_path
    manifest = write_band_group_manifest(
        images_dir,
        "capture",
        bands,
        central_wavelength_nm=None,
        identity_tolerance={},
    )

    install_stub_proposer(monkeypatch, [_candidate(0, 5.0)])
    proposed = propose_annotations(
        tmp_path, image_path=str(manifest), engine="stub", actor=None
    )
    assert "error" not in proposed, proposed
    assert proposed["staged"] is True

    accepted = stage_proposals(
        tmp_path,
        image_path=str(manifest),
        actor=None,
        assignments=kept(proposed, (0, "bud")),
    )
    assert "error" not in accepted, accepted

    assert _staged_capture(tmp_path, accepted["bucket"]) == UNDATED_BUCKET
    assert (
        len(_staged_annotations(tmp_path, accepted["bucket"], "capture")) == 1
    )


def test_propose_on_a_band_groups_member_path_refuses_naming_the_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proposing on a band group's member path refuses, naming the manifest.

    A band of a grouped capture is no logical image of its own: proposing on its path stages
    nothing, and the manifest's own path admits the proposal.
    """
    import tifffile

    import tcip_store as ts
    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest
    from tcip_mcp.tools.proposal_tools import (
        PROPOSAL_STAGING_STORE,
        propose_annotations,
    )

    images_dir = image_dir(tmp_path, UNDATED_BUCKET)
    images_dir.mkdir(parents=True)
    bands = {}
    for name in ("Red", "Green", "Blue"):
        band_path = images_dir / f"capture_{name}.tif"
        tifffile.imwrite(str(band_path), np.full((48, 48), 30, dtype=np.uint8))
        bands[name] = band_path
    manifest = write_band_group_manifest(
        images_dir,
        "capture",
        bands,
        central_wavelength_nm=None,
        identity_tolerance={},
    )

    install_stub_proposer(monkeypatch, [_candidate(0, 5.0)])
    member_path = bands["Red"]
    proposed = propose_annotations(
        tmp_path, image_path=str(member_path), engine="stub", actor=None
    )
    assert manifest.name in proposed["error"]
    assert ts.keys(PROPOSAL_STAGING_STORE, str(tmp_path)) == []

    admitted = propose_annotations(
        tmp_path, image_path=str(manifest), engine="stub", actor=None
    )
    assert admitted["staged"] is True, admitted


def test_a_second_accept_of_the_same_staged_run_refuses_and_keeps_the_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second accept of the same staged run refuses, naming the run's bucket, and keeps the first.

    The run's bucket is named for the run, so it is published once; the first document's version
    is the one the first accept wrote.
    """
    import tcip_store
    from tcip_mcp.tools.proposal_tools import (
        propose_annotations,
        stage_proposals,
    )

    img_path = image_dir(tmp_path, UNDATED_BUCKET) / "twice.jpg"
    _make_image(img_path)
    install_stub_proposer(
        monkeypatch, [_candidate(0, 5.0), _candidate(1, 40.0)]
    )

    proposed = propose_annotations(
        tmp_path, image_path=str(img_path), engine="stub", actor=None
    )
    assert "error" not in proposed, proposed

    first = stage_proposals(
        tmp_path,
        image_path=str(img_path),
        actor=None,
        assignments=kept(proposed, (0, "bud")),
    )
    assert "error" not in first, first
    document = _staged(tmp_path, first["bucket"], "twice")
    written = tcip_store.read_versioned(document).version

    second = stage_proposals(
        tmp_path,
        image_path=str(img_path),
        actor=None,
        assignments=kept(proposed, (1, "nut")),
    )
    assert "already" in second["error"] and first["bucket"] in second["error"]
    assert tcip_store.read_versioned(document).version == written


def test_two_runs_on_one_image_stay_their_own_and_each_accept_stages_its_own(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two proposal runs on one image each keep their record, and an accept stages its own run.

    The earlier run's candidates are still the ones its ``staging_id`` stages after a later run,
    and each run publishes under a bucket named for its own id.
    """
    from tcip_mcp.tools.proposal_tools import (
        propose_annotations,
        stage_proposals,
    )

    img_path = image_dir(tmp_path, UNDATED_BUCKET) / "rerun.jpg"
    _make_image(img_path)

    install_stub_proposer(monkeypatch, [_candidate(0, 5.0)])
    first = propose_annotations(
        tmp_path, image_path=str(img_path), engine="stub", actor=None
    )
    install_stub_proposer(monkeypatch, [_candidate(0, 40.0)])
    second = propose_annotations(
        tmp_path, image_path=str(img_path), engine="stub", actor=None
    )
    assert first["staging_id"] != second["staging_id"], (first, second)

    def staged_x(proposed: dict) -> float:
        accepted = stage_proposals(
            tmp_path,
            image_path=str(img_path),
            actor=None,
            assignments=kept(proposed, (0, "bud")),
        )
        assert "error" not in accepted, accepted
        assert accepted["bucket"].endswith(f"/{proposed['staging_id']}")
        (ann,) = _staged_annotations(tmp_path, accepted["bucket"], "rerun")
        return min(p[0] for ring in ann.geometry.rings for p in ring)

    assert staged_x(first) == pytest.approx(5.0)
    assert staged_x(second) == pytest.approx(40.0)


def test_a_run_finding_nothing_stages_nothing_and_leaves_an_earlier_run_staged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run that proposes nothing answers no ``staging_id`` and leaves an earlier run as it was.

    The empty run records no act, and the earlier run's id still stages its candidate.
    """
    from tcip_mcp.audit import acts_of
    from tcip_mcp.tools.proposal_tools import (
        propose_annotations,
        stage_proposals,
    )

    img_path = image_dir(tmp_path, UNDATED_BUCKET) / "goes_empty.jpg"
    _make_image(img_path)

    install_stub_proposer(monkeypatch, [_candidate(0, 5.0)])
    first = propose_annotations(
        tmp_path, image_path=str(img_path), engine="stub", actor=None
    )
    assert first["staged"] is True, first

    install_stub_proposer(monkeypatch, [])
    second = propose_annotations(
        tmp_path, image_path=str(img_path), engine="stub", actor=None
    )
    assert (second["staged"], second["staging_id"]) == (False, None), second
    acts, _cursor = acts_of(tmp_path, ("propose_annotations",))
    assert [act["arguments"]["staging_id"] for act in acts] == [
        first["staging_id"]
    ]

    accepted = stage_proposals(
        tmp_path,
        image_path=str(img_path),
        actor=None,
        assignments=kept(first, (0, "bud")),
    )
    assert "error" not in accepted, accepted


def test_accept_reports_an_unsampleable_image_as_an_error_dict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept reports an unsampleable image as a returned ``error``.

    With ``raster_content_identity`` raising for the image, the accept returns an error naming it.
    """
    from tcip_mcp.pipelines import raster_source
    from tcip_mcp.tools.proposal_tools import (
        propose_annotations,
        stage_proposals,
    )

    img_path = image_dir(tmp_path, UNDATED_BUCKET) / "unsampleable.jpg"
    _make_image(img_path)

    install_stub_proposer(monkeypatch, [_candidate(0, 5.0)])
    proposed = propose_annotations(
        tmp_path, image_path=str(img_path), engine="stub", actor=None
    )
    assert "error" not in proposed, proposed

    def _raises(*args: object, **kwargs: object) -> None:
        raise ValueError(
            f"cannot open raster {img_path!r} for a content identity: boom"
        )

    monkeypatch.setattr(raster_source, "raster_content_identity", _raises)

    accepted = stage_proposals(
        tmp_path,
        image_path=str(img_path),
        actor=None,
        assignments=kept(proposed, (0, "bud")),
    )
    assert "error" in accepted
    assert str(img_path) in accepted["error"]


def test_a_region_proposal_and_its_accept_each_open_the_image_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A region proposal and its accept each open the image once.

    The proposal, with its region crop, render and recorded identity, opens the image once, and
    the accept, verifying the identity and rendering, opens it once.
    """
    from tcip_mcp.pipelines import raster_source
    from tcip_mcp.tools.proposal_tools import (
        propose_annotations,
        stage_proposals,
    )

    img_path = image_dir(tmp_path, UNDATED_BUCKET) / "region.jpg"
    _make_image(img_path)
    install_stub_proposer(monkeypatch, [_candidate(0, 5.0)])
    opened: list = []
    real = raster_source.SourceHeader.open

    def counted(
        *args: raster_source.SourceHeader | int, **kwargs: int
    ) -> RasterSource:
        opened.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(raster_source.SourceHeader, "open", counted)
    proposed = propose_annotations(
        tmp_path,
        image_path=str(img_path),
        engine="stub",
        actor=None,
        grid_cells=["A1"],
        tile_size=32,
        overlap=0.0,
    )
    assert "error" not in proposed, proposed
    assert proposed["staged"] is True
    assert len(opened) == 1

    opened.clear()
    accepted = stage_proposals(
        tmp_path,
        image_path=str(img_path),
        actor=None,
        assignments=kept(proposed, (0, "bud")),
    )
    assert "error" not in accepted, accepted
    assert len(opened) == 1


def test_a_staging_is_one_publication_with_one_audit_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A staging is one publication with one audit line.

    ``publish`` takes encoded ``Document`` values and no callable writer, and the staging writes
    one ``prediction_bucket_published`` audit line and no ``stage_proposals`` line.
    """
    import inspect

    from tcip_mcp.buckets import Document, publish
    from tcip_mcp.tools.proposal_tools import (
        propose_annotations,
        stage_proposals,
    )
    from tests._audit_fixtures import audit_rows

    assert Document._fields == ("source", "data", "dropped")
    assert "Callable" not in str(inspect.signature(publish))
    image = image_dir(tmp_path, "2026-01-01") / "leaf.jpg"
    _make_image(image)
    install_stub_proposer(monkeypatch, [_candidate(0, 5.0)])
    proposed = propose_annotations(
        tmp_path, image_path=str(image), engine="stub", actor=None
    )
    assert "error" not in proposed, proposed

    accepted = stage_proposals(
        tmp_path,
        image_path=str(image),
        actor=None,
        assignments=kept(proposed, (0, "bud")),
    )

    assert "error" not in accepted, accepted
    assert len(audit_rows(tmp_path, "prediction_bucket_published")) == 1
    assert audit_rows(tmp_path, "stage_proposals") == []
