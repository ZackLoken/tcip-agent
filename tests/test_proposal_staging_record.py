"""Tests of the proposal staging record.

Each staged run is addressed by its image's place in the dataset and the run's own id, and carries
the content identity of the pixels the engine ran on.

``propose_annotations`` writes the record and answers its ``staging_id``; ``stage_proposals``
reads back the run that id names and refuses when the image does not match the identity that run
recorded. Both tools run over a stub engine installed at
``tcip_mcp.pipelines.proposal.resolve_proposer`` or named as a dotted ``module:factory``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import numpy as np
import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
from tests._mapping_fixtures import CROP
from tests._producer_fixtures import (
    dataset_registered,
    install_stub_proposer,
    kept,
    write_image,
)

if TYPE_CHECKING:
    from pydantic import JsonValue

    from tcip_mcp.pipelines.raster_source import RasterSource
    from tcip_mcp.tools.vision_tools import DisplayRead
    from tcip_mcp.workspace import BoundProject
    from tcip_store import Key


@pytest.fixture(autouse=True)
def _registered(tmp_path: Path) -> None:
    """Register ``tmp_path``, the dataset root every image here is written under, in itself."""
    dataset_registered(tmp_path, tmp_path, crop=CROP)


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
    assert proposed_first.staging_id is not None

    install_stub_proposer(monkeypatch, [_candidate(0, 300.0)])
    proposed_second = propose_annotations(
        tmp_path, image_path=str(second), engine="stub", actor=None
    )
    assert "error" not in proposed_second, proposed_second
    assert proposed_second.staging_id is not None

    accepted = stage_proposals(
        tmp_path,
        image_path=str(first),
        actor=None,
        assignments=kept(proposed_first, (0, "bud")),
    )
    assert "error" not in accepted, accepted

    assert _staged_capture(tmp_path, accepted.bucket) == "2026-01-01"
    anns = _staged_annotations(tmp_path, accepted.bucket, "leaf")
    assert len(anns) == 1
    xs = [p[0] for ring in anns[0].geometry.rings for p in ring]
    assert min(xs) == pytest.approx(5.0)


def test_a_staging_publishes_under_the_root_its_staging_record_is_keyed_by(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A staged proposal's bucket is published under the root its staging record is keyed by.

    The image is in a capture named ``images`` of the root ``data``, a position read two ways and
    settled by the root carrying its own state. The staging record's key is the door's reading
    of the image at entry; the publication's audit line is the publication's own.
    """
    import tcip_store as ts
    from tcip_mcp.tools.proposal_tools import (
        PROPOSAL_STAGING_STORE,
        propose_annotations,
        stage_proposals,
    )
    from tests._audit_fixtures import audit_rows

    root = dataset_registered(tmp_path, tmp_path / "data", crop=CROP)
    image = image_dir(root, "images") / "leaf.jpg"
    _make_image(image)
    install_stub_proposer(monkeypatch, [_candidate(0, 5.0)])
    proposed = propose_annotations(
        tmp_path, image_path=str(image), engine="stub", actor=None
    )
    staged = stage_proposals(
        tmp_path,
        image_path=str(image),
        actor=None,
        assignments=kept(proposed, (0, "bud")),
    )
    assert "error" not in staged, staged

    (key,) = ts.keys(PROPOSAL_STAGING_STORE, str(root))
    rows = audit_rows(Path(key.root), "prediction_bucket_published")
    assert [r["arguments"]["dataset_root"] for r in rows] == [key.root]
    assert staged.bucket.endswith("/".join(key.parts))


def test_assignments_naming_no_run_are_refused_by_their_type() -> None:
    from pydantic import ValidationError

    from tcip_mcp.tools.proposal_tools import StagedAssignments

    with pytest.raises(ValidationError, match="staging_id"):
        StagedAssignments.model_validate(
            {"candidates": [{"candidate_id": 0, "subject": "bud"}]}
        )


def test_assignments_keeping_no_candidate_are_refused_by_their_type() -> None:
    from pydantic import ValidationError

    from tcip_mcp.tools.proposal_tools import StagedAssignments

    with pytest.raises(ValidationError, match="candidates"):
        StagedAssignments.model_validate(
            {"staging_id": "run", "candidates": []}
        )


def test_a_kept_id_the_run_did_not_propose_refuses_naming_the_runs_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A kept candidate id the run's record does not hold refuses, naming it and the run's ids.

    Nothing is staged; the run's own ids, kept, still stage.
    """
    from tcip_mcp.buckets import buckets_under
    from tcip_mcp.tools.proposal_tools import (
        Staged,
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

    refused = stage_proposals(
        tmp_path,
        image_path=str(img_path),
        actor=None,
        assignments=kept(proposed, (0, "bur"), (7, "bur")),
    )

    assert isinstance(refused, dict)
    assert "[7]" in refused["error"] and "[0, 1]" in refused["error"]
    assert buckets_under(tmp_path) == []
    staged = stage_proposals(
        tmp_path,
        image_path=str(img_path),
        actor=None,
        assignments=kept(proposed, (0, "bur"), (1, "bur")),
    )
    assert isinstance(staged, Staged) and staged.proposal_count == 2


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
    assert result.staging_id is not None

    key = ts.Key(
        PROPOSAL_STAGING_STORE,
        str(tmp_path),
        ("2026-04-01", "sample", result.staging_id),
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


def test_propose_outside_a_dataset_tree_refuses_before_the_engine_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A proposal outside a dataset tree refuses, naming where a run is staged.

    A path with no dataset address to stage under runs no engine and writes no staging record;
    the same image under a dataset's images/ tree still proposes and stages.
    """
    from tcip_mcp.tools.proposal_tools import (
        ProposedCandidates,
        propose_annotations,
    )

    loose = tmp_path / "loose.jpg"
    _make_image(loose)
    install_stub_proposer(monkeypatch, [_candidate(0, 5.0)])

    refused = propose_annotations(
        tmp_path, image_path=str(loose), engine="stub", actor=None
    )
    assert isinstance(refused, dict) and "images/ tree" in refused["error"], (
        refused
    )
    assert not (tmp_path / ".tcip" / "state" / "proposals").exists()

    placed = image_dir(tmp_path, "2026-01-01") / "placed.jpg"
    _make_image(placed)
    proposed = propose_annotations(
        tmp_path, image_path=str(placed), engine="stub", actor=None
    )
    assert (
        isinstance(proposed, ProposedCandidates)
        and len(proposed.candidates) == 1
    )


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
    assert proposed.staging_id is not None

    accepted = stage_proposals(
        tmp_path,
        image_path=str(img_path),
        actor=None,
        assignments=kept(proposed, (0, "bud")),
    )
    assert "error" not in accepted, accepted

    assert _staged_capture(tmp_path, accepted.bucket) == capture
    assert len(_staged_annotations(tmp_path, accepted.bucket, "sample")) == 1


def test_propose_then_accept_through_a_band_groups_manifest_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propose then accept work through a band group's manifest path.

    A band-grouped capture is addressed by its manifest path on both sides: the proposal stages,
    and the accept writes one annotation into the capture's bucket.
    """
    import tifffile

    from tcip_mcp.tools.proposal_tools import (
        propose_annotations,
        stage_proposals,
    )
    from tests._producer_fixtures import write_band_group

    images_dir = image_dir(tmp_path, UNDATED_BUCKET)
    images_dir.mkdir(parents=True)
    bands = {}
    for name in ("Red", "Green", "Blue"):
        band_path = images_dir / f"capture_{name}.tif"
        tifffile.imwrite(str(band_path), np.full((48, 48), 30, dtype=np.uint8))
        bands[name] = band_path
    manifest = write_band_group(
        images_dir, "capture", bands, central_wavelength_nm=None
    )

    install_stub_proposer(monkeypatch, [_candidate(0, 5.0)])
    proposed = propose_annotations(
        tmp_path, image_path=str(manifest), engine="stub", actor=None
    )
    assert "error" not in proposed, proposed
    assert proposed.staging_id is not None

    accepted = stage_proposals(
        tmp_path,
        image_path=str(manifest),
        actor=None,
        assignments=kept(proposed, (0, "bud")),
    )
    assert "error" not in accepted, accepted

    assert _staged_capture(tmp_path, accepted.bucket) == UNDATED_BUCKET
    assert len(_staged_annotations(tmp_path, accepted.bucket, "capture")) == 1


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
    from tcip_mcp.tools.proposal_tools import (
        PROPOSAL_STAGING_STORE,
        propose_annotations,
    )
    from tests._producer_fixtures import write_band_group

    images_dir = image_dir(tmp_path, UNDATED_BUCKET)
    images_dir.mkdir(parents=True)
    bands = {}
    for name in ("Red", "Green", "Blue"):
        band_path = images_dir / f"capture_{name}.tif"
        tifffile.imwrite(str(band_path), np.full((48, 48), 30, dtype=np.uint8))
        bands[name] = band_path
    manifest = write_band_group(
        images_dir, "capture", bands, central_wavelength_nm=None
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
    assert admitted.staging_id is not None, admitted


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
    document = _staged(tmp_path, first.bucket, "twice")
    written = tcip_store.read_versioned(document).version

    second = stage_proposals(
        tmp_path,
        image_path=str(img_path),
        actor=None,
        assignments=kept(proposed, (1, "nut")),
    )
    assert "already" in second["error"] and first.bucket in second["error"]
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
    assert first.staging_id != second.staging_id, (first, second)

    def staged_x(proposed: dict) -> float:
        accepted = stage_proposals(
            tmp_path,
            image_path=str(img_path),
            actor=None,
            assignments=kept(proposed, (0, "bud")),
        )
        assert "error" not in accepted, accepted
        assert accepted.bucket.endswith(f"/{proposed.staging_id}")
        (ann,) = _staged_annotations(tmp_path, accepted.bucket, "rerun")
        return min(p[0] for ring in ann.geometry.rings for p in ring)

    assert staged_x(first) == pytest.approx(5.0)
    assert staged_x(second) == pytest.approx(40.0)


def test_a_run_finding_nothing_stages_nothing_and_leaves_an_earlier_run_staged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run that proposes nothing answers nothing and leaves an earlier run as it was.

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
    assert first.staging_id is not None, first

    install_stub_proposer(monkeypatch, [])
    second = propose_annotations(
        tmp_path, image_path=str(img_path), engine="stub", actor=None
    )
    assert second is None, second
    acts, _cursor = acts_of(tmp_path, ("propose_annotations",))
    assert [act.arguments["staging_id"] for act in acts] == [first.staging_id]

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
        GridRegion,
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
        region=GridRegion(cells=["A1"], tile_size=32, overlap=0.0),
    )
    assert "error" not in proposed, proposed
    assert proposed.staging_id is not None
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


STUB_ENGINE = "tests.proposal_stub:factory"
"""A dotted engine proposing one candidate box, ``tests.proposal_stub``'s."""

UNRUNNABLE_ENGINE = "tests.test_proposal_staging_record:unrunnable_proposer"
"""A dotted engine whose pass fails the test that reaches it (:func:`unrunnable_proposer`)."""


class _UnrunnableProposer:
    """A proposal engine whose pass fails the test that reaches it."""

    def propose(self, image_path: str, **params: object) -> NoReturn:
        pytest.fail(f"the proposal engine ran over {image_path}")


def unrunnable_proposer() -> _UnrunnableProposer:
    """Return the engine :data:`UNRUNNABLE_ENGINE` names."""
    return _UnrunnableProposer()


def _unregistered_image(tmp_path: Path) -> Path:
    """Ingest one 64 px image into the dataset root ``tmp_path / "field"``, never registering it.

    Returns the ingested image's path.
    """
    from tests._producer_fixtures import ingested_unregistered

    _make_image(tmp_path / "raw" / "leaf.png")
    return (
        ingested_unregistered(tmp_path / "field", tmp_path / "raw")
        / "leaf.png"
    )


def test_a_proposal_under_a_root_never_registered_refuses_before_its_engine_runs(
    tmp_path: Path, bound: BoundProject
) -> None:
    """A proposal over an image of a root never registered refuses before its engine runs.

    Called through the server, the answer is the one sentence ``require_dataset_identity``
    refuses the root with, naming it and ``register_dataset`` and nothing of the images tree,
    and nothing is staged or logged. Registered, the root admits a proposal.
    """
    from tests._producer_fixtures import (
        left_written,
        register_dataset_at,
        unregistered_refusal,
    )
    from tests._server_calls import called_through_the_server

    image = _unregistered_image(tmp_path)
    root = tmp_path / "field"
    before = left_written(tmp_path, root)

    refused = called_through_the_server(
        "propose_annotations",
        {"image_path": str(image), "engine": UNRUNNABLE_ENGINE},
        bound=bound,
    )

    assert refused == {"error": unregistered_refusal(root)}, refused
    assert left_written(tmp_path, root) == before
    register_dataset_at(tmp_path, root, crop=CROP)
    registered = left_written(tmp_path, root)
    proposed = called_through_the_server(
        "propose_annotations",
        {"image_path": str(image), "engine": STUB_ENGINE},
        bound=bound,
    )
    assert isinstance(proposed, dict) and proposed["staging_id"], proposed
    assert left_written(tmp_path, root) != registered


@pytest.mark.parametrize(
    "assigned", [False, True], ids=["explicit", "assignments"]
)
def test_a_stage_under_a_root_never_registered_refuses_naming_it_and_writes_nothing(
    tmp_path: Path, bound: BoundProject, assigned: bool
) -> None:
    """Either regime of a stage over an image of a root never registered refuses naming it.

    Called through the server, the answer is the one sentence ``require_dataset_identity``
    refuses the root with, and no bucket is published or logged. Registered, the root admits the
    stage of a proposal run over the image or of explicit shapes.
    """
    from tests._producer_fixtures import (
        left_written,
        register_dataset_at,
        unregistered_refusal,
    )
    from tests._server_calls import called_through_the_server

    image = _unregistered_image(tmp_path)
    root = tmp_path / "field"

    def staged(staging_id: str) -> JsonValue:
        shapes = (
            {
                "assignments": {
                    "staging_id": staging_id,
                    "candidates": [{"candidate_id": 0, "subject": "bud"}],
                }
            }
            if assigned
            else {
                "model_name": "detector",
                "boxes": [
                    {
                        "subject": "bud",
                        "conf": 0.9,
                        "cx": 0.5,
                        "cy": 0.5,
                        "w": 0.25,
                        "h": 0.25,
                    }
                ],
            }
        )
        return called_through_the_server(
            "stage_proposals",
            {"image_path": str(image), **shapes},
            bound=bound,
        )

    before = left_written(tmp_path, root)

    refused = staged("staging-never-proposed")

    assert refused == {"error": unregistered_refusal(root)}, refused
    assert left_written(tmp_path, root) == before
    register_dataset_at(tmp_path, root, crop=CROP)
    proposed = called_through_the_server(
        "propose_annotations",
        {"image_path": str(image), "engine": STUB_ENGINE},
        bound=bound,
    )
    assert isinstance(proposed, dict), proposed
    proposed_only = left_written(tmp_path, root)
    accepted = staged(str(proposed["staging_id"]))
    assert isinstance(accepted, dict) and accepted["bucket"], accepted
    assert left_written(tmp_path, root) != proposed_only
