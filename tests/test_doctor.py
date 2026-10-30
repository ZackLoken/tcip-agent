"""The data-state findings ``diagnose_project`` answers over a project, each at its level."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp import traits
from tcip_mcp.dataset_layout import image_dir
from tcip_mcp.model_registry import ModelRegistry
from tcip_mcp.server import binding_for
from tcip_mcp.subject_registry import Subject, SubjectRegistry
from tcip_mcp.tools import doctor_tools as doctor
from tests import _trait_fixtures as fx
from tests._audit_fixtures import audit_rows
from tests._producer_fixtures import (
    image_label_key,
    label_image,
    mark_complete,
    registry_of,
    registry_over,
    write_image,
)
from tests._web_fixtures import new_project

if TYPE_CHECKING:
    from tcip_annotation.json_io import LabelDocument
    from tcip_mcp.buckets import Proposed
    from tcip_mcp.model_registry import Producer
    from tcip_mcp.workspace import BoundProject
    from tcip_store import Key, ReadDefault, Versioned


def _answered(bound: BoundProject) -> list[tuple[str, str]]:
    """Return the findings ``diagnose_project`` answers for ``bound`` through a client's handshake.

    The call is asserted to be answered, never refused.
    """
    from tests.test_agent_identity_records import (
        _body,
        results_through_handshake,
    )

    (result,) = results_through_handshake(
        [("diagnose_project", {})], bound=bound
    )
    assert not result.is_error, result
    return [(level, message) for level, message in _body(result)["findings"]]


def _diagnosed(root: Path) -> list[tuple[str, str]]:
    """Return the findings ``diagnose_project`` answers for a server started for ``root``."""
    return _answered(binding_for(root))


def _messages(
    found: list[tuple[str, str]], needle: str
) -> list[tuple[str, str]]:
    """Return the findings of ``found`` whose message holds ``needle``."""
    return [(level, msg) for level, msg in found if needle in msg]


def _record_localizing(tmp_path: Path, name: str, localization: str) -> dict:
    """Return a trait record as the proposing producer writes it, its ``localization`` replaced.

    The record is the revision ``fx.propose`` writes under a scratch project.
    """
    revision = fx.propose(
        tmp_path / "scratch_project", fx.entry(name, ("leaf_length",))
    )
    record = {"revisions": [revision.model_dump(mode="json")]}
    record["revisions"][0]["entry"]["localization"] = localization
    return record


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    images = image_dir(root, "2026-02-11")
    images.mkdir(parents=True)
    new_project(root)
    (root / ".tcip" / "state").mkdir(parents=True, exist_ok=True)
    registry_over(root, registry_of("bud"))
    a, b, c = (
        write_image(images / f"{name}.JPG", (32, 32))
        for name in ("IMG_A", "IMG_B", "IMG_C")
    )
    # A: confirmed negative (empty + a mark). B: empty with no mark. C: holds an object.
    label_image(a, [], 32, 32, keep_empty=True)
    label_image(b, [], 32, 32, keep_empty=True)
    label_image(
        c, [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 32, 32
    )
    mark_complete(a, "bud", project=root)
    return root


def _register_checkpoint(root: Path, name: str, checkpoint_path: Path) -> Path:
    """Write a checkpoint at ``checkpoint_path`` and register it under ``name``; return its path.

    The registration goes through the registry's own producer.
    """
    from tests._verified_checkpoint_fixtures import produced_checkpoint

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    produced_checkpoint(checkpoint_path, f"{name} weights")
    ModelRegistry(str(root)).register_model(
        name, str(checkpoint_path), actor=None
    )
    return checkpoint_path


def _register_absent_checkpoint(
    root: Path, name: str, checkpoint_path: Path
) -> None:
    """Register a checkpoint (:func:`_register_checkpoint`), then delete its file."""
    _register_checkpoint(root, name, checkpoint_path).unlink()


def test_doctor_flags_registry_checkpoint_path_under_a_temp_directory(
    tmp_path: Path,
) -> None:
    """The doctor flags a registry checkpoint path under a temp directory as pollution.

    ``tmp_path`` sits under pytest's own temp tree, so a checkpoint there outside the project is
    test pollution.
    """
    root = _project(tmp_path)
    _register_absent_checkpoint(
        root, "junk", tmp_path / "elsewhere" / "model.pt"
    )

    found = _diagnosed(root)

    assert [
        level
        for level, msg in _messages(found, "not marked complete")
        if "IMG_B" in msg
    ] == ["error"], found
    assert [
        level for level, msg in _messages(found, "test/temp") if "junk" in msg
    ] == ["error"], found
    assert not _messages(found, "IMG_A"), found  # confirmed negative is clean
    assert not _messages(found, "IMG_C"), found  # an annotated image is clean


def test_doctor_reports_a_stem_collision_and_completes(tmp_path: Path) -> None:
    """The doctor reports a stem collision as an error finding and completes.

    A capture holding two identities for one stem is named as an error finding, and the call is
    answered rather than refused.
    """
    root = tmp_path / "proj"
    images = image_dir(root, "2026-02-11")
    images.mkdir(parents=True)
    new_project(root)
    registry_over(root, registry_of("bud"))
    jpeg = write_image(images / "foo.jpg", (32, 32))
    write_image(images / "foo.png", (32, 32))
    label_image(jpeg, [], 32, 32, keep_empty=True)

    found = _diagnosed(root)

    assert [
        level for level, msg in _messages(found, "foo.jpg") if "foo.png" in msg
    ] == ["error"], found


def test_doctor_flags_a_trait_record_that_will_not_read(
    tmp_path: Path,
) -> None:
    """The doctor flags a trait record that will not read.

    For a trait record the schema refuses, the doctor names the record and its refused
    localization in an error finding.
    """
    import tcip_store as ts

    root = _project(tmp_path)
    record = _record_localizing(tmp_path, "unicorn", "unicorn_match")
    ts.replace(
        traits.trait_key(root, "unicorn"), record, expect=ts.Version.ABSENT
    )

    found = _diagnosed(root)

    ((level, msg),) = _messages(found, "'unicorn' will not read")
    assert level == "error"
    assert "unicorn_match" in msg


def _clean_project(tmp_path: Path) -> Path:
    """Make a project of one image under capture ``d`` labeled with one breeder-drawn ``bud``."""
    root = tmp_path / "clean"
    images = image_dir(root, "d")
    images.mkdir(parents=True)
    new_project(root)
    registry_over(root, registry_of("bud"))
    label_image(
        write_image(images / "IMG_A.JPG", (32, 32)),
        [
            Annotation(
                subject="bud",
                geometry=BBox(1, 1, 9, 9),
                created_by="user:breeder",
            )
        ],
        32,
        32,
    )
    return root


def test_doctor_finds_nothing_on_a_clean_project(tmp_path: Path) -> None:
    assert _diagnosed(_clean_project(tmp_path)) == []


def _layout_project(
    tmp_path: Path, date: str | None, name: str = "resolved"
) -> tuple[Path, Path]:
    """Make a project whose image tree for capture ``date`` is placed by the layout resolver.

    Returns the project root and the capture directory made under it.
    """
    root = tmp_path / name
    images = image_dir(root, date)
    images.mkdir(parents=True)
    new_project(root)
    (root / ".tcip" / "state").mkdir(parents=True, exist_ok=True)
    registry_over(
        root,
        SubjectRegistry(
            subjects=(
                Subject(name="bud", description=""),
                Subject(name="leaf", description=""),
            )
        ),
    )
    return root, images


def test_the_tool_through_a_handshake_answers_the_functions_findings_and_leaves_no_line(
    tmp_path: Path,
) -> None:
    """Through a client's handshake ``diagnose_project`` answers what the function returns.

    Over a project holding an error, a warning and an info finding, whose messages sort in
    another order, the answer holds them errors first, then the warning, then the info, as the
    function returns them, and the project's audit log is left as it was.
    """
    date = "2026-03-04"
    root, images = _layout_project(tmp_path, date)
    labeled, _unlabeled = (
        write_image(images / f"{name}.JPG", (32, 32))
        for name in ("IMG_E", "IMG_U")
    )
    label_image(labeled, [], 32, 32, keep_empty=True)
    fx.propose(root, fx.entry("leaf", ("leaf_length",)))
    before = audit_rows(root)

    answered = _diagnosed(root)

    assert [level for level, _ in answered] == ["error", "warn", "info"]
    assert answered == doctor.diagnose_project(root)["findings"]
    assert audit_rows(root) == before


def test_diagnosing_a_rollback_journal_database_changes_and_locks_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``diagnose_project`` reads a database in rollback-journal mode as it is.

    The project's database, its connections released, is put in rollback-journal mode, as a
    writer other than this store would leave it. Diagnosing it through a client's handshake
    takes no transition lock, leaves the file's bytes and its journal mode as they were, and
    leaves no ``-wal`` or ``-shm`` beside it.
    """
    import sqlite3

    import tcip_store as ts
    from tcip_store import sqlite_backend
    from tcip_store.file_backend import database_file
    from tests.test_store_contract import _journal_mode

    date = "2026-03-04"
    root, images = _layout_project(tmp_path, date)
    label_image(
        write_image(images / "IMG_E.JPG", (32, 32)),
        [],
        32,
        32,
        keep_empty=True,
    )
    bound = binding_for(root)
    ts.release_root(root)
    db = database_file(str(root))
    conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        conn.execute("pragma journal_mode = delete")
    finally:
        conn.close()
    held: list[str] = []
    real_lock = sqlite_backend.transition_lock

    def counted(root: str, **kwargs: float) -> object:
        held.append(root)
        return real_lock(root, **kwargs)

    monkeypatch.setattr(sqlite_backend, "transition_lock", counted)
    before = db.read_bytes()

    assert _messages(_answered(bound), "IMG_E")
    ts.release_root(root)

    assert held == []
    assert db.read_bytes() == before
    assert _journal_mode(db) == "delete"
    assert not [
        sibling
        for sibling in (
            db.with_name(f"{db.name}-wal"),
            db.with_name(f"{db.name}-shm"),
        )
        if sibling.exists()
    ]


def test_an_unmarked_empty_label_is_named_by_its_capture_and_stem(
    tmp_path: Path,
) -> None:
    date = "2026-03-04"
    root, images = _layout_project(tmp_path, date)
    image = write_image(images / "IMG_R.JPG", (48, 32))
    label_image(image, [], 48, 32, keep_empty=True)

    found = _diagnosed(root)

    ((level, msg),) = _messages(found, "not marked complete")
    assert level == "error"
    assert f"{date}/IMG_R" in msg


def test_a_mark_for_any_subject_finishes_an_empty_label_on_a_dateless_dataset(
    tmp_path: Path,
) -> None:
    """An undated capture's empty label marked complete for a subject is a confirmed negative."""
    from tcip_mcp.dataset_layout import UNDATED_BUCKET

    root = tmp_path / "dateless"
    images = image_dir(root, UNDATED_BUCKET)
    images.mkdir(parents=True)
    new_project(root)
    registry_over(root, registry_of("bud"))
    image = write_image(images / "IMG_F.JPG", (40, 24))
    label_image(image, [], 40, 24, keep_empty=True)
    mark_complete(image, "bud", project=root)

    found = _diagnosed(root)

    assert not _messages(found, "not marked complete"), found


def test_registry_findings_are_read_through_the_registrys_own_entry_shape(
    tmp_path: Path,
) -> None:
    """Registry findings are read through the registry's own entry shape.

    Two entries written by ModelRegistry whose checkpoints are gone are each named, as an error,
    with their own name and path.
    """
    root, _images = _layout_project(tmp_path, "2026-03-04")
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    from tests._verified_checkpoint_fixtures import produced_checkpoint

    registry = ModelRegistry(str(root))
    paths = {}
    for name, payload in (
        ("currant_bud_detector_v1", "weights"),
        ("chestnut_burr_counter_v3", "other weights"),
    ):
        ckpt = produced_checkpoint(ckpt_dir / f"{name}.pt", payload)
        registry.register_model(
            name=name, checkpoint_path=str(ckpt), metrics={}, actor=None
        )
        paths[name] = ckpt
    for ckpt in paths.values():
        ckpt.unlink()

    found = _diagnosed(root)

    entries = _messages(found, "registry entry")
    assert [level for level, _ in entries] == ["error", "error"], found
    for name, ckpt in paths.items():
        assert any(name in msg and str(ckpt) in msg for _, msg in entries), (
            found
        )


def test_a_missing_checkpoint_and_a_test_checkpoint_are_distinct_registry_findings(
    tmp_path: Path,
) -> None:
    """A missing checkpoint and a temp-tree checkpoint are distinct registry findings.

    An entry pointing at a checkpoint under pytest's temp tree is reported as pollution and not
    as missing, and an entry whose checkpoint is absent inside the project is reported as
    missing and not as pollution, each as an error.
    """
    root, _images = _layout_project(tmp_path, "2026-03-04")
    # Inside the project, so never pollution; outside it under pytest's own temp tree, pollution.
    ghost = root / ".tcip" / "models" / "orchard.pt"
    scratch = tmp_path / "scratch" / "run" / "last.pt"
    _register_absent_checkpoint(root, "orchard_detector_v2", ghost)
    _register_absent_checkpoint(root, "scratch_detector", scratch)

    found = _diagnosed(root)

    entries = _messages(found, "registry entry")
    assert [level for level, _ in entries] == ["error", "error"], found
    ghost_line = next(m for _, m in entries if "orchard_detector_v2" in m)
    scratch_line = next(m for _, m in entries if "scratch_detector" in m)
    assert "checkpoint missing" in ghost_line and "test/temp" not in ghost_line
    assert (
        "test/temp" in scratch_line
        and "checkpoint missing" not in scratch_line
    )


def test_a_checkpoint_under_a_temp_rooted_project_is_not_pollution(
    tmp_path: Path,
) -> None:
    """A checkpoint under a temp-rooted project is not pollution.

    ``tmp_path`` itself sits under a temp tree (pytest's own fixture), and a checkpoint resolving
    inside the project is reported neither as pollution nor as missing.
    """
    root, _images = _layout_project(tmp_path, "2026-03-04")
    from tests._verified_checkpoint_fixtures import produced_checkpoint

    ckpt_dir = root / ".tcip" / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    produced_checkpoint(ckpt_dir / "m.pt", "weights")
    ModelRegistry(str(root)).register_model(
        name="m", checkpoint_path=str(ckpt_dir / "m.pt"), actor=None
    )

    found = _diagnosed(root)

    assert not _messages(found, "test/temp"), found
    assert not _messages(found, "checkpoint missing"), found


def test_image_census_counts_every_capture_the_loaders_admit(
    tmp_path: Path,
) -> None:
    """The doctor's image census keys an .npz capture beside a .jpg one.

    Each capture is keyed to its own capture and stem.
    """
    images = image_dir(tmp_path, "2026-03-04")
    images.mkdir(parents=True)
    (images / "plotA_0_0.npz").write_bytes(b"\x00")
    (images / "plotA_0_1.jpg").write_bytes(b"\xff\xd8")

    census = doctor._census(tmp_path, [])
    assert census is not None
    assert sorted(key.parts for key in census["images"].values()) == [
        ("2026-03-04", "plotA_0_0"),
        ("2026-03-04", "plotA_0_1"),
    ]


def test_one_stem_in_two_captures_is_two_images_each_paired_with_its_own_label(
    tmp_path: Path,
) -> None:
    """One stem in two captures stays two images, one labeled and one unlabeled.

    With two captures each holding ``same.png``, only one labeled, one image has no label record
    and no label lacks its image, at the census and at the doctor alike.
    """
    from tcip_mcp.tools.data_tools import scan_dataset

    root, first = _layout_project(tmp_path, "2026-03-04")
    labeled, _unlabeled = (
        write_image(images / "same.png", (32, 32))
        for images in (first, image_dir(root, "2026-03-05"))
    )
    label_image(
        labeled,
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))],
        32,
        32,
    )

    scan = scan_dataset(root, str(root))
    findings: list[tuple[str, str]] = []
    census = doctor._census(root, findings)
    assert census is not None, findings
    doctor.check_data_quality(root, findings, census=census)

    assert (scan["paired_images"], scan["unlabeled_images"]) == (1, 1), scan
    assert not [msg for _, msg in findings if "no matching image" in msg], (
        findings
    )
    (unlabeled,) = [msg for _, msg in findings if "no label record" in msg]
    assert unlabeled.startswith("1 of 2") and "2026-03-05/same" in unlabeled, (
        unlabeled
    )


def test_only_the_unreadable_trait_record_is_reported(tmp_path: Path) -> None:
    """Only the unreadable trait record is reported, by name and reason.

    A record the schema refuses is reported as an error by name and reason; a confirmed trait
    beside it stays silent.
    """
    import tcip_store as ts

    root, _images = _layout_project(tmp_path, "2026-03-04")
    fx.propose_and_confirm(root, fx.entry("leaf", ("leaf_length",)))
    ts.replace(
        traits.trait_key(root, "burr_size"),
        _record_localizing(tmp_path, "burr_size", "not_a_localization"),
        expect=ts.Version.ABSENT,
    )

    found = _diagnosed(root)

    ((level, msg),) = _messages(found, "trait '")
    assert level == "error"
    assert "'burr_size' will not read" in msg
    assert "not_a_localization" in msg


def test_doctor_is_silent_on_a_confirmed_latest_revision(
    tmp_path: Path,
) -> None:
    root, _images = _layout_project(tmp_path, "2026-03-04")
    fx.propose_and_confirm(root, fx.entry("leaf", ("leaf_length",)))

    found = _diagnosed(root)

    assert not _messages(found, "trait '"), found


def test_doctor_warns_on_an_unconfirmed_latest_revision(
    tmp_path: Path,
) -> None:
    """The doctor warns, in one finding and with no error, on an unconfirmed latest revision.

    The warning holds for a trait whose latest revision the breeder has not confirmed, a
    confirmed earlier one notwithstanding.
    """
    root, _images = _layout_project(tmp_path, "2026-03-04")
    fx.propose_and_confirm(root, fx.entry("leaf", ("leaf_length",)))
    fx.propose(
        root, fx.entry("leaf", ("leaf_length",), notes="a second reading")
    )

    found = _diagnosed(root)

    assert not [msg for level, msg in found if level == "error"], found
    ((level, msg),) = _messages(found, "trait '")
    assert level == "warn"
    assert "the latest revision (2) of trait 'leaf' is not confirmed" in msg
    assert "Setup tab" in msg


def test_doctor_reports_an_undecodable_trait_record_without_aborting(
    tmp_path: Path,
) -> None:
    from tests._record_damage_fixtures import damage_record

    root, _images = _layout_project(tmp_path, "2026-03-04")
    fx.propose_and_confirm(root, fx.entry("leaf", ("leaf_length",)))
    damage_record(traits.trait_key(root, "leaf"), b"{not valid json")

    found = _diagnosed(root)

    ((level, msg),) = _messages(found, "trait '")
    assert level == "error"
    assert "will not read" in msg


def test_doctor_reports_a_trait_record_holding_no_revision_without_aborting(
    tmp_path: Path,
) -> None:
    """A trait record holding no revision is an error finding, and the call is answered.

    A trait proposed and confirmed through ``propose_trait`` beside it still reads and stays
    silent.
    """
    import tcip_store as ts

    root, _images = _layout_project(tmp_path, "2026-03-04")
    fx.propose_and_confirm(root, fx.entry("leaf", ("leaf_length",)))
    ts.replace(
        traits.trait_key(root, "stem"),
        {"revisions": []},
        expect=ts.Version.ABSENT,
    )

    found = _diagnosed(root)

    ((level, msg),) = _messages(found, "trait '")
    assert level == "error"
    assert "'stem' will not read" in msg
    assert traits.read_trait("leaf", root).latest.confirmed


def _published_bucket(
    root: Path,
    images_dir: Path,
    name: str,
    stems: tuple[str, ...],
    producer: Producer | Proposed,
) -> list[Path]:
    """Publish the bucket ``name`` by ``producer`` over a new image of each of ``stems``.

    The images are written into the capture ``images_dir`` of ``root``; the bucket holds one
    ``bud`` box per image, its documents encoded as the proposal door encodes them. Returns the
    images.
    """
    from tcip_mcp.buckets import Document, publish, source_root
    from tcip_mcp.pipelines.data.selection import ClassScope

    images = [write_image(images_dir / f"{s}.JPG", (32, 32)) for s in stems]
    box = Annotation(
        subject="bud", geometry=BBox(1, 1, 9, 9), score=0.9, created_by="agent"
    )
    publish(
        root,
        source_root(images),
        name,
        [
            Document(str(i), json_io.document_payload([box], 32, 32))
            for i in images
        ],
        producer=producer,
        scope=ClassScope(),
        execution=None,
        raster_path=None,
        raster_identity=None,
        assessment_id=None,
        actor=None,
    )
    return images


def _reviewed_bucket(
    root: Path, images_dir: Path, stems: tuple[str, ...]
) -> str:
    """Publish the proposal bucket ``proposals`` (:func:`_published_bucket`) and review each image.

    A save of each image rejecting its box writes the image's verdict shard. Returns the bucket's
    name.
    """
    from tcip_mcp.buckets import Proposed
    from tcip_mcp.dataset_layout import (
        Decisions,
        Gestures,
        save_label_document,
    )

    name = "proposals"
    images = _published_bucket(
        root, images_dir, name, stems, Proposed(proposed_by="agent")
    )
    for image in images:
        save_label_document(
            root,
            image_label_key(image),
            [],
            width=32,
            height=32,
            author="user:breeder",
            actor=None,
            gestures=Gestures(decide=Decisions(bucket=name, verdicts={0: "rejected"})),
        )
    return name


def _replace_documents(
    root: Path, name: str, documents: dict[str, str] | str
) -> None:
    """Replace the ``documents`` of the stored record of bucket ``name`` under ``root``."""
    import tcip_store as ts
    from tcip_mcp.dataset_layout import bucket_key

    record = ts.read(bucket_key(root, name))
    assert isinstance(record, dict)
    ts.replace(bucket_key(root, name), record | {"documents": documents})


def test_a_diagnosis_reads_a_bucket_reviewed_on_two_images_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A diagnosis reads the record of a bucket with two verdict shards once, at the store."""
    from tcip_mcp.dataset_layout import PREDICTION_BUCKETS
    from tcip_store.sqlite_backend import SqliteBackend

    root, images = _layout_project(tmp_path, "2026-03-04")
    name = _reviewed_bucket(root, images, ("IMG_A", "IMG_B"))
    reads: list[tuple[str, ...]] = []
    real_read = SqliteBackend.read_versioned

    def counted(
        self: SqliteBackend, key: Key, **default: ReadDefault
    ) -> Versioned:
        if key.store == PREDICTION_BUCKETS:
            reads.append(key.parts)
        return real_read(self, key, **default)

    monkeypatch.setattr(SqliteBackend, "read_versioned", counted)
    found = doctor.diagnose_project(root)["findings"]

    assert reads == [(name,)]
    assert not _messages(found, "verdict shard"), found


def test_a_verdict_shard_naming_an_image_its_bucket_names_no_document_for_is_a_warning(
    tmp_path: Path,
) -> None:
    """A shard whose image the bucket's record no longer names is one warning naming that image.

    The bucket's record is rewritten after both shards were written to name the first image's
    document alone; the first image's shard stays silent.
    """
    root, images = _layout_project(tmp_path, "2026-03-04")
    name = _reviewed_bucket(root, images, ("IMG_A", "IMG_B"))
    _replace_documents(root, name, {"IMG_A": "IMG_A.JPG"})

    found = _diagnosed(root)

    ((level, msg),) = _messages(found, "verdict shard")
    assert level == "warn"
    assert f"{name}/IMG_B" in msg and "names no document for" in msg


def test_doctor_names_a_bucket_whose_record_will_not_read_without_aborting(
    tmp_path: Path,
) -> None:
    """A published bucket whose record its schema refuses is named in one error finding.

    The bucket is published and reviewed on one image, and its stored record then given
    documents that are not a mapping; the call is answered rather than refused, and the image's
    verdict shard is one warning that its bucket has no readable record.
    """
    root, images = _layout_project(tmp_path, "2026-03-04")
    name = _reviewed_bucket(root, images, ("IMG_P",))
    _replace_documents(root, name, "not a mapping")

    found = _diagnosed(root)

    ((level, msg),) = _messages(found, "bucket record will not read")
    assert level == "error"
    assert repr(name) in msg
    ((level, msg),) = _messages(found, "verdict shard")
    assert level == "warn"
    assert f"{name}/IMG_P" in msg and "has no readable record" in msg


def test_a_registry_that_will_not_read_hides_no_bucket_that_will_not(
    tmp_path: Path,
) -> None:
    """A bucket whose record will not read is named beside a registry index that will not decode.

    A readable bucket published from a registered checkpoint sits beside the damaged one, and no
    registry entry is read, so none of its digests is reported unnamed by the registry.
    """
    from tcip_mcp.model_registry import (
        load_registered_checkpoint,
        registry_index_key,
    )
    from tests._record_damage_fixtures import damage_record

    root, images = _layout_project(tmp_path, "2026-03-04")
    damaged = _reviewed_bucket(root, images, ("IMG_P",))
    _replace_documents(root, damaged, "not a mapping")
    checkpoint = _register_checkpoint(
        root, "orchard_detector_v2", root / ".tcip" / "models" / "orchard.pt"
    )
    _published_bucket(
        root,
        images,
        "predictions",
        ("IMG_Q",),
        load_registered_checkpoint(checkpoint, project=root).producer,
    )
    damage_record(registry_index_key(root), b"{not valid json")

    found = _diagnosed(root)

    assert [
        level for level, _ in _messages(found, "model registry index will not")
    ] == ["error"], found
    ((level, msg),) = _messages(found, "bucket record will not read")
    assert level == "error" and repr(damaged) in msg, found
    assert not _messages(found, "no registry entry names"), found


def test_doctor_errors_on_a_project_whose_record_does_not_decode(
    tmp_path: Path,
) -> None:
    """A project record that stops decoding after the server started is an error finding."""
    from tcip_mcp.project_record import project_record_key
    from tests._record_damage_fixtures import damage_record

    root, _images = _layout_project(tmp_path, "2026-03-04")
    bound = binding_for(root)
    damage_record(project_record_key(str(root)), b"{not valid json")

    found = _answered(bound)

    assert [level for level, _ in _messages(found, "does not decode")] == [
        "error"
    ], found


def _unreadable_label(image: Path) -> None:
    """Write a label document for ``image`` and damage its stored bytes so they do not decode."""
    from tests._record_damage_fixtures import damage_record

    label_image(image, [], 32, 32, keep_empty=True)
    damage_record(image_label_key(image), b"not json {][")


def test_doctor_flags_an_unreadable_label(tmp_path: Path) -> None:
    """The doctor flags an unreadable label as an error-level finding, not a pass.

    A corrupt label document is named, as an error, as not reading.
    """
    root, images = _layout_project(tmp_path, "2026-03-04")
    _unreadable_label(write_image(images / "IMG_S.JPG", (32, 32)))

    found = _diagnosed(root)

    assert [
        level
        for level, msg in _messages(found, "will not read")
        if "IMG_S" in msg
    ] == ["error"], found


def test_a_marked_empty_label_is_clean_to_the_data_quality_check(
    tmp_path: Path,
) -> None:
    """A marked empty label is clean to the data-quality check.

    An empty label marked complete leaves the data-quality check over the census with no
    finding.
    """
    root, images = _layout_project(tmp_path, "2026-03-04")
    image = write_image(images / "IMG_S.JPG", (32, 32))
    label_image(image, [], 32, 32, keep_empty=True)
    mark_complete(image, "bud", project=root)

    findings: list[tuple[str, str]] = []
    census = doctor._census(root, findings)
    assert census is not None, findings
    doctor.check_data_quality(root, findings, census=census)
    assert findings == []


def test_a_doctor_run_reads_each_label_once_and_reports_an_unreadable_one_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A doctor run reads each label once and reports an unreadable one once.

    Over a census and its data-quality and provenance checks, each label document is read once
    and the unreadable one is reported once.
    """
    root, images = _layout_project(tmp_path, "2026-03-04")
    _unreadable_label(write_image(images / "IMG_U.JPG", (32, 32)))
    label_image(
        write_image(images / "IMG_A.JPG", (32, 32)),
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))],
        32,
        32,
    )

    reads: list[str] = []
    real_read = json_io.read_label_document

    def _counting_read(
        key: Key, *args: str | None, **kwargs: str | None
    ) -> LabelDocument:
        reads.append(key.parts[-1])
        return real_read(key, *args, **kwargs)

    monkeypatch.setattr(json_io, "read_label_document", _counting_read)
    findings: list[tuple[str, str]] = []
    census = doctor._census(root, findings)
    assert census is not None, findings
    doctor.check_data_quality(root, findings, census=census)
    doctor.check_provenance(root, findings, census=census)

    assert sorted(reads) == ["IMG_A", "IMG_U"]
    unreadable = [
        msg for _, msg in findings if "label document will not read" in msg
    ]
    assert len(unreadable) == 1, findings
    assert "IMG_U" in unreadable[0]
