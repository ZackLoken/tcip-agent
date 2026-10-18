"""tcip doctor: the data-state findings it reports over a project and its exit codes."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp import traits
from tcip_mcp.dataset_layout import image_dir
from tcip_mcp.model_registry import ModelRegistry
from tcip_mcp.subject_registry import Subject, SubjectRegistry
from tests import _trait_fixtures as fx
from tests._cli_fixtures import run_tcip
from tests._producer_fixtures import (
    image_label_key,
    label_image,
    mark_complete,
    registry_over,
)
from tests._web_fixtures import new_project


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
    images = root / "images" / "2026-02-11"
    images.mkdir(parents=True)
    (root / ".tcip" / "state").mkdir(parents=True)
    registry_over(root, SubjectRegistry(subjects=(Subject(name="bud"),)))
    for name in ("IMG_A", "IMG_B", "IMG_C"):
        Image.new("RGB", (32, 32)).save(images / f"{name}.JPG")
    # A: confirmed negative (empty + a mark). B: empty with no mark. C: holds an object.
    label_image(images / "IMG_A.JPG", [], 32, 32, keep_empty=True)
    label_image(images / "IMG_B.JPG", [], 32, 32, keep_empty=True)
    label_image(
        images / "IMG_C.JPG",
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))],
        32,
        32,
    )
    mark_complete(images / "IMG_A.JPG", "bud", project=root)
    return root


def test_doctor_help_prints_the_dispatchers_prog_argument(capsys):
    """The doctor's help prints the ``prog`` it is given.

    Called with ``prog="tcip doctor"``, ``doctor.main`` given ``--help`` exits 0 printing that
    usage line.
    """
    from tcip_mcp.cli import doctor

    with pytest.raises(SystemExit) as exc:
        doctor.main(["--help"], prog="tcip doctor")

    assert exc.value.code == 0
    assert "usage: tcip doctor " in capsys.readouterr().out


def _register_absent_checkpoint(
    root: Path, name: str, checkpoint_path: Path
) -> None:
    """Register a checkpoint through the registry's own producer, then delete its file.

    The registry entry names a checkpoint file deleted after its registration.
    """
    from tests._verified_checkpoint_fixtures import produced_checkpoint

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    produced_checkpoint(checkpoint_path, f"{name} weights")
    ModelRegistry(str(root)).register_model(name, str(checkpoint_path))
    checkpoint_path.unlink()


def test_doctor_flags_registry_checkpoint_path_under_a_temp_directory(
    tmp_path,
):
    """The doctor flags a registry checkpoint path under a temp directory as pollution.

    ``tmp_path`` sits under pytest's own temp tree, so a checkpoint there outside the project is
    test pollution.
    """
    root = _project(tmp_path)
    _register_absent_checkpoint(
        root, "junk", tmp_path / "elsewhere" / "model.pt"
    )

    res = run_tcip("doctor", [str(root)])
    assert res.returncode == 2  # errors present
    out = res.stdout
    assert (
        "IMG_B" in out and "not marked complete" in out
    )  # unconfirmed empty -> error
    assert "junk" in out and "test/temp" in out  # registry pollution -> error
    assert "IMG_A" not in out  # confirmed negative is clean
    assert "IMG_C" not in out  # an annotated image is clean


def test_doctor_reports_a_stem_collision_and_completes(tmp_path):
    """The doctor reports a stem collision as a finding and completes.

    A capture holding two identities for one stem is named as a finding, and the doctor exits 2
    with no traceback.
    """
    root = tmp_path / "proj"
    images = root / "images" / "2026-02-11"
    images.mkdir(parents=True)
    registry_over(root, SubjectRegistry(subjects=(Subject(name="bud"),)))
    Image.new("RGB", (32, 32)).save(images / "foo.jpg")
    Image.new("RGB", (32, 32)).save(images / "foo.png")
    label_image(images / "foo.jpg", [], 32, 32, keep_empty=True)

    res = run_tcip("doctor", [str(root)])

    assert res.returncode == 2
    assert "foo.jpg" in res.stdout and "foo.png" in res.stdout
    assert "Traceback" not in res.stderr


def test_doctor_flags_a_trait_record_that_will_not_read(tmp_path):
    """The doctor flags a trait record that will not read.

    For a trait record the schema refuses, the doctor names the record and its refused
    localization and exits 2.
    """
    import tcip_store as ts

    root = _project(tmp_path)
    record = _record_localizing(tmp_path, "unicorn", "unicorn_match")
    ts.replace(
        traits.trait_key(root, "unicorn"), record, expect=ts.Version.ABSENT
    )

    res = run_tcip("doctor", [str(root)])
    assert res.returncode == 2  # errors present
    assert "'unicorn' will not read" in res.stdout
    assert "unicorn_match" in res.stdout


def _clean_project(tmp_path: Path) -> Path:
    """Write a project of one image under capture ``d`` labeled with one breeder-drawn ``bud``."""
    root = tmp_path / "clean"
    images = root / "images" / "d"
    images.mkdir(parents=True)
    (root / ".tcip" / "state").mkdir(parents=True)
    registry_over(root, SubjectRegistry(subjects=(Subject(name="bud"),)))
    Image.new("RGB", (32, 32)).save(images / "IMG_A.JPG")
    label_image(
        images / "IMG_A.JPG",
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


def test_doctor_clean_project_exits_zero(tmp_path):
    root = _clean_project(tmp_path)
    new_project(root)
    res = run_tcip("doctor", [str(root)])
    assert res.returncode == 0, res.stdout


def _layout_project(
    tmp_path: Path, date: str | None, name: str = "resolved"
) -> Path:
    """Return a project whose image tree for capture ``date`` is placed by the layout resolver."""
    root = tmp_path / name
    image_dir(root, date).mkdir(parents=True)
    (root / ".tcip" / "state").mkdir(parents=True)
    registry_over(
        root,
        SubjectRegistry(subjects=(Subject(name="bud"), Subject(name="leaf"))),
    )
    return root


def _lines(stdout: str, needle: str) -> list[str]:
    return [ln for ln in stdout.splitlines() if needle in ln]


def test_an_unmarked_empty_label_is_named_by_its_capture_and_stem(tmp_path):
    date = "2026-03-04"
    root = _layout_project(tmp_path, date)
    image = image_dir(root, date) / "IMG_R.JPG"
    Image.new("RGB", (48, 32)).save(image)
    label_image(image, [], 48, 32, keep_empty=True)

    res = run_tcip("doctor", [str(root)])
    assert res.returncode == 2, res.stdout
    unmarked = _lines(res.stdout, "not marked complete")
    assert len(unmarked) == 1, res.stdout
    assert f"{date}/IMG_R" in unmarked[0]


def test_a_mark_for_any_subject_finishes_an_empty_label_on_a_dateless_dataset(
    tmp_path,
):
    """An undated capture's empty label marked complete for a subject is a confirmed negative."""
    from tcip_mcp.dataset_layout import UNDATED_BUCKET

    root = tmp_path / "dateless"
    image_dir(root, UNDATED_BUCKET).mkdir(parents=True)
    (root / ".tcip" / "state").mkdir(parents=True)
    registry_over(root, SubjectRegistry(subjects=(Subject(name="bud"),)))
    image = image_dir(root, UNDATED_BUCKET) / "IMG_F.JPG"
    Image.new("RGB", (40, 24)).save(image)
    label_image(image, [], 40, 24, keep_empty=True)
    mark_complete(image, "bud", project=root)

    res = run_tcip("doctor", [str(root)])
    assert "not marked complete" not in res.stdout, res.stdout


def test_registry_findings_are_read_through_the_registrys_own_entry_shape(
    tmp_path,
):
    """Registry findings are read through the registry's own entry shape.

    Two entries written by ModelRegistry whose checkpoints are gone are each named with their
    own name and path.
    """
    root = _layout_project(tmp_path, "2026-03-04")
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
            name=name, checkpoint_path=str(ckpt), metrics={}
        )
        paths[name] = ckpt
    for ckpt in paths.values():
        ckpt.unlink()

    res = run_tcip("doctor", [str(root)])
    assert res.returncode == 2, res.stdout
    entry_lines = _lines(res.stdout, "registry entry")
    assert len(entry_lines) == 2, res.stdout
    for name, ckpt in paths.items():
        assert any(name in ln and str(ckpt) in ln for ln in entry_lines), (
            res.stdout
        )


def test_a_missing_checkpoint_and_a_test_checkpoint_are_distinct_registry_findings(
    tmp_path,
):
    """A missing checkpoint and a temp-tree checkpoint are distinct registry findings.

    An entry pointing at a checkpoint under pytest's temp tree is reported as pollution and not
    as missing, and an entry whose checkpoint is absent inside the project is reported as
    missing and not as pollution.
    """
    root = _layout_project(tmp_path, "2026-03-04")
    # Inside the project, so never pollution; outside it under pytest's own temp tree, pollution.
    ghost = root / ".tcip" / "models" / "orchard.pt"
    scratch = tmp_path / "scratch" / "run" / "last.pt"
    _register_absent_checkpoint(root, "orchard_detector_v2", ghost)
    _register_absent_checkpoint(root, "scratch_detector", scratch)

    res = run_tcip("doctor", [str(root)])
    assert res.returncode == 2, res.stdout
    entry_lines = _lines(res.stdout, "registry entry")
    assert len(entry_lines) == 2, res.stdout
    ghost_line = next(ln for ln in entry_lines if "orchard_detector_v2" in ln)
    scratch_line = next(ln for ln in entry_lines if "scratch_detector" in ln)
    assert "checkpoint missing" in ghost_line and "test/temp" not in ghost_line
    assert (
        "test/temp" in scratch_line
        and "checkpoint missing" not in scratch_line
    )


def test_a_checkpoint_under_a_temp_rooted_project_is_not_pollution(tmp_path):
    """A checkpoint under a temp-rooted project is not pollution.

    ``tmp_path`` itself sits under a temp tree (pytest's own fixture), and a checkpoint resolving
    inside the project is reported neither as pollution nor as missing.
    """
    root = _layout_project(tmp_path, "2026-03-04")
    from tests._verified_checkpoint_fixtures import produced_checkpoint

    ckpt_dir = root / ".tcip" / "models"
    ckpt_dir.mkdir(parents=True)
    produced_checkpoint(ckpt_dir / "m.pt", "weights")
    ModelRegistry(str(root)).register_model(
        name="m", checkpoint_path=str(ckpt_dir / "m.pt")
    )

    res = run_tcip("doctor", [str(root)])

    assert "test/temp" not in res.stdout
    assert "checkpoint missing" not in res.stdout


def test_image_census_counts_every_capture_the_loaders_admit(tmp_path):
    """The doctor's image census keys an .npz capture beside a .jpg one.

    Each capture is keyed to its own capture and stem.
    """
    from tcip_mcp.cli import doctor

    images = tmp_path / "images" / "2026-03-04"
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
    tmp_path,
):
    """One stem in two captures stays two images, one labeled and one unlabeled.

    With two captures each holding ``same.png``, only one labeled, one image has no label record
    and no label lacks its image, at the census and at the doctor alike.
    """
    from tcip_mcp.cli import doctor
    from tcip_mcp.tools.data_tools import scan_dataset

    root = _layout_project(tmp_path, "2026-03-04")
    image_dir(root, "2026-03-05").mkdir(parents=True)
    for date in ("2026-03-04", "2026-03-05"):
        Image.new("RGB", (32, 32)).save(image_dir(root, date) / "same.png")
    label_image(
        image_dir(root, "2026-03-04") / "same.png",
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))],
        32,
        32,
    )

    scan = scan_dataset(str(root))
    findings: list[tuple[str, str]] = []
    doctor.check_data_quality(
        root, findings, census=doctor._census(root, findings)
    )

    assert (scan["paired_images"], scan["unlabeled_images"]) == (1, 1), scan
    assert not [msg for _, msg in findings if "no matching image" in msg], (
        findings
    )
    (unlabeled,) = [msg for _, msg in findings if "no label record" in msg]
    assert unlabeled.startswith("1 of 2") and "2026-03-05/same" in unlabeled, (
        unlabeled
    )


def test_only_the_unreadable_trait_record_is_reported(tmp_path):
    """Only the unreadable trait record is reported, by name and reason.

    A record the schema refuses is reported by name and reason; a confirmed trait beside it
    stays silent.
    """
    import tcip_store as ts

    root = _layout_project(tmp_path, "2026-03-04")
    fx.propose_and_confirm(root, fx.entry("leaf", ("leaf_length",)))
    ts.replace(
        traits.trait_key(root, "burr_size"),
        _record_localizing(tmp_path, "burr_size", "not_a_localization"),
        expect=ts.Version.ABSENT,
    )

    res = run_tcip("doctor", [str(root)])
    assert res.returncode == 2, res.stdout
    trait_lines = _lines(res.stdout, "trait '")
    assert len(trait_lines) == 1, res.stdout
    assert "'burr_size' will not read" in trait_lines[0]
    assert "not_a_localization" in res.stdout


def _leaf_project(tmp_path: Path) -> Path:
    return new_project(_layout_project(tmp_path, "2026-03-04")).root


def test_doctor_is_silent_on_a_confirmed_latest_revision(tmp_path: Path):
    root = _leaf_project(tmp_path)
    fx.propose_and_confirm(root, fx.entry("leaf", ("leaf_length",)))

    res = run_tcip("doctor", [str(root)])

    assert "trait '" not in res.stdout, res.stdout


def test_doctor_warns_on_an_unconfirmed_latest_revision(tmp_path: Path):
    """The doctor warns, in one line and exit 1, on an unconfirmed latest trait revision.

    The warning holds for a trait whose latest revision the breeder has not confirmed, a
    confirmed earlier one notwithstanding.
    """
    root = _leaf_project(tmp_path)
    fx.propose_and_confirm(root, fx.entry("leaf", ("leaf_length",)))
    fx.propose(
        root, fx.entry("leaf", ("leaf_length",), notes="a second reading")
    )

    res = run_tcip("doctor", [str(root)])

    assert res.returncode == 1, res.stdout
    lines = _lines(res.stdout, "trait '")
    assert len(lines) == 1
    assert (
        "the latest revision (2) of trait 'leaf' is not confirmed" in lines[0]
    )
    assert "Setup tab" in lines[0]


def test_doctor_reports_an_undecodable_trait_record_without_aborting(
    tmp_path: Path,
):
    from tests._record_damage_fixtures import damage_record

    root = _leaf_project(tmp_path)
    fx.propose_and_confirm(root, fx.entry("leaf", ("leaf_length",)))
    damage_record(traits.trait_key(root, "leaf"), b"{not valid json")

    res = run_tcip("doctor", [str(root)])

    assert res.returncode == 2, res.stdout
    lines = _lines(res.stdout, "trait '")
    assert len(lines) == 1 and "will not read" in lines[0]


def test_doctor_errors_on_a_project_whose_record_does_not_decode(tmp_path):
    """A project record that does not decode is an error finding with exit 2."""
    from tcip_mcp.project_record import project_record_key
    from tests._record_damage_fixtures import damage_record

    root = new_project(_layout_project(tmp_path, "2026-03-04")).root
    key = project_record_key(str(root))
    # A genuinely undecodable byte string, written under the record's own key, so the finding
    # is the store's own decode error rather than "not a site record".
    damage_record(key, b"{not valid json")

    res = run_tcip("doctor", [str(root)])

    assert res.returncode == 2, res.stdout
    assert "does not decode" in res.stdout


def _unreadable_label(image: Path) -> None:
    """Write a label document for ``image`` and damage its stored bytes so they do not decode."""
    from tests._record_damage_fixtures import damage_record

    label_image(image, [], 32, 32, keep_empty=True)
    damage_record(image_label_key(image), b"not json {][")


def test_doctor_flags_an_unreadable_label(tmp_path):
    """The doctor flags an unreadable label as an error-level finding, not a pass.

    A corrupt label document is named as not reading, and the doctor exits 2.
    """
    date = "2026-03-04"
    root = _layout_project(tmp_path, date)
    image = image_dir(root, date) / "IMG_S.JPG"
    Image.new("RGB", (32, 32)).save(image)
    _unreadable_label(image)

    res = run_tcip("doctor", [str(root)])
    assert res.returncode == 2, res.stdout
    unreadable = _lines(res.stdout, "will not read")
    assert len(unreadable) >= 1, res.stdout
    assert any("IMG_S" in ln for ln in unreadable), res.stdout


def test_a_marked_empty_label_is_clean_to_the_data_quality_check(tmp_path):
    """A marked empty label is clean to the data-quality check.

    An empty label marked complete leaves the data-quality check over the census with no
    finding.
    """
    from tcip_mcp.cli import doctor

    date = "2026-03-04"
    root = _layout_project(tmp_path, date)
    image = image_dir(root, date) / "IMG_S.JPG"
    Image.new("RGB", (32, 32)).save(image)
    label_image(image, [], 32, 32, keep_empty=True)
    mark_complete(image, "bud", project=root)

    findings: list[tuple[str, str]] = []
    doctor.check_data_quality(
        root, findings, census=doctor._census(root, findings)
    )
    assert findings == []


def test_a_doctor_run_reads_each_label_once_and_reports_an_unreadable_one_once(
    tmp_path,
    monkeypatch,
):
    """A doctor run reads each label once and reports an unreadable one once.

    Over a census and its data-quality and provenance checks, each label document is read once
    and the unreadable one is reported once.
    """
    from tcip_mcp.cli import doctor

    date = "2026-03-04"
    root = _layout_project(tmp_path, date)
    Image.new("RGB", (32, 32)).save(image_dir(root, date) / "IMG_U.JPG")
    _unreadable_label(image_dir(root, date) / "IMG_U.JPG")
    Image.new("RGB", (32, 32)).save(image_dir(root, date) / "IMG_A.JPG")
    label_image(
        image_dir(root, date) / "IMG_A.JPG",
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))],
        32,
        32,
    )

    reads: list[str] = []
    real_read = json_io.read_label_document

    def _counting_read(key, *args, **kwargs):
        reads.append(key.parts[-1])
        return real_read(key, *args, **kwargs)

    monkeypatch.setattr(json_io, "read_label_document", _counting_read)
    findings: list[tuple[str, str]] = []
    census = doctor._census(root, findings)
    for check in (doctor.check_data_quality, doctor.check_provenance):
        check(root, findings, census=census)

    assert sorted(reads) == ["IMG_A", "IMG_U"]
    unreadable = [
        msg for _, msg in findings if "label document will not read" in msg
    ]
    assert len(unreadable) == 1, findings
    assert "IMG_U" in unreadable[0]
