"""tcip doctor: the data-state doctor catches the field-session bug family."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp import traits
from tcip_mcp.subject_registry import SubjectRegistry, Subject
from tests._producer_fixtures import image_label_key, label_image, mark_complete, registry_over
from tcip_mcp.dataset_layout import image_dir
from tcip_mcp.model_registry import ModelRegistry
from tests import _trait_fixtures as fx
from tests._cli_fixtures import run_tcip
from tests._web_fixtures import new_project


def _record_localizing(tmp_path: Path, name: str, localization: str) -> dict:
    """A trait record as the proposing producer writes it, its entry's ``localization`` then
    replaced, so a record the schema refuses carries every other field a real one does."""
    revision = fx.propose(tmp_path / "scratch_project", fx.entry(name, ("leaf_length",)))
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
    label_image(images / "IMG_C.JPG", [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))],
                32, 32)
    mark_complete(images / "IMG_A.JPG", "bud", project=root)
    return root


def test_doctor_help_prints_the_dispatchers_prog_argument(capsys):
    """The dispatcher passes ``prog=f"tcip {command}"`` into ``doctor.main``'s own keyword
    argument, which threads straight to its ``ArgumentParser``; this calls the module directly
    with that argument, so the usage line proves the parser's own ``prog``, never a global
    argv mutation."""
    from tcip_mcp.cli import doctor

    with pytest.raises(SystemExit) as exc:
        doctor.main(["--help"], prog="tcip doctor")

    assert exc.value.code == 0
    assert "usage: tcip doctor " in capsys.readouterr().out


def _register_absent_checkpoint(root: Path, name: str, checkpoint_path: Path) -> None:
    """A registry entry, written by the registry's own producer, naming a checkpoint file that
    no longer exists: registered while it did, then deleted."""
    from tests._verified_checkpoint_fixtures import produced_checkpoint

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    produced_checkpoint(checkpoint_path, f"{name} weights")
    ModelRegistry(str(root)).register_model(name, str(checkpoint_path))
    checkpoint_path.unlink()


def test_doctor_flags_registry_checkpoint_path_under_a_temp_directory(tmp_path):
    """``tmp_path`` sits under pytest's own temp tree, so a checkpoint there outside the project
    is test pollution."""
    root = _project(tmp_path)
    _register_absent_checkpoint(root, "junk", tmp_path / "elsewhere" / "model.pt")

    res = run_tcip("doctor", [str(root)])
    assert res.returncode == 2  # errors present
    out = res.stdout
    assert "IMG_B" in out and "not marked complete" in out   # unconfirmed empty -> error
    assert "junk" in out and "test/temp" in out              # registry pollution -> error
    assert "IMG_A" not in out                                # confirmed negative is clean
    assert "IMG_C" not in out                                # an annotated image is clean


def test_doctor_reports_a_stem_collision_and_completes(tmp_path):
    """A capture already holding two identities for one stem refuses at every reader; the
    doctor names it as a finding instead of crashing on the exception."""
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
    """A trait record the schema refuses reads identically to no trait at all from a listing
    alone; ``tcip doctor`` is where the agent catches the difference at session start."""
    import tcip_store as ts

    root = _project(tmp_path)
    record = _record_localizing(tmp_path, "unicorn", "unicorn_match")
    ts.replace(traits.trait_key(root, "unicorn"), record, expect=ts.Version.ABSENT)

    res = run_tcip("doctor", [str(root)])
    assert res.returncode == 2  # errors present
    assert "'unicorn' will not read" in res.stdout
    assert "unicorn_match" in res.stdout


def _clean_project(tmp_path: Path) -> Path:
    """One image under capture ``d`` whose label document holds one breeder-drawn ``bud``."""
    root = tmp_path / "clean"
    images = root / "images" / "d"
    images.mkdir(parents=True)
    (root / ".tcip" / "state").mkdir(parents=True)
    registry_over(root, SubjectRegistry(subjects=(Subject(name="bud"),)))
    Image.new("RGB", (32, 32)).save(images / "IMG_A.JPG")
    label_image(images / "IMG_A.JPG",
                [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9), created_by="user:breeder")],
                32, 32)
    return root


def test_doctor_clean_project_exits_zero(tmp_path):
    root = _clean_project(tmp_path)
    new_project(root)
    res = run_tcip("doctor", [str(root)])
    assert res.returncode == 0, res.stdout


def _layout_project(tmp_path: Path, date: str | None, name: str = "resolved") -> Path:
    """A project whose image tree for capture ``date`` is placed by the layout resolver."""
    root = tmp_path / name
    image_dir(root, date).mkdir(parents=True)
    (root / ".tcip" / "state").mkdir(parents=True)
    registry_over(root,
                   SubjectRegistry(subjects=(Subject(name="bud"), Subject(name="leaf"))))
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


def test_a_mark_for_any_subject_finishes_an_empty_label_on_a_dateless_dataset(tmp_path):
    """An empty label of the undated capture marked complete for a subject is a confirmed
    negative there too."""
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


def test_registry_findings_are_read_through_the_registrys_own_entry_shape(tmp_path):
    """Entries written by ModelRegistry are the shape ``tcip doctor`` reports on, so an entry
    whose checkpoint is gone is named with its own name and path rather than read as nothing."""
    root = _layout_project(tmp_path, "2026-03-04")
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    from tests._verified_checkpoint_fixtures import produced_checkpoint

    registry = ModelRegistry(str(root))
    paths = {}
    for name, payload in (("currant_bud_detector_v1", "weights"),
                          ("chestnut_burr_counter_v3", "other weights")):
        ckpt = produced_checkpoint(ckpt_dir / f"{name}.pt", payload)
        registry.register_model(name=name, checkpoint_path=str(ckpt), metrics={})
        paths[name] = ckpt
    for ckpt in paths.values():
        ckpt.unlink()

    res = run_tcip("doctor", [str(root)])
    assert res.returncode == 2, res.stdout
    entry_lines = _lines(res.stdout, "registry entry")
    assert len(entry_lines) == 2, res.stdout
    for name, ckpt in paths.items():
        assert any(name in ln and str(ckpt) in ln for ln in entry_lines), res.stdout


def test_a_missing_checkpoint_and_a_test_checkpoint_are_distinct_registry_findings(tmp_path):
    """Pollution and absence are different data-state problems: an entry pointing at a
    throwaway test checkpoint is not reported as merely missing, and an entry whose checkpoint
    was never written is not reported as pollution."""
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
    assert "test/temp" in scratch_line and "checkpoint missing" not in scratch_line


def test_a_checkpoint_under_a_temp_rooted_project_is_not_pollution(tmp_path):
    """``tmp_path`` itself sits under a temp tree (pytest's own fixture), so a checkpoint
    resolving inside the project is never pollution merely because the project's own location
    carries a temp-tree marker: only a checkpoint the root does not contain is scanned."""
    root = _layout_project(tmp_path, "2026-03-04")
    from tests._verified_checkpoint_fixtures import produced_checkpoint

    ckpt_dir = root / ".tcip" / "models"
    ckpt_dir.mkdir(parents=True)
    produced_checkpoint(ckpt_dir / "m.pt", "weights")
    ModelRegistry(str(root)).register_model(
        name="m", checkpoint_path=str(ckpt_dir / "m.pt"))

    res = run_tcip("doctor", [str(root)])

    assert "test/temp" not in res.stdout
    assert "checkpoint missing" not in res.stdout


def test_image_census_counts_every_capture_the_loaders_admit(tmp_path):
    """The doctor's image census reads the platform's own extension set, so an .npz capture is
    keyed to its own label document rather than reported missing."""
    from tcip_mcp.cli import doctor

    images = tmp_path / "images" / "2026-03-04"
    images.mkdir(parents=True)
    (images / "plotA_0_0.npz").write_bytes(b"\x00")
    (images / "plotA_0_1.jpg").write_bytes(b"\xff\xd8")

    census = doctor._census(tmp_path, [])
    assert census is not None
    assert sorted(key.parts for key in census["images"].values()) == [
        ("2026-03-04", "plotA_0_0"), ("2026-03-04", "plotA_0_1")]


def test_one_stem_in_two_captures_is_two_images_each_paired_with_its_own_label(tmp_path):
    """Two captures each holding ``same.png``, only one labeled: one image has no label record and
    no label lacks its image, at the census and at the doctor alike."""
    from tcip_mcp.cli import doctor
    from tcip_mcp.tools.data_tools import scan_dataset

    root = _layout_project(tmp_path, "2026-03-04")
    image_dir(root, "2026-03-05").mkdir(parents=True)
    for date in ("2026-03-04", "2026-03-05"):
        Image.new("RGB", (32, 32)).save(image_dir(root, date) / "same.png")
    label_image(image_dir(root, "2026-03-04") / "same.png",
                [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 32, 32)

    scan = scan_dataset(str(root))
    findings: list[tuple[str, str]] = []
    doctor.check_data_quality(root, findings, census=doctor._census(root, findings))

    assert (scan["paired_images"], scan["unlabeled_images"]) == (1, 1), scan
    assert not [msg for _, msg in findings if "no matching image" in msg], findings
    (unlabeled,) = [msg for _, msg in findings if "no label record" in msg]
    assert unlabeled.startswith("1 of 2") and "2026-03-05/same" in unlabeled, unlabeled


def test_only_the_unreadable_trait_record_is_reported(tmp_path):
    """A record the schema refuses is reported by name and reason; a confirmed trait beside it
    stays silent."""
    import tcip_store as ts

    root = _layout_project(tmp_path, "2026-03-04")
    fx.propose_and_confirm(root, fx.entry("leaf", ("leaf_length",)))
    ts.replace(traits.trait_key(root, "burr_size"),
               _record_localizing(tmp_path, "burr_size", "not_a_localization"),
               expect=ts.Version.ABSENT)

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
    """One line, and exit 1, for a trait whose latest revision the breeder has not confirmed,
    a confirmed earlier one notwithstanding."""
    root = _leaf_project(tmp_path)
    fx.propose_and_confirm(root, fx.entry("leaf", ("leaf_length",)))
    fx.propose(root, fx.entry("leaf", ("leaf_length",), notes="a second reading"))

    res = run_tcip("doctor", [str(root)])

    assert res.returncode == 1, res.stdout
    lines = _lines(res.stdout, "trait '")
    assert len(lines) == 1
    assert "the latest revision (2) of trait 'leaf' is not confirmed" in lines[0]
    assert "Setup tab" in lines[0]


def test_doctor_reports_an_undecodable_trait_record_without_aborting(tmp_path: Path):
    from tests._record_damage_fixtures import damage_record

    root = _leaf_project(tmp_path)
    fx.propose_and_confirm(root, fx.entry("leaf", ("leaf_length",)))
    damage_record(traits.trait_key(root, "leaf"), b"{not valid json")

    res = run_tcip("doctor", [str(root)])

    assert res.returncode == 2, res.stdout
    lines = _lines(res.stdout, "trait '")
    assert len(lines) == 1 and "will not read" in lines[0]


def test_doctor_errors_on_a_project_whose_record_does_not_decode(tmp_path):
    """A damaged record is a check that could not run, not a clean project: an error, and exit 2."""
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
    """A label document for ``image`` whose stored bytes no longer decode."""
    from tests._record_damage_fixtures import damage_record

    label_image(image, [], 32, 32, keep_empty=True)
    damage_record(image_label_key(image), b"not json {][")


def test_doctor_flags_an_unreadable_label(tmp_path):
    """A corrupt label document is an error-level finding, never a pass: the reader raises on it,
    and the doctor reports it rather than letting the corruption pass as an empty document."""
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
    """A mark written through the save door is the one fact the data-quality check reads off
    the census, on one root under the same backend."""
    from tcip_mcp.cli import doctor

    date = "2026-03-04"
    root = _layout_project(tmp_path, date)
    image = image_dir(root, date) / "IMG_S.JPG"
    Image.new("RGB", (32, 32)).save(image)
    label_image(image, [], 32, 32, keep_empty=True)
    mark_complete(image, "bud", project=root)

    findings: list[tuple[str, str]] = []
    doctor.check_data_quality(root, findings, census=doctor._census(root, findings))
    assert findings == []


def test_a_doctor_run_reads_each_label_once_and_reports_an_unreadable_one_once(
    tmp_path, monkeypatch,
):
    """The census reads each label once and only the data-quality check reports one that will not
    read: the provenance check reads the census's result, never the file."""
    from tcip_mcp.cli import doctor

    date = "2026-03-04"
    root = _layout_project(tmp_path, date)
    Image.new("RGB", (32, 32)).save(image_dir(root, date) / "IMG_U.JPG")
    _unreadable_label(image_dir(root, date) / "IMG_U.JPG")
    Image.new("RGB", (32, 32)).save(image_dir(root, date) / "IMG_A.JPG")
    label_image(image_dir(root, date) / "IMG_A.JPG",
                [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 32, 32)

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
    unreadable = [msg for _, msg in findings if "label document will not read" in msg]
    assert len(unreadable) == 1, findings
    assert "IMG_U" in unreadable[0]
