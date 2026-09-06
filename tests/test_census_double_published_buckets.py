"""The double-publish census reports a bucket whose stamp names fewer stems than it holds
documents for, a bucket whose stamp decodes with no ``image_filenames`` map at all (unjudgeable
rather than clean), a validation row sealed over a mixed-run bucket, and nothing for a bucket
whose stamp names every document. Coverage of a read-only census; it changes no behaviour."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from PIL import Image

import tcip_store as ts
from tcip_annotation.json_io import write_annotations
from tcip_mcp.dataset_layout import prediction_dir
from tcip_mcp.experiments import create_experiment, validations_key
from tcip_mcp.pipelines.resolution import operating_point_stamp, write_sidecar
from tcip_mcp.prediction_buckets import bucket_content_digest
from tcip_mcp.tools.project_tools import initialize_project, register_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "census_double_published_buckets.py"
DATE = "2026-04-02"


def _load():
    spec = importlib.util.spec_from_file_location("census_double_published_buckets", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["census_double_published_buckets"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _project(tmp_path: Path, monkeypatch) -> Path:
    """A project registering its own tree as its dataset, the way the workspace projects do."""
    project = tmp_path / "project"
    images = project / "images" / DATE
    images.mkdir(parents=True)
    Image.new("RGB", (10, 10), (1, 2, 3)).save(images / "a.png")
    Image.new("RGB", (10, 10), (3, 2, 1)).save(images / "b.png")
    initialize_project(str(project), site="north orchard")
    monkeypatch.setenv("TCIP_STATE_ROOT", str(project))
    result = register_dataset(str(project), "chestnut", str(project))
    assert "error" not in result, result
    return project


def _stamp(bucket: Path, named: list[str]) -> None:
    stamp = operating_point_stamp(
        {"conf": {"value": 0.25}}, validated=False, validated_by=None,
        tile_size_validated=None, shippable_issues=[], id_map=None, subject="bud",
        attribute=None, trait=None, dataset_hash="H", checkpoint="m",
        checkpoint_sha256="sha-detector", experiment_id=None, images_dir=None,
        raster_path=None, produced_at="2026-04-02T00:00:00+00:00",
        image_filenames={stem: f"{stem}.png" for stem in named},
    )
    write_sidecar(bucket, stamp)


def _bucket(project: Path, model: str, stems: list[str]) -> Path:
    bucket = prediction_dir(project, model, DATE)
    for stem in stems:
        write_annotations(str(bucket / f"{stem}.json"), [], img_w=10, img_h=10, keep_empty=True)
    return bucket


def test_a_stamp_naming_fewer_stems_than_the_bucket_holds_is_a_double_publish(tmp_path, monkeypatch):
    project = _project(tmp_path, monkeypatch)
    mixed = _bucket(project, "baseline", ["a", "b"])
    _stamp(mixed, ["b"])
    clean = _bucket(project, "baseline@r2", ["a", "b"])
    _stamp(clean, ["a", "b"])

    census = _load().census_project(project)

    mixed_paths = [b.bucket.resolve() for b in census.mixed]
    assert mixed_paths == [mixed.resolve()]
    assert census.mixed[0].unnamed == {"a"}
    assert not any(b.named_without_document for b in census.buckets)
    assert census.claims == []
    assert census.read_errors == []
    lines = _load().render(census)
    assert any(line.startswith("  DOUBLE-PUBLISH") and "unnamed: a" in line for line in lines)
    assert not any("baseline@r2" in line for line in lines if "DOUBLE-PUBLISH" in line)


def test_a_validation_row_sealed_over_a_mixed_bucket_is_reported_with_its_digest_state(
    tmp_path, monkeypatch,
):
    """The row is appended through the storage seam in the shape seal_validation writes
    (experiments._VALIDATION_FIELDS): seal_validation is the one producer, and running the
    count gate over a synthetic bucket to earn a real row is not what this census checks."""
    project = _project(tmp_path, monkeypatch)
    mixed = _bucket(project, "baseline", ["a", "b"])
    _stamp(mixed, ["b"])
    created = create_experiment("exp-census", {"model_source": {"builder": "x:y"}})
    assert "error" not in created, created
    sealed = bucket_content_digest(mixed)
    ts.append(validations_key("exp-census", root=project), {
        "document": "operating_point",
        "trait": "bud_count",
        "claim": {"operating_point": {"conf": {"value": 0.25, "validated_against": "holdout"}}},
        "validated_against": "holdout",
        "checkpoint_sha256": "sha-detector",
        "producing_experiment_id": None,
        "reference_identity": {},
        "covered_buckets": {mixed.resolve().relative_to(project.resolve()).as_posix(): sealed},
        "dataset_root": str(project.resolve()),
        "recorded_at": "2026-04-02T00:00:00+00:00",
        "train_disjointness": None,
        "selection_disjointness": None,
    })

    census = _load().census_project(project)

    assert len(census.claims) == 1
    claim = census.claims[0]
    assert claim.experiment_id == "exp-census"
    assert claim.bucket == mixed.resolve()
    assert claim.content_matches is True

    write_annotations(str(mixed / "c.json"), [], img_w=10, img_h=10, keep_empty=True)
    again = _load().census_project(project)
    assert again.claims[0].content_matches is False
    lines = _load().render(again)
    assert any("MIXED-RUN-CLAIM" in line and "content changed" in line for line in lines)


def test_a_project_registering_an_external_dataset_still_notes_its_own_unwalked_tree(
    tmp_path, monkeypatch,
):
    """The project registers a dataset, just not its own tree: the note names what the gate
    actually tests (the project's own prediction buckets are outside every root the census
    walked), not that the project registers no dataset at all."""
    project = tmp_path / "project"
    project.mkdir()
    initialize_project(str(project), site="north orchard")
    monkeypatch.setenv("TCIP_STATE_ROOT", str(project))

    external = tmp_path / "external"
    images = external / "images" / DATE
    images.mkdir(parents=True)
    Image.new("RGB", (10, 10), (1, 2, 3)).save(images / "a.png")
    result = register_dataset(str(external), "chestnut", str(project))
    assert "error" not in result, result

    own_bucket = _bucket(project, "baseline", ["a"])
    _stamp(own_bucket, ["a"])

    census = _load().census_project(project)

    assert census.unregistered_tree is True
    lines = _load().render(census)
    note = next(line for line in lines if line.strip().startswith("NOTE"))
    assert str(project) in note
    assert "own predictions/ tree" in note
    assert "not among the roots" in note


def test_a_bucket_whose_stamp_names_every_document_is_not_a_finding(tmp_path, monkeypatch):
    project = _project(tmp_path, monkeypatch)
    clean = _bucket(project, "baseline", ["a", "b"])
    _stamp(clean, ["a", "b"])

    census = _load().census_project(project)

    assert census.mixed == []
    assert census.claims == []
    assert [b.bucket.resolve() for b in census.buckets] == [clean.resolve()]
    assert _load().main([str(project)]) == 0


def test_main_exits_one_on_a_finding_and_two_on_a_non_project(tmp_path, monkeypatch, capsys):
    project = _project(tmp_path, monkeypatch)
    mixed = _bucket(project, "baseline", ["a", "b"])
    _stamp(mixed, ["b"])
    mod = _load()

    assert mod.main([str(project)]) == 1
    out = capsys.readouterr().out
    assert "DOUBLE-PUBLISH" in out

    assert mod.main([str(tmp_path / "nowhere")]) == 2


def test_a_bucket_whose_stamp_records_no_image_filenames_map_is_unjudgeable(tmp_path, monkeypatch):
    """A stamp decoding with no ``image_filenames`` mapping at all (the shape a producer from
    before that extension key existed writes) cannot be checked against the bucket's documents;
    it must be reported UNJUDGEABLE rather than read as clean, the pre-refusal population this
    census exists to find."""
    project = _project(tmp_path, monkeypatch)
    bucket = _bucket(project, "baseline", ["a", "b"])
    stamp = operating_point_stamp(
        {"conf": {"value": 0.25}}, validated=False, validated_by=None,
        tile_size_validated=None, shippable_issues=[], id_map=None, subject="bud",
        attribute=None, trait=None, dataset_hash="H", checkpoint="m",
        checkpoint_sha256="sha-detector", experiment_id=None, images_dir=None,
        raster_path=None, produced_at="2026-04-02T00:00:00+00:00",
    )
    write_sidecar(bucket, stamp)

    census = _load().census_project(project)

    assert census.mixed == []
    assert len(census.buckets) == 1
    entry = census.buckets[0]
    assert entry.unjudgeable is True
    assert entry.named_stems is None
    lines = _load().render(census)
    assert any(line.startswith("  UNJUDGEABLE") and str(bucket) in line for line in lines)
    assert _load().main([str(project)]) == 1


def test_a_bucket_whose_stamp_records_no_image_filenames_map_and_holds_no_document_is_not_reported(
    tmp_path, monkeypatch,
):
    """Coverage: a stamp with no image_filenames map beside a bucket holding no document at all
    published nothing, so it is not the UNJUDGEABLE finding the sibling test above pins; no line
    is rendered for it and the exit stays clean."""
    project = _project(tmp_path, monkeypatch)
    bucket = _bucket(project, "baseline", [])
    stamp = operating_point_stamp(
        {"conf": {"value": 0.25}}, validated=False, validated_by=None,
        tile_size_validated=None, shippable_issues=[], id_map=None, subject="bud",
        attribute=None, trait=None, dataset_hash="H", checkpoint="m",
        checkpoint_sha256="sha-detector", experiment_id=None, images_dir=None,
        raster_path=None, produced_at="2026-04-02T00:00:00+00:00",
    )
    write_sidecar(bucket, stamp)

    census = _load().census_project(project)

    assert census.buckets == []
    lines = _load().render(census)
    assert not any("UNJUDGEABLE" in line for line in lines)
    assert _load().main([str(project)]) == 0


def test_a_stamp_that_will_not_decode_is_read_refused_and_the_census_continues_over_the_remaining_roots(
    tmp_path, monkeypatch, capsys,
):
    """Coverage of the continuation clause the module docstring states: a stamp whose bytes are
    corrupted in place is reported READ-REFUSED for its own root, and a second, clean root passed
    alongside it still gets its own line, rather than the walk stopping at the first root."""
    from tests._record_damage_fixtures import damage_record
    from tcip_mcp.pipelines.resolution import sidecar_key

    damaged = _project(tmp_path / "damaged", monkeypatch)
    bucket = _bucket(damaged, "baseline", ["a"])
    _stamp(bucket, ["a"])
    damage_record(sidecar_key(bucket, "operating_point"), b"{not json")

    clean = _project(tmp_path / "clean", monkeypatch)
    clean_bucket = _bucket(clean, "baseline", ["a"])
    _stamp(clean_bucket, ["a"])

    exit_code = _load().main([str(damaged), str(clean)])

    assert exit_code == 2
    out = capsys.readouterr().out
    refused_at = out.index("READ-REFUSED")
    clean_at = out.index(f"project {clean}")
    assert clean_at > refused_at
