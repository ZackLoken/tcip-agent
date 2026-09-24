"""An external dataset-level COCO document becomes the dataset's per-image label documents.

The imported images train through the ordinary producer with the targets the document stated, a
crowd region keeping its flag and a run-length mask arriving as rings; every fault found before
writing is reported together and refuses with nothing written, and the writes after that are
create-only one document at a time, the audit event naming what was written. The COCO documents
here are hand-written because they are the external input the door exists for; every image,
registry and band group they refer to comes from the platform's own producers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tcip_mcp.audit as audit_module
import tcip_store as ts
from tcip_annotation import json_io

DATE = "2025-09-14"
IMG = 64
SUBJECT = "bur"
STEMS = ("tree_01", "tree_02", "tree_03")
BOX = [8.0, 10.0, 20.0, 16.0]
RING = [30.0, 30.0, 50.0, 30.0, 50.0, 54.0, 30.0, 54.0]


def _dataset(tmp_path: Path, stems=STEMS) -> Path:
    """Images brought in through ``ingest_images`` under one capture bucket, and a registry
    declaring the subject, written through its own door."""
    from PIL import Image

    from tcip_mcp.tools.annotation_tools import write_subject_registry
    from tcip_mcp.tools.ingest_tools import ingest_images

    raw = tmp_path / "raw"
    raw.mkdir()
    for index, stem in enumerate(stems):
        Image.new("RGB", (IMG, IMG), color=(40 + index, 60, 50)).save(raw / f"{stem}.png")
    root = tmp_path / "chestnut_bur_count"
    ingested = ingest_images(source=str(raw), name=root.name, site="the import test's block",
                             project_path=str(root), date_from=DATE)
    assert "error" not in ingested, ingested
    registered = write_subject_registry(str(root), subjects={
        SUBJECT: {"description": "one chestnut bur"}, "leaf": {"description": "one leaf"}})
    assert "error" not in registered, registered
    return root


def _document(path: Path, *, images=None, annotations=None, categories=None) -> Path:
    """An external COCO document: two burs as a box, one as a polygon, one image unannotated."""
    doc = {
        "images": images if images is not None else [
            {"id": index + 1, "file_name": f"{stem}.png", "width": IMG, "height": IMG}
            for index, stem in enumerate(STEMS)],
        "annotations": annotations if annotations is not None else [
            {"id": 1, "image_id": 1, "category_id": 7, "bbox": BOX, "iscrowd": 0,
             "created_by": "user:breeder", "created_at": "2025-09-15T08:00:00+00:00"},
            {"id": 2, "image_id": 2, "category_id": 7, "segmentation": [RING],
             "bbox": [30.0, 30.0, 20.0, 24.0], "iscrowd": 0},
        ],
        "categories": categories if categories is not None else [{"id": 7, "name": SUBJECT}],
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _labels(root: Path) -> Path:
    return root / "annotations" / DATE


def _import(document: Path, root: Path) -> dict:
    from tcip_mcp.tools.ingest_tools import import_coco

    return import_coco(str(document), str(root), DATE)


def _loader(task: str, root: Path):
    """The loader the ordinary producer builds over the dataset's imported documents."""
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    data_cfg = {"images_dir": str(root / "images" / DATE), "labels_dir": str(_labels(root)),
                "subject": SUBJECT, "auto_val": False}
    loader, _, _ = auto_train_val(task, data_cfg, None)
    return loader


def _targets(task: str, root: Path) -> dict:
    """Each admitted image's target, read off the loader the ordinary producer builds."""
    loader = _loader(task, root)
    return {loader.member_stem_of(key): loader[i][1] for i, key in enumerate(loader.stems)}


def test_an_imported_documents_images_train_with_the_boxes_it_stated(tmp_path: Path):
    """A detection loader over the imported documents reads back the box and the polygon's
    extent the COCO document stated; the image it gave no annotation gets no document, so it is
    not admitted as a negative."""
    pytest.importorskip("torch")

    root = _dataset(tmp_path)
    result = _import(_document(tmp_path / "external.json"), root)
    assert "error" not in result, result
    assert sorted(Path(p).name for p in result["written"]) == ["tree_01.json", "tree_02.json"]
    assert not (_labels(root) / "tree_03.json").exists()

    targets = _targets("detection", root)

    x, y, w, h = BOX
    assert sorted(targets) == ["tree_01", "tree_02"]
    assert targets["tree_01"]["boxes"].tolist() == [[x, y, x + w, y + h]]
    assert targets["tree_02"]["boxes"].tolist() == [[30.0, 30.0, 50.0, 54.0]]


def test_an_imported_documents_images_train_with_the_polygons_it_stated(tmp_path: Path):
    """An instance_seg loader over the imported documents rasterizes exactly the polygon the COCO
    document stated for each image."""
    pytest.importorskip("torch")
    import numpy as np
    from PIL import Image, ImageDraw

    shifted = [c - 20.0 for c in RING]
    root = _dataset(tmp_path)
    annotations = [{"id": 1, "image_id": 1, "category_id": 7, "segmentation": [shifted]},
                   {"id": 2, "image_id": 2, "category_id": 7, "segmentation": [RING]}]
    result = _import(_document(tmp_path / "external.json", annotations=annotations), root)
    assert "error" not in result, result

    targets = _targets("instance_seg", root)

    assert sorted(targets) == ["tree_01", "tree_02"]
    for stem, ring in (("tree_01", shifted), ("tree_02", RING)):
        stated = Image.new("L", (IMG, IMG), 0)
        ImageDraw.Draw(stated).polygon(list(zip(ring[0::2], ring[1::2])), fill=1)
        xs, ys = ring[0::2], ring[1::2]
        assert targets[stem]["boxes"].tolist() == [[min(xs), min(ys), max(xs), max(ys)]]
        assert np.array_equal(targets[stem]["masks"][0].numpy(), np.array(stated))


def test_an_imported_record_keeps_the_provenance_the_document_carried_and_gains_none(
    tmp_path: Path,
):
    root = _dataset(tmp_path)
    assert "error" not in _import(_document(tmp_path / "external.json"), root)

    (authored,) = json_io.read_annotations(_labels(root) / "tree_01.json")
    (bare,) = json_io.read_annotations(_labels(root) / "tree_02.json")
    assert (authored.created_by, authored.created_at) == (
        "user:breeder", "2025-09-15T08:00:00+00:00")
    assert (bare.created_by, bare.created_at, bare.accepted_by) == (None, None, None)


def _rows(root: Path) -> list[dict]:
    return list(ts.read_log(audit_module.audit_log_key(root)).records)


def test_the_import_leaves_one_row_with_the_documents_path_and_digest(tmp_path: Path):
    """Every row the import leaves in the dataset's log is counted, and there is one: the
    library's import event. The digest is recomputed here from the document's bytes by the stated
    convention, ``sha256(bytes)[:16]``, never through the code under test."""
    import hashlib

    root = _dataset(tmp_path)
    document = _document(tmp_path / "external.json")
    before = len(_rows(root))

    assert "error" not in _import(document, root)

    rows = _rows(root)[before:]
    assert [(row["tool"], row["status"]) for row in rows] == [("coco_document_imported", "ok")]
    (event,) = rows
    assert event["arguments"]["document"] == str(document.resolve())
    assert sorted(Path(p).name for p in event["arguments"]["written"]) == [
        "tree_01.json", "tree_02.json"]
    assert event["document_digest"] == hashlib.sha256(document.read_bytes()).hexdigest()[:16]
    for label in _labels(root).iterdir():
        assert "external.json" not in label.read_text(encoding="utf-8")


def test_the_digest_names_the_bytes_the_labels_came_from(tmp_path: Path, monkeypatch):
    """The document is read once: replacing it while the import writes leaves the event's digest
    the one of the bytes the labels were decoded from, never of a second read."""
    import hashlib

    root = _dataset(tmp_path)
    document = _document(tmp_path / "external.json")
    read_bytes = document.read_bytes()
    real_put_blob = ts.put_blob

    def replacing(key, data, **kwargs):
        document.write_text(json.dumps({"images": [], "categories": []}), encoding="utf-8")
        return real_put_blob(key, data, **kwargs)

    monkeypatch.setattr(ts, "put_blob", replacing)
    assert "error" not in _import(document, root)

    (event,) = [row for row in _rows(root) if row["tool"] == "coco_document_imported"]
    assert event["document_digest"] == hashlib.sha256(read_bytes).hexdigest()[:16]
    assert document.read_bytes() != read_bytes


def test_every_declared_category_must_be_registered(tmp_path: Path):
    """A category the document declares but no record uses is still a declaration the registry
    must be able to name, and the import refuses on it by name, writing nothing."""
    root = _dataset(tmp_path)
    document = _document(tmp_path / "external.json",
                         categories=[{"id": 7, "name": SUBJECT}, {"id": 9, "name": "husk"}])

    result = _import(document, root)

    assert "error" in result
    assert "'husk'" in result["error"] and "registry" in result["error"]
    assert not _labels(root).exists() or not any(_labels(root).iterdir())


@pytest.mark.parametrize("name", ["", None, ["husk"]], ids=["empty", "null", "list"])
def test_a_category_name_no_subject_can_carry_is_refused_by_the_registry(tmp_path: Path, name):
    """A declared name is the subject its records take, so the registry answers for it: one no
    subject can carry is named in the import's refusal, whatever its shape."""
    root = _dataset(tmp_path)
    document = _document(tmp_path / "external.json",
                         categories=[{"id": 7, "name": SUBJECT}, {"id": 9, "name": name}])

    result = _import(document, root)

    assert f"category {name!r} is not a subject" in result.get("error", ""), result
    assert not _labels(root).exists() or not any(_labels(root).iterdir())


def test_an_image_not_under_the_dataset_refuses_the_whole_import(tmp_path: Path):
    root = _dataset(tmp_path)
    images = [{"id": 1, "file_name": "tree_01.png", "width": IMG, "height": IMG},
              {"id": 2, "file_name": "tree_99.png", "width": IMG, "height": IMG}]

    result = _import(_document(tmp_path / "external.json", images=images), root)

    assert "error" in result and "'tree_99.png'" in result["error"]
    assert not (_labels(root) / "tree_01.json").exists()


@pytest.mark.parametrize("file_name", ["tree_02.jpg", "north/tree_02.png"],
                         ids=["another_extension", "directory_prefix"])
def test_an_ordinary_image_is_named_by_its_own_file_name(tmp_path: Path, file_name: str):
    """Only a ``.bandgroup`` capture is tied by stem; an ordinary image the document names by
    another file name is not the dataset's image of that stem."""
    root = _dataset(tmp_path)
    images = [{"id": 1, "file_name": "tree_01.png", "width": IMG, "height": IMG},
              {"id": 2, "file_name": file_name, "width": IMG, "height": IMG}]

    result = _import(_document(tmp_path / "external.json", images=images), root)

    assert "error" in result and f"{file_name!r} is not in" in result["error"]
    assert not (_labels(root) / "tree_01.json").exists()


def test_an_existing_per_image_document_refuses_the_whole_import(tmp_path: Path):
    """A label a person may have edited is never replaced by an import."""
    from tcip_mcp.tools.annotation_tools import save_annotations

    root = _dataset(tmp_path)
    image = root / "images" / DATE / "tree_02.png"
    assert "error" not in save_annotations(
        str(image), annotations=[{"subject": "leaf", "bbox": [1, 1, 5, 5]}])
    before = (_labels(root) / "tree_02.json").read_bytes()

    result = _import(_document(tmp_path / "external.json"), root)

    assert "error" in result and "tree_02.json" in result["error"]
    assert (_labels(root) / "tree_02.json").read_bytes() == before
    assert not (_labels(root) / "tree_01.json").exists()


def test_an_unannotated_images_faults_still_refuse_the_import(tmp_path: Path):
    """Validation is over every listed image, not only the annotated ones: the unannotated third
    image's wrong frame and its existing document both refuse, by name."""
    from tcip_mcp.tools.annotation_tools import save_annotations

    root = _dataset(tmp_path)
    image = root / "images" / DATE / "tree_03.png"
    assert "error" not in save_annotations(
        str(image), annotations=[{"subject": "leaf", "bbox": [1, 1, 5, 5]}])
    images = [{"id": index + 1, "file_name": f"{stem}.png", "width": IMG, "height": IMG}
              for index, stem in enumerate(STEMS)]
    images[2]["width"] = IMG * 2

    result = _import(_document(tmp_path / "external.json", images=images), root)

    assert "error" in result
    assert "'tree_03.png' is stated as" in result["error"]
    assert "tree_03.json already exists" in result["error"]
    assert not (_labels(root) / "tree_01.json").exists()


def test_a_frame_the_image_does_not_have_refuses_the_whole_import(tmp_path: Path):
    """A document stating another size for an image drew its geometry in another pixel frame."""
    root = _dataset(tmp_path)
    images = [{"id": 1, "file_name": "tree_01.png", "width": IMG * 2, "height": IMG * 2},
              {"id": 2, "file_name": "tree_02.png", "width": IMG, "height": IMG}]

    result = _import(_document(tmp_path / "external.json", images=images), root)

    assert "error" in result and "'tree_01.png'" in result["error"]
    assert not (_labels(root) / "tree_02.json").exists()


def test_a_repeated_image_reports_every_fault_it_carries(tmp_path: Path):
    """A second record naming the same capture, stated at another frame, is reported as both
    faults: naming one does not stop the pass from asking the rest."""
    root = _dataset(tmp_path)
    images = [{"id": 1, "file_name": "tree_01.png", "width": IMG, "height": IMG},
              {"id": 2, "file_name": "tree_01.png", "width": IMG * 2, "height": IMG}]

    result = _import(_document(tmp_path / "external.json", images=images), root)

    assert "error" in result
    assert "the capture 'tree_01' another record already names" in result["error"]
    assert "is stated as" in result["error"]
    assert not (_labels(root) / "tree_01.json").exists()


def test_image_and_category_ids_listed_twice_refuse_by_name(tmp_path: Path):
    root = _dataset(tmp_path)
    images = [{"id": 1, "file_name": "tree_01.png", "width": IMG, "height": IMG},
              {"id": 1, "file_name": "tree_02.png", "width": IMG, "height": IMG}]
    annotations = [{"id": 1, "image_id": 1, "category_id": 7, "bbox": BOX}]

    result = _import(
        _document(tmp_path / "external.json", images=images, annotations=annotations), root)

    assert "error" in result and "image id 1 is listed twice" in result["error"]
    assert not _labels(root).exists() or not any(_labels(root).iterdir())

    categories = [{"id": 7, "name": SUBJECT}, {"id": 7, "name": "leaf"}]
    result = _import(_document(tmp_path / "external.json", categories=categories), root)

    assert "error" in result and "category id 7 is declared twice" in result["error"]
    assert not _labels(root).exists() or not any(_labels(root).iterdir())


def test_a_record_the_writer_refuses_writes_nothing(tmp_path: Path):
    """Every image is encoded by the writer's one encoder before the first write, so a box the
    stored grid collapses on the second image leaves the first image unwritten too."""
    root = _dataset(tmp_path)
    annotations = [{"id": 1, "image_id": 1, "category_id": 7, "bbox": BOX},
                   {"id": 2, "image_id": 2, "category_id": 7, "bbox": [10.0, 10.0, 0.004, 5.0]}]

    result = _import(_document(tmp_path / "external.json", annotations=annotations), root)

    assert "error" in result and "'tree_02.png'" in result["error"]
    assert not (_labels(root) / "tree_01.json").exists()


def test_a_label_written_after_validation_is_never_overwritten(tmp_path: Path, monkeypatch):
    """The import's writes are create-only, one document at a time: a person's label landing on
    the second document between the validation pass and its write raises instead of being
    replaced, the first document stays written, and the one audit event names both."""
    from tcip_mcp.pipelines.data.coco_import import import_coco_document

    root = _dataset(tmp_path)
    document = _document(tmp_path / "external.json")
    person = [json_io.annotation_from_payload(
        {"subject": "leaf", "bbox": [1, 1, 5, 5]}, author="user:breeder", now="2025-09-16")]
    second = _labels(root) / "tree_02.json"
    real_put_blob = ts.put_blob

    def interleaved(key, data, **kwargs):
        if (_labels(root) / "tree_01.json").exists() and not second.exists():
            real_put_blob(*json_io.encode_annotations(second, person, IMG, IMG))
        return real_put_blob(key, data, **kwargs)

    monkeypatch.setattr(ts, "put_blob", interleaved)
    with pytest.raises(ts.VersionConflict):
        import_coco_document(document, root, date=DATE)

    (kept,) = json_io.read_annotations(second)
    assert (kept.subject, kept.created_by) == ("leaf", "user:breeder")
    (imported,) = json_io.read_annotations(_labels(root) / "tree_01.json")
    assert imported.subject == SUBJECT
    (event,) = [row for row in _rows(root) if row["tool"] == "coco_document_imported"]
    assert event["status"] == "failed"
    assert [Path(p).name for p in event["arguments"]["written"]] == ["tree_01.json"]
    assert str(second) in event["arguments"]["error"]
    assert "changed since it was read" in event["arguments"]["error"]


def test_a_partial_import_and_a_partial_publish_record_one_key_set(tmp_path: Path, monkeypatch):
    """The two producers of a partial-write record, the import and the bucket publisher, each
    leave their failed line with the documents written and the error, and nothing else beyond
    the one fact naming their own act (the document imported, the bucket published)."""
    pytest.importorskip("torch")
    import tcip_mcp.tools.inference_tools as itools
    from tcip_mcp.dataset_layout import prediction_dir
    from tcip_mcp.pipelines.data.coco_import import import_coco_document
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    root = _dataset(tmp_path)
    second = _labels(root) / "tree_02.json"
    real_put_blob = ts.put_blob

    def interleaved(key, data, **kwargs):
        if (_labels(root) / "tree_01.json").exists() and not second.exists():
            real_put_blob(*json_io.encode_annotations(second, [], IMG, IMG, keep_empty=True))
        return real_put_blob(key, data, **kwargs)

    monkeypatch.setattr(ts, "put_blob", interleaved)
    with pytest.raises(ts.VersionConflict):
        import_coco_document(_document(tmp_path / "external.json"), root, date=DATE)
    monkeypatch.setattr(ts, "put_blob", real_put_blob)

    class Detector:
        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, **kw):
            return [{"image": p, "width": IMG, "height": IMG, "boxes": [BOX], "scores": [0.9],
                     "labels": [1], "count": 1} for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", Detector)
    real_write = itools.write_predictions_json
    calls: list[str] = []

    def failing_second_write(json_path, result, **kwargs):
        calls.append(str(json_path))
        if len(calls) == 2:
            raise OSError("disk full")
        return real_write(json_path, result, **kwargs)

    monkeypatch.setattr(itools, "write_predictions_json", failing_second_write)
    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path)
    with pytest.raises(OSError):
        itools.run_inference(ckpt, str(root / "images" / DATE),
                             output_dir=str(prediction_dir(root, "detector", DATE)), tile=False)

    failed = {row["tool"]: row for row in _rows(root) if row["status"] == "failed"}
    imported = failed["coco_document_imported"]["arguments"]
    published = failed["prediction_bucket_published"]["arguments"]
    assert set(imported) - {"document", "date"} == set(published) - {"predictions_dir"} == {
        "written", "error"}
    for arguments in (imported, published):
        assert [Path(p).name for p in arguments["written"]] == ["tree_01.json"]
        assert isinstance(arguments["error"], str) and arguments["error"]


def test_an_import_that_committed_no_document_leaves_no_event(tmp_path: Path, monkeypatch):
    """An act that committed nothing leaves no line: a valid document whose images carry no
    annotation writes nothing, and one whose first write fails has written nothing either."""
    from tcip_mcp.pipelines.data.coco_import import import_coco_document

    root = _dataset(tmp_path)
    assert _import(_document(tmp_path / "empty.json", annotations=[]), root)["written"] == []
    assert not [row for row in _rows(root) if row["tool"] == "coco_document_imported"]

    person = [json_io.annotation_from_payload(
        {"subject": "leaf", "bbox": [1, 1, 5, 5]}, author="user:breeder", now="2025-09-16")]
    first = _labels(root) / "tree_01.json"
    real_put_blob = ts.put_blob

    def claimed_first(key, data, **kwargs):
        if not first.exists():
            real_put_blob(*json_io.encode_annotations(first, person, IMG, IMG))
        return real_put_blob(key, data, **kwargs)

    monkeypatch.setattr(ts, "put_blob", claimed_first)
    with pytest.raises(ts.VersionConflict):
        import_coco_document(_document(tmp_path / "external.json"), root, date=DATE)
    assert not (_labels(root) / "tree_02.json").exists()
    assert not [row for row in _rows(root) if row["tool"] == "coco_document_imported"]


def test_an_annotation_naming_no_listed_image_refuses_the_whole_import(tmp_path: Path):
    root = _dataset(tmp_path)
    annotations = [{"id": 1, "image_id": 1, "category_id": 7, "bbox": BOX},
                   {"id": 2, "image_id": 42, "category_id": 7, "bbox": BOX}]

    result = _import(_document(tmp_path / "external.json", annotations=annotations), root)

    assert "error" in result and "42" in result["error"]
    assert not (_labels(root) / "tree_01.json").exists()


@pytest.mark.parametrize("date", ["../elsewhere", "2025-09-15"], ids=["not_a_name", "no_bucket"])
def test_a_date_that_names_no_capture_refuses(tmp_path: Path, date: str):
    """A date names the capture the images sit in; a name that is no bucket of the dataset, well
    formed or not, has nothing to import into."""
    from tcip_mcp.tools.ingest_tools import import_coco

    root = _dataset(tmp_path)
    result = import_coco(str(_document(tmp_path / "external.json")), str(root), date)
    assert "error" in result and "names no capture" in result["error"]


def test_a_document_that_lists_no_image_refuses(tmp_path: Path):
    root = _dataset(tmp_path)
    result = _import(_document(tmp_path / "external.json", images=[], annotations=[]), root)
    assert "error" in result and "nothing to import" in result["error"]


@pytest.mark.parametrize("annotations", [None, [None]], ids=["null", "null_record"])
def test_a_malformed_annotations_container_refuses_by_name_through_the_tool(
    tmp_path: Path, annotations,
):
    root = _dataset(tmp_path)
    document = tmp_path / "external.json"
    document.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "tree_01.png"}],
        "categories": [{"id": 7, "name": SUBJECT}], "annotations": annotations,
    }), encoding="utf-8")

    result = _import(document, root)

    assert "error" in result and "external.json" in result["error"]


def test_a_band_grouped_capture_resolves_by_the_stem_the_document_names(tmp_path: Path):
    """An external document names a grouped capture by one of its own file names; the import ties
    it by stem to the ``.bandgroup`` manifest, and the written document is the capture's own."""
    import numpy as np
    import tifffile

    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    root = _dataset(tmp_path)
    images_dir = root / "images" / DATE
    band_g, band_r = images_dir / "plot7_G.tif", images_dir / "plot7_R.tif"
    tifffile.imwrite(str(band_g), np.full((IMG, IMG), 111, dtype=np.uint16))
    tifffile.imwrite(str(band_r), np.full((IMG, IMG), 222, dtype=np.uint16))
    write_band_group_manifest(images_dir, "plot7", {"Green": band_g, "Red": band_r})
    images = [{"id": 1, "file_name": "plot7.tif", "width": IMG, "height": IMG}]
    annotations = [{"id": 1, "image_id": 1, "category_id": 7, "bbox": BOX}]

    result = _import(
        _document(tmp_path / "external.json", images=images, annotations=annotations), root)

    assert "error" not in result, result
    (written,) = result["written"]
    assert Path(written) == _labels(root) / "plot7.json"
    (ann,) = json_io.read_annotations(written)
    assert ann.subject == SUBJECT


CROWD = [36.0, 34.0, 20.0, 18.0]


def _crowd_document(tmp_path: Path) -> Path:
    """One bur as a box and one crowd region of burs on the first image."""
    return _document(tmp_path / "external.json", annotations=[
        {"id": 1, "image_id": 1, "category_id": 7, "bbox": BOX, "iscrowd": 0},
        {"id": 2, "image_id": 1, "category_id": 7, "bbox": CROWD, "iscrowd": 1},
    ])


def test_a_crowd_region_imports_and_reaches_training_and_evaluation_as_one(tmp_path: Path):
    """The crowd flag the document stated is written, read back, carried on the detection
    loader's target under ``iscrowd``, and carried onto the evaluation ground truth both builders
    form; the built-in heads are handed the target with the crowd row removed."""
    torch = pytest.importorskip("torch")
    from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset, instance_targets
    from tcip_mcp.pipelines.training.evaluation import (
        records_from_annotation, records_from_detector,
    )

    root = _dataset(tmp_path)
    assert "error" not in _import(_crowd_document(tmp_path), root)

    stored = json.loads((_labels(root) / "tree_01.json").read_text(encoding="utf-8"))
    assert [r.get("iscrowd") for r in stored["annotations"]] == [None, True]
    gt = json_io.read_annotations(_labels(root) / "tree_01.json")
    assert [a.iscrowd for a in gt] == [False, True]

    target = _targets("detection", root)["tree_01"]
    assert target["iscrowd"].tolist() == [0, 1]
    tiled = TiledDetectionDataset(_loader("detection", root), tile_size=IMG, overlap=0.0)
    assert tiled.num_samples == 1  # tree_01, whole, in its one tile
    assert tiled[0][1]["iscrowd"].tolist() == [0, 1]
    assert tiled.class_distribution == {0: 1}  # the one object; the crowd region is none
    (heads,) = instance_targets([target])
    x, y, w, h = BOX
    assert heads["boxes"].tolist() == [[x, y, x + w, y + h]] and heads["iscrowd"].tolist() == [0]

    _, from_annotations = records_from_annotation(gt, [], width=IMG, height=IMG)
    no_detections = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64),
                     "scores": torch.zeros((0,))}
    from_detector = records_from_detector(target, no_detections, width=IMG, height=IMG)
    assert [g["iscrowd"] for g in from_annotations["gt"]] == [0, 1]
    assert [g["iscrowd"] for g in from_detector["gt"]] == [0, 1]


def _uncompressed_counts(mask) -> list[int]:
    """COCO's uncompressed run lengths of ``mask``: column-major, starting with a run of zeros."""
    flat = mask.flatten(order="F").tolist()
    counts, value, run = [], 0, 0
    for pixel in flat:
        if pixel == value:
            run += 1
        else:
            counts.append(run)
            value, run = pixel, 1
    return counts + [run]


@pytest.mark.parametrize("compressed", [True, False], ids=["compressed", "uncompressed"])
def test_a_run_length_mask_imports_as_the_rings_its_mask_yields(tmp_path: Path, compressed):
    import numpy as np
    from pycocotools import mask as mask_utils

    from tcip_annotation.mask_contours import mask_to_polygon_rings

    mask = np.zeros((IMG, IMG), dtype=np.uint8)
    mask[10:30, 12:40] = 1
    mask[40:52, 44:60] = 1
    counts: object = (mask_utils.encode(np.asfortranarray(mask))["counts"].decode("ascii")
                      if compressed else _uncompressed_counts(mask))
    root = _dataset(tmp_path)
    annotations = [{"id": 1, "image_id": 1, "category_id": 7, "iscrowd": 0,
                    "segmentation": {"size": [IMG, IMG], "counts": counts}}]

    result = _import(_document(tmp_path / "external.json", annotations=annotations), root)

    assert "error" not in result, result
    (ann,) = json_io.read_annotations(_labels(root) / "tree_01.json")
    assert ann.geometry.rings == mask_to_polygon_rings(mask)
    assert len(ann.geometry.rings) == 2


def test_a_run_length_mask_that_yields_no_ring_refuses_by_record(tmp_path: Path):
    root = _dataset(tmp_path)
    empty = [IMG * IMG]  # one run of background: nothing is foreground
    annotations = [{"id": 1, "image_id": 1, "category_id": 7, "bbox": BOX},
                   {"id": 2, "image_id": 2, "category_id": 7,
                    "segmentation": {"size": [IMG, IMG], "counts": empty}}]

    result = _import(_document(tmp_path / "external.json", annotations=annotations), root)

    assert "error" in result
    assert "record 1 a polygon needs at least one ring" in result["error"]
    assert not (_labels(root) / "tree_01.json").exists()


def test_every_fault_the_document_carries_is_reported_together(tmp_path: Path):
    """A duplicate category, an unmapped category, a wrong frame and a missing image are each
    independent of the others, and one refusal names all four."""
    root = _dataset(tmp_path)
    images = [{"id": 1, "file_name": "tree_01.png", "width": IMG * 2, "height": IMG},
              {"id": 2, "file_name": "tree_99.png", "width": IMG, "height": IMG}]
    annotations = [{"id": 1, "image_id": 1, "category_id": 7, "bbox": BOX},
                   {"id": 2, "image_id": 1, "category_id": 8, "bbox": BOX}]
    categories = [{"id": 7, "name": SUBJECT}, {"id": 7, "name": "leaf"}]

    result = _import(_document(tmp_path / "external.json", images=images,
                               annotations=annotations, categories=categories), root)

    assert "error" in result
    for fault in ("category id 7 is declared twice", "record 1 names category_id 8",
                  "'tree_01.png' is stated as", "'tree_99.png' is not in"):
        assert fault in result["error"], (fault, result["error"])


def test_a_missing_image_whose_label_path_exists_reports_both(tmp_path: Path):
    from tcip_mcp.tools.annotation_tools import save_annotations

    root = _dataset(tmp_path)
    assert "error" not in save_annotations(
        str(root / "images" / DATE / "tree_02.png"),
        annotations=[{"subject": "leaf", "bbox": [1, 1, 5, 5]}])
    images = [{"id": 1, "file_name": "tree_01.png", "width": IMG, "height": IMG},
              {"id": 2, "file_name": "tree_02.jpg", "width": IMG, "height": IMG}]

    result = _import(_document(tmp_path / "external.json", images=images), root)

    assert "error" in result
    assert "'tree_02.jpg' is not in" in result["error"]
    assert "tree_02.json already exists" in result["error"]


@pytest.mark.parametrize("change, fault", [
    ({"categories": [{"id": 7, "name": SUBJECT}, {"id": 7.5, "name": None}]}, "category 1 "),
    ({"images": [{"file_name": "tree_01.png"}]}, "image record 0 names id None"),
    ({"annotations": [{"id": 1, "category_id": 7, "bbox": BOX}]}, "record 0 names image_id None"),
], ids=["float_null_category", "no_image_id", "no_annotation_image_id"])
def test_a_malformed_identity_refuses_by_index(tmp_path: Path, change, fault):
    root = _dataset(tmp_path)
    document = _document(tmp_path / "external.json", **change)

    result = _import(document, root)

    assert "error" in result and fault in result["error"], result
    assert not _labels(root).exists() or not any(_labels(root).iterdir())


@pytest.mark.parametrize("image_id", [[], {}], ids=["list", "object"])
def test_an_image_id_that_is_no_identity_is_a_fault_never_a_lookup_key(tmp_path: Path, image_id):
    # An id found invalid is reported and never used to look the image's records up, so an
    # unhashable id is one more fault in the tool's structured error rather than a TypeError.
    root = _dataset(tmp_path)
    images = [{"id": image_id, "file_name": "tree_01.png", "width": IMG, "height": IMG},
              {"id": 2, "file_name": "tree_02.png", "width": IMG, "height": IMG}]

    result = _import(_document(tmp_path / "external.json", images=images), root)

    assert f"image record 0 names id {image_id!r}, not an integer image id" in result["error"]
    assert not _labels(root).exists() or not any(_labels(root).iterdir())


def test_a_records_identity_and_content_faults_are_both_reported(tmp_path: Path):
    root = _dataset(tmp_path)
    annotations = [{"id": 1, "image_id": 1, "category_id": 7, "bbox": BOX},
                   {"id": 2, "image_id": None, "category_id": 7, "bbox": [1.0, 2.0, 3.0]}]

    result = _import(_document(tmp_path / "external.json", annotations=annotations), root)

    assert "record 1 names image_id None" in result["error"], result
    assert "record 1 bbox [1.0, 2.0, 3.0] is not a list of 4 numbers" in result["error"], result
