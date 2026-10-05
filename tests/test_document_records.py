"""Label and prediction documents as records of their dataset root: each door that writes them
commits whole or not at all, training reads them by key, and a person reading the dump reads the
records the platform reads."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import tcip_store as ts
from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_annotation.verdicts import read_verdicts
from tcip_mcp.audit import audit_log_key
from tcip_mcp.dataset_layout import (
    Gestures, image_dir, prediction_key, save_label_document, verdict_key_of,
)
from tcip_mcp.subject_registry import Subject, SubjectRegistry
from tests._producer_fixtures import image_label_key, label_image, registry_over

DATE = "2026-03-02"
SIZE = 64
STEMS = ("a", "b", "c")
BUD = "bud"


def _capture(root: Path) -> list[Path]:
    """Three frames of capture :data:`DATE` under ``root``'s image tree, its registry declaring
    :data:`BUD`."""
    registry_over(root, SubjectRegistry(subjects=(Subject(name=BUD, description="a bud"),)))
    images = []
    for i, stem in enumerate(STEMS):
        path = image_dir(root, DATE) / f"{stem}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (SIZE, SIZE), (40 + 30 * i, 60, 60)).save(path)
        images.append(path)
    return images


def _audit_tools(root: Path) -> list[str]:
    return [entry["tool"] for entry in ts.read_log(audit_log_key(root)).records]


def _save(root: Path, image: Path, payloads: list[dict], **kwargs):
    return save_label_document(root, image_label_key(image), payloads, width=SIZE, height=SIZE,
                               author="user:breeder", actor="user:breeder", **kwargs)[0]


# ── the save door ──────────────────────────────────────────────────────────


def test_a_save_commits_its_verdicts_and_its_document_together_or_neither(
        tmp_path: Path) -> None:
    from tcip_mcp.tools.proposal_tools import stage_proposals

    image = _capture(tmp_path)[0]
    staged = stage_proposals(tmp_path, str(image), model_name="detector", boxes=[
        {"subject": BUD, "conf": 0.9, "cx": 0.5, "cy": 0.5, "w": 0.25, "h": 0.25}])
    assert "error" not in staged, staged
    bucket, shard = staged["bucket"], verdict_key_of(image_label_key(image), staged["bucket"])
    first = _save(tmp_path, image, [{"subject": BUD, "bbox": [1.0, 1.0, 9.0, 9.0]}])
    _save(tmp_path, image, [], expect=first)
    saves = _audit_tools(tmp_path).count("save_label_document")

    with pytest.raises(ts.VersionConflict):
        _save(tmp_path, image, [], expect=first,
              gestures=Gestures(bucket=bucket, accept=frozenset({0})))

    assert read_verdicts(shard) == []
    assert json_io.read_label_document(image_label_key(image)).annotations == []
    assert _audit_tools(tmp_path).count("save_label_document") == saves

    current = ts.read_versioned(image_label_key(image)).version
    _save(tmp_path, image, [], expect=current,
          gestures=Gestures(bucket=bucket, accept=frozenset({0})))

    assert [v.action for v in read_verdicts(shard)] == ["accepted"]
    assert len(json_io.read_label_document(image_label_key(image)).annotations) == 1
    assert _audit_tools(tmp_path).count("save_label_document") == saves + 1


# ── the COCO import door ───────────────────────────────────────────────────


def _coco(where: Path, images: list[Path]) -> Path:
    """An external COCO document listing ``images``, one :data:`BUD` box on each."""
    document = where / "external.json"
    document.write_text(json.dumps({
        "categories": [{"id": 1, "name": BUD}],
        "images": [{"id": i, "file_name": p.name, "width": SIZE, "height": SIZE}
                   for i, p in enumerate(images, start=1)],
        "annotations": [{"id": i, "image_id": i, "category_id": 1, "bbox": [2, 2, 10, 10]}
                        for i in range(1, len(images) + 1)],
    }), encoding="utf-8")
    return document


def test_a_coco_import_writes_every_document_or_none(tmp_path: Path, monkeypatch) -> None:
    """The import's line is appended after its documents are written, inside the one commit: a
    line that will not encode leaves none of the documents written before it."""
    import tcip_mcp.audit as audit
    from tcip_mcp.pipelines.data.coco_import import import_coco_document

    root = tmp_path / "ds"
    images = _capture(root)
    document = _coco(tmp_path, images)
    real_entry = audit.audit_entry
    monkeypatch.setattr(audit, "audit_entry",
                        lambda *a, **k: {**real_entry(*a, **k), "unencodable": float("nan")})

    with pytest.raises(ts.StoreError):
        import_coco_document(document, root, date=DATE)

    assert [ts.exists(image_label_key(p)) for p in images] == [False] * len(images)
    assert "coco_document_imported" not in _audit_tools(root)

    monkeypatch.setattr(audit, "audit_entry", real_entry)
    imported = import_coco_document(document, root, date=DATE)

    assert imported["written"] == list(STEMS)
    assert [len(json_io.read_label_document(image_label_key(p)).annotations)
            for p in images] == [1, 1, 1]
    assert _audit_tools(root).count("coco_document_imported") == 1


def test_an_import_refuses_over_a_present_record_whatever_it_holds(tmp_path: Path) -> None:
    """Presence is the store's: a label record holding ``null`` is present, and the import
    writes over it no more than over a document."""
    from tcip_mcp.pipelines.data.coco_import import import_coco_document

    root = tmp_path / "ds"
    images = _capture(root)
    ts.replace(image_label_key(images[-1]), None)

    with pytest.raises(ValueError, match=r"already holds a label document for \['c'\]"):
        import_coco_document(_coco(tmp_path, images), root, date=DATE)

    assert ts.read(image_label_key(images[-1])) is None


# ── the publication door ───────────────────────────────────────────────────


def test_a_document_its_bucket_names_is_required(tmp_path: Path) -> None:
    """A document a bucket record names and the store no longer holds refuses where it is
    counted, never reads as an image with no detections; a bucket record holding ``null`` is a
    published bucket."""
    pytest.importorskip("torch")
    from tcip_mcp.buckets import BucketExists, Document, detection_rows, publish
    from tcip_mcp.dataset_layout import bucket_key
    from tests._chain_fixtures import predicted, published

    root = tmp_path / "ds"
    images = _capture(root)
    bucket = published(tmp_path, f"m/{DATE}", [predicted(p, [BUD]) for p in images[:2]],
                       scope={"subject": BUD})
    ts.delete(prediction_key(root, bucket.name, "b"))

    with pytest.raises(json_io.UnreadableLabelDocument, match="has no record"):
        detection_rows(bucket)

    ts.replace(bucket_key(root, f"n/{DATE}"), None)
    with pytest.raises(BucketExists):
        publish(tmp_path, root, f"n/{DATE}", [Document(str(images[2]), {"annotations": []})],
                producer=bucket.producer, scope=bucket.scope, execution=bucket.execution,
                raster_path=None, raster_identity=None, assessment_id=None, actor=None)


def test_a_publication_commits_whole_and_a_second_of_one_name_writes_nothing(
        tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from tcip_mcp.buckets import BucketExists, Document, publish, read_bucket
    from tcip_mcp.pipelines.postprocessing.export import encode_predictions
    from tests._chain_fixtures import predicted, published

    root = tmp_path / "ds"
    images = _capture(root)
    first = published(tmp_path, f"m/{DATE}", [predicted(images[0], [BUD])], scope={"subject": BUD})
    assert first.document_keys == [prediction_key(root, f"m/{DATE}", "a")]
    others = [Document(str(p), *encode_predictions(predicted(p, [BUD]), "detector",
                                                   scope=first.scope)) for p in images[1:]]

    def again(name: str, raster_identity: dict | None) -> None:
        publish(tmp_path, root, name, others, producer=first.producer, scope=first.scope,
                execution=first.execution, raster_path=None, raster_identity=raster_identity,
                assessment_id=None, actor=None)

    with pytest.raises(BucketExists):
        again(first.name, None)
    with pytest.raises(ts.StoreError, match="raster_identity.width is nan"):
        again(f"n/{DATE}", {"width": float("nan")})

    assert [ts.read(prediction_key(root, name, p.stem), default=None)
            for name in (first.name, f"n/{DATE}") for p in images[1:]] == [None] * 4
    assert read_bucket(root, first.name) == first
    assert _audit_tools(root).count("prediction_bucket_published") == 1

    again(f"n/{DATE}", None)

    assert sorted(read_bucket(root, f"n/{DATE}").documents) == ["b", "c"]
    assert _audit_tools(root).count("prediction_bucket_published") == 2


# ── the reader that trains ─────────────────────────────────────────────────


def test_training_reads_each_document_of_the_capture_by_its_key(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.data.label_queries import json_det_targets
    from tests._producer_fixtures import dataset_over

    images = _capture(tmp_path)
    written = {p.stem: [Annotation(subject=BUD, geometry=BBox(2 + i, 2, 12 + 2 * i, 14))
                        for i in range(k + 1)] for k, p in enumerate(images)}
    for p in images:
        label_image(p, written[p.stem], SIZE, SIZE)

    dataset = dataset_over("detection", images[0].parent, subject=BUD)

    assert sorted(Path(location).stem for location in dataset.stems) == list(STEMS)
    for location in dataset.stems:
        stem = Path(location).stem
        read = dataset.det_targets(dataset.document(location))
        expected = json_det_targets(written[stem], dataset.scope)
        assert {k: np.asarray(v).tolist() for k, v in read.items()} == \
            {k: np.asarray(v).tolist() for k, v in expected.items()}, stem


# ── a person reads the records the platform reads ──────────────────────────


def test_the_dump_decodes_to_the_documents_the_platform_reads(tmp_path: Path) -> None:
    from tcip_mcp.cli.dump_store import dump_store, spelled
    from tcip_mcp.tools.proposal_tools import stage_proposals

    root = tmp_path / "ds"
    images = _capture(root)
    _save(tmp_path, images[0], [{"subject": BUD, "bbox": [1.0, 1.0, 9.0, 9.0]}])
    _save(tmp_path, images[1], [], gestures=Gestures(complete={BUD: True}))
    staged = stage_proposals(tmp_path, str(images[2]), model_name="detector", boxes=[
        {"subject": BUD, "conf": 0.8, "cx": 0.5, "cy": 0.5, "w": 0.25, "h": 0.25}])
    assert "error" not in staged, staged
    keys = [*(image_label_key(p) for p in images[:2]),
            prediction_key(root, staged["bucket"], images[2].stem)]

    dump_store(tmp_path, tmp_path.parent / "dump")

    for key in keys:
        dumped = (tmp_path.parent / "dump" / spelled("ds")).joinpath(
            *(spelled(name) for name in (key.store, *key.parts[:-1])),
            f"{spelled(key.parts[-1])}.json")
        assert json_io.label_document(ts.decode_value(dumped.read_bytes())) == \
            json_io.read_label_document(key), key
