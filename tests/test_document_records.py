"""Label and prediction documents as records of their dataset root.

Each door that writes them commits whole or not at all, a detection loader reads each document
of a capture by its key, and the dump decodes to the records the platform reads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

import tcip_store as ts
from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_annotation.verdicts import read_verdicts
from tcip_mcp.dataset_layout import (
    Gestures,
    image_dir,
    parse_capture_dir,
    prediction_key,
    verdict_key_of,
)
from tests._audit_fixtures import audit_rows
from tests._producer_fixtures import (
    image_label_key,
    label_image,
    registry_of,
    registry_over,
    save_labels,
    staged_box,
    write_image,
)
from tests._training_values import RATIONALE, VERTEX_SPACING

if TYPE_CHECKING:
    from tcip_mcp.pipelines.raster_source import RasterIdentity

DATE = "2026-03-02"
SIZE = 64
STEMS = ("a", "b", "c")
BUD = "bud"


def _capture(images_dir: Path) -> list[Path]:
    """Write three frames into the capture ``images_dir`` and return them.

    The registry of its dataset root (``dataset_layout.parse_capture_dir``) declares
    :data:`BUD`.
    """
    registry_over(parse_capture_dir(images_dir)[0], registry_of(BUD))
    return [
        write_image(
            images_dir / f"{stem}.png", (SIZE, SIZE), (40 + 30 * i, 60, 60)
        )
        for i, stem in enumerate(STEMS)
    ]


# ── the save door ──────────────────────────────────────────────────────────


def test_a_save_commits_its_verdicts_and_its_document_together_or_neither(
    tmp_path: Path,
) -> None:
    image = _capture(image_dir(tmp_path, DATE))[0]
    bucket = staged_box(tmp_path, image, [24.0, 24.0, 40.0, 40.0])
    shard = verdict_key_of(image_label_key(image), bucket)
    first = save_labels(
        tmp_path, image, [{"subject": BUD, "bbox": [1.0, 1.0, 9.0, 9.0]}]
    )[0]
    save_labels(tmp_path, image, [], expect=first)
    saves = len(audit_rows(tmp_path, "save_label_document"))

    with pytest.raises(ts.VersionConflictError):
        save_labels(
            tmp_path,
            image,
            [],
            expect=first,
            gestures=Gestures(bucket=bucket, accept=frozenset({0})),
        )

    assert read_verdicts(shard) == []
    assert (
        json_io.read_label_document(image_label_key(image)).annotations == []
    )
    assert len(audit_rows(tmp_path, "save_label_document")) == saves

    current = ts.read_versioned(image_label_key(image)).version
    save_labels(
        tmp_path,
        image,
        [],
        expect=current,
        gestures=Gestures(bucket=bucket, accept=frozenset({0})),
    )

    assert [v.action for v in read_verdicts(shard)] == ["accepted"]
    assert (
        len(json_io.read_label_document(image_label_key(image)).annotations)
        == 1
    )
    assert len(audit_rows(tmp_path, "save_label_document")) == saves + 1


# ── the COCO import door ───────────────────────────────────────────────────


def _coco(where: Path, images: list[Path]) -> Path:
    """Write an external COCO document listing ``images``, one :data:`BUD` box on each."""
    document = where / "external.json"
    document.write_text(
        json.dumps(
            {
                "categories": [{"id": 1, "name": BUD}],
                "images": [
                    {
                        "id": i,
                        "file_name": p.name,
                        "width": SIZE,
                        "height": SIZE,
                    }
                    for i, p in enumerate(images, start=1)
                ],
                "annotations": [
                    {
                        "id": i,
                        "image_id": i,
                        "category_id": 1,
                        "bbox": [2, 2, 10, 10],
                    }
                    for i in range(1, len(images) + 1)
                ],
            }
        ),
        encoding="utf-8",
    )
    return document


def test_a_coco_import_writes_every_document_or_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A COCO import writes every document or none.

    An audit line that will not encode raises and leaves no document and no import line; the
    same import with an encodable line writes all three documents and one line.
    """
    import tcip_mcp.audit as audit
    from tcip_mcp.pipelines.data.coco_import import import_coco_document

    root = tmp_path / "ds"
    images = _capture(image_dir(root, DATE))
    document = _coco(tmp_path, images)
    real_entry = audit.audit_entry
    monkeypatch.setattr(
        audit,
        "audit_entry",
        lambda *a, **k: {**real_entry(*a, **k), "unencodable": float("nan")},
    )

    stated = {"vertex_spacing": VERTEX_SPACING, "rationale": RATIONALE}
    with pytest.raises(ts.StoreError):
        import_coco_document(
            document, root, date=DATE, project=root, **stated, actor=None
        )

    assert [ts.exists(image_label_key(p)) for p in images] == [False] * len(
        images
    )
    assert audit_rows(root, "coco_document_imported") == []

    monkeypatch.setattr(audit, "audit_entry", real_entry)
    imported = import_coco_document(
        document, root, date=DATE, project=root, **stated, actor=None
    )

    assert imported["written"] == list(STEMS)
    assert [
        len(json_io.read_label_document(image_label_key(p)).annotations)
        for p in images
    ] == [1, 1, 1]
    assert len(audit_rows(root, "coco_document_imported")) == 1


def test_an_import_refuses_over_a_present_record_whatever_it_holds(
    tmp_path: Path,
) -> None:
    """An import refuses over a present label record, even one holding ``null``.

    Presence is the store's: a label record holding ``null`` is present, and the import writes
    over it no more than over a document.
    """
    from tcip_mcp.pipelines.data.coco_import import import_coco_document

    root = tmp_path / "ds"
    images = _capture(image_dir(root, DATE))
    ts.replace(image_label_key(images[-1]), None)

    with pytest.raises(
        ValueError, match=r"already holds a label document for \['c'\]"
    ):
        import_coco_document(
            _coco(tmp_path, images),
            root,
            date=DATE,
            vertex_spacing=VERTEX_SPACING,
            rationale=RATIONALE,
            project=root,
            actor=None,
        )

    assert ts.read(image_label_key(images[-1])) is None


# ── the publication door ───────────────────────────────────────────────────


def test_a_document_its_bucket_names_is_required(tmp_path: Path) -> None:
    """A document its bucket names is required where it is counted.

    A document a bucket record names and the store does not hold refuses where it is counted,
    and a bucket record holding ``null`` is a published bucket.
    """
    pytest.importorskip("torch")
    from tcip_mcp.buckets import (
        BucketExistsError,
        Document,
        detection_rows,
        publish,
    )
    from tcip_mcp.dataset_layout import bucket_key
    from tests._chain_fixtures import predicted, published

    root = tmp_path / "ds"
    images = _capture(image_dir(root, DATE))
    bucket = published(
        tmp_path,
        f"m/{DATE}",
        [predicted(p, [BUD]) for p in images[:2]],
        scope={"subject": BUD},
    )
    ts.delete(bucket.stem_keys["b"])

    with pytest.raises(
        json_io.UnreadableLabelDocumentError, match="has no record"
    ):
        detection_rows(bucket)

    ts.replace(bucket_key(root, f"n/{DATE}"), None)
    with pytest.raises(BucketExistsError):
        publish(
            tmp_path,
            root,
            f"n/{DATE}",
            [Document(str(images[2]), {"annotations": []})],
            producer=bucket.producer,
            scope=bucket.scope,
            execution=bucket.execution,
            raster_path=None,
            raster_identity=None,
            assessment_id=None,
            actor=None,
        )


def test_a_publication_commits_whole_and_a_second_of_one_name_writes_nothing(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from dataclasses import replace

    from tcip_mcp.buckets import (
        BucketExistsError,
        Document,
        publish,
        read_bucket,
    )
    from tcip_mcp.pipelines.data.split_construction import raster_identity
    from tcip_mcp.pipelines.postprocessing.export import encode_predictions
    from tests._chain_fixtures import predicted, published

    root = tmp_path / "ds"
    images = _capture(image_dir(root, DATE))
    unwritable = replace(
        raster_identity(images[0]), pixel_fraction=float("nan")
    )
    first = published(
        tmp_path,
        f"m/{DATE}",
        [predicted(images[0], [BUD])],
        scope={"subject": BUD},
    )
    assert first.document_keys == [prediction_key(root, f"m/{DATE}", "a")]
    others = [
        Document(
            str(p),
            *encode_predictions(
                predicted(p, [BUD]), "detector", scope=first.scope
            ),
        )
        for p in images[1:]
    ]

    def again(name: str, identity: RasterIdentity | None) -> None:
        publish(
            tmp_path,
            root,
            name,
            others,
            producer=first.producer,
            scope=first.scope,
            execution=first.execution,
            raster_path=None,
            raster_identity=identity,
            assessment_id=None,
            actor=None,
        )

    with pytest.raises(BucketExistsError):
        again(first.name, None)
    with pytest.raises(
        ts.StoreError, match=r"raster_identity\.dict\.pixel_fraction"
    ):
        again(f"n/{DATE}", unwritable)

    assert [
        ts.read(prediction_key(root, name, p.stem), default=None)
        for name in (first.name, f"n/{DATE}")
        for p in images[1:]
    ] == [None] * 4
    assert read_bucket(root, first.name) == first
    assert len(audit_rows(root, "prediction_bucket_published")) == 1

    again(f"n/{DATE}", None)

    assert sorted(read_bucket(root, f"n/{DATE}").documents) == ["b", "c"]
    assert len(audit_rows(root, "prediction_bucket_published")) == 2


def test_a_publication_writes_each_document_under_its_own_stems_key(
    tmp_path: Path,
) -> None:
    """Each published document is the one its source image's stem keys, read back as written."""
    pytest.importorskip("torch")
    from tcip_mcp.buckets import detection_rows
    from tests._chain_fixtures import predicted, published

    images = _capture(image_dir(tmp_path / "ds", DATE))
    bucket = published(
        tmp_path,
        f"m/{DATE}",
        [predicted(image, [BUD] * n) for n, image in enumerate(images)],
        scope={"subject": BUD},
    )

    assert {row["image"]: row["count"] for row in detection_rows(bucket)} == {
        image.name: n for n, image in enumerate(images)
    }


# ── the reader that trains ─────────────────────────────────────────────────


def test_training_reads_each_document_of_the_capture_by_its_key(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.data.label_queries import json_det_targets
    from tests._producer_fixtures import dataset_over

    images_dir = image_dir(tmp_path, DATE)
    images = _capture(images_dir)
    written = {
        p.stem: [
            Annotation(subject=BUD, geometry=BBox(2 + i, 2, 12 + 2 * i, 14))
            for i in range(k + 1)
        ]
        for k, p in enumerate(images)
    }
    for p in images:
        label_image(p, written[p.stem], SIZE, SIZE)

    dataset = dataset_over("detection", images_dir, subject=BUD)

    assert sorted(Path(location).stem for location in dataset.stems) == list(
        STEMS
    )
    for location in dataset.stems:
        stem = Path(location).stem
        read = dataset.det_targets(dataset.document(location))
        expected = json_det_targets(written[stem], dataset.scope)
        assert {k: np.asarray(v).tolist() for k, v in read.items()} == {
            k: np.asarray(v).tolist() for k, v in expected.items()
        }, stem


# ── a person reads the records the platform reads ──────────────────────────


def test_the_dump_decodes_to_the_documents_the_platform_reads(
    tmp_path: Path,
) -> None:
    from tcip_mcp.cli.dump_store import dump_store, spelled

    root = tmp_path / "ds"
    images = _capture(image_dir(root, DATE))
    save_labels(
        tmp_path, images[0], [{"subject": BUD, "bbox": [1.0, 1.0, 9.0, 9.0]}]
    )
    save_labels(
        tmp_path, images[1], [], gestures=Gestures(complete={BUD: True})
    )
    bucket = staged_box(tmp_path, images[2], [24.0, 24.0, 40.0, 40.0])
    keys = [
        *(image_label_key(p) for p in images[:2]),
        prediction_key(root, bucket, images[2].stem),
    ]

    dump_store(tmp_path, tmp_path.parent / "dump")

    for key in keys:
        dumped = (tmp_path.parent / "dump" / spelled("ds")).joinpath(
            *(spelled(name) for name in (key.store, *key.parts[:-1])),
            f"{spelled(key.parts[-1])}.json",
        )
        assert json_io.label_document(
            ts.decode_value(dumped.read_bytes())
        ) == json_io.read_label_document(key), key
