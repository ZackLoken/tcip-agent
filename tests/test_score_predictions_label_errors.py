"""score_predictions: a present, unreadable label or prediction document is an error naming the
file, never a raise through the MCP tool boundary, on both the single-image and folder paths.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from tcip_annotation.json_io import write_annotations
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.tools.annotation_tools import score_predictions


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), color=(120, 120, 120)).save(path)


def test_score_predictions_single_image_reports_an_unreadable_gt(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    preds = tmp_path / "predictions" / "baseline"
    labels.mkdir(parents=True)
    preds.mkdir(parents=True)
    img = images / "IMG_0000.jpg"
    _write_image(img)
    bad = labels / "IMG_0000.json"
    bad.write_bytes(b"{not json")
    write_annotations(preds / "IMG_0000.json",
                      [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5), score=0.9)], 100, 80)

    res = score_predictions(str(img))

    assert "error" in res
    assert str(bad) in res["error"]


def test_score_predictions_folder_reports_an_unreadable_prediction(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    images = root / "images"
    labels = root / "annotations"
    preds = root / "predictions" / "baseline"
    labels.mkdir(parents=True)
    preds.mkdir(parents=True)
    _write_image(images / "IMG_0000.jpg")
    write_annotations(labels / "IMG_0000.json",
                      [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))], 100, 80)
    bad = preds / "IMG_0000.json"
    bad.write_bytes(b"{not json")

    res = score_predictions(str(root))

    assert "error" in res
    assert str(bad) in res["error"]


def _seed_sidecar(pred_dir: Path, sidecar: dict) -> None:
    """A bucket's own stamp, written straight through the store: the rail refuses a fresh
    write_sidecar call missing the (subject, attribute) pair, so a pre-conform stamp (and its
    undecodable counterpart, below) has no live producer left to build it through."""
    import tcip_store
    from tcip_mcp.pipelines.resolution import sidecar_key

    tcip_store.replace(sidecar_key(pred_dir, "operating_point"), sidecar, expect=tcip_store.Version.ABSENT)


def _damage_sidecar(pred_dir: Path) -> None:
    """Corrupt a bucket's already-seeded stamp in place, wherever the bound backend keeps it."""
    import os

    from tcip_mcp.pipelines.resolution import sidecar_key
    from tcip_store.binding import BACKEND_ENV, DEFAULT_BACKEND, FILE_BACKEND
    from tcip_store.store import _backend

    key = sidecar_key(pred_dir, "operating_point")
    if (os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND) == FILE_BACKEND:
        _backend().path_for(key).write_bytes(b"{not json")
        return
    import sqlite3

    from tcip_store.sqlite_backend import database_path, encode_parts

    conn = sqlite3.connect(str(database_path(str(key.root))), isolation_level=None)
    try:
        conn.execute("update records set value = ? where store = ? and parts = ?",
                    (b"{not json", key.store, encode_parts(key.parts)))
    finally:
        conn.close()


def test_score_predictions_single_image_refuses_a_neither_key_stamp(tmp_path: Path) -> None:
    """A classified bucket's stamp predating the writer rail's (subject, attribute) pair is never
    scored as if its value-keyed record were the object class; the remedy names the conform
    script rather than silently reading the bucket as a bare, unscoped directory."""
    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    preds = tmp_path / "predictions" / "classifier"
    labels.mkdir(parents=True)
    preds.mkdir(parents=True)
    img = images / "IMG_0000.jpg"
    _write_image(img)
    write_annotations(labels / "IMG_0000.json",
                      [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))], 100, 80)
    write_annotations(preds / "IMG_0000.json",
                      [Annotation(subject="open", geometry=BBox(1, 1, 5, 5), score=0.9)], 100, 80)
    _seed_sidecar(preds, {"id_map": {"open": 0}})

    res = score_predictions(str(img))

    assert "error" in res
    assert "repair-classified-predictions" in res["error"]


def test_score_predictions_folder_refuses_an_undecodable_stamp(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    images = root / "images"
    labels = root / "annotations"
    preds = root / "predictions" / "classifier"
    labels.mkdir(parents=True)
    preds.mkdir(parents=True)
    _write_image(images / "IMG_0000.jpg")
    write_annotations(labels / "IMG_0000.json",
                      [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))], 100, 80)
    write_annotations(preds / "IMG_0000.json",
                      [Annotation(subject="open", geometry=BBox(1, 1, 5, 5), score=0.9)], 100, 80)
    _seed_sidecar(preds, {"id_map": {"open": 0}})
    _damage_sidecar(preds)

    res = score_predictions(str(root))

    assert "error" in res


def test_score_predictions_over_a_conformed_classified_bucket_scores_the_object_class(
    tmp_path: Path,
) -> None:
    """Once a classified bucket carries the object class in subject (the conformed shape), this
    scores its localization, a valid number about finding the object, never the classifier's own
    call: the pinned reading a classified bucket now gets instead of matching nothing."""
    from tcip_mcp.pipelines.resolution import write_sidecar

    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    preds = tmp_path / "predictions" / "classifier"
    labels.mkdir(parents=True)
    preds.mkdir(parents=True)
    img = images / "IMG_0000.jpg"
    _write_image(img)
    write_annotations(labels / "IMG_0000.json",
                      [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))], 100, 80)
    write_annotations(
        preds / "IMG_0000.json",
        [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5), score=0.9,
                   attributes={"opening": "open"})],
        100, 80,
    )
    write_sidecar(preds, {"id_map": {"open": 0}, "subject": "bud", "attribute": "opening"})

    res = score_predictions(str(img))

    assert "error" not in res
    assert res["tp"] == 1
