"""Tests for the canonical dataset-layout resolver."""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox

from tcip_mcp.dataset_layout import (
    annotation_dir,
    annotation_path_for_image,
    find_gt_label,
    models_with_predictions,
    parse_image_path,
    prediction_dir,
    subjects_on_date,
    subjects_with_labels,
)


def test_parse_image_path_date_nested() -> None:
    root, date, stem = parse_image_path("/ds/images/2-11-26/IMG_1.JPG")
    assert Path(root) == Path("/ds")
    assert date == "2-11-26"
    assert stem == "IMG_1"


def test_parse_image_path_flat() -> None:
    root, date, stem = parse_image_path("/ds/images/IMG_1.JPG")
    assert Path(root) == Path("/ds")
    assert date is None
    assert stem == "IMG_1"


def test_parse_image_path_refuses_an_unrecognized_shape() -> None:
    # No "images" segment anywhere in the path: neither canonical shape applies, and a guessed
    # dataset root would silently misplace a downstream write (a label file, a staged prediction).
    with pytest.raises(ValueError, match="not under a recognized dataset image tree"):
        parse_image_path("/somewhere/random/IMG_1.JPG")


def test_annotation_dir_with_and_without_date() -> None:
    # Labels are one file per image; the path no longer carries a subject or task segment.
    assert annotation_dir("/ds", "2-11-26") == Path("/ds/annotations/2-11-26")
    assert annotation_dir("/ds", None) == Path("/ds/annotations")


def test_prediction_dir_date_nested() -> None:
    assert prediction_dir("/ds", "m1", "2-11-26") == Path("/ds/predictions/m1/2-11-26")


def test_annotation_path_for_image_derives_date() -> None:
    p = annotation_path_for_image("/ds/images/2-11-26/IMG_1.JPG")
    assert p == Path("/ds/annotations/2-11-26/IMG_1.json")


def test_find_gt_label_prefers_canonical(tmp_path: Path) -> None:
    img = tmp_path / "images" / "2-11-26" / "IMG_1.JPG"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"x")
    ann = tmp_path / "annotations" / "2-11-26"
    ann.mkdir(parents=True)
    (ann / "IMG_1.json").write_text('{"annotations": []}')
    assert find_gt_label(str(img)) == ann / "IMG_1.json"


def test_find_gt_label_missing_returns_none(tmp_path: Path) -> None:
    img = tmp_path / "images" / "IMG_1.JPG"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"x")
    assert find_gt_label(str(img)) is None


def _touch(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_subjects_with_labels_is_per_date(tmp_path: Path) -> None:
    root = tmp_path
    # Subjects are read from the per-image label records (the path no longer encodes them).
    # bud labeled on 2026-02-11 only; bush labeled on 2026-03-02 only.
    json_io.write_annotations(
        str(annotation_dir(root, "2026-02-11") / "IMG_1.json"),
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 100, 100)
    json_io.write_annotations(
        str(annotation_dir(root, "2026-03-02") / "IMG_9.json"),
        [Annotation(subject="bush", geometry=BBox(1, 1, 9, 9))], 100, 100)
    # A second image on 2026-03-02 carries bud, so that date offers both subjects.
    json_io.write_annotations(
        str(annotation_dir(root, "2026-03-02") / "IMG_5.json"),
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 100, 100)

    assert subjects_with_labels(root, "2026-02-11") == ["bud"]
    # 2026-03-02 has bush and bud → both, sorted.
    assert subjects_with_labels(root, "2026-03-02") == ["bud", "bush"]
    # A date with no labels for any subject → nothing to offer.
    assert subjects_with_labels(root, "2026-03-24") == []


def test_subjects_on_date_excludes_a_bucket_sidecar(tmp_path: Path) -> None:
    """A label whose filename is a bucket's own provenance stamp is neither read nor raised
    over: it is excluded from the walk the same way every prediction bucket is."""
    root = tmp_path
    json_io.write_annotations(
        str(annotation_dir(root, "2026-02-11") / "IMG_1.json"),
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 100, 100)
    (annotation_dir(root, "2026-02-11") / "operating_point.json").write_text(
        '{"not": "a label"}', encoding="utf-8")

    assert subjects_on_date(root, "2026-02-11") == ["bud"]


def test_subjects_on_date_raises_on_an_unreadable_label(tmp_path: Path) -> None:
    from tcip_annotation.json_io import UnreadableLabelDocument

    root = tmp_path
    (annotation_dir(root, "2026-02-11")).mkdir(parents=True)
    (annotation_dir(root, "2026-02-11") / "IMG_1.json").write_bytes(b"{not json")

    with pytest.raises(UnreadableLabelDocument):
        subjects_on_date(root, "2026-02-11")


def test_models_with_predictions_is_per_date(tmp_path: Path) -> None:
    root = tmp_path
    _touch(prediction_dir(root, "baseline", "2026-02-11") / "IMG_1.json", '{"annotations": []}')
    # 'baseline' has a predictions dir on 03-24 but no files in it → not offered.
    (prediction_dir(root, "baseline", "2026-03-24")).mkdir(parents=True)

    assert models_with_predictions(root, "2026-02-11") == ["baseline"]
    assert models_with_predictions(root, "2026-03-24") == []
    assert models_with_predictions(root, "2026-03-02") == []


def test_a_bucket_holding_only_its_stamp_offers_no_predictions(tmp_path: Path) -> None:
    """A bucket whose one file is its own ``operating_point.json`` stamp, on disk the way the file
    backend writes it, has nothing to overlay, so it is not offered beside a bucket holding a
    prediction document."""
    import tcip_store as ts
    from tcip_store.file_backend import FileBackend

    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json
    from tcip_mcp.pipelines.resolution import operating_point_stamp, sidecar_key

    date = "2026-02-11"
    key = sidecar_key(prediction_dir(tmp_path, "stamped", date))
    stamp_file = FileBackend().path_for(key)
    stamp_file.parent.mkdir(parents=True)
    stamp_file.write_bytes(ts.get_descriptor(key.store).codec.encode(operating_point_stamp(
        None, validated=False, validated_by=None, tile_size_validated=None, shippable_issues=[],
        id_map=None, subject="bud", attribute=None, trait=None, dataset_hash=None,
        checkpoint=None, checkpoint_sha256=None, experiment_id=None, images_dir=None,
        raster_path=None, produced_at=None)))
    assert [f.name for f in stamp_file.parent.iterdir()] == ["operating_point.json"]
    published = prediction_dir(tmp_path, "published", date)
    published.mkdir(parents=True)
    write_predictions_json(
        published / "IMG_1.json", {"image": "IMG_1.jpg", "width": 64, "height": 64,
                                   "boxes": [[4.0, 4.0, 20.0, 20.0]], "scores": [0.9],
                                   "labels": [1], "count": 1},
        created_by="test-producer", subject="bud", attribute=None, id_map={"bud": 0})

    assert models_with_predictions(tmp_path, date) == ["published"]


def test_subjects_path_is_the_single_dataset_registry():
    from tcip_mcp.dataset_layout import subjects_path

    # One nested registry at the dataset root: no per-subject classes/<x>.json anymore.
    assert subjects_path("/ds") == Path("/ds/subjects.json")


def test_dataset_root_of_recovers_the_root_from_any_layout_dir():
    from tcip_mcp.dataset_layout import dataset_root_of

    assert dataset_root_of("/ds/annotations/2026-03-02") == Path("/ds")
    assert dataset_root_of("/ds/predictions/live/2026-03-02") == Path("/ds")
    assert dataset_root_of("/ds/images/2026-03-02") == Path("/ds")
    assert dataset_root_of("/some/where/else") is None
    # Anchors on the last dataset segment: a dataset nested under an ancestor named 'annotations'
    # (or any other segment) still resolves to the real root, not the ancestor.
    assert dataset_root_of("/data/annotations/proj/predictions/live") == Path("/data/annotations/proj")
    assert dataset_root_of("/data/images/proj/annotations/2026-03-02") == Path("/data/images/proj")
    # A bare segment with nothing above it is not inside a dataset.
    assert dataset_root_of("annotations/2026-03-02") is None


def test_the_status_derivation_makes_a_negative_take_the_human_marking() -> None:
    """A finished image with nothing on it is a negative; an unfinished empty one is not.

    The one mapping every caller shares, so a Complete on an empty image and a Complete on a
    populated one land on opposite tokens rather than degrees of the same one.
    """
    from tcip_mcp.dataset_layout import derive_status

    assert derive_status(completed=True, has_content=False) == "negative"
    assert derive_status(completed=True, has_content=True) == "complete"
    assert derive_status(completed=False, has_content=False) == "unannotated"
    assert derive_status(completed=False, has_content=True) == "partial"


def test_only_the_negative_token_reads_as_a_confirmed_negative() -> None:
    """The predicate every reader shares admits the confirmation and refuses its opposite."""
    from tcip_mcp.dataset_layout import IMAGE_STATUSES, is_confirmed_negative

    assert is_confirmed_negative("negative")
    assert not is_confirmed_negative("complete")
    assert not any(is_confirmed_negative(s) for s in IMAGE_STATUSES if s != "negative")
    assert not is_confirmed_negative(None)


def test_a_bucket_key_takes_apart_into_the_subject_and_date_it_was_built_from() -> None:
    """The published inverse round-trips every bucket the writer can build, dateless included."""
    from tcip_mcp.dataset_layout import bucket_subject_date, status_bucket

    for subject, date in (("bud", "2026-03-02"), ("bush", None)):
        assert bucket_subject_date(status_bucket(subject, date)) == (subject, date)


def test_the_tree_roots_are_the_dated_dirs_without_their_date() -> None:
    """The top-level calls and the dated ones are one implementation, so they cannot disagree."""
    from tcip_mcp.dataset_layout import (
        annotation_root,
        image_dir,
        image_root,
        prediction_root,
    )

    assert image_root("/ds") == image_dir("/ds", None) == Path("/ds/images")
    assert annotation_root("/ds") == annotation_dir("/ds", None) == Path("/ds/annotations")
    assert prediction_root("/ds") == Path("/ds/predictions")
    assert prediction_dir("/ds", "m", "2026-03-02").is_relative_to(prediction_root("/ds"))


def test_prediction_bucket_dirs_finds_a_dated_and_a_model_directory_bucket(tmp_path: Path) -> None:
    """Every model directory under predictions/ counts as a bucket in its own right, alongside
    each of its date subdirectories: the one walk the doctor command's registry check and
    tcip_mcp.store_catalog.project_roots both read through, so a directory one calls a
    bucket is a directory the other calls one too."""
    from tcip_mcp.dataset_layout import prediction_bucket_dirs

    root = tmp_path
    prediction_dir(root, "modelA", "2026-03-02").mkdir(parents=True)
    prediction_dir(root, "modelB", None).mkdir(parents=True)

    found = prediction_bucket_dirs(root, include_cleared=False)

    assert prediction_dir(root, "modelA", None) in found
    assert prediction_dir(root, "modelA", "2026-03-02") in found
    assert prediction_dir(root, "modelB", None) in found


def test_a_record_file_name_is_the_stem_the_resolver_would_have_used() -> None:
    """The name a label or prediction record takes, stated once and reused by both path builders."""
    from tcip_mcp.dataset_layout import annotation_path, label_filename, prediction_path

    assert label_filename("IMG_0001") == "IMG_0001.json"
    assert annotation_path("/ds", "2026-03-02", "IMG_0001").name == label_filename("IMG_0001")
    assert prediction_path("/ds", "m", "2026-03-02", "IMG_0001").name == label_filename("IMG_0001")


def test_resolve_image_name_recognizes_an_npz_capture(tmp_path: Path) -> None:
    """The platform's real extension set, not a private six-extension list: an ``.npz`` capture
    resolves to its own on-disk name rather than reading as absent."""
    from tcip_mcp.dataset_layout import resolve_image_name

    date = "2026-03-04"
    (tmp_path / "images" / date).mkdir(parents=True)
    (tmp_path / "images" / date / "plotA_0_0.npz").write_bytes(b"\x00")

    assert resolve_image_name(tmp_path, date, "plotA_0_0") == "plotA_0_0.npz"


def test_resolve_image_name_resolves_per_bucket_not_last_write_wins(tmp_path: Path) -> None:
    """A stem two date buckets both use resolves against its own bucket, never a flat map where
    one bucket's extension silently wins over the other's."""
    from tcip_mcp.dataset_layout import resolve_image_name

    (tmp_path / "images" / "d1").mkdir(parents=True)
    (tmp_path / "images" / "d2").mkdir(parents=True)
    (tmp_path / "images" / "d1" / "IMG_S.jpg").write_bytes(b"\x00")
    (tmp_path / "images" / "d2" / "IMG_S.png").write_bytes(b"\x00")

    assert resolve_image_name(tmp_path, "d1", "IMG_S") == "IMG_S.jpg"
    assert resolve_image_name(tmp_path, "d2", "IMG_S") == "IMG_S.png"


def test_resolve_image_name_is_none_for_an_unresolvable_stem(tmp_path: Path) -> None:
    from tcip_mcp.dataset_layout import resolve_image_name

    (tmp_path / "images" / "d1").mkdir(parents=True)

    assert resolve_image_name(tmp_path, "d1", "no_such_stem") is None


def test_canonical_prediction_bucket_recognizes_the_dated_and_undated_live_shapes(
    tmp_path: Path,
) -> None:
    from tcip_mcp.dataset_layout import canonical_prediction_bucket

    dated = prediction_dir(tmp_path, "modelA", "2026-03-02")
    undated = prediction_dir(tmp_path, "modelB", None)

    assert canonical_prediction_bucket(dated) == (tmp_path, "modelA", "2026-03-02")
    assert canonical_prediction_bucket(undated) == (tmp_path, "modelB", None)


def test_canonical_prediction_bucket_refuses_a_dot_prefixed_model_segment(tmp_path: Path) -> None:
    from tcip_mcp.dataset_layout import canonical_prediction_bucket

    assert canonical_prediction_bucket(tmp_path / "predictions" / ".cleared") is None
    assert canonical_prediction_bucket(tmp_path / "predictions" / ".cleared" / "m@x") is None


def test_canonical_prediction_bucket_is_none_outside_a_predictions_tree(tmp_path: Path) -> None:
    from tcip_mcp.dataset_layout import canonical_prediction_bucket

    assert canonical_prediction_bucket(tmp_path / "images" / "2026-03-02") is None


def test_cleared_prediction_dir_and_its_inverse_round_trip_dated_and_undated(
    tmp_path: Path,
) -> None:
    """``cleared_bucket_of`` is the declared inverse of ``cleared_prediction_dir``, for both the
    dated and undated archive shapes, and splits on the *last* ``@`` so a model whose own name
    already carries one (a resolver variant) resolves whole."""
    from tcip_mcp.dataset_layout import cleared_bucket_of, cleared_prediction_dir

    stamp = "20260906T120000Z"
    dated = cleared_prediction_dir(tmp_path, "modelA", "2026-03-02", stamp)
    undated = cleared_prediction_dir(tmp_path, "modelB", None, stamp)
    variant = cleared_prediction_dir(tmp_path, "baseline@r2", "2026-03-02", stamp)

    assert dated == tmp_path / "predictions" / ".cleared" / f"modelA@{stamp}" / "2026-03-02"
    assert cleared_bucket_of(dated) == (tmp_path, "modelA", stamp, "2026-03-02")
    assert cleared_bucket_of(undated) == (tmp_path, "modelB", stamp, None)
    assert cleared_bucket_of(variant) == (tmp_path, "baseline@r2", stamp, "2026-03-02")


def test_cleared_bucket_of_refuses_a_tail_that_does_not_parse_as_the_stamp(tmp_path: Path) -> None:
    from tcip_mcp.dataset_layout import cleared_bucket_of, is_cleared_bucket

    junk = tmp_path / "predictions" / ".cleared" / "modelA@not-a-stamp"
    assert cleared_bucket_of(junk) is None
    assert is_cleared_bucket(junk) is False


def test_is_cleared_bucket_agrees_with_the_inverse(tmp_path: Path) -> None:
    from tcip_mcp.dataset_layout import cleared_prediction_dir, is_cleared_bucket

    cleared = cleared_prediction_dir(tmp_path, "modelA", None, "20260906T120000Z")
    assert is_cleared_bucket(cleared) is True
    assert is_cleared_bucket(prediction_dir(tmp_path, "modelA", None)) is False


def test_prediction_bucket_dirs_include_cleared_walks_the_archive_alongside_the_live_walk(
    tmp_path: Path,
) -> None:
    from tcip_mcp.dataset_layout import cleared_prediction_dir, prediction_bucket_dirs

    prediction_dir(tmp_path, "modelA", "2026-03-02").mkdir(parents=True)
    dated_cleared = cleared_prediction_dir(tmp_path, "modelA", "2026-03-02", "20260906T120000Z")
    dated_cleared.mkdir(parents=True)
    undated_cleared = cleared_prediction_dir(tmp_path, "modelB", None, "20260906T130000Z")
    undated_cleared.mkdir(parents=True)

    live_only = prediction_bucket_dirs(tmp_path, include_cleared=False)
    assert dated_cleared not in live_only
    assert undated_cleared not in live_only

    with_archive = prediction_bucket_dirs(tmp_path, include_cleared=True)
    assert dated_cleared.parent in with_archive  # the model@stamp directory itself
    assert dated_cleared in with_archive
    assert undated_cleared in with_archive
    assert (tmp_path / "predictions" / ".cleared") not in with_archive


def test_prediction_bucket_dirs_include_cleared_skips_an_unrecognized_archive_directory(
    tmp_path: Path,
) -> None:
    """A directory under ``.cleared`` the inverse does not recognize (no ``@<stamp>`` tail) is
    walked by neither this walk nor anything built on it."""
    from tcip_mcp.dataset_layout import prediction_bucket_dirs

    junk = tmp_path / "predictions" / ".cleared" / "not-a-cleared-bucket"
    junk.mkdir(parents=True)

    assert junk not in prediction_bucket_dirs(tmp_path, include_cleared=True)


def test_prediction_bucket_date_answers_none_for_both_cleared_shapes(tmp_path: Path) -> None:
    from tcip_mcp.dataset_layout import cleared_prediction_dir, prediction_bucket_date

    stamp = "20260906T120000Z"
    dated = cleared_prediction_dir(tmp_path, "modelA", "2026-03-02", stamp)
    undated = cleared_prediction_dir(tmp_path, "modelB", None, stamp)

    assert prediction_bucket_date(dated) is None
    assert prediction_bucket_date(undated) is None


def test_find_prediction_refuses_an_explicit_model_failing_is_bucket_name(tmp_path: Path) -> None:
    from tcip_mcp.dataset_layout import find_prediction

    image = tmp_path / "images" / "2026-03-02" / "IMG_1.jpg"
    with pytest.raises(ValueError, match="is not a canonical model bucket name"):
        find_prediction(image, model=".cleared")
