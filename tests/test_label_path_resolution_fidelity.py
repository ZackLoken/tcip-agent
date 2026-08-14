"""The resolver's identity fidelity: an image's whole filename stem, the date bucket a caller
asks for, the date a label path can honestly report, and the fact that a format name selects a
parser rather than a different file on disk."""

from __future__ import annotations

from pathlib import Path

from tcip_mcp.dataset_layout import (
    annotation_date,
    annotation_path,
    annotation_path_for_image,
    find_gt_label,
    label_ext,
    parse_image_path,
)


def test_image_identity_keeps_every_dot_segment_of_the_filename() -> None:
    """A capture whose name carries internal dots keeps all of them in its identity stem.

    Names like ``IMG_1.2026-02-11.JPG`` (a camera stamp) and the members of a band group
    (``plot_a.ms.tif``, ``plot_a.rgb.tif``) differ only after the first dot, so truncating at
    that dot would merge two distinct images into one identity.
    """
    _root, date, stem = parse_image_path("/ds/images/2026-02-11/IMG_1.2026-02-11.JPG")
    assert date == "2026-02-11"
    assert stem == "IMG_1.2026-02-11"

    _r_ms, _d_ms, stem_ms = parse_image_path("/ds/images/2026-02-11/plot_a.ms.tif")
    _r_rgb, _d_rgb, stem_rgb = parse_image_path("/ds/images/2026-02-11/plot_a.rgb.tif")
    assert stem_ms == "plot_a.ms"
    assert stem_rgb == "plot_a.rgb"
    assert stem_ms != stem_rgb


def test_dotted_image_names_resolve_to_separate_label_files(tmp_path: Path) -> None:
    """Two images sharing a first dot-segment own separate label files, written and found."""
    img_dir = tmp_path / "images" / "2026-02-11"
    img_dir.mkdir(parents=True)
    ms = img_dir / "plot_a.ms.tif"
    rgb = img_dir / "plot_a.rgb.tif"
    ms.write_bytes(b"x")
    rgb.write_bytes(b"y")

    ms_label = annotation_path_for_image(ms)
    rgb_label = annotation_path_for_image(rgb)
    assert ms_label == tmp_path / "annotations" / "2026-02-11" / "plot_a.ms.json"
    assert rgb_label == tmp_path / "annotations" / "2026-02-11" / "plot_a.rgb.json"

    ms_label.parent.mkdir(parents=True)
    ms_label.write_text('{"annotations": []}', encoding="utf-8")
    rgb_label.write_text('{"annotations": []}', encoding="utf-8")
    assert find_gt_label(ms) == ms_label
    assert find_gt_label(rgb) == rgb_label


def test_annotation_date_refuses_a_tree_deeper_than_one_date_bucket() -> None:
    """Only ``annotations/<date>/`` yields a date; anything deeper yields ``None``.

    The canonical tree is one date level under ``annotations/``. A deeper path (a backup subtree,
    an exports subdirectory) has no recoverable capture date, and returning its first segment
    would hand every reader a fabricated one.
    """
    assert annotation_date("/ds/annotations/2026-02-11/IMG_1.json") == "2026-02-11"
    assert annotation_date("/ds/annotations/2026-02-11") == "2026-02-11"
    assert annotation_date("/ds/annotations/IMG_1.json") is None

    assert annotation_date("/ds/annotations/2026-02-11/.original/IMG_1.json") is None
    assert annotation_date("/ds/annotations/exports/2026-02-11/IMG_1.json") is None
    assert annotation_date("/ds/annotations/2026-02-11/.original") is None


def test_caller_supplied_date_wins_over_the_image_paths_own_date() -> None:
    """An explicit ``date`` decides the bucket a label is written to.

    The image's own path supplies the date only when the caller passes none: a caller labelling a
    capture under a different session's date must land there, not back in the image's bucket.
    """
    img = "/ds/images/2026-02-11/IMG_1.JPG"
    assert annotation_path_for_image(img, date="2026-03-02") == Path(
        "/ds/annotations/2026-03-02/IMG_1.json")
    assert annotation_path_for_image(img) == Path("/ds/annotations/2026-02-11/IMG_1.json")
    # A flat (non-dated) image with an explicit date lands under that date bucket.
    assert annotation_path_for_image("/ds/images/IMG_9.JPG", date="2026-03-02") == Path(
        "/ds/annotations/2026-03-02/IMG_9.json")


def test_label_format_names_all_resolve_to_the_same_file_on_disk(tmp_path: Path) -> None:
    """``fmt`` picks a parser, never a different file: every label on disk is ``.json``.

    ``json`` and ``coco`` are two readings of the same per-image JSON file, so a reader that
    asked for one and looked for a different extension would report an existing label as absent.
    """
    assert label_ext("json") == ".json"
    assert label_ext("coco") == ".json"
    assert label_ext(None) == ".json"
    assert label_ext("COCO") == ".json"
    assert annotation_path("/ds", "2026-02-11", "IMG_1", "coco") == annotation_path(
        "/ds", "2026-02-11", "IMG_1", "json")

    img = tmp_path / "images" / "2026-02-11" / "IMG_1.JPG"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"x")
    label = tmp_path / "annotations" / "2026-02-11" / "IMG_1.json"
    label.parent.mkdir(parents=True)
    label.write_text('{"annotations": []}', encoding="utf-8")
    assert find_gt_label(img, fmt="coco") == label
    assert find_gt_label(img, fmt="json") == label
    assert find_gt_label(img) == label
