"""One read of each document, each bucket record and each ground-truth table per act.

The admission's read is the one its loaders, their targets, their class counts, every item and the
assessment's retained copy consume, and the scoring's read is the one its renderers draw.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import IO, TYPE_CHECKING

import pytest

from tests._training_values import (
    IOU_THRESHOLD,
    RATIONALE,
    TRIAGE_WEIGHTS,
    drawn_split,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from _typeshed import FileDescriptorOrPath, StrOrBytesPath
    from PIL.ImageFile import ImageFile
    from torch.utils.data import Dataset

    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.data.selection import ClassScope, Sample
    from tcip_store import Key, ReadDefault, Versioned
    from tcip_store.sqlite_backend import SqliteBackend

torch = pytest.importorskip("torch")


_READ_STORES = (
    "label_documents",
    "prediction_documents",
    "prediction_buckets",
)
"""The stores whose records hold ground truth, predictions and the buckets naming them."""


def _counting_reads(
    monkeypatch: pytest.MonkeyPatch, files: tuple[str, ...] = ()
) -> Counter:
    """Count reads at the store boundary and file opens, keyed by what was read.

    Counted are each ``SqliteBackend.read_versioned`` of a record of :data:`_READ_STORES` and each
    ``builtins.open`` or ``io.open`` for reading of a file whose suffix is among ``files``.
    """
    import builtins
    import io

    from tcip_store.sqlite_backend import SqliteBackend

    counts: Counter = Counter()
    real_record = SqliteBackend.read_versioned

    def counted_record(
        self: SqliteBackend, key: Key, **default: ReadDefault
    ) -> Versioned:
        if key.store in _READ_STORES:
            counts[str(key)] += 1
        return real_record(self, key, **default)

    monkeypatch.setattr(SqliteBackend, "read_versioned", counted_record)
    real_open = builtins.open

    def counted_open(
        file: FileDescriptorOrPath,
        mode: str = "r",
        *args: int | str | bool | Callable[[str, int], int] | None,
        **kwargs: int | str | bool | Callable[[str, int], int] | None,
    ) -> IO[str] | IO[bytes]:
        if (
            isinstance(file, (str, Path))
            and mode.startswith("r")
            and Path(file).suffix in files
        ):
            counts[str(Path(file))] += 1
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", counted_open)
    monkeypatch.setattr(io, "open", counted_open)
    return counts


def _consume(*datasets: Dataset | None) -> None:
    """Read the class distribution and each item of each loader."""
    for ds in datasets:
        ds.class_distribution
        for i in range(len(ds)):
            ds[i]


def test_a_drawn_detection_run_reads_each_document_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests._producer_fixtures import train_val
    from tests.test_training_autoval import _detection_dataset

    images_dir, stems = _detection_dataset(tmp_path / "ds")
    counts = _counting_reads(monkeypatch)

    train, val, _partition, _data = train_val(
        tmp_path,
        "detection",
        {
            "images_dir": str(images_dir),
            "scope": {"subject": "bud"},
            "split": drawn_split(
                val_ratio=0.4, seed=1, grouping="tile_prefix"
            ),
        },
    )
    _consume(train, val)

    assert len(counts) == len(stems)
    assert set(counts.values()) == {1}, counts


def test_a_drawn_table_run_reads_its_table_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests._producer_fixtures import train_val
    from tests.tiny_trainer_fixtures import write_regression_dataset

    images_dir, csv_path = write_regression_dataset(
        tmp_path, intensities=[0.1, 0.3, 0.5, 0.7], values=[0.2, 0.6, 1.0, 1.4]
    )
    counts = _counting_reads(monkeypatch, files=(".csv",))

    train, val, _partition, _data = train_val(
        tmp_path,
        "regression",
        {
            "images_dir": str(images_dir),
            "labels_dir": str(csv_path),
            "split": drawn_split(val_ratio=0.5, grouping="tile_prefix"),
        },
    )
    _consume(train, val)

    assert counts == Counter({str(csv_path): 1})


def test_a_written_draw_reads_each_document_once_with_its_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcip_mcp.tools.data_tools import draw_splits
    from tests.test_training_autoval import _detection_dataset

    images_dir, stems = _detection_dataset(tmp_path / "ds")
    counts = _counting_reads(monkeypatch)

    result = draw_splits(
        tmp_path,
        str(tmp_path / "ds"),
        seed=1,
        subject="bud",
        val_ratio=0.25,
        calibration_ratio=0,
        holdout_ratio=0,
        grouping="stem",
        stratify_foreground=True,
        rationale=RATIONALE,
        output_path=str(tmp_path / "selection"),
        actor=None,
    )

    assert "error" not in result, result
    assert len(counts) == len(stems)
    assert set(counts.values()) == {1}, counts


def test_a_recorded_table_changed_under_its_digest_refuses_by_name(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    from tcip_mcp.pipelines.data.label_queries import acquired, admit
    from tests.tiny_trainer_fixtures import write_regression_dataset

    images_dir, csv_path = write_regression_dataset(
        tmp_path, intensities=[0.1, 0.3], values=[0.2, 0.6]
    )
    admitted = admit(images_dir, str(csv_path)).every_sample()
    recorded = [replace(s, read=None, stored=None) for s in admitted]
    # Admits valid work: a replay of the unchanged table reads what the admission read.
    assert [s.read for s in acquired(recorded)] == [s.read for s in admitted]

    csv_path.write_bytes(csv_path.read_bytes().replace(b"0.6", b"9.9"))
    with pytest.raises(ValueError, match="the version recorded"):
        acquired(recorded)


def test_an_assessment_reads_each_reference_document_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An assessment reads each reference document once and retains what that read answered.

    A document emptied after the assessment's re-admission read is measured as read: the
    assessment succeeds and its retained digests are those the re-admission answered.
    """
    import tcip_mcp.pipelines.data.label_queries as label_queries
    from tcip_annotation import json_io
    from tcip_mcp.assessment import read_assessment
    from tests._chain_fixtures import IMG, assess, confirm_count_trait
    from tests.test_assessment_guards import _trained

    _images_dir, drawn, selection_dir, checkpoint = _trained(
        tmp_path, "exp-reads"
    )
    confirm_count_trait(tmp_path)
    reference = sorted(
        (
            s.ground_truth
            for s in drawn.samples
            if s.side in ("calibration", "holdout")
        ),
        key=str,
    )
    readmitted: dict = {}
    real_readmit = label_queries.readmitted_samples

    def emptied_after_its_read(
        samples: Sequence[Sample], scope: ClassScope
    ) -> list[Sample]:
        out = real_readmit(samples, scope)
        readmitted.update((s.ground_truth, s.ground_truth_digest) for s in out)
        json_io.write_label_document(
            reference[0], [], IMG, IMG, keep_empty=True
        )
        return out

    monkeypatch.setattr(
        label_queries, "readmitted_samples", emptied_after_its_read
    )
    counts = _counting_reads(monkeypatch)

    record = assess(tmp_path, checkpoint, selection_dir)

    assert "error" not in record, record
    assert {str(key) for key in reference} <= set(counts), counts
    assert set(counts.values()) == {1}, counts
    retained = read_assessment(
        tmp_path, record["assessment_id"]
    ).reference.ground_truth
    assert {f.ground_truth: f.digest for f in retained} == {
        key: readmitted[key] for key in reference
    }
    assert json_io.read_label_document(reference[0]).annotations == []


def test_a_reference_document_emptied_before_the_assessment_reads_it_refuses(
    tmp_path: Path,
) -> None:
    from tcip_annotation import json_io
    from tests._chain_fixtures import IMG, assess, confirm_count_trait
    from tests.test_assessment_guards import _trained

    _images_dir, drawn, selection_dir, checkpoint = _trained(
        tmp_path, "exp-emptied"
    )
    confirm_count_trait(tmp_path)
    emptied = next(
        s.ground_truth for s in drawn.samples if s.side == "holdout"
    )
    json_io.write_label_document(emptied, [], IMG, IMG, keep_empty=True)

    record = assess(tmp_path, checkpoint, selection_dir)

    assert "no longer admissible" in record["error"], record


def test_a_semantic_run_reads_each_mask_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
    from tests._producer_fixtures import painted_frame, train_val, write_image

    images_dir, masks_dir = (
        image_dir(tmp_path, UNDATED_BUCKET),
        tmp_path / "masks",
    )
    masks_dir.mkdir()
    for i in range(4):
        write_image(images_dir / f"tree{i}.png", (8, 8), (i * 40, 0, 0))
        painted_frame(8, 8, 0, [((2, 2, 6, 6), 1)], mode="L").save(
            masks_dir / f"tree{i}.png"
        )
    counts = _counting_reads(monkeypatch, files=(".png",))

    train, val, _partition, _data = train_val(
        tmp_path,
        "semantic_seg",
        {
            "images_dir": str(images_dir),
            "labels_dir": str(masks_dir),
            "split": drawn_split(val_ratio=0.5, grouping="tile_prefix"),
        },
    )
    _consume(train, val)

    masks = {k: n for k, n in counts.items() if Path(k).parent == masks_dir}
    assert len(masks) == 4 and set(masks.values()) == {1}, counts


def test_a_freeze_reads_each_document_once_with_its_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcip_mcp.tools.data_tools import freeze_selection
    from tests.test_freeze_selection import _real_drawn_experiment
    from tests.test_selection_binding import _two_subject_two_date_dataset

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(tmp_path, root, "exp-frozen")
    counts = _counting_reads(monkeypatch)

    result = freeze_selection(tmp_path, "exp-frozen", actor=None)

    assert "error" not in result, result
    assert counts and set(counts.values()) == {1}, counts


def test_a_physical_scale_assessment_reads_its_references_and_table_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests import _trait_fixtures as fx
    from tests.test_physical_scale_assessment import (
        _author_tolerance,
        _calibrate,
        _reference,
    )

    fx.seed_delivery_traits(tmp_path)
    _author_tolerance(tmp_path)
    selection, csv_path = _reference(tmp_path)
    counts = _counting_reads(monkeypatch, files=(".csv",))

    scale = _calibrate(tmp_path, selection, csv_path)

    assert "error" not in scale, scale
    assert counts[str(csv_path)] == 1
    assert set(counts.values()) == {1}, counts


def test_a_reserved_regions_assessment_reads_its_mosaic_document_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests import _trait_fixtures as fx
    from tests.test_block_calibration import _assess, _attested

    fx.seed_confirmed_count(tmp_path, measured_subject="bud")
    exp = _attested(tmp_path)
    counts = _counting_reads(monkeypatch)

    record = _assess(exp)

    assert "error" not in record, record
    assert counts[str(exp["label"])] == 1
    assert set(counts.values()) == {1}, counts


def _published_frames(project: Path, images_dir: Path) -> list[Path]:
    """Return the image paths of two labeled frames written into the capture ``images_dir``.

    Each frame has its predictions published as ``project``'s bucket ``published``.
    """
    from tcip_annotation.state import Annotation, BBox
    from tests._chain_fixtures import published
    from tests._producer_fixtures import label_image, write_image

    frames = []
    for name in ("leaf_a", "leaf_b"):
        frame = write_image(
            images_dir / f"{name}.png", (64, 48), (90, 110, 70)
        )
        label_image(
            frame,
            [Annotation(subject="leaf", geometry=BBox(8, 8, 24, 24))],
            64,
            48,
        )
        frames.append(frame)
    published(
        project,
        "published",
        [
            {
                "image": str(frame),
                "width": 64,
                "height": 48,
                "boxes": [[8, 8, 24, 24], [40, 30, 50, 40]],
                "scores": [0.9, 0.7],
                "labels": [1, 1],
                "cap": 3,
            }
            for frame in frames
        ],
        scope={"subject": "leaf"},
    )
    return frames


def _counting_scans(monkeypatch: pytest.MonkeyPatch) -> Counter:
    """Count every scan of an image directory for its logical images, keyed by directory."""
    from tcip_mcp.pipelines import image_utils

    scans: Counter = Counter()
    real_scan = image_utils.scan_identities

    def counted_scan(directory: Path) -> dict[str, list[Path | BandGroupRef]]:
        scans[str(directory)] += 1
        return real_scan(directory)

    monkeypatch.setattr(image_utils, "scan_identities", counted_scan)
    return scans


def _counting_resolutions(monkeypatch: pytest.MonkeyPatch) -> Counter:
    """Count every path resolved to its logical image, keyed by path."""
    from tcip_mcp.pipelines import image_utils

    resolutions: Counter = Counter()
    real_resolve = image_utils.resolve_image_paths

    def counted_resolve(
        paths: Iterable[str | Path],
    ) -> list[Path | BandGroupRef]:
        paths = list(paths)
        resolutions.update(str(Path(p)) for p in paths)
        return real_resolve(paths)

    monkeypatch.setattr(image_utils, "resolve_image_paths", counted_resolve)
    return resolutions


def test_an_assessment_resolves_each_image_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests._chain_fixtures import assess, confirm_count_trait
    from tests.test_assessment_guards import _trained

    _images_dir, drawn, selection_dir, checkpoint = _trained(
        tmp_path, "exp-resolved"
    )
    confirm_count_trait(tmp_path)
    reference = {
        str(Path(s.source))
        for s in drawn.samples
        if s.side in ("calibration", "holdout")
    }
    resolutions = _counting_resolutions(monkeypatch)

    record = assess(tmp_path, checkpoint, selection_dir)

    assert "error" not in record, record
    assert reference <= set(resolutions), resolutions
    assert set(resolutions.values()) == {1}, resolutions


def test_a_selection_bound_run_resolves_each_image_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests._producer_fixtures import train_val
    from tests.test_selection_binding import (
        _draw,
        _run_data_cfg,
        _two_subject_two_date_dataset,
    )

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    drawn = _draw(tmp_path, root, tmp_path / "m")
    resolutions = _counting_resolutions(monkeypatch)

    train, val, _partition, _data = train_val(
        tmp_path, "detection", _run_data_cfg(root, tmp_path / "m")
    )
    _consume(train, val)

    trained = {
        str(Path(s.source)) for s in drawn.on("train") + drawn.on("val")
    }
    assert trained <= set(resolutions), resolutions
    assert set(resolutions.values()) == {1}, resolutions


def test_a_comparison_render_reads_its_documents_and_bucket_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
    from tcip_mcp.tools.vision_tools import ComparisonRender, visualize

    images = image_dir(tmp_path, UNDATED_BUCKET)
    frame, _other = _published_frames(tmp_path, images)
    counts = _counting_reads(monkeypatch)
    scans = _counting_scans(monkeypatch)
    opened: list = []
    real_open = Image.open

    def counted_open(
        fp: StrOrBytesPath | IO[bytes],
        *args: str | list[str] | tuple[str, ...] | None,
        **kwargs: str | list[str] | tuple[str, ...] | None,
    ) -> ImageFile:
        if Path(str(fp)) == frame:
            opened.append(fp)
        return real_open(fp, *args, **kwargs)

    monkeypatch.setattr(Image, "open", counted_open)

    result = visualize(
        tmp_path,
        ComparisonRender(
            source="comparison",
            path=str(frame),
            bucket="published",
            conf_threshold=0.0,
            iou_threshold=IOU_THRESHOLD,
        ),
    )

    assert "error" not in result, result
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)
    assert len(counts) == 3, counts
    assert set(counts.values()) == {1}, counts
    assert scans == Counter({str(images): 1}), scans
    assert len(opened) == 1, opened


def test_a_folder_scoring_resolves_its_images_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
    from tcip_mcp.tools.annotation_tools import score_predictions

    images = image_dir(tmp_path, UNDATED_BUCKET)
    frames = _published_frames(tmp_path, images)
    scans = _counting_scans(monkeypatch)

    result = score_predictions(
        tmp_path,
        str(images),
        "published",
        IOU_THRESHOLD,
        conf_threshold=0.0,
    )

    assert "error" not in result, result
    assert result["image_count"] == len(frames)
    assert scans == Counter({str(images): 1}), scans


def test_a_failure_render_reads_each_document_and_its_bucket_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
    from tcip_mcp.tools.vision_tools import render_failure_cases

    frames = _published_frames(tmp_path, image_dir(tmp_path, UNDATED_BUCKET))
    counts = _counting_reads(monkeypatch)

    result = render_failure_cases(
        tmp_path, str(tmp_path), "published", 10, **TRIAGE_WEIGHTS
    )

    assert "error" not in result, result
    assert len(result["case_images"]) == len(frames)
    assert len(counts) == 2 * len(frames) + 1, counts
    assert set(counts.values()) == {1}, counts


def test_each_whole_dataset_reader_scans_each_capture_once_in_the_one_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each whole-dataset reader lists the captures once and scans each once, inside one walk.

    The readers are the census, the image counts, the project summary and status and the plant
    mapping; a scan made outside ``image_utils.scan_captures`` is counted apart.
    """
    from tcip_mcp import dataset_layout
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.pipelines import image_utils
    from tcip_mcp.project_record import record_fields
    from tcip_mcp.tools.data_tools import _scan_dataset
    from tcip_mcp.tools.project_tools import inspect_project
    from tcip_web.routes.projects import _summarize
    from tests._mapping_fixtures import (
        geolocated_captures,
        mapping_over,
        write_plots_csv,
    )

    root = tmp_path / "ds"
    dates = ["2026-02-11", "2026-02-18"]
    geolocated_captures(root, dates)
    plants = write_plots_csv(tmp_path / "plants.csv")
    walked: Counter = Counter()
    outside: Counter = Counter()
    walking: list[bool] = []
    listings: list[str | Path] = []
    real_scan, real_list = (
        image_utils.scan_identities,
        dataset_layout.list_dates,
    )

    def counted_list(dataset_root: str | Path) -> list[str]:
        listings.append(dataset_root)
        return real_list(dataset_root)

    def counted_scan(directory: Path) -> dict[str, list[Path | BandGroupRef]]:
        (walked if walking else outside)[str(directory)] += 1
        return real_scan(directory)

    def walked_inside[**P, R](walk: Callable[P, R]) -> Callable[P, R]:
        def counted(*args: P.args, **kwargs: P.kwargs) -> R:
            walking.append(True)
            try:
                return walk(*args, **kwargs)
            finally:
                walking.pop()

        return counted

    monkeypatch.setattr(image_utils, "scan_identities", counted_scan)
    monkeypatch.setattr(dataset_layout, "list_dates", counted_list)
    monkeypatch.setattr(
        image_utils, "scan_captures", walked_inside(image_utils.scan_captures)
    )
    reads: list[Callable[[], object]] = [
        lambda: _scan_dataset(str(root)),
        lambda: image_utils.logical_image_counts(root),
        lambda: inspect_project(root),
        lambda: _summarize(root, record_fields(root)),
        lambda: mapping_over(root, [plants], tmp_path),
    ]
    for read in reads:
        walked.clear()
        outside.clear()
        listings.clear()
        read()
        assert len(listings) == 1, listings
        assert outside == Counter(), outside
        assert walked == Counter(
            {str(image_dir(root, date)): 1 for date in dates}
        ), walked


def test_a_detection_pass_reads_each_recorded_manifest_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A band-group detection pass scans its capture once, each manifest read once.

    The manifest is the one an earlier pass recorded for an explicit group.
    """
    import numpy as np
    import tifffile

    from tcip_mcp.dataset_layout import image_root
    from tcip_mcp.pipelines import image_utils
    from tcip_mcp.pipelines.data import band_groups

    images = image_root(tmp_path)
    images.mkdir()
    for name in ("cap_G.tif", "cap_R.tif"):
        tifffile.imwrite(str(images / name), np.zeros((4, 4), np.uint16))
    (manifest,) = band_groups.detect_and_write_band_groups(
        images, explicit_groups={"cap": {"G": "cap_G.tif", "R": "cap_R.tif"}}
    )["manifests"]
    scans = _counting_scans(monkeypatch)
    reads: Counter = Counter()
    real_read = band_groups.read_band_group_manifest

    def counted_read(path: Path) -> BandGroupRef:
        reads[str(path)] += 1
        return real_read(path)

    for module in (band_groups, image_utils):
        monkeypatch.setattr(module, "read_band_group_manifest", counted_read)

    band_groups.detect_and_write_band_groups(images)

    assert scans == Counter({str(images): 1})
    assert reads == Counter({manifest: 1})
