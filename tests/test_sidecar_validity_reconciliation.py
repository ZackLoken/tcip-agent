"""Reconciling a delivery's validity from what each prediction bucket's sidecar actually records.

Every reconciler here reads two independent facts off a bucket: the sidecar's own bundle-level
``validated`` flag, and the reference the named param recorded underneath it. A fixture that derives
one of those from the other cannot tell them apart, so the sidecars below set them separately, and
a delivery is assembled from buckets that disagree with each other rather than from one uniform
bucket repeated.
"""

from __future__ import annotations

from pathlib import Path

from tcip_mcp.pipelines.resolution import (
    VALIDATED_FALSE,
    VALIDATED_HELD_OUT,
    VALIDATED_PHYSICAL_MEASUREMENT,
    VALIDATED_REVIEW_CONFIRMED,
    VALIDATED_SAME_MOSAIC_IDENTITY,
    reconcile_classifier_validity,
    reconcile_operating_point_validity,
    reconcile_scale_validity,
)


def _sidecar(bucket: Path, filename: str, param_key: str, *, bundle_flag: bool,
             recorded_reference: str | None, **param_fields: object) -> str:
    """One bucket's sidecar, with its bundle-level flag and its param's recorded reference set
    independently of each other.

    Written through the seam, bypassing write_sidecar's own claim check (the stamp below asserts
    no real validated_by), so the bucket genuinely has a sidecar rather than merely a file that
    happens to sit where one would.
    """
    import tcip_store

    from tcip_mcp.pipelines.resolution import sidecar_key

    bucket.mkdir(parents=True, exist_ok=True)
    param: dict[str, object] = {"requires_validation": True,
                                "validated_against": recorded_reference}
    param.update(param_fields)
    stamp = {"validated": bundle_flag, "operating_point": {param_key: param}}
    document = filename.removesuffix(".json")
    key = sidecar_key(bucket, document)
    with tcip_store.transaction(key) as txn:
        txn.write(key, stamp)
    return str(bucket)


def _count_bucket(bucket: Path, *, bundle_flag: bool, recorded_reference: str | None,
                  conf: float) -> str:
    return _sidecar(bucket, "operating_point.json", "conf", bundle_flag=bundle_flag,
                    recorded_reference=recorded_reference, value=conf, validation_kind="annotations")


def _bound_sidecar(bucket: Path, filename: str, param_key: str, *, recorded_reference: str,
                   document: str, dataset_root: Path, experiment_id: str,
                   **param_fields: object) -> str:
    """A sidecar genuinely answered for by a validation record, the same shape :func:`_sidecar`
    writes but with a real ``validated_by`` merged in, for tests whose subject is a stamp that did
    clear rather than one that was merely claimed."""
    from tests._binding_fixtures import file_validation_record, write_prediction

    from tcip_mcp.pipelines.resolution import write_sidecar

    param: dict[str, object] = {"requires_validation": True, "validated_against": recorded_reference}
    param.update(param_fields)
    stamp = {"validated": True, "trait": "bud_opening", "operating_point": {param_key: param},
             "subject": "bud", "attribute": None}
    pred_dirs: list[Path] = []
    if document == "operating_point":
        write_prediction(bucket, "img_a")
        pred_dirs = [bucket]
    bound = file_validation_record(stamp, document=document, dataset_root=dataset_root,
                                   pred_dirs=pred_dirs, experiment_id=experiment_id)
    write_sidecar(bucket, bound, document)
    return str(bucket)


# --- a bucket already stamped unvalidated never reads back as validated ---

def test_a_count_bucket_stamped_unvalidated_never_reads_back_its_recorded_reference(tmp_path):
    """A bucket whose bundle-level flag says the delivery door refused it is unvalidated whatever
    its conf entry still records: the recorded reference is not a second, independent claim that can
    reinstate the bucket, and the threshold it names must not travel out of it either."""
    d = _count_bucket(tmp_path / "b1", bundle_flag=False,
                      recorded_reference=VALIDATED_HELD_OUT, conf=0.62)
    r = reconcile_operating_point_validity([d], trait="bud_opening", asserted=VALIDATED_HELD_OUT)
    assert r["validated"] == VALIDATED_FALSE
    assert r["per_bucket"] == {d: VALIDATED_FALSE}
    assert r["unvalidated_buckets"] == [d]
    assert r["on_disk_validated"] is False
    assert r["conf"] is None


def test_a_classifier_stamp_flagged_unvalidated_never_reads_back_its_recorded_reference(tmp_path):
    """The same two-part read on the classifier dimension's own sidecar file and param key."""
    d = _sidecar(tmp_path / "b1", "classifier_operating_point.json", "classifier",
                 bundle_flag=False, recorded_reference=VALIDATED_REVIEW_CONFIRMED, value=0.41)
    r = reconcile_classifier_validity([d], asserted=VALIDATED_REVIEW_CONFIRMED)
    assert r["validated"] == VALIDATED_FALSE
    assert r["unvalidated_buckets"] == [d]
    assert r["on_disk_validated"] is False


def test_a_scale_stamp_flagged_unvalidated_never_reads_back_its_recorded_reference(tmp_path):
    """The physical dimension reads its own sidecar the same way: a scale whose stamp says the
    check did not clear stays ungrounded even though a physical-measurement reference is recorded
    beside it."""
    d = _sidecar(tmp_path / "b1", "resolve_scale.json", "scale", bundle_flag=False,
                 recorded_reference=VALIDATED_PHYSICAL_MEASUREMENT, value=0.037, unit="mm")
    r = reconcile_scale_validity([d], unit="mm", trait="bud_opening", images_dir="unused",
                                 asserted=VALIDATED_PHYSICAL_MEASUREMENT)
    assert r["operative"] is True
    assert r["validated"] == VALIDATED_FALSE
    assert r["unvalidated_buckets"] == [d]


def test_a_classifier_stamp_that_did_clear_still_reports_its_reference(tmp_path):
    """The rail must admit valid work: a genuinely persisted classifier calibration reports the
    reference it earned, so the checks above refuse a shape rather than refusing everything."""
    d = _bound_sidecar(tmp_path / "b1", "classifier_operating_point.json", "classifier",
                       recorded_reference=VALIDATED_REVIEW_CONFIRMED,
                       document="classifier_operating_point", dataset_root=tmp_path,
                       experiment_id="exp-b1", value=0.41)
    r = reconcile_classifier_validity([d])
    assert r["validated"] == VALIDATED_REVIEW_CONFIRMED
    assert r["on_disk_validated"] is True
    assert r["unvalidated_buckets"] == []


# --- a bucket with nothing on disk floors the buckets beside it ---

def test_a_bucket_with_no_sidecar_floors_a_curve_assembled_beside_a_validated_one(tmp_path):
    """A delivery spanning several buckets is only as grounded as its least-grounded bucket: one
    bucket that never had an operating point written for it floors the whole curve, rather than the
    validated bucket beside it reporting its reference for both."""
    root = tmp_path / "ds"
    good = _bound_sidecar(root / "predictions" / "b1", "operating_point.json", "conf",
                          recorded_reference=VALIDATED_HELD_OUT, document="operating_point",
                          dataset_root=root, experiment_id="exp-b1", value=0.62,
                          validation_kind="annotations")
    absent = tmp_path / "b2"
    absent.mkdir()
    r = reconcile_operating_point_validity([good, str(absent)], trait="bud_opening")
    assert r["validated"] == VALIDATED_FALSE
    assert r["on_disk_validated"] is False
    assert r["missing_sidecars"] == [str(absent)]
    assert r["per_bucket"] == {good: VALIDATED_HELD_OUT, str(absent): VALIDATED_FALSE}


def test_a_missing_classifier_stamp_floors_a_dimension_two_other_buckets_cleared(tmp_path):
    """The same flooring on the classifier dimension, across buckets that cleared against different
    references: neither earned reference travels when a third bucket has no stamp at all."""
    held_out = _sidecar(tmp_path / "b1", "classifier_operating_point.json", "classifier",
                        bundle_flag=True, recorded_reference=VALIDATED_HELD_OUT, value=0.55)
    reviewed = _sidecar(tmp_path / "b2", "classifier_operating_point.json", "classifier",
                        bundle_flag=True, recorded_reference=VALIDATED_REVIEW_CONFIRMED, value=0.31)
    absent = tmp_path / "b3"
    absent.mkdir()
    r = reconcile_classifier_validity([held_out, reviewed, str(absent)])
    assert r["validated"] == VALIDATED_FALSE
    assert r["on_disk_validated"] is False
    assert r["missing_sidecars"] == [str(absent)]


def test_buckets_that_ran_at_different_thresholds_report_no_single_operating_point(tmp_path):
    """A reconciled curve reports one conf only when every bucket agrees on it; buckets produced at
    different thresholds have no single operating point to report, and none of theirs is picked."""
    root = tmp_path / "ds"
    a = _bound_sidecar(root / "predictions" / "b1", "operating_point.json", "conf",
                       recorded_reference=VALIDATED_HELD_OUT, document="operating_point",
                       dataset_root=root, experiment_id="exp-a", value=0.62,
                       validation_kind="annotations")
    b = _bound_sidecar(root / "predictions" / "b2", "operating_point.json", "conf",
                       recorded_reference=VALIDATED_REVIEW_CONFIRMED, document="operating_point",
                       dataset_root=root, experiment_id="exp-b", value=0.41,
                       validation_kind="annotations")
    mixed = reconcile_operating_point_validity([a, b], trait="bud_opening")
    assert mixed["validated"] == VALIDATED_HELD_OUT
    assert mixed["conf"] is None

    c = _bound_sidecar(root / "predictions" / "b3", "operating_point.json", "conf",
                       recorded_reference=VALIDATED_REVIEW_CONFIRMED, document="operating_point",
                       dataset_root=root, experiment_id="exp-c", value=0.41,
                       validation_kind="annotations")
    agreed = reconcile_operating_point_validity([b, c], trait="bud_opening")
    assert agreed["validated"] == VALIDATED_REVIEW_CONFIRMED
    assert agreed["conf"] == 0.41


# --- a reference from another dimension never clears this one ---

def test_a_raster_identity_reference_never_clears_the_count_dimension(tmp_path):
    """A raster export's content-identity claim is a shippable reference for its own dimension and
    for nothing else: recorded as the conf entry's reference it must floor the count dimension, not
    clear it because it happens to be a real reference somewhere."""
    d = _count_bucket(tmp_path / "b1", bundle_flag=True,
                      recorded_reference=VALIDATED_SAME_MOSAIC_IDENTITY, conf=0.62)
    r = reconcile_operating_point_validity([d], trait="bud_opening")
    assert r["validated"] == VALIDATED_FALSE
    assert r["per_bucket"] == {d: VALIDATED_FALSE}
    assert r["unvalidated_buckets"] == [d]
    assert r["conf"] is None


def test_an_annotations_reference_never_clears_a_bucket_scale_even_when_the_stamp_cleared(tmp_path):
    """A physical scale is checked against a known physical dimension, so annotation-based evidence
    cannot ground it however confidently the stamp itself reports success."""
    d = _sidecar(tmp_path / "b1", "resolve_scale.json", "scale", bundle_flag=True,
                 recorded_reference=VALIDATED_HELD_OUT, value=0.037, unit="mm")
    r = reconcile_scale_validity([d], unit="mm", trait="bud_opening", images_dir="unused")
    assert r["validated"] == VALIDATED_FALSE
    assert r["per_bucket"] == {d: VALIDATED_FALSE}
    assert r["unvalidated_buckets"] == [d]
