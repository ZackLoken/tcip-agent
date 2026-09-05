"""The writer-side scope rail on ``operating_point.json``: every fresh stamp must carry both
``subject`` and ``attribute``, and when it also carries an ``id_map`` the pair must agree with it
through ``scope_consistent_with_map``. Exercised directly at ``write_sidecar``/``update_sidecar``,
the one choke point every producer and every hand merge passes through; the conform script's own
rule 3 calls the identical predicate rather than holding the rule a second time
(``test_conform_classified_predictions.py::test_a_stated_detector_scope_refused_over_a_bucket_whose_map_is_keyed_by_values``
pins that call site).

A stamp missing the pair entirely has no live producer left to build it through
(``operating_point_stamp`` requires both keywords with no default), so it is seeded by writing a
hand-built dict straight through the store, the same posture
``test_review_negative_coverage.py`` and ``test_conform_classified_predictions.py`` already take
for this shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tcip_store as ts

from tcip_mcp.pipelines.resolution import (
    operating_point_stamp,
    read_operating_point_sidecar,
    sidecar_key,
    update_sidecar,
    write_sidecar,
)

SUBJECT = "bud"
ATTRIBUTE = "bud_opening"
VALUE_MAP = {"open": 0, "closed": 1}
DETECTOR_MAP = {SUBJECT: 0}


def _stamp(**overrides) -> dict:
    fields = dict(
        validated=False, validated_by=None, tile_size_validated=None, shippable_issues=[],
        id_map=None, subject=SUBJECT, attribute=None, trait=SUBJECT, dataset_hash="h",
        checkpoint="m", checkpoint_sha256="f" * 64, experiment_id=None, images_dir=None,
        raster_path=None, produced_at="2026-01-01T00:00:00+00:00",
    )
    fields.update(overrides)
    return operating_point_stamp({"conf": {"value": 0.5}}, **fields)


def test_write_sidecar_refuses_a_fresh_stamp_missing_the_pair_entirely(tmp_path: Path) -> None:
    bucket = tmp_path / "bucket"
    neither_key = {k: v for k, v in _stamp().items() if k not in ("subject", "attribute")}

    with pytest.raises(ValueError, match="carries no subject/attribute pair"):
        write_sidecar(bucket, neither_key)

    assert read_operating_point_sidecar(bucket) is None


def test_write_sidecar_refuses_a_detector_pair_over_a_value_keyed_map(tmp_path: Path) -> None:
    bucket = tmp_path / "bucket"
    stamp = _stamp(subject=SUBJECT, attribute=None, id_map=VALUE_MAP)

    with pytest.raises(ValueError, match="this looks like a classified bucket"):
        write_sidecar(bucket, stamp)

    assert read_operating_point_sidecar(bucket) is None


def test_write_sidecar_refuses_a_classified_pair_over_a_map_keyed_by_the_subject_alone(
    tmp_path: Path,
) -> None:
    bucket = tmp_path / "bucket"
    stamp = _stamp(subject=SUBJECT, attribute=ATTRIBUTE, id_map=DETECTOR_MAP, trait=ATTRIBUTE)

    with pytest.raises(ValueError, match="the shape a detector run records"):
        write_sidecar(bucket, stamp)

    assert read_operating_point_sidecar(bucket) is None


def test_write_sidecar_refuses_a_detector_pair_over_an_empty_map(tmp_path: Path) -> None:
    """An empty ``id_map`` (``{}``, distinct from ``None``, absent) names no vocabulary a detector
    pair could agree or disagree with: refused by name, never read as "not keyed by the subject"
    with an empty list printed."""
    bucket = tmp_path / "bucket"
    stamp = _stamp(subject=SUBJECT, attribute=None, id_map={})

    with pytest.raises(ValueError, match="id_map is empty"):
        write_sidecar(bucket, stamp)

    assert read_operating_point_sidecar(bucket) is None


def test_write_sidecar_refuses_a_classified_pair_over_an_empty_map(tmp_path: Path) -> None:
    """The same empty-map refusal on the classified side: an empty map is never silently admitted
    as consistent."""
    bucket = tmp_path / "bucket"
    stamp = _stamp(subject=SUBJECT, attribute=ATTRIBUTE, id_map={}, trait=ATTRIBUTE)

    with pytest.raises(ValueError, match="id_map is empty"):
        write_sidecar(bucket, stamp)

    assert read_operating_point_sidecar(bucket) is None


def test_write_sidecar_admits_a_detector_pair_over_its_own_map(tmp_path: Path) -> None:
    bucket = tmp_path / "bucket"
    stamp = _stamp(subject=SUBJECT, attribute=None, id_map=DETECTOR_MAP)

    write_sidecar(bucket, stamp)

    assert read_operating_point_sidecar(bucket)["subject"] == SUBJECT


def test_write_sidecar_admits_a_classified_pair_over_a_multi_key_map(tmp_path: Path) -> None:
    bucket = tmp_path / "bucket"
    stamp = _stamp(subject=SUBJECT, attribute=ATTRIBUTE, id_map=VALUE_MAP, trait=ATTRIBUTE)

    write_sidecar(bucket, stamp)

    stored = read_operating_point_sidecar(bucket)
    assert (stored["subject"], stored["attribute"]) == (SUBJECT, ATTRIBUTE)


def test_write_sidecar_admits_a_pair_with_no_recorded_map_at_all(tmp_path: Path) -> None:
    """``scope_consistent_with_map`` is only checked when the body also carries an ``id_map``: a
    bucket recording none makes no claim the rail can contradict."""
    bucket = tmp_path / "bucket"
    stamp = _stamp(subject=SUBJECT, attribute=ATTRIBUTE, id_map=None, trait=ATTRIBUTE)

    write_sidecar(bucket, stamp)

    assert read_operating_point_sidecar(bucket)["id_map"] is None


def test_update_sidecar_refuses_a_promotion_over_a_stored_stamp_with_no_pair_naming_the_conform_script(
    tmp_path: Path,
) -> None:
    bucket = tmp_path / "bucket"
    neither_key = {k: v for k, v in _stamp().items() if k not in ("subject", "attribute")}
    key = sidecar_key(bucket)
    ts.replace(key, neither_key, expect=ts.Version.ABSENT)

    with pytest.raises(ValueError, match="conform_classified_predictions.py"):
        update_sidecar(bucket, lambda stored: {**stored, "validated": True})

    assert read_operating_point_sidecar(bucket)["validated"] is False


def test_update_sidecar_admits_a_merge_that_only_rewrites_an_existing_keys_value(
    tmp_path: Path,
) -> None:
    """A merge that introduces no new top-level key is admitted against a stamp whose own pair
    already agrees with its recorded map, the ordinary shape a promotion or a calibrator merge
    takes over a bucket a live producer already stamped."""
    bucket = tmp_path / "bucket"
    write_sidecar(bucket, _stamp(subject=SUBJECT, attribute=None, id_map=DETECTOR_MAP))

    wrote = update_sidecar(bucket, lambda stored: {**stored, "checkpoint": "best-2"})

    assert wrote is True
    assert read_operating_point_sidecar(bucket)["checkpoint"] == "best-2"
