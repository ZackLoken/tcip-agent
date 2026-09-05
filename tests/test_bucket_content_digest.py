"""bucket_content_digest: the content identity a delivery recomputes over the buckets it reads.

Behaviour, not surface: any change to a bucket's prediction files that happened before the delivery
began is detected, a provenance stamp written beside them is not a change to what was counted, and
the memo that keeps one delivery from hashing a bucket twice never survives into the next call.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tcip_annotation.json_io import SIDECAR_FILENAMES
from tcip_mcp import prediction_buckets
from tcip_mcp.prediction_buckets import bucket_content_digest


def _write_prediction(bucket: Path, stem: str, score: float = 0.9) -> Path:
    bucket.mkdir(parents=True, exist_ok=True)
    out = bucket / f"{stem}.json"
    out.write_text(
        json.dumps({
            "image": f"{stem}.png",
            "width": 100,
            "height": 100,
            "annotations": [{"subject": "bud", "bbox": [10, 10, 30, 30], "score": score}],
        }),
        encoding="utf-8",
    )
    return out


def test_replacing_a_prediction_file_changes_the_digest(tmp_path):
    """A file swapped for different bytes under the same name is the tampering case the digest
    exists for: the stem set is identical, so only the content can tell the two buckets apart."""
    bucket = tmp_path / "predictions" / "baseline"
    _write_prediction(bucket, "img_0001")
    before = bucket_content_digest(bucket)

    _write_prediction(bucket, "img_0001", score=0.2)

    assert bucket_content_digest(bucket) != before


def test_adding_a_prediction_file_changes_the_digest(tmp_path):
    """The reason the digest is taken over the enumerated stem set rather than a published per-stem
    map: a map checked entry by entry covers replacement and deletion, but an added file is what the
    downstream count enumerates and would slip past it."""
    bucket = tmp_path / "predictions" / "baseline"
    _write_prediction(bucket, "img_0001")
    before = bucket_content_digest(bucket)

    _write_prediction(bucket, "img_0002")

    assert bucket_content_digest(bucket) != before


def test_deleting_a_prediction_file_changes_the_digest(tmp_path):
    bucket = tmp_path / "predictions" / "baseline"
    _write_prediction(bucket, "img_0001")
    _write_prediction(bucket, "img_0002")
    before = bucket_content_digest(bucket)

    (bucket / "img_0002.json").unlink()

    assert bucket_content_digest(bucket) != before


@pytest.mark.parametrize("sidecar", sorted(SIDECAR_FILENAMES))
def test_a_provenance_stamp_beside_the_predictions_leaves_the_digest_alone(tmp_path, sidecar):
    """A stamp is provenance about the bucket, not one of the records counted from it. A digest that
    moved when a stamp was written would be vouching for itself, and stamping a dimension after a
    delivery earned its record would floor the claim the stamp describes."""
    bucket = tmp_path / "predictions" / "baseline"
    _write_prediction(bucket, "img_0001")
    before = bucket_content_digest(bucket)

    (bucket / sidecar).write_text(json.dumps({"conf": {"value": 0.5}}), encoding="utf-8")
    assert bucket_content_digest(bucket) == before

    (bucket / sidecar).write_text(json.dumps({"conf": {"value": 0.7}}), encoding="utf-8")
    assert bucket_content_digest(bucket) == before


def test_a_same_length_replacement_with_a_restored_timestamp_is_detected(tmp_path):
    """Two separate calls hash the files again rather than trusting size and timestamp, which an
    actor replacing bytes in place can restore. This is why there is no cross-call cache."""
    bucket = tmp_path / "predictions" / "baseline"
    out = _write_prediction(bucket, "img_0001", score=0.9)
    stat = os.stat(out)
    before = bucket_content_digest(bucket)

    _write_prediction(bucket, "img_0001", score=0.1)
    os.utime(out, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    assert os.stat(out).st_size == stat.st_size
    assert os.stat(out).st_mtime_ns == stat.st_mtime_ns
    assert bucket_content_digest(bucket) != before


def test_a_shared_memo_hashes_each_directory_once_and_a_call_without_it_reads_again(
    tmp_path, monkeypatch
):
    """The memo spans one delivery: a bucket named by several rows of the same delivery is read
    once, and the next delivery starts from nothing so it sees whatever the files hold then."""
    detect = tmp_path / "predictions" / "baseline" / "detect"
    segment = tmp_path / "predictions" / "baseline" / "segment"
    _write_prediction(detect, "img_0001")
    _write_prediction(segment, "img_0001", score=0.4)

    hashed: list[str] = []
    real = prediction_buckets.dataset_hash

    def counting(labels_dir, stems=None):
        hashed.append(str(Path(labels_dir).resolve()))
        return real(labels_dir, stems=stems)

    monkeypatch.setattr(prediction_buckets, "dataset_hash", counting)

    memo: dict[str, str] = {}
    first = bucket_content_digest(detect, segment, memo=memo)
    assert bucket_content_digest(detect, segment, memo=memo) == first
    assert sorted(hashed) == sorted([str(detect.resolve()), str(segment.resolve())])

    hashed.clear()
    assert bucket_content_digest(detect, segment) == first
    assert len(hashed) == 2


def test_the_combined_digest_does_not_depend_on_the_argument_order(tmp_path):
    """A bucket's task dirs reach this from several callers, and no two of them are obliged to name
    them in the same order."""
    detect = tmp_path / "predictions" / "baseline" / "detect"
    segment = tmp_path / "predictions" / "baseline" / "segment"
    _write_prediction(detect, "img_0001")
    _write_prediction(segment, "img_0001", score=0.4)

    combined = bucket_content_digest(detect, segment)
    assert bucket_content_digest(segment, detect) == combined
    assert combined != bucket_content_digest(detect)
    assert combined != bucket_content_digest(segment)
