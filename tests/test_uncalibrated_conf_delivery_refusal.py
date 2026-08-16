"""A count CSV is a phenotype, so the conf behind it is either validated or declared as not.

A run with no per-dataset calibration ships whatever confidence its caller happened to pass. The
count door's only defence is the validity of the operating point the run recorded, so these drive
the whole path (a real raw run, the real CSV writer) and pin what reaches disk in each case.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.usefixtures("seed_catkin_trait_spec")

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

CALLER_PICKED_CONF = 0.37  # neither the platform default nor anything derived from data


class _CountStub:
    """A predictor returning a different detection count per image, so the CSV has real rows."""

    _COUNTS = {"row_a": 3, "row_b": 7}

    def __init__(self) -> None:
        self.model = SimpleNamespace(score_thresh=0.5, nms_thresh=0.5, detections_per_img=100)
        self.device = "cpu"
        self.score_threshold = 0.5
        self.train_tile_size = None
        self.train_overlap = None

    def predict_batch(self, paths, **kw):
        results = []
        for p in paths:
            n = self._COUNTS[Path(p).stem]
            results.append({"image": p, "width": 160, "height": 120,
                            "boxes": [[10, 10, 30, 30]] * n, "scores": [0.9] * n,
                            "labels": [1] * n, "count": n})
        return results


def _images_dir(tmp_path):
    from PIL import Image

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for stem in _CountStub._COUNTS:
        Image.new("RGB", (160, 120), color=(60, 60, 60)).save(images_dir / f"{stem}.png")
    return images_dir


def _held_out_bundle():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tests._dense_op_fixtures import dense_records

    n_images, objects_per_image = 20, 80
    miss, fp = [0] * n_images, [1] * n_images
    inputs = {
        "dataset_hash": "H",
        "calibration_records": dense_records(
            n_images=n_images, objects_per_image=objects_per_image, id_prefix="c",
            miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05),
        "holdout_records": dense_records(
            n_images=n_images, objects_per_image=objects_per_image, id_prefix="h", shift=5.0,
            miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05),
        "tiled": False, "staged_conf_floor": 0.01,
    }
    return resolve_operating_point("catkin", experiment_id=None, **inputs), inputs


def _prepare(tmp_path, monkeypatch):
    import tcip_mcp.pipelines.inference.predictor as predictor_mod

    monkeypatch.setattr(predictor_mod, "build_predictor", lambda **kw: _CountStub())
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    return str(ckpt), _images_dir(tmp_path)


def test_a_caller_chosen_conf_never_reaches_a_written_count_csv(tmp_path, monkeypatch):
    """No calibration means no validated operating point, so the door refuses and writes nothing,
    naming the conf it refused rather than reporting a bare number."""
    import tcip_mcp.tools.inference_tools as itools
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE

    ckpt, images_dir = _prepare(tmp_path, monkeypatch)
    out_csv = tmp_path / "counts.csv"

    r = itools.tabulate_counts(ckpt, str(images_dir), str(out_csv),
                               conf_threshold=CALLER_PICKED_CONF, device="cpu", tile=False)

    assert "error" in r
    assert r["operating_point"]["conf"]["value"] == pytest.approx(CALLER_PICKED_CONF)
    assert r["operating_point"]["conf"]["validated_against"] == VALIDATED_FALSE
    assert r["operating_point_validated"] == VALIDATED_FALSE
    assert r["validated"] is False
    assert not out_csv.exists()


def test_an_acknowledged_uncalibrated_count_is_written_flagged_rather_than_refused(
    tmp_path, monkeypatch,
):
    """The escape hatch stays open and honest: the same caller-chosen conf delivers a CSV whose
    every row carries the unvalidated stamp."""
    import tcip_mcp.tools.inference_tools as itools
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE

    ckpt, images_dir = _prepare(tmp_path, monkeypatch)
    out_csv = tmp_path / "counts.csv"

    r = itools.tabulate_counts(ckpt, str(images_dir), str(out_csv),
                               conf_threshold=CALLER_PICKED_CONF, device="cpu", tile=False,
                               acknowledge_unvalidated=True)

    assert "error" not in r, r
    assert r["operating_point_validated"] == VALIDATED_FALSE
    rows = out_csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1 + len(_CountStub._COUNTS)  # header plus one row per image
    assert all(VALIDATED_FALSE in row for row in rows[1:])


def test_a_calibrated_conf_delivers_the_count_csv_untouched(tmp_path, monkeypatch):
    """The refusal is about the missing validation, never about the door: a conf resolved against a
    held-out reference writes the CSV with no acknowledgement and stamps its real reference.

    The predictions it counted are persisted and stamped, and the CSV's own validity is read back
    off that stamp, so the number in the file rests on an artifact anyone can re-read.
    """
    import tcip_mcp.tools.inference_tools as itools
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_HELD_OUT, read_operating_point_sidecar, verify_stamp_binding,
    )

    ckpt, images_dir = _prepare(tmp_path, monkeypatch)
    bundle, inputs = _held_out_bundle()
    evidence = {"resolver": "resolve_operating_point", "inputs": inputs,
                "reference_inputs": {"label_dirs": {"calibration": str(images_dir)}}}
    monkeypatch.setattr(itools, "_calibrate_operating_point",
                        lambda *a, **k: (bundle, "H", 0, evidence))
    out_csv = tmp_path / "counts.csv"
    bucket = tmp_path / "predictions" / "baseline" / "2026-01-01"

    r = itools.tabulate_counts(ckpt, str(images_dir), str(out_csv), device="cpu", tile=False,
                               trait="catkin", calibration_labels_dir=str(images_dir),
                               predictions_dir=str(bucket))

    assert "error" not in r, r
    assert r["predictions_dir"] == str(bucket)
    assert r["operating_point_validated"] == VALIDATED_HELD_OUT
    assert r["total_detections"] == sum(_CountStub._COUNTS.values())
    assert VALIDATED_HELD_OUT in out_csv.read_text(encoding="utf-8")

    for stem in _CountStub._COUNTS:
        assert (bucket / f"{stem}.json").is_file()
    stamp = read_operating_point_sidecar(bucket)
    assert stamp["validated"] is True
    assert verify_stamp_binding(stamp, bucket, document="operating_point", trait="catkin").ok
