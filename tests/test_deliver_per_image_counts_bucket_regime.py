"""``deliver_per_image_counts`` over a published bucket: the bucket's own record states whose
predictions they are and what they count, and the one gate reads the assessment it was published
under. A directory that is no bucket, a mosaic bucket, a bucket counting another subject and an
unassessed bucket each refuse; an assessed bucket delivers validated, reading it imports no torch.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from tests import _trait_fixtures as fx

pytest.importorskip("torch")

SCOPE = {"subject": fx.COUNT_SUBJECT, "attribute": None, "id_map": {fx.COUNT_SUBJECT: 0}}


@pytest.fixture(autouse=True)
def _recorded_meaning(tmp_path):
    """The count trait's per-image count of :data:`SCOPE`'s subject, confirmed in the project."""
    fx.seed_confirmed_count(tmp_path, measured_subject=fx.COUNT_SUBJECT)


def _deliver(project: Path, bucket: Path, **overrides) -> dict:
    from tcip_mcp.tools.inference_tools import deliver_per_image_counts

    return deliver_per_image_counts(project, predictions_dir=str(bucket),
                                    output_path=str(project / "out" / "counts.csv"),
                                    trait=overrides.get("trait", fx.COUNT_TRAIT))


def _bucket(project: Path, scope: dict = SCOPE, **published_kw) -> Path:
    from tests._chain_fixtures import predicted, published

    name = scope["subject"] or "raster"
    bucket = project / "ds" / "predictions" / name / "2026-01-01"
    names = [scope["subject"]] * 2 if scope["subject"] else []
    return published(project, bucket, [predicted("a", names, scope["id_map"] or {})],
                     scope=scope, **published_kw).path


def test_a_directory_holding_no_bucket_record_refuses(tmp_path):
    """A directory of label documents with no record (a ground-truth tree) is refused, never
    counted: nothing states whose predictions they are."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    labels = tmp_path / "ds" / "annotations" / "2026-01-01"
    labels.mkdir(parents=True)
    json_io.write_annotations(str(labels / "a.json"),
                              [Annotation(subject=fx.COUNT_SUBJECT, geometry=BBox(1, 1, 5, 5))],
                              32, 32)

    res = _deliver(tmp_path, labels)

    assert "bucket.json" in res["error"]
    assert not (tmp_path / "out" / "counts.csv").exists()


def test_a_mosaic_bucket_refuses_naming_the_per_plant_door(tmp_path):
    import numpy as np
    import tifffile

    raster = tmp_path / "mosaic.tif"
    tifffile.imwrite(str(raster), np.zeros((32, 32, 3), dtype=np.uint8))
    from tests._chain_fixtures import predicted, published

    bucket = published(tmp_path, tmp_path / "ds" / "predictions" / "mosaic" / "run",
                       [{**predicted("mosaic", [fx.COUNT_SUBJECT], SCOPE["id_map"]),
                         "image": str(raster)}], scope=SCOPE, raster_path=raster).path

    res = _deliver(tmp_path, bucket)

    assert "deliver_orthomosaic_plant_counts" in res["error"]


def test_a_bucket_counting_another_subject_refuses_and_its_own_subject_is_admitted_to_the_gate(
    tmp_path,
):
    """The measured-subject check reads the bucket's recorded map: a bucket counting another
    subject refuses before the gate, while the confirmed subject's bucket reaches the gate (and,
    unassessed, refuses there instead)."""
    other = _bucket(tmp_path, {"subject": "leaf", "attribute": None, "id_map": {"leaf": 0}})
    own = _bucket(tmp_path)

    refused = _deliver(tmp_path, other)
    reached = _deliver(tmp_path, own)

    assert fx.COUNT_SUBJECT in refused["error"] and "leaf" in refused["error"]
    assert "no assessment answers" in reached["error"]


def test_an_unknown_trait_answers_an_error_dict(tmp_path):
    res = _deliver(tmp_path, _bucket(tmp_path), trait="no-such-trait")

    assert "no-such-trait" in res["error"]


def test_an_assessed_bucket_delivers_validated_counts_naming_the_producer(tmp_path):
    from tcip_mcp.delivery import read_delivery_events
    from tests._chain_fixtures import run_the_chain

    chain = run_the_chain(tmp_path, experiment_id="exp-counts")

    res = _deliver(tmp_path, chain.bucket)

    assert "error" not in res, res
    assert res["validated"] is True
    assert res["producer"] == chain.assessment["producer"]
    with open(res["csv_path"], newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == res["image_count"] == len(list(chain.images_dir.iterdir()))
    assert {r["validated"] for r in rows} == {"True"}
    (event,) = read_delivery_events(tmp_path)
    assert event.door == "deliver_per_image_counts"
    assert event.population == sorted(p.stem for p in chain.images_dir.iterdir())


def test_reading_a_published_bucket_to_a_delivered_csv_imports_no_torch(tmp_path):
    """Delivering from a bucket needs no GPU, no predictor and no checkpoint: the delivery runs in
    a subprocess with torch blocked from importing at all."""
    from tests._chain_fixtures import run_the_chain

    chain = run_the_chain(tmp_path, experiment_id="exp-no-torch")
    out_csv = tmp_path / "o.csv"

    script = f"""
import sys

class _BlockTorch:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "torchvision"):
            raise ImportError(f"torch blocked for this check: {{name}}")
        return None

sys.meta_path.insert(0, _BlockTorch())
from tcip_store.binding import bind_default
bind_default()
import tcip_mcp.tools.inference_tools as itools
from pathlib import Path
r = itools.deliver_per_image_counts(Path({str(tmp_path)!r}), predictions_dir={str(chain.bucket)!r},
                                    output_path={str(out_csv)!r}, trait={fx.COUNT_TRAIT!r})
assert "error" not in r, r
assert "torch" not in sys.modules, "the delivery pulled torch into sys.modules"
print("ok")
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
    assert out_csv.exists()
