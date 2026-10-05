"""``deliver_per_image_counts`` over a published bucket: the bucket's own record states whose
predictions they are and what they count, and the one gate reads the assessment it was published
under. A name that is no bucket, a mosaic bucket, a bucket counting another subject and an
unassessed bucket each refuse; an assessed bucket delivers validated, reading it imports no torch.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest
from tcip_store.file_backend import is_bookkeeping

from tests import _trait_fixtures as fx

pytest.importorskip("torch")

SCOPE = {"subject": fx.COUNT_SUBJECT}


def _images(images_dir: Path) -> list[Path]:
    """The capture's image files, without the lock sidecars beside them."""
    return sorted(p for p in images_dir.iterdir() if not is_bookkeeping(p.name))


@pytest.fixture(autouse=True)
def _recorded_meaning(tmp_path):
    """The count trait's per-image count of :data:`SCOPE`'s subject, confirmed in the project."""
    fx.seed_confirmed_count(tmp_path, measured_subject=fx.COUNT_SUBJECT)


def _deliver(project: Path, root: Path, bucket: str, **overrides) -> dict:
    from tcip_mcp.tools.inference_tools import deliver_per_image_counts

    return deliver_per_image_counts(project, str(root), bucket,
                                    str(project / "out" / "counts.csv"),
                                    trait=overrides.get("trait", fx.COUNT_TRAIT))


def _bucket(project: Path, scope: dict = SCOPE, **published_kw) -> str:
    from tests._chain_fixtures import predicted, published

    name = f"{scope['subject'] or 'raster'}/2026-01-01"
    names = [scope["subject"]] * 2 if scope["subject"] else []
    image = project / "ds" / "images" / "2026-01-01" / "a.png"
    return published(project, name, [predicted(image, names)], scope=scope, **published_kw).name


def test_a_name_no_bucket_record_answers_for_refuses(tmp_path):
    """A name no bucket is published under is refused, never counted: nothing states whose
    predictions they would be."""
    res = _deliver(tmp_path, tmp_path / "ds", "nothing/2026-01-01")

    assert "no bucket 'nothing/2026-01-01' is published" in res["error"]
    assert not (tmp_path / "out" / "counts.csv").exists()


def test_a_mosaic_bucket_refuses_naming_the_per_plant_door(tmp_path):
    import numpy as np
    import tifffile

    raster = tmp_path / "ds" / "images" / "undated" / "mosaic.tif"
    raster.parent.mkdir(parents=True)
    tifffile.imwrite(str(raster), np.zeros((32, 32, 3), dtype=np.uint8))
    from tests._chain_fixtures import predicted, published

    bucket = published(tmp_path, "mosaic/run", [predicted(raster, [fx.COUNT_SUBJECT])],
                       scope=SCOPE, raster_path=raster).name

    res = _deliver(tmp_path, tmp_path / "ds", bucket)

    assert "deliver_orthomosaic_plant_counts" in res["error"]


def test_a_bucket_counting_another_subject_refuses_and_its_own_subject_is_admitted_to_the_gate(
    tmp_path,
):
    """The measured-subject check reads the bucket's recorded subject: a bucket counting another
    subject refuses before the gate, while the confirmed subject's bucket reaches the gate (and,
    unassessed, refuses there instead)."""
    other = _bucket(tmp_path, {"subject": "leaf"})
    own = _bucket(tmp_path)

    refused = _deliver(tmp_path, tmp_path / "ds", other)
    reached = _deliver(tmp_path, tmp_path / "ds", own)

    assert fx.COUNT_SUBJECT in refused["error"] and "leaf" in refused["error"]
    assert "no assessment answers" in reached["error"]


def test_an_unknown_trait_answers_an_error_dict(tmp_path):
    res = _deliver(tmp_path, tmp_path / "ds", _bucket(tmp_path), trait="no-such-trait")

    assert "no-such-trait" in res["error"]


def test_an_assessed_bucket_delivers_validated_counts_naming_the_producer(tmp_path):
    from tcip_mcp.delivery import read_delivery_events
    from tests._chain_fixtures import run_the_chain

    chain = run_the_chain(tmp_path, experiment_id="exp-counts")

    res = _deliver(tmp_path, chain.root, chain.bucket)

    assert "error" not in res, res
    assert res["validated"] is True
    assert res["producer"] == chain.assessment["producer"]
    with open(res["csv_path"], newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == res["image_count"] == len(_images(chain.images_dir))
    assert {r["validated"] for r in rows} == {"True"}
    (event,) = read_delivery_events(tmp_path)
    assert event.door == "deliver_per_image_counts"
    assert event.population == [p.stem for p in _images(chain.images_dir)]


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
import tcip_store
tcip_store.bind()
import tcip_mcp.tools.inference_tools as itools
from pathlib import Path
r = itools.deliver_per_image_counts(Path({str(tmp_path)!r}), {str(chain.root)!r},
                                    {chain.bucket!r}, {str(out_csv)!r}, trait={fx.COUNT_TRAIT!r})
assert "error" not in r, r
assert "torch" not in sys.modules, "the delivery pulled torch into sys.modules"
print("ok")
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
    assert out_csv.exists()
