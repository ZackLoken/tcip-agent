"""The ground-truth digests a drawn selection records: each sample carries its own document's
version, the draw reads each document once, a bound run's partition alone carries them, and the
selection's own digest is the one value both the run's binding and an assessment's reference
record.
"""

from __future__ import annotations

import hashlib
from functools import partial
from pathlib import Path

import pytest

import tcip_store
from tests._producer_fixtures import label_image, write_image

torch = pytest.importorskip("torch")

from tcip_annotation.state import Annotation, BBox  # noqa: E402
from tcip_mcp.pipelines.schemas import DataSpec  # noqa: E402
from tcip_store import encode_record  # noqa: E402

IMG = 32
SUBJECT = "leaf"
DATES = ("2026-02-11", "2026-02-25")
_STEMS = ("a", "b", "c", "d", "e", "f", "g", "h")

_save_png = partial(write_image, size=(IMG, IMG))


def _dataset(root: Path, stems=_STEMS) -> Path:
    """Two capture dates, eight stems each, enough groups that a three-way draw leaves both
    train and val non-empty for either date."""
    for date in DATES:
        images_dir = root / "images" / date
        for stem in stems:
            _save_png(images_dir / f"{stem}.jpg")
            label_image(images_dir / f"{stem}.jpg",
                        [Annotation(subject=SUBJECT, geometry=BBox(2, 2, 10, 10))], IMG, IMG)
    return root


def _draw(project: Path, root: Path, out: Path, *, seed: int = 2):
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.tools.data_tools import draw_splits

    result = draw_splits(project, str(root), output_path=str(out), subject=SUBJECT, seed=seed,
                         val_ratio=0.3, calibration_ratio=0.15, holdout_ratio=0.15)
    assert "error" not in result, result
    return read_selection(out, project=project)


def _bind_run(project: Path, out: Path, experiment_id: str) -> Path:
    """A run of ``project`` bound to the selection at ``out``, its data resolved and recorded by
    the launcher's own producer, so no training body runs. Returns the run directory."""
    from tests._verified_checkpoint_fixtures import resolved_run

    return resolved_run(
        project, {"split": {"selection_dir": str(out)}}, experiment_id=experiment_id)


def _manifest_sha256(out: Path) -> str:
    """The sha256 of the selection record at ``out`` as the store holds it."""
    import tcip_store

    from tcip_mcp.pipelines.data.selection import selection_key

    return hashlib.sha256(encode_record(tcip_store.read(selection_key(out)))).hexdigest()


def test_every_drawn_sample_carries_its_own_documents_version_as_its_digest(
        tmp_path: Path) -> None:
    from tcip_mcp.dataset_layout import label_key

    root = _dataset(tmp_path / "ds")
    drawn = _draw(tmp_path, root, tmp_path / "m")

    assert len(drawn.samples) == len(DATES) * len(_STEMS)
    for sample in drawn.samples:
        assert sample.ground_truth in {label_key(root, date, sample.member) for date in DATES}
        assert sample.ground_truth_digest == tcip_store.read_versioned(
            sample.ground_truth).version.token


def test_a_withdrawn_ground_truth_refuses_its_digest_by_name(tmp_path: Path) -> None:
    """A document that is gone has no digest: the read refuses naming it rather than answering
    a stand-in value a comparison could mistake for a recorded one."""
    from tcip_mcp.dataset_layout import UNDATED_BUCKET, label_key
    from tcip_mcp.pipelines.data.selection import ground_truth_digest

    with pytest.raises(tcip_store.NotFoundError, match="absent"):
        ground_truth_digest(label_key(tmp_path, UNDATED_BUCKET, "absent"))
    with pytest.raises(tcip_store.NotFoundError, match="absent.csv"):
        ground_truth_digest(str(tmp_path / "absent.csv"))


def test_the_run_binding_records_the_selection_digest_once_never_on_a_sample(
    tmp_path: Path,
) -> None:
    """``selection_digest`` is the sha256 over ``encode_record`` of the selection's own
    document as the store holds it, and the run's resolved partition carries that value once in
    its binding block, never on a sample."""
    from tcip_mcp.experiments import run_resolution
    from tcip_mcp.pipelines.data.selection import read_selection, selection_digest

    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(tmp_path, root, out)
    resolved = run_resolution(_bind_run(tmp_path, out, "exp_selection_digest").name,
                              project=tmp_path)

    selection = read_selection(out, project=tmp_path)
    assert selection_digest(selection, tmp_path) == _manifest_sha256(out)
    assert resolved.partition["selection"]["selection_sha256"] == _manifest_sha256(out)
    assert not any("selection_sha256" in sample for sample in resolved.partition["samples"])


def test_a_document_emptied_after_its_admission_never_trains_as_empty(
        tmp_path: Path, monkeypatch) -> None:
    """The loaders train the document the admission read, at the version its sample records: one
    emptied between that read and the loaders is never trained as an image with nothing on it."""
    from tcip_mcp.pipelines.data import label_queries
    from tcip_mcp.pipelines.data.selection import ground_truth_digest
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _dataset(tmp_path / "ds")
    images_dir = root / "images" / DATES[0]
    real_samples = label_queries.Admission.samples

    def emptied_first(self, assignment, group_of):
        label_image(images_dir / "a.jpg", [], IMG, IMG, keep_empty=True)
        return real_samples(self, assignment, group_of)

    monkeypatch.setattr(label_queries.Admission, "samples", emptied_first)
    data = DataSpec.model_validate({"images_dir": str(images_dir), "scope": {"subject": SUBJECT},
                                    "split": {"seed": 0, "val_ratio": 0.15}})

    train, val, _partition, _resolved = auto_train_val(tmp_path, "detection", data, None)

    (key,) = [k for ds in (train, val) for k in ds.stems if Path(k).stem == "a"]
    held = next(ds for ds in (train, val) if key in ds.stems)
    assert held.document(key).annotations
    assert held.sample_of(key).ground_truth_digest != ground_truth_digest(
        held.sample_of(key).ground_truth)


def test_the_partition_alone_carries_the_per_sample_digests(tmp_path: Path) -> None:
    """The per-sample digests and groups ride only in the run's resolved partition: the data
    section recorded beside it (the one every checkpoint embeds) never gains them."""
    from tcip_mcp.experiments import run_resolution

    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(tmp_path, root, out)

    resolved = run_resolution(_bind_run(tmp_path, out, "exp_split_config_readback").name,
                              project=tmp_path)

    assert resolved.partition["samples"]
    assert all(sample["ground_truth_digest"] for sample in resolved.partition["samples"])
    for block in (resolved.data.record()["split"], resolved.partition["selection"]):
        assert "samples" not in block
