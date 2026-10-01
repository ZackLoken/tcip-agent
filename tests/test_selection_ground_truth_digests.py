"""The ground-truth digests a drawn selection records: each sample carries its own file's digest,
the draw reads each file once, a bound run's partition alone carries them, and the selection's own
digest is the one value both the run's binding and an assessment's reference record.
"""

from __future__ import annotations

import hashlib
from functools import partial
from pathlib import Path

import pytest

from tests._image_fixtures import write_image

torch = pytest.importorskip("torch")

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402
from tcip_store import RECORD_JSON  # noqa: E402

IMG = 32
SUBJECT = "leaf"
DATES = ("2026-02-11", "2026-02-25")
_STEMS = ("a", "b", "c", "d", "e", "f", "g", "h")

_save_png = partial(write_image, size=IMG)


def _dataset(root: Path, stems=_STEMS) -> Path:
    """Two capture dates, eight stems each, enough groups that a three-way draw leaves both
    train and val non-empty for either date."""
    for date in DATES:
        images_dir, labels_dir = root / "images" / date, root / "annotations" / date
        for stem in stems:
            _save_png(images_dir / f"{stem}.jpg")
            json_io.write_annotations(
                str(labels_dir / f"{stem}.json"),
                [Annotation(subject=SUBJECT, geometry=BBox(2, 2, 10, 10))], IMG, IMG,
            )
    return root


def _draw(project: Path, root: Path, out: Path, *, seed: int = 2):
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.tools.data_tools import draw_splits

    result = draw_splits(project, str(root), output_path=str(out), subject=SUBJECT, seed=seed,
                         train_ratio=0.4, val_ratio=0.3, calibration_ratio=0.15, holdout_ratio=0.15)
    assert "error" not in result, result
    return read_selection(out, project=project)


def _bind_run(project: Path, out: Path, experiment_id: str) -> Path:
    """A run of ``project`` bound to the selection at ``out``, its data resolved and recorded by
    the launcher's own producer, so no training body runs. Returns the run directory."""
    from tests._verified_checkpoint_fixtures import resolved_run

    return resolved_run(project, {"split": {"selection_dir": str(out)}}, experiment_id=experiment_id)


def _manifest_sha256(out: Path) -> str:
    """The sha256 of the selection record at ``out`` as the store holds it."""
    import tcip_store

    from tcip_mcp.pipelines.data.selection import selection_key

    return hashlib.sha256(RECORD_JSON.encode(tcip_store.read(selection_key(out)))).hexdigest()


def test_every_drawn_sample_carries_its_own_ground_truth_digest(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "ds")
    drawn = _draw(tmp_path, root, tmp_path / "m")

    assert drawn.samples
    for sample in drawn.samples:
        assert sample.ground_truth_digest == hashlib.sha256(
            Path(sample.ground_truth).read_bytes()).hexdigest()[:16]


def test_a_withdrawn_ground_truth_refuses_its_digest_by_name(tmp_path: Path) -> None:
    """A file that is gone has no digest: the read refuses naming it rather than answering a
    stand-in value a comparison could mistake for a recorded one."""
    from tcip_mcp.pipelines.data.selection import ground_truth_digest

    with pytest.raises(FileNotFoundError, match="absent.json"):
        ground_truth_digest(tmp_path / "absent.json")


def test_the_run_binding_records_the_selection_digest_once_never_on_a_sample(
    tmp_path: Path,
) -> None:
    """``selection_digest`` is the sha256 over ``RECORD_JSON.encode`` of the selection's own
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
    assert resolved["partition"]["selection"]["selection_sha256"] == _manifest_sha256(out)
    assert not any("selection_sha256" in sample for sample in resolved["partition"]["samples"])


def test_draw_splits_digests_every_members_document_in_one_read(tmp_path: Path) -> None:
    """The draw digests its members' ground truth in one batch, each document once however many
    members it answers for, and every sample's digest is that batch's."""
    import unittest.mock as mock

    from tcip_mcp.pipelines.data import selection

    root = _dataset(tmp_path / "ds")
    real_digests = selection.ground_truth_digests
    batches: list[dict[str, str]] = []

    def spy(paths):
        batches.append(real_digests(paths))
        return batches[-1]

    with mock.patch.object(selection, "ground_truth_digests", spy):
        drawn = _draw(tmp_path, root, tmp_path / "m")

    (batch,) = batches
    assert sorted(batch) == sorted({s.ground_truth for s in drawn.samples})
    assert all(s.ground_truth_digest == batch[s.ground_truth] for s in drawn.samples)


def test_the_partition_alone_carries_the_per_sample_digests(tmp_path: Path) -> None:
    """The per-sample digests and groups ride only in the run's resolved partition: the data
    section recorded beside it (the one every checkpoint embeds) never gains them."""
    from tcip_mcp.experiments import run_resolution

    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(tmp_path, root, out)

    resolved = run_resolution(_bind_run(tmp_path, out, "exp_split_config_readback").name,
                              project=tmp_path)

    assert resolved["partition"]["samples"]
    assert all(sample["ground_truth_digest"] for sample in resolved["partition"]["samples"])
    for block in (resolved["data"]["split"], resolved["partition"]["selection"]):
        assert "samples" not in block
        assert "ground_truth_digests" not in block
