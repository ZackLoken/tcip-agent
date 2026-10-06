"""Every standard geometry route's loaders answer for their own membership.

A detection or instance_seg run the known loaders cover names its membership once, through the
platform's own producer, and builds every loader out of the samples that producer made. Which
route the caller took decides which side each sample lands on and nothing about how a loader finds
its members: whether the caller named a validation directory, turned ``auto_val`` off, or let the
run draw its own split, each loader reads the source and the label document its own samples
state.

These drive each route through ``auto_train_val`` and read the membership back off the loaders
themselves, never off the record the run persists, so a loader that rediscovered its members from
a directory would fail here even with a faithful record beside it. Every route a run of either
geometry task can take is driven under both, each with the geometry its own loader reads, so a
dispatch that sent one task around the producer fails here rather than passing on the other's
coverage.
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_annotation.state import Annotation, BBox, Polygon  # noqa: E402
from tcip_mcp.pipelines.data.split_construction import (  # noqa: E402
    auto_train_val, partition_samples,
)
from tests._producer_fixtures import label_image, write_image  # noqa: E402
from tests._verified_checkpoint_fixtures import partition_side as recorded_side  # noqa: E402

IMG = 64
SUBJECT = "bud"
GEOMETRY_TASKS = ("detection", "instance_seg")
"""The tasks the known geometry loaders cover, which is the set the producer names membership for.
Every route below that either can take is parametrized over both."""


def _target_geometry(task: str):
    """The geometry one task's loader actually reads: a box for detection, a closed ring for
    instance_seg, whose loader reads polygons only and rasterizes their rings."""
    if task == "instance_seg":
        return Polygon(rings=[[(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)]])
    return BBox(10.0, 10.0, 30.0, 30.0)


def _labeled(root: Path, stems, *, task: str = "detection") -> Path:
    """One labeled directory: an image and a per-image label document per stem, each carrying the
    geometry ``task``'s own loader reads; the image directory."""
    images_dir = root / "images" / UNDATED_BUCKET
    for stem in stems:
        write_image(images_dir / f"{stem}.png", (IMG, IMG), (70, 90, 60))
        label_image(images_dir / f"{stem}.png",
                    [Annotation(subject=SUBJECT, geometry=_target_geometry(task))], IMG, IMG)
    return images_dir


def _one_target(ds, task: str) -> None:
    """Read one example off a loader and assert the task's own target shape came back.

    Membership alone does not prove a loader reads what its task needs: an instance_seg run whose
    samples carry only boxes indexes every member and yields no mask at all.
    """
    _image_tensor, target = ds[0]
    assert target["boxes"].tolist() == [[10.0, 10.0, 30.0, 30.0]]
    if task == "instance_seg":
        assert target["masks"].shape[0] == 1
        assert int(target["masks"].sum()) > 0


def _big_source(root: Path, stem: str, width: int, height: int) -> Path:
    """One source large enough to hold a within-image spatial split, with GT across its extent;
    the image directory."""
    from PIL import Image

    images_dir = root / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=(70, 90, 60)).save(images_dir / f"{stem}.png")
    boxes = [Annotation(subject=SUBJECT, geometry=BBox(x, y, x + 20, y + 20))
             for x in range(20, width - 20, 200) for y in range(20, height - 20, 200)]
    label_image(images_dir / f"{stem}.png", boxes, width, height)
    return images_dir


def _membership(ds) -> set[str]:
    """What one loader says its own members are, read off the samples it was built from.

    Every key the loader indexes names one of the samples it was built from, whose source is on
    disk. A tiled loader indexes one example per kept tile, so its keys are a subset of the samples
    it was handed rather than all of them.
    """
    keys = set(ds.stems)
    for key in keys:
        assert Path(ds.sample_of(key).source).is_file()
    return {ds.sample_of(key).member for key in keys}


def _recorded(project: Path, task: str, data_cfg: dict) -> dict:
    """The partition the launcher's own producer resolves for a ``task`` run of ``project`` over
    ``data_cfg``, read back from the launch record it writes: the record's members come from the
    producer, never from a loader's own keys."""
    import copy

    from tcip_mcp.experiments import run_resolution
    from tests._verified_checkpoint_fixtures import opened_run

    run_dir = opened_run(project, {"model_source": {"task": task},
                                   "data": copy.deepcopy(data_cfg),
                                   "evaluation": {"selection_metric": "loss"}})
    return run_resolution(run_dir.name, project=project)["partition"]


@pytest.mark.parametrize("task", GEOMETRY_TASKS)
def test_the_drawn_route_loaders_name_their_own_samples(tmp_path: Path, task: str):
    stems = [f"src{i}_0_0" for i in range(4)]
    images_dir = _labeled(tmp_path / "ds", stems, task=task)
    data_cfg = {"images_dir": str(images_dir), "scope": {"subject": SUBJECT}, "auto_val": True,
                "split": {"val_ratio": 0.5, "seed": 1}}

    train_ds, val_ds, partition = auto_train_val(tmp_path, task, data_cfg, None)

    assert val_ds is not None
    assert _membership(train_ds).isdisjoint(_membership(val_ds))
    assert _membership(train_ds) | _membership(val_ds) == set(stems)
    assert recorded_side(partition, "train") == sorted(_membership(train_ds))
    _one_target(train_ds, task)


@pytest.mark.parametrize("task", GEOMETRY_TASKS)
def test_auto_val_off_trains_on_every_admitted_sample_and_records_them(tmp_path: Path, task: str):
    stems = ["a_0_0", "b_0_0", "c_0_0"]
    images_dir = _labeled(tmp_path / "ds", stems, task=task)
    data_cfg = {"images_dir": str(images_dir), "scope": {"subject": SUBJECT}, "auto_val": False}
    record = _recorded(tmp_path, task, data_cfg)

    train_ds, val_ds, partition = auto_train_val(tmp_path, task, data_cfg, None)

    assert val_ds is None
    assert _membership(train_ds) == set(stems)
    assert recorded_side(partition, "train") == sorted(stems)
    assert recorded_side(partition, "val") == []
    _one_target(train_ds, task)
    assert recorded_side(record, "train") == sorted(stems)


@pytest.mark.parametrize("task", GEOMETRY_TASKS)
def test_a_starved_draw_refuses_by_the_draws_own_floor(tmp_path: Path, task: str):
    """A group policy that collapses every sample into one group leaves no side to hold out, so
    the run refuses naming the floor rather than training without validation."""
    stems = ["a_0_0", "b_0_0"]
    images_dir = _labeled(tmp_path / "ds", stems, task=task)
    data_cfg = {"images_dir": str(images_dir), "scope": {"subject": SUBJECT}, "auto_val": True,
                "split": {"val_ratio": 0.5, "seed": 1,
                          "group_key_map": {f"{UNDATED_BUCKET}/{s}": "one_group"
                                            for s in stems}}}

    with pytest.raises(ValueError, match="fewer than the 2 the requested sides need"):
        auto_train_val(tmp_path, task, data_cfg, None)


def test_one_tiled_source_still_splits_spatially_over_its_own_samples(tmp_path: Path):
    """The within-image route wraps the producer's own sample, records that one sample as its
    partition, and its region identities still name the bare stem every consumer of the manifest
    joins on.

    Detection alone: tiling wraps the detection loader, and no other task reaches this route.
    """
    stem = "mosaic"
    images_dir = _big_source(tmp_path / "ds", stem, 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir), "scope": {"subject": SUBJECT},
        "auto_val": True, "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
        "split": {"val_ratio": 0.25, "holdout_ratio": 0.1, "calibration_ratio": 0, "seed": 1},
    }

    train_ds, val_ds, partition = auto_train_val(tmp_path, "detection", data_cfg, None)

    assert val_ds is not None
    assert [s.member for s in partition_samples(partition)] == [stem]
    assert _membership(train_ds) == _membership(val_ds) == {stem}
    assert set(train_ds.tile_entries).isdisjoint(set(val_ds.tile_entries))
    manifest = data_cfg["split"]["spatial_manifest"]
    assert manifest["stem"] == stem
    assert all(i.startswith(f"{stem}::strip_") for i in manifest["train_identities"])
    assert set(manifest["train_identities"]).isdisjoint(manifest["val_identities"])


def test_the_train_only_and_drawn_routes_record_one_directory_the_same_way(tmp_path: Path):
    """The same training directory is admitted twice, once by a run training on every admitted
    sample and once by a run that drew its own split over it. The two runs partition it
    differently, which is what each route is for; what they may not do is disagree about which
    members that directory holds, where it is, or what each member's ground truth digests to
    now.
    """
    stems = ["a_0_0", "b_0_0", "c_0_0", "d_0_0"]
    images_dir = _labeled(tmp_path / "train_ds", stems)

    whole_cfg = {"images_dir": str(images_dir), "scope": {"subject": SUBJECT}, "auto_val": False}
    whole = _recorded(tmp_path, "detection", whole_cfg)

    drawn_cfg = {"images_dir": str(images_dir), "scope": {"subject": SUBJECT}, "auto_val": True,
                 "split": {"val_ratio": 0.5, "seed": 1}}
    drawn = _recorded(tmp_path, "detection", drawn_cfg)

    def _held(record: dict) -> list[tuple]:
        return sorted((s.member, s.source, s.ground_truth, s.ground_truth_digest)
                      for s in partition_samples(record))

    assert _held(whole) == _held(drawn)
    assert len(_held(whole)) == len(stems)
    assert all(digest for *_, digest in _held(whole))
    assert recorded_side(whole, "val") == [] and recorded_side(drawn, "val") != []
