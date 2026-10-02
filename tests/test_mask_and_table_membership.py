"""Membership for the tasks a per-image label document does not answer for.

A semantic segmentation run's ground truth is a mask raster beside its image; a classification,
ordinal or regression run's is one row of a table. Both are named by the platform's own producer,
drawn into a selection the same way a label-document draw is, and read back off the loaders the
run actually builds rather than off a record written beside them.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from tcip_mcp.pipelines.data.selection import ClassScope, read_selection  # noqa: E402
from tcip_mcp.pipelines.data.split_construction import (  # noqa: E402
    auto_train_val, partition_samples,
)
from tcip_mcp.tools.data_tools import draw_splits  # noqa: E402

STEMS = ("a", "b", "c", "d", "e", "f", "g", "h")


def _side(partition: dict, side: str) -> list[str]:
    """The member names a resolved partition put on ``side``."""
    return sorted({s.member for s in partition_samples(partition) if s.side == side})


def _resolved(project: Path, experiment_id: str, task: str, data_cfg: dict):
    """A run under ``project`` over ``data_cfg`` resolved by the launcher's own producer: what its
    launch record says it resolved, and the train and val datasets the child's own context
    producer builds its loaders from out of that record."""
    from tcip_mcp.experiments import observe, run_resolution
    from tcip_mcp.pipelines.training.subprocess_worker import prepare_run_context
    from tests._verified_checkpoint_fixtures import opened_run

    run_dir = opened_run(project, {"model_source": {"task": task}, "data": data_cfg},
                         experiment_id=experiment_id)
    ctx = prepare_run_context(observe(run_dir))
    val = ctx.val_loader.dataset if ctx.val_loader is not None else None
    return run_resolution(experiment_id, project=project), ctx.train_loader.dataset, val


def _mask_dataset(root: Path) -> tuple[Path, Path]:
    """A dataset whose ground truth is one ``<stem>.png`` mask per image, each carrying a distinct
    foreground block so a mask read back can be told from its neighbor's."""
    images_dir, masks_dir = root / "images", root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(STEMS):
        Image.new("RGB", (16, 16), (10 * index, 20, 30)).save(images_dir / f"{stem}.png")
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[: index + 1, : index + 1] = 1
        Image.fromarray(mask, mode="L").save(masks_dir / f"{stem}.png")
    return images_dir, masks_dir


def _table_dataset(root: Path) -> tuple[Path, Path]:
    """A dataset whose ground truth is one table row per image, each row a distinct label."""
    images_dir, csv_path = root / "images", root / "labels.csv"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, stem in enumerate(STEMS):
        Image.new("RGB", (16, 16), (10 * index, 20, 30)).save(images_dir / f"{stem}.png")
        rows.append((stem, index % 3))
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stem", "label"))
        writer.writerows(rows)
    return images_dir, csv_path


def _drawn(root: Path, ground_truth: Path, out: Path, *, seed: int = 4):
    result = draw_splits(out.parent, str(root), output_path=str(out), ground_truth=str(ground_truth),
                         seed=seed, train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125,
                         group_by="stem")
    assert "error" not in result, result
    return read_selection(out, project=out.parent)


# -- a bound run trains over exactly the selection's own samples ---------------


def test_a_bound_semantic_seg_run_trains_over_exactly_its_selections_samples(tmp_path: Path):
    """The loaders hold exactly the samples the selection recorded on each side, each reading the
    mask it names, and the held-out calibration side builds neither loader."""
    root = tmp_path / "ds"
    _images_dir, masks_dir = _mask_dataset(root)
    out = tmp_path / "m"
    drawn = _drawn(root, masks_dir, out)

    data_cfg = {"split": {"selection_dir": str(out)}}
    train_ds, val_ds, partition = auto_train_val(tmp_path, "semantic_seg", data_cfg, None)

    assert sorted(train_ds.stems) == sorted(s.location for s in drawn.on("train"))
    assert sorted(val_ds.stems) == sorted(s.location for s in drawn.on("val"))
    held_out = {s.location for s in drawn.on("calibration")}
    assert held_out and not held_out & set(train_ds.stems + val_ds.stems)
    assert _side(partition, "train") == sorted(s.member for s in drawn.on("train"))


def test_a_bound_classification_run_trains_over_exactly_its_selections_samples(tmp_path: Path):
    """The same for a table: each side holds the rows the selection put on it, by row key."""
    root = tmp_path / "ds"
    _images_dir, csv_path = _table_dataset(root)
    out = tmp_path / "m"
    drawn = _drawn(root, csv_path, out)

    data_cfg = {"split": {"selection_dir": str(out)}}
    train_ds, val_ds, partition = auto_train_val(tmp_path, "classification", data_cfg, None)

    assert sorted(train_ds.stems) == sorted(s.location for s in drawn.on("train"))
    assert sorted(val_ds.stems) == sorted(s.location for s in drawn.on("val"))
    held_out = {s.location for s in drawn.on("calibration")}
    assert held_out and not held_out & set(train_ds.stems + val_ds.stems)
    assert {s.ground_truth for s in partition_samples(partition)} == {str(csv_path)}


def test_a_directory_named_like_a_mask_is_admitted_by_neither_read_of_it(tmp_path: Path):
    """What a mask is is one predicate, so the admission that enumerates a directory and the
    re-admission that checks a recorded sample's own path cannot disagree: a directory named
    ``a.1.png`` is a mask to neither."""
    from tcip_mcp.pipelines.data.label_queries import admit, refuse_inadmissible_samples

    images_dir, masks_dir = tmp_path / "images", tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    for stem in ("a.1", "b"):
        Image.new("RGB", (16, 16), (10, 20, 30)).save(images_dir / f"{stem}.png")
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8), mode="L").save(masks_dir / "b.png")
    (masks_dir / "a.1.png").mkdir()

    admitted = admit(images_dir, masks_dir)
    assert [record.member for record in admitted.records] == ["b"]
    # Admits valid work: what it did admit still admits when it is checked again.
    refuse_inadmissible_samples(
        admitted.samples({record.member: "train" for record in admitted.records}, lambda m: m),
        admitted.scope)


# -- each sample reaches its own ground truth ----------------------------------


def test_a_mask_sample_reaches_its_own_mask(tmp_path: Path):
    """The mask a loader serves for one sample is the file that sample names, not one rebuilt from
    a directory and a stem: each fixture mask carries a distinct foreground area, so a mask served
    from the wrong sample reads as the wrong area."""
    root = tmp_path / "ds"
    _images_dir, masks_dir = _mask_dataset(root)
    out = tmp_path / "m"
    drawn = _drawn(root, masks_dir, out)

    train_ds, _val_ds, _partition = auto_train_val(
        tmp_path, "semantic_seg", {"split": {"selection_dir": str(out)}}, None)

    by_identity = {s.location: s for s in drawn.samples}
    for index, key in enumerate(train_ds.stems):
        sample = by_identity[key]
        expected = STEMS.index(sample.member) + 1
        _image, target = train_ds[index]
        assert int(target["masks"].sum()) == expected * expected
        assert Path(sample.ground_truth) == masks_dir / f"{sample.member}.png"


def test_a_row_key_sample_reaches_the_row_it_names(tmp_path: Path):
    """The label a loader serves for one sample is the row its ``row_key`` names in the table it
    names: each fixture row carries a distinct label, so a row read by position instead of by key
    reads as the wrong label."""
    root = tmp_path / "ds"
    _images_dir, csv_path = _table_dataset(root)
    out = tmp_path / "m"
    drawn = _drawn(root, csv_path, out)

    train_ds, _val_ds, _partition = auto_train_val(
        tmp_path, "classification", {"split": {"selection_dir": str(out)}}, None)

    by_identity = {s.location: s for s in drawn.samples}
    for index, key in enumerate(train_ds.stems):
        sample = by_identity[key]
        assert sample.row_key == sample.member
        assert Path(sample.ground_truth) == csv_path
        _image, target = train_ds[index]
        assert target["labels"] == STEMS.index(sample.row_key) % 3


# -- an unbound run's loaders answer for their own membership -------------------


def test_an_unbound_semantic_seg_run_reads_its_membership_off_its_own_samples(tmp_path: Path):
    """No selection: the run's own producer admits the masks beside the images and every loader is
    built from those samples, so membership is read off the loaders rather than off a record."""
    root = tmp_path / "ds"
    images_dir, masks_dir = _mask_dataset(root)
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(masks_dir),
                "split": {"val_ratio": 0.25, "seed": 7}}

    train_ds, val_ds, partition = auto_train_val(tmp_path, "semantic_seg", data_cfg, None)

    assert val_ds is not None
    members = {ds: {ds.sample_of(key).member for key in ds.stems} for ds in (train_ds, val_ds)}
    assert members[train_ds].isdisjoint(members[val_ds])
    assert members[train_ds] | members[val_ds] == set(STEMS)
    assert _side(partition, "train") == sorted(members[train_ds])
    assert _side(partition, "val") == sorted(members[val_ds])
    # Each loader indexes by its samples' own source identities and holds that sample's own
    # source, ground truth and member name.
    for ds in (train_ds, val_ds):
        for key in ds.stems:
            sample = ds.sample_of(key)
            assert sample.source == key, "a sample is indexed by its own source"
            assert Path(sample.source) == images_dir / f"{sample.member}.png"
            assert Path(sample.ground_truth) == masks_dir / f"{sample.member}.png"


def test_an_unbound_regression_run_reads_its_membership_off_its_own_samples(tmp_path: Path):
    """The same for a table: the run's own producer admits the rows naming an image that exists,
    and each loader indexes exactly the rows it was handed."""
    root = tmp_path / "ds"
    images_dir, csv_path = _table_dataset(root)
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(csv_path),
                "split": {"val_ratio": 0.25, "seed": 7}}

    train_ds, val_ds, partition = auto_train_val(tmp_path, "regression", data_cfg, None)

    assert val_ds is not None
    members = {ds: {ds.sample_of(key).member for key in ds.stems} for ds in (train_ds, val_ds)}
    assert members[train_ds].isdisjoint(members[val_ds])
    assert members[train_ds] | members[val_ds] == set(STEMS)
    assert _side(partition, "train") == sorted(members[train_ds])
    assert _side(partition, "val") == sorted(members[val_ds])
    for ds in (train_ds, val_ds):
        for key in ds.stems:
            sample = ds.sample_of(key)
            assert sample.source == key, "a sample is indexed by its own source"
            assert Path(sample.ground_truth) == csv_path
            assert Path(key) == images_dir / f"{sample.member}.png"


# -- a member's name is its row key, dots and all ------------------------------


def test_a_dotted_row_key_names_one_member_end_to_end(tmp_path: Path):
    """A row key is already the member's own name. Stripping a suffix from it would read ``a.1``
    and ``a.2`` as one member, collapsing two rows in the loaders' own membership and in the
    resolved partition the assessment's disjointness joins against by that name.
    """
    root = tmp_path / "ds"
    images_dir, csv_path = root / "images", root / "labels.csv"
    images_dir.mkdir(parents=True, exist_ok=True)
    dotted = [f"a.{index}" for index in range(8)]
    rows = []
    for index, key in enumerate(dotted):
        Image.new("RGB", (16, 16), (10 * index, 20, 30)).save(images_dir / f"{key}.png")
        rows.append((key, index % 3))
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stem", "label"))
        writer.writerows(rows)

    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(csv_path),
                "split": {"val_ratio": 0.5, "seed": 3, "group_by": "stem"}}
    resolved, train_ds, val_ds = _resolved(tmp_path, "exp-dotted", "classification", data_cfg)

    # The loaders name every row apart, not one collapsed member.
    named = {ds.sample_of(k).member for ds in (train_ds, val_ds) for k in ds.stems}
    assert named == set(dotted)
    assert len(train_ds.stems) == train_ds.num_samples

    samples = partition_samples(resolved["partition"])
    assert sorted(s.member for s in samples) == sorted(dotted)
    assert len({s.group for s in samples}) == len(dotted)


# -- freezing a run whose ground truth is not a label document -----------------


def _frozen(project: Path, experiment_id: str, task: str, data_cfg: dict) -> dict:
    """Resolve a run under ``project`` through the child's own producer, which records its
    resolved record, and freeze it."""
    from tcip_mcp.tools.data_tools import freeze_selection

    _resolved(project, experiment_id, task, data_cfg)
    return freeze_selection(project, experiment_id)


def test_a_mask_run_freezes_into_a_selection_its_bind_accepts(tmp_path: Path):
    """The staleness check freezing rests on is per member, against the digest the run's own
    producer recorded, so a mask run freezes: the frozen selection names each member's own mask
    and a later run binds to exactly that partition."""
    root = tmp_path / "ds"
    images_dir, masks_dir = _mask_dataset(root)
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(masks_dir),
                "split": {"val_ratio": 0.25, "seed": 7}}

    result = _frozen(tmp_path, "exp-mask-freeze", "semantic_seg", data_cfg)

    assert "error" not in result, result
    frozen = read_selection(result["selection_dir"], project=tmp_path)
    assert frozen.scope == ClassScope()
    assert {Path(s.ground_truth).suffix for s in frozen.samples} == {".png"}
    for sample in frozen.samples:
        assert Path(sample.ground_truth).parent == masks_dir
        assert Path(sample.ground_truth).is_file()

    train_ds, val_ds, partition = auto_train_val(
        tmp_path, "semantic_seg", {"split": {"selection_dir": result["selection_dir"]}}, None)
    assert sorted(train_ds.stems) == sorted(s.location for s in frozen.on("train"))
    assert val_ds is not None
    assert result["train"] == len(frozen.on("train"))
    assert _side(partition, "train") == sorted(s.member for s in frozen.on("train"))


def test_a_table_run_freezes_into_a_selection_its_bind_accepts(tmp_path: Path):
    """The same for a table: the frozen selection names each member's own row in the table the
    run read, and a later run binds to exactly that partition."""
    root = tmp_path / "ds"
    images_dir, csv_path = _table_dataset(root)
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(csv_path),
                "split": {"val_ratio": 0.25, "seed": 7}}

    result = _frozen(tmp_path, "exp-table-freeze", "classification", data_cfg)

    assert "error" not in result, result
    frozen = read_selection(result["selection_dir"], project=tmp_path)
    assert {s.ground_truth for s in frozen.samples} == {str(csv_path)}
    assert all(s.row_key is not None for s in frozen.samples)

    train_ds, val_ds, _partition = auto_train_val(
        tmp_path, "classification", {"split": {"selection_dir": result["selection_dir"]}}, None)
    assert sorted(train_ds.stems) == sorted(s.location for s in frozen.on("train"))
    assert val_ds is not None


def test_a_mask_edited_after_the_run_refuses_the_freeze_by_name(tmp_path: Path):
    """A member's mask replaced after the run digests differently from the ``at_run`` the record
    holds, so freezing refuses and names it rather than composing a selection over ground truth
    the run never saw."""
    root = tmp_path / "ds"
    images_dir, masks_dir = _mask_dataset(root)
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(masks_dir),
                "split": {"val_ratio": 0.25, "seed": 7}}
    from tcip_mcp.tools.data_tools import freeze_selection

    _resolved(tmp_path, "exp-mask-moved", "semantic_seg", data_cfg)

    edited = np.zeros((16, 16), dtype=np.uint8)
    edited[:12, :12] = 1
    Image.fromarray(edited, mode="L").save(masks_dir / f"{STEMS[0]}.png")

    result = freeze_selection(tmp_path, "exp-mask-moved")

    assert "error" in result and "changed since" in result["error"]
    assert STEMS[0] in result["error"]


def _three_class_masks(root: Path) -> tuple[Path, Path]:
    """Four images and their masks, one of which alone reaches class 2: a run sizing each loader
    by the classes its own half happens to hold would size the two halves differently."""
    images_dir, masks_dir = root / "images", root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(STEMS[:4]):
        Image.new("RGB", (16, 16), (10 * index, 20, 30)).save(images_dir / f"{stem}.png")
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[:8, :8] = 1
        if index == 0:
            mask[8:, 8:] = 2
        Image.fromarray(mask, mode="L").save(masks_dir / f"{stem}.png")
    return images_dir, masks_dir


def test_a_stated_class_count_refuses_on_the_run_path_whatever_its_value(tmp_path: Path):
    """A built-in loader's count is the one its ground truth derives, never an input: a config
    stating one refuses on the run path before any loader is built, whether it is below, above or
    equal to the derived count, and a config stating none records the derived count."""
    images_dir, masks_dir = _three_class_masks(tmp_path / "ds")
    base = {"images_dir": str(images_dir), "labels_dir": str(masks_dir),
            "split": {"group_by": "stem", "val_ratio": 0.25, "seed": 3}}

    for stated in (2, 3, 9):
        with pytest.raises(ValueError, match=r"states \['num_classes'\]"):
            auto_train_val(
                tmp_path, "semantic_seg", {**base, "num_classes": stated, "split": dict(base["split"])},
                None)

    unstated = {**base, "split": dict(base["split"])}
    _train_ds, val_ds, _partition = auto_train_val(tmp_path, "semantic_seg", unstated, None)
    assert val_ds is not None
    assert unstated["num_classes"] == 3 and unstated.get("num_ranks") is None


def test_a_table_run_stating_a_count_refuses(tmp_path: Path):
    """Table values ``0`` and ``1`` derive two classes; a count stated beside them refuses by
    name, and the unstated run records two."""
    from tests._producer_fixtures import run_over

    images_dir, csv_path = _table_dataset(tmp_path / "table")
    with open(csv_path, "w", newline="") as handle:
        csv.writer(handle).writerows([("stem", "label"), *((s, i % 2) for i, s in enumerate(STEMS))])
    with pytest.raises(ValueError, match=r"states \['num_classes'\]"):
        run_over("classification", str(images_dir), str(csv_path), stated={"num_classes": 9})

    _dataset, data = run_over("classification", str(images_dir), str(csv_path))
    assert data["num_classes"] == 2


def test_a_runs_unstated_class_count_is_read_once_over_every_sample(tmp_path: Path):
    """A class reaching one side only still sizes the run: the count a config states none of is
    read off every sample the run was handed, before the split, and recorded once."""
    images_dir, masks_dir = _three_class_masks(tmp_path / "ds")
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(masks_dir),
                "split": {"group_by": "stem", "val_ratio": 0.25, "seed": 3}}

    train_ds, val_ds, partition = auto_train_val(tmp_path, "semantic_seg", data_cfg, None)

    # The one mask reaching class 2 is on the validation side, which is the disagreement a
    # per-loader count would produce here.
    assert _side(partition, "val") == [STEMS[0]]
    assert val_ds is not None
    assert data_cfg["num_classes"] == 3
    assert data_cfg["num_channels"] == train_ds.expected_channels


def test_a_runs_metrics_are_reported_over_the_class_space_it_trains_in(tmp_path: Path):
    """The half being scored is never asked how many classes there are: a training half whose
    masks reach only class 1 is still measured over the three the run trains in, so the two halves
    of one run report metrics on one scale."""
    import torch
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.training.collation import task_collate
    from tcip_mcp.pipelines.training.evaluation import evaluate
    from tests import bespoke_models

    images_dir, masks_dir = _three_class_masks(tmp_path / "ds")
    data_cfg: dict = {"images_dir": str(images_dir), "labels_dir": str(masks_dir),
                      "split": {"group_by": "stem", "val_ratio": 0.25, "seed": 3}}

    train_ds, _val_ds, _partition = auto_train_val(tmp_path, "semantic_seg", data_cfg, None)
    held = {int(np.array(Image.open(masks_dir / f"{stem}.png")).max())
            for stem in STEMS[1:4]}
    assert held == {1}, "the training half reaches only class 1, which is what this measures"

    count = int(data_cfg["num_classes"])
    model = bespoke_models.build_bespoke_semantic_seg(num_classes=count)
    loader = DataLoader(train_ds, batch_size=1, collate_fn=task_collate("semantic_seg"))
    result = evaluate(model, loader, torch.device("cpu"), "semantic_seg",
                      dims={"in_chans": 3, "num_classes": count})

    assert sorted(result["per_class_iou"]) == [0, 1, 2]


def test_the_place_and_the_record_read_one_ground_truth_shape(tmp_path: Path):
    """The sniff over a place a config names and the read over a sample's own recorded ground
    truth answer with one shape, because the place asks the record's own rule of what it finds
    there: a table, a directory of masks and a directory of documents each read the same way from
    either side."""
    from tcip_mcp.pipelines.data.label_queries import ground_truth_shape
    from tcip_mcp.pipelines.data.selection import shape_of

    root = tmp_path / "ds"
    _images_dir, masks_dir = _mask_dataset(root)
    _table_images, csv_path = _table_dataset(tmp_path / "table")
    documents = root / "annotations"
    documents.mkdir(parents=True)
    (documents / "a.json").write_text("{}", encoding="utf-8")

    for place, member in ((csv_path, csv_path),
                          (masks_dir, masks_dir / f"{STEMS[0]}.png"),
                          (documents, documents / "a.json")):
        assert ground_truth_shape(str(place)) == shape_of(str(member), None), place


def test_a_mask_run_refuses_a_stated_class_space_and_admits_the_empty_one(tmp_path: Path):
    """A mask raster carries its own classes, so a scope naming a subject over it refuses by name;
    the explicit empty scope admits and is what the run records."""
    images_dir, masks_dir = _mask_dataset(tmp_path / "ds")

    def data_cfg(scope: dict) -> dict:
        return {"images_dir": str(images_dir), "labels_dir": str(masks_dir), "scope": scope,
                "split": {"group_by": "stem", "val_ratio": 0.25, "seed": 5}}

    with pytest.raises(ValueError, match="carries its own classes"):
        auto_train_val(tmp_path, "semantic_seg", data_cfg(
            {"subject": "leaf"}), None)

    admitted = data_cfg({})
    auto_train_val(tmp_path, "semantic_seg", admitted, None)
    assert admitted["scope"] == {"subject": None, "attributes": None}


# -- two producers of one record, compared against each other ------------------


def test_the_bound_and_drawn_mask_routes_record_one_directory_the_same_way(tmp_path: Path):
    """The same mask directory is admitted twice, once by a run bound to a selection drawn over it
    and once by a run that drew its own split. The two partition it differently, which is what
    each route is for; what they may not do is disagree about which members that directory holds,
    where it is, or what each member's ground truth digests to now.
    """
    root = tmp_path / "ds"
    images_dir, masks_dir = _mask_dataset(root)
    out = tmp_path / "m"
    _drawn(root, masks_dir, out)

    def trained(experiment_id: str, data_cfg: dict) -> dict:
        """Every member the run trains or validates on, with the ground truth, its digest at the
        run and the source the record names for it."""
        partition = _resolved(tmp_path, experiment_id, "semantic_seg", data_cfg)[0]["partition"]
        at_run = partition["ground_truth_digests"]
        return {s.member: (s.ground_truth, at_run[s.ground_truth], s.source)
                for s in partition_samples(partition) if s.side in ("train", "val")}

    bound = trained("exp-mask-bound", {"split": {"selection_dir": str(out)}})
    drawn = trained("exp-mask-drawn",
                    {"images_dir": str(images_dir), "labels_dir": str(masks_dir),
                     "split": {"val_ratio": 0.25, "seed": 7}})

    assert {Path(truth).parent for truth, _digest, _source in (*bound.values(), *drawn.values())
            } == {masks_dir}
    # The bound run holds out a calibration side the drawn one trains on, so the two agree on
    # every member they both name rather than on the union.
    assert bound and drawn and set(bound) <= set(drawn)
    assert {member: bound[member] for member in bound} == {member: drawn[member] for member in bound}


def test_a_document_run_and_a_mask_run_record_the_same_images_the_same_way(tmp_path: Path):
    """Two shapes of one producer, compared against each other rather than against a fixture.

    The same images are trained twice, once over a per-image label document each and once over a
    ``<stem>.png`` mask each, at the same seed and grouping policy. The two ground truths are
    genuinely different files, so the digests and the scopes differ; what the one producer may not
    do is name a different membership, a different partition, a differently shaped record or a
    different image for one shape than for the other.
    """
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import SubjectRegistry, Subject
    from tests._producer_fixtures import registry_over

    root = tmp_path / "ds"
    images_dir, masks_dir = _mask_dataset(root)
    labels_dir = root / "annotations"
    labels_dir.mkdir(parents=True, exist_ok=True)
    registry_over(root, SubjectRegistry(subjects=(Subject(name="leaf"),)))
    for stem in STEMS:
        json_io.write_annotations(labels_dir / f"{stem}.json",
                                  [Annotation(subject="leaf", geometry=BBox(2, 2, 8, 8))],
                                  16, 16, keep_empty=True)

    split = {"group_by": "stem", "val_ratio": 0.25, "seed": 11}

    def resolved(experiment_id: str, task: str, data_cfg: dict) -> tuple[dict, dict, dict]:
        """The resolved partition, and what the loaders this run actually built say each
        member's own ground truth and source are."""
        record, train_ds, val_ds = _resolved(tmp_path, experiment_id, task, data_cfg)
        served = [ds.sample_of(key) for ds in (train_ds, val_ds) for key in ds.stems]
        served_truth = {s.member: s.ground_truth for s in served}
        served_sources = {s.member: s.source for s in served}
        return record["partition"], served_truth, served_sources

    document_run, document_served, document_sources = resolved(
        "exp-shape-document", "detection", {
            "images_dir": str(images_dir), "labels_dir": str(labels_dir),
            "scope": {"subject": "leaf"}, "split": dict(split)})
    mask_run, mask_served, mask_sources = resolved(
        "exp-shape-mask", "semantic_seg", {
            "images_dir": str(images_dir), "labels_dir": str(masks_dir),
            "split": dict(split)})

    assert set(document_run) == set(mask_run), "one record shape, whatever the ground truth is"
    assert document_run["group_by"] == mask_run["group_by"] == "stem"

    def by_member(partition: dict) -> dict:
        return {s.member: s for s in partition_samples(partition)}

    document_samples, mask_samples = by_member(document_run), by_member(mask_run)
    assert {Path(s.ground_truth).parent for s in document_samples.values()} == {labels_dir}
    assert {Path(s.ground_truth).parent for s in mask_samples.values()} == {masks_dir}
    assert {m: (s.side, s.group) for m, s in document_samples.items()} == {
        m: (s.side, s.group) for m, s in mask_samples.items()}

    # One assertion over both shapes: each path the record names for a member is the path the
    # loader that trained on it read, so a record naming any other file fails here.
    for samples, served, sources in ((document_samples, document_served, document_sources),
                                     (mask_samples, mask_served, mask_sources)):
        assert {m: s.ground_truth for m, s in samples.items()} == served
        assert {m: s.source for m, s in samples.items()} == sources

    # Two runs over one set of images name the same pixels per member: a source wrong the same
    # way in a producer and in the record it writes agrees with itself, and fails only here.
    assert document_sources == mask_sources
