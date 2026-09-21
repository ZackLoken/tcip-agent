"""Training and tuning binding to a named selection (``data.split.selection_dir``).

The selection is drawn by ``draw_splits`` (see ``test_data_tools.py``); this file covers the
consumer side: ``auto_train_val``'s own branch that binds a run to one, the redraw inside a bound
selection, the refusals a launch raises, and what the run's own partition record carries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from torch.utils.data import Dataset  # noqa: E402
from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.pipelines.data.selection import read_selection
from tcip_mcp.pipelines.data.split_construction import recorded_side
from tcip_mcp.subject_registry import Attribute, SubjectRegistry, Subject, write_registry
from tcip_mcp.tools.data_tools import draw_splits

SUBJECT = "leaf"
OTHER_SUBJECT = "bud"
DATES = ("2-11-26", "2-12-01")


def _write_stem(images_dir: Path, labels_dir: Path, stem: str, annotations) -> None:
    from PIL import Image

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (100, 120, 90)).save(images_dir / f"{stem}.jpg")
    json_io.write_annotations(labels_dir / f"{stem}.json", annotations, 64, 64, keep_empty=True)


def _two_subject_two_date_dataset(root: Path) -> Path:
    """Two capture dates, six stems each: every stem carries ``leaf`` (twelve foreground groups,
    clearing a leaf-scoped draw's floor), and four of the six also carry the unrelated ``bud``
    (eight foreground groups, clearing a bud-scoped draw's floor too), so a selection drawn for
    either subject binds to a real, differently-sized draw over the identical tree."""
    write_registry(root / "subjects.json", SubjectRegistry(subjects=(
        Subject(name=SUBJECT), Subject(name=OTHER_SUBJECT),
    )))
    for date in DATES:
        images_dir, labels_dir = root / "images" / date, root / "annotations" / date
        for stem in ("a", "b"):
            _write_stem(images_dir, labels_dir, stem,
                       [Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20))])
        for stem in ("c", "d", "e", "f"):
            _write_stem(images_dir, labels_dir, stem, [
                Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20)),
                Annotation(subject=OTHER_SUBJECT, geometry=BBox(30, 30, 44, 44)),
            ])
    return root


def _attribute_scoped_dataset(root: Path) -> Path:
    """One date, five stems: four have their instance assessed for ``condition`` (clearing an
    attribute-scoped draw's floor), the fifth carries an instance never assessed for it."""
    write_registry(root / "subjects.json", SubjectRegistry(subjects=(
        Subject(name=SUBJECT, attributes=(
            Attribute(name="condition", type="categorical", values=("healthy", "damaged")),
        )),
    )))
    images_dir, labels_dir = root / "images" / DATES[0], root / "annotations" / DATES[0]
    for stem, condition in (
        ("assessed_a", "healthy"), ("assessed_b", "damaged"),
        ("assessed_c", "healthy"), ("assessed_d", "damaged"),
    ):
        _write_stem(images_dir, labels_dir, stem, [
            Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20),
                      attributes={"condition": condition})])
    _write_stem(images_dir, labels_dir, "unassessed",
               [Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20))])
    return root


def _dataset_with_a_confirmed_negative(root: Path) -> Path:
    """One date, four annotated stems (clearing a draw's foreground floor) plus a fifth stem
    whose label file is empty, for a caller to confirm negative."""
    write_registry(root / "subjects.json", SubjectRegistry(subjects=(Subject(name=SUBJECT),)))
    images_dir, labels_dir = root / "images" / DATES[0], root / "annotations" / DATES[0]
    for stem in ("a", "b", "c", "d"):
        _write_stem(images_dir, labels_dir, stem,
                   [Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20))])
    _write_stem(images_dir, labels_dir, "n", [])
    return root


def _tiled_dataset(root: Path) -> Path:
    """Four parents, three crops each, named ``<parent>_<x>_<y>`` so the default tile-prefix
    grouping puts every crop of one parent in one group."""
    write_registry(root / "subjects.json", SubjectRegistry(subjects=(Subject(name=SUBJECT),)))
    images_dir, labels_dir = root / "images" / DATES[0], root / "annotations" / DATES[0]
    for parent in ("srcA", "srcB", "srcC", "srcD"):
        for x in range(3):
            _write_stem(images_dir, labels_dir, f"{parent}_{x}_0",
                       [Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20))])
    return root


class _RecordingDataset(Dataset):
    """A bespoke dataset standing in for an agent's own builder: it records the samples, the class
    map and any data location the seam handed it, so a test can state what a run actually threads
    through and what it does not."""

    def __init__(self, samples, id_map, directories) -> None:
        self.seen_samples = list(samples)
        self.seen_id_map = id_map
        self.seen_directories = directories

    def __len__(self) -> int:
        return len(self.seen_samples)

    def __getitem__(self, idx: int):
        return torch.zeros(3, 8, 8), {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0)}


_RECORDED_BUILDS: list[_RecordingDataset] = []


def build_recording_dataset(samples=None, id_map=None, **kwargs) -> _RecordingDataset:
    """The ``dataset_source`` builder the bespoke-seam tests register through the seam's dotted
    escape. Records every data-location key it was handed, so a test can state that none was, and
    appends itself to :data:`_RECORDED_BUILDS` for a caller that cannot reach the built dataset."""
    located = {k: v for k, v in kwargs.items()
               if k in ("images_dir", "labels_dir", "csv_path", "coco_data", "stems")}
    built = _RecordingDataset(samples or [], id_map, located)
    _RECORDED_BUILDS.append(built)
    return built


def _draw(root: Path, out: Path, *, subject: str = SUBJECT, attribute: str | None = None,
         seed: int = 2):
    result = draw_splits(str(root), output_path=str(out), subject=subject, attribute=attribute,
                         seed=seed, train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in result, result
    return read_selection(out)


def _run_data_cfg(root: Path, selection_dir: Path, **overrides) -> dict:
    data_cfg: dict = {"split": {"selection_dir": str(selection_dir)}}
    data_cfg.update(overrides)
    return data_cfg


# -- binding a run to a selection ----------------------------------------------


def test_auto_train_val_binds_the_selections_own_partition(tmp_path: Path):
    """The loaders hold exactly the samples the selection recorded on each side, read by their
    own paths: nothing is rediscovered from a directory, both capture dates reach the samples the
    loaders will actually read, and the calibration side builds neither loader."""
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)
    data_cfg = _run_data_cfg(root, out)

    train_ds, val_ds, partition = auto_train_val("detection", data_cfg, None)

    assert sorted(train_ds.stems) == sorted(s.identity for s in drawn.on("train"))
    assert sorted(val_ds.stems) == sorted(s.identity for s in drawn.on("val"))
    held_out = {s.identity for s in drawn.on("calibration")}
    assert held_out and not held_out & set(train_ds.stems + val_ds.stems)
    # The sources the two loaders will open, not a record written beside them.
    read_from = {Path(train_ds.sample_sources[key]).parent.name for key in train_ds.stems}
    read_from |= {Path(val_ds.sample_sources[key]).parent.name for key in val_ds.stems}
    assert read_from == set(DATES)

    binding = data_cfg["split"]["selection_binding"]
    assert binding["selection_dir"] == str(out)
    assert binding["calibration_bound"] == len(held_out)
    assert "date" not in binding
    # The class space is the run's own, recorded once on the data config, never restated here.
    assert data_cfg["subject"] == SUBJECT
    assert not {"subject", "attribute", "id_map"} & set(binding)
    assert sorted(partition) == sorted(
        {str(Path(s.ground_truth).parent) for s in drawn.samples})


def _pixels_and_ground_truth(under: Path) -> dict[Path, int]:
    """Every image and label document under ``under`` outside a ``.tcip`` state tree, by size.

    Every tree in the workspace rather than one dataset's own, so a loader that copied its samples
    into a derived folder elsewhere would still be caught; the platform's own state directories
    are left out because a run writes its records there by design.
    """
    suffixes = {".jpg", ".png", ".tif", ".tiff", ".npy", ".json", ".bandgroup"}
    return {p: p.stat().st_size for p in under.rglob("*")
            if p.is_file() and p.suffix.lower() in suffixes and ".tcip" not in p.parts}


def test_a_multi_date_selection_trains_without_copying_anything(tmp_path: Path):
    """One selection spanning two capture dates trains for real through the worker, reading the
    images where they already sit.

    The whole worker path runs (``auto_train_val``, the class-metadata stamp, the envelope, the
    partition record), not a loader built by hand: the run's own recorded partition names members
    from both label directories, the checkpoint carries the selection's own class map, and no
    image or label document anywhere in the workspace outside a ``.tcip`` state tree was copied,
    moved or rewritten. What the loaders themselves read is
    :func:`test_auto_train_val_binds_the_selections_own_partition`'s subject.
    """
    import tcip_store as ts

    import tcip_mcp.tools.training_tools as ttools
    from tcip_mcp.experiments import create_experiment, read_run_partition, update_status
    from tcip_mcp.pipelines.training import subprocess_worker as worker

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1, "min_size": 64, "max_size": 64},
                         "task": "detection"},
        "data": {"split": {"selection_dir": str(out)}},
        "batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
        "mixed_precision": False, "device": "cpu",
        "checkpoint_every_n_epochs": 0, "early_stopping": {"enabled": False},
    }
    create_experiment("exp-multi-date", config)
    update_status("exp-multi-date", "running")
    ts.replace(ttools.launch_config_key(run_dir), config)

    before = _pixels_and_ground_truth(tmp_path)
    worker.run("exp-multi-date", str(run_dir), "")

    assert _pixels_and_ground_truth(tmp_path) == before, (
        "a bound run reads the dataset's own imagery and ground truth; nothing is copied")

    # Both dates reach the members the run actually recorded, from two label directories.
    partition = read_run_partition("exp-multi-date")
    assert {Path(d).name for d in partition["members"]} == set(DATES)
    consumed = set(recorded_side(partition["members"], "train")) | set(
        recorded_side(partition["members"], "val"))
    assert consumed == {Path(s.ground_truth).stem
                        for s in drawn.on("train") + drawn.on("val")}
    for date in DATES:
        block = next(b for d, b in partition["members"].items() if Path(d).name == date)
        assert block["train"] or block["val"]

    # The checkpoint speaks the selection's own vocabulary, not one re-read from the registry.
    checkpoint = torch.load(run_dir / "model_final.pt", map_location="cpu", weights_only=False)
    stamped = (checkpoint.get("config") or {}).get("data") or {}
    assert stamped["subject"] == SUBJECT
    assert stamped["id_map"] == drawn.id_map


def test_a_bound_run_keeps_its_selections_class_map_when_the_registry_is_reordered(
    tmp_path: Path,
):
    """The registry's declared order changes between the draw and the run. The checkpoint must
    still speak the vocabulary its samples were admitted under: a map re-resolved from the live
    registry here would stamp ids the model never trained in, which is a wrong value rather than
    a missing one."""
    import tcip_store as ts

    import tcip_mcp.tools.training_tools as ttools
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.pipelines.training import subprocess_worker as worker

    root = _attribute_scoped_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out, attribute="condition", seed=1)
    assert drawn.id_map == {"healthy": 0, "damaged": 1}

    # The same attribute, its values declared the other way round: a map re-derived here would
    # be {"damaged": 0, "healthy": 1}, a different class space than the samples were admitted in.
    write_registry(root / "subjects.json", SubjectRegistry(subjects=(
        Subject(name=SUBJECT, attributes=(
            Attribute(name="condition", type="categorical", values=("damaged", "healthy")),
        )),
    )))

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 2, "min_size": 64, "max_size": 64},
                         "task": "detection"},
        # The directories a relaunched config still carries beside its binding: exactly what a
        # registry re-read would resolve the wrong map from.
        "data": {"images_dir": str(root / "images" / DATES[0]),
                 "labels_dir": str(root / "annotations" / DATES[0]),
                 "split": {"selection_dir": str(out)}},
        "batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
        "mixed_precision": False, "device": "cpu",
        "checkpoint_every_n_epochs": 0, "early_stopping": {"enabled": False},
    }
    create_experiment("exp-reordered", config)
    update_status("exp-reordered", "running")
    ts.replace(ttools.launch_config_key(run_dir), config)

    worker.run("exp-reordered", str(run_dir), "")

    checkpoint = torch.load(run_dir / "model_final.pt", map_location="cpu", weights_only=False)
    assert ((checkpoint.get("config") or {}).get("data") or {})["id_map"] == drawn.id_map


def test_a_bound_run_trains_in_the_selections_scope_over_a_stale_config_one(tmp_path: Path):
    """A relaunched config carrying the previous run's own subject states nothing about this run:
    the selection is what says which class space its samples were admitted under, so that scope
    becomes the run's and the stale one is overwritten rather than compared against it."""
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)
    data_cfg = _run_data_cfg(root, out, subject=OTHER_SUBJECT)

    train_ds, val_ds, _partition = auto_train_val("detection", data_cfg, None)

    assert data_cfg["subject"] == drawn.subject == SUBJECT
    assert data_cfg["id_map"] == drawn.id_map
    assert train_ds.subject == SUBJECT and val_ds.subject == SUBJECT
    assert sorted(train_ds.stems) == sorted(s.identity for s in drawn.on("train"))


def test_a_selected_label_emptied_since_the_draw_refuses_the_run(tmp_path: Path):
    """A label that held the subject at draw time is emptied with nobody confirming that image
    negative. Training it would put a real object's pixels in the background class, so the bind
    refuses by name rather than reading the empty document as a negative."""
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)
    emptied = drawn.on("train")[0]
    json_io.write_annotations(emptied.ground_truth, [], 64, 64, keep_empty=True)

    with pytest.raises(ValueError, match="no longer admissible"):
        auto_train_val("detection", _run_data_cfg(root, out), None)


def test_a_selected_label_a_human_confirmed_negative_still_trains(tmp_path: Path):
    """The admitting half of that rail: the same emptied label, this time with a human marking
    the image negative for this subject, is a real negative and the run binds."""
    from tcip_mcp.dataset_layout import record_image_statuses, status_bucket
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)
    emptied = drawn.on("train")[0]
    json_io.write_annotations(emptied.ground_truth, [], 64, 64, keep_empty=True)
    record_image_statuses(
        root, status_bucket(SUBJECT, Path(emptied.ground_truth).parent.name),
        {Path(emptied.source).name: "negative"}, recorded_by="user:tester")

    train_ds, val_ds, partition = auto_train_val("detection", _run_data_cfg(root, out), None)

    assert len(train_ds) == len(drawn.on("train"))
    assert len(val_ds) == len(drawn.on("val"))
    assert Path(emptied.ground_truth).stem in set(recorded_side(partition, "train"))


def test_a_bound_run_admits_when_an_unselected_images_stem_turns_ambiguous(tmp_path: Path):
    """The recheck reads each sample's own two recorded paths, not a listing of the directory the
    sample happens to sit in. An image nobody selected picking up a second file under one stem
    makes that directory unlistable, and a recheck that listed it would refuse a selection whose
    every sample still names one real file."""
    from PIL import Image

    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    images_dir = root / "images" / DATES[0]
    Image.new("RGB", (64, 64), (10, 10, 10)).save(images_dir / "spare.jpg")
    out = tmp_path / "m"
    drawn = _draw(root, out)
    assert "spare" not in {Path(s.source).stem for s in drawn.samples}

    # The same logical image now resolves to two files: the directory can no longer be listed.
    Image.new("RGB", (64, 64), (20, 20, 20)).save(images_dir / "spare.png")

    train_ds, val_ds, _partition = auto_train_val("detection", _run_data_cfg(root, out), None)

    assert len(train_ds) == len(drawn.on("train"))
    assert len(val_ds) == len(drawn.on("val"))


def test_a_positive_named_unlike_its_image_contradicts_a_stale_negative(tmp_path: Path):
    """A sample's ground truth and its image are two trees, and a sample may name them
    independently. A human's negative for that image, stamped under a since-changed schema, is
    contradicted by the positive the sample actually records, so the sample is admitted on the
    content it names rather than quarantined for a document nobody said had to sit beside the
    image.

    The sample comes back through ``read_selection``, the platform's own reader of a recorded
    selection, which accepts an independently named pair; ``directory_samples`` cannot yet draw
    one, so no draw produces this shape today.
    """
    from tcip_mcp.dataset_layout import (
        record_image_statuses, stamp_image_status_digests, status_bucket,
    )
    from tcip_mcp.pipelines.data.label_queries import refuse_inadmissible_samples
    from tcip_mcp.pipelines.data.selection import (
        Sample, Selection, read_selection, write_selection,
    )

    from PIL import Image

    root = tmp_path / "ds"
    images_dir, labels_dir = root / "images" / DATES[0], root / "annotations" / DATES[0]
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (100, 120, 90)).save(images_dir / "photo.jpg")
    write_registry(root / "subjects.json", SubjectRegistry(subjects=(Subject(name=SUBJECT),)))
    # The positive this sample records is the only one there is, and it is not beside the image.
    json_io.write_annotations(labels_dir / "reviewed.json",
                              [Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20))],
                              64, 64, keep_empty=True)

    bucket = status_bucket(SUBJECT, DATES[0])
    record_image_statuses(root, bucket, {"photo.jpg": "negative"}, recorded_by="user:tester")
    stamp_image_status_digests(root, bucket, ["photo.jpg"], "a-schema-since-changed")

    out = tmp_path / "m"
    write_selection(out, Selection(samples=(Sample(
        source=str(images_dir / "photo.jpg"), ground_truth=str(labels_dir / "reviewed.json"),
        group="g", side="train", confirmation_bucket=bucket),
    ), subject=SUBJECT, id_map={SUBJECT: 0}))

    selection = read_selection(out)
    refuse_inadmissible_samples(selection.samples, selection.scope)


def test_a_sample_naming_a_row_of_its_ground_truth_refuses_the_geometry_loaders(tmp_path: Path):
    """``row_key`` names one row inside a ground truth that answers for many samples. A geometry
    loader reads a per-image document, and reading the file whole would take a document answering
    for many samples for a per-image one, so it refuses by naming the ground truth it does read
    rather than training on whatever the whole file holds. The same selection without the row key
    builds.

    The record comes back through ``read_selection``, the platform's own reader: the field is part
    of the recorded shape, which is why a loader has to answer for it rather than ignore it.
    """
    from PIL import Image

    from tcip_mcp.dataset_layout import status_bucket
    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.data.selection import (
        ClassScope, Sample, Selection, read_selection, write_selection,
    )

    root = tmp_path / "ds"
    images_dir, labels_dir = root / "images" / DATES[0], root / "annotations" / DATES[0]
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (100, 120, 90)).save(images_dir / "a.jpg")
    write_registry(root / "subjects.json", SubjectRegistry(subjects=(Subject(name=SUBJECT),)))
    json_io.write_annotations(labels_dir / "a.json",
                              [Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20))],
                              64, 64, keep_empty=True)

    def _selection(row_key: str | None) -> Selection:
        out = tmp_path / ("rows" if row_key else "whole")
        write_selection(out, Selection(samples=(Sample(
            source=str(images_dir / "a.jpg"), ground_truth=str(labels_dir / "a.json"),
            group="g", side="train", confirmation_bucket=status_bucket(SUBJECT, DATES[0]),
            row_key=row_key),
        ), subject=SUBJECT, id_map={SUBJECT: 0}))
        return read_selection(out)

    scope = ClassScope(subject=SUBJECT, id_map={SUBJECT: 0})
    with pytest.raises(ValueError, match="a detection loader does not read"):
        build_dataset("detection", samples=_selection("a.jpg").samples, scope=scope)

    admitted = build_dataset("detection", samples=_selection(None).samples, scope=scope)
    assert list(admitted.stems) == [str(images_dir / "a.jpg")]


def test_crops_of_one_parent_cannot_cross_sides(tmp_path: Path):
    """Every crop of one parent image shares a group key, and the reader refuses a selection
    whose group keys straddle two sides, so a run bound to one can never train on one crop of a
    parent and validate on another."""
    from tcip_mcp.pipelines.data.selection import as_selection, selection_document
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _tiled_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out, seed=1)

    side_of_parent: dict[str, str] = {}
    for sample in drawn.samples:
        parent = Path(sample.ground_truth).stem.rsplit("_", 2)[0]
        assert side_of_parent.setdefault(parent, sample.side) == sample.side
    assert len(side_of_parent) == 4

    data_cfg = _run_data_cfg(root, out)
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    parents_on = {
        side: {Path(stem).stem.rsplit("_", 2)[0] for stem in ds.stems}
        for side, ds in (("train", train_ds), ("val", val_ds))
    }
    assert not parents_on["train"] & parents_on["val"]

    # And the record itself refuses a hand-made partition that splits one parent's crops.
    document = selection_document(drawn)
    a_train = next(s for s in document["samples"] if s["side"] == "train")
    moved = dict(a_train, side="val")
    document["samples"] = [s for s in document["samples"] if s is not a_train] + [moved]
    with pytest.raises(ValueError, match="group"):
        as_selection(document, where="hand-made")


def test_auto_train_val_admits_a_confirmed_negative_the_draw_admitted(tmp_path: Path):
    """A stem whose label file is empty and whose image a human marked negative was admitted at
    the draw, so the bound run trains on it; the loader reads it as a zero-object sample rather
    than re-deciding what a negative is."""
    from tcip_mcp.dataset_layout import record_image_statuses, status_bucket
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _dataset_with_a_confirmed_negative(tmp_path / "ds")
    record_image_statuses(root, status_bucket(SUBJECT, DATES[0]), {"n.jpg": "negative"},
                          recorded_by="user:tester")
    out = tmp_path / "m"
    drawn = _draw(root, out, seed=1)
    assert "n" in {Path(s.ground_truth).stem for s in drawn.samples}

    train_ds, val_ds, _ = auto_train_val("detection", _run_data_cfg(root, out), None)

    negatives = [s for s in drawn.on("train") + drawn.on("val")
                 if Path(s.ground_truth).stem == "n"]
    assert negatives, "the negative landed on neither loader's side"
    loader = train_ds if negatives[0].identity in train_ds.stems else val_ds
    _image, target = loader[loader.stems.index(negatives[0].identity)]
    assert target["boxes"].shape[0] == 0


def test_an_unconfirmed_empty_label_never_reaches_a_bound_run(tmp_path: Path):
    """The negative invariant holds at the draw, so it holds at the bind: an empty label file
    nobody confirmed is admitted by nothing and appears in no selection."""
    root = _dataset_with_a_confirmed_negative(tmp_path / "ds")
    out = tmp_path / "m"

    drawn = _draw(root, out, seed=1)

    assert "n" not in {Path(s.ground_truth).stem for s in drawn.samples}


def test_a_bound_run_reads_class_ids_from_the_selections_own_map(tmp_path: Path):
    """The selection records the ``assign_class_ids`` map its admission used, and the loaders
    read that map: an edited registry cannot relabel a bound run's targets."""
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _attribute_scoped_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out, attribute="condition")
    assert drawn.id_map == {"healthy": 0, "damaged": 1}

    train_ds, _val_ds, _ = auto_train_val("detection", _run_data_cfg(root, out), None)

    assert train_ds.id_map == drawn.id_map
    assert train_ds.num_classes == 2
    assert train_ds.subject == SUBJECT
    assert train_ds.attribute == "condition"


def test_a_bound_run_threads_a_bespoke_dataset_source(tmp_path: Path):
    """The bespoke seam stays reachable from a selection-bound run: the agent's own builder
    receives the selection's samples and the scope it was drawn under."""
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)
    data_cfg = _run_data_cfg(root, out)
    data_cfg["dataset_source"] = {
        "builder": f"{__name__}:build_recording_dataset", "task": "detection",
    }

    train_ds, _val_ds, _ = auto_train_val("detection", data_cfg, None)

    assert sorted(s.identity for s in train_ds.seen_samples) == sorted(
        s.identity for s in drawn.on("train"))
    assert train_ds.seen_id_map == drawn.id_map


@pytest.mark.parametrize("task", ["detection", "canopy_extent"])
def test_an_unbound_bespoke_run_is_handed_the_same_samples_a_bound_one_is(tmp_path: Path, task):
    """Two producers of one fact, compared against each other rather than against a fixture.

    The same tree is trained twice through a bespoke builder, once bound to a selection drawn over
    it and once unbound over the same directory; the builder records what it was handed each time.
    The two runs partition the tree differently, which is what each route is for; what they may
    not do is disagree about which samples that tree holds, where each one's ground truth is, or
    what class map they were admitted under. Neither is handed a directory to go looking in.

    Run for a task with a built-in loader and for one without: a task with no built-in loader is
    not a task with no producer, so both routes name its samples the same way too.
    """
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)
    source = {"builder": f"{__name__}:build_recording_dataset", "task": task}

    bound_cfg = _run_data_cfg(root, out)
    bound_cfg["dataset_source"] = source
    bound_train, bound_val, _partition = auto_train_val(task, bound_cfg, None)

    unbound_cfg = {"images_dir": str(root / "images" / DATES[0]),
                   "labels_dir": str(root / "annotations" / DATES[0]),
                   "subject": SUBJECT, "dataset_source": source,
                   "split": {"val_ratio": 0.5, "seed": 3}}
    unbound_train, unbound_val, _unbound_partition = auto_train_val(
        task, unbound_cfg, None)

    assert bound_val is not None and unbound_val is not None, \
        "each route draws both sides over the samples its producer named"

    def handed(*datasets) -> list[tuple]:
        """Every sample each builder was handed, with multiplicity, as the facts that must agree.

        The side is normalized away and only that: which side a sample landed on is what the two
        routes legitimately differ about, and everything else about a sample is what they may not.
        """
        seen: list[tuple] = []
        for dataset in datasets:
            for sample in dataset.seen_samples:
                seen.append((sample.source, sample.ground_truth, sample.row_key, sample.rect,
                             sample.group, sample.confirmation_bucket, sample.member_stem))
        return sorted(seen)

    # Comparable universes: the unbound run admits one date, so the bound run's members from the
    # other one are not members the unbound run could have held.
    bound_here = [s for s in handed(bound_train, bound_val)
                  if Path(s[0]).parent.name == DATES[0]]
    unbound_here = handed(unbound_train, unbound_val)
    held_out = {s.identity for s in drawn.on("calibration")}
    unbound_trained = [s for s in unbound_here if s[0] not in held_out]

    assert bound_here and unbound_trained
    assert len(bound_here) == len(set(bound_here)), "each sample is handed once, not repeated"
    assert bound_here == unbound_trained
    assert unbound_train.seen_id_map == bound_train.seen_id_map == drawn.id_map
    # Nothing the builder can go looking in: a directory would let it read its own membership.
    for dataset in (bound_train, bound_val, unbound_train, unbound_val):
        assert dataset.seen_directories == {}


def test_the_preflight_smoke_batch_is_the_batch_the_bound_run_trains(tmp_path: Path):
    """The batch a preflight smokes comes off the run's own training loader, resolved the way the
    run resolves it. A bound config names no directories at all, and re-admitting one here would
    smoke a batch holding the validation and calibration members the run never trains on, which is
    not the batch whose measurement boundary the contract proves."""
    from tcip_mcp.tools.training_tools import _one_real_batch

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)
    config = {"data": _run_data_cfg(root, out)}
    config["data"]["dataset_source"] = {
        "builder": f"{__name__}:build_recording_dataset", "task": "detection",
    }

    before = len(_RECORDED_BUILDS)
    batch, why = _one_real_batch("detection", config)

    assert why is None and batch is not None
    smoked = _RECORDED_BUILDS[before]  # the training side, built first
    assert sorted(s.identity for s in smoked.seen_samples) == sorted(
        s.identity for s in drawn.on("train"))
    held_out = {s.identity for s in drawn.on("val") + drawn.on("calibration")}
    assert not held_out & {s.identity for s in smoked.seen_samples}
    # The caller's own config is left exactly as it was found: preflight reads, never binds.
    assert "selection_binding" not in config["data"]["split"]


def test_auto_train_val_refuses_a_selection_with_an_empty_side(tmp_path: Path):
    from tcip_mcp.pipelines.data.selection import Selection, write_selection
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)
    write_selection(out, Selection(
        samples=tuple(s for s in drawn.samples if s.side != "val"),
        subject=drawn.subject, attribute=drawn.attribute, id_map=drawn.id_map,
        seed=drawn.seed, group_by=drawn.group_by,
    ))

    with pytest.raises(ValueError, match="empty side"):
        auto_train_val("detection", _run_data_cfg(root, out), None)


def test_auto_train_val_refuses_a_selection_with_no_subject(tmp_path: Path):
    from tcip_mcp.pipelines.data.selection import Selection, write_selection
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)
    write_selection(out, Selection(samples=drawn.samples, subject=None, id_map=drawn.id_map))

    with pytest.raises(ValueError, match="no subject"):
        auto_train_val("detection", _run_data_cfg(root, out), None)


def test_auto_train_val_selection_conflicts_with_a_drawn_splits_own_parameters(tmp_path: Path):
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    data_cfg = _run_data_cfg(root, out)
    data_cfg["split"]["val_ratio"] = 0.3

    with pytest.raises(ValueError, match="val_ratio"):
        auto_train_val("detection", data_cfg, None)


def test_auto_train_val_selection_refuses_a_task_reading_another_ground_truth(tmp_path: Path):
    """What decides whether a task can bind a selection is the ground truth its samples name, not
    the task's name: a selection of per-image label documents refuses a semantic segmentation run,
    whose loader reads a mask raster, and names the ground truth that loader does read."""
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)

    with pytest.raises(ValueError, match="a semantic_seg loader does not read"):
        auto_train_val("semantic_seg", _run_data_cfg(root, out), None)


def test_auto_train_val_binding_failure_raises_rather_than_degrading(tmp_path: Path):
    """A sample whose image has moved since the draw refuses at the bind, naming it; it never
    degrades into a run that trains on whatever is left with no validation."""
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)
    gone = drawn.on("train")[0]
    Path(gone.source).unlink()

    with pytest.raises(ValueError, match="no longer admissible"):
        auto_train_val("detection", _run_data_cfg(root, out), None)


# -- redraw_within_selection ---------------------------------------------------


def _redraw_cfg(root: Path, out: Path, seed: int) -> dict:
    data_cfg = _run_data_cfg(root, out)
    data_cfg["split"]["redraw_within_selection"] = True
    data_cfg["split"]["seed"] = seed
    return data_cfg


def test_a_redraw_repartitions_the_selections_own_members_and_leaves_calibration_alone(
    tmp_path: Path,
):
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)
    pool = {s.identity for s in drawn.on("train") + drawn.on("val")}
    held_out = {s.identity for s in drawn.on("calibration")}

    data_cfg = _redraw_cfg(root, out, seed=1)
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)

    assert set(train_ds.stems) | set(val_ds.stems) == pool
    assert not (set(train_ds.stems) | set(val_ds.stems)) & held_out
    assert data_cfg["split"]["selection_binding"]["redraw"]["seed"] == 1
    assert data_cfg["split"]["resolved_seed"] == 1


def test_two_redraw_seeds_differ_and_the_same_seed_repeats(tmp_path: Path):
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)

    def _train(seed: int) -> list[str]:
        train_ds, _val, _ = auto_train_val("detection", _redraw_cfg(root, out, seed), None)
        return sorted(train_ds.stems)

    first = _train(1)
    assert _train(1) == first
    assert any(_train(seed) != first for seed in range(2, 8))


def test_a_redraw_without_a_seed_refuses(tmp_path: Path):
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    data_cfg = _run_data_cfg(root, out)
    data_cfg["split"]["redraw_within_selection"] = True

    with pytest.raises(ValueError, match="requires data.split.seed"):
        auto_train_val("detection", data_cfg, None)


def test_a_seed_without_the_redraw_flag_still_conflicts(tmp_path: Path):
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    data_cfg = _run_data_cfg(root, out)
    data_cfg["split"]["seed"] = 3

    with pytest.raises(ValueError, match="seed"):
        auto_train_val("detection", data_cfg, None)


def test_redraw_starved_issue_names_the_selection_the_seed_and_both_counts(tmp_path: Path):
    """The check every pre-Start caller shares, over a selection's own record: fewer than two
    foreground groups among its train-plus-val members is a redraw that can only leave a side
    empty, named with the selection, the seed and both group counts rather than a bare failure.
    A selection whose members do hold two is admitted."""
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.pipelines.data.splits import redraw_starved_issue

    drawn_dir = tmp_path / "drawn"
    _draw(_two_subject_two_date_dataset(tmp_path / "ds"), drawn_dir)
    assert redraw_starved_issue(
        read_selection(drawn_dir), selection_dir=str(drawn_dir), seed=3) is None

    _root, one_group = one_foreground_group_selection(tmp_path / "one")
    starved = redraw_starved_issue(
        read_selection(one_group), selection_dir=str(one_group), seed=3)
    assert starved is not None
    assert f"{str(one_group)!r}" in starved and "seed 3" in starved
    assert "1 foreground group" in starved and "2 distinct group" in starved
    assert "redraw_within_selection" in starved


def one_foreground_group_selection(tmp_path: Path) -> tuple[Path, Path]:
    """A dataset plus a selection over it whose train-and-val members hold one foreground group:
    the val side's only group is a confirmed negative. Returns ``(root, selection_dir)``."""
    from tcip_mcp.dataset_layout import record_image_statuses, status_bucket
    from tcip_mcp.pipelines.data.selection import Sample, Selection, write_selection

    root = tmp_path / "ds"
    write_registry(root / "subjects.json", SubjectRegistry(subjects=(Subject(name=SUBJECT),)))
    images_dir, labels_dir = root / "images" / DATES[0], root / "annotations" / DATES[0]
    for stem in ("fg", "neg", "held_a", "held_b"):
        _write_stem(images_dir, labels_dir, stem, [] if stem == "neg" else
                   [Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20))])
    record_image_statuses(root, status_bucket(SUBJECT, DATES[0]), {"neg.jpg": "negative"},
                          recorded_by="user:tester")

    def _sample(stem: str, side: str) -> Sample:
        return Sample(source=str(images_dir / f"{stem}.jpg"),
                      ground_truth=str(labels_dir / f"{stem}.json"), group=stem, side=side,
                      confirmation_bucket=status_bucket(SUBJECT, DATES[0]))

    out = tmp_path / "m"
    write_selection(out, Selection(
        samples=(_sample("fg", "train"), _sample("neg", "val"),
                 _sample("held_a", "calibration"), _sample("held_b", "calibration")),
        subject=SUBJECT, id_map={SUBJECT: 0}, seed=1, group_by="explicit_map"))
    return root, out


def test_preflight_flags_a_redraw_whose_members_hold_one_foreground_group(tmp_path: Path):
    """A train and a val side are always two distinct groups (the reader refuses one group on
    both), so the case left is a side whose only group carries no foreground at all: a confirmed
    negative. Preflight names it before Start rather than letting every trial draw it again."""
    from tcip_mcp.tools.training_tools import preflight_config

    root, out = one_foreground_group_selection(tmp_path)
    config = _preflight_config(root, out)
    config["data"]["split"].update({"redraw_within_selection": True, "seed": 1})

    result = preflight_config(config)

    assert any("foreground group" in issue for issue in result["issues"]), result["issues"]


# -- preflight_config's selection-level checks ---------------------------------


def _preflight_config(root: Path, selection_dir: Path, **overrides) -> dict:
    data_cfg = _run_data_cfg(root, selection_dir, **overrides)
    return {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": data_cfg, "batch_size": 2,
    }


def test_preflight_config_admits_a_bound_selection_with_no_issues(tmp_path: Path):
    from tcip_mcp.tools.training_tools import preflight_config

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)

    result = preflight_config(_preflight_config(root, out))

    # A bound config names no images_dir or labels_dir: the selection's samples carry their own
    # paths, so the directory checks must not raise an objection against it.
    assert result["issues"] == []


def test_preflight_config_flags_a_selection_with_no_subject(tmp_path: Path):
    from tcip_mcp.pipelines.data.selection import Selection, write_selection
    from tcip_mcp.tools.training_tools import preflight_config

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)
    write_selection(out, Selection(samples=drawn.samples, subject=None, id_map=drawn.id_map))

    result = preflight_config(_preflight_config(root, out))

    assert any("records no subject" in i for i in result["issues"])


def test_preflight_config_admits_the_redraw_pair_with_a_warning(tmp_path: Path):
    from tcip_mcp.tools.training_tools import preflight_config

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    config = _preflight_config(root, out)
    config["data"]["split"].update({"redraw_within_selection": True, "seed": 1})

    result = preflight_config(config)

    assert result["issues"] == []
    assert any("redraw_within_selection=true" in w for w in result["warnings"])


# -- the run's own recorded partition ------------------------------------------


def test_persist_run_partition_carries_the_selection_binding(tmp_path: Path):
    import tcip_store as ts

    from tcip_mcp.experiments import create_experiment, split_key
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_run_partition

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)
    data_cfg = _run_data_cfg(root, out)
    train_ds, val_ds, partition = auto_train_val("detection", data_cfg, None)

    create_experiment("exp-bound", {"data": data_cfg})
    persist_run_partition("exp-bound", data_cfg, partition=partition)

    record = ts.read(split_key("exp-bound"))
    assert record["selection_binding"]["selection_dir"] == str(out)
    members = record["members"]
    assert recorded_side(members, "train") == sorted(
        {Path(s.ground_truth).stem for s in drawn.on("train")})
    assert recorded_side(members, "val") == sorted(
        {Path(s.ground_truth).stem for s in drawn.on("val")})
    assert sorted(members) == sorted(partition)
    # The selection's own named grouping policy, carried rather than collapsed into the finite
    # per-stem map beside it: a stem the map does not cover is what the policy answers for.
    assert record["group_by"] == drawn.group_by == "tile_prefix"
    for block in record["members"].values():
        assert block["group_key_map"]
        assert block["label_digests"]["at_split"]
    assert "redrawn_within_selection" not in record


def test_persist_run_partition_carries_no_stale_binding_when_this_run_did_not_bind(tmp_path: Path):
    import tcip_store as ts

    from tcip_mcp.experiments import create_experiment, split_key
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_run_partition

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    data_cfg = _run_data_cfg(root, out)
    auto_train_val("detection", data_cfg, None)

    # The same config, relaunched with the binding dropped: it draws its own split instead.
    data_cfg["images_dir"] = str(root / "images" / DATES[0])
    data_cfg["labels_dir"] = str(root / "annotations" / DATES[0])
    data_cfg["subject"] = SUBJECT
    data_cfg["split"].pop("selection_dir")
    train_ds, val_ds, partition = auto_train_val("detection", data_cfg, None)

    create_experiment("exp-drawn", {"data": data_cfg})
    persist_run_partition("exp-drawn", data_cfg, partition=partition)

    record = ts.read(split_key("exp-drawn"))
    assert "selection_binding" not in record
    # A drawn run records a partition of its own; what must not survive is the earlier binding.
    assert partition is not None
    assert recorded_side(record["members"], "train")
    assert recorded_side(record["members"], "val")
