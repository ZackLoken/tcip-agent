"""Training and tuning binding to a named split manifest (``data.split.manifest_dir``).

The manifest is drawn by ``make_splits`` (see ``test_data_tools.py``); this file covers the
consumer side: ``bind_manifest_stems``, ``read_split_manifest_dir``, and ``auto_train_val``'s own
branch that binds a run to one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.class_registry import Attribute, ClassRegistry, Subject, write_registry
from tcip_mcp.tools.data_tools import make_splits

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
    clearing a leaf-scoped manifest write's floor), and four of the six also carry the unrelated
    ``bud`` (eight foreground groups, clearing a bud-scoped write's floor too), so a manifest
    drawn for either subject binds to a real, differently-sized draw over the identical tree."""
    write_registry(root / "classes.json", ClassRegistry(subjects=(
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
    attribute-scoped manifest write's floor), the fifth carries an instance never assessed for
    it."""
    write_registry(root / "classes.json", ClassRegistry(subjects=(
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


def _draw(root: Path, out: Path, *, subject: str = SUBJECT, attribute: str | None = None,
         seed: int = 2) -> dict:
    import tcip_store as ts
    from tcip_mcp.tools.data_tools import split_manifest_key

    result = make_splits(str(root), output_path=str(out), subject=subject, attribute=attribute,
                         seed=seed, train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in result, result
    return ts.read(split_manifest_key(out))


# -- bind_manifest_stems -------------------------------------------------------


def test_bind_manifest_stems_binds_the_manifests_own_partition_for_one_date(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import bind_manifest_stems

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    manifest = _draw(root, tmp_path / "m")

    binding = bind_manifest_stems(
        manifest, DATES[0], SUBJECT, None, ["a", "b", "c", "d", "e", "f"])

    assert binding.train and binding.val
    assert sorted(binding.train + binding.val + binding.calibration) == \
        ["a", "b", "c", "d", "e", "f"]
    assert binding.assigned == 6


def test_bind_manifest_stems_refuses_a_subject_mismatch(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import bind_manifest_stems

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    manifest = _draw(root, tmp_path / "m")

    with pytest.raises(ValueError, match="subject"):
        bind_manifest_stems(manifest, DATES[0], OTHER_SUBJECT, None, ["b"])


def test_bind_manifest_stems_refuses_an_attribute_mismatch(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import bind_manifest_stems

    root = _attribute_scoped_dataset(tmp_path / "ds")
    manifest = _draw(root, tmp_path / "m", subject=SUBJECT, attribute="condition")

    with pytest.raises(ValueError, match="attribute"):
        bind_manifest_stems(manifest, DATES[0], SUBJECT, None, ["assessed_a", "assessed_b"])


def test_bind_manifest_stems_refuses_a_date_the_manifest_holds_no_members_under(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import bind_manifest_stems

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    manifest = _draw(root, tmp_path / "m")

    with pytest.raises(ValueError, match="2026-01-01"):
        bind_manifest_stems(manifest, "2026-01-01", SUBJECT, None, [])


def test_bind_manifest_stems_refuses_an_admitted_stem_assigned_to_neither_side(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import bind_manifest_stems

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    manifest = _draw(root, tmp_path / "m")

    with pytest.raises(ValueError, match="no side"):
        bind_manifest_stems(manifest, DATES[0], SUBJECT, None,
                            ["a", "b", "c", "d", "e", "f", "g"])


def test_bind_manifest_stems_refuses_a_manifest_member_the_run_does_not_admit(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import bind_manifest_stems

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    manifest = _draw(root, tmp_path / "m")

    with pytest.raises(ValueError, match="not in this run's admitted"):
        bind_manifest_stems(manifest, DATES[0], SUBJECT, None, ["a"])


def test_bind_manifest_stems_refuses_an_empty_side_after_binding(tmp_path: Path):
    """No manifest write can draw an empty side any more (every ratio is refused at zero): an
    empty side after binding is exercised on a real draw with its own val members for one date
    moved onto train through the store, the shape a manifest predating that rail would read as."""
    from tcip_mcp.pipelines.data.splits import bind_manifest_stems, member_identity_parts

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    manifest = _draw(root, tmp_path / "m")
    moved = [i for i in manifest["splits"]["val"] if member_identity_parts(i)[0] == DATES[0]]
    manifest["splits"]["val"] = [i for i in manifest["splits"]["val"] if i not in moved]
    manifest["splits"]["train"] = sorted(manifest["splits"]["train"] + moved)

    with pytest.raises(ValueError, match="empty side"):
        bind_manifest_stems(manifest, DATES[0], SUBJECT, None,
                            ["a", "b", "c", "d", "e", "f"])


def test_bind_manifest_stems_places_a_calibration_stem_on_neither_loader(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import bind_manifest_stems

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    manifest = _draw(root, tmp_path / "m")

    binding = bind_manifest_stems(
        manifest, DATES[0], SUBJECT, None, ["a", "b", "c", "d", "e", "f"])

    assert binding.calibration
    assert set(binding.calibration).isdisjoint(binding.train)
    assert set(binding.calibration).isdisjoint(binding.val)
    assert binding.calibration_bound == len(binding.calibration)
    assert binding.calibration_unadmitted == 0


def test_bind_manifest_stems_a_missing_calibration_member_does_not_refuse(tmp_path: Path):
    """A calibration member the run does not admit is counted, not a reason to refuse: the
    calibration door's own floor is where an insufficient calibration side bites, never a
    training launch."""
    from tcip_mcp.pipelines.data.splits import bind_manifest_stems

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    manifest = _draw(root, tmp_path / "m")
    calibration_this_date = sorted(
        i.split("/", 1)[1] for i in manifest["splits"]["calibration"]
        if i.startswith(f"{DATES[0]}/")
    )
    assert calibration_this_date, "fixture must leave a calibration member under this date"
    dropped, kept_calibration = calibration_this_date[0], calibration_this_date[1:]
    admitted = [s for s in ("a", "b", "c", "d", "e", "f") if s != dropped]

    binding = bind_manifest_stems(manifest, DATES[0], SUBJECT, None, admitted)

    assert binding.train and binding.val
    # The manifest's own calibration membership is recorded whole, admitted or not (the run
    # never trains or selects on it either way); only the unadmitted count changes.
    assert dropped in binding.calibration
    assert set(kept_calibration) <= set(binding.calibration)
    assert binding.calibration_unadmitted == 1


# -- read_split_manifest_dir ----------------------------------------------------


def test_read_split_manifest_dir_returns_the_written_manifest(tmp_path: Path):
    from tcip_mcp.tools.data_tools import read_split_manifest_dir

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = make_splits(str(root), output_path=str(out), subject=SUBJECT, seed=1,
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in result, result

    manifest = read_split_manifest_dir(out)

    assert manifest["subject"] == SUBJECT
    assert set(manifest["members"]) == set(DATES)


def test_read_split_manifest_dir_refuses_a_directory_with_no_manifest(tmp_path: Path):
    from tcip_mcp.tools.data_tools import read_split_manifest_dir

    with pytest.raises(ValueError, match="no split manifest"):
        read_split_manifest_dir(tmp_path / "nope")


def test_read_split_manifest_dir_refuses_a_record_with_no_subject_or_members(tmp_path: Path):
    import tcip_store as ts
    from tcip_mcp.tools.data_tools import read_split_manifest_dir, split_manifest_key

    out = tmp_path / "m"
    out.mkdir()
    ts.replace(split_manifest_key(out), {"seed": 1, "splits": {"train": [], "val": []}})

    with pytest.raises(ValueError, match="subject"):
        read_split_manifest_dir(out)


# -- auto_train_val's manifest branch -------------------------------------------


def _run_data_cfg(root: Path, manifest_dir: Path, date: str, *, subject: str = SUBJECT,
                  attribute: str | None = None, images_dir: Path | None = None) -> dict:
    return {
        "images_dir": str(images_dir or (root / "images" / date)),
        "labels_dir": str(root / "annotations" / date),
        "subject": subject, "attribute": attribute, "date": date,
        "split": {"manifest_dir": str(manifest_dir)},
    }


def _dataset_with_a_confirmed_negative(root: Path) -> Path:
    """One date, four annotated stems (clearing a manifest write's foreground floor) plus a
    fifth confirmed negative for ``SUBJECT``."""
    write_registry(root / "classes.json", ClassRegistry(subjects=(Subject(name=SUBJECT),)))
    images_dir, labels_dir = root / "images" / DATES[0], root / "annotations" / DATES[0]
    for stem in ("a", "b", "c", "d"):
        _write_stem(images_dir, labels_dir, stem,
                   [Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20))])
    _write_stem(images_dir, labels_dir, "n", [])
    return root


def _single_source_mosaic_dataset(root: Path) -> tuple[Path, Path]:
    """One tileable single-source image with GT spread across its extent, enough for the spatial
    single-source split to derive disjoint regions from one image alone (the same extent and
    tile geometry ``test_spatial_region_containment.py``'s own mosaic fixture uses)."""
    from PIL import Image

    images_dir, labels_dir = root / "images", root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    write_registry(root / "classes.json", ClassRegistry(subjects=(Subject(name=SUBJECT),)))
    w, h = 4000, 3000
    Image.new("RGB", (w, h), color=(90, 90, 90)).save(images_dir / "mosaic.png")
    boxes = [Annotation(subject=SUBJECT, geometry=BBox(x, y, x + 20, y + 20))
            for x in range(20, w - 20, 200) for y in range(20, h - 20, 200)]
    json_io.write_annotations(str(labels_dir / "mosaic.json"), boxes, w, h, keep_empty=True)
    return images_dir, labels_dir


def test_auto_train_val_admits_a_confirmed_negative_with_data_date_unset(tmp_path: Path):
    """A run whose ``data.date`` is left unset still reads confirmed negatives under the tree's
    own date, the date the split manifest was drawn under, so a manifest member confirmed
    negative under that date still admits."""
    from tcip_mcp.dataset_layout import record_image_statuses, status_bucket
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _dataset_with_a_confirmed_negative(tmp_path / "ds")
    record_image_statuses(root, status_bucket(SUBJECT, DATES[0]), {"n.jpg": "negative"},
                          recorded_by="user:tester")
    out = tmp_path / "m"
    manifest = _draw(root, out, seed=1)
    assert "n" in {i.split("/", 1)[1] for ids in manifest["splits"].values() for i in ids}
    data_cfg = {
        "images_dir": str(root / "images" / DATES[0]),
        "labels_dir": str(root / "annotations" / DATES[0]),
        "subject": SUBJECT, "attribute": None,
        "split": {"manifest_dir": str(out)},
    }

    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)

    assert "n" in train_ds.stems + val_ds.stems


def test_auto_train_val_binds_to_the_manifests_own_partition_for_its_date(tmp_path: Path):
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)
    data_cfg = _run_data_cfg(root, out, DATES[0])

    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)

    date_members = {s for identity in manifest["splits"]["train"] + manifest["splits"]["val"]
                   for d, s in [identity.split("/", 1)] if d == DATES[0]}
    assert sorted(train_ds.stems + val_ds.stems) == sorted(date_members)
    binding = data_cfg["split"]["manifest_binding"]
    assert binding["manifest_dir"] == str(out)
    assert binding["subject"] == SUBJECT
    assert binding["date"] == DATES[0]
    assert binding["labels_hash_now"]
    assert "train" not in binding and "val" not in binding


def test_auto_train_val_second_bind_on_the_same_config_binds_again(tmp_path: Path):
    """The write-back lands under keys the conflict check never reads, so a second bind on the
    same config dict (a bespoke ``ctx.auto_train_val`` loop reusing ``run.config``) binds again
    rather than refusing against its own first bind."""
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    data_cfg = _run_data_cfg(root, out, DATES[0])

    auto_train_val("detection", data_cfg, None)
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)

    assert train_ds.stems and val_ds.stems


def test_auto_train_val_binds_an_explicit_map_manifest_twice_and_persists_its_narrowed_map(
        tmp_path: Path):
    """A manifest drawn with an agent-derived ``group_key_map`` binds from a fresh config, binds
    again on the same config dict, and its per-date narrowed map survives into ``split.json``."""
    from tcip_mcp.experiments import create_experiment, read_split_manifest
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_split_manifest

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    # Four groups across the two dates (p1/p2/p3/p4), clearing the manifest floor; a and b
    # share p1, c and d share p2, e and f stand alone.
    stem_groups = (("a", "p1"), ("b", "p1"), ("c", "p2"), ("d", "p2"), ("e", "p3"), ("f", "p4"))
    group_key_map = {f"{date}/{stem}": group for date in DATES for stem, group in stem_groups}
    result = make_splits(str(root), output_path=str(out), subject=SUBJECT,
                         group_key_map=group_key_map, seed=1,
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in result, result
    data_cfg = _run_data_cfg(root, out, DATES[0])

    auto_train_val("detection", data_cfg, None)
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)

    assert train_ds.stems and val_ds.stems
    assert data_cfg["split"]["resolved_group_key_map"] == dict(stem_groups)

    create_experiment("exp_explicit_map_bind", {})
    persist_split_manifest("exp_explicit_map_bind", train_ds, val_ds, data_cfg)
    persisted = read_split_manifest("exp_explicit_map_bind")
    assert persisted["group_key_map"] == dict(stem_groups)


def test_preflight_config_on_a_bound_config_admits_it_again(tmp_path: Path):
    """Preflighting a config already bound once (the same conflict-key rail this file's other
    preflight tests exercise) must not itself read as a conflict."""
    from tcip_mcp.tools.training_tools import preflight_config
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    data_cfg = _run_data_cfg(root, out, DATES[0])
    auto_train_val("detection", data_cfg, None)

    result = preflight_config(_preflight_config(root, out, DATES[0], split=data_cfg["split"]))

    manifest_issues = [i for i in result["issues"] if "manifest" in i]
    assert manifest_issues == []


def test_relaunch_from_the_durable_record_binds_again(tmp_path: Path):
    """A run relaunched from the durable ``config.json`` record a bound run's own worker patched
    binds again, rather than refusing against the very block that records it bound once."""
    import tcip_store as ts
    from tcip_mcp.experiments import config_key, create_experiment
    from tcip_mcp.pipelines.training.subprocess_worker import _patch_experiment_config_split
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    data_cfg = _run_data_cfg(root, out, DATES[0])
    auto_train_val("detection", data_cfg, None)

    create_experiment("exp_relaunch_split_bind", {"data": dict(data_cfg)})
    _patch_experiment_config_split("exp_relaunch_split_bind", data_cfg["split"])
    durable_data_cfg = ts.read(config_key("exp_relaunch_split_bind"))["data"]

    train_ds, val_ds, _ = auto_train_val("detection", durable_data_cfg, None)

    assert train_ds.stems and val_ds.stems


def test_auto_train_val_binds_the_same_tree_for_the_other_subject(tmp_path: Path):
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out, subject=OTHER_SUBJECT, seed=0)
    data_cfg = _run_data_cfg(root, out, DATES[0], subject=OTHER_SUBJECT)

    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)

    # Only stems "c", "d", "e" and "f" carry the other subject on this date; the manifest's own
    # calibration side holds out some of them, so train+val is a subset, never the whole four.
    assert set(train_ds.stems + val_ds.stems) <= {"c", "d", "e", "f"}
    assert train_ds.stems and val_ds.stems


def test_auto_train_val_binds_an_attribute_scoped_tree(tmp_path: Path):
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _attribute_scoped_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = make_splits(str(root), output_path=str(out), subject=SUBJECT, attribute="condition",
                         seed=1, train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in result, result
    data_cfg = _run_data_cfg(root, out, DATES[0], attribute="condition")

    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)

    all_assessed = {"assessed_a", "assessed_b", "assessed_c", "assessed_d"}
    assert set(train_ds.stems + val_ds.stems) <= all_assessed
    assert train_ds.stems and val_ds.stems
    assert data_cfg["split"]["manifest_binding"]["attribute"] == "condition"


def test_auto_train_val_manifest_conflicts_with_val_images_dir(tmp_path: Path):
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    data_cfg = _run_data_cfg(root, out, DATES[0])
    data_cfg["val_images_dir"] = str(root / "images" / DATES[1])

    with pytest.raises(ValueError, match="val_images_dir"):
        auto_train_val("detection", data_cfg, None)


def test_auto_train_val_manifest_conflicts_with_a_drawn_splits_own_parameters(tmp_path: Path):
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    data_cfg = _run_data_cfg(root, out, DATES[0])
    data_cfg["split"]["val_ratio"] = 0.3

    with pytest.raises(ValueError, match="val_ratio"):
        auto_train_val("detection", data_cfg, None)


def test_auto_train_val_manifest_refuses_a_task_it_does_not_admit(tmp_path: Path):
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    data_cfg = _run_data_cfg(root, out, DATES[0])

    with pytest.raises(ValueError, match="semantic_seg"):
        auto_train_val("semantic_seg", data_cfg, None)


def test_auto_train_val_manifest_refuses_a_disagreeing_date(tmp_path: Path):
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    data_cfg = _run_data_cfg(root, out, DATES[0])
    data_cfg["date"] = DATES[1]

    with pytest.raises(ValueError, match="data.date"):
        auto_train_val("detection", data_cfg, None)


def test_auto_train_val_manifest_refuses_an_images_root_mismatch(tmp_path: Path):
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    other_images = tmp_path / "elsewhere"
    other_images.mkdir()
    data_cfg = _run_data_cfg(root, out, DATES[0], images_dir=other_images)

    with pytest.raises(ValueError, match="images_root"):
        auto_train_val("detection", data_cfg, None)


def test_auto_train_val_manifest_refuses_a_moved_images_root_by_name(tmp_path: Path):
    """A manifest's recorded images_root that no longer exists on disk (a moved or renamed
    dataset) answers the named refusal, never a bare crash from comparing against a gone path."""
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    moved = tmp_path / "moved_images"
    (root / "images" / DATES[0]).rename(moved)
    data_cfg = _run_data_cfg(root, out, DATES[0], images_dir=moved)

    with pytest.raises(ValueError, match="images_root"):
        auto_train_val("detection", data_cfg, None)


def test_preflight_config_flags_a_moved_images_root_by_name(tmp_path: Path):
    from tcip_mcp.tools.training_tools import preflight_config

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    moved = tmp_path / "moved_images"
    (root / "images" / DATES[0]).rename(moved)
    config = _preflight_config(root, out, DATES[0], images_dir=str(moved))

    result = preflight_config(config)

    assert any("images_root" in i for i in result["issues"])


def test_auto_train_val_manifest_binding_failure_raises_rather_than_degrading(tmp_path: Path):
    """A stem the data no longer admits (the label emptied since the split was drawn) must raise
    to the caller, never silently degrade to training on the manifest's held-out side."""
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    json_io.write_annotations(
        root / "annotations" / DATES[0] / "b.json", [], 64, 64, keep_empty=True,
    )
    data_cfg = _run_data_cfg(root, out, DATES[0])

    with pytest.raises(ValueError, match="not in this run's admitted"):
        auto_train_val("detection", data_cfg, None)


def test_auto_train_val_manifest_refuses_a_dataset_level_coco_misrouted_as_labels_dir(
        tmp_path: Path):
    """The manifest branch reaches the same dataset-level-COCO refusal the auto path does,
    raised ahead of admission and never folded into a binding failure."""
    import json

    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    data_cfg = _run_data_cfg(root, out, DATES[0])

    labels_dir = root / "annotations" / DATES[0]
    for f in labels_dir.glob("*.json"):
        f.unlink()
    (labels_dir / "dataset.json").write_text(json.dumps(
        {"images": [], "annotations": [], "categories": []}))

    with pytest.raises(ValueError, match="data.labels_dir="):
        auto_train_val("detection", data_cfg, None)


def test_auto_train_val_reads_the_label_format_once_per_run(tmp_path: Path, monkeypatch):
    """Neither the manifest branch nor the auto path reads ``data.labels_dir``'s format more than
    once: the caller reads it ahead of its own handler, and the admission build it feeds into
    never re-reads it."""
    import tcip_mcp.tools.training_tools as ttools
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    calls: list[None] = []
    real = ttools._checked_label_format

    def counting(task, data_cfg, src):
        calls.append(None)
        return real(task, data_cfg, src)

    monkeypatch.setattr(ttools, "_checked_label_format", counting)

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    manifest_cfg = _run_data_cfg(root, out, DATES[0])
    auto_train_val("detection", manifest_cfg, None)
    assert len(calls) == 1

    calls.clear()
    auto_cfg = {
        "images_dir": str(root / "images" / DATES[0]),
        "labels_dir": str(root / "annotations" / DATES[0]),
        "subject": SUBJECT, "attribute": None,
    }
    auto_train_val("detection", auto_cfg, None)
    assert len(calls) == 1


# -- persist_split_manifest / the worker's config patch --------------------------


def test_persist_split_manifest_carries_the_manifest_binding(tmp_path: Path):
    from tcip_mcp.experiments import create_experiment, read_split_manifest
    from tcip_mcp.pipelines.data.splits import bind_manifest_stems
    from tcip_mcp.pipelines.resolution import dataset_hash
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_split_manifest

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)
    data_cfg = _run_data_cfg(root, out, DATES[0])
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)

    create_experiment("exp_manifest_binding", {})
    persist_split_manifest("exp_manifest_binding", train_ds, val_ds, data_cfg)

    persisted = read_split_manifest("exp_manifest_binding")
    assert persisted["date"] == DATES[0]
    binding = persisted["manifest_binding"]
    assert binding["manifest_dir"] == str(out)
    assert binding["date"] == DATES[0]

    # An independent bind over this run's own admitted train/val stems, checked against the
    # persisted counts and hash rather than a tautology a two-sided binder would also pass.
    expected = bind_manifest_stems(
        manifest, DATES[0], SUBJECT, None, sorted(train_ds.stems) + sorted(val_ds.stems))
    assert binding["calibration_bound"] == expected.calibration_bound == len(expected.calibration)
    assert binding["calibration_unadmitted"] == 0
    assert binding["other_dates"] > 0  # the manifest's other date's members, across all three sides
    expected_hash = dataset_hash(
        data_cfg["labels_dir"], stems=sorted(expected.train + expected.val + expected.calibration))
    assert binding["labels_hash_now"] == expected_hash


def test_persist_split_manifest_carries_no_stale_binding_when_this_run_did_not_bind(
        tmp_path: Path):
    """A config inherited from an earlier bound launch (its ``data.split`` still carrying that
    run's ``manifest_binding``) relaunched with no ``manifest_dir`` draws its own split and
    records none of the earlier binding beside the drawn membership."""
    from tcip_mcp.experiments import create_experiment, read_split_manifest
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_split_manifest

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    data_cfg = {
        "images_dir": str(root / "images" / DATES[0]),
        "labels_dir": str(root / "annotations" / DATES[0]),
        "subject": SUBJECT, "attribute": None,
        "split": {
            "manifest_binding": {"manifest_dir": "stale", "date": DATES[0]},
            "resolved_group_key_map": {"stale": "p"},
            "resolved_seed": 7,
        },
    }

    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)

    assert val_ds is not None
    assert "manifest_binding" not in data_cfg["split"]
    assert "resolved_group_key_map" not in data_cfg["split"]
    assert "resolved_seed" not in data_cfg["split"]

    create_experiment("exp_relaunch_no_manifest", {})
    persist_split_manifest("exp_relaunch_no_manifest", train_ds, val_ds, data_cfg)

    persisted = read_split_manifest("exp_relaunch_no_manifest")
    assert "manifest_binding" not in persisted
    assert persisted["date"] == DATES[0]  # every run's split.json carries its own date, bound or not


def test_patch_experiment_config_split_merges_into_the_durable_record(tmp_path: Path):
    import tcip_store as ts
    from tcip_mcp.experiments import config_key, create_experiment
    from tcip_mcp.pipelines.training.subprocess_worker import _patch_experiment_config_split

    create_experiment("exp_split_patch", {"data": {"labels_dir": "orig"}})

    _patch_experiment_config_split("exp_split_patch", {"manifest_binding": {"date": DATES[0]}})

    cfg = ts.read(config_key("exp_split_patch"))
    assert cfg["data"]["labels_dir"] == "orig"
    assert cfg["data"]["split"]["manifest_binding"]["date"] == DATES[0]


def test_worker_leaves_a_relaunched_spatial_runs_stale_binding_out_of_the_durable_config(
        tmp_path: Path, monkeypatch):
    """A launch config inherited from an earlier bound run (its ``data.split`` still carrying
    that run's ``manifest_binding``) relaunched with no ``manifest_dir`` over a single-source,
    spatially splittable tree must not mirror any binding into the durable record: the real
    ``auto_train_val`` clears the stale block before it resolves its own spatial split, so the
    worker's patch-back gate never fires."""
    import tcip_store as ts
    from tcip_mcp.experiments import config_key, create_experiment
    from tcip_mcp.pipelines.training import subprocess_worker as worker
    from tcip_mcp.tools import training_tools as ttools

    images_dir, labels_dir = _single_source_mosaic_dataset(tmp_path / "ds")
    out = tmp_path / "run"
    out.mkdir()
    data = {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": SUBJECT,
        "auto_val": True, "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
    }
    create_experiment("exp_stale_spatial_relaunch",
                      {"model_source": {"builder": "x:y", "task": "detection"}, "data": data})
    launch_config = {
        "model_source": {"builder": "x:y", "task": "detection"},
        "data": {**data,
                 "split": {"manifest_binding": {"manifest_dir": "stale", "date": DATES[0]}}},
    }
    ts.replace(ttools.launch_config_key(out), launch_config)

    class StopAfterSplit(Exception):
        pass

    def stop(*args, **kwargs):
        raise StopAfterSplit

    monkeypatch.setattr(worker, "_resolve_run_id_map", stop)
    with pytest.raises(StopAfterSplit):
        worker.run("run1", "exp_stale_spatial_relaunch", str(out), "")

    durable_split = ts.read(config_key("exp_stale_spatial_relaunch"))["data"].get("split", {})
    assert "manifest_binding" not in durable_split
    assert "spatial_manifest" not in durable_split


# -- preflight_config's manifest-level checks --------------------------------------


def _preflight_config(root: Path, manifest_dir: Path, date: str, **overrides) -> dict:
    data_cfg = _run_data_cfg(root, manifest_dir, date)
    data_cfg.update(overrides)
    return {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": data_cfg, "training": {"batch_size": 2},
    }


def test_preflight_config_admits_a_bound_manifest_with_no_issues(tmp_path: Path):
    from tcip_mcp.tools.training_tools import preflight_config

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)

    result = preflight_config(_preflight_config(root, out, DATES[0]))

    manifest_issues = [i for i in result["issues"] if "manifest" in i]
    assert manifest_issues == []


def test_preflight_config_flags_a_manifest_conflict(tmp_path: Path):
    from tcip_mcp.tools.training_tools import preflight_config

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    config = _preflight_config(root, out, DATES[0])
    config["data"]["val_images_dir"] = str(root / "images" / DATES[1])

    result = preflight_config(config)

    assert any("val_images_dir" in i for i in result["issues"])


def test_preflight_config_flags_a_subject_mismatch(tmp_path: Path):
    from tcip_mcp.tools.training_tools import preflight_config

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    config = _preflight_config(root, out, DATES[0])
    config["data"]["subject"] = OTHER_SUBJECT

    result = preflight_config(config)

    assert any("subject" in i for i in result["issues"])


def test_preflight_config_accepts_an_empty_string_attribute_the_child_normalizes(tmp_path: Path):
    """An explicit empty-string attribute normalizes to ``None`` the same way the child's own
    ``_dataset_source_kwargs`` call normalizes it, so preflight does not flag a launch the child
    binds without complaint."""
    from tcip_mcp.tools.training_tools import preflight_config

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    config = _preflight_config(root, out, DATES[0], attribute="")

    result = preflight_config(config)

    manifest_issues = [i for i in result["issues"] if "manifest" in i]
    assert manifest_issues == []


def test_hpo_trial_snapshot_carries_the_manifest_binding(tmp_path: Path, monkeypatch):
    """A tuning trial binds to the manifest its base config names through the same
    ``auto_train_val`` branch a launched run uses, and the trial's persisted resolved config
    carries that binding, so a sweep's provenance names the partition each trial trained on."""
    import torch.utils.data as tud

    import tcip_store as ts
    from tcip_mcp.pipelines.data import samplers
    from tcip_mcp.pipelines.training import generic_trainer as gt
    from tcip_mcp.tools.training_tools import _run_hpo_trial, trial_config_key

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    base_config = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": _run_data_cfg(root, out, DATES[0]),
        "training": {"batch_size": 2},
    }

    def fake_train(run, train_loader, val_loader, task="detection",
                   epoch_callback=None, resume_from=""):
        run.best_metric = 1.0
        run.status = "completed"
        return run

    monkeypatch.setattr(gt, "train", fake_train)
    monkeypatch.setattr(samplers, "build_sampler", lambda *a, **k: None)
    monkeypatch.setattr(tud, "DataLoader", lambda *a, **k: object())
    trial_dir = tmp_path / "sweep" / "trial_0"
    _run_hpo_trial({"lr": 3e-4}, [].append, base_config, str(trial_dir))

    snapshot = ts.read(trial_config_key(trial_dir.parent, trial_dir.name))
    binding = snapshot["data"]["split"]["manifest_binding"]
    assert binding["manifest_dir"] == str(out)
    assert binding["subject"] == SUBJECT
    assert binding["date"] == DATES[0]


# -- evaluate_model's split_manifest_dir ------------------------------------------


def test_evaluate_model_under_the_manifest_scores_exactly_calibration_universe_from_manifests_side(
    tmp_path: Path, monkeypatch,
):
    """evaluate_model given split_manifest_dir scores the checkpoint over exactly the stems
    calibration_universe_from_manifest hands back for the labels directory's own date (the same
    universe the calibration door draws through), and records the stem count. The universe's own
    source side within the manifest is calibration_universe_from_manifest's concern, not
    duplicated here."""
    import tcip_mcp.pipelines.training.evaluation as evaluation
    from tcip_mcp.pipelines.data.splits import calibration_universe_from_manifest, label_image_stems
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)

    images_dir, labels_dir = root / "images" / DATES[0], root / "annotations" / DATES[0]
    present, _ = label_image_stems(labels_dir, images_dir)
    expected_universe, *_rest = calibration_universe_from_manifest(manifest, DATES[0], present)
    assert expected_universe

    run = create_run({"data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                               "subject": SUBJECT}}, str(tmp_path / "runs"))
    Path(run.output_dir).mkdir(parents=True, exist_ok=True)
    registered_checkpoint(Path(run.output_dir), project_root=tmp_path, filename="model_best.pt")

    captured: dict = {}

    def _fake(ckpt, loader, device, task, output_dir, **kw):
        captured["ds"] = loader.dataset
        captured["kw"] = kw
        return {"tiled": False, "eval_regime": "tile-level"}

    monkeypatch.setattr(evaluation, "run_test_evaluation", _fake)

    res = evaluate_model(run.run_id, str(images_dir), str(labels_dir), task="detection",
                         split_manifest_dir=str(out))

    assert "error" not in res, res
    assert sorted(captured["ds"].stems) == sorted(expected_universe)
    assert captured["kw"]["split_manifest_dir"] == str(out)
    assert captured["kw"]["evaluated_stem_count"] == len(expected_universe)


def test_evaluate_model_under_manifest_writes_and_reads_back_test_results(
    tmp_path: Path, monkeypatch,
):
    """The real run_test_evaluation path under a manifest, not a stub: writes test_results.json
    for real, and the record read back through evaluation_results_key carries split_manifest_dir
    and the loader's own evaluated_stem_count."""
    import tcip_store as ts

    import tcip_mcp.pipelines.model_build as model_build
    import tcip_mcp.pipelines.training.evaluation as evaluation
    from tcip_mcp.pipelines.data.splits import calibration_universe_from_manifest, label_image_stems
    from tcip_mcp.pipelines.training.evaluation import evaluation_results_key
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_mcp.tools.training_tools import evaluate_model

    class _DummyModel:
        def load_state_dict(self, state_dict):
            pass

        def to(self, device):
            pass

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)

    images_dir, labels_dir = root / "images" / DATES[0], root / "annotations" / DATES[0]
    present, _ = label_image_stems(labels_dir, images_dir)
    expected_universe, *_rest = calibration_universe_from_manifest(manifest, DATES[0], present)
    assert expected_universe

    run = create_run({"data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                               "subject": SUBJECT}}, str(tmp_path / "runs"))
    Path(run.output_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(run.output_dir) / "model_best.pt"
    torch.save({"model_source": {"builder": "x:y"}, "model_state_dict": {}}, str(ckpt_path))
    from tcip_mcp.tools.model_tools import register_model

    reg = register_model(name="split-binding-model", checkpoint_path=str(ckpt_path), config={})
    assert "error" not in reg, reg
    monkeypatch.setattr(model_build, "build_model", lambda ckpt: _DummyModel())
    monkeypatch.setattr(evaluation, "evaluate",
                        lambda *a, **k: {"loss": 0.1, "map50": 0.5, "precision": 0.4, "recall": 0.5})

    res = evaluate_model(run.run_id, str(images_dir), str(labels_dir), task="detection",
                         split_manifest_dir=str(out))

    assert "error" not in res, res
    persisted = ts.read(evaluation_results_key(run.output_dir))
    assert persisted["split_manifest_dir"] == str(out)
    assert persisted["evaluated_stem_count"] == len(expected_universe)


def test_evaluate_model_reads_confirmed_negatives_under_the_universes_own_date(
    tmp_path: Path, monkeypatch,
):
    """With date omitted, evaluate_model derives the calibration universe's own date rather than
    reading confirmed negatives under an undated bucket (which finds none), so a confirmed
    negative on the calibration side is admitted, not dropped as an unconfirmed empty label."""
    import tcip_store as ts

    import tcip_mcp.pipelines.training.evaluation as evaluation
    from tcip_mcp.dataset_layout import record_image_statuses, status_bucket
    from tcip_mcp.pipelines.data.splits import member_identity
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_mcp.tools.data_tools import split_manifest_key
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    root = tmp_path / "ds"
    write_registry(root / "classes.json", ClassRegistry(subjects=(Subject(name=SUBJECT),)))
    images_dir, labels_dir = root / "images" / DATES[0], root / "annotations" / DATES[0]
    for i in range(4):
        _write_stem(images_dir, labels_dir, f"p{i}",
                   [Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20))])
    _write_stem(images_dir, labels_dir, "n0", [])
    record_image_statuses(root, status_bucket(SUBJECT, DATES[0]), {"n0.jpg": "negative"},
                          recorded_by="user:tester")

    out = tmp_path / "m"
    result = make_splits(str(root), output_path=str(out), subject=SUBJECT, seed=1,
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in result, result
    manifest = ts.read(split_manifest_key(out))
    negative_identity = member_identity(DATES[0], "n0")
    for side in ("train", "val", "calibration"):
        manifest["splits"][side] = [i for i in manifest["splits"][side] if i != negative_identity]
    manifest["splits"]["calibration"].append(negative_identity)
    ts.replace(split_manifest_key(out), manifest)

    run = create_run({"data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                               "subject": SUBJECT}}, str(tmp_path / "runs"))
    Path(run.output_dir).mkdir(parents=True, exist_ok=True)
    registered_checkpoint(Path(run.output_dir), project_root=tmp_path, filename="model_best.pt")

    captured: dict = {}

    def _fake(ckpt, loader, device, task, output_dir, **kw):
        captured["ds"] = loader.dataset
        captured["kw"] = kw
        return {"tiled": False, "eval_regime": "tile-level"}

    monkeypatch.setattr(evaluation, "run_test_evaluation", _fake)

    res = evaluate_model(run.run_id, str(images_dir), str(labels_dir), task="detection",
                         split_manifest_dir=str(out))

    assert "error" not in res, res
    assert "n0" in captured["ds"].stems
    assert captured["kw"]["evaluated_stem_count"] == len(captured["ds"].stems)


def test_evaluate_model_manifest_refuses_a_disagreeing_date(tmp_path: Path, monkeypatch):
    """A real (if minimal) checkpoint and a stubbed run_test_evaluation, so the refusal's own
    assertion is what fails, not an unrelated checkpoint-loading crash reached by continuing past
    a refusal the tree under test does not raise."""
    import tcip_mcp.pipelines.training.evaluation as evaluation
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_mcp.tools.training_tools import evaluate_model

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)

    images_dir, labels_dir = root / "images" / DATES[0], root / "annotations" / DATES[0]
    run = create_run({"data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                               "subject": SUBJECT}}, str(tmp_path / "runs"))
    Path(run.output_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = str(Path(run.output_dir) / "model_best.pt")
    torch.save({"model_source": {"builder": "x:y"}, "model_state_dict": {}}, ckpt_path)
    from tcip_mcp.tools.model_tools import register_model

    reg = register_model(name="split-binding-date-model", checkpoint_path=ckpt_path, config={})
    assert "error" not in reg, reg
    monkeypatch.setattr(
        evaluation, "run_test_evaluation",
        lambda ckpt, loader, device, task, output_dir, **kw:
            {"tiled": False, "eval_regime": "tile-level"})

    result = evaluate_model(run.run_id, str(images_dir), str(labels_dir), task="detection",
                            split_manifest_dir=str(out), date=DATES[1])

    assert "error" in result and "disagrees" in result["error"]


def test_evaluate_model_scores_a_one_foreground_group_calibration_side_the_door_still_refuses(
    tmp_path: Path, monkeypatch,
):
    """evaluate_model asks the shared universe resolver for one foreground group (it halves
    nothing), so a calibration side reduced to exactly one foreground group still scores under
    it, while the same universe still refuses through the shared resolver's own two-group floor
    (the locked cal/holdout draw's halving)."""
    import tcip_store as ts

    import tcip_mcp.pipelines.training.evaluation as evaluation
    from tcip_mcp.pipelines.data.splits import (
        calibration_universe_from_manifest, count_label_lines, label_image_stems,
        member_identity_parts,
    )
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_mcp.tools.data_tools import split_manifest_key
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)

    images_dir, labels_dir = root / "images" / DATES[0], root / "annotations" / DATES[0]
    present, _ = label_image_stems(labels_dir, images_dir)
    cal_ids = [i for i in manifest["splits"]["calibration"]
              if member_identity_parts(i)[0] == DATES[0]]
    fg_cal_ids = [i for i in cal_ids if count_label_lines(
        labels_dir, member_identity_parts(i)[1], subject=SUBJECT) > 0]
    assert len(fg_cal_ids) >= 2
    # Move every foreground calibration member but one onto train for this date, leaving the
    # calibration side with exactly one foreground group.
    keep, move = fg_cal_ids[0], fg_cal_ids[1:]
    manifest["splits"]["calibration"] = [
        i for i in manifest["splits"]["calibration"] if i not in move]
    manifest["splits"]["train"] = sorted(manifest["splits"]["train"] + move)
    ts.replace(split_manifest_key(out), manifest)

    run = create_run({"data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                               "subject": SUBJECT}}, str(tmp_path / "runs"))
    Path(run.output_dir).mkdir(parents=True, exist_ok=True)
    registered_checkpoint(Path(run.output_dir), project_root=tmp_path, filename="model_best.pt")

    monkeypatch.setattr(
        evaluation, "run_test_evaluation",
        lambda ckpt, loader, device, task, output_dir, **kw:
            {"tiled": False, "eval_regime": "tile-level"})

    res = evaluate_model(run.run_id, str(images_dir), str(labels_dir), task="detection",
                         split_manifest_dir=str(out))
    assert "error" not in res, res

    manifest_reread = ts.read(split_manifest_key(out))
    with pytest.raises(ValueError, match="foreground group"):
        calibration_universe_from_manifest(
            manifest_reread, DATES[0], present,
            foreground_stems={member_identity_parts(keep)[1]})


def test_evaluate_model_manifest_refuses_a_subject_mismatch(tmp_path: Path):
    from tcip_mcp.tools.training_tools import evaluate_model
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)

    images_dir, labels_dir = root / "images" / DATES[0], root / "annotations" / DATES[0]
    run = create_run({"data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                               "subject": OTHER_SUBJECT}}, str(tmp_path / "runs"))
    Path(run.output_dir).mkdir(parents=True, exist_ok=True)
    registered_checkpoint(Path(run.output_dir), project_root=tmp_path, filename="model_best.pt")

    result = evaluate_model(run.run_id, str(images_dir), str(labels_dir), task="detection",
                            split_manifest_dir=str(out))

    assert "error" in result and "subject" in result["error"]
