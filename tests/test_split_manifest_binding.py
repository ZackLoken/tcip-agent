"""Training and tuning binding to a named split manifest (``data.split.manifest_dir``).

The manifest is drawn by ``make_splits`` (see ``test_data_tools.py``); this file covers the
consumer side: ``bind_manifest_stems``, ``read_split_manifest_dir``, and ``_auto_train_val``'s own
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
    """Two capture dates, three stems each: every stem carries ``leaf``, and two of the three
    also carry the unrelated ``bud``, so a manifest drawn for either subject binds to a real,
    differently-sized draw over the identical tree, with at least two members per date on both
    sides."""
    write_registry(root / "classes.json", ClassRegistry(subjects=(
        Subject(name=SUBJECT), Subject(name=OTHER_SUBJECT),
    )))
    for date in DATES:
        images_dir, labels_dir = root / "images" / date, root / "annotations" / date
        _write_stem(images_dir, labels_dir, "a",
                   [Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20))])
        for stem in ("b", "c"):
            _write_stem(images_dir, labels_dir, stem, [
                Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20)),
                Annotation(subject=OTHER_SUBJECT, geometry=BBox(30, 30, 44, 44)),
            ])
    return root


def _attribute_scoped_dataset(root: Path) -> Path:
    """One date, three stems: two have their instance assessed for ``condition``, the third
    carries an instance never assessed for it."""
    write_registry(root / "classes.json", ClassRegistry(subjects=(
        Subject(name=SUBJECT, attributes=(
            Attribute(name="condition", type="categorical", values=("healthy", "damaged")),
        )),
    )))
    images_dir, labels_dir = root / "images" / DATES[0], root / "annotations" / DATES[0]
    for stem, condition in (("assessed_a", "healthy"), ("assessed_b", "damaged")):
        _write_stem(images_dir, labels_dir, stem, [
            Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20),
                      attributes={"condition": condition})])
    _write_stem(images_dir, labels_dir, "unassessed",
               [Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20))])
    return root


def _draw(root: Path, out: Path, *, subject: str = SUBJECT, attribute: str | None = None,
         seed: int = 1) -> dict:
    import tcip_store as ts
    from tcip_mcp.tools.data_tools import split_manifest_key

    result = make_splits(str(root), output_path=str(out), subject=subject, attribute=attribute,
                         seed=seed, train_ratio=0.5, val_ratio=0.5, test_ratio=0.0)
    assert "error" not in result, result
    return ts.read(split_manifest_key(out))


# -- bind_manifest_stems -------------------------------------------------------


def test_bind_manifest_stems_binds_the_manifests_own_partition_for_one_date(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import bind_manifest_stems

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    manifest = _draw(root, tmp_path / "m")

    binding = bind_manifest_stems(manifest, DATES[0], SUBJECT, None, ["a", "b", "c"])

    assert sorted(binding.train + binding.val) == ["a", "b", "c"]
    assert binding.train and binding.val
    assert binding.assigned == 3


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

    with pytest.raises(ValueError, match="neither side"):
        bind_manifest_stems(manifest, DATES[0], SUBJECT, None, ["a", "b", "c", "d"])


def test_bind_manifest_stems_refuses_a_manifest_member_the_run_does_not_admit(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import bind_manifest_stems

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    manifest = _draw(root, tmp_path / "m")

    with pytest.raises(ValueError, match="not in this run's admitted"):
        bind_manifest_stems(manifest, DATES[0], SUBJECT, None, ["a"])


def test_bind_manifest_stems_refuses_an_empty_side_after_binding(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import bind_manifest_stems
    from tcip_mcp.tools.data_tools import read_split_manifest_dir

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = make_splits(str(root), output_path=str(out), subject=SUBJECT, seed=1,
                         train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)
    assert "error" not in result, result
    manifest = read_split_manifest_dir(out)

    with pytest.raises(ValueError, match="empty side"):
        bind_manifest_stems(manifest, DATES[0], SUBJECT, None, ["a", "b", "c"])


# -- read_split_manifest_dir ----------------------------------------------------


def test_read_split_manifest_dir_returns_the_written_manifest(tmp_path: Path):
    from tcip_mcp.tools.data_tools import read_split_manifest_dir

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = make_splits(str(root), output_path=str(out), subject=SUBJECT, seed=1,
                         train_ratio=0.5, val_ratio=0.5, test_ratio=0.0)
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


# -- _auto_train_val's manifest branch -------------------------------------------


def _run_data_cfg(root: Path, manifest_dir: Path, date: str, *, subject: str = SUBJECT,
                  attribute: str | None = None, images_dir: Path | None = None) -> dict:
    return {
        "images_dir": str(images_dir or (root / "images" / date)),
        "labels_dir": str(root / "annotations" / date),
        "subject": subject, "attribute": attribute, "date": date,
        "split": {"manifest_dir": str(manifest_dir)},
    }


def _dataset_with_a_confirmed_negative(root: Path) -> Path:
    """One date, three annotated stems plus a fourth confirmed negative for ``SUBJECT``."""
    write_registry(root / "classes.json", ClassRegistry(subjects=(Subject(name=SUBJECT),)))
    images_dir, labels_dir = root / "images" / DATES[0], root / "annotations" / DATES[0]
    for stem in ("a", "b", "c"):
        _write_stem(images_dir, labels_dir, stem,
                   [Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20))])
    _write_stem(images_dir, labels_dir, "n", [])
    return root


def test_auto_train_val_admits_a_confirmed_negative_with_data_date_unset(tmp_path: Path):
    """A run whose ``data.date`` is left unset still reads confirmed negatives under the tree's
    own date, the date the split manifest was drawn under, so a manifest member confirmed
    negative under that date still admits."""
    from tcip_mcp.dataset_layout import record_image_statuses, status_bucket
    from tcip_mcp.tools.training_tools import _auto_train_val

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

    train_ds, val_ds = _auto_train_val("detection", data_cfg, None)

    assert "n" in train_ds.stems + val_ds.stems


def test_auto_train_val_binds_to_the_manifests_own_partition_for_its_date(tmp_path: Path):
    from tcip_mcp.tools.training_tools import _auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)
    data_cfg = _run_data_cfg(root, out, DATES[0])

    train_ds, val_ds = _auto_train_val("detection", data_cfg, None)

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
    from tcip_mcp.tools.training_tools import _auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    data_cfg = _run_data_cfg(root, out, DATES[0])

    _auto_train_val("detection", data_cfg, None)
    train_ds, val_ds = _auto_train_val("detection", data_cfg, None)

    assert train_ds.stems and val_ds.stems


def test_preflight_config_on_a_bound_config_admits_it_again(tmp_path: Path):
    """Preflighting a config already bound once (the same conflict-key rail this file's other
    preflight tests exercise) must not itself read as a conflict."""
    from tcip_mcp.tools.training_tools import _auto_train_val, preflight_config

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    data_cfg = _run_data_cfg(root, out, DATES[0])
    _auto_train_val("detection", data_cfg, None)

    result = preflight_config(_preflight_config(root, out, DATES[0], split=data_cfg["split"]))

    manifest_issues = [i for i in result["issues"] if "manifest" in i]
    assert manifest_issues == []


def test_relaunch_from_the_durable_record_binds_again(tmp_path: Path):
    """A run relaunched from the durable ``config.json`` record a bound run's own worker patched
    binds again, rather than refusing against the very block that records it bound once."""
    import tcip_store as ts
    from tcip_mcp.experiments import config_key, create_experiment
    from tcip_mcp.pipelines.training.subprocess_worker import _patch_experiment_config_split
    from tcip_mcp.tools.training_tools import _auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    data_cfg = _run_data_cfg(root, out, DATES[0])
    _auto_train_val("detection", data_cfg, None)

    create_experiment("exp_relaunch_split_bind", {"data": dict(data_cfg)})
    _patch_experiment_config_split("exp_relaunch_split_bind", data_cfg["split"])
    durable_data_cfg = ts.read(config_key("exp_relaunch_split_bind"))["data"]

    train_ds, val_ds = _auto_train_val("detection", durable_data_cfg, None)

    assert train_ds.stems and val_ds.stems


def test_auto_train_val_binds_the_same_tree_for_the_other_subject(tmp_path: Path):
    from tcip_mcp.tools.training_tools import _auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out, subject=OTHER_SUBJECT, seed=0)
    data_cfg = _run_data_cfg(root, out, DATES[0], subject=OTHER_SUBJECT)

    train_ds, val_ds = _auto_train_val("detection", data_cfg, None)

    # Only stems "b" and "c" carry the other subject on this date.
    assert sorted(train_ds.stems + val_ds.stems) == ["b", "c"]


def test_auto_train_val_binds_an_attribute_scoped_tree(tmp_path: Path):
    from tcip_mcp.tools.training_tools import _auto_train_val

    root = _attribute_scoped_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = make_splits(str(root), output_path=str(out), subject=SUBJECT, attribute="condition",
                         seed=1, train_ratio=0.5, val_ratio=0.5, test_ratio=0.0)
    assert "error" not in result, result
    data_cfg = _run_data_cfg(root, out, DATES[0], attribute="condition")

    train_ds, val_ds = _auto_train_val("detection", data_cfg, None)

    assert sorted(train_ds.stems + val_ds.stems) == ["assessed_a", "assessed_b"]
    assert data_cfg["split"]["manifest_binding"]["attribute"] == "condition"


def test_auto_train_val_manifest_conflicts_with_val_images_dir(tmp_path: Path):
    from tcip_mcp.tools.training_tools import _auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    data_cfg = _run_data_cfg(root, out, DATES[0])
    data_cfg["val_images_dir"] = str(root / "images" / DATES[1])

    with pytest.raises(ValueError, match="val_images_dir"):
        _auto_train_val("detection", data_cfg, None)


def test_auto_train_val_manifest_conflicts_with_a_drawn_splits_own_parameters(tmp_path: Path):
    from tcip_mcp.tools.training_tools import _auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    data_cfg = _run_data_cfg(root, out, DATES[0])
    data_cfg["split"]["val_ratio"] = 0.3

    with pytest.raises(ValueError, match="val_ratio"):
        _auto_train_val("detection", data_cfg, None)


def test_auto_train_val_manifest_refuses_a_task_it_does_not_admit(tmp_path: Path):
    from tcip_mcp.tools.training_tools import _auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    data_cfg = _run_data_cfg(root, out, DATES[0])

    with pytest.raises(ValueError, match="semantic_seg"):
        _auto_train_val("semantic_seg", data_cfg, None)


def test_auto_train_val_manifest_refuses_a_disagreeing_date(tmp_path: Path):
    from tcip_mcp.tools.training_tools import _auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    data_cfg = _run_data_cfg(root, out, DATES[0])
    data_cfg["date"] = DATES[1]

    with pytest.raises(ValueError, match="data.date"):
        _auto_train_val("detection", data_cfg, None)


def test_auto_train_val_manifest_refuses_an_images_root_mismatch(tmp_path: Path):
    from tcip_mcp.tools.training_tools import _auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    other_images = tmp_path / "elsewhere"
    other_images.mkdir()
    data_cfg = _run_data_cfg(root, out, DATES[0], images_dir=other_images)

    with pytest.raises(ValueError, match="images_root"):
        _auto_train_val("detection", data_cfg, None)


def test_auto_train_val_manifest_refuses_a_moved_images_root_by_name(tmp_path: Path):
    """A manifest's recorded images_root that no longer exists on disk (a moved or renamed
    dataset) answers the named refusal, never a bare crash from comparing against a gone path."""
    from tcip_mcp.tools.training_tools import _auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    moved = tmp_path / "moved_images"
    (root / "images" / DATES[0]).rename(moved)
    data_cfg = _run_data_cfg(root, out, DATES[0], images_dir=moved)

    with pytest.raises(ValueError, match="images_root"):
        _auto_train_val("detection", data_cfg, None)


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
    from tcip_mcp.tools.training_tools import _auto_train_val

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    json_io.write_annotations(
        root / "annotations" / DATES[0] / "b.json", [], 64, 64, keep_empty=True,
    )
    data_cfg = _run_data_cfg(root, out, DATES[0])

    with pytest.raises(ValueError, match="not in this run's admitted"):
        _auto_train_val("detection", data_cfg, None)


# -- _persist_split_manifest / the worker's config patch --------------------------


def test_persist_split_manifest_carries_the_manifest_binding(tmp_path: Path):
    from tcip_mcp.experiments import create_experiment, read_split_manifest
    from tcip_mcp.tools.training_tools import _auto_train_val, _persist_split_manifest

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    data_cfg = _run_data_cfg(root, out, DATES[0])
    train_ds, val_ds = _auto_train_val("detection", data_cfg, None)

    create_experiment("exp_manifest_binding", {})
    _persist_split_manifest("exp_manifest_binding", train_ds, val_ds, data_cfg)

    persisted = read_split_manifest("exp_manifest_binding")
    assert persisted["manifest_binding"]["manifest_dir"] == str(out)
    assert persisted["manifest_binding"]["date"] == DATES[0]


def test_patch_experiment_config_split_merges_into_the_durable_record(tmp_path: Path):
    import tcip_store as ts
    from tcip_mcp.experiments import config_key, create_experiment
    from tcip_mcp.pipelines.training.subprocess_worker import _patch_experiment_config_split

    create_experiment("exp_split_patch", {"data": {"labels_dir": "orig"}})

    _patch_experiment_config_split("exp_split_patch", {"manifest_binding": {"date": DATES[0]}})

    cfg = ts.read(config_key("exp_split_patch"))
    assert cfg["data"]["labels_dir"] == "orig"
    assert cfg["data"]["split"]["manifest_binding"]["date"] == DATES[0]


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
