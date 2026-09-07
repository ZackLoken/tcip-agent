"""``tcip drop-spatial-split-seed``: the conform for the seed key a spatial-strip split's
persisted record leaves, dropped because it never governed the strip layout.

Every fixture is a real ``experiment_split`` record: a real spatial-strip split, persisted by
the platform's own ``persist_split_manifest`` over a one-source tiled mosaic. Where a test needs
a record shaped like one the old writer produced (carrying ``spatial.seed``), the key is put back
through a direct store write standing in for that old writer, since a live producer can no longer
write one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402

import tcip_store as ts  # noqa: E402
from tcip_store.binding import bind_default  # noqa: E402

from tcip_mcp.experiments import create_experiment, split_key  # noqa: E402
from tests._record_damage_fixtures import damage_record  # noqa: E402

MOSAIC_W, MOSAIC_H = 4000, 3000


def _load_module():
    from tcip_mcp.cli import drop_spatial_split_seed

    return drop_spatial_split_seed


def _one_source_dataset(root: Path) -> tuple[Path, Path, str]:
    """One large single-source mosaic with GT spread across its whole extent, enough for the
    real spatial-strip split to derive a train/val layout over it."""
    from PIL import Image

    images_dir, labels_dir = root / "images", root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    stem = "mosaic"
    Image.new("RGB", (MOSAIC_W, MOSAIC_H), color=(90, 90, 90)).save(images_dir / f"{stem}.png")
    boxes = [Annotation(subject="bud", geometry=BBox(x, y, x + 20, y + 20))
             for x in range(20, MOSAIC_W - 20, 200) for y in range(20, MOSAIC_H - 20, 200)]
    json_io.write_annotations(str(labels_dir / f"{stem}.json"), boxes, MOSAIC_W, MOSAIC_H,
                              keep_empty=True)
    return images_dir, labels_dir, stem


def _persist_spatial_record(project: Path, experiment_id: str, *, config_seed: int = 7) -> None:
    """Persist a real spatial-strip split's ``experiment_split`` record for ``experiment_id``
    under ``project``, the pinned platform root: the shape ``persist_split_manifest`` writes
    going forward, its ``spatial`` block already carrying no ``seed``."""
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_split_manifest

    images_dir, labels_dir, stem = _one_source_dataset(project / "ds")
    data_cfg = {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
        "auto_val": True, "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
        "split": {"val_ratio": 0.2, "test_ratio": 0.1, "seed": config_seed},
    }
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    assert val_ds is not None
    create_experiment(experiment_id, {})
    persist_split_manifest(experiment_id, train_ds, val_ds, data_cfg)


def _inject_seed(experiment_id: str, *, root: Path, seed: int = 42) -> None:
    """Stand in for a record the old writer wrote: put ``seed`` back into the persisted
    record's ``spatial`` block through a direct store write, since a live producer can no
    longer write one."""
    key = split_key(experiment_id, root=root)
    manifest = ts.read(key)
    manifest = {**manifest, "spatial": {**manifest["spatial"], "seed": seed}}
    ts.replace(key, manifest)


def test_a_record_carrying_seed_is_rewritten_and_one_audit_line_lands(tmp_path):
    """The conform's core behaviour: a record shaped like the old writer's own (``spatial.seed``
    present) is rewritten without that key, the top-level ``seed`` (a different fact) is
    untouched, and one audit line lands naming the experiment and the key removed. The module
    is absent at the baseline, so this is new-behaviour coverage, not a regression guard."""
    bind_default()
    module = _load_module()
    project = tmp_path
    _persist_spatial_record(project, "exp-with-seed", config_seed=7)
    _inject_seed("exp-with-seed", root=project, seed=42)

    outcomes, refused = module.process_project_root(project, plan=False)

    assert refused is False
    assert any("dropped spatial.seed" in o for o in outcomes), outcomes
    record = ts.read(split_key("exp-with-seed", root=project))
    assert "seed" not in record["spatial"]
    assert record["seed"] == 7

    from tcip_mcp.tools.meta_tools import read_audit_log

    entries = read_audit_log(scope=str(project), tool=module.TOOL_NAME)["entries"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["arguments"]["experiment_id"] == "exp-with-seed"
    assert entry["key_removed"] == "spatial.seed"


def test_a_record_carrying_no_seed_is_left_untouched_and_the_command_exits_0(tmp_path):
    """A record already in the conformed shape (the ordinary case going forward, since
    ``persist_split_manifest`` no longer writes ``spatial.seed`` at all) is left byte-identical
    and the command exits 0."""
    bind_default()
    module = _load_module()
    project = tmp_path
    _persist_spatial_record(project, "exp-no-seed")
    before = ts.read(split_key("exp-no-seed", root=project))

    exit_code = module.main([str(project)])

    assert exit_code == 0
    after = ts.read(split_key("exp-no-seed", root=project))
    assert after == before


def test_plan_mode_rewrites_nothing_and_exits_2(tmp_path):
    """``--plan`` previews the rewrite a record carrying ``spatial.seed`` would get, writes
    nothing, and counts it toward the exit code the same way a refusal does."""
    bind_default()
    module = _load_module()
    project = tmp_path
    _persist_spatial_record(project, "exp-plan")
    _inject_seed("exp-plan", root=project, seed=3)
    before = ts.read(split_key("exp-plan", root=project))

    exit_code = module.main(["--plan", str(project)])

    assert exit_code == 2
    after = ts.read(split_key("exp-plan", root=project))
    assert after == before
    assert "seed" in after["spatial"]


def test_an_undecodable_record_is_reported_and_left(tmp_path):
    """A record the seam will not decode is reported by name and never rewritten: the walk
    continues past it rather than raising, and it counts toward the exit code."""
    bind_default()
    module = _load_module()
    project = tmp_path
    _persist_spatial_record(project, "exp-undecodable")
    damage_record(split_key("exp-undecodable", root=project), b"{not json")

    outcomes, refused = module.process_project_root(project, plan=False)

    assert refused is True
    assert any("will not decode" in o for o in outcomes), outcomes


def test_a_root_with_no_tcip_directory_is_refused_by_name(tmp_path):
    """A root that never held any experiment refuses by name, the way the sibling repair
    command refuses one, rather than reporting zero records and success."""
    bind_default()
    module = _load_module()
    stray = tmp_path / "not_a_project"
    stray.mkdir()

    outcomes, refused = module.process_project_root(stray, plan=False)

    assert refused is True
    assert any("no .tcip directory found" in o for o in outcomes), outcomes
