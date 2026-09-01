"""The data picker: choosing a recorded partition (or "As recorded") for a relaunch.

Covers the pieces the design record's second family adds: the shared
``manifest_compatibility`` check, the extracted ``narrow_manifest_to_date``, the checked manifest
reader, ``list_split_choices`` and its route, and the launch route's ``split_manifest_dir`` field.
The manifest-binding mechanics themselves (``bind_manifest_stems``, ``auto_train_val``'s manifest
branch, ``read_split_manifest_dir``) are ``test_split_manifest_binding.py``'s; this file reuses
its dataset fixture rather than restating it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import tcip_store as ts
from tcip_web.app import app

from tests.test_split_manifest_binding import DATES, OTHER_SUBJECT, SUBJECT, _draw, \
    _two_subject_two_date_dataset


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _bespoke_config(images_dir: Path, labels_dir: Path, *, subject: str = SUBJECT) -> dict:
    return {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1, "min_size": 64, "max_size": 64},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                 "subject": subject},
        "training": {"batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
                     "mixed_precision": False, "device": "cpu",
                     "checkpoint_every_n_epochs": 0, "early_stopping": {"enabled": False}},
    }


# -- narrow_manifest_to_date ---------------------------------------------------


def test_narrow_manifest_to_date_matches_bind_manifest_stems_own_arithmetic(tmp_path: Path):
    """The extracted narrowing and the binding it now feeds must agree: this is the same
    arithmetic the picker shows before a launch and the launch itself applies after one."""
    from tcip_mcp.pipelines.data.splits import bind_manifest_stems, narrow_manifest_to_date

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    manifest = _draw(root, tmp_path / "m")

    narrowing = narrow_manifest_to_date(manifest, DATES[0])
    binding = bind_manifest_stems(
        manifest, DATES[0], SUBJECT, None, ["a", "b", "c", "d", "e", "f"])

    assert len(narrowing.train_ids) == binding.train_bound
    assert len(narrowing.val_ids) == binding.val_bound
    assert len(narrowing.calibration_ids) == binding.calibration_bound
    assert narrowing.other_dates == binding.other_dates


# -- manifest_compatibility and the new absent-images-root refusal --------------


def _manifest_missing_images_root(root: Path, out: Path, date: str) -> dict:
    """A real ``make_splits`` draw, hand-mutated to drop one date's ``images_root``: the shape a
    hand-edited or pre-images_root manifest would read as. Every other required field stays
    exactly what ``make_splits`` wrote, so this exercises only the one field under test."""
    from tcip_mcp.pipelines.data.splits import manifest_date_key
    from tcip_mcp.tools.data_tools import split_manifest_key

    manifest = _draw(root, out)
    date_key = manifest_date_key(date)
    manifest["members"][date_key] = {
        k: v for k, v in manifest["members"][date_key].items() if k != "images_root"
    }
    ts.replace(split_manifest_key(out), manifest)
    return manifest


def test_manifest_compatibility_flags_a_members_block_with_no_images_root(tmp_path: Path):
    from tcip_mcp.tools.training_tools import manifest_compatibility

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _manifest_missing_images_root(root, out, DATES[0])
    config = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    config["data"]["split"] = {"manifest_dir": str(out)}

    issues = manifest_compatibility(config, manifest)

    assert any("images root" in i for i in issues)


def test_preflight_config_flags_a_manifest_with_no_images_root_recorded(tmp_path: Path):
    """The same refusal, reached through preflight_config: a stated root must be positively
    carried, and make_splits always writes one, so only a hand-edited manifest ever trips it."""
    from tcip_mcp.tools.training_tools import preflight_config

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _manifest_missing_images_root(root, out, DATES[0])
    config = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    config["data"]["split"] = {"manifest_dir": str(out)}

    result = preflight_config(config)

    assert any("images root" in i for i in result["issues"])


# -- read_split_manifest_dir_checked ---------------------------------------------


def test_read_split_manifest_dir_checked_tells_absence_apart_from_a_broken_record(tmp_path: Path):
    from tcip_mcp.tools.data_tools import read_split_manifest_dir_checked, split_manifest_key

    absent_manifest, absent_error = read_split_manifest_dir_checked(tmp_path / "nope")
    assert (absent_manifest, absent_error) == (None, None)

    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    ts.replace(split_manifest_key(broken_dir), {"seed": 1, "splits": {"train": [], "val": []}})
    broken_manifest, broken_error = read_split_manifest_dir_checked(broken_dir)
    assert broken_manifest is None
    assert broken_error is not None and "subject" in broken_error


# -- list_split_choices ------------------------------------------------------------


def test_list_split_choices_answers_only_as_recorded_without_images_or_labels_dir(
    tmp_path: Path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.training_tools import list_split_choices

    create_experiment("exp-no-data", {
        "model_source": {"builder": "m:f", "task": "detection"}, "data": {},
    })

    result = list_split_choices("exp-no-data")

    assert result["manifests"] == []
    assert result["as_recorded"]["case"] == "drawn"


def test_list_split_choices_route_404s_for_an_unknown_experiment(
    tmp_path: Path, monkeypatch, client: TestClient,
) -> None:
    """The route itself exists and answers a known id (proving the 404 below discriminates the
    one unknown id, never a route that failed to mount at all)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment

    create_experiment("exp-known", {
        "model_source": {"builder": "m:f", "task": "detection"}, "data": {},
    })

    known = client.get("/api/training/configs/exp-known/splits")
    assert known.status_code == 200
    assert known.json()["as_recorded"]["case"] == "drawn"

    resp = client.get("/api/training/configs/nope/splits")
    assert resp.status_code == 404


def test_list_split_choices_offers_every_recorded_partition_with_the_bindings_own_counts(
    tmp_path: Path, monkeypatch,
):
    """Through the platform's own producers: a manifest ``make_splits`` writes under the
    dataset's own ``splits`` directory, a second one elsewhere that a run bound to through
    ``launch_training``, a third for another subject reached through a bound run (listed
    disabled with the compatibility text), and a directory whose record will not decode (listed
    disabled, never absent)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})
    import subprocess

    class _StubChild:
        def __init__(self, *a, **k) -> None:
            self.pid = 4242

        def __class_getitem__(cls, item):
            return cls

    monkeypatch.setattr(subprocess, "Popen", _StubChild)

    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.pipelines.data.splits import bind_manifest_stems
    from tcip_mcp.tools.data_tools import split_manifest_key
    from tcip_mcp.tools.training_tools import launch_training, list_split_choices

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    dataset_default = root / "splits"
    default_manifest = _draw(root, dataset_default)

    elsewhere = tmp_path / "elsewhere"
    _draw(root, elsewhere, seed=2)
    bound_cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    bound_cfg["data"]["split"] = {"manifest_dir": str(elsewhere)}
    launched = launch_training(bound_cfg, str(tmp_path / "out_bound"))
    assert "error" not in launched, launched

    other_subject_dir = tmp_path / "other_subject"
    _draw(root, other_subject_dir, subject=OTHER_SUBJECT, seed=3)
    create_experiment("exp-other-subject", {
        "data": {"split": {"manifest_dir": str(other_subject_dir)}},
    })

    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    ts.replace(split_manifest_key(broken_dir), {"seed": 1, "splits": {"train": [], "val": []}})
    create_experiment("exp-broken-manifest", {"data": {"split": {"manifest_dir": str(broken_dir)}}})

    picked_cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    create_experiment("exp-picked", picked_cfg)

    result = list_split_choices("exp-picked")
    by_dir = {m["manifest_dir"]: m for m in result["manifests"]}

    expected = bind_manifest_stems(
        default_manifest, DATES[0], SUBJECT, None, ["a", "b", "c", "d", "e", "f"])
    default_entry = by_dir[str(dataset_default)]
    assert default_entry["enabled"] is True
    assert default_entry["train"] == expected.train_bound
    assert default_entry["val"] == expected.val_bound
    assert default_entry["calibration"] == expected.calibration_bound
    assert default_entry["other_dates"] == expected.other_dates

    elsewhere_entry = by_dir[str(elsewhere)]
    assert elsewhere_entry["enabled"] is True
    assert elsewhere_entry["train"] > 0 and elsewhere_entry["val"] > 0

    other_subject_entry = by_dir[str(other_subject_dir)]
    assert other_subject_entry["enabled"] is False
    assert "subject" in other_subject_entry["reason"]

    broken_entry = by_dir[str(broken_dir)]
    assert broken_entry["enabled"] is False
    assert broken_entry["reason"] is not None


# -- POST /api/training/runs's split_manifest_dir --------------------------------


def test_relaunch_route_409s_for_a_split_manifest_dir_outside_the_enabled_set(
    tmp_path: Path, monkeypatch, client: TestClient,
) -> None:
    """The browser's string is verified against the platform's own listing: a string this
    listing never offered, and one it offered but disabled, both refuse the same way."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.data_tools import split_manifest_key

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    create_experiment("exp-picker-guard", cfg)

    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    ts.replace(split_manifest_key(broken_dir), {"seed": 1, "splits": {"train": [], "val": []}})
    create_experiment("exp-broken-source", {"data": {"split": {"manifest_dir": str(broken_dir)}}})

    resp_unknown = client.post("/api/training/runs", json={
        "experiment_id": "exp-picker-guard", "split_manifest_dir": str(tmp_path / "never-listed"),
    })
    assert resp_unknown.status_code == 409

    resp_disabled = client.post("/api/training/runs", json={
        "experiment_id": "exp-picker-guard", "split_manifest_dir": str(broken_dir),
    })
    assert resp_disabled.status_code == 409


def test_relaunch_route_leaves_the_snapshots_data_unchanged_when_no_partition_is_chosen(
    tmp_path: Path, monkeypatch, client: TestClient,
) -> None:
    """"As recorded" launches the stored config's own data section, whether it was bound or
    drawn, byte for byte: the route never rewrites data.split unless a partition is chosen."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    import tcip_mcp.tools.training_tools as training_tools_module
    from tcip_mcp.experiments import create_experiment

    captured: dict = {}

    def fake_launch_training(config, *a, **k):
        captured["data"] = config.get("data")
        return {"run_id": "r1", "experiment_id": config.get("experiment_id"), "status": "launched"}

    monkeypatch.setattr(training_tools_module, "launch_training", fake_launch_training)

    drawn_data = {"images_dir": "/data/images", "labels_dir": "/data/labels", "subject": SUBJECT}
    create_experiment("exp-drawn", {
        "model_source": {"builder": "m:f", "task": "detection"}, "data": dict(drawn_data),
    })
    resp = client.post("/api/training/runs", json={"experiment_id": "exp-drawn"})
    assert resp.status_code == 200, resp.json()
    assert captured["data"] == drawn_data

    bound_data = {"images_dir": "/data/images", "labels_dir": "/data/labels", "subject": SUBJECT,
                 "split": {"manifest_dir": "/some/manifest"}}
    create_experiment("exp-bound", {
        "model_source": {"builder": "m:f", "task": "detection"}, "data": dict(bound_data),
    })
    resp = client.post("/api/training/runs", json={"experiment_id": "exp-bound"})
    assert resp.status_code == 200, resp.json()
    assert captured["data"] == bound_data


def test_relaunch_route_launches_with_a_chosen_manifest_and_refreshes_the_pristine_config(
    tmp_path: Path, monkeypatch, client: TestClient,
) -> None:
    """A launch with a chosen partition submits a string the server itself listed; on the
    pristine branch the existing first-run refresh stores that candidate, data.split replaced
    wholesale and val_images_dir removed, as the experiment's own config."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})
    import subprocess

    class _StubChild:
        def __init__(self, *a, **k) -> None:
            self.pid = 4242

        def __class_getitem__(cls, item):
            return cls

    monkeypatch.setattr(subprocess, "Popen", _StubChild)

    from tcip_mcp.experiments import config_key, create_experiment, read_member

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    chosen = root / "splits"
    _draw(root, chosen)

    cfg = _bespoke_config(root / "images" / DATES[0], root / "annotations" / DATES[0])
    cfg["data"]["val_images_dir"] = str(root / "images" / DATES[1])
    create_experiment("exp-choose-partition", cfg)

    resp = client.post("/api/training/runs", json={
        "experiment_id": "exp-choose-partition", "split_manifest_dir": str(chosen),
    })
    assert resp.status_code == 200, resp.json()

    snapshot = read_member(config_key("exp-choose-partition"))
    assert snapshot["data"]["split"] == {"manifest_dir": str(chosen)}
    assert "val_images_dir" not in snapshot["data"]


def test_a_chosen_manifest_binds_and_the_runs_own_split_record_names_it(tmp_path: Path) -> None:
    """The candidate config the route builds for a chosen partition, bound the exact way the
    child binds it (auto_train_val then persist_split_manifest, the sequence
    subprocess_worker.run follows, called directly so no training subprocess is needed): the
    run's own split.json names the chosen manifest as what it bound to."""
    from tcip_mcp.experiments import create_experiment, read_split_manifest
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_split_manifest

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    chosen = tmp_path / "chosen"
    _draw(root, chosen, seed=5)

    data_cfg = {
        "images_dir": str(root / "images" / DATES[0]),
        "labels_dir": str(root / "annotations" / DATES[0]),
        "subject": SUBJECT, "split": {"manifest_dir": str(chosen)},
    }
    train_ds, val_ds, label_digests = auto_train_val("detection", data_cfg, None)
    create_experiment("exp-bound-split-record", {})
    persist_split_manifest("exp-bound-split-record", train_ds, val_ds, data_cfg,
                           label_digests=label_digests)

    split_record = read_split_manifest("exp-bound-split-record")
    assert split_record["manifest_binding"]["manifest_dir"] == str(chosen)
