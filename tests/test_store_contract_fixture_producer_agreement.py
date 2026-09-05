"""Six of ``test_store_contract.py``'s ``REGISTERED`` goldens (the schema stability audit's
cross-cutting note) pinned a shape their producer no longer writes: a golden there proves
placement and encoding, never shape, so nothing caught the drift. Each case here re-derives the
true shape from the same real producer and checks the registered golden still agrees, so a
golden hand-edited back to a stale shape is caught here rather than silently standing again.
"""

from __future__ import annotations

from tcip_mcp.experiments import create_experiment, read_split_manifest
from tcip_mcp.pipelines.data import splits
from tcip_mcp.pipelines.data.split_construction import persist_split_manifest
from tcip_mcp.tools import data_tools
from tcip_web.routes.inference import InferenceJob, _summary

from tests.test_experiment_validations import _real_selection_disjointness
from tests.test_experiment_validations import _row as validation_row
from tests.test_store_contract import LOCK_IDENTITY, REGISTERED


def test_the_split_manifest_golden_nests_dataset_hash_and_labels_root_under_members(tmp_path):
    """``draw_splits``' writer, ``compose_split_manifest``, writes ``dataset_hash`` and
    ``labels_root`` only inside a date's own ``members`` block, never at the record's top level."""
    fresh = data_tools.compose_split_manifest(
        tmp_path / "splits", seed=42, group_by="stem_prefix", dataset_fingerprint="7ac1",
        subject="bud", attribute=None, id_map={"bud": 0},
        members={"2026-03-04": {"labels_root": "annotations", "images_root": "images",
                                 "dataset_hash": "9f2c",
                                 "label_digests": {"a_1": "7f3a1b9c2d4e5f60"}}},
        splits={"train": ["a_1"], "val": [], "calibration": []},
        admission_counts={"a_1": 1}, calibration_foreground_groups_by_date={},
        realized_ratios={"train": 1.0, "val": 0.0, "calibration": 0.0},
    )
    golden = REGISTERED["split_manifest"].golden
    assert isinstance(golden, dict)

    assert "dataset_hash" not in golden and "labels_root" not in golden
    assert set(golden["members"][next(iter(golden["members"]))]) >= {
        "labels_root", "images_root", "dataset_hash", "label_digests"}
    assert set(golden["splits"]) == {"train", "val", "calibration"}
    assert set(golden) == set(fresh)


def test_the_cal_holdout_lock_golden_carries_every_key_the_resolver_writes(tmp_path):
    fresh = splits.resolve_locked_cal_holdout_split(
        ["a_1", "b_2", "c_3", "d_4"], identity_hash=LOCK_IDENTITY, scope_root=tmp_path)
    golden = REGISTERED["cal_holdout_split_lock"].golden
    assert isinstance(golden, dict)

    assert set(golden) == set(fresh) == {
        "identity_hash", "calibration", "holdout", "group_by", "group_key_map", "seed",
        "holdout_ratio", "split_manifest_dir", "redraw_history",
    }


def test_the_job_registry_golden_carries_job_id_not_id(tmp_path):
    job = InferenceJob(
        job_id="j1", checkpoint_path="model_best.pt", images_dir="images/2026-03-04",
        output_dir="predictions/live/2026-03-04", conf=0.5, iou=0.5,
        slice_hw=(512, 512), overlap=0.2, status="completed",
    )
    fresh = _summary(job)
    golden = REGISTERED["job_registry"].golden[0]

    assert "id" not in golden and golden["job_id"] == "j1"
    assert set(golden) == set(fresh)


def test_the_experiment_validations_golden_carries_the_resolvers_full_selection_disjointness():
    golden = REGISTERED["experiment_validations"].golden["selection_disjointness"]
    fresh = _real_selection_disjointness()

    assert golden == fresh
    assert len(golden) == 12, "the resolver never produces the four-key shape this used to pin"


def test_the_shared_validation_row_fixtures_selection_disjointness_agrees_too():
    row = validation_row()
    assert row["selection_disjointness"] == _real_selection_disjointness()


def test_the_resolve_scale_sidecar_golden_carries_every_key_the_writer_stamps(tmp_path):
    """Re-derives the shape from ``calibrate_physical_scale`` itself, the smallest producer that
    writes a ``resolve_scale.json``, rather than checking the golden only against itself."""
    from tcip_mcp.pipelines.resolution import read_scale_sidecar
    from tcip_mcp.tools.scale_tools import calibrate_physical_scale

    from tests import _operationalization_fixtures as fx
    from tests.test_delivery_gate import _author_scale_tolerance, _calibration_setup

    fx.seed_delivery_traits(tmp_path)
    _author_scale_tolerance(tmp_path, "plant_surface_area")
    pred_dir, labels_dir, ref_csv, _stems, group_key_map, images_dir = _calibration_setup(
        tmp_path, lengths_px=[100.0, 100.0, 100.0, 100.0])
    calibrate_physical_scale(
        trait="plant_surface_area", pred_dir=pred_dir, dataset_root=str(tmp_path / "ds"),
        images_dir=images_dir, unit="mm", reference_subject="cal_bar", labels_dir=labels_dir,
        reference_csv=ref_csv, group_key_map=group_key_map)

    fresh = read_scale_sidecar(pred_dir)
    golden = REGISTERED["resolve_scale_sidecar"].golden
    assert isinstance(golden, dict)
    assert set(golden) == set(fresh)

    scale_golden, scale_fresh = golden["operating_point"]["scale"], fresh["operating_point"]["scale"]
    assert set(scale_golden) == set(scale_fresh)
    assert "unit" in scale_golden and "units" not in scale_golden


def test_the_experiment_split_golden_carries_every_key_persist_split_manifest_writes(tmp_path):
    class _Drawn:
        def __init__(self, stems: list[str]) -> None:
            self.stems = stems

    experiment_id = "exp-fixture-shape-check"
    create_experiment(experiment_id, {"model_source": {"builder": "my_module:build"}})
    persist_split_manifest(
        experiment_id, _Drawn(["img_001"]), _Drawn(["img_002"]), {"labels_dir": ""},
        dataset_id="a1", dataset_fingerprint="7ac1",
    )
    fresh = read_split_manifest(experiment_id)
    golden = REGISTERED["experiment_split"].golden
    assert isinstance(golden, dict)

    assert set(golden) == set(fresh)
