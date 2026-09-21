"""Several of ``test_store_contract.py``'s ``REGISTERED`` goldens, checked against their producers.

A golden there proves placement and encoding, never shape. Each case here derives the shape from
the same producer the platform ships and checks the registered golden agrees with it, so a
golden carrying a shape no producer writes is caught here.
"""

from __future__ import annotations

from tcip_mcp.experiments import create_experiment, read_run_partition
from tcip_mcp.pipelines.data import selection, splits
from tcip_mcp.pipelines.data.split_construction import persist_run_partition
from tcip_web.routes.inference import InferenceJob, _summary

from tests.test_experiment_validations import _real_selection_disjointness
from tests.test_experiment_validations import _row as validation_row
from tests.test_store_contract import LOCK_IDENTITY, REGISTERED


def test_the_selection_golden_carries_each_sample_s_own_source_label_group_and_side(tmp_path):
    """A selection's record is its sample list: each entry names its own source and label rather
    than a shared root, so the golden cannot carry a per-date members block or a bare id list."""
    fresh = selection.selection_document(selection.write_selection(
        tmp_path / "splits",
        selection.Selection(
            samples=(
                selection.Sample(source="images/2026-03-04/a_1.jpg",
                                 ground_truth="annotations/2026-03-04/a_1.json",
                                 group="a", side="train",
                                 confirmation_bucket="bud/2026-03-04",
                                 ground_truth_digest="7f3a1b9c2d4e5f60"),
            ),
            subject="bud", attribute=None, id_map={"bud": 0}, seed=42, group_by="stem",
            dataset_fingerprint="7ac1", admission_counts={"annotated": 1},
            realized_ratios={"train": 1.0, "val": 0.0, "calibration": 0.0},
        ),
    ))
    golden = REGISTERED["selection"].golden
    assert isinstance(golden, dict)

    assert "members" not in golden and "splits" not in golden and "date" not in golden
    assert set(golden["samples"][0]) >= {
        "source", "ground_truth", "group", "side", "confirmation_bucket"}
    assert set(golden) == set(fresh)


def test_the_cal_holdout_lock_golden_carries_every_key_the_resolver_writes(tmp_path):
    fresh = splits.resolve_locked_cal_holdout_split(
        ["a_1", "b_2", "c_3", "d_4"], identity_hash=LOCK_IDENTITY, scope_root=tmp_path)
    golden = REGISTERED["cal_holdout_split_lock"].golden
    assert isinstance(golden, dict)

    assert set(golden) == set(fresh) == {
        "identity_hash", "calibration", "holdout", "group_by", "group_key_map", "seed",
        "holdout_ratio", "selection_dir", "redraw_history",
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
    assert len(golden) == 12, "the resolver produces twelve keys, never a four-key shape"


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


def test_the_experiment_split_golden_carries_every_key_persist_run_partition_writes(tmp_path):
    """Over a partition the platform's own producer named, so the golden is checked against a
    record carrying real per-scope membership rather than against the four keys a member-less run
    writes."""
    from tcip_mcp.pipelines.data.split_construction import _recorded_partition

    from tests._producer_fixtures import samples_over

    images_dir, table = tmp_path / "images", tmp_path / "ranks.csv"
    images_dir.mkdir()
    for stem in ("img_001", "img_002"):
        (images_dir / f"{stem}.jpg").write_bytes(b"")
    table.write_text("image,rank\nimg_001,1\nimg_002,2\n", encoding="utf-8", newline="\n")
    samples = samples_over(images_dir, table)
    partition = _recorded_partition(samples[:1], samples[1:], samples)

    experiment_id = "exp-fixture-shape-check"
    create_experiment(experiment_id, {"model_source": {"builder": "my_module:build"}})
    persist_run_partition(
        experiment_id, {"split": {"resolved_group_by": "stem_prefix"}},
        dataset_id="a1", dataset_fingerprint="7ac1", partition=partition,
    )
    fresh = read_run_partition(experiment_id)
    golden = REGISTERED["experiment_split"].golden
    assert isinstance(golden, dict)

    assert set(golden) == set(fresh)
    golden_block = next(iter(golden["members"].values()))
    fresh_block = next(iter(fresh["members"].values()))
    assert set(golden_block) == set(fresh_block)
    assert set(golden_block["label_digests"]) == set(fresh_block["label_digests"])


def test_the_trait_specs_golden_carries_every_field_the_encoder_writes():
    """``_encode_spec`` writes every ``TraitSpec`` field plus the ``schema_version`` stamp; the
    golden carries exactly those keys, so a field added to the dataclass is caught here."""
    from tcip_mcp import traits

    from tests.test_store_contract import TRAIT_UNDER_TEST

    fresh = traits._encode_spec(traits.TraitSpec(name=TRAIT_UNDER_TEST))
    golden = REGISTERED["trait_specs"].golden
    assert isinstance(golden, dict)

    assert set(golden) == set(fresh)
    assert golden["schema_version"] == fresh["schema_version"] == traits.TRAIT_SPEC_SCHEMA_VERSION
