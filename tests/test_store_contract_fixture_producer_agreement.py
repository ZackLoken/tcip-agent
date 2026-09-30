"""Several of ``test_store_contract.py``'s ``REGISTERED`` goldens, checked against their producers.

A golden there proves placement and encoding, never shape. Each case here derives the shape from
the same producer the platform ships and checks the registered golden agrees with it, so a
golden carrying a shape no producer writes is caught here.
"""

from __future__ import annotations

from tcip_mcp.pipelines.data import selection, splits

from tests.test_experiment_validations import _real_selection_disjointness
from tests.test_experiment_validations import _row as validation_row
from tests.test_store_contract import LOCK_IDENTITY, REGISTERED


def test_the_selection_golden_carries_each_sample_s_own_source_label_group_and_side(tmp_path):
    """A selection's record is its sample list: each entry names its own source and label rather
    than a shared root, so the golden cannot carry a per-date members block or a bare id list."""
    import tcip_store as ts

    selection.write_selection(
        tmp_path / "splits",
        selection.Selection(
            samples=(
                selection.Sample(member="a_1", source=str(tmp_path / "images/2026-03-04/a_1.jpg"),
                                 ground_truth=str(tmp_path / "annotations/2026-03-04/a_1.json"),
                                 group="a", side="train",
                                 confirmation_bucket="bud/2026-03-04",
                                 ground_truth_digest="7f3a1b9c2d4e5f60"),
            ),
            scope=selection.ClassScope(subject="bud", id_map={"bud": 0}), seed=42,
            group_by="stem", dataset_fingerprint="7ac1",
        ),
        project=tmp_path,
    )
    fresh = ts.read(selection.selection_key(tmp_path / "splits"))
    golden = REGISTERED["selection"].golden
    assert isinstance(golden, dict)

    assert "members" not in golden and "splits" not in golden and "date" not in golden
    assert set(golden["samples"][0]) >= {
        "member", "source", "ground_truth", "group", "side", "confirmation_bucket"}
    assert set(golden) == set(fresh)


def test_the_cal_holdout_lock_golden_carries_every_key_the_resolver_writes(tmp_path):
    import tcip_store as ts

    splits.resolve_locked_cal_holdout_split(
        ["a_1", "b_2", "c_3", "d_4"], identity_hash=LOCK_IDENTITY, scope_root=tmp_path)
    fresh = ts.read(splits.cal_holdout_lock_key(LOCK_IDENTITY, scope_root=tmp_path))
    golden = REGISTERED["cal_holdout_split_lock"].golden
    assert isinstance(golden, dict)

    assert set(golden) == set(fresh) == {
        "identity_hash", "calibration", "holdout", "group_by", "group_key_map", "seed",
        "holdout_ratio", "redraw_history",
    }


def test_the_shared_validation_row_fixtures_selection_disjointness_is_the_resolvers_own(tmp_path):
    row = validation_row(tmp_path)
    assert row["selection_disjointness"] == _real_selection_disjointness(tmp_path)
    assert len(row["selection_disjointness"]) == 11, (
        "the resolver produces eleven keys, never a four-key shape")


def test_the_resolve_scale_sidecar_golden_carries_every_key_the_writer_stamps(tmp_path):
    """Re-derives the shape from ``calibrate_physical_scale`` itself, the smallest producer that
    writes a ``resolve_scale.json``, rather than checking the golden only against itself."""
    from tcip_mcp.pipelines.resolution import read_scale_sidecar
    from tcip_mcp.tools.scale_tools import calibrate_physical_scale

    from tests import _trait_fixtures as fx
    from tests.test_delivery_gate import _author_scale_tolerance, _calibration_setup

    fx.seed_delivery_traits(tmp_path)
    _author_scale_tolerance(tmp_path, "plant_surface_area")
    pred_dir, labels_dir, ref_csv, _stems, group_key_map, images_dir = _calibration_setup(
        tmp_path, lengths_px=[100.0, 100.0, 100.0, 100.0])
    calibrate_physical_scale(
        tmp_path, trait="plant_surface_area", pred_dir=pred_dir, dataset_root=str(tmp_path / "ds"),
        images_dir=images_dir, unit="mm", reference_subject="cal_bar", labels_dir=labels_dir,
        reference_csv=ref_csv, group_key_map=group_key_map)

    fresh = read_scale_sidecar(pred_dir)
    golden = REGISTERED["resolve_scale_sidecar"].golden
    assert isinstance(golden, dict)
    assert set(golden) == set(fresh)

    scale_golden, scale_fresh = golden["operating_point"]["scale"], fresh["operating_point"]["scale"]
    assert set(scale_golden) == set(scale_fresh)
    assert "unit" in scale_golden and "units" not in scale_golden


def test_the_traits_golden_carries_every_field_the_proposing_and_confirming_producers_write(
    tmp_path,
):
    """The record ``propose_trait`` and ``confirm_revision`` leave on the store carries exactly
    the golden's keys, at the record, the revision, the entry and the operationalization, so a
    field added to the schema is caught here."""
    import tcip_store as ts
    from tcip_mcp import traits

    from tests import _trait_fixtures as fx

    fx.seed_confirmed_count(tmp_path)
    fresh = ts.read(traits.trait_key(tmp_path, fx.COUNT_TRAIT))
    golden = REGISTERED["traits"].golden
    assert isinstance(golden, dict)

    (golden_revision,), fresh_revision = golden["revisions"], fresh["revisions"][-1]
    assert set(golden) == set(fresh)
    assert set(golden_revision) == set(fresh_revision)
    assert set(golden_revision["entry"]) == set(fresh_revision["entry"])
    assert set(golden_revision["proposing_agent"]) == set(fresh_revision["proposing_agent"])
    (golden_op,), (fresh_op,) = (golden_revision["entry"]["operationalizations"].values(),
                                 fresh_revision["entry"]["operationalizations"].values())
    assert set(golden_op) == set(fresh_op)
