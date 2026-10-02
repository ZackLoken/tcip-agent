"""Several of ``test_store_contract.py``'s ``REGISTERED`` goldens, checked against their producers.

A golden there proves placement and encoding, never shape. Each case here derives the shape from
the same producer the platform ships and checks the registered golden agrees with it, so a
golden carrying a shape no producer writes is caught here.
"""

from __future__ import annotations

from tcip_mcp.pipelines.data import selection

from tests.test_store_contract import REGISTERED


def test_the_selection_golden_carries_each_sample_s_own_source_label_group_and_side(tmp_path):
    """A selection's record is its sample list: each entry names its own source and label rather
    than a shared root, so the golden cannot carry a per-date members block or a bare id list."""
    import tcip_store as ts

    from tcip_mcp.pipelines.data.label_queries import registry_scope

    selection.write_selection(
        tmp_path / "splits",
        selection.Selection(
            samples=(
                selection.Sample(member="a_1", source=str(tmp_path / "images/2026-03-04/a_1.jpg"),
                                 ground_truth=str(tmp_path / "annotations/2026-03-04/a_1.json"),
                                 group="a", side="train",
                                 ground_truth_digest="7f3a1b9c2d4e5f60"),
            ),
            scope=registry_scope(tmp_path, "bud"), seed=42,
            group_by="stem", dataset_fingerprint="7ac1",
        ),
        project=tmp_path,
    )
    fresh = ts.read(selection.selection_key(tmp_path / "splits"))
    golden = REGISTERED["selection"].golden
    assert isinstance(golden, dict)

    assert "members" not in golden and "splits" not in golden and "date" not in golden
    assert set(golden["samples"][0]) == set(fresh["samples"][0]) >= {
        "member", "source", "ground_truth", "group", "side"}
    assert set(golden) == set(fresh)


def test_the_delivery_events_golden_carries_every_key_a_delivery_records(tmp_path):
    """A real delivery through the chain's own producers leaves an event whose keys, and whose
    bucket finding's keys, are exactly the golden's."""
    import pytest

    pytest.importorskip("torch")
    from tcip_mcp.delivery import read_delivery_events
    from tcip_mcp.tools.inference_tools import deliver_per_image_counts

    from tests import _trait_fixtures as fx
    from tests._chain_fixtures import run_the_chain

    chain = run_the_chain(tmp_path, experiment_id="exp-golden-event")
    delivered = deliver_per_image_counts(tmp_path, predictions_dir=str(chain.bucket),
                                         output_path=str(tmp_path / "out.csv"),
                                         trait=fx.COUNT_TRAIT)
    assert "error" not in delivered, delivered
    (event,) = read_delivery_events(tmp_path)
    fresh = event.model_dump(mode="json")
    golden = REGISTERED["delivery_events"].golden
    assert isinstance(golden, dict)
    assert set(golden) == set(fresh)
    assert set(golden["buckets"][0]) == set(fresh["buckets"][0])
    assert set(golden["producer"]) == set(fresh["producer"])


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
    (golden_op,), (fresh_op,) = (golden_revision["entry"]["operationalizations"].values(),
                                 fresh_revision["entry"]["operationalizations"].values())
    assert set(golden_op) == set(fresh_op)
