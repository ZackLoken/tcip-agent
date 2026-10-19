"""Trait entries and their admission.

The tests cover what the schema and a proposal admit, how the calibration path reads an entry,
and the positive state a bucket's scope classifies.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from tcip_mcp import subject_registry as cr, traits
from tcip_mcp.operationalization import latest_confirmed
from tcip_mcp.pipelines.postprocessing import phenology
from tcip_mcp.traits import TraitEntry, TraitUnknownError, trait_names
from tests._trait_fixtures import (
    BUD_OPENING,
    entry,
    latest,
    propose,
    propose_and_confirm,
)

if TYPE_CHECKING:
    from tcip_mcp.buckets import Bucket

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")


# ── what the schema and a proposal admit ─────────────────────────────────────


def test_an_entry_delivering_a_vocabulary_phenotype_is_admitted_and_reads_back(
    tmp_path: Path,
) -> None:
    propose(
        tmp_path,
        entry(
            "leaf",
            ("leaf_length",),
            localization="iou_match",
            count_objective="detection_f1",
        ),
    )

    spec = latest("leaf", tmp_path)
    assert (
        spec.delivers == ("leaf_length",) and spec.localization == "iou_match"
    )


@pytest.mark.parametrize(
    "fields, refusal",
    [
        ({"not_a_field": 3}, "not_a_field"),
        ({"provenance": ["name: vocabulary_derived"]}, "provenance"),
        ({"localization": "box"}, "localization"),
    ],
)
def test_an_entry_the_schema_refuses_names_why(
    fields: dict, refusal: str
) -> None:
    """The closed field set and the stated localization vocabulary refuse by name."""
    base = entry("leaf", ("leaf_length",)).model_dump()
    with pytest.raises(ValidationError, match=refusal):
        TraitEntry.model_validate({**base, **fields})


@pytest.mark.parametrize(
    "delivers, refusal",
    [
        (("unicorn_horn_length",), "unicorn_horn_length"),
        ((), "delivers must name at least one"),
    ],
)
def test_a_proposal_off_the_vocabulary_refuses_and_appends_nothing(
    tmp_path: Path,
    delivers: tuple,
    refusal: str,
) -> None:
    """A proposal whose ``delivers`` is empty or leaves crops.yml refuses by name.

    No revision is appended.
    """
    with pytest.raises(ValueError, match=refusal):
        propose(tmp_path, entry("leaf", delivers))
    assert "leaf" not in trait_names(tmp_path)


def test_every_picker_names_an_admitted_count_objective_and_unset_stays_empty() -> (
    None
):
    """Admission accepts each count objective a picker implements.

    An unset count objective stays empty.
    """
    from tcip_mcp.pipelines.operating_point import COUNT_OBJECTIVE_PICKERS

    assert entry("undecided", ("leaf_length",)).count_objective == ""
    for objective in COUNT_OBJECTIVE_PICKERS:
        assert (
            entry(
                "t", ("leaf_length",), count_objective=objective
            ).count_objective
            == objective
        )


# ── how the calibration path reads an entry ──────────────────────────────────


def test_an_unset_count_objective_refuses_the_proposal_asking_the_breeder(
    tmp_path: Path,
) -> None:
    """A count operationalization proposed with no count objective refuses, asking the breeder.

    No platform default stands in for the objective a count is fitted under; a stated one is
    admitted.
    """
    from tests._trait_fixtures import with_operationalization

    floors = {
        "localization": "center_match",
        "count_bias_tolerance_frac": 0.1,
        "count_error_tolerance": 1.0,
        "holdout_match_quality_floor": 0.5,
    }
    undecided = with_operationalization(
        entry("undecided", ("leaf_length",), **floors), traits.PER_IMAGE_COUNT
    )
    decided = with_operationalization(
        entry(
            "decided",
            ("leaf_length",),
            count_objective="detection_f1",
            **floors,
        ),
        traits.PER_IMAGE_COUNT,
    )

    with pytest.raises(traits.UnauthoredFieldError, match="count_objective"):
        propose(tmp_path, undecided)
    assert "undecided" not in trait_names(tmp_path)
    assert propose(tmp_path, decided).entry.count_objective == "detection_f1"


@pytest.mark.parametrize(
    "field",
    [
        "count_objective",
        "crossing_rule",
        "regression_criterion",
        "plant_aggregation",
        "localization",
    ],
)
def test_an_entry_naming_what_no_table_implements_refuses_naming_the_field(
    field: str,
) -> None:
    """An entry naming a value outside a field's vocabulary refuses, naming the field.

    The refusal comes where the entry a proposal takes is built.
    """
    with pytest.raises(ValidationError, match=field):
        entry("custom", ("leaf_length",), **{field: "a_name_no_table_has"})


@pytest.mark.parametrize(
    "vocabulary, table",
    [
        (
            "CountObjective",
            "tcip_mcp.pipelines.operating_point:COUNT_OBJECTIVE_PICKERS",
        ),
        (
            "CrossingRule",
            "tcip_mcp.pipelines.postprocessing.phenology:CROSSING_RULES",
        ),
        (
            "RegressionCriterion",
            "tcip_mcp.pipelines.operating_point:REGRESSION_CRITERIA",
        ),
        (
            "PlantAggregation",
            "tcip_mcp.pipelines.postprocessing.aggregation:PLANT_AGGREGATIONS",
        ),
        ("Localization", "tcip_mcp.traits:LOCALIZATION_FIELDS"),
    ],
)
def test_every_authorable_name_has_an_implementation_and_no_other(
    vocabulary: str, table: str
) -> None:
    """A field's vocabulary and the table its reader subscripts name the same set.

    The empty unauthored member is set aside.
    """
    import importlib
    from typing import get_args

    module, name = table.split(":")
    implemented = getattr(importlib.import_module(module), name)
    authorable = set(get_args(getattr(traits, vocabulary))) - {""}
    assert authorable == set(implemented)


# ── the record ───────────────────────────────────────────────────────────────


def test_every_trait_record_is_read_and_the_latest_revision_answers(
    tmp_path: Path,
) -> None:
    propose(
        tmp_path,
        entry("leaf", ("leaf_length",), count_bias_tolerance_frac=1.0),
    )
    propose(
        tmp_path,
        entry("leaf", ("leaf_length",), count_bias_tolerance_frac=99.0),
    )

    assert set(trait_names(tmp_path)) == {"bud_opening", "leaf"}
    assert latest("leaf", tmp_path).count_bias_tolerance_frac == 99.0


def test_an_unknown_trait_hard_fails(tmp_path: Path) -> None:
    with pytest.raises(TraitUnknownError):
        traits.read_trait("banana", tmp_path)


def test_a_measurement_reader_refuses_a_trait_with_no_confirmed_revision_by_name(
    tmp_path: Path,
) -> None:
    """A measurement reader refuses a trait with no confirmed revision by name.

    A measurement reads the latest confirmed revision: an entry proposed but not confirmed
    refuses by name, and a later unconfirmed proposal leaves what the confirmed one says.
    """
    from tcip_mcp.operationalization import OperationalizationRefusedError

    propose(
        tmp_path,
        entry("pending", ("leaf_length",), count_objective="detection_f1"),
    )
    with pytest.raises(OperationalizationRefusedError, match="'pending'"):
        latest_confirmed("pending", tmp_path)

    propose_and_confirm(
        tmp_path,
        entry("leaf", ("leaf_length",), count_objective="detection_f1"),
    )
    propose(
        tmp_path,
        entry("leaf", ("leaf_length",), count_objective="count_unbiased"),
    )
    assert (
        latest_confirmed("leaf", tmp_path).entry.count_objective
        == "detection_f1"
    )


def test_a_proposal_made_while_another_holds_the_record_lands_as_the_next_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two concurrent proposals both land as successive revisions.

    One started while the other holds the record lands after it, as revision 2.
    """
    import threading

    real_clock = traits.now_iso
    other_landed = threading.Event()
    other: list = []

    def propose_other() -> None:
        other.append(
            propose(
                tmp_path,
                entry("leaf", ("leaf_length",), notes="the other writer"),
            )
        )
        other_landed.set()

    def clock_while_holding_the_record() -> str:
        monkeypatch.setattr(traits, "now_iso", real_clock)
        threading.Thread(target=propose_other).start()
        other_landed.wait(timeout=2)
        return real_clock()

    monkeypatch.setattr(traits, "now_iso", clock_while_holding_the_record)
    mine = propose(
        tmp_path, entry("leaf", ("leaf_length",), notes="this writer")
    )
    assert other_landed.wait(timeout=30)

    record = traits.read_trait("leaf", tmp_path)
    assert [(r.number, r.entry.notes) for r in record.revisions] == [
        (1, "this writer"),
        (2, "the other writer"),
    ]
    assert (mine.number, other[0].number) == (1, 2)


def test_bud_opening_reads_back_as_the_reference_fixture(
    tmp_path: Path,
) -> None:
    t = latest_confirmed("bud_opening", tmp_path).entry
    assert t == BUD_OPENING
    assert t.positive_state == traits.PositiveState(
        attribute="opening", value="open"
    )
    assert t.localization_tolerance_frac == 0.5
    assert t.majority_milestone == "95per"
    assert (
        t.count_bias_tolerance_frac is None
    )  # not yet authored by the domain expert
    assert set(t.delivers) == {"leaf_out_05per_date", "leaf_out_50per_date"}


def test_a_trait_record_lands_in_the_project_state_database(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "fresh"
    propose(project_root, entry("leaf", ("leaf_length",)))

    assert (project_root / ".tcip" / "state" / ".tcip" / "store.db").is_file()
    assert traits.trait_names(project_root) == ["leaf"]


# ── the positive state, one attribute and one of its values ──────────────────

OPENING = cr.Attribute("opening", "categorical", ("closed", "open"))
"""The attribute :data:`BUD_OPENING`'s positive state names."""
GRADE = cr.Attribute("grade", "ordinal", ("low", "high"))


def _crossing(**state: str) -> TraitEntry:
    """Return :data:`BUD_OPENING` stating ``state`` as its positive state.

    The entry carries a crossing operationalization over ``bud``, with its count bias and error
    tolerances and classifier agreement floor authored.
    """
    from tests._trait_fixtures import with_fields, with_operationalization

    return with_operationalization(
        with_fields(
            BUD_OPENING,
            positive_state=state,
            count_bias_tolerance_frac=0.1,
            count_error_tolerance=1.0,
            classifier_agreement_floor=0.6,
        ),
        traits.STATE_CROSSING_DATES,
        measured_subject="bud",
        delivered_phenotypes=BUD_OPENING.delivers,
    )


def test_a_positive_state_its_registry_does_not_declare_refuses_at_proposal(
    tmp_path: Path,
) -> None:
    """A positive state its registry does not declare refuses at proposal.

    The positive state names its attribute and its value as one fact, each held to the
    registry: a value the attribute does not list refuses, an attribute the subject does not
    declare refuses, and a value of the subject's second attribute is admitted.
    """
    from tests._producer_fixtures import registry_over

    registry_over(
        tmp_path,
        cr.SubjectRegistry(
            subjects=(
                cr.Subject(
                    name="bud", description="", attributes=(OPENING, GRADE)
                ),
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="attribute 'opening' of 'bud' declares no value 'shed'",
    ):
        propose(tmp_path, _crossing(attribute="opening", value="shed"))
    with pytest.raises(
        ValueError, match="subject 'bud' declares no attribute 'color'"
    ):
        propose(tmp_path, _crossing(attribute="color", value="open"))

    admitted = propose(tmp_path, _crossing(attribute="grade", value="high"))
    assert admitted.entry.positive_state == traits.PositiveState(
        attribute="grade", value="high"
    )


@pytest.mark.parametrize(
    "kind, field",
    [
        (traits.STATE_CROSSING_DATES, "crossing_rule"),
        (traits.PER_PLANT_COUNT_AGGREGATE, "plant_aggregation"),
        (traits.PER_PLANT_ORDINAL_AGGREGATE, "plant_aggregation"),
        (traits.PER_PLANT_REGRESSION_AGGREGATE, "plant_aggregation"),
    ],
)
def test_an_operationalization_leaving_its_breeder_definition_unauthored_refuses_asking_for_it(
    tmp_path: Path, kind: str, field: str
) -> None:
    """An operationalization leaving its breeder definition unauthored refuses, asking for it.

    The rule a milestone's day is read under and the statistic a plant's images combine into
    are the trait's definition: an operationalization resting on one refuses unauthored at the
    proposal, asking the breeder its question, and appends nothing; authored, it is admitted.
    """
    import re

    from tests._trait_fixtures import (
        with_fields,
        with_floors,
        with_operationalization,
    )

    stated = with_operationalization(
        with_floors(with_fields(BUD_OPENING, name="leaf_timing")),
        kind,
        measured_subject="bud",
        delivered_phenotypes=BUD_OPENING.delivers,
        delivered_value_keys=(
            ("count",) if kind in traits.VALUE_KEY_KINDS else ()
        ),
    )
    with pytest.raises(
        traits.UnauthoredFieldError,
        match=re.escape(f"{field}: {traits.QUESTIONS[field]}"),
    ):
        propose(tmp_path, with_fields(stated, **{field: ""}))
    assert "leaf_timing" not in trait_names(tmp_path)
    if kind != traits.STATE_CROSSING_DATES:
        assert propose(tmp_path, stated).entry == stated


# ── the positive state resolved from a prediction bucket's own recorded scope ──


def _bucket(
    project: Path,
    date: str,
    *,
    attributes: tuple,
    images: list[Path] | None = None,
) -> Bucket:
    """Publish and return the bucket ``run/<date>`` of one ``bud`` detection per image.

    Each of ``images`` (by default one frame ``P1.png`` of capture ``date`` of the dataset
    ``ds``) holds one ``bud`` detection carrying the last value of each of ``attributes``,
    published from a checkpoint whose scope declares them (``_chain_fixtures.published``).
    """
    pytest.importorskip("torch")
    from tests._chain_fixtures import published

    registry = cr.SubjectRegistry(
        subjects=(
            cr.Subject(name="bud", description="", attributes=attributes),
        )
    )
    return published(
        project,
        f"run/{date}",
        [
            {
                "image": str(image),
                "width": 8,
                "height": 8,
                "boxes": [[1.0, 1.0, 3.0, 3.0]],
                "scores": [0.9],
                "labels": [1],
                "cap": 2,
                **(
                    {"attributes": [[len(a.values) - 1 for a in attributes]]}
                    if attributes
                    else {}
                ),
            }
            for image in images
            or [project / "ds" / "images" / date / "P1.png"]
        ],
        scope={"subject": "bud"},
        registry=registry,
    )


def test_the_positive_state_resolves_by_name_from_the_buckets_own_scope(
    tmp_path: Path,
) -> None:
    named = _bucket(tmp_path, "2026-02-11", attributes=(OPENING,))
    absent = _bucket(
        tmp_path,
        "2026-02-25",
        attributes=(
            cr.Attribute("opening", "categorical", ("closed", "bud")),
        ),
    )

    assert named.scope.state_ids(BUD_OPENING.positive_state) == (
        0,
        OPENING.values.index("open"),
    )
    assert absent.scope.state_ids(BUD_OPENING.positive_state) is None


# ── end-to-end through the phenology delivery ─────────────────────────────────


def _deliver_series(tmp_path: Path, *, attributed: bool) -> dict:
    """Deliver two dated buckets over two mapped plots under a breeder's acknowledgment.

    The buckets' scope declares the positive state's attribute (``attributed``) or no
    attribute, and they are delivered through the mapping over them; a refusal answers
    ``{"error": ...}``.
    """
    from tests._chain_fixtures import acknowledged
    from tests._mapping_fixtures import PLOTS, map_captures
    from tests._trait_fixtures import seed_positive_class

    captures = map_captures(
        tmp_path, tmp_path / "ds", ["2026-02-11", "2026-03-09"]
    )
    buckets = [
        _bucket(
            tmp_path,
            d,
            attributes=(OPENING,) if attributed else (),
            images=images,
        ).name
        for d, images in captures.items()
    ]
    seed_positive_class(tmp_path / "ds", "bud", BUD_OPENING.positive_state)
    try:
        measurement = phenology.measure_phenology(
            tmp_path,
            trait="bud_opening",
            mapping_name="valley",
            dataset_root=tmp_path / "ds",
            buckets=buckets,
            plants=list(PLOTS),
        )
        return acknowledged(
            tmp_path,
            lambda ack: phenology.deliver_phenology(
                tmp_path,
                measurement,
                curves=False,
                output_path=tmp_path / "out.csv",
                acknowledgment_id=ack,
                door="test_trait_authoring",
                actor=None,
            ),
            reason="no assessment in this fixture",
        )
    except ValueError as exc:
        return {"error": str(exc)}


@pytest.mark.usefixtures("seed_bud_operationalization")
def test_a_series_classifying_the_states_attribute_delivers(
    tmp_path: Path,
) -> None:
    """A series classifying the positive state's attribute delivers.

    The delivered milestones are read under the crossing rule the shipped revision states, and
    the rows pin that revision by number and digest and carry no crossing_rule column.
    """
    from tcip_mcp.delivery import read_delivery_events
    from tests import csv_rows

    res = _deliver_series(tmp_path, attributed=True)

    assert "error" not in res, res
    rows = csv_rows(tmp_path / "out.csv")
    (event,) = read_delivery_events(tmp_path)
    (shipped,) = [
        r
        for r in traits.read_trait(event.trait, tmp_path).revisions
        if r.number == event.trait_revision
    ]
    assert shipped.entry_sha256 == event.trait_revision_sha256
    assert shipped.entry.crossing_rule == BUD_OPENING.crossing_rule
    assert {
        (r["trait_revision"], r["trait_revision_sha256"]) for r in rows
    } == {(str(shipped.number), shipped.entry_sha256)}
    assert all("crossing_rule" not in r for r in rows)


@pytest.mark.usefixtures("seed_bud_operationalization")
def test_a_series_that_never_classified_the_positive_state_refuses(
    tmp_path: Path,
) -> None:
    res = _deliver_series(tmp_path, attributed=False)

    assert "classify no opening='open'" in res["error"]
    assert not (tmp_path / "out.csv").exists()
