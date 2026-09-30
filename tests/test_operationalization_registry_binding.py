"""A state trait's positive class is a class the delivered dataset's registry declares.

A crossing operationalization names a subject and a positive class the delivered dataset never
chose for itself; the registry is where the dataset says what a subject's instances can be called.
These cases pin the predicate that answers whether a class is declared, the proposal's registry
check with the tool's ``dataset_root`` resolution, and the delivery-time refusal a registry that
stops declaring the class produces.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp import subject_registry as cr
from tcip_mcp.operationalization import OperationalizationRefused, confirmed_revision
from tcip_mcp.tools.trait_tools import propose_trait
from tcip_mcp.traits import PER_IMAGE_COUNT, STATE_CROSSING_DATES
from tests import _trait_fixtures as fx
from tests._population import mapped_plants

_CROSSING = fx.with_operationalization(
    fx.CROSSING_SPEC, STATE_CROSSING_DATES, measured_subject="flower",
    delivered_phenotypes=fx.CROSSING_SPEC.delivers)
_WITHOUT_OPEN = cr.SubjectRegistry(subjects=(
    cr.Subject(name="flower", attributes=(
        cr.Attribute(name="state", type="categorical", values=("closed",)),
    )),
))


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    fx.seed_positive_class(root, "flower", fx.CROSSING_SPEC.positive_value)
    return root


def _propose(project: Path, **kwargs) -> dict:
    return propose_trait(project, _CROSSING, rationale="the breeder's words", **kwargs)


# ── the predicate ─────────────────────────────────────────────────────────────


def test_positive_value_problem_is_none_when_the_registry_declares_it(project: Path) -> None:
    registry = cr.registry_for_dataset_root(project)
    assert registry is not None
    assert cr.positive_value_problem(registry, "flower", "open") is None


def test_positive_value_problem_names_an_unknown_subject(project: Path) -> None:
    registry = cr.registry_for_dataset_root(project)
    assert registry is not None
    problem = cr.positive_value_problem(registry, "no_such_subject", "open")
    assert problem is not None and "no subject" in problem


def test_positive_value_problem_names_a_subject_with_no_attributes() -> None:
    registry = cr.SubjectRegistry(subjects=(cr.Subject(name="bush"),))
    problem = cr.positive_value_problem(registry, "bush", "open")
    assert problem is not None and "no attributes" in problem


def test_positive_value_problem_names_the_value_not_among_the_attributes(project: Path) -> None:
    registry = cr.registry_for_dataset_root(project)
    assert registry is not None
    problem = cr.positive_value_problem(registry, "flower", "shed")
    assert problem is not None and "'shed'" in problem and "'flower'" in problem


# ── the proposal's registry check, through the tool ──────────────────────────


def test_the_tool_resolves_the_project_roots_own_registry_when_unambiguous(project: Path) -> None:
    result = _propose(project)

    assert "error" not in result, result
    assert result["entry"]["operationalizations"][STATE_CROSSING_DATES]["measured_subject"] == (
        "flower")


def test_a_count_entry_needs_no_registry(tmp_path: Path) -> None:
    """Only a crossing operationalization is checked against a registry; a project with none
    still takes a count entry."""
    result = propose_trait(
        tmp_path, fx.with_operationalization(fx.COUNT_SPEC, PER_IMAGE_COUNT),
        rationale="the breeder's words")

    assert "error" not in result, result


def test_the_tool_refuses_a_project_registering_two_datasets_with_no_dataset_root(
    project: Path, tmp_path: Path,
) -> None:
    from tcip_mcp.tools.project_tools import register_dataset

    dataset_a, dataset_b = tmp_path / "dataset_a", tmp_path / "dataset_b"
    dataset_a.mkdir()
    dataset_b.mkdir()
    register_dataset(project, str(dataset_a), "chestnut")
    register_dataset(project, str(dataset_b), "currant")

    result = _propose(project)

    assert "registers 2 datasets" in result["error"]
    assert dataset_a.name in result["error"] and dataset_b.name in result["error"]
    assert "dataset_root" in result["error"]


def test_the_tool_resolves_an_explicit_dataset_root_over_the_project_roots_own(
    project: Path, tmp_path: Path,
) -> None:
    other_dataset = tmp_path / "other_dataset"
    other_dataset.mkdir()
    fx.seed_positive_class(other_dataset, "flower", "shed")

    result = _propose(project, dataset_root=str(other_dataset))

    assert "is not among subject 'flower'" in result["error"]


def test_the_tool_refuses_an_explicit_dataset_root_with_no_registry_by_name(
    project: Path, tmp_path: Path,
) -> None:
    bare_dataset = tmp_path / "bare_dataset"
    bare_dataset.mkdir()

    result = _propose(project, dataset_root=str(bare_dataset))

    assert bare_dataset.name in result["error"]
    assert "no subject registry" in result["error"]


# ── the delivery-time refusal ────────────────────────────────────────────────


def test_a_confirmed_crossing_whose_delivered_registry_lost_the_class_refuses_with_its_problem(
    project: Path,
) -> None:
    fx.propose_and_confirm(project, _CROSSING)

    with pytest.raises(OperationalizationRefused) as excinfo:
        confirmed_revision(STATE_CROSSING_DATES, project=project, trait=fx.CROSSING_TRAIT,
                           registry=_WITHOUT_OPEN)

    problem = cr.positive_value_problem(_WITHOUT_OPEN, "flower", "open")
    assert problem is not None and problem in str(excinfo.value)


def test_confirming_a_new_revision_does_not_clear_a_live_registry_problem(project: Path) -> None:
    fx.propose_and_confirm(project, _CROSSING)
    fx.propose_and_confirm(project, fx.with_fields(_CROSSING, notes="confirmed again"))

    with pytest.raises(OperationalizationRefused):
        confirmed_revision(STATE_CROSSING_DATES, project=project, trait=fx.CROSSING_TRAIT,
                           registry=_WITHOUT_OPEN)


def test_a_confirmed_crossing_whose_registry_still_declares_the_class_delivers(
    project: Path,
) -> None:
    revision = fx.propose_and_confirm(project, _CROSSING)

    assert confirmed_revision(
        STATE_CROSSING_DATES, project=project, trait=fx.CROSSING_TRAIT,
        registry=cr.registry_for_dataset_root(project)) == revision


# ── registry_for_pred_dirs ─────────────────────────────────────────────────────


def test_registry_for_pred_dirs_refuses_directories_spanning_two_dataset_roots(
    tmp_path: Path,
) -> None:
    bucket_a = tmp_path / "ds_a" / "predictions" / "run" / "2026-02-11"
    bucket_b = tmp_path / "ds_b" / "predictions" / "run" / "2026-02-11"
    bucket_a.mkdir(parents=True)
    bucket_b.mkdir(parents=True)

    with pytest.raises(cr.RegistryError, match="more than one dataset root"):
        cr.registry_for_pred_dirs([str(bucket_a), str(bucket_b)])


def test_registry_for_pred_dirs_resolves_the_registry_through_deliver_phenology_milestoness_own_path(
    tmp_path: Path,
) -> None:
    """deliver_phenology_milestones resolves its registry from the buckets it delivers, not from
    the project root: a registry written where the buckets resolve to is what a crossing
    delivery's positive-class check reads."""
    from tcip_mcp.dataset_layout import subjects_path
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones
    from tests._binding_fixtures import write_plant_mapping
    from tests.test_phenology_tools import _bucket, _ds_root, _write_op_sidecar, _write_preds

    fx.seed_positive_class(tmp_path, "flower", "open")
    fx.propose_and_confirm(tmp_path, _CROSSING)
    ds_root = _ds_root(tmp_path)
    bucket = _bucket(tmp_path, "2026-02-11")
    _write_preds(bucket, "PLANT_A_2026-02-11", ["open"])
    _write_op_sidecar(bucket, dataset_root=ds_root, validated=False,
                      id_map={"closed": 0, "open": 1}, trait=fx.CROSSING_TRAIT)
    cr.write_registry(subjects_path(ds_root), cr.SubjectRegistry(subjects=(
        cr.Subject(name="flower", attributes=(
            cr.Attribute(name="state", type="categorical", values=("closed", "open")),
        )),
    )))
    write_plant_mapping(tmp_path, "valley", {
        "2026-02-11": [{"stem": "PLANT_A_2026-02-11", "plot_name": "P1", "accession_name": "acc-9"}],
    }, dataset_root=ds_root)

    def deliver(out: str) -> dict:
        return deliver_phenology_milestones(
            tmp_path, trait=fx.CROSSING_TRAIT, mapping_name="valley",
            plants=mapped_plants(tmp_path, "valley"),
            predictions_by_date={"2026-02-11": str(bucket)}, output_csv_path=str(tmp_path / out))

    res = deliver("out.csv")

    # Reached the unvalidated-evidence gate, past the meaning and registry check.
    assert "validated" in res["error"]
    assert "no subject registry" not in res["error"]
    assert not (tmp_path / "out.csv").exists()

    cr.write_registry(subjects_path(ds_root), _WITHOUT_OPEN)
    refused = deliver("out2.csv")

    assert "no longer declares" in refused["error"]
