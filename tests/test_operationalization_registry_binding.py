"""A state trait's positive class is a class the delivered dataset's registry declares.

A crossing statement names a subject and a positive class the delivered dataset never chose for
itself; the registry is where the dataset actually says what a subject's instances can be called.
These cases pin the predicate that answers whether a class is really declared, the statement
writer's registry requirement, the tool's ``dataset_root`` resolution, and the delivery-time
supersession a registry that stops declaring the class produces.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp import class_registry as cr
from tcip_mcp import operationalization as op
from tests import _operationalization_fixtures as fx


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return fx.seed_project(tmp_path / "project")


# ── the predicate ─────────────────────────────────────────────────────────────


def test_positive_class_problem_is_none_when_the_registry_declares_it(project: Path) -> None:
    registry = cr.registry_for_dataset_root(project)
    assert cr.positive_class_problem(registry, "flower", "open") is None


def test_positive_class_problem_names_an_unknown_subject(project: Path) -> None:
    registry = cr.registry_for_dataset_root(project)
    problem = cr.positive_class_problem(registry, "no_such_subject", "open")
    assert problem is not None and "no subject" in problem


def test_positive_class_problem_names_a_subject_with_no_attributes() -> None:
    registry = cr.ClassRegistry(subjects=(cr.Subject(name="bush"),))
    problem = cr.positive_class_problem(registry, "bush", "open")
    assert problem is not None and "no attributes" in problem


def test_positive_class_problem_names_the_value_not_among_the_attributes(project: Path) -> None:
    registry = cr.registry_for_dataset_root(project)
    problem = cr.positive_class_problem(registry, "flower", "shed")
    assert problem is not None and "'shed'" in problem and "'flower'" in problem


# ── the statement writer's registry requirement ──────────────────────────────


def test_a_crossing_statement_with_no_registry_refuses_by_name(project: Path) -> None:
    with pytest.raises(ValueError, match="needs the delivered dataset's registry"):
        op.state_operationalization(
            project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES,
            statement="s", mechanism="m", measured_subject="flower",
            delivered_phenotypes=["bloom_05per_date", "bloom_50per_date"],
        )


def test_a_crossing_statement_naming_a_class_absent_from_the_registry_refuses(project: Path) -> None:
    registry = cr.ClassRegistry(subjects=(
        cr.Subject(name="flower", attributes=(
            cr.Attribute(name="state", type="categorical", values=("closed",)),
        )),
    ))

    with pytest.raises(ValueError, match="is not among subject 'flower'"):
        op.state_operationalization(
            project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES,
            statement="s", mechanism="m", measured_subject="flower",
            delivered_phenotypes=["bloom_05per_date", "bloom_50per_date"],
            registry=registry,
        )


def test_a_crossing_statement_over_a_registry_that_declares_the_class_succeeds(project: Path) -> None:
    registry = cr.registry_for_dataset_root(project)

    record = op.state_operationalization(
        project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES,
        statement="s", mechanism="m", measured_subject="flower",
        delivered_phenotypes=["bloom_05per_date", "bloom_50per_date"],
        registry=registry,
    )

    assert record["measured_subject"] == "flower"


def test_other_delivery_kinds_ignore_the_registry_keyword(project: Path) -> None:
    """A per_image_count statement needs no registry: passing one explicitly changes nothing."""
    record = op.state_operationalization(
        project, fx.COUNT_TRAIT, op.PER_IMAGE_COUNT,
        statement="s", mechanism="m", measured_subject=fx.COUNT_SUBJECT,
        delivered_phenotypes=[], registry=cr.ClassRegistry(),
    )

    assert record["measured_subject"] == fx.COUNT_SUBJECT


# ── the tool's dataset_root resolution ────────────────────────────────────────


def test_the_tool_refuses_a_project_registering_two_datasets_with_no_dataset_root(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.operationalization_tools import state_trait_operationalization
    from tcip_mcp.tools.project_tools import register_dataset

    project = fx.seed_project(tmp_path / "project")
    dataset_a, dataset_b = tmp_path / "dataset_a", tmp_path / "dataset_b"
    dataset_a.mkdir()
    dataset_b.mkdir()
    register_dataset(str(dataset_a), "chestnut", project_root=str(project))
    register_dataset(str(dataset_b), "currant", project_root=str(project))

    result = state_trait_operationalization(
        project_root=str(project), trait=fx.CROSSING_TRAIT,
        delivery_kind=op.STATE_CROSSING_DATES,
        statement="s", mechanism="m", measured_subject="flower",
        delivered_phenotypes=["bloom_05per_date", "bloom_50per_date"],
    )

    assert "error" in result
    assert "registers 2 datasets" in result["error"]
    assert dataset_a.name in result["error"] and dataset_b.name in result["error"]
    assert "dataset_root" in result["error"]


def test_the_tool_resolves_the_project_roots_own_registry_when_unambiguous(project: Path) -> None:
    from tcip_mcp.tools.operationalization_tools import state_trait_operationalization

    result = state_trait_operationalization(
        project_root=str(project), trait=fx.CROSSING_TRAIT,
        delivery_kind=op.STATE_CROSSING_DATES,
        statement="the date each plant reached the state the breeder scores in the field",
        mechanism="the calibrated state classifier over the isolated flowers of one plant",
        measured_subject="flower",
        delivered_phenotypes=["bloom_05per_date", "bloom_50per_date"],
    )

    assert "error" not in result, result
    assert result["measured_subject"] == "flower"


def test_the_tool_resolves_an_explicit_dataset_root_over_the_project_roots_own(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.operationalization_tools import state_trait_operationalization

    project = fx.seed_project(tmp_path / "project")  # declares "open" for flower
    other_dataset = tmp_path / "other_dataset"
    other_dataset.mkdir()
    fx.seed_positive_class(other_dataset, "flower", "shed")  # declares "shed", not "open"

    result = state_trait_operationalization(
        project_root=str(project), trait=fx.CROSSING_TRAIT,
        delivery_kind=op.STATE_CROSSING_DATES,
        statement="s", mechanism="m", measured_subject="flower",
        delivered_phenotypes=["bloom_05per_date", "bloom_50per_date"],
        dataset_root=str(other_dataset),
    )

    assert "error" in result
    assert "is not among subject 'flower'" in result["error"]


def test_the_tool_refuses_an_explicit_dataset_root_with_no_registry_by_name(
    tmp_path: Path,
) -> None:
    """An explicit dataset_root with no classes.json refuses through the tool's own error
    channel rather than raising FileNotFoundError out of it."""
    from tcip_mcp.tools.operationalization_tools import state_trait_operationalization

    project = fx.seed_project(tmp_path / "project")
    bare_dataset = tmp_path / "bare_dataset"
    bare_dataset.mkdir()

    result = state_trait_operationalization(
        project_root=str(project), trait=fx.CROSSING_TRAIT,
        delivery_kind=op.STATE_CROSSING_DATES,
        statement="s", mechanism="m", measured_subject="flower",
        delivered_phenotypes=["bloom_05per_date", "bloom_50per_date"],
        dataset_root=str(bare_dataset),
    )

    assert "error" in result
    assert bare_dataset.name in result["error"]
    assert "no class registry" in result["error"]


# ── the delivery-time supersession ────────────────────────────────────────────


def test_a_confirmed_crossing_delivery_whose_registry_lost_the_class_reports_a_registry_problem(
    project: Path,
) -> None:
    record = fx.state_crossing(project)
    fx.confirm(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, record)
    spec, stored, _ = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)
    registry_without_class = cr.ClassRegistry(subjects=(
        cr.Subject(name="flower", attributes=(
            cr.Attribute(name="state", type="categorical", values=("closed",)),
        )),
    ))

    check = op.check_operationalization(
        spec, stored, op.STATE_CROSSING_DATES, registry=registry_without_class,
    )

    assert not check.ok
    assert check.state is None
    assert check.superseded == ()
    assert check.registry_problem == cr.positive_class_problem(
        registry_without_class, "flower", "open")


def test_reconfirming_does_not_clear_a_live_registry_problem(project: Path) -> None:
    record = fx.state_crossing(project)
    fx.confirm(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, record)
    registry_without_class = cr.ClassRegistry(subjects=(
        cr.Subject(name="flower", attributes=(
            cr.Attribute(name="state", type="categorical", values=("closed",)),
        )),
    ))
    spec, stored, _ = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)
    before = op.check_operationalization(
        spec, stored, op.STATE_CROSSING_DATES, registry=registry_without_class)
    assert before.registry_problem is not None

    op.confirm_trait_operationalization(
        project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES,
        user="user:breeder", record_seen=op.record_seen_hash(stored.value),
        identity_from_request=True,
    )

    spec2, stored2, _ = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)
    after = op.check_operationalization(
        spec2, stored2, op.STATE_CROSSING_DATES, registry=registry_without_class)
    assert after.registry_problem is not None
    assert not after.ok


def test_a_confirmed_crossing_delivery_whose_registry_still_declares_the_class_is_ok(
    project: Path,
) -> None:
    record = fx.state_crossing(project)
    fx.confirm(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, record)
    spec, stored, _ = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)
    registry = cr.registry_for_dataset_root(project)

    check = op.check_operationalization(spec, stored, op.STATE_CROSSING_DATES, registry=registry)

    assert check.ok


def test_the_results_panel_reports_a_registry_dropped_class_as_not_current(project: Path) -> None:
    from fastapi.testclient import TestClient

    from tcip_mcp.dataset_layout import classes_path
    from tcip_web.app import app

    record = fx.state_crossing(project)
    fx.confirm(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, record)
    cr.write_registry(classes_path(project), cr.ClassRegistry(subjects=(
        cr.Subject(name="flower", attributes=(
            cr.Attribute(name="state", type="categorical", values=("closed",)),
        )),
    )))

    client = TestClient(app, base_url="http://127.0.0.1")
    body = client.get(
        "/api/results/operationalization",
        params={"project_root": str(project), "trait": fx.CROSSING_TRAIT,
                "delivery_kind": op.STATE_CROSSING_DATES},
    ).json()

    assert body["confirmed_current"] is False
    assert body["superseded"] == []
    assert body["registry_problem"] is not None


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


def test_registry_for_pred_dirs_resolves_the_registry_through_compute_phenologys_own_path(
    tmp_path: Path,
) -> None:
    """compute_phenology resolves its registry from the buckets it delivers
    (registry_for_pred_dirs), not from the project root: a registry written where the buckets
    actually resolve to is what a crossing delivery's positive-class check reads."""
    from tcip_mcp.dataset_layout import classes_path
    from tcip_mcp.tools.phenology_tools import compute_phenology
    from tests.test_phenology_tools import _bucket, _ds_root, _write_op_sidecar, _write_preds

    fx.seed_project(tmp_path)
    record = fx.state_crossing(tmp_path)
    fx.confirm(tmp_path, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, record)
    ds_root = _ds_root(tmp_path)
    bucket = _bucket(tmp_path, "2026-02-11")
    _write_preds(bucket, "PLANT_A_2026-02-11", ["open"])
    id_map = {"closed": 0, "open": 1}
    _write_op_sidecar(bucket, dataset_root=ds_root, validated=False, id_map=id_map,
                      trait=fx.CROSSING_TRAIT)
    cr.write_registry(classes_path(ds_root), cr.ClassRegistry(subjects=(
        cr.Subject(name="flower", attributes=(
            cr.Attribute(name="state", type="categorical", values=("closed", "open")),
        )),
    )))
    mapping_name = "valley"
    import tcip_store as ts
    from tcip_mcp.pipelines.postprocessing.plant_mapping import plant_mapping_key

    ts.replace(plant_mapping_key(tmp_path, mapping_name), {
        "2026-02-11": [{"stem": "PLANT_A_2026-02-11", "plot_name": "P1", "accession_name": "acc-9"}],
    })

    res = compute_phenology(
        trait=fx.CROSSING_TRAIT,
        mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(bucket)},
        output_csv_path=str(tmp_path / "out.csv"),
    )

    # Reached the unvalidated-evidence gate, past the operationalization/registry check: the
    # registry at ds_root, resolved from the bucket itself, declared the positive class.
    assert "error" in res
    assert "validated" in res["error"]
    assert "no class registry" not in res["error"]
    assert not (tmp_path / "out.csv").exists()

    # A registry at the same resolved root that drops the class refuses at the earlier check.
    cr.write_registry(classes_path(ds_root), cr.ClassRegistry(subjects=(
        cr.Subject(name="flower", attributes=(
            cr.Attribute(name="state", type="categorical", values=("closed",)),
        )),
    )))
    refused = compute_phenology(
        trait=fx.CROSSING_TRAIT,
        mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(bucket)},
        output_csv_path=str(tmp_path / "out2.csv"),
    )
    assert "error" in refused
    assert "no longer holds" in refused["error"]
