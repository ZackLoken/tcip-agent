"""Tests for the phenology MCP tools (build_plant_mapping + deliver_phenology_milestones).

The tools are the agent-facing surface for the per-plant phenology pipeline. These tests pin:
(1) build_plant_mapping wraps build + persist and reports a compact summary + error paths;
(2) deliver_phenology_milestones writes the canonical column schema from classified predictions + a
persisted plant mapping, resolving the positive class id from the prediction buckets' own
recorded id_map and gating both the classifier (reconciled from
classifier_operating_point.json) and the count operating point; and (3) its measurement-
integrity guard refuses to deliver a CSV when no bucket ever classified along the trait's
positive-class axis.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import tcip_store as ts
from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.pipelines.postprocessing.plant_mapping import (
    NEAREST_MATCH_FACTOR,
    NN_TOLERANCE_METERS,
    plant_mapping_key,
)
from tcip_mcp.traits import CENTER_MATCH, get_trait
from tcip_mcp.tools.phenology_tools import (
    _classification_items,
    build_plant_mapping,
    calibrate_classifier_operating_point,
    deliver_phenology_milestones,
)
from tests._binding_fixtures import write_bound_sidecar

# seed_bud_operationalization writes the spec plus the confirmed crossing record this root needs.
pytestmark = pytest.mark.usefixtures("seed_bud_operationalization")


def _plant_csv(path: Path) -> None:
    path.write_text(
        "plot_name,accession_name,WGS84_centroid_x,WGS84_centroid_y\n"
        "P1,acc-9,-90.058,43.197\n",
        encoding="utf-8",
    )


def test_build_plant_mapping_wraps_build_and_persists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import datetime

    from tcip_mcp.tools.project_tools import initialize_project, register_dataset
    from tcip_mcp.traits import registered_crops

    from tests.test_plant_mapping_binding import _write_geo_image

    # tmp_path sits directly under this test's workspace; point the workspace elsewhere so
    # initialize_project's naming rail (which only holds under the workspace) doesn't apply here.
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    assert "error" not in initialize_project(str(tmp_path), site="orchard block")
    images_root = tmp_path / "images"
    _write_geo_image(
        images_root / "2026-02-11" / "img1.jpg", 43.19670, -90.058000,
        datetime(2026, 2, 11, 9, 30))
    register_dataset(str(tmp_path), crop=sorted(registered_crops())[0])
    csv_path = tmp_path / "plants.csv"
    _plant_csv(csv_path)
    name = "valley"

    from tests._binding_fixtures import register_plant_registry_for

    registry = register_plant_registry_for([csv_path])
    res = build_plant_mapping(
        name=name,
        images_root=str(images_root),
        plant_registry=registry,
    )

    assert "error" not in res, res
    assert res["name"] == name
    assert res["project_root"] == str(tmp_path)
    assert res["dataset_root"] == str(tmp_path)
    assert res["n_dates"] == 1
    assert res["n_images"] == 1
    assert res["n_mapped"] + res["n_unattributed"] == 1
    assert "2026-02-11" in res["per_date"]
    assert res["nn_tolerance_m"] == {"value": NN_TOLERANCE_METERS, "source": "fallback"}
    assert res["max_match_distance_m"] == pytest.approx(NN_TOLERANCE_METERS * NEAREST_MATCH_FACTOR)
    persisted = ts.read(plant_mapping_key(tmp_path, name))
    assert list(persisted["assignments"].keys()) == ["2026-02-11"]
    assert persisted["assignments"]["2026-02-11"][0]["stem"] == "img1"
    assert "confidence" not in persisted["assignments"]["2026-02-11"][0]


def test_build_plant_mapping_missing_images_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_mcp.tools.project_tools import initialize_project

    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    assert "error" not in initialize_project(str(tmp_path), site="orchard block")
    res = build_plant_mapping(
        images_root=str(tmp_path / "nope"),
        plant_registry="unregistered",
        name="m",
    )
    assert "error" in res
    assert "images_root not found" in res["error"]


def test_build_plant_mapping_missing_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_mcp.tools.project_tools import initialize_project, register_dataset
    from tcip_mcp.traits import registered_crops

    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    assert "error" not in initialize_project(str(tmp_path), site="orchard block")
    images_root = tmp_path / "images"
    (images_root / "2026-02-11").mkdir(parents=True)
    register_dataset(str(tmp_path), crop=sorted(registered_crops())[0])
    res = build_plant_mapping(
        images_root=str(images_root),
        plant_registry="does-not-exist",
        name="m",
    )
    assert "error" in res
    assert "plant registry not found" in res["error"]
    assert "register_plant_registry" in res["error"]


def _write_mapping(project_root: Path, name: str, mapping: dict) -> None:
    from tests._binding_fixtures import write_plant_mapping

    write_plant_mapping(project_root, name, mapping, dataset_root=_ds_root(project_root))


def _write_preds(dir_path: Path, stem: str, subjects: list[str], *,
                 attribute: str | None = "opening", object_subject: str = "bud") -> None:
    """Detector shape (``attribute=None``): each decoded name lands straight in ``subject``.
    Classified shape (the default, matching :data:`ID_MAP`): every record carries
    ``object_subject`` with its value under ``attribute``."""
    dir_path.mkdir(parents=True, exist_ok=True)
    if attribute is None:
        anns = [Annotation(subject=s, geometry=BBox(1.0, 1.0, 3.0, 3.0), score=0.9)
                for s in subjects]
    else:
        anns = [Annotation(subject=object_subject, geometry=BBox(1.0, 1.0, 3.0, 3.0), score=0.9,
                           attributes={attribute: s}) for s in subjects]
    json_io.write_annotations(dir_path / f"{stem}.json", anns, 8, 8)


def _ds_root(tmp_path: Path) -> Path:
    return tmp_path / "ds"


def _bucket(tmp_path: Path, date: str) -> Path:
    """A prediction bucket under the dataset's own predictions layout, so a count claim's
    covered-bucket key (relative to dataset_root) resolves against a real root."""
    return _ds_root(tmp_path) / "predictions" / "run" / date


def _write_bud_opening_registry(root: Path) -> None:
    """A ``classes.json`` declaring ``bud``'s ``opening`` axis at ``root``: the vocabulary
    ``_classification_items`` resolves for a prediction bucket recording no ``id_map`` of its own.
    """
    from tcip_mcp import class_registry

    class_registry.write_registry(root / "classes.json", class_registry.ClassRegistry(subjects=(
        class_registry.Subject(name="bud", attributes=(
            class_registry.Attribute(name="opening", type="categorical",
                                     values=("open", "closed")),
        )),
    )))


def _write_stamp_bypassing_claim_rail(dir_path: Path, stamp: dict, document: str) -> None:
    """Write one bucket's stamp through the storage seam without the writer-side claim check.

    The real writer (``write_sidecar``) refuses a stamp claiming ``validated`` with no well-formed
    ``validated_by`` pointer, so a fixture standing in for a hand-edited or foreign file (one no real
    producer could write) goes around that check here, the same way it went straight to disk under
    the file backend: the storage location is still the seam's, only the writer-side rail is skipped.
    """
    from tcip_mcp.pipelines.resolution import sidecar_key

    dir_path.mkdir(parents=True, exist_ok=True)
    key = sidecar_key(dir_path, document)
    with ts.transaction(key) as txn:
        txn.write(key, stamp)


def _write_op_sidecar(dir_path: Path, *, dataset_root: Path, validated: bool, conf: float = 0.4,
                      id_map: dict | None = None, experiment_id: str | None = None,
                      checkpoint_sha256: str | None = None,
                      tile_size_prov: dict | None = None, trait: str = "bud_opening",
                      subject: str = "bud", attribute: str | None = "opening") -> None:
    """The operating_point.json a calibrated run_inference writes.

    ``tile_size_prov`` is the tile_size param's own provenance entry, present for a run that
    actually tiled; omitted here for an untiled run, which carries no gating tile scale.
    ``experiment_id`` doubles as the record's producing_experiment_id when ``validated``: the run
    that produced these predictions is the run a genuinely-bound claim names. ``subject`` and
    ``attribute`` default to the classified scope :data:`ID_MAP` decodes; a caller writing a bare
    detector map states ``attribute=None``.
    """
    ref = "held_out_annotations" if validated else "false"
    dir_path.mkdir(parents=True, exist_ok=True)
    op: dict = {"conf": {"value": conf, "validated_against": ref}}
    if tile_size_prov is not None:
        op["tile_size"] = tile_size_prov
    stamp = {
        "validated": validated,
        "trait": trait,
        "operating_point": op,
        "id_map": id_map,
        "experiment_id": experiment_id,
        "checkpoint_sha256": checkpoint_sha256,
        "subject": subject,
        "attribute": attribute,
    }
    if validated:
        write_bound_sidecar(dir_path, stamp, dataset_root=dataset_root,
                            experiment_id=f"exp-record-{dir_path.name}",
                            producing_experiment_id=experiment_id)
    else:
        _write_stamp_bypassing_claim_rail(dir_path, stamp, "operating_point")


def _tiled(ref: str, value: int = 640) -> dict:
    return {"value": value, "requires_validation": True, "validation_kind": "geometry",
            "validated_against": ref}


def _write_classifier_sidecar(dir_path: Path, *, dataset_root: Path, validated: bool,
                              trait: str | None = None, experiment_id: str | None = None) -> None:
    """The classifier_operating_point.json calibrate_classifier_operating_point writes."""
    ref = "held_out_annotations" if validated else "false"
    dir_path.mkdir(parents=True, exist_ok=True)
    stamp = {
        "validated": validated,
        "operating_point": {"classifier": {"value": "open", "validated_against": ref}},
        "trait": trait,
        "experiment_id": experiment_id,
    }
    if validated and trait:
        write_bound_sidecar(dir_path, stamp, document="classifier_operating_point",
                            dataset_root=dataset_root, experiment_id=f"exp-classifier-{dir_path.name}",
                            producing_experiment_id=experiment_id, trait=trait)
    else:
        _write_stamp_bypassing_claim_rail(dir_path, stamp, "classifier_operating_point")


ID_MAP = {"closed": 0, "open": 1}


def test_deliver_phenology_milestones_delivers_when_both_validated(tmp_path: Path) -> None:
    root = _ds_root(tmp_path)
    d1, d2 = _bucket(tmp_path, "2026-02-11"), _bucket(tmp_path, "2026-03-09")
    _write_preds(d1, "P1_a", ["closed"])
    _write_preds(d2, "P1_b", ["open"])
    _write_op_sidecar(d1, dataset_root=root, validated=True, id_map=ID_MAP)
    _write_op_sidecar(d2, dataset_root=root, validated=True, id_map=ID_MAP)
    _write_classifier_sidecar(d1, dataset_root=root, validated=True, trait="bud_opening")
    mapping_name = "valley"
    _write_mapping(tmp_path, mapping_name, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_phenology_milestones(
        trait="bud_opening",
        mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        classifier_pred_dirs=[str(d1)],
    )

    assert "error" not in res, res
    assert res["positive_class_assessed"] is True
    assert out_csv.exists()


def test_deliver_phenology_milestones_signature_names_neither_retired_parameter() -> None:
    """The two caller-stated operating-point parameters are gone; nothing replaces them."""
    import inspect

    names = set(inspect.signature(deliver_phenology_milestones).parameters)
    assert "operating_point_conf" not in names
    assert "operating_point_validated" not in names


def test_deliver_phenology_milestones_derives_one_conf_from_two_stamps_with_no_caller_parameter(
    tmp_path: Path,
) -> None:
    """Coverage: two validated stamps recording the same conf write it into every row, with no
    operating-point parameter this door no longer accepts."""
    root = _ds_root(tmp_path)
    d1, d2 = _bucket(tmp_path, "2026-02-11"), _bucket(tmp_path, "2026-03-09")
    _write_preds(d1, "P1_a", ["closed"])
    _write_preds(d2, "P1_b", ["open"])
    _write_op_sidecar(d1, dataset_root=root, validated=True, id_map=ID_MAP, conf=0.4)
    _write_op_sidecar(d2, dataset_root=root, validated=True, id_map=ID_MAP, conf=0.4)
    _write_classifier_sidecar(d1, dataset_root=root, validated=True, trait="bud_opening")
    mapping_name = "valley"
    _write_mapping(tmp_path, mapping_name, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_phenology_milestones(
        trait="bud_opening", mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv), classifier_pred_dirs=[str(d1)],
    )

    assert "error" not in res, res
    rows = _delivered_rows(out_csv)
    assert rows
    assert all(row["operating_point_conf"] == "0.4" for row in rows)


def test_deliver_phenology_milestones_joins_confs_from_dates_calibrated_apart(tmp_path: Path) -> None:
    """Two validated stamps recording different confs deliver both, joined, in delivered-date
    order, rather than a blank cell."""
    root = _ds_root(tmp_path)
    d1, d2 = _bucket(tmp_path, "2026-02-11"), _bucket(tmp_path, "2026-03-09")
    _write_preds(d1, "P1_a", ["closed"])
    _write_preds(d2, "P1_b", ["open"])
    _write_op_sidecar(d1, dataset_root=root, validated=True, id_map=ID_MAP, conf=0.4)
    _write_op_sidecar(d2, dataset_root=root, validated=True, id_map=ID_MAP, conf=0.6)
    _write_classifier_sidecar(d1, dataset_root=root, validated=True, trait="bud_opening")
    mapping_name = "valley"
    _write_mapping(tmp_path, mapping_name, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_phenology_milestones(
        trait="bud_opening", mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv), classifier_pred_dirs=[str(d1)],
    )

    assert "error" not in res, res
    rows = _delivered_rows(out_csv)
    assert rows
    assert all(row["operating_point_conf"] == "0.4;0.6" for row in rows)


def test_deliver_phenology_milestones_joins_confs_in_dates_delivered_order_not_the_callers_dict_order(
    tmp_path: Path,
) -> None:
    """The conf cell follows the sorted delivered-date order, not the caller's dict order: a
    ``predictions_by_date`` passed with the later date first still delivers the earlier date's
    conf first, matching the sorted ``dates_delivered`` cell beside it."""
    root = _ds_root(tmp_path)
    d1, d2 = _bucket(tmp_path, "2026-02-11"), _bucket(tmp_path, "2026-03-09")
    _write_preds(d1, "P1_a", ["closed"])
    _write_preds(d2, "P1_b", ["open"])
    _write_op_sidecar(d1, dataset_root=root, validated=True, id_map=ID_MAP, conf=0.4)
    _write_op_sidecar(d2, dataset_root=root, validated=True, id_map=ID_MAP, conf=0.6)
    _write_classifier_sidecar(d1, dataset_root=root, validated=True, trait="bud_opening")
    mapping_name = "valley"
    _write_mapping(tmp_path, mapping_name, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_phenology_milestones(
        trait="bud_opening", mapping_name=mapping_name,
        predictions_by_date={"2026-03-09": str(d2), "2026-02-11": str(d1)},
        output_csv_path=str(out_csv), classifier_pred_dirs=[str(d1)],
    )

    assert "error" not in res, res
    rows = _delivered_rows(out_csv)
    assert rows
    assert all(row["operating_point_conf"] == "0.4;0.6" for row in rows)
    assert all(row["dates_delivered"] == "2026-02-11;2026-03-09" for row in rows)


def test_deliver_phenology_milestones_blanks_a_bucket_with_no_numeric_conf_in_the_joined_cell(
    tmp_path: Path,
) -> None:
    """A validated stamp with no numeric conf beside one at 0.4 delivers "0.4;", an empty entry
    for the bucket with no conf, rather than the scalar for both."""
    root = _ds_root(tmp_path)
    d1, d2 = _bucket(tmp_path, "2026-02-11"), _bucket(tmp_path, "2026-03-09")
    _write_preds(d1, "P1_a", ["closed"])
    _write_preds(d2, "P1_b", ["open"])
    _write_op_sidecar(d1, dataset_root=root, validated=True, id_map=ID_MAP, conf=0.4)
    _write_op_sidecar(d2, dataset_root=root, validated=True, id_map=ID_MAP, conf=None)
    _write_classifier_sidecar(d1, dataset_root=root, validated=True, trait="bud_opening")
    mapping_name = "valley"
    _write_mapping(tmp_path, mapping_name, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_phenology_milestones(
        trait="bud_opening", mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv), classifier_pred_dirs=[str(d1)],
    )

    assert "error" not in res, res
    rows = _delivered_rows(out_csv)
    assert rows
    assert all(row["operating_point_conf"] == "0.4;" for row in rows)


def test_deliver_phenology_milestones_reports_an_unreadable_prediction_by_name(tmp_path: Path) -> None:
    """A present, unreadable prediction document is an error naming the file, never a raise
    through the tool boundary and never silently read as this plant's date contributing nothing."""
    root = _ds_root(tmp_path)
    d1, d2 = _bucket(tmp_path, "2026-02-11"), _bucket(tmp_path, "2026-03-09")
    _write_preds(d1, "P1_a", ["closed"])
    d2.mkdir(parents=True, exist_ok=True)
    bad = d2 / "P1_b.json"
    bad.write_text("not json {][", encoding="utf-8")
    _write_op_sidecar(d1, dataset_root=root, validated=True, id_map=ID_MAP)
    _write_op_sidecar(d2, dataset_root=root, validated=True, id_map=ID_MAP)
    _write_classifier_sidecar(d1, dataset_root=root, validated=True, trait="bud_opening")
    mapping_name = "valley"
    _write_mapping(tmp_path, mapping_name, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_phenology_milestones(
        trait="bud_opening",
        mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        classifier_pred_dirs=[str(d1)],
    )

    assert "error" in res
    assert str(bad) in res["error"]


def test_deliver_phenology_milestones_re_reads_the_registry_at_the_second_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registry is re-resolved immediately before the write, not reused from the first check:
    a registry edit racing the delivery is exactly the mid-run move that check exists to catch."""
    from tcip_mcp import class_registry as cr

    root = _ds_root(tmp_path)
    d1, d2 = _bucket(tmp_path, "2026-02-11"), _bucket(tmp_path, "2026-03-09")
    _write_preds(d1, "P1_a", ["closed"])
    _write_preds(d2, "P1_b", ["open"])
    _write_op_sidecar(d1, dataset_root=root, validated=True, id_map=ID_MAP)
    _write_op_sidecar(d2, dataset_root=root, validated=True, id_map=ID_MAP)
    _write_classifier_sidecar(d1, dataset_root=root, validated=True, trait="bud_opening")
    mapping_name = "valley"
    _write_mapping(tmp_path, mapping_name, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    declares_it = cr.ClassRegistry(subjects=(
        cr.Subject(name="bud", attributes=(
            cr.Attribute(name="state", type="categorical", values=("closed", "open")),
        )),
    ))
    drops_it = cr.ClassRegistry(subjects=(
        cr.Subject(name="bud", attributes=(
            cr.Attribute(name="state", type="categorical", values=("closed",)),
        )),
    ))
    calls = {"n": 0}

    def racing_registry(pred_dirs: object) -> cr.ClassRegistry:
        calls["n"] += 1
        return declares_it if calls["n"] == 1 else drops_it

    monkeypatch.setattr("tcip_mcp.class_registry.registry_for_pred_dirs", racing_registry)

    res = deliver_phenology_milestones(
        trait="bud_opening",
        mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        classifier_pred_dirs=[str(d1)],
    )

    assert calls["n"] >= 2
    assert "error" in res
    assert "no longer holds" in res["error"]
    assert not out_csv.exists()


def test_deliver_phenology_milestones_floors_a_count_stamp_earned_for_a_different_trait(tmp_path: Path) -> None:
    """A count stamp validated for one trait must not answer for a phenology delivery under a
    different trait: the refusal names the sidecar and both traits."""
    root = _ds_root(tmp_path)
    d1, d2 = _bucket(tmp_path, "2026-02-11"), _bucket(tmp_path, "2026-03-09")
    _write_preds(d1, "P1_a", ["closed"])
    _write_preds(d2, "P1_b", ["open"])
    _write_op_sidecar(d1, dataset_root=root, validated=True, id_map=ID_MAP, trait="second_trait")
    _write_op_sidecar(d2, dataset_root=root, validated=True, id_map=ID_MAP, trait="second_trait")
    _write_classifier_sidecar(d1, dataset_root=root, validated=True, trait="bud_opening")
    mapping_name = "valley"
    _write_mapping(tmp_path, mapping_name, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_phenology_milestones(
        trait="bud_opening",
        mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        classifier_pred_dirs=[str(d1)],
    )

    assert "error" in res
    assert not out_csv.exists()
    assert "second_trait" in res["error"] and "bud_opening" in res["error"]
    assert str(d1) in res["error"] or str(d2) in res["error"]


def test_deliver_phenology_milestones_reports_n_images_unattributed_when_never_assessed(tmp_path: Path) -> None:
    """The measurement-integrity guard's early return (no bucket anywhere classified the trait's
    positive class) must disclose n_images_unattributed the same way the success path does -- both
    read it off the same delivery disclosure, so an early refusal is not missing a field a later
    success would have carried."""
    root = _ds_root(tmp_path)
    d1 = _bucket(tmp_path, "2026-02-11")
    d1.mkdir(parents=True)
    _write_op_sidecar(d1, dataset_root=root, validated=True, id_map=ID_MAP)
    # No P1_a.json written under d1 -- the mapping names it but nothing was ever inferred for it,
    # so the only date on record is missing, never classified.
    mapping_name = "valley"
    _write_mapping(tmp_path, mapping_name, {
        "2026-02-11": [
            {"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"},
            {"stem": "P1_b", "accession_name": "acc-9"},  # no plot_name -> unmapped
        ],
    })
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_phenology_milestones(
        trait="bud_opening",
        mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1)},
        output_csv_path=str(out_csv),
    )

    assert "error" in res
    assert res["positive_class_assessed"] is False
    assert res["n_images_unattributed"] == 1
    assert not out_csv.exists()


def test_deliver_phenology_milestones_rejects_classifier_stamp_from_unrelated_run(tmp_path: Path) -> None:
    """A genuinely-validated classifier_operating_point.json calibrated for a
    different trait/experiment must not validate an unrelated delivery -- classifier_pred_dirs is a
    separate, caller-supplied list, so reconcile_classifier_validity's own on-disk check alone can't
    see this; the stamp's own recorded trait/experiment_id must agree with what's being delivered."""
    root = _ds_root(tmp_path)
    d1, d2 = _bucket(tmp_path, "2026-02-11"), _bucket(tmp_path, "2026-03-09")
    _write_preds(d1, "P1_a", ["closed"])
    _write_preds(d2, "P1_b", ["open"])
    _write_op_sidecar(d1, dataset_root=root, validated=True, id_map=ID_MAP, experiment_id="run-B")
    _write_op_sidecar(d2, dataset_root=root, validated=True, id_map=ID_MAP, experiment_id="run-B")
    # Genuinely validated (validated=True), but calibrated for a different trait and a different
    # producing run than the one being delivered here.
    other_trait_dir = tmp_path / "unrelated_calibration"
    _write_classifier_sidecar(other_trait_dir, dataset_root=root, validated=True,
                              trait="some_other_trait", experiment_id="run-A-different-model")
    mapping_name = "valley"
    _write_mapping(tmp_path, mapping_name, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_phenology_milestones(
        trait="bud_opening",
        mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        classifier_pred_dirs=[str(other_trait_dir)],
    )

    assert "error" in res
    assert res["positive_state_classifier_validated"] == "false"
    assert not out_csv.exists()


def test_deliver_phenology_milestones_rejects_classifier_stamp_with_no_trait_recorded(tmp_path: Path) -> None:
    """The real writer (calibrate_classifier_operating_point) always
    records a real trait name -- unlike experiment_id, there is no legitimate producer path that
    omits it. A sidecar with trait=None (a hand-edited or foreign file, not one the real writer could
    produce) must not be trusted just because neither the trait-mismatch nor the experiment-mismatch
    branch fires against a null -- both being null would otherwise bypass the binding check entirely."""
    root = _ds_root(tmp_path)
    d1, d2 = _bucket(tmp_path, "2026-02-11"), _bucket(tmp_path, "2026-03-09")
    _write_preds(d1, "P1_a", ["closed"])
    _write_preds(d2, "P1_b", ["open"])
    _write_op_sidecar(d1, dataset_root=root, validated=True, id_map=ID_MAP)
    _write_op_sidecar(d2, dataset_root=root, validated=True, id_map=ID_MAP)
    # Genuinely "validated", but with neither trait nor experiment_id recorded -- the shape a
    # hand-edited/foreign sidecar could carry, never one calibrate_classifier_operating_point writes.
    _write_classifier_sidecar(d1, dataset_root=root, validated=True, trait=None, experiment_id=None)
    mapping_name = "valley"
    _write_mapping(tmp_path, mapping_name, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_phenology_milestones(
        trait="bud_opening",
        mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        classifier_pred_dirs=[str(d1)],
    )

    assert "error" in res
    assert res["positive_state_classifier_validated"] == "false"
    assert not out_csv.exists()


def test_deliver_phenology_milestones_refuses_unvalidated_classifier(tmp_path: Path) -> None:
    root = _ds_root(tmp_path)
    d1, d2 = _bucket(tmp_path, "2026-02-11"), _bucket(tmp_path, "2026-03-09")
    _write_preds(d1, "P1_a", ["closed"])
    _write_preds(d2, "P1_b", ["open"])
    _write_op_sidecar(d1, dataset_root=root, validated=True, id_map=ID_MAP)
    _write_op_sidecar(d2, dataset_root=root, validated=True, id_map=ID_MAP)
    # No classifier_operating_point.json anywhere -> classifier dimension floors to unvalidated.
    mapping_name = "valley"
    _write_mapping(tmp_path, mapping_name, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    out_csv = tmp_path / "out" / "bud_phenology.csv"
    res = deliver_phenology_milestones(
        trait="bud_opening",
        mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
    )
    assert "error" in res
    assert res["positive_state_classifier_validated"] == "false"
    assert not out_csv.exists()


def _deliver_via_writer(
    *, trait: str, mapping_name: str, predictions_by_date: dict[str, str],
    output_csv_path: Path, classifier_pred_dirs: list[str] | None = None,
    acknowledgement,
) -> dict:
    """Deliver through the canonical writer directly, built from the same reconciliation, basis
    and mapping ``deliver_phenology_milestones`` itself resolves before calling it.

    Writer-level, not tool-level: the MCP tool takes no acknowledgement any more, so a test
    proving what an acknowledged, unvalidated delivery stamps on the CSV runs through this
    instead. The producer path (a real request through the web export route) is exercised by
    ``tests/test_tcip_web_results_routes.py``, not here.
    """
    from tcip_mcp.operationalization import (
        STATE_CROSSING_DATES, check_operationalization, resolve_trait_and_record,
    )
    from tcip_mcp.pipelines.postprocessing import phenology, plant_mapping
    from tcip_mcp.pipelines.resolution import (
        bind_classifier_validity, reconcile_classifier_validity, reconcile_operating_point_validity,
        reconcile_tile_size_validity,
    )
    from tcip_mcp.project_paths import platform_state_root

    platform_root = platform_state_root()
    mapping_build, verified = plant_mapping.resolve_delivery_mapping(
        platform_root, mapping_name, predictions_by_date)
    disclosure = mapping_build.delivery_disclosure(verified, list(predictions_by_date))

    pred_dirs = list(predictions_by_date.values())
    recon = reconcile_operating_point_validity(pred_dirs, trait=trait)
    classifier_recon = reconcile_classifier_validity(classifier_pred_dirs or [])
    classifier_state, _note = bind_classifier_validity(
        classifier_recon["validated"], classifier_pred_dirs, pred_dirs, trait=trait)
    tile_recon = reconcile_tile_size_validity(pred_dirs)
    flags = phenology.phenology_delivery_flags(classifier_state, recon["validated"], tile_recon)

    spec, record, _specs_dir = resolve_trait_and_record(trait, STATE_CROSSING_DATES)
    stated = check_operationalization(spec, record, STATE_CROSSING_DATES)
    result = phenology.per_plant_phenology(
        mapping_build.rows(), predictions_by_date,
        positive_class_name=spec.positive_class_name, spec=spec)

    return phenology.write_phenology_csv(
        "test", result["rows"], Path(output_csv_path), spec, flags=flags,
        acknowledgement=acknowledgement, basis=stated.basis,
        operating_point_confs=recon["confs"],
        producer={}, bindings=recon["bindings"], predictions_by_date=predictions_by_date,
        project_root=platform_root, plant_mapping=disclosure)


def test_an_acknowledged_delivery_stamps_the_unvalidated_dimension_false(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.resolution import Acknowledgement

    root = _ds_root(tmp_path)
    d1, d2 = _bucket(tmp_path, "2026-02-11"), _bucket(tmp_path, "2026-03-09")
    _write_preds(d1, "P1_a", ["closed"])
    _write_preds(d2, "P1_b", ["open"])
    _write_op_sidecar(d1, dataset_root=root, validated=True, id_map=ID_MAP)
    _write_op_sidecar(d2, dataset_root=root, validated=True, id_map=ID_MAP)
    mapping_name = "valley"
    _write_mapping(tmp_path, mapping_name, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    out_csv = tmp_path / "out" / "bud_phenology.csv"
    cells = _deliver_via_writer(
        trait="bud_opening",
        mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=out_csv,
        acknowledgement=Acknowledgement(  # provisional delivery, clearly flagged
            acknowledged_by="user:tester", reason="test acknowledgement"),
    )
    assert cells["positive_state_classifier_validated"] == "false"
    assert out_csv.exists()


def test_deliver_phenology_milestones_refuses_asymmetric_validation(tmp_path: Path) -> None:
    # Classifier validated but the count operating point isn't. The gate requires both.
    root = _ds_root(tmp_path)
    d1, d2 = _bucket(tmp_path, "2026-02-11"), _bucket(tmp_path, "2026-03-09")
    _write_preds(d1, "P1_a", ["closed"])
    _write_preds(d2, "P1_b", ["open"])
    _write_op_sidecar(d1, dataset_root=root, validated=False, id_map=ID_MAP)
    _write_op_sidecar(d2, dataset_root=root, validated=False, id_map=ID_MAP)
    _write_classifier_sidecar(d1, dataset_root=root, validated=True, trait="bud_opening")
    mapping_name = "valley"
    _write_mapping(tmp_path, mapping_name, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    out_csv = tmp_path / "out" / "bud_phenology.csv"
    res = deliver_phenology_milestones(
        trait="bud_opening",
        mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        classifier_pred_dirs=[str(d1)],
    )
    assert "error" in res
    assert not out_csv.exists()


def test_writer_acknowledge_stamps_each_dimension_independently(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.resolution import Acknowledgement

    root = _ds_root(tmp_path)
    d1, d2 = _bucket(tmp_path, "2026-02-11"), _bucket(tmp_path, "2026-03-09")
    _write_preds(d1, "P1_a", ["closed"])
    _write_preds(d2, "P1_b", ["open"])
    _write_op_sidecar(d1, dataset_root=root, validated=True, id_map=ID_MAP)
    _write_op_sidecar(d2, dataset_root=root, validated=True, id_map=ID_MAP)
    # Classifier not validated; op point IS validated on disk.
    mapping_name = "valley"
    _write_mapping(tmp_path, mapping_name, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    out_csv = tmp_path / "out" / "bud_phenology.csv"
    cells = _deliver_via_writer(
        trait="bud_opening",
        mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=out_csv,
        acknowledgement=Acknowledgement(acknowledged_by="user:tester", reason="test acknowledgement"),
    )
    assert cells["positive_state_classifier_validated"] == "false"  # never upgraded
    assert cells["operating_point_validated"] == "held_out_annotations"
    assert out_csv.exists()


def _tile_gate_fixture(tmp_path: Path, tile_size_prov: dict | None) -> dict:
    """A fully-validated two-date phenology delivery, varying only the tile scale's own basis."""
    root = _ds_root(tmp_path)
    d1, d2 = _bucket(tmp_path, "2026-02-11"), _bucket(tmp_path, "2026-03-09")
    _write_preds(d1, "P1_a", ["closed"])
    _write_preds(d2, "P1_b", ["open"])
    _write_op_sidecar(d1, dataset_root=root, validated=True, id_map=ID_MAP, tile_size_prov=tile_size_prov)
    _write_op_sidecar(d2, dataset_root=root, validated=True, id_map=ID_MAP, tile_size_prov=tile_size_prov)
    _write_classifier_sidecar(d1, dataset_root=root, validated=True, trait="bud_opening")
    mapping_name = "valley"
    _write_mapping(tmp_path, mapping_name, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    return {
        "trait": "bud_opening",
        "mapping_name": mapping_name,
        "predictions_by_date": {"2026-02-11": str(d1), "2026-03-09": str(d2)},
        "output_csv_path": str(tmp_path / "out" / "bud_phenology.csv"),
        "classifier_pred_dirs": [str(d1)],
    }


def test_deliver_phenology_milestones_refuses_a_fabricated_tile_size(tmp_path: Path) -> None:
    """A phenology fraction is built from per-image counts, and the tile edge scales those counts.
    A tiled bucket whose tile_size fell back to the fabricated default, with no persisted training
    geometry and no explicit caller override, must refuse here even though the classifier and the
    conf beside it are both genuinely validated."""
    args = _tile_gate_fixture(tmp_path, _tiled("false"))
    res = deliver_phenology_milestones(**args)
    assert "error" in res
    assert res["tile_size_validated"] == "false"
    assert res["operating_point_validated"] == "held_out_annotations"  # conf is not what refused
    assert not Path(args["output_csv_path"]).exists()


def test_deliver_phenology_milestones_delivers_when_the_tile_scale_has_a_real_basis(tmp_path: Path) -> None:
    """The rail must admit valid work: a tile edge derived from the checkpoint's own persisted
    training geometry delivers cleanly."""
    args = _tile_gate_fixture(tmp_path, _tiled("persisted_training_geometry", 224))
    res = deliver_phenology_milestones(**args)
    assert "error" not in res, res
    assert res["tile_size_validated"] == "persisted_training_geometry"
    assert Path(args["output_csv_path"]).exists()


def test_deliver_phenology_milestones_never_gates_an_untiled_delivery_on_tile_size(tmp_path: Path) -> None:
    """Buckets from untiled runs carry a non-gating tile_size entry; it must not manufacture a
    refusal over a dimension that was never operative."""
    args = _tile_gate_fixture(tmp_path, {"value": None, "requires_validation": False,
                                         "validation_kind": None, "validated_against": None})
    res = deliver_phenology_milestones(**args)
    assert "error" not in res, res
    assert res["tile_size_validated"] is None
    assert Path(args["output_csv_path"]).exists()


def test_writer_acknowledged_tile_size_floors_the_csv_operating_point_stamp(
    tmp_path: Path,
) -> None:
    """The CSV's operating_point_validated column is the count operating point's only count-side
    stamp, and the tile scale has no column of its own. A delivery whose tile edge only shipped
    through an acknowledgement must not read as fully validated there."""
    import csv

    from tcip_mcp.pipelines.resolution import Acknowledgement

    args = _tile_gate_fixture(tmp_path, _tiled("false"))
    cells = _deliver_via_writer(
        **args, acknowledgement=Acknowledgement(acknowledged_by="user:tester",
                                                reason="test acknowledgement"))
    assert cells["operating_point_validated"] == "false"
    rows = list(csv.DictReader(Path(args["output_csv_path"]).open(encoding="utf-8")))
    assert rows and all(r["operating_point_validated"] == "false" for r in rows)


def test_writer_acknowledged_tile_size_floors_the_csv_classifier_stamp(
    tmp_path: Path,
) -> None:
    """The classifier column has no less claim on the tile scale's floor than the count operating
    point's own column does: a tile edge that only shipped through an acknowledgement must not
    let a genuinely-validated classifier read as fully validated beside it."""
    import csv

    from tcip_mcp.pipelines.resolution import Acknowledgement

    args = _tile_gate_fixture(tmp_path, _tiled("false"))
    cells = _deliver_via_writer(
        **args, acknowledgement=Acknowledgement(acknowledged_by="user:tester",
                                                reason="test acknowledgement"))
    assert cells["positive_state_classifier_validated"] == "false"
    rows = list(csv.DictReader(Path(args["output_csv_path"]).open(encoding="utf-8")))
    assert rows and all(r["positive_state_classifier_validated"] == "false" for r in rows)


def test_deliver_phenology_milestones_refuses_unclassified_predictions(tmp_path: Path) -> None:
    # Predictions from a bare detector (no opening axis at all) must refuse,
    # never report full coverage.
    root = _ds_root(tmp_path)
    d1 = _bucket(tmp_path, "2026-02-11")
    _write_preds(d1, "P1_a", ["bud"], attribute=None)
    _write_op_sidecar(d1, dataset_root=root, validated=True, id_map={"bud": 0}, attribute=None)
    mapping_name = "valley"
    _write_mapping(tmp_path, mapping_name, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_phenology_milestones(
        trait="bud_opening",
        mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1)},
        output_csv_path=str(out_csv),
    )

    assert "error" in res
    assert not out_csv.exists()


def test_deliver_phenology_milestones_missing_mapping(tmp_path: Path) -> None:
    res = deliver_phenology_milestones(
        trait="bud_opening",
        mapping_name="nope",
        predictions_by_date={},
        output_csv_path=str(tmp_path / "out.csv"),
    )
    assert "error" in res
    assert "not found" in res["error"]


def test_deliver_phenology_milestones_unknown_trait_refuses(tmp_path: Path) -> None:
    res = deliver_phenology_milestones(
        trait="not-a-real-trait",
        mapping_name="nope",
        predictions_by_date={},
        output_csv_path=str(tmp_path / "out.csv"),
    )
    assert "error" in res


# ── calibrate_classifier_operating_point: the classifier-validity producer ────────────

def _write_calibration_image(
    gt_dir: Path, pred_dir: Path, stem: str, calls: list[tuple[bool | None, bool]], *,
    image_offset: float,
) -> None:
    """One (GT, pred) file pair with several classified instances, GT and pred boxes at the
    same position per instance (spaced far apart from each other) so they match regardless of
    the exact derived center-match tolerance. ``calls`` is
    ``[(is_true_positive, is_pred_positive), ...]``; ``is_true_positive=None`` writes a GT
    instance with no ``opening`` attribute at all (never assessed), instead of a real value.
    ``image_offset`` shifts every box in this image by a unique amount so distinct images never
    collide on content hash: two images with the same classification pattern must still carry
    different geometry, the same way two different real photos would.
    """
    gt_anns, pred_anns = [], []
    for i, (is_tp, is_pred_pos) in enumerate(calls):
        x = image_offset + i * 20.0
        box = BBox(x, 0.0, x + 8.0, 8.0)
        attrs = {} if is_tp is None else {"opening": "open" if is_tp else "closed"}
        gt_anns.append(Annotation(subject="bud", geometry=box, attributes=attrs))
        pred_anns.append(Annotation(
            subject="bud", geometry=box, score=0.9,
            attributes={"opening": "open" if is_pred_pos else "closed"}))
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    w = int(image_offset + len(calls) * 20.0) + 8
    json_io.write_annotations(gt_dir / f"{stem}.json", gt_anns, w, 8)
    json_io.write_annotations(pred_dir / f"{stem}.json", pred_anns, w, 8)


def _write_split(gt_dir: Path, pred_dir: Path, *, prefix: str, n_images: int,
                 per_image_calls, offset: int = 0, offset_stride: float = 1000.0) -> None:
    """``per_image_calls(image_index) -> list[(is_tp, is_pred_pos)]`` builds one split of
    ``n_images`` files, each named ``f"{prefix}_{offset+i}"``, a distinct stem per split so
    calibration and holdout are disjoint by construction unless a test deliberately reuses one.
    Each image also gets a unique geometry offset (``offset_stride`` apart), so two splits built
    with the same ``offset``/``prefix`` sequence (as a genuine-duplication test wants) land on
    identical geometry, while two splits meant to be independent don't collide by accident.
    """
    for i in range(n_images):
        _write_calibration_image(gt_dir, pred_dir, f"{prefix}_{offset + i}", per_image_calls(i),
                                 image_offset=(offset + i) * offset_stride)


def _one_positive_one_negative(_i: int) -> list[tuple[bool, bool]]:
    """Every image: one correctly-classified positive + one correctly-classified negative."""
    return [(True, True), (False, False)]


def _alternating_calls(i: int) -> list[tuple[bool, bool]]:
    """Every image: one correctly-classified instance, alternating positive/negative by index."""
    return [(i % 2 == 0, i % 2 == 0), (i % 2 == 1, i % 2 == 1)]


def test_calibrate_classifier_operating_point_passes_for_well_formed_reference(tmp_path: Path) -> None:
    root = _ds_root(tmp_path)
    _write_bud_opening_registry(root)
    cal_gt, cal_pred = root / "annotations" / "cal", tmp_path / "cal_pred"
    hold_gt, hold_pred = root / "annotations" / "hold", tmp_path / "hold_pred"
    # 20 images/split, 2 correctly-classified instances each (1 positive, 1 negative) -> perfect,
    # balanced, disjoint (different stems) references on both sides.
    calls = _one_positive_one_negative
    _write_split(cal_gt, cal_pred, prefix="cal", n_images=20, per_image_calls=calls)
    _write_split(hold_gt, hold_pred, prefix="hold", n_images=20, per_image_calls=calls, offset=1000)

    res = calibrate_classifier_operating_point(
        trait_name="bud_opening", subject="bud", attribute="opening",
        calibration_gt_dir=str(cal_gt), calibration_pred_dir=str(cal_pred),
        holdout_gt_dir=str(hold_gt), holdout_pred_dir=str(hold_pred),
        output_dir=str(tmp_path / "out"), dataset_root=str(root), experiment_id=None,
    )

    assert res["passed"] is True, res
    assert res["failures"] == []
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_HELD_OUT,
        read_classifier_operating_point_sidecar,
        reconcile_classifier_validity,
    )

    sidecar = read_classifier_operating_point_sidecar(tmp_path / "out")
    assert sidecar["validated"] is True
    # The writer's field name must be the one the shared reader reads, checked by running the real
    # reader over the real output, not by re-asserting a key name in two places.

    assert sidecar["operating_point"]["classifier"]["validated_against"] == VALIDATED_HELD_OUT
    assert reconcile_classifier_validity([str(tmp_path / "out")])["validated"] == VALIDATED_HELD_OUT


def test_calibrate_classifier_operating_point_reports_an_undecodable_pred_stamp(
    tmp_path: Path,
) -> None:
    """A prediction bucket whose operating_point.json will not decode refuses through this door
    with the seam's own message, rather than a raw StoreError escaping it."""
    cal_gt, cal_pred = tmp_path / "cal_gt", tmp_path / "cal_pred"
    hold_gt, hold_pred = tmp_path / "hold_gt", tmp_path / "hold_pred"
    calls = _one_positive_one_negative
    _write_split(cal_gt, cal_pred, prefix="cal", n_images=2, per_image_calls=calls)
    _write_split(hold_gt, hold_pred, prefix="hold", n_images=2, per_image_calls=calls, offset=1000)

    import os

    from tcip_mcp.pipelines.resolution import sidecar_key
    from tcip_store.binding import BACKEND_ENV, DEFAULT_BACKEND, FILE_BACKEND

    key = sidecar_key(cal_pred, "operating_point")
    ts.replace(key, {"id_map": {"bud": 0}}, expect=ts.Version.ABSENT)
    name = os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND
    if name == FILE_BACKEND:
        from tcip_store.store import _backend

        _backend().path_for(key).write_bytes(b"{not json")
    else:
        import sqlite3

        from tcip_store.sqlite_backend import database_path, encode_parts

        conn = sqlite3.connect(str(database_path(str(key.root))), isolation_level=None)
        try:
            conn.execute(
                "update records set value = ? where store = ? and parts = ?",
                (b"{not json", key.store, encode_parts(key.parts)),
            )
        finally:
            conn.close()

    res = calibrate_classifier_operating_point(
        trait_name="bud_opening", subject="bud", attribute="opening",
        calibration_gt_dir=str(cal_gt), calibration_pred_dir=str(cal_pred),
        holdout_gt_dir=str(hold_gt), holdout_pred_dir=str(hold_pred),
        output_dir=str(tmp_path / "out"), dataset_root=str(tmp_path), experiment_id=None,
    )

    assert "error" in res


def test_calibrate_classifier_operating_point_earns_a_record_a_later_bucket_binds_to(
    tmp_path: Path,
) -> None:
    """The whole legitimate route for this door: no checkpoint anywhere in its inputs, ground
    truth and predictions placed under one stated dataset root carrying the registry the
    vocabulary is resolved from. The claim it earns is a real record its own stamp verifies
    against, and a later date's bucket from the same producing run reads that stamp as validated."""
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_HELD_OUT,
        bind_classifier_validity,
        read_classifier_operating_point_sidecar,
        reconcile_classifier_validity,
        verify_stamp_binding,
    )

    root, out = _ds_root(tmp_path), tmp_path / "out"
    _write_bud_opening_registry(root)
    cal_gt, cal_pred = root / "annotations" / "2026-02-11", _bucket(tmp_path, "2026-02-11")
    hold_gt, hold_pred = root / "annotations" / "2026-03-09", _bucket(tmp_path, "2026-03-09")
    calls = _one_positive_one_negative
    _write_split(cal_gt, cal_pred, prefix="cal", n_images=20, per_image_calls=calls)
    _write_split(hold_gt, hold_pred, prefix="hold", n_images=20, per_image_calls=calls, offset=1000)

    res = calibrate_classifier_operating_point(
        trait_name="bud_opening", subject="bud", attribute="opening",
        calibration_gt_dir=str(cal_gt), calibration_pred_dir=str(cal_pred),
        holdout_gt_dir=str(hold_gt), holdout_pred_dir=str(hold_pred),
        output_dir=str(out), dataset_root=str(root), experiment_id=None,
    )

    assert res["passed"] is True, res
    stamp = read_classifier_operating_point_sidecar(out)
    assert stamp["checkpoint_sha256"] is None  # no bucket in the inputs carried one to copy
    binding = verify_stamp_binding(stamp, out, document="classifier_operating_point",
                                   trait="bud_opening")
    assert binding.ok and binding.claimed, binding.note
    assert binding.experiment_id == res["validated_by"]["experiment_id"]
    assert binding.producing_experiment_id is None  # the calibration hangs off its own experiment

    from tcip_mcp.experiments import find_validation

    row = find_validation(res["validated_by"]["experiment_id"], res["validated_by"]["record_digest"])
    # Sealed-disjointness liveness through the real classifier resolver: gate_evidence is read
    # correctly end to end, not silently lost to a stale key lookup that would leave these null.
    assert row["train_disjointness"] is not None
    assert row["selection_disjointness"] is not None

    later = _bucket(tmp_path, "2026-04-06")
    _write_preds(later, "P1_a", ["open"])
    _write_op_sidecar(later, dataset_root=root, validated=True, id_map=ID_MAP, experiment_id=None)
    state = reconcile_classifier_validity([str(out)])["validated"]

    assert state == VALIDATED_HELD_OUT
    assert bind_classifier_validity(state, [str(out)], [str(later)], trait="bud_opening") == (
        VALIDATED_HELD_OUT, "")


def test_calibrate_classifier_operating_point_refuses_a_dataset_root_its_gt_dirs_contradict(
    tmp_path: Path,
) -> None:
    """A GT directory the layout does place names its own dataset root, and recording the reference
    against a root it does not live under would leave a reader resolving it somewhere else."""
    ds, stated = tmp_path / "ds", tmp_path / "elsewhere"
    cal_gt, cal_pred = ds / "annotations" / "cal", tmp_path / "cal_pred"
    hold_gt, hold_pred = ds / "annotations" / "hold", tmp_path / "hold_pred"
    calls = _one_positive_one_negative
    _write_split(cal_gt, cal_pred, prefix="cal", n_images=20, per_image_calls=calls)
    _write_split(hold_gt, hold_pred, prefix="hold", n_images=20, per_image_calls=calls, offset=1000)

    res = calibrate_classifier_operating_point(
        trait_name="bud_opening", subject="bud", attribute="opening",
        calibration_gt_dir=str(cal_gt), calibration_pred_dir=str(cal_pred),
        holdout_gt_dir=str(hold_gt), holdout_pred_dir=str(hold_pred),
        output_dir=str(tmp_path / "out"), dataset_root=str(stated), experiment_id=None,
    )

    # Both roots, quoted the way the message quotes every path it names.
    assert f"{str(ds.resolve())!r}" in res["error"], res
    assert f"{str(stated.resolve())!r}" in res["error"], res
    assert not (tmp_path / "out").exists()  # a refused calibration stamps nothing


def test_calibrate_classifier_operating_point_refuses_genuinely_shared_content_holdout(
    tmp_path: Path,
) -> None:
    """A holdout whose GT content is cloned from calibration (same classification calls, same
    geometry) must refuse content_shared_with_calibration even under different image ids: the whole
    point of a content hash, not an image-id check."""
    root = _ds_root(tmp_path)
    _write_bud_opening_registry(root)
    cal_gt, cal_pred = root / "annotations" / "cal", tmp_path / "cal_pred"
    hold_gt, hold_pred = root / "annotations" / "hold", tmp_path / "hold_pred"
    calls = _alternating_calls
    _write_split(cal_gt, cal_pred, prefix="cal", n_images=20, per_image_calls=calls)
    # Different stems ("dup" vs "cal"), but identical per-image geometry+calls -> disjoint by
    # image_id, still content-shared.
    _write_split(hold_gt, hold_pred, prefix="dup", n_images=20, per_image_calls=calls)

    res = calibrate_classifier_operating_point(
        trait_name="bud_opening", subject="bud", attribute="opening",
        calibration_gt_dir=str(cal_gt), calibration_pred_dir=str(cal_pred),
        holdout_gt_dir=str(hold_gt), holdout_pred_dir=str(hold_pred),
        output_dir=str(tmp_path / "out"), dataset_root=str(root), experiment_id=None,
    )

    assert res["passed"] is False
    assert "content_shared_with_calibration" in res["failures"]


def test_calibrate_classifier_operating_point_partial_flip_fails_compensating_error_floor(
    tmp_path: Path,
) -> None:
    """A classifier that flips a substantial, symmetric fraction of calls (net count-bias ~0)
    must still fail: a bare kappa>0 floor alone would let a 40%-wrong classifier through, since
    symmetric flips cancel out in the net bias."""
    root = _ds_root(tmp_path)
    _write_bud_opening_registry(root)
    cal_gt, cal_pred = root / "annotations" / "cal", tmp_path / "cal_pred"
    hold_gt, hold_pred = root / "annotations" / "hold", tmp_path / "hold_pred"
    good = _one_positive_one_negative
    _write_split(cal_gt, cal_pred, prefix="cal", n_images=30, per_image_calls=good)

    def flipped(i):
        # 40% of images get a fully-flipped pair (both calls wrong) -- symmetric, net bias 0.
        if i % 5 < 2:
            return [(True, False), (False, True)]
        return [(True, True), (False, False)]

    _write_split(hold_gt, hold_pred, prefix="hold", n_images=50, per_image_calls=flipped, offset=1000)

    res = calibrate_classifier_operating_point(
        trait_name="bud_opening", subject="bud", attribute="opening",
        calibration_gt_dir=str(cal_gt), calibration_pred_dir=str(cal_pred),
        holdout_gt_dir=str(hold_gt), holdout_pred_dir=str(hold_pred),
        output_dir=str(tmp_path / "out"), dataset_root=str(root), experiment_id=None,
    )

    assert res["passed"] is False
    assert "compensating_error_floor_failed" in res["failures"], res["failures"]
    from tcip_mcp.pipelines.resolution import read_classifier_operating_point_sidecar

    sidecar = read_classifier_operating_point_sidecar(tmp_path / "out")
    assert "schema_version" not in sidecar
    gate_evidence = sidecar["gate_evidence"]
    assert gate_evidence["kappa"] is not None
    assert gate_evidence["kappa"] <= gate_evidence["kappa_floor"]
    assert gate_evidence["kappa_floor_source"] == "default"  # bud_opening sets none


def test_resolve_classifier_operating_point_refuses_single_image_holdout() -> None:
    """A single-image holdout has no images to vary the count-bias
    across, so its std is trivially 0 -- the SE penalty the equivalence test relies on vanishes.
    Must refuse (insufficient_holdout_images), the same minimum the detection path requires,
    rather than pass at exactly the tolerance with zero uncertainty discount."""
    from tcip_mcp.pipelines.operating_point import resolve_classifier_operating_point

    cal = [
        {"image_id": "c0", "is_true_positive": True, "is_pred_positive": True, "bbox": [0.0, 0.0, 10.0, 10.0]},
        {"image_id": "c1", "is_true_positive": False, "is_pred_positive": False, "bbox": [20.0, 0.0, 30.0, 10.0]},
    ]
    # 20 instances, all on one image -- a real net bias of +1, which the equivalence test's SE=0
    # (n_images=1) would let through regardless of tolerance.
    hold = [
        {"image_id": "h0", "is_true_positive": i < 10, "is_pred_positive": i < 11,
         "bbox": [float(100 + i), 0.0, float(110 + i), 10.0]}
        for i in range(20)
    ]
    res = resolve_classifier_operating_point(
        "bud_opening", calibration_items=cal, holdout_items=hold, experiment_id=None)

    assert res["passed"] is False
    assert "insufficient_holdout_images" in res["failures"], res["failures"]


def test_resolve_classifier_operating_point_bias_is_scoped_to_present_images() -> None:
    """count_bias/count_bias_std must be measured over the same population typical_positive_count
    is already scoped to (images carrying a true or predicted positive), the same present-scoped fix
    the pooled detector gate already has, mirroring _count_stats_at_conf's own `if gt or dt`. Before
    the fix, an all-negative image (no true positive, no predicted positive -- a confirmed-closed
    bud the classifier correctly called negative) contributed a certain zero to the bias mean/std
    while never counting toward typical_positive_count, diluting a real systematic miscall by
    n_bias_images/n_present exactly as the detector path's own dilution did."""
    from tcip_mcp.pipelines.operating_point import resolve_classifier_operating_point

    cal = [
        {"image_id": "c0", "is_true_positive": True, "is_pred_positive": True, "bbox": [0.0, 0.0, 10.0, 10.0]},
        {"image_id": "c1", "is_true_positive": False, "is_pred_positive": False, "bbox": [20.0, 0.0, 30.0, 10.0]},
    ]
    hold = []
    # 10 informative images: 100 real positives each, systematically over-called by +2 (102
    # predicted positive) -- a real 2% relative over-count.
    for i in range(10):
        for k in range(100):
            hold.append({"image_id": f"h{i}", "is_true_positive": True, "is_pred_positive": True,
                        "bbox": [float(k), 0.0, float(k + 1), 10.0]})
        for k in range(2):
            hold.append({"image_id": f"h{i}", "is_true_positive": False, "is_pred_positive": True,
                        "bbox": [float(9000 + k), 0.0, float(9001 + k), 10.0]})
    # 40 uninformative images: every instance confirmed negative and correctly called negative.
    # No true positive, no predicted positive anywhere on these -- exactly the population
    # typical_positive_count already excludes.
    for i in range(40):
        hold.append({"image_id": f"empty{i}", "is_true_positive": False, "is_pred_positive": False,
                    "bbox": [-100.0 - i, 0.0, -90.0 - i, 10.0]})

    res = resolve_classifier_operating_point(
        "bud_opening", calibration_items=cal, holdout_items=hold, experiment_id=None)

    assert res["gate_evidence"]["typical_positive_count"] == pytest.approx(100.0)
    assert res["gate_evidence"]["count_bias_n_images"] == 10  # scoped to the 10 informative images only
    assert res["gate_evidence"]["count_bias"] == pytest.approx(2.0)  # not diluted toward 0.4 by the 40 empties
    assert res["passed"] is False
    assert "count_bias_exceeds_tolerance" in res["failures"], res["failures"]


def _classifier_items(prefix, n_images, pos_per_image, *, miscall_images=(), image_offset=0):
    """``pos_per_image`` correctly-called positives per image, plus one token negative per image (so
    kappa stays defined). On ``miscall_images`` (indices), one extra false-positive-called instance
    is added -- the same absolute miscall, regardless of density. Every image's whole geometry is
    offset by its own index (``image_offset`` shifts a sibling split's sequence further still), so
    two images never share content by construction unless a test deliberately reuses one.
    """
    items = []
    for i in range(n_images):
        base = (image_offset + i) * 1000.0
        for k in range(pos_per_image):
            items.append({"image_id": f"{prefix}{i}", "is_true_positive": True, "is_pred_positive": True,
                         "bbox": [base + k * 15.0, 0.0, base + k * 15.0 + 10.0, 10.0]})
        if i in miscall_images:
            items.append({"image_id": f"{prefix}{i}", "is_true_positive": False, "is_pred_positive": True,
                         "bbox": [base + 9000.0, 0.0, base + 9010.0, 10.0]})
        items.append({"image_id": f"{prefix}{i}", "is_true_positive": False, "is_pred_positive": False,
                     "bbox": [base - 100.0, 0.0, base - 90.0, 10.0]})
    return items


def test_resolve_classifier_operating_point_relative_tolerance_refuses_a_sparse_class_the_dense_admits():
    """The classifier path's count-bias tolerance is relative too (same field, same
    `_bias_equivalence_ok`); the one passing classifier fixture in this file has bias exactly 0.0,
    true at any tolerance, so this pins both directions with the identical absolute miscall (one extra
    false-positive-called instance, on one holdout image out of 20): sparse (1 real positive/image)
    refuses; dense (150 real positives/image) admits the same miscall as a small relative fraction.
    """
    from tcip_mcp.pipelines.operating_point import resolve_classifier_operating_point

    sparse = resolve_classifier_operating_point(
        "bud_opening", calibration_items=_classifier_items("c", 20, 1),
        holdout_items=_classifier_items("h", 20, 1, miscall_images=[0], image_offset=20),
        experiment_id=None)
    assert sparse["gate_evidence"]["typical_positive_count"] == pytest.approx(1.0)
    assert sparse["passed"] is False
    assert "count_bias_exceeds_tolerance" in sparse["failures"]

    dense = resolve_classifier_operating_point(
        "bud_opening", calibration_items=_classifier_items("c", 20, 150),
        holdout_items=_classifier_items("h", 20, 150, miscall_images=[0], image_offset=20),
        experiment_id=None)
    assert dense["gate_evidence"]["typical_positive_count"] == pytest.approx(150.0)
    # Same count_bias/count_bias_std as the sparse case (the miscall pattern is identical) -- only
    # the derived tolerance differs, proving density is what changed the outcome.
    assert dense["gate_evidence"]["count_bias"] == pytest.approx(sparse["gate_evidence"]["count_bias"])
    assert dense["gate_evidence"]["count_bias_std"] == pytest.approx(sparse["gate_evidence"]["count_bias_std"])
    assert dense["gate_evidence"]["count_bias_tolerance_absolute"] > sparse["gate_evidence"]["count_bias_tolerance_absolute"]
    assert "count_bias_exceeds_tolerance" not in dense["failures"]


def test_resolve_classifier_operating_point_honors_trait_authored_agreement_floor(
    tmp_path: Path, monkeypatch,
) -> None:
    """TraitSpec.classifier_agreement_floor, when a trait authors one,
    must be the floor actually applied -- not the platform's interim default."""
    from dataclasses import replace

    from tcip_mcp.pipelines import operating_point as op_mod
    from tests._trait_fixtures import BUD_OPENING

    strict_bud_opening = replace(BUD_OPENING, classifier_agreement_floor=0.9)
    monkeypatch.setattr(op_mod, "get_trait", lambda name: strict_bud_opening)

    # A holdout with kappa=0.8 -- clears the platform's interim default (0.41) but not the
    # trait's own stricter authored floor (0.9).
    def make_items(n, flips):
        items = []
        for i in range(n):
            is_tp = i < n // 2
            pred = (not is_tp) if i in flips else is_tp
            items.append({"image_id": f"i{i}", "is_true_positive": is_tp, "is_pred_positive": pred,
                         "bbox": [float(i), 0.0, float(i + 10), 10.0]})
        return items

    cal = make_items(20, flips=set())
    hold = make_items(100, flips=set(range(10)))  # 10% symmetric flip -> kappa=0.8
    res = op_mod.resolve_classifier_operating_point(
        "bud_opening", calibration_items=cal, holdout_items=hold, experiment_id=None)

    assert 0.75 < res["gate_evidence"]["kappa"] < 0.85, res["gate_evidence"]
    assert res["gate_evidence"]["kappa_floor"] == 0.9
    assert res["gate_evidence"]["kappa_floor_source"] == "trait"
    assert res["passed"] is False
    assert "compensating_error_floor_failed" in res["failures"]


def test_resolve_classifier_operating_point_count_bias_tolerance_frac_source(monkeypatch) -> None:
    """TraitSpec.count_bias_tolerance_frac mirrors classifier_agreement_floor's own provenance
    stamp: unauthored (BUD_OPENING's own state) resolves to the platform's interim default fraction and
    stamps that; a trait that authors its own value stamps ``"trait"`` instead."""
    from dataclasses import replace

    from tcip_mcp.pipelines import operating_point as op_mod
    from tests._trait_fixtures import BUD_OPENING

    def make_items(n, flips):
        items = []
        for i in range(n):
            is_tp = i < n // 2
            pred = (not is_tp) if i in flips else is_tp
            items.append({"image_id": f"i{i}", "is_true_positive": is_tp, "is_pred_positive": pred,
                         "bbox": [float(i), 0.0, float(i + 10), 10.0]})
        return items

    cal = make_items(20, flips=set())
    hold = make_items(100, flips=set())  # clean, zero bias, so this stamp is reachable regardless

    monkeypatch.setattr(op_mod, "get_trait", lambda name: BUD_OPENING)
    res_default = op_mod.resolve_classifier_operating_point(
        "bud_opening", calibration_items=cal, holdout_items=hold, experiment_id=None)
    assert res_default["gate_evidence"]["count_bias_tolerance_frac"] == pytest.approx(0.01)
    assert res_default["gate_evidence"]["count_bias_tolerance_frac_source"] == "default"

    authored_bud_opening = replace(BUD_OPENING, count_bias_tolerance_frac=0.2)
    monkeypatch.setattr(op_mod, "get_trait", lambda name: authored_bud_opening)
    res_trait = op_mod.resolve_classifier_operating_point(
        "bud_opening", calibration_items=cal, holdout_items=hold, experiment_id=None)
    assert res_trait["gate_evidence"]["count_bias_tolerance_frac"] == pytest.approx(0.2)
    assert res_trait["gate_evidence"]["count_bias_tolerance_frac_source"] == "trait"


# --------------------------------------------------------------------------
# resolve_ordinal_operating_point / resolve_regression_operating_point: the ordinal/regression
# calibration gates, mirroring resolve_classifier_operating_point's own test coverage shape.
# --------------------------------------------------------------------------

def _ordinal_items(prefix, true_ranks, pred_ranks, offset=0):
    return [{"image_id": f"{prefix}{i + offset}", "true_rank": t, "predicted_rank": p}
           for i, (t, p) in enumerate(zip(true_ranks, pred_ranks))]


def _regression_items(prefix, true_values, pred_values, offset=0):
    return [{"image_id": f"{prefix}{i + offset}", "true_value": t, "predicted_value": p}
           for i, (t, p) in enumerate(zip(true_values, pred_values))]


def test_resolve_ordinal_operating_point_passes_on_clean_disjoint_split() -> None:
    from tcip_mcp.pipelines.operating_point import resolve_ordinal_operating_point

    ranks = [0, 1, 2, 1, 0, 2, 1, 0, 2, 1] * 2  # 20 items, every rank represented repeatedly
    cal = _ordinal_items("c", ranks, ranks)
    hold = _ordinal_items("h", ranks, ranks)  # perfect agreement -> kappa=1.0

    res = resolve_ordinal_operating_point(
        "bud_opening", criterion="quadratic_weighted_kappa", calibration_items=cal, holdout_items=hold,
        experiment_id=None)

    assert res["passed"] is True, res
    assert res["failures"] == []
    assert res["gate_evidence"]["criterion"] == "quadratic_weighted_kappa"
    assert res["gate_evidence"]["score"] == pytest.approx(1.0)
    assert res["gate_evidence"]["floor_source"] == "default"


def test_resolve_ordinal_operating_point_fails_closed_on_non_disjoint_split() -> None:
    from tcip_mcp.pipelines.operating_point import resolve_ordinal_operating_point

    ranks = [0, 1, 2, 1, 0, 2, 1, 0, 2, 1] * 2
    cal = _ordinal_items("shared", ranks, ranks)
    hold = _ordinal_items("shared", ranks, ranks)  # identical image_ids -> not disjoint

    res = resolve_ordinal_operating_point(
        "bud_opening", criterion="quadratic_weighted_kappa", calibration_items=cal, holdout_items=hold,
        experiment_id=None)

    assert res["passed"] is False
    assert "not_disjoint" in res["failures"]
    assert res["validated_against"] == "false"


def test_resolve_ordinal_operating_point_fails_closed_at_or_below_the_floor() -> None:
    """A holdout with mostly-adjacent-rank disagreement scores well below the interim default
    kappa floor (0.41) and must refuse, not merely score low."""
    from tcip_mcp.pipelines.operating_point import resolve_ordinal_operating_point

    true_ranks = [0, 1, 2] * 8
    pred_ranks = [2, 0, 1] * 8  # every item off by a full rank, cyclically -> poor agreement
    cal = _ordinal_items("c", true_ranks, true_ranks)
    hold = _ordinal_items("h", true_ranks, pred_ranks)

    res = resolve_ordinal_operating_point(
        "bud_opening", criterion="quadratic_weighted_kappa", calibration_items=cal, holdout_items=hold,
        experiment_id=None)

    assert res["gate_evidence"]["score"] is not None
    assert res["gate_evidence"]["score"] <= res["gate_evidence"]["floor"]
    assert res["passed"] is False
    assert "compensating_error_floor_failed" in res["failures"]


def test_resolve_ordinal_operating_point_fails_closed_on_missing_items() -> None:
    from tcip_mcp.pipelines.operating_point import resolve_ordinal_operating_point

    res = resolve_ordinal_operating_point(
        "bud_opening", criterion="quadratic_weighted_kappa", calibration_items=None, holdout_items=None,
        experiment_id=None)

    assert res["passed"] is False
    assert res["failures"] == ["no_calibration_or_holdout"]
    assert res["validated_against"] == "false"


def test_resolve_ordinal_operating_point_unknown_criterion_raises() -> None:
    from tcip_mcp.pipelines.operating_point import resolve_ordinal_operating_point

    with pytest.raises(ValueError, match="not a registered ordinal criterion"):
        resolve_ordinal_operating_point(
            "bud_opening", criterion="not_a_real_criterion",
            calibration_items=[{"image_id": "c0", "true_rank": 0, "predicted_rank": 0}],
            holdout_items=[{"image_id": "h0", "true_rank": 0, "predicted_rank": 0}],
            experiment_id=None)


def test_resolve_regression_operating_point_passes_on_clean_disjoint_split() -> None:
    from tcip_mcp.pipelines.operating_point import resolve_regression_operating_point

    values = [float(i) for i in range(20)]
    cal = _regression_items("c", values, values)
    hold = _regression_items("h", values, values)  # perfect fit -> r_squared=1.0

    res = resolve_regression_operating_point(
        "bud_opening", criterion="r_squared", calibration_items=cal, holdout_items=hold,
        experiment_id=None)

    assert res["passed"] is True, res
    assert res["failures"] == []
    assert res["gate_evidence"]["criterion"] == "r_squared"
    assert res["gate_evidence"]["score"] == pytest.approx(1.0)
    assert res["gate_evidence"]["floor_source"] == "default"


def test_resolve_regression_operating_point_fails_closed_on_non_disjoint_split() -> None:
    from tcip_mcp.pipelines.operating_point import resolve_regression_operating_point

    values = [float(i) for i in range(20)]
    cal = _regression_items("shared", values, values)
    hold = _regression_items("shared", values, values)

    res = resolve_regression_operating_point(
        "bud_opening", criterion="r_squared", calibration_items=cal, holdout_items=hold,
        experiment_id=None)

    assert res["passed"] is False
    assert "not_disjoint" in res["failures"]
    assert res["validated_against"] == "false"


def test_resolve_regression_operating_point_fails_closed_at_or_below_the_floor() -> None:
    """A holdout whose predictions carry no real relationship to the true values scores well below
    the interim default skill floor (0.5) and must refuse."""
    from tcip_mcp.pipelines.operating_point import resolve_regression_operating_point

    true_values = [float(i) for i in range(20)]
    cal = _regression_items("c", true_values, true_values)
    hold = _regression_items("h", true_values, [10.0] * 20)  # constant prediction, no skill

    res = resolve_regression_operating_point(
        "bud_opening", criterion="r_squared", calibration_items=cal, holdout_items=hold,
        experiment_id=None)

    assert res["gate_evidence"]["score"] is not None
    assert res["gate_evidence"]["score"] <= res["gate_evidence"]["floor"]
    assert res["passed"] is False
    assert "compensating_error_floor_failed" in res["failures"]


def test_resolve_regression_operating_point_fails_closed_on_missing_items() -> None:
    from tcip_mcp.pipelines.operating_point import resolve_regression_operating_point

    res = resolve_regression_operating_point(
        "bud_opening", criterion="r_squared", calibration_items=[], holdout_items=[],
        experiment_id=None)

    assert res["passed"] is False
    assert res["failures"] == ["no_calibration_or_holdout"]
    assert res["validated_against"] == "false"


def test_resolve_regression_operating_point_unknown_criterion_raises() -> None:
    from tcip_mcp.pipelines.operating_point import resolve_regression_operating_point

    with pytest.raises(ValueError, match="not a registered regression criterion"):
        resolve_regression_operating_point(
            "bud_opening", criterion="not_a_real_criterion",
            calibration_items=[{"image_id": "c0", "true_value": 1.0, "predicted_value": 1.0}],
            holdout_items=[{"image_id": "h0", "true_value": 1.0, "predicted_value": 1.0}],
            experiment_id=None)


def test_resolve_regression_operating_point_ccc_criterion_is_selectable() -> None:
    """The criterion toolkit is genuinely dispatched, not hardcoded to r_squared: a caller who
    states concordance_correlation_coefficient gets that statistic recorded, not silently ignored."""
    from tcip_mcp.pipelines.operating_point import resolve_regression_operating_point

    values = [float(i) for i in range(20)]
    cal = _regression_items("c", values, values)
    hold = _regression_items("h", values, values)  # perfect fit -> CCC=1.0 too

    res = resolve_regression_operating_point(
        "bud_opening", criterion="concordance_correlation_coefficient", calibration_items=cal,
        holdout_items=hold, experiment_id=None)

    assert res["gate_evidence"]["criterion"] == "concordance_correlation_coefficient"
    assert res["gate_evidence"]["score"] == pytest.approx(1.0)
    assert res["passed"] is True


def test_calibrate_classifier_operating_point_foreign_checkpoint_stamp_still_reachable(
    tmp_path: Path,
) -> None:
    """experiment_id=None (a foreign/unregistered checkpoint) skips train-disjointness rather than
    failing closed -- the classifier-validity stamp is still reachable for an otherwise clean
    reference."""
    root = _ds_root(tmp_path)
    _write_bud_opening_registry(root)
    cal_gt, cal_pred = root / "annotations" / "cal", tmp_path / "cal_pred"
    hold_gt, hold_pred = root / "annotations" / "hold", tmp_path / "hold_pred"
    calls = _one_positive_one_negative
    _write_split(cal_gt, cal_pred, prefix="cal", n_images=20, per_image_calls=calls)
    _write_split(hold_gt, hold_pred, prefix="hold", n_images=20, per_image_calls=calls, offset=1000)

    res = calibrate_classifier_operating_point(
        trait_name="bud_opening", subject="bud", attribute="opening",
        calibration_gt_dir=str(cal_gt), calibration_pred_dir=str(cal_pred),
        holdout_gt_dir=str(hold_gt), holdout_pred_dir=str(hold_pred),
        output_dir=str(tmp_path / "out"), dataset_root=str(root), experiment_id=None,
    )

    assert res["passed"] is True, res
    assert "train_disjointness_unresolvable" not in res["failures"]
    assert "train_disjointness_leaked" not in res["failures"]


def test_calibrate_classifier_operating_point_unassessed_gt_never_fabricates_a_negative(
    tmp_path: Path,
) -> None:
    """A GT instance never assessed for `attribute` (no opening key at all)
    must be excluded from the reference, never coerced into "not positive". A perfect classifier
    scored against a reference where every image also carries unassessed instances must still pass
    cleanly -- the unassessed instances contribute no fabricated disagreement."""
    root = _ds_root(tmp_path)
    _write_bud_opening_registry(root)
    cal_gt, cal_pred = root / "annotations" / "cal", tmp_path / "cal_pred"
    hold_gt, hold_pred = root / "annotations" / "hold", tmp_path / "hold_pred"

    def perfect_plus_unassessed(_i: int) -> list[tuple[bool | None, bool]]:
        # 2 correctly-classified instances (1 pos, 1 neg) + 2 never-assessed instances. The
        # classifier's own call on the unassessed instances is irrelevant -- they must not enter
        # the reference at all, so their is_pred_positive value here is deliberately inconsistent
        # (would fabricate a disagreement if the bug were still present).
        return [(True, True), (False, False), (None, True), (None, False)]

    _write_split(cal_gt, cal_pred, prefix="cal", n_images=20, per_image_calls=perfect_plus_unassessed)
    _write_split(hold_gt, hold_pred, prefix="hold", n_images=20,
                per_image_calls=perfect_plus_unassessed, offset=1000)

    res = calibrate_classifier_operating_point(
        trait_name="bud_opening", subject="bud", attribute="opening",
        calibration_gt_dir=str(cal_gt), calibration_pred_dir=str(cal_pred),
        holdout_gt_dir=str(hold_gt), holdout_pred_dir=str(hold_pred),
        output_dir=str(tmp_path / "out"), dataset_root=str(root), experiment_id=None,
    )

    assert res["passed"] is True, res
    from tcip_mcp.pipelines.resolution import read_classifier_operating_point_sidecar

    sidecar = read_classifier_operating_point_sidecar(tmp_path / "out")
    gate_evidence = sidecar["gate_evidence"]
    assert gate_evidence["kappa"] == 1.0, gate_evidence  # a perfect classifier, not degraded by phantom errors
    assert gate_evidence["count_bias"] == 0.0, gate_evidence
    assert res["n_holdout_items"] == 20 * 2  # only the 2 assessed instances/image counted, not 4


def test_classification_items_derives_center_match_tolerance_across_the_whole_split(
    tmp_path: Path,
) -> None:
    """The center-match tolerance must be derived from the whole
    split's GT, not one image at a time. A small-object image's real pair, offset by more than
    that image's own (too-tight) per-image tolerance but less than the split-wide tolerance
    (pulled up by a large-object image elsewhere in the same split), must still match."""
    root = _ds_root(tmp_path)
    _write_bud_opening_registry(root)
    gt_dir, pred_dir = root / "annotations" / "date", tmp_path / "pred"
    gt_dir.mkdir(parents=True)
    pred_dir.mkdir()
    # Precondition guard: this test's math depends on
    # the recorded kind (BUD_OPENING's localization=CENTER_MATCH, seeded by seed_bud_trait_spec)
    # governing, not a live derivation; this is not redundant with derivation: this
    # split's average characteristic size (sqrt(200*200)=200 and sqrt(20*20)=20, mean 110px)
    # exceeds derive_localization_kind's ~45px crossover and would derive to iou_match if
    # unrecorded. resolve_match_criterion uses a recorded kind as-is (only warns on divergence,
    # never overrides it (see test_evaluation_metrics.py's dedicated coverage of that behavior),
    # so this assertion is what actually fails fast, with a clear message, if the BUD_OPENING fixture's
    # localization value ever changes and silently stops exercising the center-match code path
    # this test exists to cover.
    spec = get_trait("bud_opening")
    assert spec.localization == CENTER_MATCH and spec.localization_tolerance_frac == 0.5

    # "big": a 200x200 GT box -> char_size=200 -> per-image tolerance would be 100px; offset 40px
    # (comfortably under both the per-image AND the split-wide tolerance derived below).
    big_box = BBox(0.0, 0.0, 200.0, 200.0)
    json_io.write_annotations(gt_dir / "big.json", [
        Annotation(subject="bud", geometry=big_box, attributes={"opening": "open"}),
    ], 400, 400)
    json_io.write_annotations(pred_dir / "big.json", [
        Annotation(subject="bud", geometry=BBox(40.0, 0.0, 240.0, 200.0), score=0.9,
                   attributes={"opening": "open"}),
    ], 400, 400)

    # "small": a 20x20 GT box -> char_size=20 -> per-image tolerance would be only 10px; the
    # prediction is offset 15px, which a per-image derivation would drop as unmatched.
    small_box = BBox(1000.0, 0.0, 1020.0, 20.0)
    json_io.write_annotations(gt_dir / "small.json", [
        Annotation(subject="bud", geometry=small_box, attributes={"opening": "closed"}),
    ], 1100, 100)
    json_io.write_annotations(pred_dir / "small.json", [
        Annotation(subject="bud", geometry=BBox(1015.0, 0.0, 1035.0, 20.0), score=0.9,
                   attributes={"opening": "closed"}),
    ], 1100, 100)

    # Split-wide avg char_size = (200 + 20) / 2 = 110 -> tolerance = 0.5 * 110 = 55px, comfortably
    # above both the small image's 15px offset and the big image's 40px offset.
    items = _classification_items(str(gt_dir), str(pred_dir), trait_name="bud_opening", subject="bud",
                                  positive_value="open", attribute="opening")

    by_image = {it["image_id"]: it for it in items}
    assert "small" in by_image, (
        "the small-object image's pair was dropped -- tolerance was derived per-image, not "
        "across the whole split"
    )
    assert by_image["small"]["is_true_positive"] is False  # "closed"
    assert by_image["small"]["is_pred_positive"] is False
    assert "big" in by_image
    assert by_image["big"]["is_true_positive"] is True  # "open"


def test_classification_items_scopes_gt_to_the_run_subject(tmp_path: Path) -> None:
    """A labels dir isn't guaranteed to hold only one kind of
    annotation -- a dataset that also isolates an enabling subject (e.g. "bush", root CLAUDE.md's
    "a subject is not a trait") must not let that unrelated box enter the match pool. Here a "bush"
    annotation sits exactly on the prediction's center (distance 0) while the real "bud" GT is a
    few px off -- if subject weren't scoped, greedy center-match (closest first) would steal the
    match for "bush" and either drop the real bud pair or attribute it to the wrong box/attribute
    entirely."""
    root = _ds_root(tmp_path)
    _write_bud_opening_registry(root)
    gt_dir, pred_dir = root / "annotations" / "date", tmp_path / "pred"
    gt_dir.mkdir(parents=True)
    pred_dir.mkdir()

    # bud center is ~7px from the prediction, inside tolerance; bush center is an exact match, so
    # unscoped greedy center-match would steal it for bush ahead of the real bud pair.
    bud_box = BBox(105.0, 105.0, 145.0, 145.0)
    bush_box = BBox(114.0, 114.0, 126.0, 126.0)
    json_io.write_annotations(gt_dir / "a.json", [
        Annotation(subject="bud", geometry=bud_box, attributes={"opening": "open"}),
        Annotation(subject="bush", geometry=bush_box),
    ], 400, 400)
    json_io.write_annotations(pred_dir / "a.json", [
        Annotation(subject="bud", geometry=BBox(114.0, 114.0, 126.0, 126.0), score=0.9,
                   attributes={"opening": "open"}),
    ], 400, 400)

    items = _classification_items(str(gt_dir), str(pred_dir), trait_name="bud_opening", subject="bud",
                                  positive_value="open", attribute="opening")

    assert len(items) == 1
    assert items[0]["bbox"] == [105.0, 105.0, 145.0, 145.0]  # the bud box, not bush's
    assert items[0]["is_true_positive"] is True  # bud's own "open" attribute


def _write_pair(gt_dir: Path, pred_dir: Path, *, gt_value: str, pred_value: str = "open") -> Path:
    """One matched (GT, pred) instance pair, same box, for a refusal test that never reaches
    the matching machinery's own edge cases. Returns the GT document written."""
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_file = gt_dir / "a.json"
    json_io.write_annotations(gt_file, [
        Annotation(subject="bud", geometry=BBox(0.0, 0.0, 10.0, 10.0), attributes={"opening": gt_value}),
    ], 20, 20)
    json_io.write_annotations(pred_dir / "a.json", [
        Annotation(subject="bud", geometry=BBox(0.0, 0.0, 10.0, 10.0), score=0.9,
                   attributes={"opening": pred_value}),
    ], 20, 20)
    return gt_file


def test_classification_items_refuses_a_bare_split_with_no_registry(tmp_path: Path) -> None:
    """A bare prediction bucket (no stamp at all) is held to the registry its ground truth's own
    dataset root carries; with no such root, this refuses naming both directories and the
    remedy, never reading a vocabulary off the reference's own values."""
    gt_dir, pred_dir = tmp_path / "gt", tmp_path / "pred"
    _write_pair(gt_dir, pred_dir, gt_value="open")

    with pytest.raises(ValueError) as exc:
        _classification_items(str(gt_dir), str(pred_dir), trait_name="bud_opening", subject="bud",
                              positive_value="open", attribute="opening")

    message = str(exc.value)
    assert str(Path(pred_dir)) in message and str(Path(gt_dir)) in message
    assert "dataset root" in message


def test_classification_items_refuses_a_classified_bucket_with_no_map_and_no_registry(
    tmp_path: Path,
) -> None:
    """A classified stamp recording no usable id_map falls to the same registry requirement a
    bare bucket does; with no registry under the ground truth's own root, this refuses the same
    way, naming the classified scope's own absent map rather than a bare directory."""
    from tcip_mcp.pipelines.resolution import operating_point_stamp, write_sidecar

    gt_dir, pred_dir = tmp_path / "gt", tmp_path / "pred"
    _write_pair(gt_dir, pred_dir, gt_value="open")
    stamp = operating_point_stamp(
        {}, validated=False, validated_by=None, tile_size_validated=None, shippable_issues=[],
        id_map=None, subject="bud", attribute="opening", trait=None, dataset_hash=None,
        checkpoint=None, checkpoint_sha256=None, experiment_id=None, images_dir=None,
        raster_path=None, produced_at=None,
    )
    write_sidecar(pred_dir, stamp)

    with pytest.raises(ValueError) as exc:
        _classification_items(str(gt_dir), str(pred_dir), trait_name="bud_opening", subject="bud",
                              positive_value="open", attribute="opening")

    message = str(exc.value)
    assert str(Path(pred_dir)) in message and str(Path(gt_dir)) in message
    assert "classified scope with no usable id_map" in message


def test_classification_items_refuses_a_registry_not_declaring_the_positive_value(
    tmp_path: Path,
) -> None:
    """The declared values a classifier is calibrated against must include the trait's own
    positive value: nothing else checks for it, since require_classified_record only checks
    predictions against whatever the registry happens to declare."""
    from tcip_mcp import class_registry

    root = _ds_root(tmp_path)
    class_registry.write_registry(root / "classes.json", class_registry.ClassRegistry(subjects=(
        class_registry.Subject(name="bud", attributes=(
            class_registry.Attribute(name="opening", type="categorical",
                                     values=("closed", "other")),
        )),
    )))
    gt_dir, pred_dir = root / "annotations" / "date", tmp_path / "pred"
    # Both sides carry a value the registry does declare, so require_classified_record's own
    # membership check never fires: only the missing-positive-value check under test can raise.
    _write_pair(gt_dir, pred_dir, gt_value="closed", pred_value="closed")

    with pytest.raises(ValueError) as exc:
        _classification_items(str(gt_dir), str(pred_dir), trait_name="bud_opening", subject="bud",
                              positive_value="open", attribute="opening")

    message = str(exc.value)
    assert "'open'" in message
    assert "closed" in message and "other" in message


def test_classification_items_refuses_a_ground_truth_value_outside_the_registry(
    tmp_path: Path,
) -> None:
    """A ground-truth value the registry does not declare refuses naming the file and the value,
    the same fact on the reference side that require_classified_record already checks on the
    prediction side."""
    root = _ds_root(tmp_path)
    _write_bud_opening_registry(root)
    gt_dir, pred_dir = root / "annotations" / "date", tmp_path / "pred"
    gt_file = _write_pair(gt_dir, pred_dir, gt_value="budding")

    with pytest.raises(ValueError) as exc:
        _classification_items(str(gt_dir), str(pred_dir), trait_name="bud_opening", subject="bud",
                              positive_value="open", attribute="opening")

    message = str(exc.value)
    assert str(gt_file) in message
    assert "budding" in message


# --------------------------------------------------------------------------
# calibrate_scalar_operating_point: an end-to-end run against a real trained
# checkpoint, no pre-staged files (there is no staging mechanism for a CSV-sourced scalar trait,
# unlike calibrate_classifier_operating_point's paired GT/prediction dirs).
# --------------------------------------------------------------------------

def test_calibrate_scalar_operating_point_ordinal_e2e(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.training.generic_trainer import train
    from tcip_mcp.pipelines.training.collation import task_collate
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_mcp.tools.calibration_tools import calibrate_scalar_operating_point
    from tests.test_e2e_tasktypes import _model_source, _save_png, _train_config, _write_csv

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    images_dir = tmp_path / "images"
    rows = []
    for i in range(10):
        _save_png(images_dir / f"img{i}.png", bright=(i % 3 == 0))
        rows.append((f"img{i}", i % 3))
    csv_path = tmp_path / "ranks.csv"
    _write_csv(csv_path, rows, ("stem", "rank"))

    dataset = build_dataset("ordinal", images_dir=str(images_dir), csv_path=str(csv_path), num_ranks=3)
    loader = DataLoader(dataset, batch_size=5, collate_fn=task_collate("ordinal"))
    model_source = _model_source("build_bespoke_ordinal", num_ranks=3)
    # Seeded through the trainer's own config key so this run's init and shuffling repeat.
    run = create_run({**_train_config(model_source), "seed": 0}, str(tmp_path / "out"))
    run = train(run, loader, val_loader=None, task="ordinal")
    assert run.status == "completed", getattr(run, "error", run.status)

    from tcip_mcp.tools.model_tools import register_model

    ckpt_path = str(tmp_path / "out" / "model_best.pt")
    reg = register_model(name="ordinal-e2e", checkpoint_path=ckpt_path, config={},
                         project_path=str(tmp_path))
    assert "error" not in reg, reg

    result = calibrate_scalar_operating_point(
        trait_name="bud_opening", task="ordinal",
        checkpoint_path=str(tmp_path / "out" / "model_best.pt"),
        images_dir=str(images_dir), csv_path=str(csv_path),
        criterion="quadratic_weighted_kappa", output_dir=str(tmp_path / "calib"),
        dataset_root=str(tmp_path),
    )

    assert "error" not in result, result
    assert result["criterion"] == "quadratic_weighted_kappa"
    assert result["n_calibration_items"] + result["n_holdout_items"] == 10

    from tcip_mcp.pipelines.resolution import read_ordinal_operating_point_sidecar

    sidecar = read_ordinal_operating_point_sidecar(tmp_path / "calib")
    assert "schema_version" not in sidecar
    assert sidecar["trait"] == "bud_opening"
    assert sidecar["operating_point"]["ordinal"]["criterion"] == "quadratic_weighted_kappa"
    assert sidecar["operating_point"]["ordinal"]["validated_against"] == result["validated_against"]
    assert sidecar["validated"] == result["passed"]
    assert "gate_evidence" in sidecar and sidecar["gate_evidence"]["criterion"] == "quadratic_weighted_kappa"
    import hashlib

    assert sidecar["checkpoint_sha256"] == hashlib.sha256(
        (tmp_path / "out" / "model_best.pt").read_bytes()).hexdigest()


def test_calibrate_scalar_operating_point_regression_e2e(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.training.generic_trainer import train
    from tcip_mcp.pipelines.training.collation import task_collate
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_mcp.tools.calibration_tools import calibrate_scalar_operating_point
    from tests.test_e2e_tasktypes import _model_source, _save_png, _train_config, _write_csv

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    images_dir = tmp_path / "images"
    rows = []
    for i in range(10):
        _save_png(images_dir / f"img{i}.png", bright=(i % 2 == 0))
        rows.append((f"img{i}", float(i) / 10.0))
    csv_path = tmp_path / "values.csv"
    _write_csv(csv_path, rows, ("stem", "value"))

    dataset = build_dataset("regression", images_dir=str(images_dir), csv_path=str(csv_path))
    loader = DataLoader(dataset, batch_size=5, collate_fn=task_collate("regression"))
    model_source = _model_source("build_bespoke_regressor")
    # Seeded through the trainer's own config key so this run's init and shuffling repeat.
    run = create_run({**_train_config(model_source), "seed": 0}, str(tmp_path / "out"))
    run = train(run, loader, val_loader=None, task="regression")
    assert run.status == "completed", getattr(run, "error", run.status)

    from tcip_mcp.tools.model_tools import register_model

    reg = register_model(name="regression-e2e", checkpoint_path=str(tmp_path / "out" / "model_best.pt"),
                         config={}, project_path=str(tmp_path))
    assert "error" not in reg, reg

    result = calibrate_scalar_operating_point(
        trait_name="bud_opening", task="regression",
        checkpoint_path=str(tmp_path / "out" / "model_best.pt"),
        images_dir=str(images_dir), csv_path=str(csv_path),
        criterion="r_squared", output_dir=str(tmp_path / "calib"),
        dataset_root=str(tmp_path),
    )

    assert "error" not in result, result
    assert result["criterion"] == "r_squared"
    assert result["n_calibration_items"] + result["n_holdout_items"] == 10

    from tcip_mcp.pipelines.resolution import read_regression_operating_point_sidecar

    sidecar = read_regression_operating_point_sidecar(tmp_path / "calib")
    assert "schema_version" not in sidecar
    assert sidecar["trait"] == "bud_opening"
    assert sidecar["operating_point"]["regression"]["criterion"] == "r_squared"
    assert sidecar["operating_point"]["regression"]["validated_against"] == result["validated_against"]
    assert sidecar["validated"] == result["passed"]
    assert "gate_evidence" in sidecar and sidecar["gate_evidence"]["criterion"] == "r_squared"


def test_calibrate_scalar_operating_point_unknown_task_returns_error(tmp_path: Path) -> None:
    from tcip_mcp.tools.calibration_tools import calibrate_scalar_operating_point

    result = calibrate_scalar_operating_point(
        trait_name="bud_opening", task="classification", checkpoint_path="m.pt",
        images_dir=str(tmp_path), csv_path=str(tmp_path / "x.csv"), criterion="r_squared",
        output_dir=str(tmp_path / "calib"), dataset_root=str(tmp_path),
    )
    assert "error" in result


def _rank_csv(csv_path: Path, ranks: dict[str, int]) -> None:
    csv_path.write_text("stem,rank\n" + "".join(f"{s},{r}\n" for s, r in ranks.items()),
                        encoding="utf-8")


def test_calibrate_scalar_operating_point_admits_a_loose_images_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CSV over an images directory the dataset layout cannot place is legitimate bespoke work:
    the stated root is what the record's locations are written against, and a root nothing else
    contradicts refuses nothing. The predictor is a stand-in returning each image's recorded rank,
    so what is exercised is the door's own earning path rather than a fresh model's accuracy.

    The checkpoint is a real file, since the stamp and the record both name it by content hash."""
    pytest.importorskip("torch")
    from tcip_mcp.dataset_layout import dataset_root_of
    from tcip_mcp.experiments import find_validation
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_HELD_OUT,
        read_ordinal_operating_point_sidecar,
        reconcile_ordinal_validity,
        verify_stamp_binding,
    )
    from tcip_mcp.tools.calibration_tools import calibrate_scalar_operating_point

    frames, out = tmp_path / "frames", tmp_path / "calib"
    frames.mkdir()
    ranks = {}
    for i in range(20):
        Image.new("RGB", (4, 4)).save(frames / f"img{i}.png")
        ranks[f"img{i}"] = i % 3
    csv_path = tmp_path / "ranks.csv"
    _rank_csv(csv_path, ranks)
    assert dataset_root_of(frames) is None
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "model_best.pt"
    torch.save({"model_state_dict": {}, "kind": "tcip_module"}, checkpoint)

    from tcip_mcp.tools.model_tools import register_model

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    reg = register_model(name="loose-dir", checkpoint_path=str(checkpoint), config={},
                         project_path=str(tmp_path))
    assert "error" not in reg, reg

    class _RecordedRanks:
        def predict_batch(self, sources):
            return [{"head0_ranks": [float(ranks[Path(s).stem])]} for s in sources]

    monkeypatch.setattr("tcip_mcp.pipelines.inference.predictor.build_predictor",
                        lambda *a, **kw: _RecordedRanks())

    res = calibrate_scalar_operating_point(
        trait_name="bud_opening", task="ordinal", checkpoint_path=str(checkpoint),
        images_dir=str(frames), csv_path=str(csv_path),
        criterion="quadratic_weighted_kappa", output_dir=str(out),
        dataset_root=str(tmp_path), group_by="stem",
    )

    assert res["passed"] is True, res
    stamp = read_ordinal_operating_point_sidecar(out)
    binding = verify_stamp_binding(stamp, out, document="ordinal_operating_point", trait="bud_opening")
    assert binding.ok and binding.claimed, binding.note
    assert binding.experiment_id == res["validated_by"]["experiment_id"]
    assert reconcile_ordinal_validity(
        [str(out)], trait="bud_opening")["validated"] == VALIDATED_HELD_OUT

    # The door ran this checkpoint, so stamp and record both name it and the equality check holds.
    import hashlib

    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert stamp["checkpoint_sha256"] == sha
    row = find_validation(res["validated_by"]["experiment_id"], res["validated_by"]["record_digest"])
    assert row["checkpoint_sha256"] == sha
    assert binding.checkpoint_sha256 == sha
    # Sealed-disjointness liveness: the resolver's gate_evidence key is read correctly end to end,
    # not silently lost to a stale key lookup that would leave these null.
    assert row["train_disjointness"] is not None
    assert row["selection_disjointness"] is not None


def test_calibrate_scalar_operating_point_refuses_a_dataset_root_its_images_contradict(
    tmp_path: Path,
) -> None:
    """The images directory places itself under a dataset root, so a stated root that disagrees
    would record the reference against a dataset it does not live under."""
    from tcip_mcp.tools.calibration_tools import calibrate_scalar_operating_point

    ds, stated = tmp_path / "ds", tmp_path / "elsewhere"
    (ds / "images").mkdir(parents=True)
    csv_path = tmp_path / "ranks.csv"
    _rank_csv(csv_path, {f"img{i}": i % 3 for i in range(4)})

    res = calibrate_scalar_operating_point(
        trait_name="bud_opening", task="ordinal", checkpoint_path=str(tmp_path / "model_best.pt"),
        images_dir=str(ds / "images"), csv_path=str(csv_path),
        criterion="quadratic_weighted_kappa", output_dir=str(tmp_path / "calib"),
        dataset_root=str(stated),
    )

    # Both roots, quoted the way the message quotes every path it names.
    assert f"{str(ds.resolve())!r}" in res["error"], res
    assert f"{str(stated.resolve())!r}" in res["error"], res
    assert not (tmp_path / "calib").exists()  # a refused calibration stamps nothing


# The delivered producer tail: what a phenology CSV may name, and what the delivery records.

def _delivery_setup(tmp_path: Path, *, experiment_id: str | None,
                    checkpoint_sha256: str | None) -> tuple[str, Path, Path]:
    """Two classified, count-validated buckets plus a mapping, ready for a phenology delivery."""
    root = _ds_root(tmp_path)
    d1, d2 = _bucket(tmp_path, "2026-02-11"), _bucket(tmp_path, "2026-03-09")
    _write_preds(d1, "P1_a", ["closed"])
    _write_preds(d2, "P1_b", ["open"])
    for d in (d1, d2):
        _write_op_sidecar(d, dataset_root=root, validated=True, id_map=ID_MAP,
                          experiment_id=experiment_id, checkpoint_sha256=checkpoint_sha256)
    _write_classifier_sidecar(d1, dataset_root=root, validated=True, trait="bud_opening",
                              experiment_id=experiment_id)
    mapping_name = "valley"
    _write_mapping(tmp_path, mapping_name, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    return mapping_name, d1, d2


def _named_records(*pred_dirs: Path) -> str:
    """The records these buckets' stamps point at, joined the way a delivered cell joins them."""
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    pointers = set()
    for d in pred_dirs:
        by = read_operating_point_sidecar(d)["validated_by"]
        pointers.add(f"{by['experiment_id']}:{by['record_digest']}")
    return "; ".join(sorted(pointers))


def _delivered_rows(out_csv: Path) -> list[dict]:
    import csv as _csv

    with out_csv.open(newline="", encoding="utf-8") as f:
        return list(_csv.DictReader(f))


def test_deliver_phenology_milestones_names_the_record_and_producer_a_bound_bucket_earned(tmp_path: Path) -> None:
    """A delivery every bucket of which is bound names the records that bound it and the producing
    run those records were earned under, rather than leaving a reader to trust the stamps."""
    from tests._binding_fixtures import record_producing_run

    sha = record_producing_run(tmp_path, "exp-producer")
    mapping_name, d1, d2 = _delivery_setup(
        tmp_path, experiment_id="exp-producer", checkpoint_sha256=sha)
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_phenology_milestones(
        trait="bud_opening", mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv), classifier_pred_dirs=[str(d1)],
    )

    assert "error" not in res, res
    assert "validation_record" in res["columns"]
    rows = _delivered_rows(out_csv)
    assert rows
    for row in rows:
        assert row["validation_record"] == _named_records(d1, d2)
        assert row["validation_record"] != ""
        assert row["producer_model_sha256"] == sha
        assert row["producing_experiment_id"] == "exp-producer"


def test_writer_delivers_a_forged_stamp_acknowledged_with_no_producer_names(
    tmp_path: Path,
) -> None:
    """A stamp claiming validation no record answers for floors the count, and the acknowledged
    CSV says the producer is unknown instead of repeating the names the stamp asserted for itself.

    Delivered through ``_deliver_via_writer`` (the MCP tool takes no acknowledgement).
    """
    from tcip_mcp.pipelines.resolution import Acknowledgement, update_sidecar

    mapping_name, d1, d2 = _delivery_setup(
        tmp_path, experiment_id="exp-producer", checkpoint_sha256="a" * 64)

    def _forge(stamp: dict) -> dict:
        stamp["validated_by"] = {"experiment_id": "exp_that_never_ran", "record_digest": "0" * 16}
        return stamp

    for d in (d1, d2):
        update_sidecar(d, _forge, "operating_point")

    out_csv = tmp_path / "out" / "bud_phenology.csv"
    cells = _deliver_via_writer(
        trait="bud_opening",
        mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=out_csv,
        classifier_pred_dirs=[str(d1)],
        acknowledgement=Acknowledgement(acknowledged_by="user:tester", reason="test acknowledgement"),
    )

    assert cells["operating_point_validated"] == "false"
    rows = _delivered_rows(out_csv)
    assert rows
    for row in rows:
        assert row["producer_model_sha256"] == ""
        assert row["producing_experiment_id"] == ""
        assert row["validation_record"] == ""


def test_deliver_phenology_milestones_records_what_verification_found_in_the_datasets_own_log(
    tmp_path: Path,
) -> None:
    """The delivery's audited arguments say what was asked for; this says which buckets stood behind
    the numbers and which records answered for them, in the log that travels with the data."""
    from tests._binding_fixtures import record_producing_run

    sha = record_producing_run(tmp_path, "exp-producer")
    mapping_name, d1, d2 = _delivery_setup(
        tmp_path, experiment_id="exp-producer", checkpoint_sha256=sha)
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_phenology_milestones(
        trait="bud_opening", mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv), classifier_pred_dirs=[str(d1)],
    )

    assert "error" not in res, res
    from tcip_mcp.audit import audit_log_key

    page = ts.read_log(audit_log_key(_ds_root(tmp_path)))
    assert page.records, "the delivery wrote nothing to the log of the dataset its buckets sit in"
    events = [e for e in page.records if e["tool"] == "deliver_phenology_milestones" and "verified_buckets" in e]
    assert len(events) == 1, page.records
    verified = events[0]["verified_buckets"]
    assert set(verified) == {str(d1), str(d2)}
    assert all(v["verified"] for v in verified.values())
    assert events[0]["record_digests"], events[0]
