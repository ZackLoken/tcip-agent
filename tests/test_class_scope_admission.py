"""``ClassScope``: an attribute stated with no subject refuses at construction, and a classified
scope an admission produces is the one a selection and a run's own data section carry back.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from tcip_mcp.pipelines.data.selection import ClassScope, read_selection
from tcip_mcp.tools.data_tools import draw_splits

from tests.test_selection_binding import DATES, SUBJECT, _attribute_scoped_dataset


def test_an_attribute_with_no_subject_refuses_where_the_scope_is_built() -> None:
    with pytest.raises(ValueError, match="attribute 'condition' is stated with no subject"):
        ClassScope(attribute="condition")
    with pytest.raises(ValueError, match="stated with no subject"):
        ClassScope.of({"scope": {"attribute": "condition", "id_map": {"healthy": 0}}})


def test_a_classified_scope_the_admission_produced_round_trips_through_the_selection_and_the_run(
    tmp_path: Path,
) -> None:
    """The admitting case, through the producers: the draw records the scope its admission read
    targets under, the selection reads it back whole, and a run bound to it records that same
    scope on its own data section."""
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _attribute_scoped_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = draw_splits(tmp_path, str(root), output_path=str(out), subject=SUBJECT,
                         attribute="condition", seed=1, train_ratio=0.5, val_ratio=0.25,
                         calibration_ratio=0.125, holdout_ratio=0.125)
    assert "error" not in result, result

    drawn = read_selection(out, project=tmp_path)
    assert drawn.scope == ClassScope(SUBJECT, "condition", {"healthy": 0, "damaged": 1})
    assert result["scope"] == asdict(drawn.scope)

    data_cfg: dict = {"split": {"selection_dir": str(out)}}
    auto_train_val(tmp_path, "detection", data_cfg, None)
    assert ClassScope.of(data_cfg) == drawn.scope


REVERSED = {"healthy": 1, "damaged": 0}
"""The reverse of the map the fixture's registry assigns ``condition``."""


def _dirs(root: Path) -> tuple[str, str]:
    return str(root / "images" / DATES[0]), str(root / "annotations" / DATES[0])


def test_a_run_stating_its_map_is_admitted_under_that_map_not_the_registrys(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.data.label_queries import admit_run

    images_dir, labels_dir = _dirs(_attribute_scoped_dataset(tmp_path / "ds"))
    admitted = admit_run({"images_dir": images_dir, "labels_dir": labels_dir,
                          "scope": {"subject": SUBJECT, "attribute": "condition",
                                    "id_map": REVERSED}})

    assert admitted.scope.id_map == REVERSED


def test_a_recorded_document_scope_with_no_map_refuses_and_a_fresh_statement_gets_one(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.data.label_queries import admit, admit_run

    images_dir, labels_dir = _dirs(_attribute_scoped_dataset(tmp_path / "ds"))
    with pytest.raises(ValueError, match="records no id_map"):
        admit(images_dir, labels_dir, scope=ClassScope(SUBJECT, "condition"))

    fresh = admit_run({"images_dir": images_dir, "labels_dir": labels_dir,
                       "scope": {"subject": SUBJECT, "attribute": "condition"}})
    assert fresh.scope.id_map == {"healthy": 0, "damaged": 1}
