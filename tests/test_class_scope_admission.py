"""``ClassScope``: the scope an admission produces carries every attribute the registry declares
for its subject, read back whole through a selection and a run's own data section, and a document
scope whose attributes were never read refuses at admission.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from tcip_mcp.pipelines.data.selection import ClassScope, read_selection
from tcip_mcp.subject_registry import Attribute
from tcip_mcp.tools.data_tools import draw_splits

from tests.test_selection_binding import DATES, SUBJECT, _attribute_scoped_dataset

CONDITION = Attribute(name="condition", type="categorical", values=("healthy", "damaged"))
"""The attribute the fixture's registry declares on its subject."""


def test_a_scope_the_admission_produced_round_trips_through_the_selection_and_the_run(
    tmp_path: Path,
) -> None:
    """The admitting case, through the producers: the draw records the scope its admission read
    targets under, every declared attribute in it, the selection reads it back whole, and a run
    bound to it records that same scope on its own data section."""
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = _attribute_scoped_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = draw_splits(tmp_path, str(root), output_path=str(out), subject=SUBJECT,
                         seed=1, train_ratio=0.5, val_ratio=0.25,
                         calibration_ratio=0.125, holdout_ratio=0.125)
    assert "error" not in result, result

    drawn = read_selection(out, project=tmp_path)
    assert drawn.scope == ClassScope(SUBJECT, (CONDITION,))
    assert result["scope"] == asdict(drawn.scope)

    data_cfg: dict = {"split": {"selection_dir": str(out)}}
    auto_train_val(tmp_path, "detection", data_cfg, None)
    assert ClassScope.of(data_cfg) == drawn.scope


REVERSED = Attribute(name="condition", type="categorical", values=("damaged", "healthy"))
"""The reverse of the order the fixture's registry declares ``condition`` in."""


def _images_dir(root: Path) -> str:
    return str(root / "images" / DATES[0])


def test_a_run_recording_its_attributes_is_admitted_under_them_not_the_registrys(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.data.split_construction import run_membership

    images_dir = _images_dir(_attribute_scoped_dataset(tmp_path / "ds"))
    membership = run_membership({"images_dir": images_dir,
                                 "scope": asdict(ClassScope(SUBJECT, (REVERSED,)))})

    assert membership.scope.attributes == (REVERSED,)


def test_a_document_scope_with_no_attributes_read_refuses_and_a_fresh_statement_reads_them(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.data.label_queries import admit
    from tcip_mcp.pipelines.data.split_construction import run_membership

    images_dir = _images_dir(_attribute_scoped_dataset(tmp_path / "ds"))
    with pytest.raises(ValueError, match="records no attributes"):
        admit(images_dir, scope=ClassScope(SUBJECT))

    fresh = run_membership({"images_dir": images_dir, "scope": {"subject": SUBJECT}})
    assert fresh.scope.attributes == (CONDITION,)
