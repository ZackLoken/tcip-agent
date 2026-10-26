"""``ClassScope``: the scope an admission produces and its round trip through a selection and a run.

A drawn scope carries the attributes the registry declares for its subject, read back whole
through a selection and a run's own data section, and a scope recording no attributes refuses at
admission.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tcip_mcp.dataset_layout import image_dir
from tcip_mcp.pipelines.data.selection import ClassScope, read_selection
from tcip_mcp.subject_registry import Attribute
from tcip_mcp.tools.data_tools import draw_splits
from tests._training_values import drawn_split
from tests.test_selection_binding import (
    DATES,
    SUBJECT,
    _attribute_scoped_dataset,
)

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.split_construction import Membership

CONDITION = Attribute(
    name="condition", type="categorical", values=("healthy", "damaged")
)
"""The attribute the fixture's registry declares on its subject."""


def test_a_scope_the_admission_produced_round_trips_through_the_selection_and_the_run(
    tmp_path: Path,
) -> None:
    """The scope a draw's admission produced round-trips through the selection and a bound run.

    Through the producers, the draw records the scope its admission read targets under, the
    registry's declared ``condition`` attribute in it, the selection reads it back whole, and a
    run bound to it records that same scope on its own data section.
    """
    pytest.importorskip("torch")
    from tests._producer_fixtures import train_val

    root = _attribute_scoped_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = draw_splits(
        tmp_path,
        str(root),
        output_path=str(out),
        subject=SUBJECT,
        **drawn_split(
            seed=1,
            val_ratio=0.25,
            calibration_ratio=0.125,
            holdout_ratio=0.125,
            grouping="tile_prefix",
        ),
        actor=None,
    )
    assert "error" not in result, result

    drawn = read_selection(out, project=tmp_path)
    assert drawn.scope == ClassScope(SUBJECT, (CONDITION,))
    assert ClassScope.of(result) == drawn.scope

    *_loaders, resolved = train_val(
        tmp_path, "detection", {"split": {"selection_dir": str(out)}}
    )
    assert resolved.recorded_scope == drawn.scope


REVERSED = Attribute(
    name="condition", type="categorical", values=("damaged", "healthy")
)
"""The reverse of the order the fixture's registry declares ``condition`` in."""


def _images_dir(root: Path) -> str:
    return str(image_dir(root, DATES[0]))


def run_membership(block: dict) -> Membership:
    """Return the membership a run's data ``block``, drawing its own split, admits."""
    from tcip_mcp.pipelines.data.split_construction import (
        run_membership as admitted,
    )
    from tcip_mcp.pipelines.schemas import DataSpec, DrawnSplit

    data = DataSpec.model_validate(block)
    assert isinstance(data.split, DrawnSplit)
    return admitted(data, data.split)


def test_a_run_recording_its_attributes_is_admitted_under_them_not_the_registrys(
    tmp_path: Path,
) -> None:
    images_dir = _images_dir(_attribute_scoped_dataset(tmp_path / "ds"))
    membership = run_membership(
        {
            "images_dir": images_dir,
            "scope": asdict(ClassScope(SUBJECT, (REVERSED,))),
            "split": drawn_split(grouping="tile_prefix"),
        }
    )

    assert membership.scope.attributes == (REVERSED,)


def test_a_document_scope_with_no_attributes_read_refuses_and_a_fresh_statement_reads_them(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.data.label_queries import admit

    images_dir = _images_dir(_attribute_scoped_dataset(tmp_path / "ds"))
    with pytest.raises(ValueError, match="records no attributes"):
        admit(images_dir, scope=ClassScope(SUBJECT))

    fresh = run_membership(
        {
            "images_dir": images_dir,
            "scope": {"subject": SUBJECT},
            "split": drawn_split(grouping="tile_prefix"),
        }
    )
    assert fresh.scope.attributes == (CONDITION,)
