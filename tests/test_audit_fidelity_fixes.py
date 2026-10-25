"""Data fidelity at the MCP save door, the prediction encoder and the split stratifier.

An emptied document survives the MCP save door, inference predictions carry model provenance,
and stratified splits count JSON objects, not JSON lines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import tcip_store as ts
from tcip_annotation import json_io
from tests._producer_fixtures import (
    blank_image,
    box_annotation,
    image_label_key,
    label_image,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tcip_mcp.workspace import BoundProject


def test_mcp_save_annotations_empty_writes_the_empty_document_the_route_writes(
    bound: BoundProject,
) -> None:
    """An empty MCP save writes an empty label document over an annotated one.

    The emptied document stays, holding nothing, and the save answers the capture it wrote.
    """
    from tcip_mcp.tools.annotation_tools import save_annotations

    img = blank_image(bound.root)
    label_image(img, [box_annotation(1, 1, 9, 9)], 100, 80)
    res = save_annotations(bound, str(img), annotations=[], actor=None)
    assert res == {"capture": "undated", "written": [img.stem], "count": 0}
    assert ts.read(image_label_key(img), default=None) is not None
    assert json_io.read_label_document(image_label_key(img)).annotations == []


@pytest.mark.parametrize(
    "bad",
    [{"bbox": [1, 1, 5, 5]}, {"subject": ""}],
    ids=["no_subject", "empty_subject"],
)
def test_mcp_save_annotations_refuses_by_index_through_the_decoders_checks(
    bound: BoundProject, bad: dict[str, list[int]] | dict[str, str]
) -> None:
    """The MCP save refuses a malformed annotation by its index and writes nothing.

    An annotation that names no subject or names an empty one is refused naming its index; the
    valid one beside it is not written alone.
    """
    from tcip_mcp.tools.annotation_tools import save_annotations

    img = blank_image(bound.root)
    res = save_annotations(
        bound,
        str(img),
        annotations=[{"subject": "bur", "bbox": [1, 1, 5, 5]}, bad],
        actor=None,
    )
    assert res["error"].startswith("annotation 1 ")
    assert ts.read(image_label_key(img), default=None) is None


def test_encode_predictions_stamps_model_provenance(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.data.label_queries import registry_scope
    from tcip_mcp.pipelines.postprocessing.export import encode_predictions

    data, _dropped = encode_predictions(
        {
            "image": "pred.jpg",
            "width": 100,
            "height": 80,
            "boxes": [[10, 10, 30, 30]],
            "scores": [0.9],
            "labels": [1],
            "cap": 10,
        },
        created_by="model:best_bud",
        scope=registry_scope(tmp_path, "bud"),
    )
    obj = data["annotations"][0]
    assert obj["created_by"] == "model:best_bud"
    assert obj["created_at"]
    assert obj["score"] == pytest.approx(0.9)


def test_draw_splits_counts_json_objects_not_lines(tmp_path: Path) -> None:
    """``draw_splits`` counts a document's annotation objects, not its text lines.

    Over documents of three, one, zero and zero boxes, the split's foreground annotations sum
    to 4.
    """
    from tcip_mcp.tools.data_tools import draw_splits
    from tests._producer_fixtures import mark_complete
    from tests._training_values import RATIONALE

    images = [blank_image(tmp_path, name=f"img_{i}.JPG") for i in range(4)]

    box = box_annotation(1, 1, 9, 9)
    label_image(images[0], [box, box, box], 100, 80)
    label_image(images[1], [box], 100, 80)
    for negative in images[2:]:
        label_image(negative, [], 100, 80, keep_empty=True)
        mark_complete(negative, "bud", project=tmp_path)

    res = draw_splits(
        tmp_path,
        str(tmp_path),
        seed=1,
        val_ratio=0.5,
        calibration_ratio=0.0,
        holdout_ratio=0.0,
        grouping="stem",
        stratify_foreground=True,
        rationale=RATIONALE,
        subject="bud",
        actor=None,
    )
    assert "error" not in res
    # foreground_annotations sums per split: true total is 3+1+0+0. Counting raw JSON text
    # lines instead reports dozens, since negatives alone read as several each.
    assert sum(res["foreground_annotations"].values()) == 4
