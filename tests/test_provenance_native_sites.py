"""Tests of native provenance stamping at the non-web write sites and on the read path.

The read path is the one that carries the stamped provenance back out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
from tests._producer_fixtures import write_image

if TYPE_CHECKING:
    from pathlib import Path

    from tcip_annotation.state import Annotation
    from tcip_mcp.workspace import BoundProject

# ── MCP save_annotations: the tool is the producer ───────────────────────────


def _records(img: Path) -> list[Annotation]:
    """Return the records of ``img``'s label document as stored."""
    from tcip_annotation.json_io import read_label_document
    from tests._producer_fixtures import image_label_key

    return read_label_document(image_label_key(img)).annotations


def test_save_annotations_takes_no_producer_and_names_itself(
    tmp_path: Path, bound: BoundProject
) -> None:
    """``save_annotations`` takes no producer and stamps each record with its own and a time.

    The producer is the tool's own, which no caller states.
    """
    import inspect

    from tcip_mcp.tools.annotation_tools import save_annotations

    assert "created_by" not in inspect.signature(save_annotations).parameters
    img = write_image(image_dir(tmp_path, UNDATED_BUCKET) / "IMG_0001.JPG")
    save_annotations(
        bound,
        str(img),
        annotations=[{"subject": "bud", "bbox": [10, 10, 30, 30]}],
        actor=None,
    )
    obj = _records(img)[0]
    assert obj.created_by == "save_annotations"
    assert obj.created_at


def test_save_annotations_never_reads_provenance_off_a_shape(
    tmp_path: Path, bound: BoundProject
) -> None:
    """A shape's own ``created_by`` is not an author: both new shapes carry the door's."""
    from tcip_mcp.tools.annotation_tools import save_annotations

    img = write_image(image_dir(tmp_path, UNDATED_BUCKET) / "IMG_0001.JPG")
    save_annotations(
        bound,
        str(img),
        annotations=[
            {
                "subject": "bud",
                "bbox": [10, 10, 30, 30],
                "created_by": "someone-else",
            },
            {"subject": "bud", "bbox": [40, 40, 60, 60]},
        ],
        actor=None,
    )
    assert [o.created_by for o in _records(img)] == [
        "save_annotations",
        "save_annotations",
    ]


# ── MCP read_annotations: authorship travels back out of the read path ───────


def test_the_read_path_carries_authorship_out_of_the_record() -> None:
    """``ClientAnnotation`` carries a record's authorship out on the read path.

    ``ClientAnnotation`` is what ``read_annotations`` returns per annotation, dumped; it carries
    the record's created_by, accepted_by, created_at and accepted_at.
    """
    from tcip_annotation.json_io import ClientAnnotation
    from tcip_annotation.state import Annotation, BBox

    d = ClientAnnotation.of(
        Annotation(
            subject="bud",
            geometry=BBox(1, 2, 3, 4),
            created_by="model:m_best@c9f632ba98b2",
            created_at="2026-01-01T00:00:00+00:00",
            accepted_by="user:breeder",
            accepted_at="2026-01-02T00:00:00+00:00",
        )
    )

    assert d.created_by == "model:m_best@c9f632ba98b2"
    assert d.accepted_by == "user:breeder"
    assert d.created_at == "2026-01-01T00:00:00+00:00"
    assert d.accepted_at == "2026-01-02T00:00:00+00:00"


def test_the_read_path_states_no_provenance_a_record_does_not_carry() -> None:
    """An unattributed label reads back with no ``created_by`` or ``accepted_by``."""
    from tcip_annotation.json_io import ClientAnnotation
    from tcip_annotation.state import Annotation, BBox

    d = ClientAnnotation.of(
        Annotation(subject="bud", geometry=BBox(1, 2, 3, 4))
    )

    assert d.created_by is None and d.accepted_by is None
