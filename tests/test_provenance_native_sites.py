"""Native provenance stamping at the non-web write sites, the read path that carries it back out,
plus the identity helper.

The web save door is covered in test_tcip_web_routes.py; the proposal engines (stage_proposals)
in test_vision.py / test_review_channel.py.
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

import pytest
from PIL import Image


# ── the one actor function ───────────────────────────────────────────────────

def test_actor_prefixes_a_person_idempotently():
    from tcip_mcp.identity import actor
    assert actor("breeder") == "user:breeder"
    assert actor("user:breeder") == "user:breeder"   # idempotent, never doubles


@pytest.mark.parametrize("name", [None, "", "  "])
def test_actor_refuses_a_request_that_names_no_one_whatever_the_process_runs_as(
        monkeypatch, name):
    from tcip_mcp import identity
    monkeypatch.setenv("TCIP_USER", "osuser")
    with pytest.raises(ValueError, match="names no one"):
        identity.actor(name)


# ── MCP save_annotations: optional producer created_by ───────────────────────

def _img(tmp_path):
    p = tmp_path / "images" / UNDATED_BUCKET / "IMG_0001.JPG"
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80)).save(p)
    return p


def _records(img) -> list[dict]:
    """The records of ``img``'s label document as stored."""
    import tcip_store

    from tests._producer_fixtures import image_label_key

    return tcip_store.read(image_label_key(img))["annotations"]


def test_save_annotations_takes_no_producer_and_names_itself(tmp_path):
    """Every saved record carries a producer and its time: the tool's own, which no caller
    states."""
    import inspect

    from tcip_mcp.tools.annotation_tools import save_annotations
    assert "created_by" not in inspect.signature(save_annotations).parameters
    img = _img(tmp_path)
    save_annotations(tmp_path, tmp_path.parent, str(img),
                     annotations=[{"subject": "bud", "bbox": [10, 10, 30, 30]}])
    obj = _records(img)[0]
    assert obj["created_by"] == "save_annotations"
    assert obj["created_at"]


def test_save_annotations_never_reads_provenance_off_a_shape(tmp_path):
    """A shape's own ``created_by`` is not an author: every new shape carries the door's."""
    from tcip_mcp.tools.annotation_tools import save_annotations
    img = _img(tmp_path)
    save_annotations(
        tmp_path, tmp_path.parent, str(img),
        annotations=[
            {"subject": "bud", "bbox": [10, 10, 30, 30], "created_by": "someone-else"},
            {"subject": "bud", "bbox": [40, 40, 60, 60]},
        ],
    )
    assert [o["created_by"] for o in _records(img)] == ["save_annotations", "save_annotations"]


# ── MCP read_annotations: authorship travels back out of the read path ───────

def test_the_read_path_carries_authorship_out_of_the_record():
    """``client_annotation`` is what ``read_annotations`` returns per annotation. Reference
    admissibility turns on created_by and accepted_by, so a read path that dropped them would leave
    a stamped author knowable only by opening the label file."""
    from tcip_annotation.json_io import client_annotation
    from tcip_annotation.state import Annotation, BBox

    d = client_annotation(Annotation(subject="bud", geometry=BBox(1, 2, 3, 4),
                             created_by="model:m_best@c9f632ba98b2",
                             created_at="2026-01-01T00:00:00+00:00",
                             accepted_by="user:breeder",
                             accepted_at="2026-01-02T00:00:00+00:00"))

    assert d["created_by"] == "model:m_best@c9f632ba98b2"
    assert d["accepted_by"] == "user:breeder"
    assert d["created_at"] == "2026-01-01T00:00:00+00:00"
    assert d["accepted_at"] == "2026-01-02T00:00:00+00:00"


def test_the_read_path_omits_provenance_a_record_does_not_carry():
    """An unattributed label reads back unattributed, not with null authorship keys that a reader
    could mistake for a recorded absence."""
    from tcip_annotation.json_io import client_annotation
    from tcip_annotation.state import Annotation, BBox

    d = client_annotation(Annotation(subject="bud", geometry=BBox(1, 2, 3, 4)))

    assert "created_by" not in d and "accepted_by" not in d
