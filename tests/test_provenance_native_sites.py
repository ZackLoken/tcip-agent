"""Native provenance stamping at the non-web write sites, the read path that carries it back out,
plus the identity helper.

Web sites (annotate save, review accept/edit) are covered in test_tcip_web_routes.py; the
proposal engines (stage_proposals) and stage_proposals in test_vision.py / test_review_channel.py.
"""

from __future__ import annotations

import json

from PIL import Image


# ── identity helper ──────────────────────────────────────────────────────────

def test_user_id_prefixes_humans_idempotently():
    from tcip_web.identity import user_id
    assert user_id("breeder") == "user:breeder"
    assert user_id("user:breeder") == "user:breeder"   # idempotent, never doubles
    assert user_id("") == "user:gui"             # never bare / empty


def test_resolve_user_prefers_explicit(monkeypatch):
    from tcip_web import identity
    monkeypatch.setenv("TCIP_USER", "osuser")
    assert identity.resolve_user("emily") == "emily"     # GUI value wins
    assert identity.resolve_user(None) == "osuser"       # falls back to env
    assert identity.resolve_user("  ") == "osuser"       # blank -> fallback


def test_current_user_env_override(monkeypatch):
    from tcip_web import identity
    monkeypatch.setenv("TCIP_USER", "breeder")
    assert identity.current_user() == "breeder"


# ── MCP save_annotations: optional producer created_by ───────────────────────

def _img(tmp_path):
    p = tmp_path / "images" / "IMG_0001.JPG"
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80)).save(p)
    return p


def test_save_annotations_stamps_created_by_when_given(tmp_path):
    from tcip_mcp.tools.annotation_tools import save_annotations
    img = _img(tmp_path)
    out = tmp_path / "labels.json"
    save_annotations(str(img), annotations=[{"subject": "bud", "bbox": [10, 10, 30, 30]}],
                     path=str(out), created_by="claude")
    obj = json.loads(out.read_text())["annotations"][0]
    assert obj["created_by"] == "claude"        # producer named by the agent
    assert obj["created_at"]


def test_save_annotations_no_provenance_by_default(tmp_path):
    """No created_by arg -> provenance stays unset (honest: don't fabricate an author)."""
    from tcip_mcp.tools.annotation_tools import save_annotations
    img = _img(tmp_path)
    out = tmp_path / "labels.json"
    save_annotations(str(img), annotations=[{"subject": "bud", "bbox": [10, 10, 30, 30]}],
                     path=str(out))
    obj = json.loads(out.read_text())["annotations"][0]
    assert "created_by" not in obj
    assert "created_at" not in obj


def test_save_annotations_per_shape_created_by_overrides(tmp_path):
    from tcip_mcp.tools.annotation_tools import save_annotations
    img = _img(tmp_path)
    out = tmp_path / "labels.json"
    save_annotations(
        str(img),
        annotations=[
            {"subject": "bud", "bbox": [10, 10, 30, 30], "created_by": "sam"},
            {"subject": "bud", "bbox": [40, 40, 60, 60]},
        ],
        path=str(out), created_by="claude",
    )
    objs = json.loads(out.read_text())["annotations"]
    assert objs[0]["created_by"] == "sam"       # per-shape wins
    assert objs[1]["created_by"] == "claude"    # falls back to the param


# ── MCP read_annotations: authorship travels back out of the read path ───────

def test_the_read_path_carries_authorship_out_of_the_record():
    """``_ann_dict`` is what ``read_annotations`` returns per annotation. Reference admissibility
    turns on created_by and accepted_by, so a read path that dropped them would leave a stamped
    author knowable only by opening the label file."""
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.annotation_tools import _ann_dict

    d = _ann_dict(Annotation(subject="bud", geometry=BBox(1, 2, 3, 4),
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
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.annotation_tools import _ann_dict

    d = _ann_dict(Annotation(subject="bud", geometry=BBox(1, 2, 3, 4)))

    assert "created_by" not in d and "accepted_by" not in d
