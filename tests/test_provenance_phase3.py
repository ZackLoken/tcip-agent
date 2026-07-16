"""Phase 3 — native provenance stamping at the non-web write sites + the identity helper.

Web sites (annotate save, review accept/edit) are covered in test_tcip_web_routes.py; SAM
(accept_candidates) and stage_proposals in test_vision.py / test_review_channel.py.
"""

from __future__ import annotations

import json

from PIL import Image


# ── identity helper ──────────────────────────────────────────────────────────

def test_user_id_prefixes_humans_idempotently():
    from tcip_web.identity import user_id
    assert user_id("zack") == "user:zack"
    assert user_id("user:zack") == "user:zack"   # idempotent, never doubles
    assert user_id("") == "user:gui"             # never bare / empty


def test_resolve_user_prefers_explicit(monkeypatch):
    from tcip_web import identity
    monkeypatch.setenv("TCIP_USER", "osuser")
    assert identity.resolve_user("emily") == "emily"     # GUI value wins
    assert identity.resolve_user(None) == "osuser"       # falls back to env
    assert identity.resolve_user("  ") == "osuser"       # blank -> fallback


def test_current_user_env_override(monkeypatch):
    from tcip_web import identity
    monkeypatch.setenv("TCIP_USER", "zack")
    assert identity.current_user() == "zack"
    monkeypatch.delenv("TCIP_USER", raising=False)
    monkeypatch.setenv("TCIP_REVIEW_USER", "emily")
    assert identity.current_user() == "emily"            # legacy key still honored


# ── MCP save_annotations: optional producer created_by ───────────────────────

def _img(tmp_path):
    p = tmp_path / "images" / "IMG_0001.JPG"
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80)).save(p)
    return p


def test_save_annotations_stamps_created_by_when_given(tmp_path):
    from tcip_mcp.tools.annotation_tools import save_annotations
    img = _img(tmp_path)
    det = tmp_path / "detect.json"
    save_annotations(str(img), boxes=[{"x1": 10, "y1": 10, "x2": 30, "y2": 30, "class_id": 0}],
                     detect_path=str(det), created_by="claude")
    obj = json.loads(det.read_text())["objects"][0]
    assert obj["created_by"] == "claude"        # producer named by the agent
    assert obj["created_at"]


def test_save_annotations_no_provenance_by_default(tmp_path):
    """No created_by arg -> provenance stays unset (honest: don't fabricate an author)."""
    from tcip_mcp.tools.annotation_tools import save_annotations
    img = _img(tmp_path)
    det = tmp_path / "detect.json"
    save_annotations(str(img), boxes=[{"x1": 10, "y1": 10, "x2": 30, "y2": 30, "class_id": 0}],
                     detect_path=str(det))
    obj = json.loads(det.read_text())["objects"][0]
    assert "created_by" not in obj
    assert "created_at" not in obj


def test_save_annotations_per_shape_created_by_overrides(tmp_path):
    from tcip_mcp.tools.annotation_tools import save_annotations
    img = _img(tmp_path)
    det = tmp_path / "detect.json"
    save_annotations(
        str(img),
        boxes=[
            {"x1": 10, "y1": 10, "x2": 30, "y2": 30, "class_id": 0, "created_by": "sam"},
            {"x1": 40, "y1": 40, "x2": 60, "y2": 60, "class_id": 0},
        ],
        detect_path=str(det), created_by="claude",
    )
    objs = json.loads(det.read_text())["objects"]
    assert objs[0]["created_by"] == "sam"       # per-shape wins
    assert objs[1]["created_by"] == "claude"    # falls back to the param
