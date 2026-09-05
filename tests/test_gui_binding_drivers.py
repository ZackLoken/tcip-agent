"""The live-GUI binding rail, shared: ``focus_human_attention`` and ``push_panel_event`` now
refuse under the same ``canvas_open_binding`` mismatch ``capture_live_canvas`` already refused
under, through the one predicate ``web_client.gui_binding_matches`` all three read.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402
from tcip_mcp.dataset_layout import annotation_dir, image_dir  # noqa: E402
from tcip_mcp.tools.gui_tools import focus_human_attention, push_panel_event  # noqa: E402
from tcip_web.app import app  # noqa: E402

from tests.test_canvas_liveview import _mint_binding, _select  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _scene(root: Path, date: str) -> None:
    idir = Path(image_dir(root, date))
    idir.mkdir(parents=True, exist_ok=True)
    (idir / "IMG_0001.JPG").write_bytes(b"x")
    adir = Path(annotation_dir(root, date))
    adir.mkdir(parents=True, exist_ok=True)
    json_io.write_annotations(
        str(adir / "IMG_0001.json"),
        [Annotation(subject="bud", geometry=BBox(10.0, 10.0, 20.0, 20.0))],
        100, 100,
    )


def _delivery_spy(monkeypatch) -> list[tuple[str, str, dict]]:
    """Patch the one HTTP push so a "delivers" test proves it reached the post, without a real
    HTTP call or a live GUI to steer: both drivers import ``post_panel_event`` from
    ``tcip_mcp.web_client`` at call time, so patching the module attribute here is what a lazy
    re-import inside the tool sees.
    """
    import tcip_mcp.web_client as web_client_module

    calls: list[tuple[str, str, dict]] = []

    def _spy(panel: str, event_type: str, data: dict, **kwargs) -> dict:
        calls.append((panel, event_type, data))
        return {"status": "ok", "delivered": True}

    monkeypatch.setattr(web_client_module, "post_panel_event", _spy)
    return calls


# ── foreign binding: the stand-in is enough, since the shape under test is the mismatch
# itself, not the writer's own field derivation (`_mint_binding`'s own docstring names this) ──

def test_focus_human_attention_refuses_under_a_foreign_binding(
    tmp_path: Path, monkeypatch, tmp_path_factory,
) -> None:
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    date = "2026-03-02"
    _scene(tmp_path, date)
    other = tmp_path_factory.mktemp("elsewhere")
    _mint_binding(other, generation=3, project_name="currant_bud_valley")

    res = focus_human_attention("annotate", str(tmp_path), str(tmp_path), "bud", date)

    assert "error" in res
    assert res["bound_project"] == "currant_bud_valley"
    assert res["compared_root"] == str(tmp_path)


def test_push_panel_event_refuses_under_a_foreign_binding(
    tmp_path: Path, monkeypatch, tmp_path_factory,
) -> None:
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    other = tmp_path_factory.mktemp("elsewhere")
    _mint_binding(other, generation=3, project_name="currant_bud_valley")

    res = push_panel_event(panel="training", event_type="metrics_update", data={"epoch": 1},
                           project_root=str(tmp_path))

    assert "error" in res
    assert res["delivered"] is False
    assert res["bound_project"] == "currant_bud_valley"
    assert res["panel"] == "training"


# ── matching binding: minted through the real writer, delivery proven by a spy on the post ──

def test_focus_human_attention_delivers_under_a_matching_binding(
    tmp_path: Path, monkeypatch, client,
) -> None:
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    date = "2026-03-02"
    _scene(tmp_path, date)
    _select(client, tmp_path)
    calls = _delivery_spy(monkeypatch)

    res = focus_human_attention("annotate", str(tmp_path), str(tmp_path), "bud", date)

    assert "error" not in res, res
    assert res["image"] == "IMG_0001.JPG"
    assert res["delivered"] is True
    assert len(calls) == 1
    assert calls[0][0] == "app"


def test_push_panel_event_delivers_under_a_matching_binding(
    tmp_path: Path, monkeypatch, client,
) -> None:
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    _select(client, tmp_path)
    calls = _delivery_spy(monkeypatch)

    res = push_panel_event(panel="training", event_type="metrics_update", data={"epoch": 1},
                           project_root=str(tmp_path))

    assert "bound_project" not in res  # not refused by the binding rail
    assert res["panel"] == "training"
    assert res["delivered"] is True
    assert calls == [("training", "metrics_update", {"epoch": 1})]


# ── no binding at all ────────────────────────────────────────────────────────

def test_focus_human_attention_refuses_with_no_binding_at_all(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    date = "2026-03-02"
    _scene(tmp_path, date)

    res = focus_human_attention("annotate", str(tmp_path), str(tmp_path), "bud", date)

    assert "error" in res
    assert res["bound_project"] is None
    assert res["bound_root"] is None
    assert res["delivered"] is False


def test_push_panel_event_refuses_with_no_binding_at_all(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))

    res = push_panel_event(panel="training", event_type="metrics_update", data={"epoch": 1},
                           project_root=str(tmp_path))

    assert "error" in res
    assert res["bound_project"] is None
    assert res["bound_root"] is None
    assert res["delivered"] is False


# ── the binding store itself unreadable ──────────────────────────────────────

def test_push_panel_event_refuses_when_the_binding_cannot_be_read(
    tmp_path: Path, monkeypatch,
) -> None:
    import tcip_store
    from tcip_mcp import web_client as web_client_module

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))

    def _boom(*args, **kwargs):
        raise tcip_store.StoreError("boom")

    monkeypatch.setattr(web_client_module.tcip_store, "read", _boom)

    res = push_panel_event(panel="training", event_type="metrics_update", data={"epoch": 1},
                           project_root=str(tmp_path))

    assert "error" in res
    assert "Could not read the canvas-open binding" in res["error"]
    assert res["delivered"] is False


def test_focus_human_attention_refuses_when_the_binding_cannot_be_read(
    tmp_path: Path, monkeypatch,
) -> None:
    import tcip_store
    from tcip_mcp import web_client as web_client_module

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    date = "2026-03-02"
    _scene(tmp_path, date)

    def _boom(*args, **kwargs):
        raise tcip_store.StoreError("boom")

    monkeypatch.setattr(web_client_module.tcip_store, "read", _boom)

    res = focus_human_attention("annotate", str(tmp_path), str(tmp_path), "bud", date)

    assert "error" in res
    assert "Could not read the canvas-open binding" in res["error"]
    assert res["delivered"] is False


def test_gui_binding_matches_refuses_a_record_with_no_root_field(
    tmp_path: Path, monkeypatch,
) -> None:
    """A stored binding record missing its own required ``root`` field reads as unreadable,
    not as a ``KeyError`` escaping the comparison: this seam never writes one without a
    ``root``, so a caller seeing one anyway must be told the record is illegible."""
    import tcip_store
    from tcip_mcp.web_client import GuiBindingUnreadable, canvas_open_binding_key, gui_binding_matches

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    key = canvas_open_binding_key()
    stored = tcip_store.read_versioned(key, default=None)
    tcip_store.replace(key, {"generation": 1, "project_name": None}, expect=stored.version)

    with pytest.raises(GuiBindingUnreadable):
        gui_binding_matches(str(tmp_path))
