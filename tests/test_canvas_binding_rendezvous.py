"""The canvas_open_binding record as the source at every rendezvous with a browser: the
lifespan's startup read, the WS connect/resync re-read, and the select route's own ordering
against the dataset it names. Covers the fix-up round's three named live-reproduced cases:
a restart, a deleted record, and an external bump followed by a resync.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import tcip_store as ts  # noqa: E402
from tcip_mcp.web_client import canvas_open_binding_key  # noqa: E402
from tcip_web.app import app  # noqa: E402
from tcip_web.state import GuiState, store  # noqa: E402

from tests.test_canvas_liveview import SHAPES, _meta, _payload, _select  # noqa: E402


def _reset_in_memory_store() -> None:
    """Wipe the module-singleton ``StateStore`` back to a fresh process's own starting values,
    leaving every durable record (the binding, gui.json) untouched: a restart simulation."""
    store._binding_generation = None
    store._project_root = None
    store._state = GuiState()
    store._version = 0


@pytest.fixture(autouse=True)
def _restore_store_after():
    yield
    _reset_in_memory_store()


def test_a_restarted_process_replays_the_records_generation_and_a_push_succeeds(
    tmp_path: Path,
) -> None:
    """(a) restart simulation: a fresh app process's lifespan replays the record's own
    generation, and a push naming it succeeds."""
    with TestClient(app, base_url="http://127.0.0.1") as client:
        sel = _select(client, tmp_path)
    generation = sel["generation"]

    _reset_in_memory_store()

    with TestClient(app, base_url="http://127.0.0.1") as client:
        with client.websocket_connect("ws://127.0.0.1/ws/state") as ws:
            replay = ws.receive_json()
        assert replay["generation"] == generation

        r = client.post(
            "/api/canvas/state",
            json=_payload("C:/img/a.jpg", generation, shapes=SHAPES),
        )
        assert r.status_code == 200
        assert _meta(tmp_path)["image_path"] == "C:/img/a.jpg"


def test_b_a_record_deleted_after_adoption_replays_null_and_the_gate_trips(
    tmp_path: Path,
) -> None:
    """(b) record deleted after adoption: the next replay carries a null generation (the
    client-side presence gate trips on it), and the route's 409 names the missing record."""
    with TestClient(app, base_url="http://127.0.0.1") as client:
        sel = _select(client, tmp_path)
        generation = sel["generation"]

        key = canvas_open_binding_key()
        ts.delete(key, expect=ts.read_versioned(key).version)

        with client.websocket_connect("ws://127.0.0.1/ws/state") as ws:
            replay = ws.receive_json()
        assert replay["generation"] is None

        r = client.post(
            "/api/canvas/state",
            json=_payload("C:/img/a.jpg", generation, shapes=SHAPES),
        )
        assert r.status_code == 409
        assert r.json()["detail"]["generation"] is None


def test_c_an_external_bump_converges_after_a_resync(tmp_path: Path) -> None:
    """(c) an external bump (another process's own select, simulated by writing the record
    directly through the store seam): a resync's replay carries the bumped generation, and
    the next push succeeds against it."""
    with TestClient(app, base_url="http://127.0.0.1") as client:
        sel = _select(client, tmp_path)
        stale_generation = sel["generation"]

        key = canvas_open_binding_key()
        current = ts.read_versioned(key)
        bumped = {**current.value, "generation": current.value["generation"] + 1}
        ts.replace(key, bumped, expect=current.version)

        stale = client.post(
            "/api/canvas/state",
            json=_payload("C:/img/a.jpg", stale_generation, shapes=SHAPES),
        )
        assert stale.status_code == 409

        with client.websocket_connect("ws://127.0.0.1/ws/state") as ws:
            replay = ws.receive_json()
        assert replay["generation"] == bumped["generation"]

        fresh = client.post(
            "/api/canvas/state",
            json=_payload("C:/img/b.jpg", bumped["generation"], shapes=SHAPES),
        )
        assert fresh.status_code == 200
        assert _meta(tmp_path)["image_path"] == "C:/img/b.jpg"
