"""GUARDS proof, kept isolated: ``record_delivery_binding_event`` stamps a delivered file's own
digest, exercised directly against a hand-written file so the proof needs nothing from the
plant-registry family (``resolution.py`` is the only file this test's fix touches).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import tcip_store as ts

from tcip_mcp.pipelines import resolution


def test_a_delivered_files_own_bytes_are_the_recorded_digest(tmp_path: Path) -> None:
    """GUARDS: record_delivery_binding_event stamps output_sha256 from the file it names."""
    out_csv = tmp_path / "out.csv"
    out_csv.write_text("plant_id,count\nP1,3\n", encoding="utf-8")

    resolution.record_delivery_binding_event(
        "test_door", str(out_csv), [], {}, measurement_documents=["operating_point"],
        scale_document=None, trait="astringency", delivery_kind="state_crossing_dates",
        project_root=tmp_path, plant_mapping=None,
    )

    scope = resolution.delivery_events_scope(tmp_path)
    keys = ts.keys(resolution.DELIVERY_EVENTS_STORE, str(scope))
    events = [ts.read(k) for k in keys if ts.read(k)["door"] == "test_door"]
    assert len(events) == 1, events
    assert events[0]["output_sha256"] == hashlib.sha256(out_csv.read_bytes()).hexdigest()
