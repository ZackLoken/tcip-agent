"""``deliver_csv`` records a delivered file's own digest."""

from __future__ import annotations

import hashlib
from pathlib import Path

from tcip_mcp import delivery
from tests._trait_fixtures import seed_confirmed_count
from tests.test_delivery_events import _record


def test_a_delivered_files_own_bytes_are_the_recorded_digest(tmp_path: Path) -> None:
    _record(tmp_path, seed_confirmed_count(tmp_path))

    (event,) = delivery.read_delivery_events(tmp_path)
    delivered = Path(event.output_path).read_bytes()
    assert delivered and event.output_sha256 == hashlib.sha256(delivered).hexdigest()
