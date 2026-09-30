"""The trait listings the frontend tests read are the platform's own current output: regenerated
through the generator's own producer path, they equal the checked-in fixture."""

from __future__ import annotations

import json
from pathlib import Path

from tools.generate_trait_fixture import GENERATED_PATH, listings


def test_the_checked_in_listings_are_what_the_producers_serve_today(tmp_path: Path) -> None:
    checked_in = json.loads(GENERATED_PATH.read_text(encoding="utf-8"))

    assert listings(tmp_path.parent) == checked_in
