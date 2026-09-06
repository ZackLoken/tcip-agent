"""frozen-formats.json is the shipped freeze commitment, held to the registry by this test.

The manifest is generated from the store registry by tools/generate_frozen_manifest.py and
committed; this test runs that same generator's --check in a fresh interpreter (one
implementation of the derivation, and a clean registry: an in-process check would also see
every throwaway store other tests registered in this worker) and refuses any drift, in both
directions: a registered store the manifest misses and a manifest entry no store backs are the
same inequality. Coverage of the commitment mechanism, not a behavior fix.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_GENERATOR = _REPO / "tools" / "generate_frozen_manifest.py"
_MANIFEST = _REPO / "frozen-formats.json"


def test_the_committed_manifest_matches_the_live_registry_exactly():
    completed = subprocess.run(
        [sys.executable, str(_GENERATOR), "--check"],
        capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "OK" in completed.stdout


def test_the_manifest_declares_a_nonempty_frozen_set_and_coherent_entries():
    committed = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    assert committed["stores"]
    assert any(entry["frozen"] for entry in committed["stores"].values())
    for name, entry in committed["stores"].items():
        if entry["cannot_carry_field"]:
            assert entry["frozen"], f"{name} declares cannot_carry_field but is not frozen"
        assert entry["schema_version"] >= 1, f"{name} declares a ceiling below 1"
