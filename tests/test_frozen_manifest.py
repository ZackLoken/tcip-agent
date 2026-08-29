"""frozen-formats.json is the shipped freeze commitment, held to the registry by this test.

The manifest is generated from the store registry by scripts/generate_frozen_manifest.py and
committed; this test regenerates it in process through that same generator (one implementation,
never a second spelling of the derivation) and compares documents. A store added, dropped,
reclassified, or version-bumped without regenerating and committing the manifest fails here,
in both directions: a registered store the manifest misses and a manifest entry no store backs
are the same inequality. Coverage of the commitment mechanism, not a behavior fix.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_GENERATOR = _REPO / "scripts" / "generate_frozen_manifest.py"
_MANIFEST = _REPO / "frozen-formats.json"


def _generator_module():
    spec = importlib.util.spec_from_file_location("generate_frozen_manifest", _GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_committed_manifest_matches_the_live_registry_exactly():
    committed = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    live = _generator_module().manifest()
    assert committed == live, (
        "frozen-formats.json no longer matches the store registry; regenerate it with "
        "python scripts/generate_frozen_manifest.py and commit the result with the change "
        "that moved the registry"
    )


def test_the_manifest_declares_every_store_and_a_nonempty_frozen_set():
    committed = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    from tcip_mcp.store_catalogue import bootstrapped_stores

    assert set(committed["stores"]) == set(bootstrapped_stores())
    assert any(s["frozen"] for s in committed["stores"].values())
    for name, entry in committed["stores"].items():
        if entry["cannot_carry_field"]:
            assert entry["frozen"], f"{name} declares cannot_carry_field but is not frozen"
