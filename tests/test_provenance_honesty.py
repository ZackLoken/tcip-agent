"""Provenance honesty: every registered derivation label names a real implementation.

A label naming a derivation no function computes is the failure this catches. Every entry in
DERIVATION_IMPLEMENTATIONS maps to an importable callable or is explicitly marked as a
non-derivation ("caller-input"/"placeholder")."""

from __future__ import annotations

import importlib

from tcip_mcp.pipelines.derivations import DERIVATION_IMPLEMENTATIONS


def test_registered_implementations_exist_and_are_callable():
    for label, target in DERIVATION_IMPLEMENTATIONS.items():
        if target in ("caller-input", "placeholder"):
            continue
        module, _, attr = str(target).rpartition(".")
        fn = getattr(importlib.import_module(module), attr, None)
        assert callable(fn), f"{label!r} points at {target} which is not an importable callable"
