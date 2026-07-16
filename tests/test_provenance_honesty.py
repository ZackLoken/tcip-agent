"""Provenance honesty: a `derived()` stamp must trace to a real implementation.

Guards against the cross_tile_nms costume bug: code once stamped derived_from="GT neighbor-IoU
distribution" while no function computed it. Every derived_from label stamped anywhere in
tcip_mcp must be a reviewed entry in DERIVATION_IMPLEMENTATIONS — mapped to an importable
callable, or explicitly marked as a non-derivation ("caller-input"/"placeholder")."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

from tcip_mcp.pipelines.derivations import DERIVATION_IMPLEMENTATIONS

SRC = Path(__file__).parent.parent / "packages" / "tcip-mcp" / "src" / "tcip_mcp"


def _stamped_labels() -> list[tuple[str, str]]:
    """(file, derived_from literal-or-prefix) for every derived(...) call in tcip_mcp."""
    out = []
    for py in SRC.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "id", getattr(node.func, "attr", "")) == "derived"):
                continue
            for kw in node.keywords:
                if kw.arg != "derived_from":
                    continue
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    out.append((py.name, kw.value.value))
                elif isinstance(kw.value, ast.JoinedStr):  # f-string: match by leading literal
                    head = kw.value.values[0]
                    if isinstance(head, ast.Constant):
                        out.append((py.name, head.value.strip()))
    return out


def test_every_derived_stamp_has_a_registered_implementation():
    labels = _stamped_labels()
    assert labels, "expected derived() stamp sites in tcip_mcp"
    for fname, label in labels:
        hit = label in DERIVATION_IMPLEMENTATIONS or any(
            label.startswith(k) for k in DERIVATION_IMPLEMENTATIONS)
        assert hit, (f"{fname}: derived_from={label!r} is stamped but not registered in "
                     "DERIVATION_IMPLEMENTATIONS — add the implementing callable "
                     "(or an explicit non-derivation marker) before stamping it.")


def test_registered_implementations_exist_and_are_callable():
    for label, target in DERIVATION_IMPLEMENTATIONS.items():
        if target in ("caller-input", "placeholder"):
            continue
        module, _, attr = str(target).rpartition(".")
        fn = getattr(importlib.import_module(module), attr, None)
        assert callable(fn), f"{label!r} points at {target} which is not an importable callable"
