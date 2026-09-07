"""Fault-injection helpers for clear_prediction_bucket's interrupted-clear tests.

The door's own module (``tcip_mcp.tools.inference_tools``) binds the storage seam's module
object once, at import time, as the plain name ``store``; every helper function inside it reaches
the seam through that same module-level name at call time. Rebinding that one name to a proxy
that intercepts a single method call, and delegates everything else (including every other
module's own separately-bound ``store`` reference, the audited decorator's log append and the
experiment writes among them) to the real module, is what isolates a fault to the door's own call
without breaking the machinery around it.
"""

from __future__ import annotations

from typing import Any, Callable


class _StoreFaultProxy:
    """Delegates every attribute to ``real`` except ``method_name``, whose calls matching
    ``predicate`` raise ``exc`` once (the first such call) and pass through afterward."""

    def __init__(
        self, real: Any, method_name: str, *,
        predicate: Callable[[tuple, dict], bool] | None = None,
        exc: Exception | None = None,
    ) -> None:
        self._real = real
        self._method_name = method_name
        self._predicate = predicate or (lambda args, kwargs: True)
        self._exc = exc or RuntimeError("clear_prediction_bucket: injected fault")
        self.fired = False

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._real, name)
        if name != self._method_name:
            return attr

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self.fired and self._predicate(args, kwargs):
                self.fired = True
                raise self._exc
            return attr(*args, **kwargs)

        return wrapper


def inject_store_fault(
    monkeypatch, *, method_name: str, predicate: Callable[[tuple, dict], bool] | None = None,
    exc: Exception | None = None,
) -> _StoreFaultProxy:
    """Rebind ``tcip_mcp.tools.inference_tools.store`` to a proxy that raises once on the first
    call to ``method_name`` matching ``predicate``, restoring ordinary behavior after. ``exc``
    defaults to a bare ``RuntimeError``; a caller proving the door's own ``VersionConflict``
    handling passes a real one instead, so the fault reads exactly as a concurrent write the seam
    itself would raise."""
    import tcip_mcp.tools.inference_tools as inference_tools_mod
    from tcip_store import store as real_store

    proxy = _StoreFaultProxy(real_store, method_name, predicate=predicate, exc=exc)
    monkeypatch.setattr(inference_tools_mod, "store", proxy)
    return proxy


def key_in_store(*store_names: str) -> Callable[[tuple, dict], bool]:
    """A predicate for :func:`inject_store_fault`: true when the call's first positional
    argument is a ``tcip_store.Key`` whose ``store`` is one of ``store_names``."""

    def predicate(args: tuple, kwargs: dict) -> bool:
        key = args[0] if args else kwargs.get("key")
        return getattr(key, "store", None) in store_names

    return predicate


def raise_on_nth_call(monkeypatch, module: Any, attr_name: str, n: int) -> None:
    """Patch ``module.attr_name`` (a plain function) to raise on its ``n``-th call (1-indexed)
    and delegate to the original every other time. For a function the caller under test imports
    fresh, locally, on each call (so the patched module attribute is what a later local import
    picks up), never a name already bound at module-import time."""
    original = getattr(module, attr_name)
    state = {"count": 0}

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        state["count"] += 1
        if state["count"] == n:
            raise RuntimeError(f"clear_prediction_bucket: injected fault ({attr_name} call {n})")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, attr_name, wrapper)
