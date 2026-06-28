"""Component registry — generic registry pattern for all ML primitives.

Every composable ML component (backbone, neck, head, loss, optimizer,
scheduler, sampler, augmentation) registers into a typed registry with
self-describing metadata so the agent can query, reason about, and
compose components at runtime.

Uses a plugin-style registry with layered capability pools.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


# Canonical (optional) metadata fields and their expected types. Registration stays
# non-breaking: unknown keys are allowed; known keys are type-checked and a mismatch is
# logged as a warning (surfaces typos / wrong types at import time without crashing).
_METADATA_FIELD_TYPES: dict[str, tuple[type, ...]] = {
    "description": (str,),
    "valid_tasks": (list,),
    "input_format": (str,),
    "output_format": (str,),
    "supported_channels": (list, str),  # e.g. [3] or "any"
    "required_backbone_stages": (int,),
    "min_pyramid_levels": (int,),
    "max_pyramid_levels": (int,),
    "required_deps": (list,),
    "params_M": (int, float),
    "options": (dict,),
}


def validate_component_metadata(
    metadata: dict[str, Any], *, registry: str, component: str
) -> list[str]:
    """Type-check recognized metadata fields; log (don't raise) on mismatch."""
    issues = []
    for key, expected in _METADATA_FIELD_TYPES.items():
        if key in metadata and not isinstance(metadata[key], expected):
            issues.append(
                f"{registry}.{component}: metadata['{key}'] should be "
                f"{' | '.join(t.__name__ for t in expected)}, got "
                f"{type(metadata[key]).__name__}"
            )
    for msg in issues:
        logger.warning(msg)
    return issues


class ComponentRegistry:
    """A typed registry for ML components.

    Usage::

        BACKBONES = ComponentRegistry("backbones")

        @BACKBONES.register("resnet50", category="cnn", metadata={...})
        def build_resnet50(**kwargs):
            ...

        model = BACKBONES.build("resnet50", pretrained=True)
        info = BACKBONES.describe("resnet50")
        all_cnns = BACKBONES.list(category="cnn")
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._entries: dict[str, _Entry] = {}

    def register(
        self,
        component_name: str,
        *,
        category: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> Callable:
        """Decorator that registers a factory function.

        Args:
            component_name: Unique name within this registry.
            category: Grouping label (e.g. "cnn", "vit", "anchor_based").
            metadata: Self-describing dict for agent reasoning. Suggested keys:
                description, valid_tasks, input_format, output_format,
                constraints, default_params, optional_deps.
        """
        def decorator(factory_fn: Callable) -> Callable:
            if component_name in self._entries:
                raise ValueError(
                    f"Duplicate registration in {self.name}: '{component_name}'"
                )
            validate_component_metadata(
                metadata or {}, registry=self.name, component=component_name
            )
            self._entries[component_name] = _Entry(
                name=component_name,
                category=category,
                factory_fn=factory_fn,
                metadata=metadata or {},
            )
            return factory_fn
        return decorator

    def register_factory(
        self,
        component_name: str,
        factory_fn: Callable,
        *,
        category: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Imperative registration (non-decorator form)."""
        if component_name in self._entries:
            raise ValueError(
                f"Duplicate registration in {self.name}: '{component_name}'"
            )
        validate_component_metadata(
            metadata or {}, registry=self.name, component=component_name
        )
        self._entries[component_name] = _Entry(
            name=component_name,
            category=category,
            factory_fn=factory_fn,
            metadata=metadata or {},
        )

    def get(self, name: str) -> Callable:
        """Return the raw factory function."""
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(
                f"Unknown component '{name}' in {self.name}. "
                f"Available: {sorted(self._entries)}"
            )
        return entry.factory_fn

    def build(self, name: str, **kwargs: Any) -> Any:
        """Instantiate a component by name."""
        return self.get(name)(**kwargs)

    def describe(self, name: str) -> dict[str, Any]:
        """Return metadata dict for agent reasoning."""
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(f"Unknown component '{name}' in {self.name}")
        return {
            "name": entry.name,
            "category": entry.category,
            "registry": self.name,
            **copy.deepcopy(entry.metadata),
        }

    def list(
        self,
        category: str | None = None,
        filter_fn: Callable[[dict], bool] | None = None,
    ) -> list[dict[str, Any]]:
        """List registered components with optional filtering.

        Args:
            category: Filter to this category only.
            filter_fn: Predicate on the metadata dict.
        """
        results = []
        for entry in self._entries.values():
            if category is not None and entry.category != category:
                continue
            info = {
                "name": entry.name,
                "category": entry.category,
                **copy.deepcopy(entry.metadata),
            }
            if filter_fn is not None and not filter_fn(info):
                continue
            results.append(info)
        return results

    # -- Query API (let the agent discover compatible components) -----------

    def list_by_format(
        self, *, input_format: str | None = None, output_format: str | None = None
    ) -> list[dict[str, Any]]:
        """Components matching the given input/output ``*_format`` metadata."""
        def _match(info: dict) -> bool:
            if input_format is not None and info.get("input_format") != input_format:
                return False
            if output_format is not None and info.get("output_format") != output_format:
                return False
            return True

        return [i for i in self.list() if _match(i)]

    def list_compatible_with(self, producer_metadata: dict[str, Any]) -> list[dict[str, Any]]:
        """Components whose ``input_format`` matches a producer's ``output_format``."""
        out_fmt = (producer_metadata or {}).get("output_format")
        if out_fmt is None:
            return []
        return [i for i in self.list() if i.get("input_format") == out_fmt]

    def supports_channels(self, name: str, num_channels: int) -> bool:
        """Whether component ``name`` can consume ``num_channels``-channel input.

        Reads ``supported_channels`` metadata: ``"any"`` → True; a list → membership.
        If unspecified, assumes the current RGB reality (3 channels only).
        """
        sc = self.describe(name).get("supported_channels")
        if sc is None:
            return num_channels == 3
        if isinstance(sc, str):
            return sc.lower() == "any"
        return num_channels in sc

    def find_by_constraint(self, **constraints: Any) -> list[dict[str, Any]]:
        """Components whose metadata matches all ``key=value`` constraints.

        List-valued metadata (e.g. ``valid_tasks``) is matched by membership.
        """
        def _match(info: dict) -> bool:
            for k, v in constraints.items():
                actual = info.get(k)
                if isinstance(actual, list):
                    if v not in actual:
                        return False
                elif actual != v:
                    return False
            return True

        return [i for i in self.list() if _match(i)]

    def register_external(
        self,
        component_name: str,
        factory_fn: Callable,
        *,
        category: str = "external",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a component defined outside this package (plugins/experiments)."""
        self.register_factory(
            component_name, factory_fn, category=category, metadata=metadata
        )

    # -- Test isolation -----------------------------------------------------

    def clear(self) -> None:
        """Remove all registrations (use with snapshot/restore for test isolation)."""
        self._entries.clear()

    def snapshot(self) -> dict[str, _Entry]:
        """Shallow-copy the current registrations for later :meth:`restore`."""
        return dict(self._entries)

    def restore(self, snap: dict[str, _Entry]) -> None:
        """Restore a snapshot taken with :meth:`snapshot`."""
        self._entries = dict(snap)

    def names(self) -> list[str]:
        """Return sorted list of registered component names."""
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"ComponentRegistry({self.name!r}, {len(self._entries)} entries)"


class _Entry:
    __slots__ = ("name", "category", "factory_fn", "metadata")

    def __init__(
        self,
        name: str,
        category: str,
        factory_fn: Callable,
        metadata: dict[str, Any],
    ) -> None:
        self.name = name
        self.category = category
        self.factory_fn = factory_fn
        self.metadata = metadata


# ---------------------------------------------------------------------------
# Global singleton registries
# ---------------------------------------------------------------------------

BACKBONES = ComponentRegistry("backbones")
NECKS = ComponentRegistry("necks")
HEADS = ComponentRegistry("heads")
LOSSES = ComponentRegistry("losses")
OPTIMIZERS = ComponentRegistry("optimizers")
SCHEDULERS = ComponentRegistry("schedulers")
SAMPLERS = ComponentRegistry("samplers")
AUGMENTATIONS = ComponentRegistry("augmentations")

_ALL_REGISTRIES = (
    BACKBONES, NECKS, HEADS, LOSSES, OPTIMIZERS, SCHEDULERS, SAMPLERS, AUGMENTATIONS,
)

# Setuptools entry-point group external packages publish component-registering hooks under.
PLUGIN_ENTRY_POINT_GROUP = "tcip.components"


def load_plugins(group: str = PLUGIN_ENTRY_POINT_GROUP) -> list[str]:
    """Invoke entry-point hooks that register external components.

    Each entry point must resolve to a zero-arg callable that performs its own
    ``REGISTRY.register_external(...)`` calls. Returns the names successfully loaded;
    a failing plugin is logged and skipped (never breaks the host).
    """
    import importlib.metadata as importlib_metadata

    loaded: list[str] = []
    for ep in importlib_metadata.entry_points(group=group):
        try:
            ep.load()()
            loaded.append(ep.name)
        except Exception:
            logger.exception("Failed to load component plugin '%s'", ep.name)
    return loaded


def snapshot_all() -> dict[str, dict]:
    """Snapshot every global registry (pair with :func:`restore_all` in tests)."""
    return {r.name: r.snapshot() for r in _ALL_REGISTRIES}


def restore_all(snap: dict[str, dict]) -> None:
    """Restore registries from :func:`snapshot_all`."""
    by_name = {r.name: r for r in _ALL_REGISTRIES}
    for name, entries in snap.items():
        if name in by_name:
            by_name[name].restore(entries)
