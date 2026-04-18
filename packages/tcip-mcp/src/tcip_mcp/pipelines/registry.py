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
