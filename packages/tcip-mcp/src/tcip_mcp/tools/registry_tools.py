"""Registry query tools — list crops, traits, pipeline configs.

All queries load ``registry/crops.yml`` directly via ``yaml.safe_load``.
No schema.py, no Pydantic, no Python enums.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from tcip_mcp.server import mcp

# ── Registry loader ─────────────────────────────────────────────────────────

_REGISTRY: dict | None = None
_REGISTRY_PATH: str | None = None


def _find_registry() -> Path:
    """Locate ``registry/crops.yml`` by walking up from CWD or using env var."""
    env = os.environ.get("TCIP_REGISTRY_PATH")
    if env:
        p = Path(env)
        if p.exists():
            return p

    # Walk up from CWD
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / "registry" / "crops.yml"
        if candidate.exists():
            return candidate

    # Fall back to location relative to this package
    pkg_root = Path(__file__).resolve().parents[4]  # tcip-agent repo root
    candidate = pkg_root / "registry" / "crops.yml"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Cannot find registry/crops.yml. Set TCIP_REGISTRY_PATH or run from repo root."
    )


def _load() -> dict:
    global _REGISTRY, _REGISTRY_PATH
    if _REGISTRY is None:
        path = _find_registry()
        _REGISTRY_PATH = str(path)
        with open(path, "r", encoding="utf-8") as f:
            _REGISTRY = yaml.safe_load(f)
    return _REGISTRY


# ── Tools ───────────────────────────────────────────────────────────────────


@mcp.tool()
def list_crops() -> dict:
    """List all crops with trait counts (total and automatable)."""
    reg = _load()
    crops: dict[str, dict] = {}
    for group_key, group in reg.items():
        if not isinstance(group, dict):
            continue
        has_pipeline = all(
            f in group
            for f in ("image_perspective", "sensor_type", "isolation_task", "ml_task")
        )
        for trait in group.get("traits", []):
            for crop in trait.get("crops", []):
                entry = crops.setdefault(crop, {"total": 0, "automatable": 0, "categories": set()})
                entry["total"] += 1
                entry["categories"].add(trait.get("category", "unknown"))
                if has_pipeline:
                    entry["automatable"] += 1

    # Convert sets to sorted lists for JSON serialisation
    return {
        name: {
            "total": d["total"],
            "automatable": d["automatable"],
            "categories": sorted(d["categories"]),
        }
        for name, d in sorted(crops.items())
    }


@mcp.tool()
def get_crop_traits(crop_name: str) -> dict:
    """Get all traits for a crop, grouped by pipeline status.

    Args:
        crop_name: One of the 6 supported crops.
    """
    reg = _load()
    automatable: list[dict] = []
    non_automatable: list[dict] = []

    for group_key, group in reg.items():
        if not isinstance(group, dict):
            continue
        has_pipeline = all(
            f in group
            for f in ("image_perspective", "sensor_type", "isolation_task", "ml_task")
        )
        for trait in group.get("traits", []):
            if crop_name not in trait.get("crops", []):
                continue
            entry = {
                "name": trait["name"],
                "definition": trait.get("definition", ""),
                "format": trait.get("format", ""),
                "category": trait.get("category", ""),
            }
            if has_pipeline:
                entry.update(
                    {
                        "pipeline_group": group_key,
                        "image_perspective": group["image_perspective"],
                        "sensor_type": group["sensor_type"],
                        "isolation_task": group["isolation_task"],
                        "ml_task": group["ml_task"],
                    }
                )
                automatable.append(entry)
            else:
                non_automatable.append(entry)

    return {
        "crop": crop_name,
        "automatable": automatable,
        "non_automatable": non_automatable,
        "total": len(automatable) + len(non_automatable),
        "automatable_count": len(automatable),
    }


@mcp.tool()
def get_trait_info(crop_name: str, trait_name: str) -> dict:
    """Get full pipeline configuration for a specific trait.

    Args:
        crop_name: Crop name (e.g. 'hazelnut').
        trait_name: Trait name (e.g. 'catkin_05per_date').
    """
    reg = _load()
    for group_key, group in reg.items():
        if not isinstance(group, dict):
            continue
        for trait in group.get("traits", []):
            if trait["name"] != trait_name or crop_name not in trait.get("crops", []):
                continue
            has_pipeline = all(
                f in group
                for f in ("image_perspective", "sensor_type", "isolation_task", "ml_task")
            )
            result = {
                "crop": crop_name,
                "trait": trait["name"],
                "definition": trait.get("definition", ""),
                "format": trait.get("format", ""),
                "category": trait.get("category", ""),
                "units": trait.get("units"),
                "pipeline_group": group_key,
                "automatable": has_pipeline,
            }
            if has_pipeline:
                result.update(
                    {
                        "image_perspective": group["image_perspective"],
                        "sensor_type": group["sensor_type"],
                        "isolation_task": group["isolation_task"],
                        "ml_task": group["ml_task"],
                    }
                )
            return result

    return {"error": f"Trait '{trait_name}' not found for crop '{crop_name}'."}


@mcp.tool()
def find_traits_by_task(crop_name: str, ml_task: str) -> dict:
    """Find all traits for a crop that use a given ML task.

    Args:
        crop_name: Crop name.
        ml_task: ML task type (e.g. 'object_detection', 'classification').
    """
    reg = _load()
    results: list[dict] = []
    for group_key, group in reg.items():
        if not isinstance(group, dict):
            continue
        if group.get("ml_task") != ml_task:
            continue
        for trait in group.get("traits", []):
            if crop_name not in trait.get("crops", []):
                continue
            results.append(
                {
                    "name": trait["name"],
                    "category": trait.get("category", ""),
                    "format": trait.get("format", ""),
                    "pipeline_group": group_key,
                    "sensor_type": group.get("sensor_type"),
                    "image_perspective": group.get("image_perspective"),
                }
            )
    return {"crop": crop_name, "ml_task": ml_task, "traits": results, "count": len(results)}


@mcp.tool()
def find_traits_by_sensor(crop_name: str, sensor_type: str) -> dict:
    """Find all traits for a crop that use a given sensor type.

    Args:
        crop_name: Crop name.
        sensor_type: Sensor type (e.g. 'rgb', 'lidar', 'nirs').
    """
    reg = _load()
    results: list[dict] = []
    for group_key, group in reg.items():
        if not isinstance(group, dict):
            continue
        if group.get("sensor_type") != sensor_type:
            continue
        for trait in group.get("traits", []):
            if crop_name not in trait.get("crops", []):
                continue
            results.append(
                {
                    "name": trait["name"],
                    "category": trait.get("category", ""),
                    "format": trait.get("format", ""),
                    "pipeline_group": group_key,
                    "ml_task": group.get("ml_task"),
                    "image_perspective": group.get("image_perspective"),
                }
            )
    return {"crop": crop_name, "sensor_type": sensor_type, "traits": results, "count": len(results)}


@mcp.tool()
def get_registry_summary() -> dict:
    """Get aggregate statistics about the trait registry."""
    reg = _load()
    crops: set[str] = set()
    tasks: set[str] = set()
    sensors: set[str] = set()
    perspectives: set[str] = set()
    categories: set[str] = set()
    total_traits = 0
    automatable_traits = 0

    for group_key, group in reg.items():
        if not isinstance(group, dict):
            continue
        has_pipeline = all(
            f in group
            for f in ("image_perspective", "sensor_type", "isolation_task", "ml_task")
        )
        if has_pipeline:
            tasks.add(group["ml_task"])
            sensors.add(group["sensor_type"])
            perspectives.add(group["image_perspective"])
        for trait in group.get("traits", []):
            total_traits += 1
            if has_pipeline:
                automatable_traits += 1
            for crop in trait.get("crops", []):
                crops.add(crop)
            categories.add(trait.get("category", "unknown"))

    return {
        "crops": sorted(crops),
        "crop_count": len(crops),
        "total_traits": total_traits,
        "automatable_traits": automatable_traits,
        "ml_tasks": sorted(tasks),
        "sensor_types": sorted(sensors),
        "perspectives": sorted(perspectives),
        "categories": sorted(categories),
    }
