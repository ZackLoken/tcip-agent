"""Core runtime dependencies import, and an absent one fails its test.

These tests fail, not skip, on a missing dependency.
"""

import importlib

import pytest

# Declared (non-optional) tcip-mcp dependencies; see packages/tcip-mcp/pyproject.toml.
CORE_DEPS = [
    "mcp",
    "torch",
    "torchvision",
    "timm",
    "pycocotools",
    "pydantic",
    "numpy",
    "PIL",
    "shapely",
    "yaml",
    "rasterio",
    "fiona",
]


@pytest.mark.parametrize("module_name", CORE_DEPS)
def test_core_dependency_importable(module_name: str) -> None:
    importlib.import_module(module_name)
