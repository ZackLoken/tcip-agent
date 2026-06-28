"""CI trust: core runtime dependencies must be importable, not silently skipped.

If any of these is missing, the bulk of the suite would module-level ``importorskip``
while CI still reported green (a smaller-than-expected suite passing). These tests FAIL
(not skip) so a broken/incomplete environment is loud. The set mirrors tcip-mcp's declared
dependencies — anything CI's ``pip install -e tcip-mcp[dev]`` is expected to provide.
"""

import importlib

import pytest

# Declared (non-optional) tcip-mcp dependencies — see packages/tcip-mcp/pyproject.toml.
CORE_DEPS = [
    "mcp", "torch", "torchvision", "timm", "pycocotools",
    "pydantic", "numpy", "PIL", "shapely", "yaml",
]


@pytest.mark.parametrize("module_name", CORE_DEPS)
def test_core_dependency_importable(module_name):
    importlib.import_module(module_name)
