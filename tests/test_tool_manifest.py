"""The MCP server imports and registers its tools with torch unimportable."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

_BLOCK_TORCH = """
class _BlockTorch:
    def find_spec(self, name, path=None, target=None):
        if (name == "torch" or name.startswith("torch.")
                or name == "torchvision" or name.startswith("torchvision.")):
            raise ImportError(f"torch blocked for this check: {name}")
        return None
"""

_BLOCK_TORCH_AND_IMPORT_SERVER = f"""
import sys

{_BLOCK_TORCH}
sys.meta_path.insert(0, _BlockTorch())
import tcip_mcp.server
assert "torch" not in sys.modules, "importing the server pulled torch into sys.modules"
print(len(tcip_mcp.server.list_registered_tools()))
"""

_BLOCK_TORCH_AND_TRY_IMPORT = f"""
import sys

{_BLOCK_TORCH}
sys.meta_path.insert(0, _BlockTorch())
try:
    import torch
except ImportError:
    print("blocked")
else:
    print("not blocked")
"""


def test_server_imports_with_torch_absent() -> None:
    """The MCP server imports and registers tools while torch imports are blocked.

    The check runs in a subprocess whose meta-path finder blocks torch and torchvision from
    importing.
    """
    assert importlib.util.find_spec("torch") is not None, (
        "torch must be installed in this test environment for this check to mean anything"
    )
    result = subprocess.run(
        [sys.executable, "-c", _BLOCK_TORCH_AND_IMPORT_SERVER],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) > 0


def test_block_torch_finder_actually_blocks_torch_import() -> None:
    """The meta-path finder the subprocess checks install really prevents importing torch."""
    result = subprocess.run(
        [sys.executable, "-c", _BLOCK_TORCH_AND_TRY_IMPORT],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "blocked"
