"""The MCP server registers every tool with torch unimportable: a tool module that needs torch
imports it inside its own functions."""

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


def test_server_imports_with_torch_absent():
    """Every tool module registers even when torch cannot be imported at all.

    The server's own tool imports must never require torch at module load time; a tool
    module that needs torch imports it inside its own functions. Runs the check in a
    subprocess that blocks torch (and torchvision) from importing through a meta-path finder,
    since this process already has torch loaded once any other test has imported it.
    """
    assert importlib.util.find_spec("torch") is not None, (
        "torch must be installed in this test environment for this check to mean anything"
    )
    result = subprocess.run(
        [sys.executable, "-c", _BLOCK_TORCH_AND_IMPORT_SERVER],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) > 0


def test_block_torch_finder_actually_blocks_torch_import():
    """The meta-path finder the subprocess checks above install really prevents importing
    torch, rather than implementing the pre-3.4 find_module/load_module protocol Python 3.12
    never calls, which would leave torch importable while looking like it was blocked."""
    result = subprocess.run(
        [sys.executable, "-c", _BLOCK_TORCH_AND_TRY_IMPORT],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "blocked"
