"""Guards against MCP tool-registry drift.

Two failure modes this catches:
  1. A tool that is decorated `@mcp.tool()` but never actually registers (import
     error, decorator mistake, duplicate name).
  2. Docs that hard-code a tool count which then goes stale (the original sin:
     README said 54, copilot-instructions said 57, reality was 57).
"""

from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "packages" / "tcip-mcp" / "src" / "tcip_mcp" / "tools"


def _decorated_tool_names() -> set[str]:
    """Function names decorated with `@mcp.tool(...)` across tools/*.py (via AST).

    MCPServer registers a tool under its function name by default, and every tool in
    this repo uses a bare `@mcp.tool()`, so the function name is the tool name.
    """
    names: set[str] = set()
    for py in TOOLS_DIR.glob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                if isinstance(target, ast.Attribute) and target.attr == "tool":
                    names.add(node.name)
    return names


def test_decorated_tools_exist():
    """Sanity: the AST actually finds the decorated tools."""
    assert len(_decorated_tool_names()) > 0


def test_registered_tool_names_unique():
    from tcip_mcp.server import list_registered_tools

    names = list_registered_tools()
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f"duplicate MCP tool names registered: {dupes}"


def test_every_decorated_tool_registers():
    """Registry must match the decorated functions exactly, torch installed or not: a tool
    module that needs torch imports it inside its own functions rather than at module level."""
    from tcip_mcp.server import list_registered_tools

    registered = set(list_registered_tools())
    decorated = _decorated_tool_names()
    missing = decorated - registered
    extra = registered - decorated
    assert not missing and not extra, f"missing from registry={missing}, unexpected={extra}"


_BLOCK_TORCH = """
class _BlockTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch.") or name == "torchvision" or name.startswith("torchvision."):
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


def test_consolidated_tools_present_and_removed_absent():
    """Merged tools register and the tools they replaced do not.

    An explicit presence/absence check (not a hard-coded count) so the eval-on-disk /
    splits merges and the machine-plumbing de-registrations stay put.
    """
    from tcip_mcp.server import list_registered_tools

    registered = set(list_registered_tools())
    for present in (
        "make_splits", "focus", "get_experiment",
        "register_model", "load_project_memory",
        # Renamed tools: the new names must register.
        "archive_project", "inspect_project", "scan_dataset", "read_annotations",
        "render_failure_cases", "overlay_reference_grid", "capture_live_canvas",
        # Renamed tools: the new names must register.
        "preflight_config", "select_best_model", "score_predictions",
        "tabulate_counts", "view_gui_state",
        # Method-neutral auto-labeling seam: no longer SAM-specific names.
        "propose_annotations", "accept_proposals", "segment_prompt",
    ):
        assert present in registered, f"{present} should be registered"
    removed = {
        "evaluate_detections", "evaluate_dataset", "split_dataset",
        "log_metrics", "record_artifact", "get_training_metrics_path",
        # collapse to model_source-only: the menu/composer/spec tools are gone.
        "recommend_model", "list_components", "validate_model_spec",
        "validate_pipeline_spec", "compose_and_summarize",
        "get_worst_predictions", "run_pipeline",
        # focus_annotate + focus_review merged into focus(tab=).
        "focus_annotate", "focus_review",
        # get_experiment_lineage merged into get_experiment(view='lineage').
        "get_experiment_lineage",
        # register_model_from_experiment merged into register_model(experiment_id=).
        "register_model_from_experiment",
        # load_reports + load_retrospectives merged into load_project_memory(kind=).
        "load_reports", "load_retrospectives",
        # Old names: must no longer register.
        "export_project", "get_project_status", "load_dataset", "load_annotations",
        "visualize_worst_predictions", "visualize_grid_overlay", "visualize_canvas",
        "sam_auto_label",
        # The old SAM-hardcoded names must no longer register.
        "generate_mask_candidates", "accept_candidates", "sam_predict",
        # Old names: must no longer register.
        "validate_config", "get_best_model", "evaluate_predictions",
        "export_results_csv", "get_active_context",
    }
    assert not (removed & registered), f"removed tools still registered: {removed & registered}"


def test_docs_do_not_hardcode_tool_count():
    """Docs must point at scripts/list_tools.py, not cite a literal count.

    The regex allows an optional run of adjectives before "tool(s)" so the
    project's idiomatic phrasings are all caught: "57 MCP tools",
    "54 domain tools", "56 specialized tools", "57 total tools", bare "56 tools".
    It does not match prose like "all MCP tool calls" (no leading number).
    """
    pattern = re.compile(r"\b\d+\s+(?:[A-Za-z][A-Za-z-]*\s+){0,2}tools?\b", re.IGNORECASE)
    docs = [REPO_ROOT / "README.md", REPO_ROOT / "CLAUDE.md"]
    docs += sorted((REPO_ROOT / ".github").rglob("*.md"))
    offenders = {}
    for doc in docs:
        if not doc.exists():
            continue
        hits = pattern.findall(doc.read_text(encoding="utf-8"))
        if hits:
            offenders[str(doc.relative_to(REPO_ROOT))] = hits
    assert not offenders, (
        f"docs hard-code a tool count {offenders}; "
        "remove the number and reference `python scripts/list_tools.py` instead"
    )
