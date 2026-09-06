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

    MCPServer registers a tool under its function name by default, and every tool in this repo
    uses a bare `@mcp.tool()` except `serve_domain_knowledge`, whose description is composed from the
    knowledge documents at import time; either form is still matched by `target.attr == "tool"`,
    so the function name is the tool name regardless.
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
        "draw_splits", "focus_human_attention", "get_experiment",
        "register_model", "load_project_memory", "read_audit_log",
        # Renamed tools: the new names must register.
        "inspect_project",
        "capture_live_canvas",
        # Renamed tools: the new names must register.
        "rank_registered_models",
        "deliver_per_image_counts", "view_gui_state",
        # The general per-plant CSV door, over a caller's own aggregation and mapping.
        "deliver_per_plant_csv",
        # Method-neutral auto-labeling seam: no longer SAM-specific names.
        "propose_annotations", "stage_proposals", "segment_prompt",
        # Re-admitted count calibrator, beside its two sibling calibrators.
        "calibrate_count_operating_point",
    ):
        assert present in registered, f"{present} should be registered"
    removed = {
        # Demoted to scripts under the admission standard (packages/tcip-mcp/CLAUDE.md): each
        # function stays importable, only the tool registration is gone.
        "scan_dataset", "inspect_compute_resources", "render_failure_cases", "archive_project",
        "import_project",
        "evaluate_detections", "evaluate_dataset", "split_dataset",
        "log_metrics", "record_artifact", "get_training_metrics_path",
        # collapse to model_source-only: the menu/composer/spec tools are gone.
        "recommend_model", "list_components", "validate_model_spec",
        "validate_pipeline_spec", "compose_and_summarize",
        "get_worst_predictions", "run_pipeline",
        # focus_annotate + focus_review merged into focus_human_attention(tab=).
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
        # Renamed: the docstring stopped denying its own verb under the old name.
        "accept_proposals",
        # Renamed tools: the old names must no longer register.
        "focus", "make_splits", "tabulate_counts", "select_best_model",
        "calibrate_ordinal_regression_operating_point", "cancel_hpo", "run_hpo",
        "claude_reports", "project_retrospective", "domain_knowledge",
        "force_redraw_cal_holdout_split", "push_panel_data", "update_trait_spec_fields",
        "compute_phenology", "check_training_status", "init_project", "set_active_project",
        # Merged away: list_experiments(launched_only=True) serves the launched-runs view.
        "list_training_runs",
        # Merged away: rank_registered_models(metric="") serves the listing view.
        "list_registered_models",
        # Merged away: stage_proposals(assignments=...) serves the accepted-candidates regime.
        "stage_accepted_proposals",
        # Demoted to a library call plus scripts/preflight_config.py: launch_training calls the
        # function directly, no web route calls it.
        "preflight_config",
        # Demoted to a library call: the agent reads a label file through it directly.
        "read_annotations",
        # Demoted to a library call plus scripts/score_predictions.py.
        "score_predictions",
        # Demoted to a library call plus scripts/triage_predictions.py.
        "triage_predictions",
        # Demoted to a library call: the web compare route calls the function directly.
        "compare_experiments",
        # Demoted to a library call plus scripts/overlay_reference_grid.py; kept @audited.
        "overlay_reference_grid",
        # Demoted to a library call plus scripts/visualize.py; kept @audited.
        "visualize",
        # Folded into scripts/doctor.py's check_data_quality; its own function is deleted.
        "validate_data_quality",
        # Merged into run_inference, which persists the bucket both doors used to.
        "export_predictions",
    }
    assert not (removed & registered), f"removed tools still registered: {removed & registered}"


def test_docs_do_not_hardcode_tool_count():
    """Docs must point at tools/list_tools.py, not cite a literal count.

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
        "remove the number and reference `python tools/list_tools.py` instead"
    )
