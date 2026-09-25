"""``tcip``: the operator console command, dispatching to one subcommand per operator command.

Each subcommand's module (``tcip_mcp.cli.<name>``, or ``distill-learnings`` in this package)
exposes ``main(argv)``, returning the exit code; the dispatcher only routes.

Run as ``python -m tcip_web.cli <command> [args...]`` or, once installed, as ``tcip <command>
[args...]``.
"""

from __future__ import annotations

import importlib
import sys

COMMANDS: dict[str, str] = {
    "adopt-store": "tcip_mcp.cli.adopt_store",
    "export-store": "tcip_mcp.cli.export_store",
    "doctor": "tcip_mcp.cli.doctor",
    "archive-project": "tcip_mcp.cli.archive_project",
    "import-project": "tcip_mcp.cli.import_project",
    "complete-removals": "tcip_mcp.cli.complete_removals",
    "complete-renames": "tcip_mcp.cli.complete_renames",
    "calibrate-operating-point": "tcip_mcp.cli.calibrate_operating_point",
    "check-dataset-identity": "tcip_mcp.cli.check_dataset_identity",
    "write-project-site": "tcip_mcp.cli.write_project_site",
    "distill-learnings": "tcip_web.cli.distill_learnings",
    "scan-dataset": "tcip_mcp.cli.scan_dataset",
    "score-predictions": "tcip_mcp.cli.score_predictions",
    "triage-predictions": "tcip_mcp.cli.triage_predictions",
    "overlay-reference-grid": "tcip_mcp.cli.overlay_reference_grid",
    "visualize": "tcip_mcp.cli.visualize",
    "render-failure-cases": "tcip_mcp.cli.render_failure_cases",
    "preflight-config": "tcip_mcp.cli.preflight_config",
    "inspect-compute-resources": "tcip_mcp.cli.inspect_compute_resources",
    "plant-aware-group-splits": "tcip_mcp.cli.plant_aware_group_splits",
    "shp-to-plant-csv": "tcip_mcp.cli.shp_to_plant_csv",
}
"""Command name (as typed after ``tcip``) to the module exposing its ``main(argv)``. A command
name is that module's own final name with underscores respelled as hyphens."""


def _usage() -> str:
    names = "\n".join(f"  {name}" for name in sorted(COMMANDS))
    return f"usage: tcip <command> [args...]\n\ncommands:\n{names}\n"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if not args or args[0] in ("-h", "--help"):
        print(_usage())
        return 0 if args and args[0] in ("-h", "--help") else 2
    command, rest = args[0], args[1:]
    module_name = COMMANDS.get(command)
    if module_name is None:
        print(f"tcip: unknown command {command!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(module_name)
    return module.main(rest, prog=f"tcip {command}")


if __name__ == "__main__":
    sys.exit(main())
