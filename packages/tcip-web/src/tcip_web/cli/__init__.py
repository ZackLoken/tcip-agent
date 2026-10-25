"""``tcip``: the operator console command, dispatching to one subcommand per command.

An operator-only command is its own module (:data:`COMMANDS`), exposing ``main(argv)`` and
returning the exit code; every registered MCP tool is the command of its own name
(:func:`tcip_mcp.cli.hyphenated`), run by :func:`tcip_mcp.cli.run_tool`, which builds the one
server the invocation lists and calls. The dispatcher only routes.

Run as ``python -m tcip_web.cli <command> [args...]`` or, once installed, as ``tcip <command>
[args...]``.
"""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING

from tcip_mcp.cli import hyphenated

if TYPE_CHECKING:
    from mcp.types import Tool

_OPERATOR_MODULES = (
    "tcip_mcp.cli.dump_store",
    "tcip_mcp.cli.doctor",
    "tcip_mcp.cli.archive_project",
    "tcip_mcp.cli.import_project",
    "tcip_mcp.cli.write_project_record",
    "tcip_mcp.cli.distill_learnings",
    "tcip_web.cli.report_conversation",
)

COMMANDS: dict[str, str] = {
    hyphenated(module.rpartition(".")[2]): module
    for module in _OPERATOR_MODULES
}
"""Each operator-only command, its module's own final name :func:`hyphenated`, to the module
exposing its ``main(argv)``."""


def tool_listing() -> list[Tool]:
    """Return the tools a server for no project and no person lists."""
    import anyio

    from tcip_mcp.server import Session, build_server

    return anyio.run(build_server(Session()).list_tools)


def _commands(listing: list[Tool]) -> dict[str, tuple[str, str | None]]:
    """Return every command the dispatcher runs, sorted, to what runs it and what it does.

    An operator command maps to its module and the module's docstring; a tool's command, one per
    tool of ``listing``, to the tool's name and the description the server lists it with.
    """
    listed = {
        hyphenated(tool.name): (tool.name, tool.description)
        for tool in listing
    }
    operator = {
        name: (module, importlib.import_module(module).__doc__)
        for name, module in COMMANDS.items()
    }
    return dict(sorted((listed | operator).items()))


def command_names(listing: list[Tool]) -> list[str]:
    """Return every command the dispatcher runs, each operator-only one and each tool's, sorted.

    The tools are those of ``listing``, a server's listing.
    """
    return list(_commands(listing))


def _usage(listing: list[Tool]) -> str:
    """Return the usage: every command beside the first line of what it states it does."""
    said = {name: doc for name, (_runs, doc) in _commands(listing).items()}
    width = max(map(len, said))
    lines = "\n".join(
        f"  {name:<{width}}  {(doc or '').strip().partition(chr(10))[0]}"
        for name, doc in said.items()
    )
    return f"usage: tcip <command> [args...]\n\ncommands:\n{lines}\n"


def main(argv: list[str] | None = None) -> int:
    """Route ``argv`` to its command and return that exit code.

    ``argv`` defaults to ``sys.argv[1:]``. No command prints the usage and returns 2, ``-h`` or
    ``--help`` prints it and returns 0, and an unknown command prints its name and the usage to
    stderr and returns 2.
    """
    args = sys.argv[1:] if argv is None else list(argv)
    if not args or args[0] in ("-h", "--help"):
        print(_usage(tool_listing()))
        return 0 if args and args[0] in ("-h", "--help") else 2
    command, rest = args[0], args[1:]
    prog = f"tcip {command}"
    if command in COMMANDS:
        return importlib.import_module(COMMANDS[command]).main(rest, prog=prog)
    from tcip_mcp.cli import run_tool

    def unknown(listing: list[Tool]) -> int:
        """Print that no command is ``command``, beside the usage ``listing`` gives; return 2."""
        print(
            f"tcip: unknown command {command!r}\n\n{_usage(listing)}",
            file=sys.stderr,
        )
        return 2

    return run_tool(command, rest, prog=prog, unknown=unknown)


if __name__ == "__main__":
    sys.exit(main())
