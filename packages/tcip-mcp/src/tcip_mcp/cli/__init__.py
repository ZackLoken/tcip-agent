"""Operator command sub-package: one module per operator-only ``tcip`` subcommand, and one adapter.

Each operator module exposes ``main(argv)``, which returns the exit code; :func:`run_tool` runs
any registered MCP tool as the command of its own name.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.types import Tool


def bound_project(value: str) -> Path:
    """Bind this process's store and return the project at ``value``, resolved.

    Exits naming the path when it holds no readable project record.
    """
    from tcip_mcp.project_record import existing_project
    from tcip_store import bind

    bind()
    try:
        return existing_project(value)[0]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def hyphenated(name: str) -> str:
    """Return ``name`` as the console spells a command or an option: underscores as hyphens."""
    return name.replace("_", "-")


def run_tool(
    command: str,
    argv: list[str],
    *,
    prog: str,
    unknown: Callable[[list[Tool]], int],
) -> int:
    """Call the registered tool whose command is ``command`` once and print its answer.

    The answer is printed as an MCP client reads it. One server answers the invocation
    (:func:`~tcip_mcp.server.server_for`): its listing names the tool and its schema, and when no
    tool is ``command`` the answer is ``unknown`` given that listing. The whole of ``argv`` is
    parsed once, ``--help`` printing before ``--user`` is required, and the server's session is
    then bound to ``--project`` and to ``--user``, the person the call is made by
    (:func:`~tcip_mcp.identity.actor`), whom it gives every tool taking ``actor``. Each argument
    of the tool's advertised input schema is
    stated ``--<argument> VALUE``, the argument :func:`hyphenated`: its JSON value, or the text
    itself where the schema admits a string; an argument stated more than once takes the list of
    its values. A value ``@<path>`` is always the contents of the file at that path, located
    against ``--project``, so a value that itself begins with ``@`` is given through a file
    holding it. The server alone admits the arguments, and each text block of its answer is
    printed as it renders it (a JSON document for an object answer, one line per item for a
    list). Returns 0 once answered, the tool's own ``error`` included, and 2 when the call is
    refused: no ``--user`` or one naming no one, a ``@<path>`` that cannot be read, an argument
    the schema admits no value for, or a project tool with no ``--project``.
    """
    import anyio
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import CallToolResult, TextContent

    from tcip_mcp.identity import NoActorError, actor
    from tcip_mcp.registry_paths import located
    from tcip_mcp.server import Session, project_binding, server_for

    session = Session()
    server = server_for(session)
    listing = anyio.run(server.list_tools)
    tools = {hyphenated(t.name): t for t in listing}
    if command not in tools:
        return unknown(listing)
    listed = tools[command]
    parser = argparse.ArgumentParser(
        prog=prog,
        description=listed.description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--project", default=None, help="the project the tool acts on"
    )
    parser.add_argument(
        "--user", required=True, help="the person this call is made by"
    )
    schema = listed.input_schema
    for argument, spec in schema["properties"].items():
        parser.add_argument(
            f"--{hyphenated(argument)}",
            dest=argument,
            action="append",
            help=json.dumps(spec),
        )
    stated = vars(parser.parse_args(argv))
    project = stated.pop("project")
    try:
        person = actor(stated.pop("user"))
    except NoActorError as exc:
        print(f"{prog}: --user: {exc}", file=sys.stderr)
        return 2
    session.bind(project_binding(project), person)

    def admits_text(spec: dict) -> bool:
        """Return whether ``spec`` (or one of its alternatives) admits a string."""
        return any(
            schema.get("$defs", {})
            .get(s.get("$ref", "").rpartition("/")[2], s)
            .get("type")
            == "string"
            for s in (spec, *spec.get("anyOf", ()))
        )

    def value(text: str, spec: dict) -> object:
        """Return ``text`` as the transport value ``spec`` reads: itself, or its JSON."""
        if text.startswith("@"):
            text = located(text[1:], project).read_text(encoding="utf-8")
        if admits_text(spec):
            return text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    try:
        arguments = {
            argument: value(texts[0], schema["properties"][argument])
            if len(texts) == 1
            else [
                value(t, schema["properties"][argument].get("items", {}))
                for t in texts
            ]
            for argument, texts in stated.items()
        }
    except (OSError, ValueError) as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        return 2
    try:
        result = anyio.run(server.call_tool, listed.name, arguments)
    except ToolError as exc:
        print(exc, file=sys.stderr)
        return 2
    assert isinstance(result, CallToolResult), "no tool asks for client input"
    for block in result.content:
        assert isinstance(block, TextContent), "every tool answers in text"
        print(block.text)
    return 0
