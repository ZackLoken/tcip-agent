"""Verify that code examples in knowledge documents and source docstrings actually work.

A worked example that does not run is worse than no example: the agent follows it, gets a
TypeError, and learns to distrust the surface it came from. Both a skill table and a source
docstring can drift from the signature they describe, and neither is self-correcting, so this
checks them the only way that cannot go stale: against `inspect.signature` at import time.

What it checks, without executing anything:
  1. every example parses;
  2. every symbol it imports actually exists;
  3. every call to an imported symbol would bind against the real signature, which is what
     catches a documented call that omits a required keyword-only argument.

    python scripts/verify_doc_examples.py            # whole repo; non-zero exit on any problem
    python scripts/verify_doc_examples.py --list     # also list the examples that passed
"""
from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import re
import textwrap
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOTS = [REPO / "packages" / "tcip-mcp" / "src", REPO / "packages" / "tcip-annotation" / "src"]

_MD_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_PLACEHOLDER = object()


@dataclass
class Example:
    origin: str          # "file:line" where the block starts
    code: str


@dataclass
class Problem:
    origin: str
    kind: str            # syntax | missing-symbol | signature
    detail: str


def _md_examples(path: Path) -> list[Example]:
    text = path.read_text(encoding="utf-8")
    out = []
    for m in _MD_FENCE.finditer(text):
        line = text[: m.start()].count("\n") + 2  # +1 for the fence line itself
        out.append(Example(f"{path.relative_to(REPO)}:{line}", m.group(1)))
    return out


# A docstring literal block is only an example if it is code. A return-value shape or a prose
# paragraph is neither runnable nor wrong, and reporting one is a false alarm.
_CODE_LINE = re.compile(r"^\s*(?:from\s+\w|import\s+\w|\w[\w.]*\s*=\s*\w|\w[\w.]*\()")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _docstring_examples(path: Path) -> list[Example]:
    """reST literal blocks: a line ending in `::` followed by a more-indented run."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        doc = ast.get_docstring(node, clean=False) if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) else None
        if not doc or "::" not in doc:
            continue
        lines = doc.splitlines()
        i = 0
        while i < len(lines):
            if not lines[i].rstrip().endswith("::"):
                i += 1
                continue
            marker_indent = _indent(lines[i])
            i += 1
            while i < len(lines) and not lines[i].strip():  # skip the blank after ::
                i += 1
            if i >= len(lines):
                break
            # The literal block is the run indented deeper than the line introducing it;
            # it ends at the first non-blank line that dedents back to or past that marker.
            block_indent, block = _indent(lines[i]), []
            while i < len(lines):
                if not lines[i].strip():
                    block.append(lines[i])
                    i += 1
                    continue
                if _indent(lines[i]) < block_indent or _indent(lines[i]) <= marker_indent:
                    break
                block.append(lines[i])
                i += 1
            # dedent, not cleandoc: cleandoc treats the first line specially and would flatten
            # the block's relative indentation, breaking any suite (a `with`/`for`/`def` body).
            code = textwrap.dedent("\n".join(block)).strip()
            if code and any(_CODE_LINE.match(ln) for ln in code.splitlines()):
                start = getattr(node, "lineno", 1)
                out.append(Example(f"{path.relative_to(REPO)}:~{start}", code))
    return out


def _resolve_imports(tree: ast.AST) -> tuple[dict[str, object], list[str]]:
    """Map the names an example imports to real objects. Returns (resolved, missing)."""
    resolved: dict[str, object] = {}
    missing: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("tcip"):
            try:
                mod = importlib.import_module(node.module)
            except Exception as exc:  # noqa: BLE001 (any import failure is the finding)
                missing.append(f"cannot import {node.module}: {type(exc).__name__}: {exc}")
                continue
            for alias in node.names:
                if hasattr(mod, alias.name):
                    resolved[alias.asname or alias.name] = getattr(mod, alias.name)
                else:
                    missing.append(f"{node.module} has no attribute {alias.name!r}")
    return resolved, missing


def _dispatch_target(func: object, node: ast.Call) -> tuple[Callable[..., object], str] | None:
    """Resolve a name-dispatching factory to the builder it would actually call.

    ``build_detector("faster_rcnn", ...)`` accepts anything through ``**kwargs``, so binding
    against the factory proves nothing: the arity error surfaces inside the dispatched builder.
    Find the registry dict in the factory's module and bind against the real target instead.
    """
    if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
        return None
    key = node.args[0].value
    module = sys.modules.get(getattr(func, "__module__", ""))
    if module is None:
        return None
    for attr in vars(module).values():
        if isinstance(attr, dict) and key in attr and callable(attr.get(key)):
            return attr[key], key
    return None


def _check_calls(tree: ast.AST, resolved: dict[str, object]) -> list[str]:
    """Bind each call to an imported symbol against its real signature."""
    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        target = resolved.get(node.func.id)
        if target is None or not callable(target):
            continue
        # An example using *args/**kwargs tells us nothing about arity.
        if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
            continue
        try:
            sig = inspect.signature(target)
        except (TypeError, ValueError):
            continue
        args = [_PLACEHOLDER] * len(node.args)
        kwargs = {k.arg: _PLACEHOLDER for k in node.keywords if k.arg}
        try:
            sig.bind(*args, **kwargs)
        except TypeError as exc:
            problems.append(f"{node.func.id}(...) would not bind: {exc}")
            continue

        # The factory bound, but it may only forward **kwargs to a dispatched builder.
        dispatched = _dispatch_target(target, node)
        if dispatched is None:
            continue
        inner, key = dispatched
        try:
            inner_sig = inspect.signature(inner)
        except (TypeError, ValueError):
            continue
        try:  # drop the dispatch key itself; the builder receives the rest
            inner_sig.bind(*args[1:], **kwargs)
        except TypeError as exc:
            problems.append(f"{node.func.id}({key!r}, ...) reaches {inner.__name__}, which would not bind: {exc}")
    return problems


def check(example: Example) -> list[Problem]:
    try:
        tree = ast.parse(example.code)
    except SyntaxError as exc:
        return [Problem(example.origin, "syntax", f"{exc.msg} (line {exc.lineno})")]
    resolved, missing = _resolve_imports(tree)
    out = [Problem(example.origin, "missing-symbol", m) for m in missing]
    out += [Problem(example.origin, "signature", p) for p in _check_calls(tree, resolved)]
    return out


def collect() -> list[Example]:
    from tcip_mcp.knowledge import document_paths

    examples: list[Example] = []
    for md in sorted(document_paths()):
        examples += _md_examples(md)
    for root in SOURCE_ROOTS:
        for py in sorted(root.rglob("*.py")):
            if "__pycache__" in py.parts:
                continue
            examples += _docstring_examples(py)
    return examples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="also list examples that passed")
    args = ap.parse_args()

    examples = collect()
    problems: list[Problem] = []
    for ex in examples:
        found = check(ex)
        problems += found
        if args.list and not found:
            print(f"  ok   {ex.origin}")

    print(f"\nchecked {len(examples)} code examples in knowledge documents and source docstrings")
    if not problems:
        print("all examples parse, import, and bind against the real signatures")
        return 0
    print(f"{len(problems)} problem(s):\n")
    for p in problems:
        print(f"  [{p.kind}] {p.origin}\n      {p.detail}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
