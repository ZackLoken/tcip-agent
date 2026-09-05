"""Builds a module inventory and real import graph for the repo's Python and
TypeScript source trees.

Scope: packages/tcip-mcp/src, packages/tcip-annotation/src, packages/tcip-web/src,
packages/tcip-store/src, scripts/ (all Python, parsed with the ast module), and
packages/tcip-web/frontend/src (TypeScript, import statements extracted with a regex
parser since no TS compiler is invoked here).

Python import resolution: every .py file under the four package src/ roots gets a
dotted module name (packages/tcip-mcp/src/tcip_mcp/tools/foo.py -> tcip_mcp.tools.foo;
an __init__.py's dotted name is its containing package). scripts/ is not an installed
package; its modules are indexed both under a bare stem (foo.py -> "foo", the name
another script in scripts/ uses via `from foo import x` because scripts/ ends up on
sys.path when a script is run directly) and under "scripts.foo" (the dotted form a
module elsewhere in the repo would need). Both ast.Import and ast.ImportFrom
(including relative imports, resolved against the importing module's own package) are
walked; a `from pkg import name` is resolved to the submodule pkg.name when that
submodule exists in the index, else treated as an attribute pulled from pkg itself.
Only names that resolve to a file inside this repo are recorded as edges; stdlib and
third-party imports are not in-repo and are dropped.

TypeScript import resolution: regex extraction of `import ... from '...'`,
`export ... from '...'`, and dynamic `import('...')` specifiers. Relative specifiers
resolve against the importing file's directory; `@/...` resolves against
packages/tcip-web/frontend/src (the alias in tsconfig.json / vite.config.ts). Bare
package specifiers (react, zustand, ...) are external and dropped.

Tracked at scripts/build_module_inventory.py so `check_architecture_doc.py` can run it
fresh and compare its counts against ARCHITECTURE.md's tables. Run from anywhere; the
repo root is found by walking up from this file to the first ancestor containing a
.git directory. Writes JSON to --out when given (with a markdown twin beside it), or
prints the JSON to stdout when --out is omitted.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    cur = start
    while cur != cur.parent:
        if (cur / ".git").exists():
            return cur
        cur = cur.parent
    raise SystemExit("could not locate repo root (no .git ancestor found)")


REPO_ROOT = find_repo_root(SCRIPT_DIR)

PY_PACKAGE_ROOTS = [
    REPO_ROOT / "packages" / "tcip-mcp" / "src",
    REPO_ROOT / "packages" / "tcip-annotation" / "src",
    REPO_ROOT / "packages" / "tcip-web" / "src",
    REPO_ROOT / "packages" / "tcip-store" / "src",
]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
TS_ROOT = REPO_ROOT / "packages" / "tcip-web" / "frontend" / "src"

TS_EXTENSIONS = (".ts", ".tsx")


def rel(p: Path) -> str:
    return p.resolve().relative_to(REPO_ROOT).as_posix()


def iter_py_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out = []
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        out.append(p)
    return sorted(out)


def iter_ts_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out = []
    for p in root.rglob("*"):
        if p.suffix in TS_EXTENSIONS and p.is_file():
            out.append(p)
    return sorted(out)


class PyModule:
    def __init__(self, path: Path, dotted: str, is_init: bool, root_label: str):
        self.path = path
        self.dotted = dotted
        self.is_init = is_init
        self.root_label = root_label
        self.lines = 0
        self.owns: str | None = None
        self.owns_source: str | None = None  # "docstring" | "header_comment" | None
        self.parse_error: str | None = None
        self.imports: set[str] = set()  # repo-relative posix paths this module imports
        self.imported_by: set[str] = set()  # repo-relative posix paths that import this module


def py_dotted_name(file_path: Path, src_root: Path) -> tuple[str, bool]:
    relp = file_path.relative_to(src_root)
    parts = list(relp.parts)
    is_init = parts[-1] == "__init__.py"
    if is_init:
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(parts), is_init


def build_python_index():
    modules: dict[str, PyModule] = {}  # relpath -> PyModule
    dotted_index: dict[str, str] = {}  # dotted name -> relpath

    for src_root in PY_PACKAGE_ROOTS:
        label = src_root.parent.name  # e.g. "tcip-mcp"
        for f in iter_py_files(src_root):
            dotted, is_init = py_dotted_name(f, src_root)
            relpath = rel(f)
            modules[relpath] = PyModule(f, dotted, is_init, label)
            dotted_index[dotted] = relpath

    scripts_bare: dict[str, str] = {}
    for f in iter_py_files(SCRIPTS_ROOT):
        stem = f.stem
        relpath = rel(f)
        modules[relpath] = PyModule(f, stem, False, "scripts")
        scripts_bare[stem] = relpath
        dotted_index[f"scripts.{stem}"] = relpath

    return modules, dotted_index, scripts_bare


PY_HEADER_COMMENT_SKIP = re.compile(r"^#!|^#.*coding[:=]")


def first_sentence_of_first_paragraph(text: str) -> str | None:
    """Collapse the first paragraph's line-wraps and cut at the first sentence end."""
    text = text.strip()
    if not text:
        return None
    first_para = text.split("\n\n", 1)[0]
    collapsed = " ".join(first_para.split())
    m = re.search(r"(.*?[.!?])(\s|$)", collapsed)
    return m.group(1) if m else collapsed


def extract_py_owns(source: str, tree: ast.Module) -> tuple[str | None, str | None]:
    doc = ast.get_docstring(tree, clean=True)
    if doc:
        return first_sentence_of_first_paragraph(doc), "docstring"

    lines = source.splitlines()
    comment_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if comment_lines:
                break
            continue
        if stripped.startswith("#"):
            if PY_HEADER_COMMENT_SKIP.match(stripped) and not comment_lines:
                continue
            comment_lines.append(stripped.lstrip("#").strip())
        else:
            break
    if comment_lines:
        return first_sentence_of_first_paragraph(" ".join(comment_lines)), "header_comment"
    return None, None


def resolve_from_import(
    base_parts: list[str], alias_name: str, dotted_index: dict[str, str]
) -> str | None:
    """Resolve one name of a `from <base> import <alias_name>` to a relpath, if local."""
    candidate = ".".join(base_parts + [alias_name])
    if candidate in dotted_index:
        return dotted_index[candidate]
    base_dotted = ".".join(base_parts)
    if base_dotted in dotted_index:
        return dotted_index[base_dotted]
    return None


def parse_python_imports(
    pymod: PyModule, dotted_index: dict[str, str], scripts_bare: dict[str, str]
):
    try:
        source = pymod.path.read_text(encoding="utf-8")
    except OSError as e:
        pymod.parse_error = f"could not read file: {e}"
        return
    pymod.lines = len(source.splitlines())

    try:
        tree = ast.parse(source, filename=str(pymod.path))
    except SyntaxError as e:
        pymod.parse_error = f"SyntaxError: {e}"
        return

    pymod.owns, pymod.owns_source = extract_py_owns(source, tree)

    in_scripts = pymod.root_label == "scripts"
    self_parts = pymod.dotted.split(".")
    # a package's own __init__ counts as being inside that package for `.` resolution
    self_package_parts = self_parts if pymod.is_init else self_parts[:-1]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = dotted_index.get(alias.name)
                if target is None and in_scripts:
                    target = scripts_bare.get(alias.name)
                if target and target != rel(pymod.path):
                    pymod.imports.add(target)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                up = node.level - 1
                if up > len(self_package_parts):
                    continue  # climbs above repo root; unresolvable
                base_parts = self_package_parts[: len(self_package_parts) - up] if up else self_package_parts
                if node.module:
                    base_parts = base_parts + node.module.split(".")
                for alias in node.names:
                    target = resolve_from_import(base_parts, alias.name, dotted_index)
                    if target and target != rel(pymod.path):
                        pymod.imports.add(target)
            else:
                if not node.module:
                    continue
                base_parts = node.module.split(".")
                resolved_any = False
                for alias in node.names:
                    target = resolve_from_import(base_parts, alias.name, dotted_index)
                    if target and target != rel(pymod.path):
                        pymod.imports.add(target)
                        resolved_any = True
                if not resolved_any and in_scripts and node.module in scripts_bare:
                    target = scripts_bare[node.module]
                    if target != rel(pymod.path):
                        pymod.imports.add(target)


class TsModule:
    def __init__(self, path: Path):
        self.path = path
        self.lines = 0
        self.owns: str | None = None
        self.owns_source: str | None = None
        self.parse_error: str | None = None
        self.imports: set[str] = set()
        self.imported_by: set[str] = set()
        self.unresolved_local_specifiers: set[str] = set()


TS_FROM_CLAUSE = re.compile(
    r"(?:^|\n)\s*(?:import|export)\s+(?:type\s+)?[^;]*?\bfrom\s+['\"]([^'\"]+)['\"]"
)
TS_SIDE_EFFECT = re.compile(r"(?:^|\n)\s*import\s+['\"]([^'\"]+)['\"]")
TS_DYNAMIC = re.compile(r"\bimport\(\s*['\"]([^'\"]+)['\"]")
TS_BLOCK_COMMENT = re.compile(r"^\s*/\*\*?(.*?)\*/", re.DOTALL)


def extract_ts_specifiers(source: str) -> set[str]:
    specs: set[str] = set()
    for pat in (TS_FROM_CLAUSE, TS_SIDE_EFFECT, TS_DYNAMIC):
        for m in pat.finditer(source):
            specs.add(m.group(1))
    return specs


def extract_ts_owns(source: str) -> tuple[str | None, str | None]:
    m = TS_BLOCK_COMMENT.match(source)
    if m:
        body = m.group(1)
        cleaned_lines = []
        for line in body.splitlines():
            line = line.strip()
            line = re.sub(r"^\*\s?", "", line)
            if line:
                cleaned_lines.append(line)
        if cleaned_lines:
            return first_sentence_of_first_paragraph(" ".join(cleaned_lines)), "docstring"
    comment_lines: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped:
            if comment_lines:
                break
            continue
        if stripped.startswith("//"):
            comment_lines.append(stripped.lstrip("/").strip())
        else:
            break
    if comment_lines:
        return first_sentence_of_first_paragraph(" ".join(comment_lines)), "header_comment"
    return None, None


def resolve_ts_specifier(spec: str, importing_file: Path) -> str | None:
    if spec.startswith("."):
        base = (importing_file.parent / spec).resolve()
    elif spec.startswith("@/"):
        base = (TS_ROOT / spec[2:]).resolve()
    else:
        return None  # external package, not in-repo

    if base.suffix in TS_EXTENSIONS:
        candidates = [base]
    else:
        candidates = [base.with_name(base.name + ext) for ext in TS_EXTENSIONS]
        candidates += [base / f"index{ext}" for ext in TS_EXTENSIONS]

    for c in candidates:
        if c.is_file():
            try:
                return rel(c)
            except ValueError:
                return None
    return None


def parse_ts_imports(tsmod: TsModule):
    try:
        source = tsmod.path.read_text(encoding="utf-8")
    except OSError as e:
        tsmod.parse_error = f"could not read file: {e}"
        return
    tsmod.lines = len(source.splitlines())
    tsmod.owns, tsmod.owns_source = extract_ts_owns(source)

    for spec in extract_ts_specifiers(source):
        if not (spec.startswith(".") or spec.startswith("@/")):
            continue
        target = resolve_ts_specifier(spec, tsmod.path)
        if target:
            if target != rel(tsmod.path):
                tsmod.imports.add(target)
        else:
            tsmod.unresolved_local_specifiers.add(spec)


def build_inventory() -> dict:
    py_modules, dotted_index, scripts_bare = build_python_index()
    for pymod in py_modules.values():
        parse_python_imports(pymod, dotted_index, scripts_bare)
    for pymod in py_modules.values():
        for target in pymod.imports:
            if target in py_modules:
                py_modules[target].imported_by.add(rel(pymod.path))

    ts_files = iter_ts_files(TS_ROOT)
    ts_modules: dict[str, TsModule] = {}
    for f in ts_files:
        ts_modules[rel(f)] = TsModule(f)
    for tsmod in ts_modules.values():
        parse_ts_imports(tsmod)
    for tsmod in ts_modules.values():
        for target in tsmod.imports:
            if target in ts_modules:
                ts_modules[target].imported_by.add(rel(tsmod.path))

    py_records = []
    for relpath, m in sorted(py_modules.items()):
        py_records.append(
            {
                "path": relpath,
                "root": m.root_label,
                "lines": m.lines,
                "owns": m.owns,
                "owns_source": m.owns_source,
                "dotted_name": m.dotted,
                "parse_error": m.parse_error,
                "imports": sorted(m.imports),
                "imported_by": sorted(m.imported_by),
                "imported_by_count": len(m.imported_by),
            }
        )

    ts_records = []
    for relpath, m in sorted(ts_modules.items()):
        ts_records.append(
            {
                "path": relpath,
                "root": "tcip-web-frontend",
                "lines": m.lines,
                "owns": m.owns,
                "owns_source": m.owns_source,
                "parse_error": m.parse_error,
                "imports": sorted(m.imports),
                "imported_by": sorted(m.imported_by),
                "imported_by_count": len(m.imported_by),
                "unresolved_local_specifiers": sorted(m.unresolved_local_specifiers),
            }
        )

    parse_errors = [r for r in py_records + ts_records if r["parse_error"]]

    return {
        "repo_root": str(REPO_ROOT),
        "python_modules": py_records,
        "typescript_modules": ts_records,
        "counts": {
            "python_total": len(py_records),
            "python_by_root": {
                label: sum(1 for r in py_records if r["root"] == label)
                for label in sorted({r["root"] for r in py_records})
            },
            "typescript_total": len(ts_records),
            "parse_errors": len(parse_errors),
        },
    }


def render_markdown(out: dict) -> str:
    lines = []
    lines.append("# Module inventory and import graph")
    lines.append("")
    lines.append(
        "Generated by `scripts/build_module_inventory.py`. Python imports are resolved by "
        "parsing each file's AST (ast.Import / ast.ImportFrom, including relative imports) "
        "against a dotted-name index built from every .py file under the four package src/ "
        "roots and scripts/. TypeScript imports are extracted with a regex parser over "
        "`import`/`export ... from` and dynamic `import()` specifiers, resolved against "
        "relative paths and the `@/` -> packages/tcip-web/frontend/src alias. Only "
        "specifiers that resolve to a file in this repo are recorded as edges; "
        "standard-library, third-party, and CSS/asset imports are excluded."
    )
    lines.append("")
    counts = out["counts"]
    lines.append(
        f"Python modules: {counts['python_total']} "
        f"({', '.join(f'{k}: {v}' for k, v in counts['python_by_root'].items())}). "
        f"TypeScript modules: {counts['typescript_total']}. "
        f"Parse errors: {counts['parse_errors']}."
    )
    lines.append("")

    def emit_table(title: str, records: list[dict]):
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Path | Lines | Owns | Imports (in-repo) | Imported by (count) |")
        lines.append("|---|---|---|---|---|")
        for r in records:
            owns = r["owns"] or "(none found)"
            owns = owns.replace("|", "\\|")
            imports = "; ".join(r["imports"]) if r["imports"] else "(none)"
            imports = imports.replace("|", "\\|")
            path = r["path"].replace("|", "\\|")
            lines.append(f"| {path} | {r['lines']} | {owns} | {imports} | {r['imported_by_count']} |")
        lines.append("")

    for label in sorted(counts["python_by_root"]):
        records = [r for r in out["python_modules"] if r["root"] == label]
        emit_table(f"Python: {label}", records)

    emit_table("TypeScript: tcip-web frontend", out["typescript_modules"])

    parse_errors = [
        r for r in out["python_modules"] + out["typescript_modules"] if r["parse_error"]
    ]
    lines.append("## Modules that could not be parsed")
    lines.append("")
    if parse_errors:
        for r in parse_errors:
            lines.append(f"- `{r['path']}`: {r['parse_error']}")
    else:
        lines.append("None. Every discovered .py, .ts, and .tsx file under the scoped roots parsed.")
    lines.append("")

    unresolved = [
        (r["path"], r["unresolved_local_specifiers"])
        for r in out["typescript_modules"]
        if r.get("unresolved_local_specifiers")
    ]
    lines.append("## TypeScript local-looking specifiers that did not resolve to a file")
    lines.append("")
    lines.append(
        "These matched the regex as relative (`.`/`..`) or `@/`-aliased specifiers but no "
        "file on disk matched after trying the .ts/.tsx/index extensions the script tried. "
        "Recorded rather than silently dropped."
    )
    lines.append("")
    if unresolved:
        for path, specs in unresolved:
            lines.append(f"- `{path}`: {', '.join(specs)}")
    else:
        lines.append("None.")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=None,
        help="Write the inventory JSON here (and a markdown twin beside it, same stem, "
        ".md suffix). Omit to print the JSON to stdout.",
    )
    args = ap.parse_args()

    out = build_inventory()
    py_records = out["python_modules"]
    ts_records = out["typescript_modules"]
    parse_errors = out["counts"]["parse_errors"]

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        out_path.with_suffix(".md").write_text(render_markdown(out), encoding="utf-8")
        print(
            f"wrote {out_path} and {out_path.with_suffix('.md')}: "
            f"python modules: {len(py_records)}, typescript modules: {len(ts_records)}, "
            f"parse errors: {parse_errors}",
            file=sys.stderr,
        )
    else:
        print(json.dumps(out, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
