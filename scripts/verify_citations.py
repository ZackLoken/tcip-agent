"""Check that literature citations point at real code, real papers, and real sentences.

A fabricated citation is worse than no citation: it manufactures authority a reviewer will
trust. Language models invent plausible DOIs and paraphrase a paper into a quote that never
appears in it, and neither failure is visible by reading. So every check here is mechanical.

Four things are asserted. Each reference names a paper that can be retrieved (a DOI or a stable
URL). Each anchor resolves to a symbol that still exists at the path given, so a rename breaks
loudly instead of rotting silently. Each `# cite:` marker in the codebase resolves to a
reference, and each reference is reachable from at least one marker. And every quote appears
verbatim in the text of the stored PDF, which is the check an invented quote cannot survive.

Anchors are `path::symbol`, never line numbers, because line numbers are wrong within a week and
a rotted anchor cannot be told apart from a wrong one. Nested symbols use `path::Class.method`.

    python scripts/verify_citations.py              # skip quote checks for absent PDFs
    python scripts/verify_citations.py --require-pdf  # absent PDF is a failure

PDFs are not tracked: redistributing paywalled papers is infringement, and they are large. So the
quote check runs for whoever has fetched them and CI checks everything else. Record the retrieval
URL so fetching is reproducible.
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REFERENCES = REPO_ROOT / "references"

REQUIRED_TOP = ("key", "title", "year", "supports")
REQUIRED_SUPPORT = ("anchor", "decision", "quote")

CITE_MARKER = re.compile(r"#\s*cite:\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
TS_SYMBOL = re.compile(
    r"\b(?:function|class|const|let|var|interface|type|enum)\s+{name}\b|"
    r"\b{name}\s*[:=]\s*(?:\(|function|async|=>)"
)


def load_yaml():
    try:
        import yaml
    except ImportError:
        sys.exit("pyyaml is required. It reads crops.yml too, so it should already be present.")
    return yaml


def read_pdf_text(path: pathlib.Path) -> str | None:
    try:
        import pypdf
    except ImportError:
        sys.exit(
            "pypdf is required for the quote check. It is currently an undeclared transitive "
            "dependency; add it to environment.yml before relying on this script."
        )
    try:
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:  # pypdf raises a wide range on malformed files
        print(f"    could not read {path.name}: {exc}")
        return None


def normalize(text: str) -> str:
    """Collapse the line breaks and hyphenation PDF extraction inserts mid-sentence."""
    text = text.replace("­", "")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    return re.sub(r"\s+", " ", text).strip().lower()


def parse_reference(path: pathlib.Path, yaml) -> dict | None:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        print(f"  {path.name}: no YAML frontmatter")
        return None
    _, _, rest = raw.partition("---")
    front, sep, _body = rest.partition("\n---")
    if not sep:
        print(f"  {path.name}: frontmatter not closed")
        return None
    try:
        data = yaml.safe_load(front)
    except Exception as exc:
        print(f"  {path.name}: frontmatter does not parse: {exc}")
        return None
    if not isinstance(data, dict):
        print(f"  {path.name}: frontmatter is not a mapping")
        return None
    data["_path"] = path
    return data


def python_symbol_exists(tree: ast.AST, dotted: str) -> bool:
    parts = dotted.split(".")

    def walk(node, remaining):
        head, rest = remaining[0], remaining[1:]
        for child in ast.iter_child_nodes(node):
            name = getattr(child, "name", None)
            if name is None and isinstance(child, (ast.Assign, ast.AnnAssign)):
                targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                if any(isinstance(t, ast.Name) and t.id == head for t in targets):
                    return not rest
                continue
            if name == head:
                return True if not rest else walk(child, rest)
        return False

    return walk(tree, parts)


def anchor_resolves(anchor: str) -> tuple[bool, str]:
    if "::" not in anchor:
        return False, "anchor is not path::symbol"
    rel, symbol = anchor.split("::", 1)
    target = REPO_ROOT / rel
    if not target.exists():
        return False, f"file not found: {rel}"
    if target.suffix == ".py":
        try:
            tree = ast.parse(target.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            return False, f"file does not parse: {exc}"
        if python_symbol_exists(tree, symbol):
            return True, ""
        return False, f"symbol not found: {symbol}"
    if target.suffix in (".ts", ".tsx"):
        leaf = symbol.split(".")[-1]
        pattern = TS_SYMBOL.pattern.replace("{name}", re.escape(leaf))
        if re.search(pattern, target.read_text(encoding="utf-8")):
            return True, ""
        return False, f"symbol not found: {symbol}"
    return True, ""


def collect_markers() -> dict[str, list[str]]:
    """Every `# cite: key` in tracked source, mapped key to the places it appears."""
    found: dict[str, list[str]] = {}
    for pattern in ("packages/**/*.py", "scripts/*.py", "packages/**/*.ts", "packages/**/*.tsx"):
        for path in REPO_ROOT.glob(pattern):
            if "node_modules" in path.parts or path.name == "verify_citations.py":
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8",
                                                         errors="replace").splitlines(), 1):
                match = CITE_MARKER.search(line)
                if match:
                    rel = path.relative_to(REPO_ROOT).as_posix()
                    found.setdefault(match.group(1), []).append(f"{rel}:{lineno}")
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--require-pdf", action="store_true",
                    help="treat an absent PDF as a failure rather than a skipped quote check")
    args = ap.parse_args()

    if not REFERENCES.is_dir():
        print(f"no references directory at {REFERENCES}")
        return 0

    yaml = load_yaml()
    files = sorted(p for p in REFERENCES.glob("*.md")
                   if not p.name.startswith("_") and p.name != "README.md")
    if not files:
        print("no reference files yet")
        return 0

    problems: list[str] = []
    skipped: list[str] = []
    checked_quotes = 0
    keys: set[str] = set()

    for path in files:
        ref = parse_reference(path, yaml)
        if ref is None:
            problems.append(f"{path.name}: unreadable")
            continue

        missing = [f for f in REQUIRED_TOP if not ref.get(f)]
        if missing:
            problems.append(f"{path.name}: missing {', '.join(missing)}")
            continue
        if ref["key"] != path.stem:
            problems.append(f"{path.name}: key '{ref['key']}' does not match filename")
        if not (ref.get("doi") or ref.get("url")):
            problems.append(f"{path.name}: no doi and no url, so the paper cannot be retrieved")
        keys.add(ref["key"])

        pdf_text = None
        pdf_rel = ref.get("pdf")
        if pdf_rel:
            pdf_path = REPO_ROOT / pdf_rel
            if pdf_path.exists():
                raw = read_pdf_text(pdf_path)
                pdf_text = normalize(raw) if raw else None
            elif args.require_pdf:
                problems.append(f"{path.name}: pdf missing at {pdf_rel}")
            else:
                skipped.append(f"{path.name}: pdf absent, quote checks skipped")
        elif args.require_pdf:
            problems.append(f"{path.name}: no pdf field")
        else:
            skipped.append(f"{path.name}: no pdf field, quote checks skipped")

        supports = ref["supports"]
        if not isinstance(supports, list) or not supports:
            problems.append(f"{path.name}: supports must be a non-empty list")
            continue

        for i, item in enumerate(supports, 1):
            where = f"{path.name} supports[{i}]"
            if not isinstance(item, dict):
                problems.append(f"{where}: not a mapping")
                continue
            absent = [f for f in REQUIRED_SUPPORT if not item.get(f)]
            if absent:
                problems.append(f"{where}: missing {', '.join(absent)}")
                continue
            ok, why = anchor_resolves(str(item["anchor"]))
            if not ok:
                problems.append(f"{where}: {why}")
            if pdf_text is not None:
                checked_quotes += 1
                if normalize(str(item["quote"])) not in pdf_text:
                    problems.append(
                        f"{where}: quote does not appear verbatim in the stored PDF")

    markers = collect_markers()
    for key, places in sorted(markers.items()):
        if key not in keys:
            problems.append(f"cite marker '{key}' has no reference file ({places[0]})")
    for key in sorted(keys - set(markers)):
        problems.append(f"reference '{key}' is not cited by any code marker")

    print(f"{len(files)} reference(s), {len(markers)} cited key(s) in code, "
          f"{checked_quotes} quote(s) checked against stored PDFs")
    for note in skipped:
        print(f"  skipped: {note}")
    if not problems:
        print("no problems found")
        return 0
    print(f"\n{len(problems)} problem(s):")
    for problem in problems:
        print(f"  {problem}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
