"""List every claim-shaped sentence this change *adds* to comments and docstrings.

`verify_doc_examples.py` checks that code examples run. Nothing checked the prose, and prose is
where the expensive mistakes went: "Falls back to torchvision when timm is unavailable" (no
fallback exists), "Any timm name is accepted" (convnext/swin raise), "an MCP tool cannot carry a
real batch" (it can, as JSON). Each was reasoning that felt settled, so no "remember to verify"
rule caught it.

This does not judge truth; it cannot. It enumerates, from the diff, every sentence that asserts
something checkable, so each one is looked at deliberately instead of slipping through as
background reasoning. Output empty = the change adds no unverified assertion.

    python scripts/verify_claims.py                 # vs HEAD (unstaged + staged)
    python scripts/verify_claims.py --against <rev>
    python scripts/verify_claims.py --strict        # exit 1 if any claim is found

Retire a listed claim by verifying it and leaving it, or by rewriting the sentence to state what
the code *does* rather than why it exists or what it cannot do.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

# Language that asserts a fact beyond "here is what this line does": purpose, capability,
# necessity, exclusivity, causation. These are the shapes that cannot be checked by running the
# function, which is exactly why they need naming.
_CLAIM = re.compile(
    r"\b("
    r"falls?\s+back|fallback|exists?\s+(?:for|to|because)|is\s+there\s+(?:for|so)|"
    r"cannot|can'?t|never|always|only\s+(?:way|path|place|works?|when)|"
    r"is\s+required|are\s+required|must\s+be|has\s+to\s+be|"
    r"unavailable|unsupported|not\s+supported|impossible|"
    r"because|so\s+that|otherwise|instead\s+of|rather\s+than|"
    r"guarantees?|ensures?|prevents?|"
    r"nothing\s+(?:calls?|uses?|reads?)|no\s+(?:caller|consumer|reader)|"
    # Capability claims: "works for everything" and "you may do X" are the two shapes that
    # slipped through as instructions rather than assertions.
    r"(?:accepts?|supports?|allows?|handles?)\s+any|"
    r"any\s+\S+(?:\s+\S+)?\s+(?:is|are)\s+(?:accepted|supported|allowed|honou?red)|"
    r"to\s+override|pass\s+your\s+own|you\s+can\s+(?:pass|use|override|supply)"
    r")\b",
    re.IGNORECASE,
)

# Only prose carries these claims: comments, and lines inside a docstring/markdown.
_CODE_COMMENT = re.compile(r"^\s*#")
_PY = (".py",)
_PROSE = (".md",)


def _added_lines(against: str) -> list[tuple[str, int, str]]:
    """(path, line-number-in-new-file, text) for every line this change adds."""
    diff = subprocess.run(["git", "diff", "-U0", against, "--", "*.py", "*.md"],
                          capture_output=True, text=True, check=False).stdout
    out: list[tuple[str, int, str]] = []
    path, lineno = "", 0
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
        elif line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            lineno = int(m.group(1)) if m else 0
        elif line.startswith("+") and not line.startswith("+++"):
            out.append((path, lineno, line[1:]))
            lineno += 1
    return out


def _is_prose(path: str, text: str) -> bool:
    if path.endswith(_PROSE):
        return True
    if not path.endswith(_PY):
        return False
    if _CODE_COMMENT.match(text):
        return True
    # A docstring body line: prose-looking, not code. Cheap heuristic: over-including a code line
    # only costs a false listing, while missing a docstring line costs the thing this exists for.
    stripped = text.strip()
    return bool(stripped) and not re.match(r"^[\w.\[\]\"']+\s*[=(:]|^(from|import|return|raise)\b",
                                           stripped)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--against", default="HEAD", help="revision to diff against (default HEAD)")
    ap.add_argument("--strict", action="store_true", help="exit 1 when any claim is listed")
    args = ap.parse_args()

    found: list[tuple[str, int, str]] = []
    for path, lineno, text in _added_lines(args.against):
        if _is_prose(path, text) and _CLAIM.search(text):
            found.append((path, lineno, text.strip()))

    if not found:
        print(f"no claim-shaped prose added vs {args.against}")
        return 0

    print(f"{len(found)} claim-shaped sentence(s) added vs {args.against}.")
    print("Verify each against the code, or rewrite it to state behaviour instead of purpose:\n")
    for path, lineno, text in found:
        print(f"  {path}:{lineno}")
        print(f"      {text}")
    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
