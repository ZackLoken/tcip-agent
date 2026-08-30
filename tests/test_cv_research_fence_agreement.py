"""Consistency rail between cv-research's academic-source list and the WebFetch fence.

`agent_terminal.settings.json`'s own `_comment` pairs its WebFetch allowlist with the cv-research
skill: the skill names the academic hosts an agent may reason it can fetch, and the allowlist is
the one that actually grants fetching them. When two hand-maintained declarations state the same
fact, they drift apart silently unless something checks them against each other (CLAUDE.md: when
two code paths must agree, a consistency check needs an independent recheck, not shared
implementation, to mean anything). This is that recheck, parsed from both files directly rather
than from a copy of either.

Coverage, not a fail-before proof: the two sides already agree as of this test's introduction, so
a run today passes by construction. It exists to catch the moment either side adds or drops a host
without the other.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CV_RESEARCH_SKILL = REPO / ".github" / "skills" / "cv-research" / "SKILL.md"
FENCE_SETTINGS = REPO / "packages" / "tcip-web" / "src" / "tcip_web" / "agent_terminal.settings.json"

_DOMAIN = re.compile(r"`([a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,})`")
_WEBFETCH_DOMAIN = re.compile(r"^WebFetch\(domain:(.+)\)$")


def _strip_www(host: str) -> str:
    return host[len("www.") :] if host.startswith("www.") else host


def _skill_hosts() -> set[str]:
    """The backticked host domains named in cv-research SKILL.md's section 1 (academic sources)."""
    text = CV_RESEARCH_SKILL.read_text(encoding="utf-8")
    section = re.search(r"## 1\. Research.*?(?=\n## 2\.)", text, re.DOTALL)
    assert section, "cv-research SKILL.md has no '## 1. Research' section to read hosts from"
    return {_strip_www(h) for h in _DOMAIN.findall(section.group(0))}


def _fence_hosts() -> set[str]:
    """The domains `agent_terminal.settings.json`'s WebFetch allowlist actually grants."""
    data = json.loads(FENCE_SETTINGS.read_text(encoding="utf-8"))
    allow = data["permissions"]["allow"]
    hosts = []
    for rule in allow:
        m = _WEBFETCH_DOMAIN.match(rule)
        if m:
            hosts.append(m.group(1))
    return {_strip_www(h) for h in hosts}


def test_cv_research_academic_hosts_agree_with_the_webfetch_fence():
    skill_hosts = _skill_hosts()
    fence_hosts = _fence_hosts()
    only_in_skill = sorted(skill_hosts - fence_hosts)
    only_in_fence = sorted(fence_hosts - skill_hosts)
    assert skill_hosts == fence_hosts, (
        f"cv-research names host(s) the fence does not grant: {only_in_skill}; "
        f"the fence grants host(s) cv-research does not name: {only_in_fence}"
    )
