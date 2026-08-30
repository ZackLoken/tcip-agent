"""Consistency rail between cv-research's academic-source list and the WebFetch fence.

`agent_terminal.settings.json`'s own `_comment` pairs its WebFetch allowlist with the cv-research
skill: the skill names the academic hosts an agent may reason it can fetch, and the allowlist is
the one that actually grants fetching them. When two hand-maintained declarations state the same
fact, they drift apart silently unless something checks them against each other (CLAUDE.md: when
two code paths must agree, a consistency check needs an independent recheck, not shared
implementation, to mean anything). This is that recheck, parsed from both files directly rather
than from a copy of either.

Host agreement is checked one-directionally on the www prefix, matching how the fence actually
grants hosts: `arxiv.org` and `semanticscholar.org` each carry a bare and a www rule side by
side, while every other academic host here has only a bare rule. Every skill-named host (always
written bare in the skill) must be granted under its exact bare string: a fence that grants only
the www form no longer covers the bare URLs a paper's own link actually uses (a real regression),
so that direction is never normalized. A granted host is checked against the skill only after
stripping any www prefix, so a www rule granted alongside its own bare counterpart is not
reported as an extra host the skill never named.

The skill side is parsed structurally, scoped to the source-list bullets between "The
allowed/preferred set:" and "Search discipline:", not the whole section's prose: a backticked
filename mentioned anywhere else in section 1 is never read as a host.

Coverage, not a fail-before proof: the two sides already agree as of this test's introduction, so
a run today passes by construction. It exists to catch the moment either side adds or drops a host
without the other; the fixture cases below construct that drift directly against each parser
rather than waiting for it to happen to the live files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CV_RESEARCH_SKILL = REPO / ".github" / "skills" / "cv-research" / "SKILL.md"
FENCE_SETTINGS = REPO / "packages" / "tcip-web" / "src" / "tcip_web" / "agent_terminal.settings.json"

_SECTION_1 = re.compile(r"## 1\. Research.*?(?=\n## 2\.)", re.DOTALL)
_SOURCE_LIST = re.compile(r"The allowed/preferred set:\s*\n\n(.*?)\n\nSearch discipline:", re.DOTALL)
_LIST_ITEM = re.compile(r"^- .+(?:\n(?!- ).+)*", re.MULTILINE)
_BACKTICK_HOST = re.compile(r"`([a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,})`")
_WEBFETCH_DOMAIN = re.compile(r"^WebFetch\(domain:(.+)\)$")


def _strip_www(host: str) -> str:
    return host[len("www.") :] if host.startswith("www.") else host


def _hosts_from_source_list(section_text: str) -> set[str]:
    """Backticked host domains named in section 1's own academic-source bullets.

    Scoped to the source-list block (the "- Name (`host`): ..." / "- Name: `host`, ...: ..."
    bullets between "The allowed/preferred set:" and "Search discipline:"), never the whole
    section's prose, so a backticked filename in an unrelated sentence there is never read as
    a host.
    """
    block = _SOURCE_LIST.search(section_text)
    assert block, "cv-research SKILL.md's academic-source bullet list has moved or changed shape"
    hosts: set[str] = set()
    for item in _LIST_ITEM.findall(block.group(1)):
        assert ":" in item, f"source-list bullet has no colon description: {item!r}"
        hosts |= set(_BACKTICK_HOST.findall(item))
    return hosts


def _skill_hosts() -> set[str]:
    """The academic hosts cv-research SKILL.md's section 1 names, read from the live file."""
    text = CV_RESEARCH_SKILL.read_text(encoding="utf-8")
    section = _SECTION_1.search(text)
    assert section, "cv-research SKILL.md has no '## 1. Research' section to read hosts from"
    return _hosts_from_source_list(section.group(0))


def _hosts_from_fence(allow: list[str]) -> set[str]:
    """The literal ``WebFetch(domain:...)`` hosts an allowlist grants, unnormalized."""
    hosts: set[str] = set()
    for rule in allow:
        m = _WEBFETCH_DOMAIN.match(rule)
        if m:
            hosts.add(m.group(1))
    return hosts


def _fence_hosts() -> set[str]:
    """The domains `agent_terminal.settings.json`'s WebFetch allowlist actually grants, read
    from the live file."""
    data = json.loads(FENCE_SETTINGS.read_text(encoding="utf-8"))
    return _hosts_from_fence(data["permissions"]["allow"])


def _check_agreement(skill_hosts: set[str], fence_hosts: set[str]) -> tuple[list[str], list[str]]:
    """The two drift directions, normalized asymmetrically to match how the fence grants hosts.

    Skill -> fence: every skill-named host (always written bare) must be granted under its
    exact bare string, unnormalized; a fence that grants only the www form of a host no longer
    covers the bare URLs a paper's own link uses, so that drop is reported, not hidden.

    Fence -> skill: a granted host is compared to the skill after stripping any www prefix, so
    a www rule granted alongside its own bare counterpart (`arxiv.org` and `semanticscholar.org`
    each carry both today) is not reported as an extra host the skill never named.
    """
    only_in_skill = sorted(skill_hosts - fence_hosts)
    only_in_fence = sorted({_strip_www(h) for h in fence_hosts} - skill_hosts)
    return only_in_skill, only_in_fence


def test_cv_research_academic_hosts_agree_with_the_webfetch_fence():
    skill_hosts = _skill_hosts()
    fence_hosts = _fence_hosts()
    only_in_skill, only_in_fence = _check_agreement(skill_hosts, fence_hosts)
    assert not only_in_skill and not only_in_fence, (
        f"cv-research names host(s) the fence does not grant under their bare form: {only_in_skill}; "
        f"the fence grants host(s) cv-research does not name: {only_in_fence}"
    )


# Fixture-driven drift cases below: each constructs a synthetic input and checks a parser catch
# the drift directly, rather than waiting for it to happen to the live files.

_SKILL_TEXT = """
## 1. Research: academic sources only

The allowed/preferred set:

- arXiv (`arxiv.org`): preprints.
- Semantic Scholar (`semanticscholar.org`): citation graph.

Search discipline:

- Read the ablation, not the headline number: see `crops.yml` for an unrelated example.

## 2. Implement
"""


def _section_1(text: str) -> str:
    match = _SECTION_1.search(text)
    assert match
    return match.group(0)


def test_skill_host_parser_ignores_a_filename_outside_the_source_list():
    """A backticked, dot-bearing token in the Search discipline prose (a filename, not a host)
    is not picked up: the parser is scoped to the source-list bullets, not the whole section."""
    hosts = _hosts_from_source_list(_section_1(_SKILL_TEXT))
    assert hosts == {"arxiv.org", "semanticscholar.org"}


def test_skill_host_parser_catches_a_host_added_to_the_skill():
    """Adding a host to the source list is visible to the parser (the skill -> fence drift
    direction the agreement test relies on)."""
    added = _SKILL_TEXT.replace(
        "- Semantic Scholar (`semanticscholar.org`): citation graph.",
        "- Semantic Scholar (`semanticscholar.org`): citation graph.\n"
        "- OpenReview (`openreview.net`): peer reviews.",
    )
    hosts = _hosts_from_source_list(_section_1(added))
    assert hosts == {"arxiv.org", "semanticscholar.org", "openreview.net"}


def test_fence_host_parser_catches_a_host_added_to_the_fence():
    """Adding a `WebFetch(domain:...)` rule is visible to the parser (the fence -> skill drift
    direction the agreement test relies on)."""
    allow = ["WebSearch", "WebFetch(domain:arxiv.org)", "WebFetch(domain:openreview.net)"]
    assert _hosts_from_fence(allow) == {"arxiv.org", "openreview.net"}


def test_agreement_check_catches_a_dropped_bare_grant_kept_as_www_only():
    """The scenario the www normalization must not hide: the fence drops the bare
    ``WebFetch(domain:arxiv.org)`` rule while keeping ``WebFetch(domain:www.arxiv.org)``. The
    skill still names the bare host (the form a paper URL actually uses), which the fence no
    longer grants; the one-directional normalization reports this rather than silently agreeing.
    """
    skill_hosts = {"arxiv.org"}
    fence_hosts = {"www.arxiv.org"}  # bare grant dropped, www kept
    only_in_skill, only_in_fence = _check_agreement(skill_hosts, fence_hosts)
    assert only_in_skill == ["arxiv.org"]
    assert only_in_fence == []


def test_agreement_check_does_not_flag_a_www_companion_rule():
    """A www rule granted alongside its own bare counterpart (today's actual arXiv/Semantic
    Scholar shape) is not reported as an extra host the skill never named."""
    skill_hosts = {"arxiv.org"}
    fence_hosts = {"arxiv.org", "www.arxiv.org"}
    only_in_skill, only_in_fence = _check_agreement(skill_hosts, fence_hosts)
    assert only_in_skill == []
    assert only_in_fence == []
