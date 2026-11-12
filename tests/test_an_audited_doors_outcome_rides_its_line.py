"""An audited door's ``outcome`` puts facts of its answer on its ``ok`` line, as extra facts.

``outcome`` is called once for a call that returns its answer, and never for one that returns its
error dict or raises. Its facts follow the rules a line's extra facts follow: a key the line
declares is dropped, a ``None`` fact is absent from the stored line and a false one is kept. An
outcome that raises, or one returning a fact with no JSON form, is a line that could not be
written: the door's own write stands and nothing is appended.

Each door here is made for the test through the envelope itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tcip_mcp.audit import MutationCommittedWithoutAuditLineError, audited
from tests._audit_fixtures import audit_rows

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path


def test_the_outcome_is_called_once_for_an_answer_and_never_for_an_error_dict_or_a_raise(
    tmp_path: Path,
) -> None:
    """The error dict leaves no line, the raise an ``exception`` line without the facts."""
    answers: list[dict] = []

    def outcome(answer: dict) -> dict[str, object]:
        answers.append(answer)
        return {"written": answer["written"]}

    @audited(outcome=outcome)
    def act(project: Path, how: str) -> dict:
        if how == "refuse":
            return {"error": "refused"}
        if how == "raise":
            raise ValueError("raised")
        return {"written": how}

    assert act(tmp_path, "refuse") == {"error": "refused"}
    with pytest.raises(ValueError, match="raised"):
        act(tmp_path, "raise")
    assert act(tmp_path, "answer") == {"written": "answer"}

    assert answers == [{"written": "answer"}]
    raised, answered = audit_rows(tmp_path, "act")
    assert raised["status"] == "exception"
    assert "written" not in raised
    assert answered["status"] == "ok"
    assert answered["written"] == "answer"


def _raising(answer: dict) -> Mapping[str, object]:
    """Raise, as an outcome that cannot read its answer does."""
    raise KeyError("no such fact")


def _unjsonable(answer: dict) -> Mapping[str, object]:
    """Return a fact with no JSON form."""
    return {"handle": object()}


@pytest.mark.parametrize(
    "outcome", [_raising, _unjsonable], ids=["raises", "no_json_form"]
)
def test_an_outcome_that_cannot_be_recorded_commits_without_a_line(
    tmp_path: Path, outcome: Callable[[dict], Mapping[str, object]]
) -> None:
    """The call raises the committed-without-line error; the door's write stands, no line lands."""
    written = tmp_path / "written.txt"

    @audited(outcome=outcome)
    def act(project: Path) -> dict:
        written.write_text("the door's own write", encoding="utf-8")
        return {"written": str(written)}

    with pytest.raises(MutationCommittedWithoutAuditLineError) as caught:
        act(tmp_path)

    assert caught.value.tool == "act"
    assert written.read_text(encoding="utf-8") == "the door's own write"
    assert audit_rows(tmp_path, "act") == []


def test_outcome_facts_follow_the_extra_fact_rules(tmp_path: Path) -> None:
    """A declared key is dropped, a ``None`` fact absent, a false one kept; arguments unchanged."""

    @audited(
        outcome=lambda answer: {
            "status": "declared",
            "arguments": {"declared": True},
            "absent": None,
            "minted": False,
            "kept": answer["kept"],
        }
    )
    def act(project: Path, given: int) -> dict:
        return {"kept": given}

    act(tmp_path, 3)

    (line,) = audit_rows(tmp_path, "act")
    assert line["status"] == "ok"
    assert line["arguments"] == {"given": 3}
    assert "absent" not in line
    assert line["minted"] is False
    assert line["kept"] == 3
