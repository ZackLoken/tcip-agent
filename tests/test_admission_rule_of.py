"""admission_rule_of and parse_validation_reference: the one reading of a bucket's validated
count operating point as a rule that admits a prediction score, and the one parser of the
validation-reference marker a rule-admitted record names.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp.pipelines.resolution import admission_rule_of, parse_validation_reference


@pytest.mark.parametrize("value,expected", [
    ("exp-1:0123456789abcdef", ("exp-1", "0123456789abcdef")),
    ("exp:with:colons:0123456789abcdef", ("exp:with:colons", "0123456789abcdef")),
    ("no-colon-here", None),
    (":0123456789abcdef", None),
    ("exp-1:", None),
    ("exp-1:not-hex-characters-", None),
    ("exp-1:0123456789abcde", None),  # fifteen hex characters, one short of sixteen
])
def test_parse_validation_reference(value: str, expected: tuple[str, str] | None) -> None:
    assert parse_validation_reference(value) == expected


def test_admission_rule_of_answers_none_for_no_stamp(tmp_path: Path) -> None:
    """An absent stamp reads as its own reason, distinct from a stamp claiming nothing: the file
    itself does not exist, so nothing here asserts what its content says."""
    resolution = admission_rule_of(None, tmp_path)
    assert resolution.rule is None
    assert "no operating_point.json" in resolution.reason
    assert "does not claim validated" not in resolution.reason


def test_admission_rule_of_answers_none_for_a_stamp_claiming_nothing(tmp_path: Path) -> None:
    resolution = admission_rule_of({"validated": False}, tmp_path)
    assert resolution.rule is None
    assert "does not claim validated" in resolution.reason


def test_admission_rule_of_answers_none_for_a_bound_claim_with_no_readable_conf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stamp whose claim binds (verify_stamp_binding answers claimed and ok) but whose own
    conf value is not a finite, non-boolean number: this shape has no honest real producer (a
    resolver's own conf is always numeric), so the binding is stubbed to isolate the check."""
    import tcip_mcp.pipelines.resolution as resolution_mod

    def fake_binding(sidecar, pred_dir, *, document, **kw):
        return resolution_mod.StampBinding(
            ok=True, claimed=True, experiment_id="exp-1", record_digest="0123456789abcdef")

    monkeypatch.setattr(resolution_mod, "verify_stamp_binding", fake_binding)
    stamp = {"validated": True, "operating_point": {"conf": {"value": "not-a-number"}}}
    resolution = admission_rule_of(stamp, tmp_path)
    assert resolution.rule is None
    assert "no readable conf" in resolution.reason
