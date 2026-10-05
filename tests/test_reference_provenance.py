"""A reference has to be a measurement, not the model's own output.

Point an assessment's reference at the model's own predictions and every numeric criterion
clears, because the model agrees with itself. The provenance the records carry is the only thing
that can catch it, so the reference's label documents are refused whole
(:func:`~tcip_annotation.json_io.require_reference_ground_truth`) when a record carries a
prediction score, when an agent authored it and no reviewer accepted it, or when it carries a
rule-based admission nobody signed off. Ordinary ground truth, ground truth a person authored, a
prediction a reviewer accepted and a signed rule-based admission all remain a reference.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox

IMG = 32
PRODUCER = "model:m_best@c9f632ba98b2"  # the shape a publication stamps on every prediction


def _documents(root: Path, stems, annotations) -> list[Annotation]:
    """One label document per stem holding ``annotations(stem)``; every annotation they hold,
    read back."""
    from tcip_mcp.dataset_layout import UNDATED_BUCKET, label_key

    held = []
    for s in stems:
        key = label_key(root, UNDATED_BUCKET, s)
        json_io.write_label_document(key, annotations(s), IMG, IMG, keep_empty=True)
        held += json_io.read_label_document(key).annotations
    return held


def _prediction(box=(2, 2, 10, 10)):
    return Annotation(subject="bud", geometry=BBox(*box), score=0.87, created_by=PRODUCER,
                      created_at="2026-01-01T00:00:00+00:00")


def _hand(box=(2, 2, 10, 10), **kw):
    return Annotation(subject="bud", geometry=BBox(*box), **kw)


STEMS = [f"src{g}_{t}_0" for g in range(4) for t in range(2)]


def test_a_reference_of_the_models_own_predictions_refuses_whole(tmp_path):
    documents = _documents(tmp_path, STEMS,
                           lambda s: [_prediction(), _prediction(box=(15, 15, 25, 25))])

    with pytest.raises(ValueError) as exc:
        json_io.require_reference_ground_truth(documents)

    assert "16 of 16 annotations" in str(exc.value)
    assert "accept the model's proposals in the editor" in str(exc.value)


def test_a_mixed_reference_refuses_whole_rather_than_keeping_its_clean_subset(tmp_path):
    documents = _documents(tmp_path, STEMS, lambda s: [_hand()] + (
        [_prediction()] if s == "src0_0_0" else []))

    with pytest.raises(ValueError, match="1 of 9 annotations"):
        json_io.require_reference_ground_truth(documents)


def test_ground_truth_an_agent_authored_that_nobody_ruled_on_refuses(tmp_path):
    """The same output with its score dropped: no reviewer took responsibility for any of it."""
    documents = _documents(tmp_path, STEMS, lambda s: [
        _hand(created_by=PRODUCER, created_at="2026-01-01T00:00:00+00:00")])

    with pytest.raises(ValueError) as exc:
        json_io.require_reference_ground_truth(documents)

    message = str(exc.value)
    assert "8 of 8 annotations" in message
    assert PRODUCER in message
    assert "confirm these records in the editor" in message


def test_a_bare_tool_name_reads_as_a_machine_author(tmp_path):
    documents = _documents(tmp_path, ["a"], lambda s: [_hand(created_by="sam")])

    with pytest.raises(ValueError, match="authored by sam"):
        json_io.require_reference_ground_truth(documents)


@pytest.mark.parametrize("producer", ["user:", "user:   "])
def test_the_prefix_naming_no_one_is_no_persons_authorship(tmp_path, producer):
    documents = _documents(tmp_path, ["a"], lambda s: [_hand(created_by=producer)])

    with pytest.raises(ValueError, match=f"authored by {producer}"):
        json_io.require_reference_ground_truth(documents)


@pytest.mark.parametrize("record", [
    _hand(),
    _hand(created_by="user:breeder", created_at="2026-01-01T00:00:00+00:00"),
    _hand(created_by=PRODUCER, accepted_by="user:breeder",
          accepted_at="2026-01-02T00:00:00+00:00"),
], ids=["unattributed", "a-persons", "reviewer-accepted"])
def test_ground_truth_a_person_stands_behind_is_a_reference(tmp_path, record):
    json_io.require_reference_ground_truth(_documents(tmp_path, STEMS, lambda s: [record]))


def test_the_assessment_refuses_a_reference_of_the_models_own_predictions(tmp_path):
    """The rail runs inside the assessment, before any inference over the reference."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import (
        assess, confirm_count_trait, draw_reference_selection, synthetic_capture, train_on,
    )

    root = tmp_path / "ds"
    synthetic_capture(root)
    selection = draw_reference_selection(tmp_path, root, tmp_path / "selection")
    checkpoint = train_on(tmp_path / "selection", tmp_path, "exp-self-reference")
    confirm_count_trait(tmp_path)
    held = selection.on("holdout")[0]
    json_io.write_label_document(held.ground_truth, [_prediction()], 64, 64)

    record = assess(tmp_path, checkpoint, tmp_path / "selection")

    assert "carry a prediction score" in record.get("error", ""), record
