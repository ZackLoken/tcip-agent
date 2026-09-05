"""``write_predictions_json``: the run's own ``(subject, attribute)`` scope decides where a decoded
label lands. With an attribute, the decoded value lands under ``attributes[attribute]`` and
``subject`` carries the object class; without one, the output is a detector run's, byte-identical
to before this shape existed. Every refusal below runs before the writer's own persistence call,
so a run that cannot be decoded honestly leaves no document on disk at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_annotation.json_io import read_annotations
from tcip_mcp.pipelines.postprocessing.export import write_predictions_json

SUBJECT = "bud"
ATTRIBUTE = "bud_opening"
ID_MAP = {"open": 0, "closed": 1}


def _result(*, boxes, scores, labels, width=100, height=80) -> dict:
    return {"boxes": boxes, "scores": scores, "labels": labels, "width": width, "height": height}


def test_an_attribute_with_no_subject_refuses_before_any_write(tmp_path: Path) -> None:
    path = tmp_path / "img1.json"
    result = _result(boxes=[[1, 1, 5, 5]], scores=[0.9], labels=[1])

    with pytest.raises(ValueError, match="was given with no subject"):
        write_predictions_json(str(path), result, subject=None, attribute=ATTRIBUTE, id_map=ID_MAP)

    assert not path.exists()


def test_a_classified_run_decodes_the_value_under_the_attribute(tmp_path: Path) -> None:
    path = tmp_path / "img1.json"
    result = _result(boxes=[[1, 1, 5, 5]], scores=[0.9], labels=[1])

    write_predictions_json(str(path), result, subject=SUBJECT, attribute=ATTRIBUTE, id_map=ID_MAP)

    (written,) = read_annotations(str(path))
    assert written.subject == SUBJECT
    assert written.attributes == {ATTRIBUTE: "open"}
    assert written.score == 0.9


def test_a_classified_result_carrying_a_label_outside_the_map_refuses_naming_id_and_map(
    tmp_path: Path,
) -> None:
    path = tmp_path / "img1.json"
    # label 3 decodes to 0-indexed id 2, not a key of ID_MAP (which only has ids 0 and 1).
    result = _result(boxes=[[1, 1, 5, 5]], scores=[0.9], labels=[3])

    with pytest.raises(ValueError, match=r"detection 0 decoded to id 2") as excinfo:
        write_predictions_json(str(path), result, subject=SUBJECT, attribute=ATTRIBUTE, id_map=ID_MAP)

    assert "[0, 1]" in str(excinfo.value)
    assert not path.exists()


def test_a_classified_run_refuses_before_the_first_document_of_a_multi_detection_result(
    tmp_path: Path,
) -> None:
    """The refusal fires per-detection, but nothing lands on disk for the whole result: the
    decodable detection ahead of the unmapped one is not written either."""
    path = tmp_path / "img1.json"
    result = _result(boxes=[[1, 1, 5, 5], [10, 10, 20, 20]], scores=[0.9, 0.8], labels=[1, 3])

    with pytest.raises(ValueError):
        write_predictions_json(str(path), result, subject=SUBJECT, attribute=ATTRIBUTE, id_map=ID_MAP)

    assert not path.exists()


def test_a_detector_run_with_no_id_map_keeps_the_raw_index_name(tmp_path: Path) -> None:
    path = tmp_path / "img1.json"
    result = _result(boxes=[[1, 1, 5, 5]], scores=[0.9], labels=[1])

    write_predictions_json(str(path), result, subject=None, attribute=None, id_map=None)

    (written,) = read_annotations(str(path))
    assert written.subject == "0"
    assert written.attributes == {}


def test_a_detector_run_with_a_recorded_map_decodes_the_name_into_subject(tmp_path: Path) -> None:
    path = tmp_path / "img1.json"
    result = _result(boxes=[[1, 1, 5, 5]], scores=[0.9], labels=[1])

    write_predictions_json(str(path), result, subject=None, attribute=None, id_map={SUBJECT: 0})

    (written,) = read_annotations(str(path))
    assert written.subject == SUBJECT
    assert written.attributes == {}


def test_a_detector_result_carrying_a_label_outside_the_map_never_refuses(tmp_path: Path) -> None:
    """A detector run's raw-index fallback is an honest, degraded name, never a refusal: the
    rule that a per-detection decode failure refuses applies only under a declared attribute."""
    path = tmp_path / "img1.json"
    result = _result(boxes=[[1, 1, 5, 5]], scores=[0.9], labels=[9])

    write_predictions_json(str(path), result, subject=None, attribute=None, id_map={SUBJECT: 0})

    (written,) = read_annotations(str(path))
    assert written.subject == "8"


def test_omitting_the_scope_keywords_refuses_at_the_call_boundary(tmp_path: Path) -> None:
    """Coverage of the signature, not a guard: ``subject``/``attribute`` are required keyword-only
    parameters with no default, so a caller that omits them fails before the writer's own body
    runs at all. The three real call sites are static code that always states them; this is not
    what proves that."""
    path = tmp_path / "img1.json"
    result = _result(boxes=[], scores=[], labels=[])

    with pytest.raises(TypeError):
        write_predictions_json(str(path), result)  # type: ignore[call-arg]
