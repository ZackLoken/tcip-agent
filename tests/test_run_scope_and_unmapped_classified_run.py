"""``run_scope`` and ``unmapped_classified_run``: the one read of a run's own recorded class
space, and the one composed refusal for a classified run that resolved no ``id_map`` to decode
its predictions with.

Every publishing door (the shared image publisher, the raster regime, the web inference worker)
calls these two functions rather than re-reading ``config["data"]`` or composing its own remedy
text; these tests pin the functions directly, at the one seam every door shares.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tcip_mcp.pipelines.data.selection import ClassScope
from tcip_mcp.tools.inference_tools import run_scope, unmapped_classified_run


def _predictor(config: dict, path: str = "/models/best.pt") -> SimpleNamespace:
    return SimpleNamespace(config=config, path=path)


def _scope(data_cfg: dict) -> ClassScope:
    """The run's own recorded class space, read the way every door reads it."""
    return run_scope(_predictor({"data": data_cfg}))


def test_run_scope_reads_the_recorded_class_space() -> None:
    scope = _scope({"subject": "bud", "attribute": "bud_opening", "id_map": {"open": 0}})
    assert scope == ClassScope("bud", "bud_opening", {"open": 0})


def test_run_scope_reads_a_detector_runs_bare_subject() -> None:
    assert _scope({"subject": "bud"}) == ClassScope("bud", None, None)


def test_run_scope_reads_no_scope_at_all_as_an_empty_class_space() -> None:
    assert _scope({}) == ClassScope()


def test_run_scope_refusal_names_the_attribute_and_the_checkpoint_path() -> None:
    predictor = _predictor({"data": {"attribute": "bud_opening"}}, path="/models/bespoke.pt")
    with pytest.raises(ValueError, match="declares attribute 'bud_opening' with no subject") as excinfo:
        run_scope(predictor)
    assert "/models/bespoke.pt" in str(excinfo.value)


def test_unmapped_classified_run_answers_none_for_a_resolved_map() -> None:
    scope = _scope({"subject": "bud", "attribute": "bud_opening"})
    assert unmapped_classified_run(
        scope, {"open": 0, "closed": 1}, images_dir="/data/images") is None


def test_unmapped_classified_run_answers_none_for_a_detector_run() -> None:
    assert unmapped_classified_run(
        _scope({"subject": "bud"}), None, images_dir="/data/images") is None


def test_unmapped_classified_run_names_images_dir_for_a_registry_derived_run_called_with_none() -> None:
    scope = _scope({"subject": "bud", "attribute": "bud_opening"})
    message = unmapped_classified_run(scope, None, images_dir=None)
    assert message is not None
    assert "no images_dir was given" in message
    assert "bud_opening" in message and "bud" in message


def test_unmapped_classified_run_names_write_subject_registry_for_a_registry_derived_run_with_no_registry() -> None:
    scope = _scope({"subject": "bud", "attribute": "bud_opening"})
    message = unmapped_classified_run(scope, None, images_dir="/data/images")
    assert message is not None
    assert "write_subject_registry" in message
    assert "/data/images" in message
