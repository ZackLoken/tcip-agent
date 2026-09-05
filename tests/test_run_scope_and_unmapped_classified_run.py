"""``run_scope`` and ``unmapped_classified_run``: the one read of a run's own recorded
``(subject, attribute)`` pair, and the one composed refusal for a classified run that resolved no
``id_map`` to decode its predictions with.

Every publishing door (the shared image publisher, the raster regime, the web inference worker)
calls these two functions rather than re-reading ``config["data"]`` or composing its own remedy
text; these tests pin the functions directly, at the one seam every door shares.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tcip_mcp.tools.inference_tools import run_scope, unmapped_classified_run


def _predictor(config: dict, path: str = "/models/best.pt") -> SimpleNamespace:
    return SimpleNamespace(config=config, path=path)


def test_run_scope_reads_the_recorded_pair() -> None:
    predictor = _predictor({"data": {"subject": "bud", "attribute": "bud_opening"}})
    assert run_scope(predictor) == ("bud", "bud_opening")


def test_run_scope_reads_a_detector_runs_bare_subject() -> None:
    predictor = _predictor({"data": {"subject": "bud"}})
    assert run_scope(predictor) == ("bud", None)


def test_run_scope_reads_no_scope_at_all_as_a_pair_of_nones() -> None:
    predictor = _predictor({"data": {}})
    assert run_scope(predictor) == (None, None)


def test_run_scope_refusal_names_the_attribute_and_the_checkpoint_path() -> None:
    predictor = _predictor({"data": {"attribute": "bud_opening"}}, path="/models/bespoke.pt")
    with pytest.raises(ValueError, match="declares attribute 'bud_opening' with no subject") as excinfo:
        run_scope(predictor)
    assert "/models/bespoke.pt" in str(excinfo.value)


def test_unmapped_classified_run_answers_none_for_a_resolved_map() -> None:
    data_cfg = {"subject": "bud", "attribute": "bud_opening"}
    assert unmapped_classified_run(data_cfg, {"open": 0, "closed": 1}, images_dir="/data/images") is None


def test_unmapped_classified_run_answers_none_for_a_detector_run() -> None:
    data_cfg = {"subject": "bud"}
    assert unmapped_classified_run(data_cfg, None, images_dir="/data/images") is None


def test_unmapped_classified_run_names_images_dir_for_a_registry_derived_run_called_with_none() -> None:
    data_cfg = {"subject": "bud", "attribute": "bud_opening"}
    message = unmapped_classified_run(data_cfg, None, images_dir=None)
    assert message is not None
    assert "no images_dir was given" in message
    assert "bud_opening" in message and "bud" in message


def test_unmapped_classified_run_names_write_class_map_for_a_registry_derived_run_with_no_registry() -> None:
    data_cfg = {"subject": "bud", "attribute": "bud_opening"}
    message = unmapped_classified_run(data_cfg, None, images_dir="/data/images")
    assert message is not None
    assert "write_class_map" in message
    assert "/data/images" in message


def test_unmapped_classified_run_names_retrain_for_a_bespoke_dataset_source() -> None:
    data_cfg = {"subject": "bud", "attribute": "bud_opening", "dataset_source": "pkg.mod:build"}
    message = unmapped_classified_run(data_cfg, None, images_dir="/data/images")
    assert message is not None
    assert "data.id_map" in message and "retrain" in message
    assert "write_class_map" not in message


def test_unmapped_classified_run_names_retrain_for_a_coco_sourced_run() -> None:
    data_cfg = {"subject": "bud", "attribute": "bud_opening", "coco_json": "/data/anns.json"}
    message = unmapped_classified_run(data_cfg, None, images_dir="/data/images")
    assert message is not None
    assert "data.id_map" in message and "retrain" in message


def test_unmapped_classified_run_a_non_registry_derived_run_names_retrain_even_with_no_images_dir() -> None:
    """A bespoke or COCO-sourced run's remedy never depends on ``images_dir`` at all (the raster
    regime, which always calls with ``images_dir=None``, reaches exactly this branch when its own
    targets are not registry-derived): the message names the retrain route, never the
    registry-derived branch's ``images_dir``/``write_class_map`` text."""
    data_cfg = {"subject": "bud", "attribute": "bud_opening", "dataset_source": "pkg.mod:build"}
    message = unmapped_classified_run(data_cfg, None, images_dir=None)
    assert message is not None
    assert "data.id_map" in message and "retrain" in message
    assert "images_dir" not in message
