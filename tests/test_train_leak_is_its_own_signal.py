"""Calibration-versus-holdout separation and train-disjointness are two different questions.

``sweep["disjoint"]`` answers only the first: did the two sides of the reference share an image. It
reads True on a reference whose every image was also a training image, so the gate cannot rest on it
and a reader must not take it for the disjointness the operating contract requires. The
train-disjointness result travels beside it, and a leak found there has to refuse on its own,
whichever of its two mechanisms (group-level or exact-stem) actually found it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import tcip_store  # noqa: E402
from tcip_mcp.experiments import split_key  # noqa: E402
from tcip_mcp.pipelines.operating_point import resolve_operating_point  # noqa: E402
from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, VALIDATED_HELD_OUT  # noqa: E402

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

N_IMAGES = 4
OBJECTS_PER_IMAGE = 8

CAL_STEMS = [f"cal_{i}" for i in range(N_IMAGES)]
HOLD_STEMS = [f"hold_{i}" for i in range(N_IMAGES)]


def _records(stems: list[str], offset: float) -> list[dict]:
    """One record per stem, every object matched exactly by one detection: a reference that clears
    every gate except whatever the train-disjointness check says about it.
    """
    recs = []
    for i, stem in enumerate(stems):
        gt, dt = [], []
        for k in range(OBJECTS_PER_IMAGE):
            box = [offset + 100.0 * k, 50.0 + 10.0 * i, 40.0, 40.0]
            gt.append({"bbox": box, "category_id": 1})
            dt.append({"bbox": box, "category_id": 1, "score": 0.9})
        recs.append({"image_id": stem, "width": 4000, "height": 1000, "gt": gt, "dt": dt})
    return recs


def _write_split(experiment_id: str, train_stems: list[str]) -> None:
    tcip_store.replace(split_key(experiment_id), {"train": train_stems, "group_by": "stem"})


def _resolve(experiment_id: str):
    return resolve_operating_point(
        "bud_opening", tiled=True, dataset_hash="h", staged_conf_floor=0.05,
        experiment_id=experiment_id,
        calibration_records=_records(CAL_STEMS, 0.0),
        holdout_records=_records(HOLD_STEMS, 100000.0))


def test_a_reference_drawn_entirely_from_the_training_split_is_refused():
    """Calibration and holdout share no image with each other, so the cal-versus-holdout signal
    reads clean, yet every one of their images was trained on. The refusal comes from the
    train-disjointness result alone, which here reports its leak at group level with nothing in the
    exact-stem list beside it.
    """
    _write_split("exp_leaky", CAL_STEMS + HOLD_STEMS)

    b = _resolve("exp_leaky")
    sweep = b.params["conf"].gate_evidence
    td = sweep["train_disjointness"]

    assert sweep["disjoint"] is True  # the cal/holdout signal says nothing about the training leak
    assert td["leaked_groups"] == sorted(CAL_STEMS + HOLD_STEMS)
    assert td["leaked_stems"] == []
    assert td["unresolvable"] is False
    assert "train_disjointness_leaked" in sweep["failures"]
    assert b.params["conf"].validated_against == VALIDATED_FALSE


def test_the_same_reference_validates_against_a_training_split_it_never_touched():
    """The companion obligation: the identical records, against a run trained on other images
    entirely, must earn the held-out stamp.
    """
    _write_split("exp_clean", [f"other_{i}" for i in range(6)])

    b = _resolve("exp_clean")
    sweep = b.params["conf"].gate_evidence
    td = sweep["train_disjointness"]

    assert sweep["disjoint"] is True
    assert td["leaked_groups"] == [] and td["leaked_stems"] == []
    assert sweep["failures"] == []
    assert b.params["conf"].validated_against == VALIDATED_HELD_OUT
