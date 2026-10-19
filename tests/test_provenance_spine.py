"""The provenance identity spine.

Covered: a checkpoint's producing run read off the run's own final status beside its sha256, the
enriched capture_env, a selection's own digests and seed, and the producing model named on the
delivery event and not on its rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

# ── capture_env records code + library fingerprint ────────────────────────────


def test_capture_env_records_code_and_libraries() -> None:
    from tcip_mcp.pipelines.model_build import capture_env

    env = capture_env()
    # git commit + numpy + CUDA are present as keys (values best-effort/null), never fatal.
    assert "tcip_git_commit" in env
    assert "numpy" in env
    assert "cuda" in env
    assert env["python"]


# ── identity resolved off a verified, registry-matched checkpoint ──────────────


def test_the_producer_of_a_completed_runs_checkpoint_is_that_run(
    tmp_path: Path,
) -> None:
    """The producer of a completed run's checkpoint is that run.

    The checkpoint resolves its producer through the binding the run's own final status recorded,
    not a caller-asserted tag or a stamp in the payload.
    """
    pytest.importorskip("torch")
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path, experiment_id="expR")

    producer = load_registered_checkpoint(ckpt, project=tmp_path).producer
    assert len(producer.checkpoint_sha256) == 64
    assert producer.experiment_id == "expR"


def test_the_producer_of_a_foreign_checkpoint_names_no_run(
    tmp_path: Path,
) -> None:
    """A foreign checkpoint's producer names its digest and no run.

    A checkpoint a run of the project wrote beside its deliverable, whose digest no completion
    names, registered in explicit mode, loads with ``experiment_id`` null.
    """
    pytest.importorskip("torch")
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tests._verified_checkpoint_fixtures import foreign_checkpoint

    producer = load_registered_checkpoint(
        foreign_checkpoint(tmp_path), project=tmp_path
    ).producer
    assert producer.checkpoint_sha256
    assert producer.experiment_id is None


def test_a_payloads_own_experiment_id_names_no_producer(
    tmp_path: Path,
) -> None:
    """A payload's own ``experiment_id`` names no producer.

    A checkpoint a run saved through its own writer with an ``experiment_id`` beside its weights,
    whose digest no completion names, keeps that field in its payload and resolves no producing
    run, its digest still named.
    """
    pytest.importorskip("torch")
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tests._verified_checkpoint_fixtures import (
        register_checkpoint,
        secondary_checkpoint,
    )

    ckpt = secondary_checkpoint(
        tmp_path, fixture_state={"experiment_id": "expStamped"}
    )
    register_checkpoint(tmp_path, ckpt, name="stamped")

    loaded = load_registered_checkpoint(ckpt, project=tmp_path)
    assert loaded.payload["experiment_id"] == "expStamped"
    assert loaded.producer.experiment_id is None
    assert loaded.producer.checkpoint_sha256


# ── draw_splits records each sample's digest and the seed ─────────────────────


def test_draw_splits_selection_embeds_digests_and_seed(
    data_dir: Path, tmp_path: Path
) -> None:
    # A selection needs at least four foreground groups to clear the floor; the fixture's own
    # three (img_001..003) need one more, added here rather than in the shared fixture.
    from PIL import Image

    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.tools.data_tools import draw_splits
    from tests._producer_fixtures import label_image
    from tests._training_values import drawn_split

    images_dir = data_dir / "images" / "2-11-26"
    Image.new("RGB", (640, 480), color=(128, 128, 128)).save(
        images_dir / "img_004.jpg"
    )
    label_image(
        images_dir / "img_004.jpg",
        [Annotation(subject="bud", geometry=BBox(288, 216, 352, 264))],
        640,
        480,
    )

    out = tmp_path / "splits"
    result = draw_splits(
        tmp_path,
        str(data_dir),
        output_path=str(out),
        subject="bud",
        **drawn_split(
            seed=7,
            val_ratio=0.25,
            calibration_ratio=0.125,
            holdout_ratio=0.125,
            grouping="tile_prefix",
        ),
    )
    assert result["seed"] == 7
    drawn = read_selection(out, project=tmp_path)
    assert drawn.seed == 7
    assert all(sample.ground_truth_digest for sample in drawn.samples)
    assert set(drawn.counts()) == {"train", "val", "calibration", "holdout"}


# ── the producing model is named once, on the delivery event ──────────────────


def test_a_delivery_event_names_its_producer_and_write_time_and_its_rows_repeat_neither(
    tmp_path: Path,
) -> None:
    """A delivery event names its producer and write time, and its rows repeat neither.

    The delivery's one event carries the bucket's producer and an ISO write time; its one row
    carries the delivery columns and no producer column.
    """
    pytest.importorskip("torch")
    from datetime import datetime

    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.delivery import DELIVERY_COLUMNS, read_delivery_events
    from tcip_mcp.pipelines.postprocessing.export import (
        deliver_per_image_counts_csv,
    )
    from tests import _trait_fixtures as fx, csv_rows
    from tests._chain_fixtures import acknowledged, predicted, published

    fx.seed_delivery_traits(tmp_path)
    fx.seed_confirmed_count(tmp_path)
    image = tmp_path / "ds" / "images" / "2026-01-01" / "a.png"
    bucket = published(
        tmp_path,
        "m/2026-01-01",
        [predicted(image, [fx.COUNT_SUBJECT] * 3)],
        scope={"subject": fx.COUNT_SUBJECT},
    )
    out = tmp_path / "counts.csv"

    acknowledged(
        tmp_path,
        lambda ack: deliver_per_image_counts_csv(
            tmp_path,
            bucket.root,
            bucket.name,
            str(out),
            trait=fx.COUNT_TRAIT,
            acknowledgment_id=ack,
            door="test_provenance",
            actor=None,
        ),
    )

    (event,) = read_delivery_events(tmp_path)
    assert event.producer == read_bucket(bucket.root, bucket.name).producer
    datetime.fromisoformat(event.produced_at)
    (row,) = csv_rows(out)
    assert set(DELIVERY_COLUMNS) <= set(row)
    assert not {
        "producer_model_sha256",
        "producing_experiment_id",
        "produced_at",
    } & set(row)


def test_the_phenology_schema_carries_the_delivery_columns_and_no_producer_column() -> (
    None
):
    from tcip_mcp.delivery import DELIVERY_COLUMNS
    from tcip_mcp.pipelines.postprocessing.phenology import (
        phenology_csv_columns,
    )
    from tests._trait_fixtures import BUD_OPENING

    columns = phenology_csv_columns(BUD_OPENING)
    assert tuple(columns[-len(DELIVERY_COLUMNS) :]) == DELIVERY_COLUMNS
    assert "producer_model_sha256" not in columns
