"""Registry ranking applies no default metric and ranks by a stated one.

``rank_registered_models`` lists and does not rank when ``metric`` is left empty, and a stated
metric that no model carries is a distinct error from an empty registry. These tests also cover
lower-is-better ranking, the metrics source a registration derives, and a completed run's
checkpoint metrics sourced from the checkpoint's own epoch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from tcip_mcp.model_registry import Listed, ModelRegistry


def _checkpoint(tmp_path: Path, name: str) -> str:
    """Write a checkpoint file of its own under ``tmp_path`` for ``name`` and return its path."""
    pytest.importorskip("torch")
    from tests._verified_checkpoint_fixtures import produced_checkpoint

    return str(produced_checkpoint(tmp_path / f"{name}.pt", name))


def _registered(
    tmp_path: Path, *entries: tuple[str, dict | None]
) -> ModelRegistry:
    """Return a registry holding one foreign entry per ``(name, metrics)``.

    Each entry is over a checkpoint file of its own.
    """
    from tcip_mcp.model_registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path))
    for name, metrics in entries:
        reg.register_model(
            name,
            _checkpoint(tmp_path, name),
            metrics=metrics,
            tags=[],
            actor=None,
        )
    return reg


def _named(reg: ModelRegistry, name: str) -> Listed:
    """Return the one entry ``reg`` lists under ``name``."""
    (listed,) = [m for m in reg.list_models() if m.entry.name == name]
    return listed


def test_a_listing_answers_each_entry_flat_with_its_checkpoints_metrics_and_source(
    tmp_path: Path,
) -> None:
    """A listing answers each entry's fields flat, its checkpoint's metrics and source over them.

    The listing's model and the ranked answer the training route projects are each the entry's
    fields with the facts' ``metrics`` and ``metrics_source``, one flat mapping.
    """
    from tcip_mcp.tools.model_tools import (
        ranked_registered_model,
        registered_listing,
    )

    reg = _registered(tmp_path, ("a", {"val_map50": 0.5}))
    (listed,) = reg.list_models()
    flat = {
        **listed.entry.model_dump(),
        "metrics": listed.facts.metrics,
        "metrics_source": listed.facts.metrics_source,
    }

    assert registered_listing(tmp_path)["models"] == [flat]
    ranked = ranked_registered_model(
        tmp_path,
        "val_map50",
        higher_is_better=True,
        include_unverified=True,
        experiment_ids=None,
        tag=None,
    )
    assert {key: ranked[key] for key in flat} == flat
    assert (ranked["metrics_source"], ranked["metrics"]) == (
        "caller",
        {"val_map50": 0.5},
    )


def test_rank_registered_models_lists_rather_than_ranking_on_an_empty_metric(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.model_tools import rank_registered_models

    project = tmp_path

    # Empty registry, no metric: an empty listing, not a refusal.
    assert rank_registered_models(project) == {
        "models": [],
        "count": 0,
        "available_metrics": [],
    }

    _registered(
        tmp_path, ("a", {"val_map50": 0.70}), ("b", {"val_map50": 0.90})
    )

    # A populated registry with no metric: the listing, not a required-metric refusal.
    res = rank_registered_models(project)
    assert res["count"] == 2
    assert {m["name"] for m in res["models"]} == {"a", "b"}
    assert res["available_metrics"] == [
        {
            "metric": "val_map50",
            "role": "comparability_only",
            "direction": "higher",
            "sources": ["caller"],
        }
    ]

    # An explicit, legitimate metric still succeeds: a rail must admit valid work.
    res = rank_registered_models(
        project, metric="val_map50", include_unverified=True
    )
    assert res["name"] == "b"
    assert res["ranking_basis"] == "val_map50"
    assert (res["higher_is_better"], res["direction_source"]) == (
        True,
        "declared",
    )

    # A declared metric no model carries: a distinct error listing what's actually available.
    res = rank_registered_models(
        project, metric="val_loss", include_unverified=True
    )
    assert "No registered model has metric" in res["error"]
    assert res["available_metrics"][0]["metric"] == "val_map50"
    assert res["n_models"] == 2

    # A metric with no declared ranking direction: refused before ever looking for it.
    res = rank_registered_models(project, metric="val_map99")
    assert "no declared ranking direction" in res["error"]


def test_rank_registered_models_excludes_unverified_entries_by_default(
    tmp_path: Path,
) -> None:
    """Ranking excludes unverified entries by default and ranks them with ``include_unverified``."""
    from tcip_mcp.tools.model_tools import rank_registered_models

    _registered(tmp_path, ("asserted", {"val_map50": 0.99}))

    res = rank_registered_models(tmp_path, metric="val_map50")
    assert "unverified" in res["error"]
    assert res["excluded_unverified"] == [
        {"name": "asserted", "metrics_source": "caller"}
    ]

    res = rank_registered_models(
        tmp_path, metric="val_map50", include_unverified=True
    )
    assert res["name"] == "asserted"
    assert res["unverified_included"] is True
    assert res["excluded_unverified"] == []


def test_rank_registered_models_refusals_name_no_argument_a_breeder_would_not_pass(
    tmp_path: Path,
) -> None:
    """The undeclared-direction and all-unverified ranking refusals name no tool parameter."""
    from tcip_mcp.tools.model_tools import rank_registered_models

    _registered(tmp_path, ("asserted", {"val_map50": 0.99}))

    no_direction = rank_registered_models(tmp_path, metric="val_map99")[
        "error"
    ]
    all_unverified = rank_registered_models(tmp_path, metric="val_map50")[
        "error"
    ]

    assert no_direction == (
        "'val_map99' has no declared ranking direction (schemas.HIGHER_IS_BETTER_BY_METRIC "
        "names no entry for it). State a direction to rank by it anyway, or pick one of the "
        "available metrics."
    )
    assert all_unverified == (
        "every registered model carrying 'val_map50' is unverified (metrics_source is not "
        "'trainer'); include unverified models to rank them, or register a verified run."
    )
    for text in (no_direction, all_unverified):
        for parameter in (
            "higher_is_better",
            "include_unverified",
            "available_metrics",
            "rank_registered_models",
        ):
            assert parameter not in text


def test_register_model_refuses_a_nonexistent_checkpoint(
    tmp_path: Path,
) -> None:
    """``register_model`` refuses a nonexistent checkpoint and stores no entry."""
    from tcip_mcp.model_registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        reg.register_model(
            "ghost", str(tmp_path / "nonexistent.pt"), actor=None
        )
    assert reg.list_models() == []


def test_register_model_refuses_bytes_the_verified_reader_refuses(
    tmp_path: Path,
) -> None:
    """``register_model`` refuses bytes the verified reader refuses.

    Bytes that are no checkpoint refuse with nothing stored, and a readable checkpoint
    registers.
    """
    from tcip_mcp.model_registry import (
        ModelRegistry,
        UnregisteredCheckpointError,
    )

    reg = ModelRegistry(str(tmp_path))
    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"not a checkpoint")

    with pytest.raises(UnregisteredCheckpointError):
        reg.register_model("garbage", str(garbage), actor=None)
    assert reg.list_models() == []

    reg.register_model("real", _checkpoint(tmp_path, "real"), actor=None)
    assert [m.entry.name for m in reg.list_models()] == ["real"]


def test_a_foreign_registrations_source_is_read_off_whether_it_carries_metrics(
    tmp_path: Path,
) -> None:
    """A foreign registration's source reads ``caller`` with metrics and ``None`` without."""
    from tcip_mcp.model_registry import ModelRegistry
    from tcip_mcp.tools.model_tools import register_model

    register_model(
        name="a",
        checkpoint_path=_checkpoint(tmp_path, "a"),
        project=tmp_path,
        metrics={"val_map50": 0.5},
        actor=None,
    )
    register_model(
        name="b",
        checkpoint_path=_checkpoint(tmp_path, "b"),
        project=tmp_path,
        actor=None,
    )

    reg = ModelRegistry(str(tmp_path))
    assert _named(reg, "a").facts.metrics_source == "caller"
    assert _named(reg, "b").facts.metrics_source is None


def test_best_model_lower_is_better_for_loss(tmp_path: Path) -> None:
    from tcip_mcp.model_registry import best_model

    models = _registered(
        tmp_path, ("hi", {"val_loss": 0.9}), ("lo", {"val_loss": 0.2})
    ).list_models()

    best = best_model(
        models, "val_loss", higher_is_better=False, include_unverified=True
    )
    assert best is not None and best.entry.name == "lo"
    # A metric no model has: None, cleanly distinguishable from "no models".
    assert (
        best_model(
            models,
            "nonexistent",
            higher_is_better=False,
            include_unverified=True,
        )
        is None
    )


def test_best_model_excludes_an_entry_with_no_metrics_from_ranking(
    tmp_path: Path,
) -> None:
    """``best_model`` excludes an entry with no metrics from ranking.

    Beside an entry carrying ``val_loss``, an entry registered with no metrics raises nothing
    and is passed over, and the entry carrying the metric is ranked best.
    """
    from tcip_mcp.model_registry import best_model

    models = _registered(
        tmp_path, ("empty", None), ("real", {"val_loss": 0.5})
    ).list_models()

    best = best_model(
        models, "val_loss", higher_is_better=False, include_unverified=True
    )
    assert best is not None and best.entry.name == "real"


def test_a_completed_runs_metrics_come_from_its_best_checkpoint_not_its_last_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed run's metrics come from its best checkpoint, not its last epoch.

    A run the default trainer completes names its best checkpoint, whose metrics are that
    epoch's own, sourced ``trainer``; the registry lists it under the run's id.
    """
    torch = pytest.importorskip("torch")
    from tcip_mcp.experiments import METRICS_FILE, partition_rows, read_rows
    from tcip_mcp.model_registry import ModelRegistry
    from tests._training_values import drawn_split
    from tests._verified_checkpoint_fixtures import worker_run
    from tests.tiny_trainer_fixtures import (
        regressor_config,
        write_regression_dataset,
    )

    images_dir, csv_path = write_regression_dataset(
        tmp_path, intensities=[0.1, 0.3, 0.5, 0.7], values=[0.2, 0.6, 1.0, 1.4]
    )
    run_dir = worker_run(
        tmp_path,
        regressor_config(
            3,
            data={
                "images_dir": str(images_dir),
                "labels_dir": str(csv_path),
                "split": drawn_split(val_ratio=0.0),
            },
            project=tmp_path,
        ),
        experiment_id="exp-best",
    )

    facts = _named(ModelRegistry(str(tmp_path)), "exp-best").facts
    assert facts.metrics_source == "trainer"
    best = torch.load(run_dir / "model_best.pt", weights_only=False)
    (row,) = [
        r
        for r in partition_rows(read_rows(run_dir / METRICS_FILE)[0])[0]
        if r["epoch"] == best["epoch"]
    ]
    assert facts.metrics["selection"] == row["selection"]


def test_a_semantic_seg_runs_checkpoint_lists_with_its_per_class_metrics(
    tmp_path: Path,
) -> None:
    """A semantic segmentation run's checkpoint lists with the per-class metrics it carries.

    A run the default trainer completes over three classes with a validation side, its three
    training images in one batch, registers its best checkpoint, and the listing answers its
    ``val_per_class_iou`` keyed by each class.
    """
    pytest.importorskip("torch")
    from tcip_mcp.experiments import observe
    from tcip_mcp.tools.model_tools import registered_listing
    from tests._chain_fixtures import (
        BESPOKE_MODELS,
        BESPOKE_SEMANTIC_SEG,
        FRAME_STATISTICS,
        training_config,
    )
    from tests._training_values import drawn_split, evaluation_block
    from tests._verified_checkpoint_fixtures import worker_run
    from tests.test_mask_and_table_membership import _three_class_masks

    images_dir, masks_dir, _masks = _three_class_masks(tmp_path / "ds")
    run_dir = worker_run(
        tmp_path,
        training_config(
            tmp_path,
            {
                "builder": BESPOKE_SEMANTIC_SEG,
                "source_files": [BESPOKE_MODELS],
                "task": "semantic_seg",
                "normalization": FRAME_STATISTICS,
            },
            {
                "images_dir": str(images_dir),
                "labels_dir": str(masks_dir),
                "split": drawn_split(val_ratio=0.25, seed=3),
            },
            batch_size=3,
            evaluation=evaluation_block(selection_metric="loss"),
        ),
        experiment_id="exp-seg",
    )
    observation = observe(run_dir)
    assert observation.state == "completed", observation.final

    (listed,) = [
        m
        for m in registered_listing(tmp_path)["models"]
        if m["name"] == "exp-seg"
    ]
    assert set(listed["metrics"]["val_per_class_iou"]) == {"0", "1", "2"}


UNENCODABLE = "bespoke_models:save_built_weights_with_unencodable_metrics"
"""A ``training_source`` saving the ``bespoke_models.UNENCODABLE_METRICS`` case its config names."""


@pytest.mark.parametrize(
    ("body", "stated", "named"),
    [
        (
            UNENCODABLE,
            {"fixture_unencodable": "integer_key"},
            "val_per_class_iou",
        ),
        (
            UNENCODABLE,
            {"fixture_unencodable": "non_finite"},
            "val_loss",
        ),
        (
            UNENCODABLE,
            {"fixture_unencodable": "bytes_key"},
            "b'val_loss'",
        ),
        (
            "bespoke_models:save_built_weights",
            {"fixture_metrics": [0.5]},
            "dict",
        ),
        (
            "bespoke_models:save_built_weights",
            {"fixture_metrics": 0.5},
            "dict",
        ),
    ],
    ids=["integer_key", "non_finite", "bytes_key", "list", "scalar"],
)
def test_a_bespoke_checkpoint_whose_metrics_the_registry_cannot_read_is_refused_at_its_write(
    tmp_path: Path, body: str, stated: dict, named: str
) -> None:
    """A bespoke loop's checkpoint whose metrics the registry cannot read is refused as written.

    Metrics keyed by an integer or by bytes, carrying a ``nan``, or not a mapping at all fail the
    run naming what refused, no checkpoint is written, and the listing answers.
    """
    pytest.importorskip("torch")
    from tcip_mcp.experiments import observe
    from tcip_mcp.pipelines.training.generic_trainer import checkpoint_path
    from tcip_mcp.tools.model_tools import registered_listing
    from tests._verified_checkpoint_fixtures import finished_run

    run_dir = finished_run(tmp_path, training_source=body, **stated)

    final = observe(run_dir).final
    assert final is not None and final.state == "failed", final
    assert named in (final.status_error or "")
    assert not checkpoint_path(run_dir, "model_final").exists()
    assert registered_listing(tmp_path)["count"] == 0


def test_a_bespoke_checkpoint_whose_metrics_json_holds_writes_and_lists(
    tmp_path: Path,
) -> None:
    """A bespoke loop's checkpoint carrying JSON metrics writes, and lists with those metrics."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.model_tools import registered_listing
    from tests._verified_checkpoint_fixtures import finished_run

    finished_run(
        tmp_path,
        experiment_id="exp-bespoke",
        metrics={"val_per_class_iou": {"0": 0.5}},
    )

    (listed,) = registered_listing(tmp_path)["models"]
    assert (listed["name"], listed["metrics"]) == (
        "exp-bespoke",
        {"val_per_class_iou": {"0": 0.5}},
    )


@pytest.mark.parametrize(
    "stated", [None, [], 0, False], ids=["none", "list", "zero", "false"]
)
def test_a_checkpoints_metrics_are_absent_or_a_mapping_never_read_as_empty(
    stated: object,
) -> None:
    """A checkpoint carrying no metrics reads as none; one stating a non-mapping is refused.

    ``checkpoint_metrics`` reads a payload without ``METRICS_KEY`` as an empty mapping and refuses
    ``None``, a list, a zero or a boolean stated under it rather than reading it as empty.
    """
    from pydantic import ValidationError

    from tcip_mcp.model_registry import checkpoint_metrics
    from tcip_mcp.pipelines.model_build import METRICS_KEY

    assert checkpoint_metrics({}) == {}
    with pytest.raises(ValidationError):
        checkpoint_metrics({METRICS_KEY: stated})


def test_an_entrys_metrics_changed_after_admission_are_refused_where_they_are_read(
    tmp_path: Path,
) -> None:
    """A registered entry's metrics changed in memory to ``nan`` are refused by ``entry_facts``.

    The entry is a foreign registration read back from the registry index, its checkpoint path
    resolved; its nested metrics are changed after it was admitted.
    """
    from pydantic import ValidationError

    from tcip_mcp.model_registry import entry_facts, read_registry_index
    from tcip_mcp.registry_paths import resolved_registry_path

    _registered(tmp_path, ("a", {"val_loss": 0.5}))
    (stored,) = read_registry_index(tmp_path)
    entry = stored.model_copy(
        update={
            "checkpoint_path": str(
                resolved_registry_path(str(tmp_path), stored.checkpoint_path)
            )
        }
    )
    assert entry_facts(entry, tmp_path).metrics == {"val_loss": 0.5}

    entry.metrics["val_loss"] = float("nan")

    with pytest.raises(ValidationError, match="val_loss"):
        entry_facts(entry, tmp_path)


def test_a_registry_payload_that_json_cannot_hold_is_refused_at_register_model(
    tmp_path: Path,
) -> None:
    """A registry payload JSON cannot hold is refused at ``register_model``.

    An infinite metric refuses naming ``metrics.val_map50``, and the registry lists nothing.
    """
    from pydantic import ValidationError

    from tcip_mcp.model_registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path))
    ckpt = _checkpoint(tmp_path, "m")

    with pytest.raises(ValidationError) as metrics_refused:
        reg.register_model(
            "a", str(ckpt), metrics={"val_map50": float("inf")}, actor=None
        )
    assert "metrics.val_map50" in str(metrics_refused.value)

    assert reg.list_models() == []


def test_an_ordinary_registry_payload_is_still_registered(
    tmp_path: Path,
) -> None:
    """An ordinary metrics payload registers, lists and ranks."""
    from tcip_mcp.model_registry import ModelRegistry, best_model

    reg = ModelRegistry(str(tmp_path))
    listed = reg.register_model(
        "a",
        _checkpoint(tmp_path, "a"),
        metrics={"val_map50": 0.70},
        actor=None,
    )

    assert listed.entry.name == "a"
    assert [m.entry.name for m in reg.list_models()] == ["a"]
    best = best_model(
        reg.list_models(),
        "val_map50",
        higher_is_better=True,
        include_unverified=True,
    )
    assert best is not None and best.facts.metrics["val_map50"] == 0.70
