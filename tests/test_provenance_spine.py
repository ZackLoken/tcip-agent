"""The provenance identity spine.

Locks the provenance across the spine: a checkpoint's producing run read off the run's own
final status beside a computed-once sha256, the enriched capture_env, a selection's own digests
and seed, and the producing-model stamps on the delivery CSV and manifest surfaces.
"""

from __future__ import annotations

import pytest


# ── capture_env records code + library fingerprint ────────────────────────────

def test_capture_env_records_code_and_libraries():
    from tcip_mcp.pipelines.model_build import capture_env

    env = capture_env()
    # git commit + numpy + CUDA are present as keys (values best-effort/null), never fatal.
    assert "tcip_git_commit" in env
    assert "numpy" in env
    assert "cuda" in env
    assert env["python"]


# ── identity resolved off a verified, registry-matched checkpoint ──────────────

def test_resolve_model_identity_from_the_runs_final_status(tmp_path, monkeypatch):
    """A completed run's checkpoint resolves its producer through the binding the run's own
    final status recorded, not a caller-asserted tag or a stamp in the payload."""
    pytest.importorskip("torch")
    from tcip_mcp.model_registry import load_registered_checkpoint, resolve_model_identity
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = registered_checkpoint(None, experiment_id="expR")

    checkpoint = load_registered_checkpoint(ckpt, project_path=str(tmp_path))
    ident = resolve_model_identity(checkpoint)
    assert ident["sha256"] and len(ident["sha256"]) == 64
    assert ident["experiment_id"] == "expR"
    assert ident["checkpoint"] == "model_final"


def test_resolve_model_identity_foreign_checkpoint(tmp_path):
    """A registered checkpoint with no producing experiment (explicit-mode registration, no
    stamp) resolves the sha and leaves ``experiment_id`` null rather than failing."""
    torch = pytest.importorskip("torch")
    from tcip_mcp.model_registry import (
        ModelRegistry, load_registered_checkpoint, resolve_model_identity,
    )

    ckpt = tmp_path / "foreign.pt"
    torch.save({"model_state_dict": {}}, ckpt)
    ModelRegistry(str(tmp_path)).register_model("foreign", str(ckpt), {})

    checkpoint = load_registered_checkpoint(ckpt, project_path=str(tmp_path))
    ident = resolve_model_identity(checkpoint)
    assert ident["sha256"]                 # sha still recorded
    assert ident["experiment_id"] is None  # no run -> honest null, not a failure


def test_a_payloads_own_experiment_id_names_no_producer(tmp_path):
    """A checkpoint payload that states an ``experiment_id`` of its own is a claim nothing
    answers for: the producer is only ever the run whose final status names the digest, so a
    foreign checkpoint resolves none, whatever its payload says."""
    torch = pytest.importorskip("torch")
    from tcip_mcp.model_registry import (
        ModelRegistry, load_registered_checkpoint, resolve_model_identity,
    )

    ckpt = tmp_path / "stamped.pt"
    torch.save({"model_state_dict": {}, "experiment_id": "expStamped"}, ckpt)
    ModelRegistry(str(tmp_path)).register_model("stamped", str(ckpt), {})

    checkpoint = load_registered_checkpoint(ckpt, project_path=str(tmp_path))
    ident = resolve_model_identity(checkpoint)
    assert ident["experiment_id"] is None
    assert ident["sha256"]


# ── draw_splits manifest embeds dataset_hash + seed ─────────────────────────────

def test_draw_splits_selection_embeds_digests_and_seed(data_dir, tmp_path):
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.tools.data_tools import draw_splits

    # A selection needs at least four foreground groups to clear the floor; the fixture's own
    # three (img_001..003) need one more, added here rather than in the shared fixture.
    from PIL import Image

    images_dir = data_dir / "images" / "2-11-26"
    labels_dir = data_dir / "annotations" / "2-11-26"
    Image.new("RGB", (640, 480), color=(128, 128, 128)).save(images_dir / "img_004.jpg")
    json_io.write_annotations(
        labels_dir / "img_004.json",
        [Annotation(subject="bud", geometry=BBox(288, 216, 352, 264))], 640, 480,
    )

    out = tmp_path / "splits"
    result = draw_splits(str(data_dir), output_path=str(out), seed=7, subject="bud",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert result["seed"] == 7
    drawn = read_selection(out)
    assert drawn.seed == 7
    assert all(sample.ground_truth_digest for sample in drawn.samples)
    assert set(drawn.counts()) == {"train", "val", "calibration"}


# ── delivery CSVs carry the producing-model provenance columns ─────────────────

def _run_with_a_recorded_checkpoint(tmp_path, experiment_id):
    """A run whose own record answers for the checkpoint a delivery names it by."""
    from tests._binding_fixtures import record_producing_run

    return record_producing_run(experiment_id)


def test_export_detection_csv_carries_provenance(tmp_path):
    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv
    from tcip_mcp.pipelines.resolution import VALIDATED_HELD_OUT

    from tests import _trait_fixtures as fx
    from tests._binding_fixtures import write_bound_sidecar, write_prediction

    revision = fx.seed_confirmed_count(tmp_path)
    sha = _run_with_a_recorded_checkpoint(tmp_path, "expE")
    # This door takes no acknowledgment, so the delivery is made genuinely validated: a real
    # bucket bound to the checkpoint that produced it, rather than a provisional escape.
    root = tmp_path / "ds"
    bucket = root / "predictions" / "preds"
    write_prediction(bucket, "img_a")
    stamp = {
        "scope": {"subject": fx.COUNT_SUBJECT, "attribute": None},
        "validated": True, "trait": fx.COUNT_TRAIT, "checkpoint_sha256": sha,
        "operating_point": {"conf": {"value": 0.4, "requires_validation": True,
                                     "validation_kind": "annotations",
                                     "validated_against": VALIDATED_HELD_OUT}},
    }
    write_bound_sidecar(bucket, stamp, dataset_root=root, producing_experiment_id="expE")
    out = tmp_path / "counts.csv"
    export_detection_csv(
        [{"image": "a.jpg", "count": 3, "scores": [0.9, 0.8, 0.7]}], str(out),
        revision=revision, provenance={"operating_point_conf": 0.42}, pred_dirs=[str(bucket)])
    rows = list(__import__("csv").DictReader(out.open()))
    assert rows[0]["producer_model_sha256"] == sha
    assert rows[0]["producing_experiment_id"] == "expE"
    assert rows[0]["operating_point_conf"] == "0.42"


def test_export_aggregated_csv_carries_provenance(tmp_path):
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv
    from tcip_mcp.pipelines.resolution import VALIDATED_HELD_OUT

    from tests import _trait_fixtures as fx
    from tests._binding_fixtures import write_bound_sidecar, write_prediction

    fx.seed_delivery_traits(tmp_path)
    fx.seed_confirmed_aggregate(tmp_path, "stem_count", value_keys=["count"])
    sha = _run_with_a_recorded_checkpoint(tmp_path, "expA")
    # This door takes no acknowledgment either, so the same real-bucket route as above
    # is what earns a genuinely validated delivery.
    root = tmp_path / "ds"
    bucket = root / "predictions" / "preds"
    write_prediction(bucket, "img_a")
    stamp = {
        "scope": {"subject": fx.COUNT_SUBJECT, "attribute": None},
        "validated": True, "trait": fx.COUNT_TRAIT, "checkpoint_sha256": sha,
        "operating_point": {"conf": {"value": 0.4, "requires_validation": True,
                                     "validation_kind": "annotations",
                                     "validated_against": VALIDATED_HELD_OUT}},
    }
    write_bound_sidecar(bucket, stamp, dataset_root=root, producing_experiment_id="expA")
    out = tmp_path / "agg.csv"
    export_aggregated_csv(
        [{"plant_id": "p1", "value": 5, "observations": 2, "value_key": "count",
          "measurement_document": "operating_point", "plant_attribution": "image"}],
        str(out), delivered_phenotype="stem_count", pred_dirs=[str(bucket)])
    rows = list(__import__("csv").DictReader(out.open()))
    assert rows[0]["producer_model_sha256"] == sha
    assert rows[0]["producing_experiment_id"] == "expA"


def test_export_aggregated_csvs_produced_at_is_the_write_time_never_a_buckets_own(tmp_path):
    """A bucket's own operating-point sidecar can carry a ``produced_at`` of its own (the run
    that produced it stamped one); the delivered CSV's own ``produced_at`` column is always
    ``delivered_tail``'s write-time timestamp, never that value, since it is computed fresh on
    every delivery and never read off a bucket. Coverage, not a regression guard."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv
    from tcip_mcp.pipelines.resolution import VALIDATED_HELD_OUT
    from tests import _trait_fixtures as fx
    from tests._binding_fixtures import write_bound_sidecar, write_prediction

    fx.seed_delivery_traits(tmp_path)
    fx.seed_confirmed_aggregate(tmp_path, "stem_count", value_keys=["count"])
    root = tmp_path / "ds"
    bucket = root / "predictions" / "preds"
    write_prediction(bucket, "img_a")
    stamp = {
        "scope": {"subject": fx.COUNT_SUBJECT, "attribute": None},
        "validated": True, "trait": fx.COUNT_TRAIT,
        "operating_point": {"conf": {"value": 0.4, "requires_validation": True,
                                     "validation_kind": "annotations",
                                     "validated_against": VALIDATED_HELD_OUT}},
        "produced_at": "2020-01-01T00:00:00+00:00",
    }
    write_bound_sidecar(bucket, stamp, dataset_root=root, experiment_id="exp-old-stamp")
    out = tmp_path / "agg.csv"

    export_aggregated_csv(
        [{"plant_id": "p1", "value": 5, "observations": 2, "value_key": "count",
          "measurement_document": "operating_point", "plant_attribution": "image"}],
        str(out), delivered_phenotype="stem_count", pred_dirs=[str(bucket)])

    rows = list(__import__("csv").DictReader(out.open()))
    assert rows[0]["produced_at"] != "2020-01-01T00:00:00+00:00"


def test_export_detection_csvs_produced_at_is_present_and_iso_parseable(tmp_path):
    """The detection CSV's own ``produced_at`` cell is a real write-time timestamp, not merely a
    non-empty string."""
    from datetime import datetime

    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv
    from tcip_mcp.pipelines.resolution import VALIDATED_HELD_OUT
    from tests import _trait_fixtures as fx
    from tests._binding_fixtures import write_bound_sidecar, write_prediction

    revision = fx.seed_confirmed_count(tmp_path)
    # This door takes no acknowledgment, so the delivery is made genuinely validated.
    root = tmp_path / "ds"
    bucket = root / "predictions" / "preds"
    write_prediction(bucket, "img_a")
    stamp = {
        "scope": {"subject": fx.COUNT_SUBJECT, "attribute": None},
        "validated": True, "trait": fx.COUNT_TRAIT,
        "operating_point": {"conf": {"value": 0.4, "requires_validation": True,
                                     "validation_kind": "annotations",
                                     "validated_against": VALIDATED_HELD_OUT}},
    }
    write_bound_sidecar(bucket, stamp, dataset_root=root)
    out = tmp_path / "counts.csv"

    export_detection_csv(
        [{"image": "a.jpg", "count": 3, "scores": [0.9, 0.8, 0.7]}], str(out),
        revision=revision, pred_dirs=[str(bucket)])

    rows = list(__import__("csv").DictReader(out.open()))
    datetime.fromisoformat(rows[0]["produced_at"])


def test_delivered_tail_treats_a_none_valued_produced_at_key_as_absent(tmp_path):
    """A caller composing its own asserted dict over a sidecar carrying no ``produced_at`` of its
    own (``{"produced_at": sidecar.get("produced_at")}``) carries the key with a ``None`` value,
    never a caller-stated one; ``delivered_tail`` must not refuse that the way it refuses a
    caller that actually asserts a ``produced_at``, matching ``corroborated_producer``'s own
    absence convention two functions up."""
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_FALSE, Acknowledgment, check_delivery_gate, delivered_tail,
    )

    gate = check_delivery_gate(
        {"operating_point": VALIDATED_FALSE},
        acknowledgment=Acknowledgment(acknowledged_by="user:tester", reason="test acknowledgment"))
    columns = ("produced_at", "operating_point_validated")

    tail = delivered_tail({"produced_at": None}, {}, gate, columns=columns)
    assert tail["produced_at"]  # the write's own timestamp, not refused

    with pytest.raises(ValueError, match="produced_at"):
        delivered_tail({"produced_at": "2020-01-01T00:00:00+00:00"}, {}, gate, columns=columns)


# ── phenology CSV schema carries producing-model identity ──────────────────────

def test_phenology_columns_include_producer_identity():
    from tcip_mcp.pipelines.postprocessing.phenology import phenology_csv_columns
    from tests._trait_fixtures import BUD_OPENING

    columns = phenology_csv_columns(BUD_OPENING)
    assert "producer_model_sha256" in columns
    assert "producing_experiment_id" in columns
    assert "validation_record" in columns
