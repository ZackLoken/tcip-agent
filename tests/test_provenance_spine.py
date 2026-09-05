"""The provenance identity spine.

Locks the additive provenance stamping across the spine: checkpoint experiment_id + a
computed-once sha256, the terminal-state lock (additive-only), the enriched capture_env,
the split manifest, and the producing-model stamps on the delivery CSV/manifest surfaces.
These are provenance additions, no measurement changes.
"""

from __future__ import annotations

import pytest


# ── R4: capture_env records code + library fingerprint ────────────────────────

def test_capture_env_records_code_and_libraries():
    from tcip_mcp.pipelines.model_build import capture_env

    env = capture_env()
    # git commit + numpy + CUDA are present as keys (values best-effort/null), never fatal.
    assert "tcip_git_commit" in env
    assert "numpy" in env
    assert "cuda" in env
    assert env["python"]


# ── R1: stamp_model_ref carries experiment_id (optional) ──────────────────────

def test_stamp_model_ref_stamps_experiment_id():
    from tcip_mcp.pipelines.model_build import stamp_model_ref

    src = {"model_source": {"builder": "x:y", "task": "detection"}}
    payload = stamp_model_ref({"model_state_dict": {}}, src, experiment_id="expZ")
    assert payload["experiment_id"] == "expZ"

    # From config when not passed explicitly.
    payload2 = stamp_model_ref({"model_state_dict": {}}, {**src, "experiment_id": "expC"})
    assert payload2["experiment_id"] == "expC"

    # Absent id -> no key fabricated (raw/foreign checkpoints legitimately have none).
    payload3 = stamp_model_ref({"model_state_dict": {}}, src)
    assert "experiment_id" not in payload3


# ── R2: identity resolved off a verified, registry-matched checkpoint ───────

def test_resolve_model_identity_from_registry(tmp_path, monkeypatch):
    """A registered checkpoint that carries no stamped experiment_id still resolves one, through
    the binding a run's own completion recorded (``checkpoint.producer``), not a caller-asserted
    tag."""
    torch = pytest.importorskip("torch")
    from tcip_mcp.experiments import complete_run, create_experiment, register_model_from_experiment
    from tcip_mcp.model_registry import load_registered_checkpoint, resolve_model_identity

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = tmp_path / "best.pt"
    torch.save({"model_state_dict": {}}, ckpt)
    create_experiment("expR", {"model_source": {"builder": "x:y"}})
    assert "error" not in complete_run("expR", str(ckpt))
    reg = register_model_from_experiment("expR", str(ckpt), name="m1")
    assert "error" not in reg, reg

    checkpoint = load_registered_checkpoint(ckpt, project_path=str(tmp_path))
    ident = resolve_model_identity(checkpoint)
    assert ident["sha256"] and len(ident["sha256"]) == 64
    assert ident["experiment_id"] == "expR"
    assert ident["checkpoint"] == "best"


def test_resolve_model_identity_foreign_checkpoint(tmp_path):
    """A registered checkpoint with no producing experiment (explicit-mode registration, no
    stamp) resolves the sha and leaves ``experiment_id`` null rather than failing."""
    torch = pytest.importorskip("torch")
    from tcip_mcp.model_registry import (
        ModelRegistry, load_registered_checkpoint, resolve_model_identity,
    )

    ckpt = tmp_path / "foreign.pt"
    torch.save({"model_state_dict": {}}, ckpt)
    ModelRegistry(str(tmp_path)).register_model("foreign", str(ckpt), {}, metrics_source=None)

    checkpoint = load_registered_checkpoint(ckpt, project_path=str(tmp_path))
    ident = resolve_model_identity(checkpoint)
    assert ident["sha256"]                 # sha still recorded
    assert ident["experiment_id"] is None  # no run -> honest null, not a failure


def test_resolve_model_identity_reads_checkpoint_own_experiment_id(tmp_path):
    """The ordinary train-then-calibrate workflow saves a checkpoint stamped with its own
    ``experiment_id`` (via ``stamp_model_ref``); ``resolve_model_identity`` must read that stamp
    directly rather than resolving it only from the registry's own ``experiment:`` tag, which
    would otherwise silently bypass the train-disjointness gate whenever the two disagree."""
    torch = pytest.importorskip("torch")
    from tcip_mcp.model_registry import (
        ModelRegistry, load_registered_checkpoint, resolve_model_identity,
    )

    ckpt = tmp_path / "stamped.pt"
    torch.save({"model_state_dict": {}, "experiment_id": "expStamped"}, ckpt)
    ModelRegistry(str(tmp_path)).register_model("stamped", str(ckpt), {}, metrics_source=None)

    checkpoint = load_registered_checkpoint(ckpt, project_path=str(tmp_path))
    ident = resolve_model_identity(checkpoint)
    assert ident["experiment_id"] == "expStamped"
    assert ident["sha256"]


def test_resolve_model_identity_caller_experiment_id_wins_over_stamp(tmp_path):
    torch = pytest.importorskip("torch")
    from tcip_mcp.model_registry import (
        ModelRegistry, load_registered_checkpoint, resolve_model_identity,
    )

    ckpt = tmp_path / "stamped.pt"
    torch.save({"model_state_dict": {}, "experiment_id": "expStamped"}, ckpt)
    ModelRegistry(str(tmp_path)).register_model("stamped", str(ckpt), {}, metrics_source=None)

    checkpoint = load_registered_checkpoint(ckpt, project_path=str(tmp_path))
    ident = resolve_model_identity(checkpoint, experiment_id="expCaller")
    assert ident["experiment_id"] == "expCaller"


# ── R3: terminal-state lock is additive-only ───────────────────────────────

@pytest.fixture()
def exp_store(tmp_path, monkeypatch):
    import tcip_mcp.experiments as exp
    monkeypatch.setattr(exp, "EXPERIMENTS_DIR", tmp_path / "experiments")
    # Route the refused-mutation audit to the tmp project so it never touches the repo log.
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    return exp


def test_update_status_refuses_to_leave_terminal(exp_store):
    import tcip_store as ts

    exp = exp_store
    exp.create_experiment("t1", {})
    exp.update_status("t1", "completed")
    res = exp.update_status("t1", "running")
    assert "error" in res
    assert ts.read(exp.status_key("t1"))["state"] == "completed"


def test_log_metrics_refuses_new_epoch_when_terminal(exp_store):
    exp = exp_store
    exp.create_experiment("t2", {})
    exp.log_metrics("t2", 0, {"loss": 1.0})
    exp.update_status("t2", "completed")
    res = exp.log_metrics("t2", 1, {"loss": 0.5})
    assert "error" in res
    assert len(exp.read_metrics("t2")) == 1  # no new epoch appended


def test_lineage_additive_first_write_allowed_overwrite_refused(exp_store):
    import tcip_store as ts

    exp = exp_store
    exp.create_experiment("t3", {})
    exp.update_status("t3", "completed")
    # First write into a still-empty field is permitted (R1's predictions link relies on this).
    exp.update_lineage("t3", predictions="/preds/run")
    lin = ts.read(exp.lineage_key("t3"))
    assert lin["predictions"] == "/preds/run"
    # Overwriting the now-populated field is refused; the recorded value stands.
    exp.update_lineage("t3", predictions="/preds/other")
    lin2 = ts.read(exp.lineage_key("t3"))
    assert lin2["predictions"] == "/preds/run"


def test_record_artifact_additive_only_when_terminal(exp_store):
    import tcip_store as ts

    exp = exp_store
    exp.create_experiment("t4", {})
    exp.update_status("t4", "completed")
    exp.record_artifact("t4", "predictions", "/preds")   # new name -> allowed
    res = exp.record_artifact("t4", "predictions", "/other")  # existing -> refused
    assert "error" in res
    arts = ts.read(exp.artifacts_key("t4"))
    assert arts["predictions"]["path"] == "/preds"


# ── R5: draw_splits manifest embeds dataset_hash + seed ───────────────────────

def test_draw_splits_manifest_embeds_hash_and_seed(data_dir, tmp_path):
    import tcip_store as ts
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.data_tools import draw_splits, split_manifest_key

    # A manifest write needs at least four foreground groups to clear the floor; the fixture's
    # own three (img_001..003) need one more, added here rather than in the shared fixture.
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
    manifest = ts.read(split_manifest_key(out))
    assert manifest["seed"] == 7
    assert manifest["members"]["2-11-26"]["dataset_hash"]
    assert set(manifest["splits"]) == {"train", "val", "calibration"}


# ── R2: delivery CSVs carry the producing-model provenance columns ────────────

def _run_with_a_recorded_checkpoint(tmp_path, experiment_id):
    """A run whose own record answers for the checkpoint a delivery names it by."""
    from tests._binding_fixtures import record_producing_run

    return record_producing_run(tmp_path, experiment_id)


def test_export_detection_csv_carries_provenance(tmp_path):
    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv
    from tcip_mcp.pipelines.resolution import VALIDATED_HELD_OUT

    from tests import _operationalization_fixtures as fx
    from tests._binding_fixtures import write_bound_sidecar, write_prediction

    fx.seed_confirmed_count(tmp_path)
    sha = _run_with_a_recorded_checkpoint(tmp_path, "expE")
    # This door takes no acknowledgement, so the delivery is made genuinely validated: a real
    # bucket bound to the checkpoint that produced it, rather than a provisional escape.
    root = tmp_path / "ds"
    bucket = root / "predictions" / "preds"
    write_prediction(bucket, "img_a")
    stamp = {
        "subject": fx.COUNT_SUBJECT, "attribute": None,
        "validated": True, "trait": fx.COUNT_TRAIT, "checkpoint_sha256": sha,
        "operating_point": {"conf": {"value": 0.4, "requires_validation": True,
                                     "validation_kind": "annotations",
                                     "validated_against": VALIDATED_HELD_OUT}},
    }
    write_bound_sidecar(bucket, stamp, dataset_root=root, producing_experiment_id="expE")
    out = tmp_path / "counts.csv"
    export_detection_csv(
        [{"image": "a.jpg", "count": 3, "scores": [0.9, 0.8, 0.7]}], str(out),
        trait=fx.COUNT_TRAIT, provenance={"operating_point_conf": 0.42}, pred_dirs=[str(bucket)])
    rows = list(__import__("csv").DictReader(out.open()))
    assert rows[0]["producer_model_sha256"] == sha
    assert rows[0]["producing_experiment_id"] == "expE"
    assert rows[0]["operating_point_conf"] == "0.42"


def test_export_aggregated_csv_carries_provenance(tmp_path):
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv
    from tcip_mcp.pipelines.resolution import VALIDATED_HELD_OUT

    from tests import _operationalization_fixtures as fx
    from tests._binding_fixtures import write_bound_sidecar, write_prediction

    fx.seed_delivery_traits(tmp_path)
    fx.seed_confirmed_aggregate(tmp_path, "stem_count", value_keys=["count"])
    sha = _run_with_a_recorded_checkpoint(tmp_path, "expA")
    # This door takes no acknowledgement either, so the same real-bucket route as above
    # is what earns a genuinely validated delivery.
    root = tmp_path / "ds"
    bucket = root / "predictions" / "preds"
    write_prediction(bucket, "img_a")
    stamp = {
        "subject": fx.COUNT_SUBJECT, "attribute": None,
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
    from tests import _operationalization_fixtures as fx
    from tests._binding_fixtures import write_bound_sidecar, write_prediction

    fx.seed_delivery_traits(tmp_path)
    fx.seed_confirmed_aggregate(tmp_path, "stem_count", value_keys=["count"])
    root = tmp_path / "ds"
    bucket = root / "predictions" / "preds"
    write_prediction(bucket, "img_a")
    stamp = {
        "subject": fx.COUNT_SUBJECT, "attribute": None,
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
    """Coverage for the family's stated ``produced_at`` meaning: the detection CSV's own cell is
    a real write-time timestamp, not merely a non-empty string."""
    from datetime import datetime

    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv
    from tcip_mcp.pipelines.resolution import VALIDATED_HELD_OUT
    from tests import _operationalization_fixtures as fx
    from tests._binding_fixtures import write_bound_sidecar, write_prediction

    fx.seed_confirmed_count(tmp_path)
    # This door takes no acknowledgement, so the delivery is made genuinely validated.
    root = tmp_path / "ds"
    bucket = root / "predictions" / "preds"
    write_prediction(bucket, "img_a")
    stamp = {
        "subject": fx.COUNT_SUBJECT, "attribute": None,
        "validated": True, "trait": fx.COUNT_TRAIT,
        "operating_point": {"conf": {"value": 0.4, "requires_validation": True,
                                     "validation_kind": "annotations",
                                     "validated_against": VALIDATED_HELD_OUT}},
    }
    write_bound_sidecar(bucket, stamp, dataset_root=root)
    out = tmp_path / "counts.csv"

    export_detection_csv(
        [{"image": "a.jpg", "count": 3, "scores": [0.9, 0.8, 0.7]}], str(out),
        trait=fx.COUNT_TRAIT, pred_dirs=[str(bucket)])

    rows = list(__import__("csv").DictReader(out.open()))
    datetime.fromisoformat(rows[0]["produced_at"])


def test_delivered_tail_treats_a_none_valued_produced_at_key_as_absent(tmp_path):
    """A caller composing its own asserted dict over a sidecar carrying no ``produced_at`` of its
    own (``{"produced_at": sidecar.get("produced_at")}``) carries the key with a ``None`` value,
    never a caller-stated one; ``delivered_tail`` must not refuse that the way it refuses a
    caller that actually asserts a ``produced_at``, matching ``corroborated_producer``'s own
    absence convention two functions up."""
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_FALSE, Acknowledgement, check_delivery_gate, delivered_tail,
    )

    gate = check_delivery_gate(
        {"operating_point": VALIDATED_FALSE},
        acknowledgement=Acknowledgement(acknowledged_by="user:tester", reason="test acknowledgement"))
    columns = ("produced_at", "operating_point_validated")

    tail = delivered_tail({"produced_at": None}, {}, gate, columns=columns)
    assert tail["produced_at"]  # the write's own timestamp, not refused

    with pytest.raises(ValueError, match="produced_at"):
        delivered_tail({"produced_at": "2020-01-01T00:00:00+00:00"}, {}, gate, columns=columns)


# ── R2: phenology CSV schema carries producing-model identity ─────────────────

def test_phenology_columns_include_producer_identity():
    from tcip_mcp.pipelines.postprocessing.phenology import phenology_csv_columns
    from tests._trait_fixtures import BUD_OPENING

    columns = phenology_csv_columns(BUD_OPENING)
    assert "producer_model_sha256" in columns
    assert "producing_experiment_id" in columns
    assert "validation_record" in columns
