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


# ── R2: sha256 computed once (cached) + identity resolved from the registry ─

def test_checkpoint_sha256_is_cached(tmp_path, monkeypatch):
    import tcip_mcp.model_registry as reg

    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"weights")

    calls = {"n": 0}
    real = reg._compute_sha256

    def _counting(p):
        calls["n"] += 1
        return real(p)

    monkeypatch.setattr(reg, "_compute_sha256", _counting)
    reg._SHA_CACHE.clear()
    a = reg.checkpoint_sha256(ckpt)
    b = reg.checkpoint_sha256(ckpt)
    assert a == b and len(a) == 64
    assert calls["n"] == 1  # second call served from cache, no re-hash


def test_resolve_model_identity_from_registry(tmp_path):
    from tcip_mcp.model_registry import ModelRegistry, resolve_model_identity

    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"weights")
    ModelRegistry(str(tmp_path)).register_model(
        "m1", str(ckpt), {"model_source": {"builder": "x:y"}}, tags=["experiment:expR"],
        metrics_source=None)

    ident = resolve_model_identity(ckpt, project_path=str(tmp_path))
    assert ident["sha256"] and len(ident["sha256"]) == 64
    assert ident["experiment_id"] == "expR"
    assert ident["checkpoint"] == "best"


def test_resolve_model_identity_foreign_checkpoint(tmp_path):
    from tcip_mcp.model_registry import resolve_model_identity

    ckpt = tmp_path / "foreign.pt"
    ckpt.write_bytes(b"weights")
    ident = resolve_model_identity(ckpt, project_path=str(tmp_path))
    assert ident["sha256"]                 # sha still recorded
    assert ident["experiment_id"] is None  # no run -> honest null, not a failure


def test_resolve_model_identity_reads_checkpoint_own_experiment_id(tmp_path):
    """The ordinary train-then-calibrate workflow saves a checkpoint stamped with its own
    ``experiment_id`` (via ``stamp_model_ref``) but never registers it before calibration runs;
    ``resolve_model_identity`` must read that stamp directly rather than resolving None just
    because no registry entry exists yet, which would otherwise silently bypass the
    train-disjointness gate for every checkpoint that hadn't been explicitly registered."""
    torch = pytest.importorskip("torch")
    from tcip_mcp.model_registry import resolve_model_identity

    ckpt = tmp_path / "stamped.pt"
    torch.save({"model_state_dict": {}, "experiment_id": "expStamped"}, ckpt)

    ident = resolve_model_identity(ckpt, project_path=str(tmp_path))
    assert ident["experiment_id"] == "expStamped"
    assert ident["sha256"]


def test_resolve_model_identity_caller_experiment_id_wins_over_stamp(tmp_path):
    torch = pytest.importorskip("torch")
    from tcip_mcp.model_registry import resolve_model_identity

    ckpt = tmp_path / "stamped.pt"
    torch.save({"model_state_dict": {}, "experiment_id": "expStamped"}, ckpt)

    ident = resolve_model_identity(ckpt, experiment_id="expCaller", project_path=str(tmp_path))
    assert ident["experiment_id"] == "expCaller"


# ── R3: terminal-state lock is additive-only ───────────────────────────────

@pytest.fixture()
def exp_store(tmp_path, monkeypatch):
    import tcip_mcp.experiments as exp
    monkeypatch.setattr(exp, "EXPERIMENTS_DIR", tmp_path / "experiments")
    # Route the refused-mutation audit to the tmp project so it never touches the repo log.
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
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


# ── R5: make_splits manifest embeds dataset_hash + seed ───────────────────────

def test_make_splits_manifest_embeds_hash_and_seed(data_dir, tmp_path):
    import tcip_store as ts
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.data_tools import make_splits, split_manifest_key

    # A manifest write needs at least four foreground groups to clear the floor; the fixture's
    # own three (img_001..003) need one more, added here rather than in the shared fixture.
    from PIL import Image

    images_dir = data_dir / "images" / "2-11-26"
    labels_dir = data_dir / "annotations" / "2-11-26"
    Image.new("RGB", (640, 480), color=(128, 128, 128)).save(images_dir / "img_004.jpg")
    json_io.write_annotations(
        labels_dir / "img_004.json",
        [Annotation(subject="catkin", geometry=BBox(288, 216, 352, 264))], 640, 480,
    )

    out = tmp_path / "splits"
    result = make_splits(str(data_dir), output_path=str(out), seed=7, subject="catkin",
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

    from tests import _operationalization_fixtures as fx

    fx.seed_confirmed_count(tmp_path)
    sha = _run_with_a_recorded_checkpoint(tmp_path, "expE")
    out = tmp_path / "counts.csv"
    export_detection_csv(
        [{"image": "a.jpg", "count": 3, "scores": [0.9, 0.8, 0.7]}], str(out),
        trait=fx.COUNT_TRAIT,
        provenance={"producer_model_sha256": sha, "experiment_id": "expE",
                    "operating_point_conf": 0.42, "produced_at": "2026-07-19T00:00:00Z"},
        acknowledge_unvalidated=True)
    rows = list(__import__("csv").DictReader(out.open()))
    assert rows[0]["producer_model_sha256"] == sha
    assert rows[0]["experiment_id"] == "expE"
    assert rows[0]["operating_point_conf"] == "0.42"


def test_export_aggregated_csv_carries_provenance(tmp_path):
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    from tests import _operationalization_fixtures as fx

    fx.seed_delivery_traits(tmp_path)
    fx.seed_confirmed_aggregate(tmp_path, "stem_count", value_keys=["count"])
    sha = _run_with_a_recorded_checkpoint(tmp_path, "expA")
    out = tmp_path / "agg.csv"
    # No pred_dirs means no on-disk validity source, so the provisional delivery is acknowledged.
    export_aggregated_csv(
        [{"plant_id": "p1", "value": 5, "observations": 2, "value_key": "count",
          "measurement_document": "operating_point"}],
        str(out), trait_name="stem_count",
        provenance={"producer_model_sha256": sha, "experiment_id": "expA",
                    "produced_at": "2026-07-19T00:00:00Z"},
        measurement_validated="held_out_annotations", acknowledge_unvalidated=True)
    rows = list(__import__("csv").DictReader(out.open()))
    assert rows[0]["producer_model_sha256"] == sha
    assert rows[0]["experiment_id"] == "expA"


# ── R2: phenology CSV schema carries producing-model identity ─────────────────

def test_phenology_columns_include_producer_identity():
    from tcip_mcp.pipelines.postprocessing.phenology import phenology_csv_columns
    from tests._trait_fixtures import CATKIN

    columns = phenology_csv_columns(CATKIN)
    assert "producer_model_sha256" in columns
    assert "producer_experiment_id" in columns
    assert "validation_record" in columns
