"""W5 — the provenance identity spine.

Locks the additive provenance stamping across the spine: checkpoint experiment_id + a
computed-once sha256, the terminal-state lock (additive-only), the enriched capture_env,
the split manifest, and the producing-model stamps on the delivery CSV/manifest surfaces.
These are provenance additions — no measurement changes.
"""

from __future__ import annotations

import json

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


# ── R2/G7: sha256 computed once (cached) + identity resolved from the registry ─

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
        "m1", str(ckpt), {"model_source": {"builder": "x:y"}}, tags=["experiment:expR"])

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
    """K1: the ordinary train-then-calibrate workflow saves a checkpoint stamped with its own
    ``experiment_id`` (via ``stamp_model_ref``) but never registers it before calibration runs —
    ``resolve_model_identity`` must read that stamp directly rather than resolving None just
    because no registry entry exists yet (which used to silently bypass the train-disjointness
    gate for every checkpoint that hadn't been explicitly registered)."""
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


# ── R3/D6: terminal-state lock is additive-only ───────────────────────────────

@pytest.fixture()
def exp_store(tmp_path, monkeypatch):
    import tcip_mcp.experiments as exp
    monkeypatch.setattr(exp, "EXPERIMENTS_DIR", tmp_path / "experiments")
    # Route the refused-mutation audit to the tmp project so it never touches the repo log.
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    return exp


def test_update_status_refuses_to_leave_terminal(exp_store):
    exp = exp_store
    exp.create_experiment("t1", {})
    exp.update_status("t1", "completed")
    res = exp.update_status("t1", "running")
    assert "error" in res
    d = exp.experiments_dir() / "t1"
    assert json.loads((d / "status.json").read_text())["state"] == "completed"


def test_log_metrics_refuses_new_epoch_when_terminal(exp_store):
    exp = exp_store
    exp.create_experiment("t2", {})
    exp.log_metrics("t2", 0, {"loss": 1.0})
    exp.update_status("t2", "completed")
    res = exp.log_metrics("t2", 1, {"loss": 0.5})
    assert "error" in res
    lines = (exp.experiments_dir() / "t2" / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1  # no new epoch appended


def test_lineage_additive_first_write_allowed_overwrite_refused(exp_store):
    exp = exp_store
    exp.create_experiment("t3", {})
    exp.update_status("t3", "completed")
    # First write into a still-empty field is permitted (R1's predictions link relies on this).
    exp.update_lineage("t3", predictions="/preds/run")
    lin = json.loads((exp.experiments_dir() / "t3" / "lineage.json").read_text())
    assert lin["predictions"] == "/preds/run"
    # Overwriting the now-populated field is refused; the recorded value stands.
    exp.update_lineage("t3", predictions="/preds/other")
    lin2 = json.loads((exp.experiments_dir() / "t3" / "lineage.json").read_text())
    assert lin2["predictions"] == "/preds/run"


def test_record_artifact_additive_only_when_terminal(exp_store):
    exp = exp_store
    exp.create_experiment("t4", {})
    exp.update_status("t4", "completed")
    exp.record_artifact("t4", "predictions", "/preds")   # new name -> allowed
    res = exp.record_artifact("t4", "predictions", "/other")  # existing -> refused
    assert "error" in res
    arts = json.loads((exp.experiments_dir() / "t4" / "artifacts.json").read_text())
    assert arts["predictions"]["path"] == "/preds"


# ── R5: make_splits manifest embeds dataset_hash + seed ───────────────────────

def test_make_splits_manifest_embeds_hash_and_seed(data_dir, tmp_path):
    from tcip_mcp.tools.data_tools import make_splits

    out = tmp_path / "splits"
    result = make_splits(str(data_dir), output_path=str(out), seed=7)
    assert result["dataset_hash"]
    assert result["seed"] == 7
    manifest = json.loads((out / "split_manifest.json").read_text())
    assert manifest["seed"] == 7
    assert manifest["dataset_hash"] == result["dataset_hash"]
    assert set(manifest["splits"]) == {"train", "val", "test"}


# ── R2: delivery CSVs carry the producing-model provenance columns ────────────

def test_export_detection_csv_carries_provenance(tmp_path):
    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

    out = tmp_path / "counts.csv"
    export_detection_csv(
        [{"image": "a.jpg", "count": 3, "scores": [0.9, 0.8, 0.7]}], str(out),
        provenance={"producer_model_sha256": "abc", "experiment_id": "expE",
                    "operating_point_conf": 0.42, "produced_at": "2026-07-19T00:00:00Z"},
        measurement_validated="validated_held_out")
    rows = list(__import__("csv").DictReader(out.open()))
    assert rows[0]["producer_model_sha256"] == "abc"
    assert rows[0]["experiment_id"] == "expE"
    assert rows[0]["operating_point_conf"] == "0.42"


def test_export_aggregated_csv_carries_provenance(tmp_path):
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    out = tmp_path / "agg.csv"
    # K3: no on-disk measurement-validity source for a bare trait_name="count" call with no
    # pred_dirs — acknowledge the provisional delivery explicitly (the provenance stamp itself is
    # unaffected by that).
    export_aggregated_csv(
        [{"plant_id": "p1", "value": 5, "observations": 2}], str(out), trait_name="count",
        provenance={"producer_model_sha256": "def", "experiment_id": "expA",
                    "produced_at": "2026-07-19T00:00:00Z"},
        measurement_validated="validated_held_out", acknowledge_unvalidated=True)
    rows = list(__import__("csv").DictReader(out.open()))
    assert rows[0]["producer_model_sha256"] == "def"
    assert rows[0]["experiment_id"] == "expA"


# ── R2: phenology CSV schema carries producing-model identity ─────────────────

def test_phenology_columns_include_producer_identity():
    from tcip_mcp.pipelines.postprocessing.phenology import phenology_csv_columns
    from tests._trait_fixtures import CATKIN

    columns = phenology_csv_columns(CATKIN)
    assert "producer_model_sha256" in columns
    assert "producer_experiment_id" in columns
