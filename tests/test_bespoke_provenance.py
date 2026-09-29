"""Code provenance for a bespoke (model_source) run.

Locks: snapshot_model_source (copy source files + sha256), KIND_TCIP_MODULE stamping, the
_kind_from_ckpt structural fallback, build_predictor rebuilding a bespoke model from its importable
builder (no exec) + predicting, and a completed run's registry entry round-tripping the bespoke
kind.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_mcp.pipelines.inference.predictor import (  # noqa: E402
    KIND_TCIP_MODULE,
    _kind_from_ckpt,
    build_predictor,
    detect_kind,
)
from tcip_mcp.pipelines.model_build import (  # noqa: E402
    build_model,
    snapshot_model_source,
    stamp_model_ref,
)
from tests import bespoke_models  # noqa: E402


def _model_source() -> dict:
    return {"builder": "tests.bespoke_models:build_bespoke_detector",
            "builder_kwargs": {"gt_boxes_wh": [[15, 36], [16, 40], [17, 44]],
                               "min_size": 64, "max_size": 128},
            "task": "detection", "source_files": [__file__]}


_DATA = {"num_channels": 3, "scope": {"subject": "bud", "id_map": {"bud": 0}}}
"""The data section a one-subject, three-band run records, which its checkpoint carries."""

_DIMS = {"in_chans": 3, "num_classes": 1}
"""What :func:`~tcip_mcp.pipelines.model_build.recorded_model_dims` reads off :data:`_DATA`."""


# --------------------------------------------------------------------------
# snapshot_model_source: source files + sha256
# --------------------------------------------------------------------------

def test_snapshot_model_source_copies_files_and_records_provenance(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    manifest = snapshot_model_source({"model_source": _model_source(), "seed": 123}, exp_dir)

    assert manifest is not None
    expected_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    entry = next(e for e in manifest["files"] if e["sha256"] == expected_sha)
    stored = (exp_dir / "model_src" / entry["file"]).read_bytes()  # content-addressed destination
    assert hashlib.sha256(stored).hexdigest() == expected_sha
    assert manifest["builder"].endswith(":build_bespoke_detector")
    assert "seed" not in manifest
    assert manifest["missing"] == []
    assert manifest["snapshot_errors"] == []


# --------------------------------------------------------------------------
# silent partial capture is self-describing
# --------------------------------------------------------------------------

def test_snapshot_model_source_records_missing_files(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    src = _model_source()
    missing_path = str(tmp_path / "does_not_exist.py")
    src["source_files"] = [__file__, missing_path]
    manifest = snapshot_model_source({"model_source": src}, exp_dir)

    assert manifest["missing"] == [missing_path]
    assert any(e["src"] == __file__ for e in manifest["files"])  # the real file still captured


def test_snapshot_model_source_records_import_error(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    manifest = snapshot_model_source(
        {"model_source": {"builder": "definitely_not_a_real_module_xyz:build"}}, exp_dir)

    assert manifest["snapshot_errors"]
    assert "definitely_not_a_real_module_xyz" in manifest["snapshot_errors"][0]


# --------------------------------------------------------------------------
# content-addressed destination: no basename clobber, no double-count
# --------------------------------------------------------------------------

def test_snapshot_model_source_dedups_same_file_reached_two_ways(tmp_path):
    """The auto-appended builder module __file__ (absolute) and a differently-spelled
    source_files entry for the same physical file (e.g. via a relative/dotted path) must dedup
    by content, not merely by exact path-string equality: a naive ``str(p) in seen`` dedup
    misses this because the two spellings never compare equal as strings."""
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    real = Path(__file__).resolve()
    # A second, distinct string that resolves to the exact same file on disk: relative to cwd.
    import os
    alt_spelling = os.path.relpath(real, Path.cwd())
    assert alt_spelling != str(real)  # genuinely a different string, not a no-op fixture

    src = _model_source()
    src["source_files"] = [str(real), alt_spelling]
    manifest = snapshot_model_source({"model_source": src}, exp_dir)

    expected_sha = hashlib.sha256(real.read_bytes()).hexdigest()
    matches = [e for e in manifest["files"] if e["sha256"] == expected_sha]
    assert len(matches) == 1  # one physical file, one entry, regardless of how many ways it was named


def test_snapshot_model_source_basename_collision_does_not_clobber(tmp_path):
    """Two distinct source files sharing a basename must both survive on disk with distinct
    content, not silently overwrite each other."""
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    (a_dir / "model.py").write_text("# builder A")
    (b_dir / "model.py").write_text("# builder B, different content")

    src = {"builder": "tests.bespoke_models:build_bespoke_detector",
          "source_files": [str(a_dir / "model.py"), str(b_dir / "model.py")]}
    manifest = snapshot_model_source({"model_source": src}, exp_dir)

    a_path, b_path = str(a_dir / "model.py"), str(b_dir / "model.py")
    file_entries = [e for e in manifest["files"] if e["src"] in (a_path, b_path)]
    assert len(file_entries) == 2
    shas = {e["sha256"] for e in file_entries}
    assert len(shas) == 2  # distinct content, distinct hashes, distinct destination keys
    for e in file_entries:
        dst = exp_dir / "model_src" / e["file"]
        assert dst.is_file()
        assert hashlib.sha256(dst.read_bytes()).hexdigest() == e["sha256"]  # not clobbered


# --------------------------------------------------------------------------
# KIND_TCIP_MODULE stamping + structural fallback
# --------------------------------------------------------------------------

def test_stamp_names_the_kind_from_the_config_and_the_sniff_reads_it():
    payload = stamp_model_ref({"model_state_dict": {},
                               "config": {"model_source": _model_source()}})
    assert payload["kind"] == KIND_TCIP_MODULE
    assert _kind_from_ckpt(payload, "x.pt") == KIND_TCIP_MODULE


def test_stamp_model_ref_refuses_a_payload_with_no_weights():
    """A payload sniffed as a loadable tcip module (model_source present) must carry its weights,
    or the predictor fails at inference with a bare KeyError naming no contract."""
    import pytest

    with pytest.raises(ValueError, match="model_state_dict"):
        stamp_model_ref({"metrics": {"val_loss": 0.2},
                         "config": {"model_source": _model_source()}})


# --------------------------------------------------------------------------
# build_predictor rebuilds the bespoke model from its builder (no exec) + predicts
# --------------------------------------------------------------------------

def test_build_predictor_rebuilds_bespoke_and_predicts(tmp_path):
    from PIL import Image

    src = _model_source()
    model = build_model({"model_source": src}, _DIMS)
    assert isinstance(model, bespoke_models.BespokeGNDetector)  # built via the importable builder

    ckpt = tmp_path / "model_best.pt"
    payload = stamp_model_ref(
        {"model_state_dict": model.state_dict(), "metrics": {"val_loss": 0.3, "epoch": 1},
         "config": {"model_source": src, "data": _DATA}})
    torch.save(payload, ckpt)

    assert detect_kind(str(ckpt)) == KIND_TCIP_MODULE  # kind sniffed from disk

    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.tools.model_tools import register_model

    reg_result = register_model(name="bespoke-detector", checkpoint_path=str(ckpt), config={},
                                project_path=str(tmp_path))
    assert "error" not in reg_result, reg_result
    checkpoint = load_registered_checkpoint(str(ckpt), project_path=str(tmp_path))

    predictor = build_predictor(checkpoint, device="cpu", score_threshold=0.0)
    assert predictor.kind == KIND_TCIP_MODULE
    assert predictor.task == "detection"
    assert predictor.in_chans == 3

    img = tmp_path / "a.png"
    Image.new("RGB", (64, 64), (120, 120, 120)).save(img)
    out = predictor.predict(str(img))
    assert {"boxes", "scores", "labels", "count"} <= set(out)  # measurable detection output


def test_predictor_loads_at_the_two_channels_its_run_recorded(tmp_path):
    """The width a run records on its data section is the one its model was built at, and the
    predictor reads that checkpoint's images at it: a two-band run loads at two channels, not a
    silent default of 3."""
    import numpy as np

    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tcip_mcp.pipelines.model_build import recorded_model_dims

    src = {"builder": "tests.bespoke_models:build_bespoke_classifier", "task": "classification"}
    config = {"model_source": src, "data": {"num_channels": 2, "num_classes": 2, "scope": {}}}
    model = build_model(config, recorded_model_dims(config))
    ckpt = tmp_path / "model_best.pt"
    payload = stamp_model_ref({"model_state_dict": model.state_dict(), "config": config})
    torch.save(payload, ckpt)

    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.tools.model_tools import register_model

    reg_result = register_model(name="bespoke-classifier", checkpoint_path=str(ckpt), config={},
                                project_path=str(tmp_path))
    assert "error" not in reg_result, reg_result
    checkpoint = load_registered_checkpoint(str(ckpt), project_path=str(tmp_path))

    predictor = GenericPredictor(checkpoint, device="cpu")
    assert predictor.in_chans == 2

    arr = (np.random.rand(16, 16, 2) * 255).astype(np.uint8)
    img = tmp_path / "two_band.npy"
    np.save(img, arr)
    out = predictor.predict(str(img))
    assert out  # decoded and forwarded at 2 channels with no shape-mismatch error


# --------------------------------------------------------------------------
# a completed run's registry entry round-trips the bespoke kind
# --------------------------------------------------------------------------

def test_register_round_trips_bespoke_kind(tmp_path):
    from tcip_mcp.model_registry import ModelRegistry
    from tests._verified_checkpoint_fixtures import finished_run

    finished_run(tmp_path, experiment_id="expB", model_source=_model_source(), data=_DATA,
                 metrics={"val_loss": 0.3, "epoch": 1})

    [entry] = [m for m in ModelRegistry(str(tmp_path)).list_models()
               if m["experiment_id"] == "expB"]
    assert entry["metrics"]["val_loss"] == pytest.approx(0.3)
    assert entry["kind"] == KIND_TCIP_MODULE   # read from the stamped checkpoint's payload
    assert entry["sha256"] and len(entry["sha256"]) == 64
    assert entry["experiment_id"] == "expB"
