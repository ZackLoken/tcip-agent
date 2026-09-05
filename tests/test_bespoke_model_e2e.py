"""An agent-authored detector with modified internals and a custom train(ctx) loop, run end to
end through the audited envelope.

Proves the whole CV-scientist vision at once:
  * (a) architecture modifications take effect end to end: the ``AnchorGenerator`` uses aspect ratios
    derived from the synthetic GT (via ``derivations.gt_aspect_ratios``) and sizes from the GT
    size distribution (not torchvision defaults), and every norm layer is GroupNorm (no BatchNorm);
  * (b) the custom ``train(ctx)`` loop (not ``ctx.default_train``) had its metrics, checkpoint, audit
    bracket, and source/env provenance recorded by the envelope (``kind=KIND_TCIP_MODULE`` with a
    source+env snapshot present) and the model registered on completion;
  * the module actually learns (``overfit_check``), and build -> resolve_operating_point -> predict
    close the measurement loop.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

# No built-in traits: seed_bud_trait_spec (conftest.py) writes a real bud.yml into this
# test's pinned platform state root so resolve_operating_point("bud_opening", ...) keeps resolving by default.
pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

# Component registration side-effects (backbones/necks/heads used by build_dataset + eval).
import tcip_mcp.pipelines.components.backbones  # noqa: F401,E402
import tcip_mcp.pipelines.components.necks  # noqa: F401,E402
import tcip_mcp.pipelines.components.heads  # noqa: F401,E402
import tcip_mcp.pipelines.components.losses  # noqa: F401,E402

from torch.utils.data import DataLoader  # noqa: E402

from tests import bespoke_models  # noqa: E402 (the agent-authored bespoke model + train loop)
from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402

IMG = 64


def _save_png(path: Path) -> None:
    from torchvision.utils import save_image

    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.rand(3, IMG, IMG), str(path))


def _audit_events(root: Path, tool: str = "training_run") -> list[dict]:
    import tcip_store as ts
    from tcip_mcp.audit import audit_log_key

    page = ts.read_log(audit_log_key(root))
    return [record for record in page.records if record.get("tool") == tool]


def test_bespoke_detector_end_to_end(tmp_path: Path):
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.model_registry import ModelRegistry, load_registered_checkpoint
    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.derivations import gt_aspect_ratios
    from tcip_mcp.pipelines.inference.predictor import KIND_TCIP_MODULE, build_predictor
    from tcip_mcp.pipelines.model_build import build_model
    from tcip_mcp.pipelines.model_contract import overfit_check
    from tcip_mcp.pipelines.operating_point import records_over_loader, resolve_operating_point
    from tcip_mcp.pipelines.training.envelope import TrainContext, run_training_envelope
    from tcip_mcp.pipelines.training.collation import task_collate
    from tcip_mcp.pipelines.training.run_registry import create_run

    # 1. Synthetic detection data: open (tall) boxes so GT-derived anchors differ from defaults.
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    shapes = [(15, 36), (16, 40), (17, 44)]
    gt_wh: list[tuple[int, int]] = []
    for i in range(6):
        _save_png(images_dir / f"img{i}.png")
        w, h = shapes[i % len(shapes)]
        x1, y1 = 32 - w / 2, 32 - h / 2
        json_io.write_annotations(str(labels_dir / f"img{i}.json"),
                                  [Annotation(subject="bud", geometry=BBox(x1, y1, x1 + w, y1 + h))],
                                  IMG, IMG, keep_empty=True)
        gt_wh.append((w, h))

    dataset = build_dataset("detection", images_dir=str(images_dir),
                            labels_dir=str(labels_dir), subject="bud")
    train_loader = DataLoader(dataset, batch_size=2, collate_fn=task_collate("detection"))
    val_loader = DataLoader(dataset, batch_size=2, collate_fn=task_collate("detection"))

    # 2. Bespoke model_source + custom training_source, run through the audited envelope.
    src_file = bespoke_models.__file__
    config = {
        "model_source": {
            "builder": "tests.bespoke_models:build_bespoke_detector",
            "builder_kwargs": {"gt_boxes_wh": gt_wh, "num_classes": 1,
                               "min_size": IMG, "max_size": IMG * 2},
            "task": "detection", "in_chans": 3, "source_files": [src_file],
        },
        "training_source": "tests.bespoke_models:train_bespoke",
        "device": "cpu", "epochs": 2, "seed": 0,
    }
    out = tmp_path / "out"
    create_experiment("expBespoke", config, data_source=str(images_dir))
    update_status("expBespoke", "running")
    run = create_run(config, str(out))
    ctx = TrainContext(run=run, train_loader=train_loader, val_loader=val_loader,
                       task="detection", experiment_id="expBespoke")

    run_training_envelope(ctx)

    # ---- the custom loop completed through the envelope ----
    assert run.status == "completed", run.error
    ckpt = out / "model_best.pt"
    assert ckpt.is_file()

    # ---- (a) modifications took effect end to end (rebuilt from the trained checkpoint) ----
    expected_ratios = tuple(gt_aspect_ratios(gt_wh))
    expected_sizes = bespoke_models.gt_anchor_sizes(gt_wh)
    assert expected_ratios != (0.5, 1.0, 2.0)              # not torchvision's default ratios
    assert expected_sizes != (32, 64, 128, 256, 512)       # not torchvision's default sizes

    checkpoint = load_registered_checkpoint(str(ckpt), project_path=str(tmp_path))
    predictor = build_predictor(checkpoint, device="cpu", score_threshold=0.0)
    assert predictor.kind == KIND_TCIP_MODULE
    anchor_gen = predictor.model.detector.rpn.anchor_generator
    assert anchor_gen.aspect_ratios == (expected_ratios,)   # anchors are the GT-derived ratios
    assert anchor_gen.sizes == (expected_sizes,)            # anchor sizes are the GT-derived scales
    assert any(isinstance(m, torch.nn.GroupNorm) for m in predictor.model.modules())  # BN->GN present
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                   for m in predictor.model.modules())      # no BatchNorm survived the modification

    # ---- (b) the custom loop's checkpoint is stamped bespoke; provenance snapshot present ----
    best = torch.load(ckpt, weights_only=False)
    assert best["kind"] == KIND_TCIP_MODULE
    assert best["model_source"]["builder"].endswith(":build_bespoke_detector")

    import tcip_store as ts
    from tcip_mcp.experiments import env_key
    from tcip_mcp.pipelines.model_build import snapshot_manifest_key

    exp_dir = tmp_path / ".tcip" / "experiments" / "expBespoke"
    env = ts.read(env_key("expBespoke"))
    assert env["model_kind"] == KIND_TCIP_MODULE and env["env"]["torch"]
    manifest = ts.read(snapshot_manifest_key(exp_dir))
    assert manifest["training_source"] == "tests.bespoke_models:train_bespoke"
    assert any(e["src"] == src_file and len(e["sha256"]) == 64
               for e in manifest["files"])                  # source snapshotted with sha256
    assert (exp_dir / "model_src" / next(
        e["file"] for e in manifest["files"] if e["src"] == src_file)).is_file()

    # ---- (b) the custom loop's metrics + the audit bracket were recorded via ctx/envelope ----
    # The loop's rows land on the experiment's own metrics log, not beside the weights in a
    # separately-computed output dir.
    from tcip_mcp.experiments import read_metrics

    assert not (out / "metrics.jsonl").exists()
    metric_rows = read_metrics("expBespoke")
    assert metric_rows and all("train_loss" in r for r in metric_rows)
    events = _audit_events(tmp_path)
    assert [e["status"] for e in events] == ["running", "completed"]  # opened + closed around the body
    assert events[-1]["arguments"]["run_id"] == run.run_id

    # ---- completion registered the bespoke model into the immutable registry ----
    entry = ModelRegistry(str(tmp_path)).get_model("expBespoke")
    assert entry is not None and entry["kind"] == KIND_TCIP_MODULE
    assert entry["sha256"] and len(entry["sha256"]) == 64

    # ---- the module actually learns; resolve_operating_point + predict close the measurement loop ----
    overfit = overfit_check(build_model(config), "detection", steps=30, lr=5e-3)
    assert overfit["passed"], overfit["issue"]

    records = records_over_loader(predictor.model, val_loader, torch.device("cpu"), "detection")
    bundle = resolve_operating_point("bud_opening", tiled=True, dataset_hash="test",
                                     calibration_records=records, holdout_records=records)
    assert "conf" in bundle.params                          # operating point resolved over bespoke outputs

    pred = predictor.predict(str(images_dir / "img0.png"))
    assert {"boxes", "scores", "labels", "count"} <= set(pred)  # measurable detection output
