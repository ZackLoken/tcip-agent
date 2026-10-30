"""An agent-authored detector with modified internals and a custom train(ctx) loop, end to end.

The run goes through the audited envelope. The test checks that:
  * (a) architecture modifications take effect end to end: the ``AnchorGenerator`` uses the
    aspect ratios and base size the checkpoint records under ``DETECTOR_GEOMETRY_DERIVATION``,
    and the model holds GroupNorm layers and no BatchNorm;
  * (b) the custom ``train(ctx)`` loop had its metrics, checkpoint and audit bracket recorded by
    the envelope, the source and environment in the launch record, and the model registered by
    completing;
  * the module passes ``overfit_check``, and its predictor answers a detection record.
"""

from __future__ import annotations

import hashlib
from functools import partial
from pathlib import Path

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
from tests._audit_fixtures import audit_rows
from tests._chain_fixtures import (
    FRAME_STATISTICS,
    GT_ANCHOR_DETECTOR,
    TRAIN_BESPOKE,
    training_config,
)

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

# Component registration side-effects (backbones/necks/heads used by build_dataset + eval).
import tcip_mcp.pipelines.components.backbones  # noqa: F401,E402
import tcip_mcp.pipelines.components.heads  # noqa: F401,E402
import tcip_mcp.pipelines.components.losses  # noqa: F401,E402
import tcip_mcp.pipelines.components.necks  # noqa: F401,E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402
from tcip_mcp.pipelines.model_build import CONFIG_KEY  # noqa: E402
from tests import bespoke_models  # noqa: E402 (the agent-authored bespoke model + train loop)
from tests._image_fixtures import write_noise_image  # noqa: E402
from tests._producer_fixtures import dataset_over, label_image  # noqa: E402
from tests._training_values import (  # noqa: E402
    RATIONALE,
    adamw_optimizer,
    drawn_split,
    schedule,
)

IMG = 64

_save_png = partial(write_noise_image, size=IMG, span=1.0)


def test_bespoke_detector_end_to_end(tmp_path: Path) -> None:
    from dataclasses import asdict

    from tcip_mcp.experiments import (
        METRICS_FILE,
        observe,
        read_rows,
        read_run,
    )
    from tcip_mcp.model_registry import (
        ModelRegistry,
        load_registered_checkpoint,
    )
    from tcip_mcp.pipelines.derivations import DETECTOR_GEOMETRY_DERIVATION
    from tcip_mcp.pipelines.execution import Stated, prepare
    from tcip_mcp.pipelines.model_contract import overfit_check

    # 1. Synthetic detection data: open (tall) boxes so GT-derived anchors are tall too.
    images_dir = image_dir(tmp_path, UNDATED_BUCKET)
    shapes = [(15, 36), (16, 40), (17, 44)]
    for i in range(6):
        _save_png(images_dir / f"img{i}.png")
        w, h = shapes[i % len(shapes)]
        x1, y1 = 32 - w / 2, 32 - h / 2
        label_image(
            images_dir / f"img{i}.png",
            [Annotation(subject="bud", geometry=BBox(x1, y1, x1 + w, y1 + h))],
            IMG,
            IMG,
            keep_empty=True,
        )

    dataset = dataset_over("detection", str(images_dir), subject="bud")

    # 2. Bespoke model_source + custom training_source, run through the audited envelope.
    src_file = bespoke_models.__file__
    # train_bespoke runs its scheduler's horizon beside the loaders' batch and the blocks
    # ctx.build_optimizer, ctx.build_scheduler and ctx.evaluate read.
    config = training_config(
        tmp_path,
        {
            "builder": GT_ANCHOR_DETECTOR,
            "task": "detection",
            "source_files": [src_file],
            "normalization": FRAME_STATISTICS,
        },
        {
            "images_dir": str(images_dir),
            "num_channels": 3,
            "scope": asdict(dataset.scope),
            "split": drawn_split(val_ratio=0.15),
        },
        training_source=TRAIN_BESPOKE,
        scheduler=schedule("cosine", horizon_epochs=2),
        optimizer=adamw_optimizer(backbone_lr=1e-3),
    )
    from tests._verified_checkpoint_fixtures import worker_run

    out = worker_run(tmp_path, config, experiment_id="expBespoke")

    # ---- the custom loop completed through the envelope ----
    final = observe(out).final
    assert final is not None and final.state == "completed", final
    ckpt = out / "model_best.pt"
    assert ckpt.is_file()

    # ---- (a) modifications took effect end to end (rebuilt from the trained checkpoint) ----
    checkpoint = load_registered_checkpoint(str(ckpt), project=tmp_path)
    geometry = checkpoint.geometry.detector
    assert geometry is not None
    assert geometry.source == DETECTOR_GEOMETRY_DERIVATION
    expected_ratios = tuple(geometry.aspect_ratios)
    # Every ground-truth box is tall: taller than any fixed ratio set reaches.
    assert max(expected_ratios) > 2.0
    base = geometry.anchor_base_size
    expected_sizes = (base, base * 2, base * 4)

    p = prepare(
        checkpoint,
        Stated(tile=False, conf=0.0, rationale=RATIONALE),
        device="cpu",
    ).runnable()
    predictor = p.predictor
    anchor_gen = predictor.model.detector.rpn.anchor_generator
    assert anchor_gen.aspect_ratios == (
        expected_ratios,
    )  # anchors are the GT-derived ratios
    assert anchor_gen.sizes == (
        expected_sizes,
    )  # anchor sizes are the GT-derived scales
    # BN->GN present
    assert any(
        isinstance(m, torch.nn.GroupNorm) for m in predictor.model.modules()
    )
    assert not any(
        isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
        for m in predictor.model.modules()
    )  # no BatchNorm survived the modification

    # ---- (b) the custom loop's checkpoint names its builder; provenance snapshot present ----
    best = torch.load(ckpt, weights_only=False)
    assert best[CONFIG_KEY]["model_source"]["builder"] == GT_ANCHOR_DETECTOR

    launch = read_run(out)
    assert launch.environment["torch"]
    assert launch.config["training_source"] == TRAIN_BESPOKE
    [entry] = launch.source["files"].values()  # source snapshotted with sha256
    copy = out / entry["file"]
    assert copy.read_bytes() == Path(src_file).read_bytes()
    assert entry["sha256"] == hashlib.sha256(copy.read_bytes()).hexdigest()
    assert [
        Path(f).resolve() for f in checkpoint.spec.model_source.source_files
    ] == [copy.resolve()]  # the checkpoint binds that copy

    # ---- (b) the custom loop's metrics + the audit bracket were recorded via ctx/envelope ----
    metric_rows = read_rows(out / METRICS_FILE)[0]
    assert metric_rows and all(
        "train_loss" in r.model_dump() for r in metric_rows
    )
    events = audit_rows(tmp_path, "training_run")
    # opened + closed around the body
    assert [e["status"] for e in events] == ["running", "completed"]
    assert events[-1]["arguments"]["experiment_id"] == "expBespoke"

    # ---- completion registered the bespoke model into the immutable registry ----
    [registered] = [
        m.entry
        for m in ModelRegistry(str(tmp_path)).list_models()
        if m.entry.experiment_id == "expBespoke"
    ]
    assert registered.sha256 and len(registered.sha256) == 64

    # ---- the module actually learns, and its predictor answers a detection record ----
    from tests._chain_fixtures import built_model

    overfit = overfit_check(
        built_model(config, geometry),
        "detection",
        seed=0,
        deterministic=False,
        device=torch.device("cpu"),
        steps=30,
        lr=5e-3,
        dims={"in_chans": 3, "num_classes": 1, "img_size": (64, 64)},
    )
    assert overfit["passed"], overfit["issue"]

    (pred,) = p.predict([str(images_dir / "img0.png")])
    assert {"boxes", "scores", "labels", "count"} <= set(
        pred
    )  # measurable detection output
