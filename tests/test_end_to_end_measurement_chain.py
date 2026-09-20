"""The measurement chain end to end: ingest, train, publish, assess, confirm, deliver.

Three detectors over one flow. The first delivers a CSV whose validated column reads true, and
is the admitting half for the two refusals beside it: the same flow with the reference labels
edited after the assessment refuses, and the same flow with the bucket published twice refuses
the second publish.

The subject is the chain, not the fit: the model is tiny and its predictions are stable, so a
run that stops delivering a validated number says the chain broke rather than that a fit
wandered.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402

IMG = 64
SUBJECT = "bud"
DATE = "2-11-26"
STEMS = tuple(f"s{i:02d}" for i in range(40))


def _object_at(index: int) -> tuple[int, int, int]:
    """Where this frame's single object sits, and how big it is.

    Every frame differs in both, so no two frames are the same pixels: a reference whose
    calibration and holdout halves shared content would be measuring the model against itself,
    and the calibration refuses one.
    """
    x0 = 4 + (index % 5) * 8
    y0 = 4 + ((index // 5) % 5) * 8
    size = 14 + (index % 3) * 4
    return x0, y0, size


def _synthetic_capture(root: Path) -> tuple[Path, Path]:
    """Ingest one capture date of dim frames, each holding one bright square its label names.

    The frames are written to a raw folder and brought in through ``ingest_images``, the door a
    breeder's pile of photos actually arrives by, so the chain starts where they start. Exactly
    one object per frame, so the counted quantity is known without measuring it, while position,
    size and background all vary frame to frame.
    """
    from PIL import Image, ImageDraw

    from tcip_mcp.tools.ingest_tools import ingest_images

    raw = root.parent / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(STEMS):
        x0, y0, size = _object_at(index)
        shade = 28 + (index % 7)
        frame = Image.new("RGB", (IMG, IMG), color=(shade, shade, shade))
        ImageDraw.Draw(frame).rectangle([x0, y0, x0 + size - 1, y0 + size - 1],
                                        fill=(230, 230, 230))
        frame.save(raw / f"{stem}.png")

    ingested = ingest_images(
        source=str(raw), name="block_bud_count", site="the chain detector's synthetic block",
        project_path=str(root), date_from=DATE,
    )
    assert "error" not in ingested, ingested
    assert ingested["copied"] == len(STEMS), ingested

    images_dir, labels_dir = root / "images" / DATE, root / "annotations" / DATE
    assert images_dir.is_dir(), sorted(p.name for p in root.iterdir())
    labels_dir.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(STEMS):
        x0, y0, size = _object_at(index)
        json_io.write_annotations(
            str(labels_dir / f"{stem}.json"),
            [Annotation(subject=SUBJECT, geometry=BBox(x0, y0, x0 + size, y0 + size))], IMG, IMG,
        )
    return images_dir, labels_dir


def _draw_reference_selection(root: Path, out: Path):
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.tools.data_tools import draw_splits

    result = draw_splits(str(root), output_path=str(out), subject=SUBJECT, seed=2,
                         train_ratio=0.4, val_ratio=0.3, calibration_ratio=0.3)
    assert "error" not in result, result
    return read_selection(out)


BUILDER = "tests.bespoke_models:build_bright_region_detector"


def _run_config(selection_dir: Path) -> dict:
    """A run that binds the drawn selection rather than drawing a partition of its own."""
    return {
        "model_source": {"builder": BUILDER, "builder_kwargs": {}, "task": "detection"},
        "data": {"split": {"selection_dir": str(selection_dir)}},
        "batch_size": 2,
        "stages": [{"freeze_to": -1, "epochs": 1}],
        "mixed_precision": False,
        "device": "cpu",
        "checkpoint_every_n_epochs": 1,
        "early_stopping": {"enabled": False},
        "optimizer": {"name": "sgd", "backbone_lr": 1e-3, "head_lr": 1e-2, "weight_decay": 0},
        "scheduler": {"type": "cosine"},
        "gradient_accumulation_steps": 1,
    }


def _train_on(selection_dir: Path, out_dir: Path, project_root: Path, experiment_id: str) -> str:
    """Train the tiny detector over the selection's train side and register the checkpoint.

    Leaves what the rest of the chain binds to: a run record carrying the selection binding and
    the partition it trained on, and a registered checkpoint with an identity of its own.
    """
    from torch.utils.data import DataLoader

    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_run_partition
    from tcip_mcp.pipelines.training.collation import task_collate
    from tcip_mcp.pipelines.training.generic_trainer import train
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_mcp.tools.model_tools import register_model

    config = _run_config(selection_dir)
    # The draw resolves the scope off the selection and records it on the config the run then
    # keeps, so the checkpoint is stamped with the vocabulary it was actually trained for.
    data_cfg = config["data"]
    train_ds, val_ds, partition = auto_train_val("detection", data_cfg, None)

    create_experiment(experiment_id, config)
    persist_run_partition(experiment_id, train_ds, val_ds, data_cfg, partition=partition)

    collate = task_collate("detection")
    loader = DataLoader(train_ds, batch_size=2, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=2, collate_fn=collate)
    run = create_run(config, str(out_dir), id=experiment_id)
    completed = train(run, loader, val_loader=val_loader, task="detection")
    assert completed.status == "completed", completed.status

    checkpoint = out_dir / "model_best.pt"
    assert checkpoint.is_file(), sorted(p.name for p in out_dir.iterdir())
    registered = register_model(name="chain-detector", checkpoint_path=str(checkpoint),
                                config={}, project_path=str(project_root))
    assert "error" not in registered, registered
    return str(checkpoint)


class Chain:
    """What one run of the flow leaves behind, for a detector to deliver from or disturb."""

    def __init__(self, root: Path, images_dir: Path, labels_dir: Path, selection_dir: Path,
                 checkpoint_path: str, bucket: Path, published: dict) -> None:
        self.root = root
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.selection_dir = selection_dir
        self.checkpoint_path = checkpoint_path
        self.bucket = bucket
        self.published = published


def _confirm_trait_revision(project_root: Path, trait: str, fields: dict) -> dict:
    """Revise the trait's own spec statement and confirm the revision as the breeder would.

    A changed record is a new unconfirmed revision, so the confirmation here is of the revision,
    never of the statement it replaced.
    """
    from tcip_store import read_versioned

    from tcip_mcp import traits

    traits.write_trait_spec_fields(
        trait, fields, project_root=project_root,
        rationale="the breeder tightened what this trait's number has to clear",
    )
    scope = traits.trait_spec_statements_scope(project_root)
    record = read_versioned(traits.trait_spec_statement_key(scope, trait), default=None).value or {}
    assert not record.get("confirmed_by"), "a revised statement must land unconfirmed"
    return traits.confirm_trait_spec(
        project_root, trait, user="chain-breeder",
        record_seen=traits.trait_spec_statement_seen_hash(record),
        identity_from_request=False,
    )


def _run_the_chain(tmp_path: Path, *, experiment_id: str, bucket_name: str = "chain") -> Chain:
    """Ingest, train, publish and assess: the flow every detector here varies one step of."""
    from tests import _operationalization_fixtures as fx

    from tcip_mcp.tools.inference_tools import run_inference

    root = tmp_path / "ds"
    images_dir, labels_dir = _synthetic_capture(root)
    selection_dir = tmp_path / "selection"
    _draw_reference_selection(root, selection_dir)
    checkpoint_path = _train_on(selection_dir, tmp_path / "run", tmp_path, experiment_id)

    fx.write_spec(tmp_path, fx.COUNT_SPEC)

    bucket = root / "predictions" / bucket_name / DATE
    published = run_inference(
        checkpoint_path=checkpoint_path,
        images_dir=str(images_dir),
        output_dir=str(bucket),
        trait=fx.COUNT_TRAIT,
        calibration_labels_dir=str(labels_dir),
        selection_dir=str(selection_dir),
        experiment_id=experiment_id,
    )
    assert "error" not in published, published
    assert published["validated"] is True, published.get("shippable_issues")
    return Chain(root, images_dir, labels_dir, selection_dir, checkpoint_path, bucket, published)


def test_a_drawn_reference_selection_records_each_samples_ground_truth_digest(tmp_path: Path):
    """The fact the edited-reference refusal rests on: a selection records, per sample, the
    digest of the ground truth the draw held out, so a reader can say that file moved since
    without re-reading the draw."""
    root = tmp_path / "ds"
    _synthetic_capture(root)
    selection = _draw_reference_selection(root, tmp_path / "selection")

    calibration = selection.on("calibration")
    assert calibration, selection.counts()
    assert all(s.ground_truth_digest for s in calibration), [
        (s.ground_truth, s.ground_truth_digest) for s in calibration
    ]


def test_the_draw_and_the_delivery_check_digest_a_ground_truth_the_same_way(tmp_path: Path):
    """The two sides of one fact, compared against each other rather than against a fixture.

    The draw records each sample's ground-truth digest; the delivery-time check recomputes one to
    ask whether the reference moved. If those two spelled the digest differently, the check would
    report every untouched reference as moved, and a guard test on either side alone would stay
    green over it.
    """
    from tcip_mcp.pipelines.resolution import _ground_truth_digest

    root = tmp_path / "ds"
    _synthetic_capture(root)
    selection = _draw_reference_selection(root, tmp_path / "selection")

    for sample in selection.on("calibration"):
        assert _ground_truth_digest(Path(sample.ground_truth)) == sample.ground_truth_digest, (
            sample.ground_truth)


def test_the_tiny_detector_trains_and_finds_one_object_per_frame(tmp_path: Path):
    """The model the chain rests on: a real training pass, and predictions stable enough that a
    count measured over them is a fact about the chain rather than about a fit."""
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

    root = tmp_path / "ds"
    images_dir, _labels_dir = _synthetic_capture(root)
    selection_dir = tmp_path / "selection"
    _draw_reference_selection(root, selection_dir)

    checkpoint_path = _train_on(selection_dir, tmp_path / "run", tmp_path, "exp-chain-train")

    checkpoint = load_registered_checkpoint(checkpoint_path, project_path=str(tmp_path))
    predictor = GenericPredictor(checkpoint, device="cpu", score_threshold=0.5)
    results = predictor.predict_batch(
        [str(images_dir / f"{stem}.png") for stem in STEMS[:3]])

    assert [r["count"] for r in results] == [1, 1, 1], results
    for index, result in enumerate(results):
        x0, y0, size = _object_at(index)
        assert result["boxes"][0] == pytest.approx(
            [float(x0), float(y0), float(x0 + size), float(y0 + size)], abs=1.0)


def test_the_calibrated_door_publishes_a_bucket_and_earns_a_record_for_it(tmp_path: Path):
    """Publish and assess: the calibrated export door measures the operating point on the
    selection's calibration side and seals a record the bucket's stamp then names."""
    from tests import _operationalization_fixtures as fx

    from tcip_mcp.tools.inference_tools import run_inference

    root = tmp_path / "ds"
    images_dir, labels_dir = _synthetic_capture(root)
    selection_dir = tmp_path / "selection"
    _draw_reference_selection(root, selection_dir)
    checkpoint_path = _train_on(selection_dir, tmp_path / "run", tmp_path, "exp-chain-publish")

    fx.write_spec(tmp_path, fx.COUNT_SPEC)

    bucket = root / "predictions" / "chain" / DATE
    published = run_inference(
        checkpoint_path=checkpoint_path,
        images_dir=str(images_dir),
        output_dir=str(bucket),
        trait=fx.COUNT_TRAIT,
        calibration_labels_dir=str(labels_dir),
        selection_dir=str(selection_dir),
        experiment_id="exp-chain-publish",
    )

    assert "error" not in published, published
    assert (published.get("gate_evidence_summary") or {}).get("failures") == []
    assert published.get("shippable_issues") == []
    assert published["validated"] is True


# -- the three detectors -------------------------------------------------------


def test_the_chain_delivers_a_csv_whose_validated_column_reads_true(tmp_path: Path):
    """The admitting flow, and the half that makes the two refusals below mean something.

    Ingest synthetic images, train a tiny model, publish a bucket, assess it against a reference
    selection, confirm a trait revision, deliver a CSV whose validated column reads true. Nothing
    is edited and nothing is republished, so the delivery stands.
    """
    import csv

    from tests import _operationalization_fixtures as fx

    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE
    from tcip_mcp.tools.inference_tools import deliver_per_image_counts

    chain = _run_the_chain(tmp_path, experiment_id="exp-chain-delivers")

    _confirm_trait_revision(tmp_path, fx.COUNT_TRAIT, {"count_error_tolerance": 0.25})
    fx.seed_confirmed_count(tmp_path, measured_subject=SUBJECT)

    out_csv = tmp_path / "per_image_counts.csv"
    delivered = deliver_per_image_counts(
        predictions_dir=str(chain.bucket), output_path=str(out_csv), trait=fx.COUNT_TRAIT)

    assert "error" not in delivered, delivered
    assert out_csv.is_file()

    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == len(STEMS), len(rows)
    assert {row["operating_point_validated"] for row in rows} != {VALIDATED_FALSE}
    for row in rows:
        assert row["operating_point_validated"] != VALIDATED_FALSE, row
        assert row["unvalidated_dimensions"] == "", row
        assert row["validation_record"], row
        assert int(row["detection_count"]) == 1, row


def test_editing_the_reference_labels_after_the_assessment_refuses_the_delivery(tmp_path: Path):
    """The same flow with the reference labels edited after the assessment: the delivery refuses.

    The operating point was measured against particular ground truth. Change that ground truth
    afterwards and the number on file was earned against a reference that no longer exists, so
    the claim no longer answers for the delivery and the CSV must not be written.
    """
    from tests import _operationalization_fixtures as fx

    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.tools.inference_tools import deliver_per_image_counts

    chain = _run_the_chain(tmp_path, experiment_id="exp-chain-edited")

    _confirm_trait_revision(tmp_path, fx.COUNT_TRAIT, {"count_error_tolerance": 0.25})
    fx.seed_confirmed_count(tmp_path, measured_subject=SUBJECT)

    # Move one object on the calibration side: the reference the gate was measured against.
    selection = read_selection(chain.selection_dir)
    edited = Path(selection.on("calibration")[0].ground_truth)
    json_io.write_annotations(
        str(edited), [Annotation(subject=SUBJECT, geometry=BBox(1, 1, 9, 9))], IMG, IMG)

    out_csv = tmp_path / "per_image_counts.csv"
    delivered = deliver_per_image_counts(
        predictions_dir=str(chain.bucket), output_path=str(out_csv), trait=fx.COUNT_TRAIT)

    assert "error" in delivered, delivered
    assert edited.name in str(delivered["error"]), delivered["error"]
    assert not out_csv.exists(), "a refused delivery writes no CSV"


def test_a_reference_selection_that_can_no_longer_be_read_refuses_the_delivery(tmp_path: Path):
    """A claim whose reference nobody can open is a claim whose reference cannot be confirmed.

    The sibling of the edited-labels refusal: deleting the selection leaves the operating point
    resting on evidence that is no longer there to check, which the delivery refuses rather than
    treating an unanswerable question as an answer of no.
    """
    import tcip_store

    from tests import _operationalization_fixtures as fx

    from tcip_mcp.pipelines.data.selection import read_selection, selection_key
    from tcip_mcp.tools.inference_tools import deliver_per_image_counts

    chain = _run_the_chain(tmp_path, experiment_id="exp-chain-unreadable")

    _confirm_trait_revision(tmp_path, fx.COUNT_TRAIT, {"count_error_tolerance": 0.25})
    fx.seed_confirmed_count(tmp_path, measured_subject=SUBJECT)

    # Through the store's own door, so the selection is gone under either storage backend.
    tcip_store.delete(selection_key(chain.selection_dir))
    with pytest.raises(ValueError):
        read_selection(chain.selection_dir)

    out_csv = tmp_path / "per_image_counts.csv"
    delivered = deliver_per_image_counts(
        predictions_dir=str(chain.bucket), output_path=str(out_csv), trait=fx.COUNT_TRAIT)

    assert "error" in delivered, delivered
    assert "cannot be read now" in str(delivered["error"]), delivered["error"]
    assert not out_csv.exists(), "a refused delivery writes no CSV"


def test_publishing_the_same_bucket_twice_refuses_the_second_publish(tmp_path: Path):
    """The same flow with the bucket published twice: the second refuses.

    A bucket already holding prediction documents is not republished into, whatever overwrite
    says, so the predictions a delivered number rests on cannot be replaced underneath it.
    """
    from tests import _operationalization_fixtures as fx

    from tcip_mcp.tools.inference_tools import run_inference

    chain = _run_the_chain(tmp_path, experiment_id="exp-chain-republish")
    # Every file the bucket holds, the provenance stamp included, which is a loose file under one
    # storage backend and a record inside the other.
    before = {path.name: path.read_bytes() for path in sorted(chain.bucket.glob("*.json"))}
    assert before

    republished = run_inference(
        checkpoint_path=chain.checkpoint_path,
        images_dir=str(chain.images_dir),
        output_dir=str(chain.bucket),
        trait=fx.COUNT_TRAIT,
        calibration_labels_dir=str(chain.labels_dir),
        selection_dir=str(chain.selection_dir),
        experiment_id="exp-chain-republish",
        overwrite=True,
    )

    # document_stem_count is the document guard's own key, so this pins the refusal to the
    # second publish rather than to any error the call might have raised.
    assert "error" in republished, republished
    assert republished["document_stem_count"] == len(STEMS), republished
    after = {path.name: path.read_bytes() for path in sorted(chain.bucket.glob("*.json"))}
    assert after == before, "the refused publish must leave the bucket exactly as it was"
