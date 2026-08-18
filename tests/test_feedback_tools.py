"""Review->retrain MCP tools (materialize + queue + lineage + registration)."""

from __future__ import annotations

from pathlib import Path

from tcip_mcp.tools.feedback_tools import materialize_review_dataset, prioritize_review_queue

# The prediction bucket these verdicts were recorded against, as bucket_key_of spells one.
BUCKET = "predictions/detector/2026-03-04"


def _seed_verdicts(state_dir: Path) -> Path:
    """Record one accepted and one rejected image's verdicts in the store at ``state_dir``."""
    state = {"verdicts": {
        (BUCKET, "imgA.png"): {"img_status": "completed", "detections": [
            {"action": "accepted", "class_name": "catkin", "gt_bbox_norm": [0.5, 0.5, 0.2, 0.2], "pred_bbox_norm": None}]},
        (BUCKET, "imgB.png"): {"img_status": "completed", "detections": [
            {"action": "rejected", "class_name": "catkin", "gt_bbox_norm": None, "pred_bbox_norm": [0.8, 0.8, 0.1, 0.1]}]},
    }}
    # Seed through the engine so the fixture cannot drift from the real shard format.
    from tcip_annotation.review_engine import ReviewEngine

    engine = ReviewEngine(str(state_dir))
    engine.raw_state.update(state)
    engine.save_review_state()
    return state_dir


def _source_images(src: Path) -> Path:
    from PIL import Image
    src.mkdir(parents=True, exist_ok=True)
    for name in ("imgA.png", "imgB.png"):
        Image.new("RGB", (64, 64), (120, 120, 120)).save(src / name)
    return src


def _own_store(dataset_root: Path) -> Path:
    from tcip_mcp.prediction_buckets import review_state_dir_of

    return review_state_dir_of(dataset_root)


def _setup(tmp_path: Path):
    """A dataset whose own verdict store holds the review, plus its reviewed source images."""
    dataset_root = tmp_path / "dataset"
    _seed_verdicts(_own_store(dataset_root))
    return dataset_root, _source_images(tmp_path / "src")


def test_materialize_review_dataset_end_to_end(tmp_path):
    dataset_root, src = _setup(tmp_path)
    out = tmp_path / "out"
    r = materialize_review_dataset(str(dataset_root), str(src), str(out))
    assert "error" not in r
    assert r["positive"] == 1 and r["hard_negative"] == 1
    assert (out / "images" / "imgA.png").is_file()
    assert (out / "annotations" / "imgA.json").is_file()


def test_materialize_reads_the_dataset_s_own_store_when_none_is_stated(tmp_path):
    """The dataset root alone names the store: no second argument, no second location."""
    dataset_root, src = _setup(tmp_path)
    r = materialize_review_dataset(str(dataset_root), str(src), str(tmp_path / "out"))
    assert "error" not in r
    assert r["dataset_root"] == str(dataset_root)
    assert r["review_state_stated"] is False
    assert r["review_state"] == str(_own_store(dataset_root) / "review")
    assert str(_own_store(dataset_root)) in r["review_state_origin"]


def test_materialize_consumes_a_stated_store_outside_the_dataset(tmp_path):
    """A review recorded outside the dataset is still curated, and the response says from where.

    The dataset here has no store of its own, so the shards can only have come from the stated
    location, and the caller is told which one it was.
    """
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    external = _seed_verdicts(tmp_path / "elsewhere" / "state")
    src = _source_images(tmp_path / "src")

    r = materialize_review_dataset(
        str(dataset_root), str(src), str(tmp_path / "out"), review_state_dir=str(external))
    assert "error" not in r
    assert r["positive"] == 1 and r["hard_negative"] == 1
    assert r["dataset_root"] == str(dataset_root)
    assert r["review_state_stated"] is True
    assert r["review_state"] == str(external / "review")
    assert str(external) in r["review_state_origin"]
    assert str(_own_store(dataset_root)) in r["review_state_origin"]


def test_materialize_refuses_an_empty_stated_store_rather_than_the_dataset_s_own(tmp_path):
    """A stated store holding no shards is refused, never answered from the dataset's own.

    The dataset's own store holds a full review here, so a fallback would succeed and quietly
    curate a review the caller did not name.
    """
    dataset_root, src = _setup(tmp_path)
    stated = tmp_path / "elsewhere" / "state"
    stated.mkdir(parents=True)

    r = materialize_review_dataset(
        str(dataset_root), str(src), str(tmp_path / "out"), review_state_dir=str(stated))
    assert str(stated) in r["error"]
    assert "positive" not in r


def test_materialize_review_dataset_records_lineage(tmp_path, monkeypatch):
    import tcip_store as ts
    import tcip_mcp.experiments as experiments
    monkeypatch.setattr(experiments, "EXPERIMENTS_DIR", tmp_path / "exp")
    experiments.create_experiment("exp1", {"x": 1})

    dataset_root, src = _setup(tmp_path)
    out = tmp_path / "out"
    r = materialize_review_dataset(str(dataset_root), str(src), str(out), experiment_id="exp1")
    assert r["experiment_id"] == "exp1"

    lineage = ts.read(experiments.lineage_key("exp1"))
    assert lineage["data_source"] == str(dataset_root)
    assert lineage["review_session"]["dataset_root"] == str(dataset_root)
    assert lineage["review_session"]["review_state_dir"] == str(_own_store(dataset_root))
    artifacts = ts.read(experiments.artifacts_key("exp1"))
    assert artifacts["curated_dataset"]["path"] == str(out)


def test_lineage_records_a_stated_store_beside_the_dataset_it_curates(tmp_path, monkeypatch):
    """Both facts are recorded: which dataset the review was of, and where its shards were read."""
    import tcip_store as ts
    import tcip_mcp.experiments as experiments
    monkeypatch.setattr(experiments, "EXPERIMENTS_DIR", tmp_path / "exp")

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    external = _seed_verdicts(tmp_path / "elsewhere" / "state")
    src = _source_images(tmp_path / "src")

    materialize_review_dataset(
        str(dataset_root), str(src), str(tmp_path / "out"),
        experiment_id="ext1", review_state_dir=str(external))

    lineage = ts.read(experiments.lineage_key("ext1"))
    assert lineage["data_source"] == str(dataset_root)
    assert lineage["review_session"]["dataset_root"] == str(dataset_root)
    assert lineage["review_session"]["review_state_dir"] == str(external)


def test_materialize_creates_experiment_when_absent(tmp_path, monkeypatch):
    import tcip_store as ts
    import tcip_mcp.experiments as experiments
    monkeypatch.setattr(experiments, "EXPERIMENTS_DIR", tmp_path / "exp")

    dataset_root, src = _setup(tmp_path)
    r = materialize_review_dataset(str(dataset_root), str(src), str(tmp_path / "out"), experiment_id="new1")
    assert r["experiment_id"] == "new1"
    lineage = ts.read(experiments.lineage_key("new1"))
    assert "review_session" in lineage


def test_materialize_invalid_inputs_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()  # a dataset root whose own store holds no shards
    r = materialize_review_dataset(str(empty), str(tmp_path), str(tmp_path / "o1"))
    assert str(_own_store(empty)) in r["error"]

    dataset_root, _src = _setup(tmp_path)
    assert "error" in materialize_review_dataset(
        str(dataset_root), str(tmp_path / "nope"), str(tmp_path / "o2"))

    assert "dataset_root" in materialize_review_dataset("", str(tmp_path), str(tmp_path / "o3"))["error"]


def test_prioritize_review_queue_checkpoint_missing(tmp_path):
    r = prioritize_review_queue(str(tmp_path / "nope.pt"), str(tmp_path))
    assert "error" in r  # early guard, no torch import needed


def test_prioritize_review_queue_skips_what_the_dataset_s_own_store_holds(tmp_path):
    """The dataset root is enough to find the verdicts: both reviewed images drop out of the queue.

    A store the tool could not find would rank every image again and send the breeder back through
    a review they already finished.
    """
    dataset_root, images = _setup(tmp_path)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")

    r = prioritize_review_queue(
        checkpoint_path=str(ckpt), images_dir=str(images), dataset_root=str(dataset_root),
        strategy="confidence_triage")
    assert r["reviewed_skipped"] == 2
    assert r["total_images"] == 0


def test_prioritize_review_queue_skips_what_a_stated_store_holds(tmp_path):
    """A review recorded outside the dataset still filters the queue when its store is stated."""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    external = _seed_verdicts(tmp_path / "elsewhere" / "state")
    images = _source_images(tmp_path / "src")
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")

    r = prioritize_review_queue(
        checkpoint_path=str(ckpt), images_dir=str(images), dataset_root=str(dataset_root),
        strategy="confidence_triage", review_state_dir=str(external))
    assert r["reviewed_skipped"] == 2


def test_prioritize_review_queue_rejects_non_composed_kind(tmp_path, monkeypatch):
    """Active-learning uncertainty scoring reads model logits, which a non-composed predictor
    kind (a bespoke tcip model was never built for) doesn't expose, so it must fail with a
    clear error, not crash on ``.model``.
    """
    from types import SimpleNamespace

    import tcip_mcp.pipelines.inference.predictor as predmod

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"x")
    # prioritize_review_queue imports build_predictor from the predictor module at call time.
    monkeypatch.setattr(predmod, "build_predictor",
                        lambda *a, **k: SimpleNamespace(kind="foreign_kind"))

    r = prioritize_review_queue(checkpoint_path=str(ckpt), images_dir=str(images))
    assert "error" in r and "foreign_kind" in r["error"]


def test_prioritize_review_queue_confidence_triage_surfaces_unscoreable(tmp_path, monkeypatch):
    """A regression checkpoint's predictions carry no confidence signal at all (RegressionHead's
    point estimate, deliberately no distributional output). confidence_triage must route them
    into review rather than silently drop them from every output, and tag them distinctly via
    unscoreable_images so a caller can tell this apart from a genuinely medium-confidence item."""
    from types import SimpleNamespace

    import tcip_mcp.pipelines.inference.predictor as predmod

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"x")

    predictions = [{"image": "a.jpg", "width": 4, "height": 4, "head0_values": [0.42]}]
    monkeypatch.setattr(
        predmod, "build_predictor",
        lambda *a, **k: SimpleNamespace(predict_batch=lambda sources: predictions))

    r = prioritize_review_queue(
        checkpoint_path=str(ckpt), images_dir=str(images), strategy="confidence_triage")
    assert r["needs_review"] == 1
    assert r["review_images"] == ["a.jpg"]
    assert r["unscoreable_images"] == ["a.jpg"]
    assert r["auto_accepted_images"] == []


def test_unresolvable_scorer_raises_valueerror_not_an_import_error():
    """The refusal is a ValueError whatever the name looks like.

    ``build_scorer``'s callers catch ``ValueError`` to turn a refusal into an error dict. A dotted
    name that fails to import used to raise ``ModuleNotFoundError`` straight out of the audited
    MCP tool instead.
    """
    import pytest

    from tcip_mcp.pipelines.active_learning.helpers import build_scorer

    with pytest.raises(ValueError):
        build_scorer("no_such_scorer", "detection")
    with pytest.raises(ValueError, match="Could not import scorer"):
        build_scorer("not_a_module.at_all:make", "detection")


def test_unresolvable_proposal_engine_raises_valueerror():
    import pytest

    from tcip_mcp.pipelines.proposal import resolve_proposer

    with pytest.raises(ValueError):
        resolve_proposer("no_such_engine")
    with pytest.raises(ValueError, match="Could not import proposal engine"):
        resolve_proposer("not_a_module.at_all:make")


def test_feedback_tools_register_in_manifest():
    from tcip_mcp.server import list_registered_tools
    names = list_registered_tools()
    assert "materialize_review_dataset" in names
    assert "prioritize_review_queue" in names
