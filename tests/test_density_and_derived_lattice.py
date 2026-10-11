"""Each frame's detection cap is the pass's object density times that frame's own pixels, the
density the checkpoint recorded of its training regions where no reference stands behind the pass,
every forward, prediction and validation alike, run under it; and a tile lattice nobody stated or
recorded is the one the ground truth's object sides derive, the same one for the tiler, the
spatial split and the pass."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_annotation.state import Annotation, BBox  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

from tcip_mcp.dataset_layout import UNDATED_BUCKET  # noqa: E402
from tcip_mcp.pipelines.execution import ExecutionRefusedError, Stated  # noqa: E402
from tests import REPO_ROOT  # noqa: E402
from tests._verified_checkpoint_fixtures import SAMPLE_CONF, SAMPLE_DETECTOR_PASS  # noqa: E402

MOSAIC_WIDTH, MOSAIC_HEIGHT = 5000, 600
"""A mosaic wide enough for a strip split at the lattice its 100 px long objects derive."""

THIS_FILE = str(REPO_ROOT / "tests" / "test_density_and_derived_lattice.py")
"""The file this module's dataset builders are defined in, which a run naming one declares."""


def build_regions_ds(*, samples, scope, transforms=None, **_ignored):
    """A ``dataset_source`` builder's detection dataset over ``samples``: the platform's own
    three-band loader, exposing its ``regions``."""
    from tcip_mcp.pipelines.data.datasets import build_dataset

    return build_dataset("detection", samples=samples, scope=scope, transforms=transforms,
                         sizes={"num_channels": 3})


class _NoRegions(Dataset):
    """A detection dataset serving ``inner``'s items and exposing no ``regions``."""

    def __init__(self, inner) -> None:
        self.inner = inner

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index):
        return self.inner[index]


def build_no_regions_ds(**kwargs) -> _NoRegions:
    """A ``dataset_source`` builder's detection dataset exposing no ``regions``."""
    return _NoRegions(build_regions_ds(**kwargs))


COMPOSED_LATTICE = (32, 0.25)
"""The lattice :func:`compose_tiles` composes its tiler at."""


def compose_tiles(ctx) -> None:
    """A training body tiling its own training loader's dataset at :data:`COMPOSED_LATTICE`
    (``ctx.tiled_dataset``), logging the lattice its tiler serves, validating the model as built
    over those tiles (``ctx.evaluate``) and saving it."""
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.model_build import STATE_DICT_KEY

    edge, overlap = COMPOSED_LATTICE
    tiled = ctx.tiled_dataset(ctx.train_loader.dataset, tile_size=edge, overlap=overlap,
                              sliver_frac=0.5)
    model = ctx.build_model()
    val = ctx.evaluate(model, DataLoader(tiled, batch_size=2, collate_fn=ctx.task_collate()))
    ctx.log_metrics(1, {"tile_size": tiled.tile_size, "overlap": tiled.overlap,
                        "val_loss": val["loss"]})
    ctx.save_checkpoint({STATE_DICT_KEY: model.state_dict()}, "model_final")


def _in_model_caps(model) -> list:
    """The ``detections_per_img`` each forward of ``model`` runs under, appended as it runs."""
    from tcip_mcp.pipelines.operating_point import detector_operating_point_holder

    holder, _path = detector_operating_point_holder(model)
    caps: list = []
    model.register_forward_pre_hook(lambda _module, _args: caps.append(holder.detections_per_img))
    return caps


def test_a_pass_with_no_reference_caps_each_frame_at_the_checkpoints_density_times_its_pixels(
        tmp_path: Path):
    """Two frames of different sizes in one pass: each runs its forward under, and its row
    carries, the cap its own pixels give the density the checkpoint recorded, so the caps stand in
    the ratio of the frames' areas and neither row keeps more than its own."""
    from PIL import Image

    from tcip_mcp.pipelines.derivations import detection_cap
    from tcip_mcp.pipelines.execution import prepare
    from tests._verified_checkpoint_fixtures import verified_checkpoint

    checkpoint = verified_checkpoint(tmp_path)
    frames = {"small": (64, 48), "large": (256, 192)}
    paths = []
    for name, size in frames.items():
        Image.new("RGB", size, (120, 120, 120)).save(tmp_path / f"{name}.png")
        paths.append(str(tmp_path / f"{name}.png"))

    p = prepare(checkpoint, Stated(tile=False, conf=0.0), device="cpu").runnable()
    forwards = _in_model_caps(p.predictor.model)
    small, large = p.predict(paths)

    density = checkpoint.spec.data.train_object_density
    caps = [detection_cap(density, w * h) for w, h in frames.values()]
    assert (p.execution.density, p.execution.sources["density"]) == (density, "derived")
    assert [small["cap"], large["cap"]] == caps
    assert large["cap"] / small["cap"] == (256 * 192) / (64 * 48)
    assert small["count"] <= small["cap"] and large["count"] <= large["cap"]
    assert sorted(forwards) == sorted(caps)


def _without_density(project: Path):
    """A checkpoint no run produced: the bytes of :func:`project_checkpoint`'s with the density
    its config records removed, registered in explicit mode; loaded through the registry."""
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.model_build import CONFIG_KEY
    from tests._verified_checkpoint_fixtures import project_checkpoint, register_checkpoint

    payload = torch.load(project_checkpoint(project), map_location="cpu", weights_only=False)
    del payload[CONFIG_KEY]["data"]["train_object_density"]
    path = project / "foreign.pt"
    torch.save(payload, path)
    register_checkpoint(project, str(path), name="foreign")
    return load_registered_checkpoint(str(path), project=project)


def test_a_checkpoint_recording_no_density_refuses_a_detector_pass_naming_the_primitive(
        tmp_path: Path):
    """A detector checkpoint no run of the platform produced records no density of its training
    regions: a pass with no reference refuses naming the door that records one, and the
    platform-built detector beside it predicts."""
    from PIL import Image

    from tcip_mcp.pipelines.execution import prepare
    from tests._verified_checkpoint_fixtures import verified_checkpoint

    foreign = _without_density(tmp_path)
    assert foreign.spec.data.train_object_density is None

    with pytest.raises(ExecutionRefusedError, match="launch_training"):
        prepare(foreign, Stated(tile=False, conf=SAMPLE_CONF), device="cpu").runnable()
    Image.new("RGB", (64, 48), (120, 120, 120)).save(tmp_path / "frame.png")
    admitted = prepare(verified_checkpoint(tmp_path), Stated(tile=False, conf=SAMPLE_CONF),
                       device="cpu").runnable()
    (result,) = admitted.predict([str(tmp_path / "frame.png")])
    assert result["cap"] >= 1 and result["count"] <= result["cap"]


def _mosaic(root: Path) -> Path:
    """One mosaic under ``root``'s undated capture holding two rows of 100 x 4 px buds; its images
    directory."""
    import numpy as np
    import tifffile

    from tests._producer_fixtures import label_image

    images_dir = root / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    raster = images_dir / "mosaic.tif"
    rng = np.random.default_rng(0)
    tifffile.imwrite(str(raster), rng.integers(0, 255, size=(MOSAIC_HEIGHT, MOSAIC_WIDTH, 3),
                                               dtype=np.uint8))
    label_image(raster, [Annotation(subject="bud", geometry=BBox(x, y, x + 100, y + 4))
                         for x in range(10, MOSAIC_WIDTH - 110, 150) for y in (100, 400)],
                MOSAIC_WIDTH, MOSAIC_HEIGHT, keep_empty=True)
    return images_dir


def _resolved(project: Path, data: dict, monkeypatch) -> tuple:
    """``(run, lattice derivations)``: the run a detection config over ``data`` resolves under
    ``project`` (``split_construction.resolve_run``), and how many times its resolution called
    ``derivations.derive_tile_geometry``."""
    import tcip_mcp.pipelines.derivations as derivations
    from tcip_mcp.pipelines.data.split_construction import resolve_run
    from tcip_mcp.pipelines.model_build import staged_sources
    from tcip_mcp.pipelines.schemas import train_config
    from tests._chain_fixtures import training_config
    from tests._verified_checkpoint_fixtures import unbuilt_source

    calls: list = []
    real = derivations.derive_tile_geometry

    def counted(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(derivations, "derive_tile_geometry", counted)
    spec = train_config(training_config(unbuilt_source("detection"), data))
    run = resolve_run(spec, staged_sources(spec, project).layout, project=project)
    monkeypatch.setattr(derivations, "derive_tile_geometry", real)
    return run, len(calls)


def test_the_tiler_the_spatial_split_and_the_pass_derive_one_lattice_from_one_ground_truth(
        tmp_path: Path, monkeypatch):
    """One mosaic's ground truth of long, thin objects, three routes' own records: the lattice a
    run's resolution recorded and drew its spatial split at, the one its tiler serves, and the one
    a tiled pass over a checkpoint recording none ran at under that mosaic as its reference, all
    the lattice the mosaic's objects derive, resolved once for the run, in some tile of which
    every object lies whole."""
    from tcip_mcp.pipelines.derivations import derive_tile_geometry
    from tcip_mcp.pipelines.slicing import slice_lattice
    from tcip_mcp.pipelines.training.eval_runners import run_full_frame_evaluation
    from tests._predictor_fixtures import StubPredictor, install
    from tests._producer_fixtures import checkpoint_admission, dataset_over
    from tests._verified_checkpoint_fixtures import verified_checkpoint

    images_dir = _mosaic(tmp_path / "ds")
    run, derivations = _resolved(tmp_path, {
        "images_dir": str(images_dir), "scope": {"subject": "bud"},
        "tiling": {"enabled": True, "sliver_frac": 0.5},
        "split": {"val_ratio": 0.2, "seed": 1}}, monkeypatch)
    assert run.spatial is not None and derivations == 1
    run_lattice = (run.data.tiling.tile_size, run.data.tiling.overlap)
    tiler_lattice = (run.train_ds.tile_size, run.train_ds.overlap)

    checkpoint = verified_checkpoint(tmp_path)
    install(monkeypatch, StubPredictor(task="detection", in_chans=3, width=MOSAIC_WIDTH,
                                       height=MOSAIC_HEIGHT, boxes=(), scores=()))
    execution = run_full_frame_evaluation(
        checkpoint, checkpoint_admission(checkpoint, images_dir),
        stated=Stated(**SAMPLE_DETECTOR_PASS))["execution"]

    objects = dataset_over("detection", str(images_dir), subject="bud",
                           stated={"num_channels": 3}).regions
    assert run_lattice == tiler_lattice == (execution["tile_size"], execution["overlap"])
    assert run_lattice == derive_tile_geometry(objects, tile_size=None, overlap=None)
    slices = slice_lattice(MOSAIC_HEIGHT, MOSAIC_WIDTH, *tiler_lattice)
    for x0, y0, x1, y1 in objects[0].boxes.tolist():
        assert any(sx0 <= x0 and sy0 <= y0 and x1 <= sx1 and y1 <= sy1
                   for sx0, sy0, sx1, sy1 in slices), (x0, y0, x1, y1)


def test_a_drawn_tiled_run_resolves_its_lattice_once(tmp_path: Path, monkeypatch):
    """A tiled run drawing train and val over several frames resolves its lattice once, before
    its loaders are built, and its tiler serves that one."""
    from tests._producer_fixtures import seed_bud_images

    images_dir = seed_bud_images(tmp_path / "ds" / "images" / UNDATED_BUCKET, n=4, size=128)
    run, derivations = _resolved(tmp_path, {
        "images_dir": str(images_dir), "scope": {"subject": "bud"},
        "tiling": {"enabled": True, "tile_size": 64, "sliver_frac": 0.5},
        "split": {"val_ratio": 0.25, "seed": 0}}, monkeypatch)

    assert derivations == 1
    assert (run.train_ds.tile_size, run.train_ds.overlap) == (run.data.tiling.tile_size,
                                                              run.data.tiling.overlap)


def test_an_object_as_long_as_the_derived_overlap_lies_whole_in_a_tile_at_a_stated_edge(
        tmp_path: Path):
    """A frame holding one 29 px object, at a stated 100 px edge: the lattice its run resolves
    overlaps neighbors by at least the object's length, so its tiler holds it whole in some
    tile."""
    from tcip_mcp.pipelines.slicing import slice_lattice
    from tests._producer_fixtures import label_image, run_over, write_image

    image = write_image(tmp_path / "images" / UNDATED_BUCKET / "frame.png", (300, 100))
    label_image(image, [Annotation(subject="bud", geometry=BBox(71.5, 5, 100.5, 9))], 300, 100)
    tiler, data = run_over("detection", image.parent, subject="bud",
                           tiling={"enabled": True, "tile_size": 100, "sliver_frac": 0.5})
    (region,) = run_over("detection", image.parent, subject="bud")[0].regions
    (x0, y0, x1, y1), = region.boxes.tolist()

    assert (tiler.tile_size, data["tiling"]["tile_size"]) == (100, 100)
    assert any(sx0 <= x0 and sy0 <= y0 and x1 <= sx1 and y1 <= sy1
               for sx0, sy0, sx1, sy1 in slice_lattice(100, 300, tiler.tile_size, tiler.overlap))


@pytest.mark.parametrize("stated", [{}, {"tile_size": 64}], ids=["nothing", "an edge alone"])
def test_a_tiled_pass_with_nothing_stated_recorded_or_referenced_refuses_naming_the_three(
        tmp_path: Path, stated: dict):
    """A checkpoint that trained untiled on frames of no square edge records no lattice: a tiled
    pass stating none of it and holding no reference refuses naming what supplies it, never a
    fallback lattice; stated whole, the same pass runs."""
    from tests._producer_fixtures import gray_frame
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_OVERLAP, predicted_over, project_checkpoint,
    )

    checkpoint = project_checkpoint(tmp_path)
    images_dir = str(Path(gray_frame(tmp_path)).parent)

    with pytest.raises(ExecutionRefusedError) as refused:
        predicted_over(tmp_path, checkpoint, images_dir, device="cpu", tile=True, **stated)
    message = str(refused.value)
    assert "state it" in message and "train the checkpoint tiled" in message
    assert "reference" in message

    p, _results = predicted_over(tmp_path, checkpoint, images_dir, device="cpu", tile=True,
                                 tile_size=64, overlap=SAMPLE_OVERLAP)
    assert (p.execution.tile_size, p.execution.overlap) == (64, SAMPLE_OVERLAP)


def test_validation_runs_every_frame_under_the_cap_its_density_gives_it(tmp_path: Path,
                                                                        monkeypatch):
    """The trainer's validation predicts each frame under the cap the run's density gives it, so
    a frame whose cap is one keeps one detection though the same detector under a cap of two
    keeps more than one."""
    import tcip_mcp.pipelines.training.evaluation as evaluation
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tcip_mcp.pipelines.model_build import model_dims
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point
    from tcip_mcp.pipelines.training.generic_trainer import _validate
    from tests._producer_fixtures import run_over, seed_bud_images
    from tests._verified_checkpoint_fixtures import verified_checkpoint

    images_dir = seed_bud_images(tmp_path / "val" / "images" / UNDATED_BUCKET, n=2, size=128)
    dataset, data = run_over("detection", images_dir, subject="bud", stated={"num_channels": 3})
    images, targets = zip(*(dataset[i] for i in range(len(dataset))))
    model = GenericPredictor(verified_checkpoint(tmp_path), device="cpu").model
    model.eval()
    set_detector_operating_point(model, score_thresh=0.0, detections_per_img=2)
    with torch.no_grad():
        raw = model([images[0]])[0]
    assert len(raw["boxes"]) > 1, "the built detector must keep more than one box under two"

    kept: list[int] = []
    real = evaluation.records_from_detector

    def counting(target, output, **kwargs):
        kept.append(len(output["boxes"]))
        return real(target, output, **kwargs)

    monkeypatch.setattr(evaluation, "records_from_detector", counting)
    _validate(model, [(list(images), list(targets))], torch.device("cpu"), "detection",
              dims=model_dims(ClassScope.of(data), {"num_channels": 3}), conf_threshold=0.0,
              iou_threshold=0.5, score_weights=None, trait=None, density=1 / (128 * 128))

    assert len(kept) == 2 and max(kept) <= 1


def test_validation_keeps_every_box_the_cap_admits_whatever_the_builders_own_floor(
        tmp_path: Path, monkeypatch):
    """The trainer's validation predicts at no score floor: a built detector holding a score
    threshold above every score it gives, as a builder's own threshold would, still reports in
    each frame's row every box the frame's cap admits."""
    import tcip_mcp.pipelines.training.evaluation as evaluation
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.derivations import detection_cap
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tcip_mcp.pipelines.model_build import model_dims
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point
    from tcip_mcp.pipelines.training.generic_trainer import _validate
    from tests._producer_fixtures import run_over, seed_bud_images
    from tests._verified_checkpoint_fixtures import verified_checkpoint

    images_dir = seed_bud_images(tmp_path / "val" / "images" / UNDATED_BUCKET, n=2, size=128)
    dataset, data = run_over("detection", images_dir, subject="bud", stated={"num_channels": 3})
    images, targets = zip(*(dataset[i] for i in range(len(dataset))))
    model = GenericPredictor(verified_checkpoint(tmp_path), device="cpu").model
    model.eval()
    density = 300 / (128 * 128)
    cap = detection_cap(density, 128 * 128)
    set_detector_operating_point(model, score_thresh=0.0, detections_per_img=cap)
    with torch.no_grad():
        unfloored = [model([image])[0] for image in images]
    scores = [float(s) for out in unfloored for s in out["scores"]]
    assert scores, "the fixture must give boxes for a floor to drop"
    set_detector_operating_point(model, score_thresh=min(1.0, max(scores) + 1e-3),
                                 detections_per_img=cap)

    rows: list[dict] = []
    real = evaluation.records_from_detector

    def recording(target, output, **kwargs):
        rows.append(real(target, output, **kwargs))
        return rows[-1]

    monkeypatch.setattr(evaluation, "records_from_detector", recording)
    _validate(model, [(list(images), list(targets))], torch.device("cpu"), "detection",
              dims=model_dims(ClassScope.of(data), {"num_channels": 3}), conf_threshold=0.5,
              iou_threshold=0.5, score_weights=None, trait=None, density=density)

    assert [row["count"] for row in rows] == [len(out["boxes"]) for out in unfloored]


def _frames_config(project: Path, name: str, **data) -> dict:
    """A one-epoch run of the built detector over four labeled frames of its own, drawing a
    validation side and selecting on its loss, ``data``'s keys beside its data section's own."""
    from tests._chain_fixtures import training_config
    from tests._training_values import evaluation_block
    from tests._verified_checkpoint_fixtures import (
        SCOPED_DATA, detection_images, detector_declaring, fixture_data_dir,
    )

    return training_config(detector_declaring(THIS_FILE), {
        **detection_images(fixture_data_dir(project, name), SCOPED_DATA["scope"], n=4),
        **SCOPED_DATA, "split": {"seed": 0, "val_ratio": 0.25}, **data},
        evaluation=evaluation_block(selection_metric="loss"))


def test_the_density_the_trainer_validates_under_is_the_one_its_checkpoint_records(
        tmp_path: Path, monkeypatch):
    """A run's validation is handed the density its own resolution recorded, the one its
    checkpoint carries."""
    import tcip_mcp.pipelines.training.generic_trainer as trainer
    from tcip_mcp.experiments import observe
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tests._verified_checkpoint_fixtures import worker_run

    handed: list = []
    real = trainer._validate

    def recording(*args, **kwargs):
        handed.append(kwargs["density"])
        return real(*args, **kwargs)

    monkeypatch.setattr(trainer, "_validate", recording)
    observation = observe(worker_run(tmp_path, _frames_config(tmp_path, "frames")))
    assert observation.checkpoint is not None, observation.final

    recorded = load_registered_checkpoint(observation.checkpoint["path"], project=tmp_path)
    assert handed and set(handed) == {recorded.spec.data.train_object_density}


def test_a_bespoke_detection_dataset_exposing_regions_records_its_density_and_validates(
        tmp_path: Path, monkeypatch):
    """A run whose loaders a ``dataset_source`` builder makes records the density of the regions
    its dataset exposes and validates under it; a builder's dataset exposing none refuses naming
    the interface."""
    import tcip_mcp.pipelines.training.generic_trainer as trainer
    from tcip_mcp.experiments import observe
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tests._verified_checkpoint_fixtures import opened_run, worker_run

    validated: list = []
    real = trainer._validate

    def recording(*args, **kwargs):
        validated.append(kwargs["density"])
        return real(*args, **kwargs)

    monkeypatch.setattr(trainer, "_validate", recording)
    observation = observe(worker_run(tmp_path, _frames_config(
        tmp_path, "bespoke", dataset_source={
            "builder": "test_density_and_derived_lattice:build_regions_ds",
            "source_files": [THIS_FILE]})))
    assert observation.checkpoint is not None, observation.final
    trained = load_registered_checkpoint(observation.checkpoint["path"], project=tmp_path)
    assert trained.spec.data.train_object_density == pytest.approx(1 / (64 * 48))
    assert validated == [trained.spec.data.train_object_density]

    with pytest.raises(ValueError, match="regions"):
        opened_run(tmp_path, _frames_config(tmp_path, "opaque", dataset_source={
            "builder": "test_density_and_derived_lattice:build_no_regions_ds",
            "source_files": [THIS_FILE]}))


def test_a_bespoke_body_tiles_at_the_lattice_it_states_and_validates_over_its_tiles(
        tmp_path: Path):
    """A run whose loaders a ``dataset_source`` builder makes records no lattice; its training
    body composes the tiler at the pair it states, the tiler serves that pair, and the run
    validates over those tiles and completes."""
    from tcip_mcp.experiments import observe, read_rows
    from tests._verified_checkpoint_fixtures import worker_run

    observation = observe(worker_run(tmp_path, _frames_config(
        tmp_path, "composed", dataset_source={
            "builder": "test_density_and_derived_lattice:build_regions_ds",
            "source_files": [THIS_FILE]}) | {
        "training_source": "test_density_and_derived_lattice:compose_tiles"}))
    assert observation.checkpoint is not None, observation.final

    (row,) = [r for r in read_rows(observation.metrics_log)[0] if "tile_size" in r]
    assert (row["tile_size"], row["overlap"]) == COMPOSED_LATTICE
    assert observation.record["resolved"]["data"].get("tiling") is None
    assert observation.record["resolved"]["data"].get("train_native_size") is None


def test_a_tiling_stated_beside_a_builder_refuses_at_admission_naming_both(tmp_path: Path):
    """A config stating ``data.tiling`` beside a ``dataset_source`` refuses at the door naming
    both, before any run opens; the same config without the tiling opens."""
    from tests._verified_checkpoint_fixtures import opened_run

    source = {"builder": "test_density_and_derived_lattice:build_regions_ds",
              "source_files": [THIS_FILE]}
    with pytest.raises(ValueError) as refused:
        opened_run(tmp_path, _frames_config(tmp_path, "both", dataset_source=source,
                                            tiling={"enabled": True}))
    assert all(name in str(refused.value) for name in ("data.tiling", "data.dataset_source"))
    assert opened_run(tmp_path, _frames_config(tmp_path, "builder", dataset_source=source))


def test_an_enabled_tiling_without_its_lattice_refuses_naming_its_resolver(tmp_path: Path):
    """A tiling block reaching the factory enabled with no edge or overlap refuses naming both
    and the resolver that states them, and the same block resolved builds its tiler."""
    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.data.label_queries import registry_scope
    from tcip_mcp.pipelines.data.split_construction import resolved_tiling
    from tcip_mcp.pipelines.schemas import TilingSpec
    from tests._producer_fixtures import samples_over, seed_bud_images

    images_dir = seed_bud_images(tmp_path / "images" / UNDATED_BUCKET, n=2, size=128)
    samples, scope = samples_over(images_dir, subject="bud"), registry_scope(images_dir, "bud")
    sizes, tiling = {"num_channels": 3}, TilingSpec.model_validate({"sliver_frac": 0.5})

    with pytest.raises(ValueError) as refused:
        build_dataset("detection", samples=samples, scope=scope, sizes=sizes, tiling=tiling)
    assert all(name in str(refused.value)
               for name in ("tile_size", "overlap", "resolved_tiling"))
    resolved = resolved_tiling("detection", tiling, samples, scope, sizes)
    tiler = build_dataset("detection", samples=samples, scope=scope, sizes=sizes, tiling=resolved)
    assert (tiler.tile_size, tiler.overlap) == (resolved.tile_size, resolved.overlap)


@pytest.mark.parametrize("built", ["recording", "counting"])
def test_a_stand_in_datasets_density_is_the_one_its_served_objects_give(built: str):
    """A stand-in detection dataset's regions are read off the objects it serves, so the density
    a run records of it counts exactly those."""
    from tcip_mcp.pipelines.derivations import derive_object_density
    from tests.test_dataset_source_seam import build_bespoke_ds
    from tests.test_selection_binding import build_recording_dataset

    dataset = (build_recording_dataset(samples=["s0", "s1"]) if built == "recording"
               else build_bespoke_ds(rows=["s0", "s1"]))
    served = sum(len(dataset[i][1]["boxes"]) for i in range(len(dataset)))

    assert derive_object_density(dataset.regions) == served / len(dataset) / (8 * 8)


def test_a_bespoke_detector_under_cap_one_keeps_one_through_validation_and_a_pass(
        tmp_path: Path, monkeypatch):
    """A producer-built bespoke detector over frames holding two objects each keeps one under a
    cap of one, in a pass and in the trainer's validation alike, though ungoverned it finds two."""
    import tcip_mcp.pipelines.training.evaluation as evaluation
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.execution import prepare
    from tcip_mcp.pipelines.model_build import model_dims
    from tcip_mcp.pipelines.training.generic_trainer import _validate
    from tests._chain_fixtures import BLOB_BUILDER
    from tests._producer_fixtures import run_over, seed_labeled_images
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    images_dir = seed_labeled_images(
        tmp_path / "pair" / "images" / UNDATED_BUCKET,
        [Annotation(subject="bud", geometry=BBox(4, 4, 20, 20)),
         Annotation(subject="bud", geometry=BBox(36, 30, 56, 44))], n=2, width=64, height=48)
    checkpoint = load_registered_checkpoint(
        registered_checkpoint(tmp_path, model_source=BLOB_BUILDER), project=tmp_path)
    p = prepare(checkpoint, Stated(tile=False, conf=0.0), images_dir=str(images_dir),
                device="cpu").runnable()
    dataset, data = run_over("detection", images_dir, subject="bud", stated={"num_channels": 3})
    images, targets = zip(*(dataset[i] for i in range(len(dataset))))
    model = p.predictor.model
    model.eval()
    with torch.no_grad():
        assert len(model([images[0]])[0]["boxes"]) == 2

    one = 1 / (64 * 48)
    rows = p.predictor.predict_batch(p.paths, p.execution.with_value("density", one, "explicit"))
    assert [(r["cap"], r["count"]) for r in rows] == [(1, 1), (1, 1)]

    kept: list[int] = []
    real = evaluation.records_from_detector

    def counting(target, output, **kwargs):
        kept.append(len(output["boxes"]))
        return real(target, output, **kwargs)

    monkeypatch.setattr(evaluation, "records_from_detector", counting)
    _validate(model, [(list(images), list(targets))], torch.device("cpu"), "detection",
              dims=model_dims(ClassScope.of(data), {"num_channels": 3}), conf_threshold=0.0,
              iou_threshold=0.5, score_weights=None, trait=None, density=one)
    assert kept == [1, 1]


@pytest.mark.parametrize(
    "knob", ["box_detections_per_img", "detections_per_img", "box_score_thresh", "score_thresh"])
def test_an_operating_point_knob_stated_to_the_builder_refuses_by_name(knob: str):
    """The cap and the score threshold are set at every forward: a detector builder handed
    either refuses naming it, and the same call without it builds at its other keywords."""
    from tests.bespoke_models import build_bespoke_detection

    with pytest.raises(ValueError, match=knob):
        build_bespoke_detection(min_size=64, max_size=128, box_nms_thresh=0.4, **{knob: 7})
    assert build_bespoke_detection(min_size=64, max_size=128, box_nms_thresh=0.4) is not None


def test_a_restored_record_missing_its_overlap_or_density_refuses_and_a_complete_one_predicts(
        tmp_path: Path):
    """A stored detector's tiled execution record states its whole lattice and its conf beside
    its density or restores nothing: one missing its overlap or its density refuses naming it,
    and the complete record restores and predicts."""
    from tcip_mcp.pipelines.execution import Execution, prepare
    from tests._producer_fixtures import gray_frame
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_OVERLAP, predicted_over, project_checkpoint, verified_checkpoint,
    )

    checkpoint = verified_checkpoint(tmp_path)
    images_dir = str(Path(gray_frame(tmp_path)).parent)
    p, _results = predicted_over(tmp_path, project_checkpoint(tmp_path), images_dir, device="cpu",
                                 tile=True, tile_size=64, overlap=SAMPLE_OVERLAP)
    record = p.execution.record()

    with pytest.raises(ValueError, match="overlap"):
        Execution.of({**record, "overlap": None})
    with pytest.raises(ValueError, match="density"):
        Execution.of({**record, "density": None})
    restored = prepare(checkpoint, images_dir=images_dir, device="cpu",
                       restored=Execution.of(record)).runnable()
    (result,) = restored.predict(restored.paths)
    assert result["tiles"] >= 1


def test_the_published_cap_reaches_the_row_scoring_its_bucket(tmp_path: Path, monkeypatch):
    """The cap a frame was predicted under, published on its document, is the cap on the row
    scoring that bucket against the frame's ground truth; a proposal bucket's row carries none,
    no forward having capped it."""
    import tcip_mcp.pipelines.training.evaluation as evaluation
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.pipelines.derivations import detection_cap
    from tcip_mcp.pipelines.image_utils import resolve_image_path
    from tcip_mcp.pipelines.training.evaluation import score_bucket
    from tcip_mcp.tools.inference_tools import run_inference
    from tcip_mcp.tools.proposal_tools import stage_proposals
    from tests._producer_fixtures import gray_frame, label_image
    from tests._verified_checkpoint_fixtures import project_checkpoint, verified_checkpoint

    image = Path(gray_frame(tmp_path / "images" / UNDATED_BUCKET, 96, "a.png"))
    label_image(image, [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], 96, 96)
    published = run_inference(tmp_path, project_checkpoint(tmp_path), images_dir=str(image.parent),
                              bucket="capped", device="cpu",
                              stated=Stated(tile=False, conf=SAMPLE_CONF))
    assert "error" not in published, published
    staged = stage_proposals(tmp_path, str(image), model_name="detector", boxes=[
        {"subject": "bud", "conf": 0.9, "cx": 0.5, "cy": 0.5, "w": 0.25, "h": 0.25}])
    assert "error" not in staged, staged

    scored: list[dict] = []
    real = evaluation.records_from_annotation

    def recording(*args, **kwargs):
        scored.append(real(*args, **kwargs))
        return scored[-1]

    monkeypatch.setattr(evaluation, "records_from_annotation", recording)

    def row(name: str) -> dict:
        score_bucket([resolve_image_path(image)], read_bucket(tmp_path, name),
                     iou_threshold=0.5, conf_threshold=0.0, trait=None)
        return scored[-1]

    density = verified_checkpoint(tmp_path).spec.data.train_object_density
    capped, proposed = row("capped"), row(staged["bucket"])
    assert capped["cap"] == detection_cap(density, 96 * 96)
    assert capped["count"] == len(capped["dt"]) <= capped["cap"]
    assert (proposed["cap"], proposed["count"]) == (None, 1)
