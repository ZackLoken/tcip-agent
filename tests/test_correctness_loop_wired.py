"""The correctness loop is wired to a sanctioned surface.

``check_model_contract`` / ``overfit_check`` must have production callers, so a broken
bespoke builder cannot waste a full audited run: ``preflight_config(smoke=True)`` builds + smokes
at the resolved dims and blocks a broken builder, and ``TrainContext`` exposes the checks + craft
primitives so a hand-rolled ``train(ctx)`` self-proves.
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

import pytest

from tests._chain_fixtures import BESPOKE_CLASSIFIER, BESPOKE_SEMANTIC_SEG, training_config

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_mcp.pipelines.model_build import resolve_contract_dims  # noqa: E402
from tcip_mcp.pipelines.training.envelope import TrainContext  # noqa: E402
from tests.tiny_trainer_fixtures import trainer_run  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402


def _broken_builder(**kwargs):
    """An 'agent-written' builder that imports fine but fails the measurement contract: its
    eval-mode forward returns a bare tensor, not the list[dict] detection scorers consume."""
    import torch.nn as nn

    class _Broken(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, images, targets=None):
            if self.training:
                return {"loss": self.lin(torch.rand(1, 4)).sum()}
            return torch.rand(3)  # not list[dict], violates the detection boundary

    return _Broken()


class _ValuesDataset(Dataset):
    """Four blank images, each target one scalar value."""

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return torch.zeros(3, 32, 32), {"values": torch.tensor(float(idx))}


def _bespoke_task_dataset(**_kwargs):
    """Agent-authored dataset for a task the platform does not enumerate."""
    return _ValuesDataset()


def _strict_bespoke_dataset(samples=None, scope=None, transforms=None, task=None):
    """Declares only what the training path passes: no `**kwargs` catch-all to absorb stray keys."""
    return _ValuesDataset()


def _unbuildable_dataset(**_kwargs):
    raise RuntimeError("cannot open the source for this task")


def _bespoke_classification_dataset(samples=None, scope=None, transforms=None, task=None):
    """An agent-authored classification dataset that owns its own class space: the platform
    resolves no count for a run built through a builder, which is why such a run must smoke the
    batch it holds rather than one synthesized at a count nobody resolved."""
    from torch.utils.data import Dataset

    class _DS(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, idx):
            return torch.zeros(3, 32, 32), {"labels": idx % 3}

    return _DS()


def _admitted_tree(tmp_path):
    """An images directory whose label documents the platform's own producer admits samples from,
    for a bespoke run: a builder does not exempt its run from naming the data the producer
    reads."""
    from PIL import Image
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import SubjectRegistry, Subject
    from tests._producer_fixtures import label_image, registry_over

    root = tmp_path / "ds"
    imgs = root / "images" / UNDATED_BUCKET
    imgs.mkdir(parents=True)
    registry_over(root, SubjectRegistry(subjects=(Subject(name="leaf"),)))
    for stem in ("a", "b", "c", "d"):
        Image.new("RGB", (32, 32)).save(imgs / f"{stem}.png")
        label_image(imgs / f"{stem}.png", [Annotation(subject="leaf", geometry=BBox(2, 2, 10, 10))],
                    32, 32, keep_empty=True)
    return imgs


TASK_MODEL = f"{__name__}:_bespoke_task_model"
"""The ``model_source`` builder of :func:`_bespoke_task_model`."""


def _bespoke_task_model(**_kwargs):
    """Agent-authored model for that same task: trains and emits a scored output."""
    import torch.nn as nn

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(3 * 32 * 32, 1)

        def forward(self, images, targets=None):
            pred = self.head(images.flatten(1)).squeeze(-1)
            if self.training:
                return {"loss": ((pred - targets["values"]) ** 2).mean()}
            return {"values": pred}

    return _Net()


# --------------------------------------------------------------------------
# resolve_contract_dims: the size the run will actually see
# --------------------------------------------------------------------------

def test_resolve_contract_dims_prefers_tile_edge_over_default():
    cfg = {"model_source": {"task": "detection"},
           "data": {"tiling": {"enabled": True, "tile_size": 512}}}

    dims = resolve_contract_dims(cfg, "detection", {"in_chans": 4, "num_classes": 5})
    assert dims == {"in_chans": 4, "num_classes": 5, "img_size": 512}


def test_model_dims_states_only_what_the_run_holds():
    """A run that records no width refuses by name rather than building at a guess; a run that
    holds no count states none, leaving the contract to say whether its own task needed one; a rank
    count answers for an ordinal run under its own name, and img_size is the only value the
    contract resolver adds."""
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.model_build import model_dims

    cfg, scope = {"model_source": {}}, ClassScope()

    with pytest.raises(ValueError, match="data.num_channels"):
        model_dims(scope, {})
    detector = model_dims(scope, {"num_channels": 3})
    assert detector == {"in_chans": 3}
    assert resolve_contract_dims(cfg, "detection", detector) == {
        "in_chans": 3, "img_size": 224}  # a detector's synthetic box needs no count
    ordinal = model_dims(scope, {"num_channels": 3, "num_ranks": 4})
    assert ordinal == {"in_chans": 3, "num_ranks": 4}
    assert resolve_contract_dims(cfg, "ordinal", ordinal) == {
        "in_chans": 3, "num_classes": 4, "img_size": 224}


def test_model_dims_hands_the_admitted_subject_and_every_attribute(tmp_path):
    """A scoped run's model is built at the one subject its samples were admitted under and one
    head per attribute the registry declares, never a count stated beside them."""
    from tcip_mcp.pipelines.data.label_queries import registry_scope
    from tcip_mcp.pipelines.model_build import model_dims
    from tcip_mcp.subject_registry import Attribute, Subject, SubjectRegistry
    from tests._producer_fixtures import registry_over

    attributes = (Attribute("color", "categorical", ("red", "blue")),
                  Attribute("grade", "ordinal", ("low", "mid", "high")))
    registry_over(tmp_path, SubjectRegistry(subjects=(Subject(name="bud",
                                                               attributes=attributes),)))
    scope = registry_scope(tmp_path / "images", "bud")

    assert model_dims(scope, {"num_channels": 3}) == {"in_chans": 3, "num_classes": 1,
                                                      "attributes": attributes}


# --------------------------------------------------------------------------
# preflight_config(smoke=True): builds + smokes, blocks a broken builder
# --------------------------------------------------------------------------

def test_preflight_smoke_blocks_broken_builder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = _admitted_tree(tmp_path)
    cfg = training_config(
        {"builder": f"{__name__}:_broken_builder", "task": "detection"},
        {"images_dir": str(imgs), "scope": {"subject": "leaf"},
         "split": {"seed": 0, "val_ratio": 0.15}},
        batch_size=1, stages=[{"freeze_to": 0, "epochs": 1}])
    # Fast path (no smoke) is structurally valid: the builder imports fine.
    assert preflight_config(tmp_path, cfg)["valid"] is True
    # Smoke path builds + runs the contract and catches the measurement-boundary violation.
    r = preflight_config(tmp_path, cfg, smoke=True)
    assert r["valid"] is False
    assert any("model contract" in i for i in r["issues"])
    assert r["smoke"]["dims"]["img_size"] == 224  # the untiled fallback edge, resolved


def test_preflight_smoke_passes_valid_builder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import preflight_config
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR

    imgs = _admitted_tree(tmp_path)
    cfg = training_config(
        BUILT_DETECTOR, {"images_dir": str(imgs), "scope": {"subject": "leaf"},
                         "split": {"seed": 0, "val_ratio": 0.15}},
        batch_size=1, stages=[{"freeze_to": 0, "epochs": 1}])
    r = preflight_config(tmp_path, cfg, smoke=True, overfit=True)
    assert r["valid"] is True, r["issues"]
    assert r["smoke"]["ok"] is True
    assert "overfit_check" in r  # voluntary diagnostic reported, non-gating


# --------------------------------------------------------------------------
# A task the contract has no synthetic schema for is smoked against a real batch:
# no task taxonomy, and no run launching with the contract silently skipped.
# --------------------------------------------------------------------------

def test_preflight_builds_and_smokes_at_the_count_the_run_resolved(tmp_path, monkeypatch):
    """The model is built and smoked at the class count this run's own ground truth carries, read
    the way its loaders read it, and at the band count its sources carry: the platform hands the
    builder both, so no head can be declared wider than the data."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import preflight_config
    from tests.test_mask_and_table_membership import _three_class_masks

    images_dir, masks_dir = _three_class_masks(tmp_path / "ds")
    cfg = training_config(
        {"builder": BESPOKE_SEMANTIC_SEG, "task": "semantic_seg"},
        {"images_dir": str(images_dir), "labels_dir": str(masks_dir),
         "split": {"seed": 0, "val_ratio": 0.15}},
        batch_size=1, stages=[{"freeze_to": 0, "epochs": 1}])

    r = preflight_config(tmp_path, cfg, smoke=True)

    assert r["smoke"]["dims"]["num_classes"] == 3, r["smoke"]
    assert r["smoke"]["dims"]["in_chans"] == 3


def test_preflight_smokes_a_single_class_run_within_its_own_count(tmp_path, monkeypatch):
    """The synthetic batch is shaped within the count the run resolved: an all-background mask
    run trains one class, and a batch shaped at a wider minimum would index past the head the run
    actually builds while its own data passes."""
    import numpy as np
    from PIL import Image

    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import preflight_config

    images_dir, masks_dir = tmp_path / "ds" / "images" / UNDATED_BUCKET, tmp_path / "ds" / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    for stem in ("a", "b", "c", "d"):
        Image.new("RGB", (32, 32), (80, 90, 100)).save(images_dir / f"{stem}.png")
        Image.fromarray(np.zeros((32, 32), dtype=np.uint8), mode="L").save(
            masks_dir / f"{stem}.png"
        )
    cfg = training_config(
        {"builder": BESPOKE_SEMANTIC_SEG, "task": "semantic_seg"},
        {"images_dir": str(images_dir), "labels_dir": str(masks_dir),
         "split": {"seed": 0, "val_ratio": 0.15}},
        batch_size=1, stages=[{"freeze_to": 0, "epochs": 1}])

    synthetic = preflight_config(tmp_path, cfg, smoke=True)

    assert synthetic["smoke"]["dims"]["num_classes"] == 1
    assert synthetic["smoke"]["batch_source"] == "synthetic"
    assert synthetic["smoke"]["ok"] is True, synthetic["smoke"]["issues"]
    # The same model over the run's own batch reaches the same verdict.
    from tcip_mcp.pipelines.data.split_construction import resolve_run
    from tcip_mcp.pipelines.model_build import build_model
    from tcip_mcp.pipelines.model_contract import check_model_contract
    from tcip_mcp.pipelines.schemas import train_config
    from tcip_mcp.tools.training_tools import _one_real_batch

    batch, why = _one_real_batch(
        "semantic_seg", resolve_run(cfg, train_config(cfg), project=tmp_path).train_ds)
    assert batch is not None, why
    smoked = synthetic["smoke"]["dims"]
    model = build_model(cfg, {"in_chans": smoked["in_chans"], "num_classes": smoked["num_classes"]})
    assert check_model_contract(model, "semantic_seg", sample_batch=batch)["ok"]


def test_preflight_smokes_bespoke_task_on_a_real_batch(tmp_path, monkeypatch):
    """A bespoke dataset_source run declaring no width has no synthetic shape to smoke at either,
    and the batch from its own dataset answers for both."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = _admitted_tree(tmp_path)
    cfg = training_config(
        {"builder": TASK_MODEL, "task": "bunch_compactness"},
        {"images_dir": str(imgs), "scope": {"subject": "leaf"},
         "split": {"seed": 0, "val_ratio": 0.15},
         "dataset_source": {"builder": f"{__name__}:_bespoke_task_dataset",
                            "task": "bunch_compactness"}},
        stages=[{"freeze_to": 0, "epochs": 1}])
    r = preflight_config(tmp_path, cfg, smoke=True, overfit=True)
    assert r["valid"] is True, r["issues"]
    # The contract actually ran: a real batch stood in for the missing synthetic schema.
    assert r["smoke"]["not_smokeable"] is None
    assert r["smoke"]["ok"] is True
    assert r["smoke"]["train_loss"] is not None
    assert r["smoke"]["batch_source"] == "dataset"  # provenance: which reference proved it
    # The same batch reaches overfit_check; re-synthesizing here would report a false "does not
    # learn" for exactly the bespoke tasks the real-batch path exists to serve.
    assert r["overfit_check"]["issue"] is None, r["overfit_check"]
    assert r["overfit_check"]["passed"] is True, r["overfit_check"]


def test_preflight_smoke_batch_matches_what_the_run_will_build(tmp_path, monkeypatch):
    """The smoked dataset is the one the run's own resolution builds, which a bespoke builder
    accepting only the producer's own four names builds fine."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.pipelines.data.split_construction import resolve_run
    from tcip_mcp.pipelines.schemas import train_config
    from tcip_mcp.tools.training_tools import _one_real_batch

    imgs = _admitted_tree(tmp_path)
    data = {"images_dir": str(imgs), "scope": {"subject": "leaf"},
            "split": {"seed": 0, "val_ratio": 0.15},
            "dataset_source": {"builder": f"{__name__}:_strict_bespoke_dataset",
                               "task": "bunch_compactness"}}
    config = training_config({"task": "bunch_compactness"}, data)
    resolution = resolve_run(config, train_config(config), project=tmp_path)

    batch, why = _one_real_batch("bunch_compactness", resolution.train_ds)
    assert why is None, why
    assert batch is not None


def test_preflight_blocks_when_no_batch_can_be_built(tmp_path, monkeypatch):
    """A dataset that cannot produce an item blocks the launch at its resolution, naming the
    builder's own reason, rather than reaching a skipped smoke check."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = _admitted_tree(tmp_path)
    # Structurally valid, but the dataset cannot produce an item.
    cfg = training_config(
        {"builder": TASK_MODEL, "task": "bunch_compactness"},
        {"images_dir": str(imgs), "scope": {"subject": "leaf"},
         "split": {"seed": 0, "val_ratio": 0.15},
         "dataset_source": {"builder": f"{__name__}:_unbuildable_dataset",
                            "task": "bunch_compactness"}},
        stages=[{"freeze_to": 0, "epochs": 1}])
    r = preflight_config(tmp_path, cfg, smoke=True)
    assert r["valid"] is False
    # Only the builder raises this text, so the refusal is the unbuildable dataset's own.
    assert any("cannot open the source" in i for i in r["issues"]), r["issues"]


# --------------------------------------------------------------------------
# ctx surface: a custom loop self-proves + reuses the craft primitives
# --------------------------------------------------------------------------

def _ctx_for(project, task: str, builder: str, data: dict):
    """A context over a run of ``project`` whose table ground truth recorded the empty scope
    admission writes."""
    config = training_config({"builder": builder, "task": task}, {"scope": {}, **data})
    run = trainer_run(config, "out", project=project, has_val_loader=False, id="auto-run-6")
    return TrainContext(run=run, train_loader=None, val_loader=None)


def test_ctx_check_contract_and_overfit_check(tmp_path):
    """The model is built and smoked at what the run recorded: its own width and class count, the
    sizes its loaders were built at."""
    ctx = _ctx_for(tmp_path, "classification", BESPOKE_CLASSIFIER,
                   data={"num_channels": 3, "num_classes": 2})
    report = ctx.check_contract()
    assert report["ok"], report["issues"]
    # overfit is voluntary + non-gating; steps flow through as an override kwarg.
    over = ctx.overfit_check(steps=15)
    assert over["passed"], over["issue"]


def test_ctx_smokes_a_bespoke_dataset_run_at_the_count_its_data_states(tmp_path, monkeypatch):
    """A bespoke dataset owns its class space, so its run states the count on its data section;
    the platform records the band count its sources carry beside it and builds the model at both,
    and both proofs run at those dimensions."""
    import csv

    from PIL import Image
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.data.split_construction import auto_train_val
    from tcip_mcp.pipelines.training.collation import task_collate

    monkeypatch.chdir(tmp_path)
    imgs, table = tmp_path / "images" / UNDATED_BUCKET, tmp_path / "labels.csv"
    imgs.mkdir(parents=True)
    rows = []
    for index in range(4):
        Image.new("RGB", (32, 32), (40 * index, 90, 120)).save(imgs / f"img{index}.png")
        rows.append((f"img{index}", index % 3))
    with open(table, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stem", "label"))
        writer.writerows(rows)
    data = {"images_dir": str(imgs), "labels_dir": str(table), "num_classes": 3,
            "split": {"seed": 0, "val_ratio": 0.15},
            "dataset_source": {"builder": f"{__name__}:_bespoke_classification_dataset"}}
    config = training_config({"builder": BESPOKE_CLASSIFIER, "task": "classification"}, data)
    train_ds, _val_ds, _partition = auto_train_val(tmp_path, "classification", data, None)
    assert (data["num_channels"], data["num_classes"]) == (3, 3)
    loader = DataLoader(train_ds, batch_size=2, collate_fn=task_collate("classification"))
    ctx = TrainContext(run=trainer_run(config, tmp_path / "out", project=tmp_path,
                                       has_val_loader=False, id="auto-run-62"),
                       train_loader=loader, val_loader=None)

    report = ctx.check_contract()
    over = ctx.overfit_check(steps=3)

    assert report["not_smokeable"] is None, report
    assert report["ok"], report["issues"]
    assert report["train_loss"] is not None
    assert over["issue"] is None, over


def test_ctx_refuses_a_classification_run_recording_no_class_count(tmp_path):
    """A classification run recording no class count is refused by name by both proofs: a batch
    shaped at a count nobody resolved would prove the model against a class space the run does
    not train in."""
    ctx = _ctx_for(tmp_path, "classification", TASK_MODEL,
                   data={"num_channels": 3})

    with pytest.raises(ValueError, match="records no num_classes"):
        ctx.check_contract()
    with pytest.raises(ValueError, match="records no num_classes"):
        ctx.overfit_check(steps=2)


def test_ctx_apply_stage_freeze_matches_trainer_guard(tmp_path):
    from tcip_mcp.pipelines.training.generic_trainer import apply_stage_freeze

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    ctx = TrainContext(run=trainer_run(training_config({"task": "regression"}, {}), "out",
                                       project=tmp_path, has_val_loader=False, id="auto-run-7"),
                       train_loader=None)
    full = ctx.apply_stage_freeze(model, 0)
    assert full == sum(p.numel() for p in model.parameters())
    # A shrink relative to the previous stage violates the monotonic guard.
    with pytest.raises(RuntimeError, match="Non-decreasing unfreeze"):
        apply_stage_freeze(model, 0, prev_trainable=full + 1)
