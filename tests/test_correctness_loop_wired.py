"""The correctness loop is wired to a sanctioned surface.

``check_model_contract`` / ``overfit_check`` must have production callers, so a broken
bespoke builder cannot waste a full audited run: ``preflight_config(smoke=True)`` builds + smokes
at the resolved dims and blocks a broken builder, and ``TrainContext`` exposes the checks + craft
primitives so a hand-rolled ``train(ctx)`` self-proves.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_mcp.pipelines.model_build import resolve_contract_dims  # noqa: E402
from tcip_mcp.pipelines.training.envelope import TrainContext  # noqa: E402
from tcip_mcp.pipelines.training.run_registry import create_run  # noqa: E402


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


def _bespoke_task_dataset(**_kwargs):
    """Agent-authored dataset for a task the platform does not enumerate."""
    from torch.utils.data import Dataset

    class _DS(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return torch.zeros(3, 32, 32), {"values": torch.tensor(float(idx))}

    return _DS()


def _strict_bespoke_dataset(samples=None, id_map=None, transforms=None, task=None):
    """Declares only what the training path passes: no `**kwargs` catch-all to absorb stray keys."""
    from torch.utils.data import Dataset

    class _DS(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return torch.zeros(3, 32, 32), {"values": torch.tensor(float(idx))}

    return _DS()


def _unbuildable_dataset(**_kwargs):
    raise RuntimeError("cannot open the source for this task")


def _bespoke_classification_dataset(samples=None, id_map=None, transforms=None, task=None):
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
    """An images directory and a label tree the platform's own producer admits samples from, for a
    bespoke run: a builder does not exempt its run from naming the data the producer reads."""
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import SubjectRegistry, Subject, write_registry

    root = tmp_path / "ds"
    imgs, lbls = root / "images", root / "annotations"
    imgs.mkdir(parents=True)
    lbls.mkdir(parents=True)
    write_registry(root / "subjects.json", SubjectRegistry(subjects=(Subject(name="leaf"),)))
    for stem in ("a", "b", "c", "d"):
        Image.new("RGB", (32, 32)).save(imgs / f"{stem}.png")
        json_io.write_annotations(lbls / f"{stem}.json",
                                  [Annotation(subject="leaf", geometry=BBox(2, 2, 10, 10))],
                                  32, 32, keep_empty=True)
    return imgs, lbls


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
    from tcip_mcp.pipelines.data.selection import ClassScope

    cfg = {
        "model_source": {"builder_kwargs": {"in_chans": 4}, "task": "detection"},
        "data": {"tiling": {"enabled": True, "tile_size": 512}},
    }

    dims = resolve_contract_dims(cfg, "detection", scope=ClassScope(),
                                 sizes={"num_classes": 5})
    assert dims == {"in_chans": 4, "num_classes": 5, "img_size": 512}


def test_resolve_contract_dims_answers_none_rather_than_inventing_a_size():
    """A run that states no width has no synthetic shape at all and says so; a run that states no
    count states none here, leaving the contract to say whether its own task needed one. The rank
    count answers for an ordinal run, and img_size is the only value this resolver supplies."""
    from tcip_mcp.pipelines.data.selection import ClassScope

    cfg, scope = {"model_source": {}}, ClassScope()

    assert resolve_contract_dims(cfg, "detection", scope=scope, sizes={}) is None
    assert resolve_contract_dims(cfg, "detection", scope=scope, sizes={"num_channels": 3}) == {
        "in_chans": 3, "img_size": 224}  # a detector's synthetic box needs no count
    assert resolve_contract_dims(cfg, "ordinal", scope=scope,
                                 sizes={"num_channels": 3, "num_ranks": 4}) == {
        "in_chans": 3, "num_classes": 4, "img_size": 224}


def test_resolve_contract_dims_takes_the_admitted_map_over_the_heads_declared_count(tmp_path):
    """The count the smoke forwards at is the one the run's samples were admitted under, whatever
    the head declares: a smoke proving a model against a head wider than the run's own class space
    proves it against a model that will not train."""
    from tcip_mcp.pipelines.data.selection import ClassScope

    cfg = {
        "model_source": {"builder_kwargs": {"num_classes": 5}},
        "data": {"subject": "bud", "attribute": "opening", "labels_dir": str(tmp_path / "labels")},
    }
    scope = ClassScope(subject="bud", attribute="opening", id_map={"open": 0, "closed": 1})

    assert resolve_contract_dims(cfg, "detection", scope=scope,
                                 sizes={"num_channels": 3})["num_classes"] == 2


# --------------------------------------------------------------------------
# preflight_config(smoke=True): builds + smokes, blocks a broken builder
# --------------------------------------------------------------------------

def test_preflight_smoke_blocks_broken_builder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import preflight_config

    imgs, lbls = _admitted_tree(tmp_path)
    cfg = {
        "model_source": {"builder": f"{__name__}:_broken_builder", "task": "detection",
                         "in_chans": 3},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls), "subject": "leaf"},
        "batch_size": 1, "stages": [{"freeze_to": 0, "epochs": 1}],
    }
    # Fast path (no smoke) is structurally valid: the builder imports fine.
    assert preflight_config(cfg)["valid"] is True
    # Smoke path builds + runs the contract and catches the measurement-boundary violation.
    r = preflight_config(cfg, smoke=True)
    assert r["valid"] is False
    assert any("model contract" in i for i in r["issues"])
    assert r["smoke"]["dims"]["img_size"] == 224  # the untiled fallback edge, resolved


def test_preflight_smoke_passes_valid_builder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import preflight_config

    imgs, lbls = _admitted_tree(tmp_path)
    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1, "in_chans": 3, "min_size": 64,
                                            "max_size": 128},
                         "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls), "subject": "leaf"},
        "batch_size": 1, "stages": [{"freeze_to": 0, "epochs": 1}],
    }
    r = preflight_config(cfg, smoke=True, overfit=True)
    assert r["valid"] is True, r["issues"]
    assert r["smoke"]["ok"] is True
    assert "overfit_check" in r  # voluntary diagnostic reported, non-gating


# --------------------------------------------------------------------------
# A task the contract has no synthetic schema for is smoked against a real batch:
# no task taxonomy, and no run launching with the contract silently skipped.
# --------------------------------------------------------------------------

def test_preflight_smokes_at_the_count_the_run_resolved_not_the_heads(tmp_path, monkeypatch):
    """The smoke forwards at the class count this run's own ground truth carries, read the way
    its loaders read it: a head declared wider than the data never widens the batch the contract
    is earned against."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import preflight_config
    from tests.test_mask_and_table_membership import _three_class_masks

    images_dir, masks_dir = _three_class_masks(tmp_path / "ds")
    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_semantic_seg",
                         "builder_kwargs": {"num_classes": 9, "in_chans": 3},
                         "task": "semantic_seg"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(masks_dir)},
        "batch_size": 1, "stages": [{"freeze_to": 0, "epochs": 1}],
    }

    r = preflight_config(cfg, smoke=True)

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

    images_dir, masks_dir = tmp_path / "ds" / "images", tmp_path / "ds" / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    for stem in ("a", "b", "c", "d"):
        Image.new("RGB", (32, 32), (80, 90, 100)).save(images_dir / f"{stem}.png")
        Image.fromarray(np.zeros((32, 32), dtype=np.uint8), mode="L").save(masks_dir / f"{stem}.png")
    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_semantic_seg",
                         "builder_kwargs": {"num_classes": 1, "in_chans": 3},
                         "task": "semantic_seg"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(masks_dir)},
        "batch_size": 1, "stages": [{"freeze_to": 0, "epochs": 1}],
    }

    synthetic = preflight_config(cfg, smoke=True)

    assert synthetic["smoke"]["dims"]["num_classes"] == 1
    assert synthetic["smoke"]["batch_source"] == "synthetic"
    assert synthetic["smoke"]["ok"] is True, synthetic["smoke"]["issues"]
    # The same model over the run's own batch reaches the same verdict.
    from tcip_mcp.pipelines.model_build import build_model
    from tcip_mcp.pipelines.model_contract import check_model_contract
    from tcip_mcp.tools.training_tools import _one_real_batch

    batch, why = _one_real_batch("semantic_seg", cfg)
    assert batch is not None, why
    assert check_model_contract(build_model(cfg), "semantic_seg", sample_batch=batch)["ok"]


def test_preflight_smokes_bespoke_task_on_a_real_batch(tmp_path, monkeypatch):
    """A bespoke dataset_source run declaring no width has no synthetic shape to smoke at either,
    and the batch from its own dataset answers for both."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import preflight_config

    imgs, lbls = _admitted_tree(tmp_path)
    cfg = {
        "model_source": {"builder": f"{__name__}:_bespoke_task_model", "task": "bunch_compactness"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls), "subject": "leaf",
                 "dataset_source": {"builder": f"{__name__}:_bespoke_task_dataset",
                                    "task": "bunch_compactness"}},
        "batch_size": 2, "stages": [{"freeze_to": 0, "epochs": 1}],
    }
    r = preflight_config(cfg, smoke=True, overfit=True)
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
    """The smoked dataset is built the way the training path builds it, not from a private key
    list.

    A bespoke builder only has to accept what the training path passes it, which is the producer's
    own four names: forwarding a directory here would reject a dataset that trains fine, turning
    the contract rail into a blocker for valid work.
    """
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import _one_real_batch

    imgs, lbls = _admitted_tree(tmp_path)
    data = {"images_dir": str(imgs), "labels_dir": str(lbls), "subject": "leaf",
            "dataset_source": {"builder": f"{__name__}:_strict_bespoke_dataset",
                               "task": "bunch_compactness"}}

    batch, why = _one_real_batch("bunch_compactness", {"data": data})
    assert why is None, why
    assert batch is not None


def test_preflight_blocks_when_no_batch_can_be_built(tmp_path, monkeypatch):
    """Unsmokeable is a blocked launch, not a skipped check: the boundary stays proven."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import preflight_config

    imgs, lbls = _admitted_tree(tmp_path)
    cfg = {  # structurally valid, but the dataset cannot produce an item
        "model_source": {"builder": f"{__name__}:_bespoke_task_model", "task": "bunch_compactness"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls), "subject": "leaf",
                 "dataset_source": {"builder": f"{__name__}:_unbuildable_dataset",
                                    "task": "bunch_compactness"}},
        "batch_size": 2, "stages": [{"freeze_to": 0, "epochs": 1}],
    }
    r = preflight_config(cfg, smoke=True)
    assert r["valid"] is False
    # Assert on what only the blocking path produces: "valid is False" alone is vacuous here,
    # since an unrelated route (an unknown task falling through to a classification-shaped
    # synthetic batch, which this model also rejects) could reach False too.
    assert r["smoke"]["not_smokeable"]
    assert any("measurement boundary is unproven" in i for i in r["issues"]), r["issues"]
    # The real reason the batch could not be built is surfaced, not swallowed into the log.
    assert any("cannot open the source" in i for i in r["issues"]), r["issues"]


# --------------------------------------------------------------------------
# ctx surface: a custom loop self-proves + reuses the craft primitives
# --------------------------------------------------------------------------

def _ctx_for(task: str, builder: str, data: dict | None = None, **builder_kwargs):
    config = {"model_source": {"builder": builder, "builder_kwargs": builder_kwargs, "task": task},
              "device": "cpu", "data": data or {}}
    run = create_run(config, "out", id="auto-run-6")
    return TrainContext(run=run, train_loader=None, val_loader=None, task=task)


def test_ctx_check_contract_and_overfit_check():
    """The smoke forwards at what the run recorded: its own width and class count, the sizes its
    loaders were built at."""
    ctx = _ctx_for("classification", "tests.bespoke_models:build_bespoke_classifier",
                   data={"num_channels": 3, "num_classes": 2}, num_classes=2, in_chans=3)
    report = ctx.check_contract()
    assert report["ok"], report["issues"]
    # overfit is voluntary + non-gating; steps flow through as an override kwarg.
    over = ctx.overfit_check(steps=15)
    assert over["passed"], over["issue"]


def test_ctx_smokes_against_its_own_batch_when_the_run_records_no_width(tmp_path):
    """A hand-rolled loop whose run states no width has its own train loader, and that batch is
    what the contract is proved against: the smoke never invents a width to synthesize one at."""
    import csv

    from PIL import Image
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.training.collation import task_collate
    from tests._producer_fixtures import dataset_over

    images_dir, csv_path = tmp_path / "images", tmp_path / "labels.csv"
    images_dir.mkdir()
    rows = []
    for index in range(4):
        Image.new("RGB", (32, 32), (40 * index, 90, 120)).save(images_dir / f"img{index}.png")
        rows.append((f"img{index}", index % 2))
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stem", "label"))
        writer.writerows(rows)
    loader = DataLoader(dataset_over("classification", str(images_dir), str(csv_path)),
                        batch_size=2, collate_fn=task_collate("classification"))

    config = {"model_source": {"builder": "tests.bespoke_models:build_bespoke_classifier",
                               "builder_kwargs": {"num_classes": 2}, "task": "classification"},
              "device": "cpu"}
    ctx = TrainContext(run=create_run(config, str(tmp_path / "out"), id="auto-run-61"),
                       train_loader=loader, val_loader=None, task="classification")

    report = ctx.check_contract()

    assert report["not_smokeable"] is None, report
    assert report["ok"], report["issues"]
    assert report["train_loss"] is not None


def test_ctx_smokes_its_own_batch_when_the_run_states_a_width_but_no_count(tmp_path, monkeypatch):
    """A run whose model declares a width and whose bespoke dataset owns its class count holds a
    loader, and that batch is what both proofs run against.

    Its resolved dimensions name the width and no count, which is a shape nothing can be
    synthesized at; the loader answers for it. A caller deciding that for itself, by asking only
    whether dimensions exist, refuses work the contract would admit.
    """
    import csv

    from PIL import Image
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.data.split_construction import auto_train_val
    from tcip_mcp.pipelines.training.collation import task_collate

    monkeypatch.chdir(tmp_path)
    imgs, table = tmp_path / "images", tmp_path / "labels.csv"
    imgs.mkdir()
    rows = []
    for index in range(4):
        Image.new("RGB", (32, 32), (40 * index, 90, 120)).save(imgs / f"img{index}.png")
        rows.append((f"img{index}", index % 3))
    with open(table, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stem", "label"))
        writer.writerows(rows)
    data = {"images_dir": str(imgs), "labels_dir": str(table),
            "dataset_source": {"builder": f"{__name__}:_bespoke_classification_dataset",
                               "task": "classification"}}
    config = {"model_source": {"builder": "tests.bespoke_models:build_bespoke_classifier",
                               "builder_kwargs": {"num_classes": 3, "in_chans": 3},
                               "task": "classification"},
              "device": "cpu", "data": data}
    train_ds, _val_ds, _partition = auto_train_val("classification", data, None)
    loader = DataLoader(train_ds, batch_size=2, collate_fn=task_collate("classification"))
    ctx = TrainContext(run=create_run(config, str(tmp_path / "out"), id="auto-run-62"),
                       train_loader=loader, val_loader=None, task="classification")

    report = ctx.check_contract()
    over = ctx.overfit_check(steps=3)

    assert report["not_smokeable"] is None, report
    assert report["ok"], report["issues"]
    assert report["train_loss"] is not None
    assert over["issue"] is None, over


def test_ctx_says_one_thing_when_it_can_smoke_nothing():
    """A classification run recording no class count, holding no batch to smoke, gets the same
    named condition from both proofs: a batch shaped at a count nobody resolved would prove the
    model against a class space the run does not train in."""
    ctx = _ctx_for("classification", "tests.bespoke_models:build_bespoke_classifier",
                   num_classes=2, in_chans=3)

    report = ctx.check_contract()
    over = ctx.overfit_check(steps=2)

    assert report["not_smokeable"] == over["issue"]
    assert "no dimensions to synthesize a classification batch at" in report["not_smokeable"]
    assert "class count" in report["not_smokeable"]
    assert report["ok"] is False and over["passed"] is False


def test_ctx_apply_stage_freeze_matches_trainer_guard():
    from tcip_mcp.pipelines.training.generic_trainer import apply_stage_freeze

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    ctx = TrainContext(run=create_run({}, "out", id="auto-run-7"), train_loader=None, task="classification")
    full = ctx.apply_stage_freeze(model, 0)
    assert full == sum(p.numel() for p in model.parameters())
    # A shrink relative to the previous stage violates the monotonic guard.
    with pytest.raises(RuntimeError, match="Non-decreasing unfreeze"):
        apply_stage_freeze(model, 0, prev_trainable=full + 1)
