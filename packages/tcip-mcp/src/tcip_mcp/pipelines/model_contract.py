"""The model-side contract: the measurement boundary, as a behavioral check.

* ``check_model_contract(model, task)``: a behavioral smoke: a train-mode forward yields a finite
loss with a gradient, and an eval-mode forward yields the documented shape for the task
(``list[dict]`` per image for detection/instance_seg, a ``dict`` of tensors otherwise).
* ``overfit_check(model, task, steps)``: a few optimizer steps on one fixed tiny batch (seeded,
CPU) must drive the loss down.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypedDict, Unpack

from pydantic import JsonValue

from tcip_mcp.pipelines.model_build import ContractDims
from tcip_store.values import finite_or_none, stored_number

if TYPE_CHECKING:
    import torch

    from tcip_mcp.pipelines.training.collation import Batch

DETECTOR_TASKS = frozenset({"detection", "instance_seg"})
"""The tasks whose model answers scored boxes and labels per image: the heads a detection cap,
a confidence and an operating point govern."""
TILING_TASKS = frozenset({"detection", "instance_seg"})
"""The tasks a run trains and predicts over tiles of, each tile's targets clipped to it."""
# The tasks ``_synth_batch`` can shape a batch for. A task outside this set is not refused as
# unsupported (the platform has no fixed task taxonomy), but it cannot be smoked blind, so the
# caller supplies ``sample_batch=`` instead of the contract inventing a shape for it.
_SYNTHESIZABLE_TASKS = DETECTOR_TASKS | {
    "classification",
    "ordinal",
    "regression",
    "semantic_seg",
}
_COUNTED_TASKS = _SYNTHESIZABLE_TASKS - DETECTOR_TASKS - {"regression"}
"""The tasks whose synthetic target carries a class or rank id, so a batch for one cannot be
shaped without the run's own count. A detector labels its one box foreground and a regression
target is a value, so neither reads one."""


class ContractKeywords(TypedDict, total=False):
    """The keywords :func:`check_model_contract` takes beside its model and task.

    ``dims`` is the run's resolved dimensions, ``device`` the device it runs on (``"cpu"``
    unstated) and ``sample_batch`` an ``(images, targets)`` pair from the run's own dataset.
    """

    dims: ContractDims | None
    device: str
    sample_batch: Batch | None


class OverfitKeywords(ContractKeywords, total=False):
    """The keywords :func:`overfit_check` takes beside its model, task, seed and determinism.

    Those of :class:`ContractKeywords`, and the optimizer ``steps`` (20 unstated) and learning
    rate ``lr`` (0.01 unstated).
    """

    steps: int
    lr: float


class ContractReport(TypedDict):
    """What :func:`check_model_contract` found.

    ``ok`` is whether no issue was found and ``issues`` each one found; ``train_loss`` the
    train-mode loss, ``eval_output_type`` the output shape the eval pass was held to,
    ``not_smokeable`` why nothing could be smoked, and ``gradient_magnitudes`` each parameter's
    gradient norm; each is ``None`` when its pass did not produce it.
    """

    ok: bool
    issues: list[str]
    train_loss: float | None
    eval_output_type: str | None
    not_smokeable: str | None
    gradient_magnitudes: dict[str, float] | None


class OverfitReport(TypedDict):
    """What :func:`overfit_check` found.

    ``passed`` is whether the loss fell with no ``issue``; ``losses`` each step's loss, and
    ``initial`` and ``final`` the first and last, ``None`` with no step taken.
    """

    passed: bool
    losses: list[float]
    initial: float | None
    final: float | None
    issue: str | None


class LaunchContract(TypedDict):
    """The model contract a launch records on its run.

    ``subject`` names what was checked and ``gating`` whether it gates the launch;
    ``batch_source`` is the batch that drove it (``"dataset"`` or ``"synthetic"``), ``dims`` a
    synthetic batch's dimensions (``None`` for a dataset batch), ``issues`` and
    ``gradient_magnitudes`` the :class:`ContractReport`'s, and ``overfit_check`` the rendered
    :func:`overfit_check` report (:func:`render_overfit_report`), ``None`` when none ran.
    """

    subject: str
    gating: bool
    batch_source: str | None
    dims: ContractDims | None
    issues: list[str]
    gradient_magnitudes: dict[str, float] | None
    overfit_check: dict[str, JsonValue] | None


def no_batch_reason(
    task: str, dims: ContractDims | None, sample_batch: object
) -> str | None:
    """Return why this call can smoke nothing, or ``None`` when it can."""
    if sample_batch is not None:
        return None
    if task not in _SYNTHESIZABLE_TASKS:
        return (
            f"no synthetic batch schema for task {task!r}; pass sample_batch= (an (images, "
            f"targets) pair from this run's dataset) to smoke it. Schemas exist for "
            f"{sorted(_SYNTHESIZABLE_TASKS)}."
        )
    missing = (
        "a frame size"
        if dims is None or dims.get("img_size") is None
        else "a class count"
        if task in _COUNTED_TASKS and dims.get("num_classes") is None
        else None
    )
    if missing is not None:
        return (
            f"no dimensions to synthesize a {task} batch at: this run states {missing} "
            f"nowhere. Pass dims= (its own resolved dimensions) or sample_batch= (an "
            f"(images, targets) pair from its dataset)."
        )
    return None


def _synth_batch(task: str, dims: ContractDims, device: torch.device) -> Batch:
    """Synthesize a minimal batch for ``task`` at ``dims``.

    The batch is per-sample ``(image, target)`` items shaped like a dataset's ``__getitem__``,
    collated with ``task_collate``, each image ``in_chans`` bands at ``img_size``, ``(width,
    height)``. ``num_classes`` is the run's own count, which only a ``_COUNTED_TASKS`` target
    reads; a detection target carries an ``attributes`` row, each attribute's first value, when
    the run's ``attributes`` name any, so every head is driven.
    """
    import torch

    from tcip_mcp.pipelines.data.datasets import target_tensors
    from tcip_mcp.pipelines.training.collation import task_collate

    collate = task_collate(task)
    in_chans, (width, height) = dims["in_chans"], dims["img_size"]
    attributes = dims.get("attributes", ())
    num_classes = dims.get("num_classes")

    if task in DETECTOR_TASKS:
        img = torch.rand(in_chans, height, width, device=device)
        box = [width * 0.2, height * 0.2, width * 0.7, height * 0.7]
        # One foreground instance (labels are 1-indexed), through the loaders' own tensor builder.
        tensors = target_tensors(
            {
                "boxes": [box],
                "labels": [1],
                "iscrowd": [False],
                **(
                    {"attributes": [[0] * len(attributes)]}
                    if attributes
                    else {}
                ),
            }
        )
        target: dict[str, torch.Tensor | int] = {
            k: v.to(device) for k, v in tensors.items()
        }
        target["image_id"] = 0
        if task == "instance_seg":
            mask = torch.zeros(
                (1, height, width), dtype=torch.uint8, device=device
            )
            mask[
                0,
                int(height * 0.2) : int(height * 0.7),
                int(width * 0.2) : int(width * 0.7),
            ] = 1
            target["masks"] = mask
        return collate([(img, target)])

    if task == "regression":
        return collate(
            [
                (
                    torch.rand(in_chans, height, width, device=device),
                    {"values": torch.rand((), device=device)},
                )
                for _ in range(2)
            ]
        )

    assert num_classes is not None, (
        "a counted task reaches here with its own count"
    )
    items = []
    for _ in range(2):
        img = torch.rand(in_chans, height, width, device=device)
        if task == "ordinal":
            # A 0-dim tensor, not a python scalar: the stacking collate rebuilds a scalar on the
            # cpu.
            target = {
                "ranks": torch.randint(0, num_classes, (), device=device)
            }
        elif task == "semantic_seg":
            target = {
                "masks": torch.randint(
                    0, num_classes, (height, width), device=device
                )
            }
        else:
            target = {
                "labels": torch.randint(0, num_classes, (), device=device)
            }
        items.append((img, target))
    return collate(items)


def _driving_batch(
    task: str,
    dims: ContractDims | None,
    sample_batch: Batch | None,
    device: torch.device,
) -> Batch:
    """Return the ``(images, targets)`` a model is driven with.

    That is ``sample_batch`` when given, else synthesized at ``dims``, which
    :func:`no_batch_reason` then admitted; per-image targets through ``instance_targets``.
    """
    from tcip_mcp.pipelines.data.datasets import instance_targets

    batch = sample_batch
    if batch is None:
        assert dims is not None, (
            "no_batch_reason admits no synthesis without dims"
        )
        batch = _synth_batch(task, dims, device)
    match batch:
        case (images, list() as targets):
            return images, instance_targets(targets)
    return batch


def _contains_tensor(
    value: object, _depth: int = 0, _max_depth: int = 4
) -> bool:
    """Return whether ``value`` is a tensor, or a list/tuple/dict that carries one.

    Any depth up to ``_max_depth`` is searched.
    """
    import torch

    if isinstance(value, torch.Tensor):
        return True
    if _depth >= _max_depth:
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(v, _depth + 1, _max_depth) for v in value)
    if isinstance(value, dict):
        return any(
            _contains_tensor(v, _depth + 1, _max_depth) for v in value.values()
        )
    return False


def _forward_loss(model: torch.nn.Module, batch: Batch) -> torch.Tensor:
    """Return a single scalar loss tensor from a train-mode forward (a loss dict summed)."""
    out = model(*batch)
    if isinstance(out, dict):
        terms = [v for v in out.values() if hasattr(v, "requires_grad")]
        if not terms:
            raise ValueError(
                "train-mode forward returned a dict with no tensor loss terms"
            )
        loss = terms[0]
        for t in terms[1:]:
            loss = loss + t
        return loss
    return out  # already a scalar loss tensor


def check_model_contract(
    model: torch.nn.Module,
    task: str,
    **keywords: Unpack[ContractKeywords],
) -> ContractReport:
    """Check the model's behavior at the measurement boundary.

    Returns the :class:`ContractReport`; a failed train or eval pass is recorded in it, not
    raised. The ``keywords`` are :class:`ContractKeywords`.

    ``dims`` is :func:`~tcip_mcp.pipelines.model_build.resolve_contract_dims`' output, stated only
    when a batch is synthesized here; a caller handing over ``sample_batch`` states none. A
    detector whose ``dims`` carry ``attributes`` must also return ``attributes`` per image.

    ``sample_batch`` is an ``(images, targets)`` pair from this run's own dataset, required for a
    task outside ``_SYNTHESIZABLE_TASKS``.
    """
    import torch

    dims = keywords.get("dims")
    sample_batch = keywords.get("sample_batch")
    issues: list[str] = []
    # Never invent a batch shape: a green report earned against a guessed one proves nothing, so
    # a caller with no schema or no dimensions is told to smoke a real batch instead.
    report = ContractReport(
        ok=False,
        issues=issues,
        train_loss=None,
        eval_output_type=None,
        not_smokeable=no_batch_reason(task, dims, sample_batch),
        gradient_magnitudes=None,
    )
    if report["not_smokeable"] is not None:
        return report
    dev = torch.device(keywords.get("device", "cpu"))
    model.to(dev)

    # Train pass: finite loss with a gradient.
    try:
        model.train()
        loss = _forward_loss(
            model, _driving_batch(task, dims, sample_batch, dev)
        )
        if not loss.requires_grad:
            issues.append(
                "train-mode loss does not require grad (no learnable path)"
            )
        lv = float(loss.detach())
        report["train_loss"] = lv
        if not torch.isfinite(torch.tensor(lv)):
            issues.append(f"train-mode loss is not finite ({lv})")
        loss.backward()
        named_grads = [
            (name, p.grad)
            for name, p in model.named_parameters()
            if p.grad is not None
        ]
        grads = [g for _, g in named_grads]
        # float64: an L2 norm summed in the gradient's own float32 dtype can overflow to inf even
        # when every element is finite, which would fail check_json_value once this is persisted.
        gradient_magnitudes = {
            name: float(g.detach().to(torch.float64).norm())
            for name, g in named_grads
        }
        if not grads:
            issues.append(
                "no parameter received a gradient from the train-mode loss"
            )
        elif not all(torch.isfinite(g).all() for g in grads):
            issues.append("some parameter gradients are not finite")
        elif not all(math.isfinite(v) for v in gradient_magnitudes.values()):
            issues.append("some parameter gradient norms are not finite")
        report["gradient_magnitudes"] = gradient_magnitudes
    except Exception as exc:  # noqa: BLE001
        issues.append(f"train-mode forward/backward failed: {exc}")

    # Eval pass: documented output shape.
    try:
        model.eval()
        images, _ = _driving_batch(task, dims, sample_batch, dev)
        with torch.no_grad():
            out = model(images)
        if task in DETECTOR_TASKS:
            report["eval_output_type"] = "list[dict]"
            required = {"boxes", "scores", "labels"}
            if task == "instance_seg":
                # instance_seg trains on real masks (_synth_batch above already supplies
                # target["masks"]); a model whose eval output drops them passes as a detector
                # while the platform's only sanctioned dimensional measurement can never reach it.
                required = required | {"masks"}
            if dims and dims.get("attributes"):
                required = required | {"attributes"}
            if not (
                isinstance(out, list)
                and out
                and isinstance(out[0], dict)
                and required <= set(out[0])
            ):
                issues.append(
                    f"detection eval output is not list[dict] with {sorted(required)}"
                )
        else:
            report["eval_output_type"] = "dict"
            if not isinstance(out, dict):
                issues.append(
                    f"non-detection eval output is not a dict (got {type(out).__name__})"
                )
            elif not any(_contains_tensor(v) for v in out.values()):
                # Which keys a task uses is the agent's to choose, so none are named here. But a
                # dict carrying no tensor, anywhere in a value's own list/tuple/dict nesting, not
                # only at the top level, is not a prediction, and no scorer can consume it.
                issues.append(
                    f"non-detection eval output carries no tensor value (keys: {sorted(out)}); "
                    "nothing downstream can read a measurement from it"
                )
    except Exception as exc:  # noqa: BLE001
        issues.append(f"eval-mode forward failed: {exc}")

    report["ok"] = not issues
    return report


def overfit_check(
    model: torch.nn.Module,
    task: str,
    *,
    seed: int,
    deterministic: bool,
    **keywords: Unpack[OverfitKeywords],
) -> OverfitReport:
    """Drive ``steps`` optimizer updates on one fixed tiny batch; the loss must fall.

    The ``keywords`` are :class:`OverfitKeywords`; ``dims`` is as for
    :func:`check_model_contract`. Seeded at the run's own ``seed`` and
    ``deterministic``, on CPU by default. Returns the :class:`OverfitReport`, ``passed`` iff
    every step completes with no recorded ``issue``, every loss is finite and the final loss is
    strictly below the initial one.

    ``sample_batch`` is an ``(images, targets)`` pair from your own dataset, required for a task
    outside ``_SYNTHESIZABLE_TASKS``. Without it such a task returns ``passed: False`` with the
    reason in ``issue``; it never raises.
    """
    import torch

    from tcip_mcp.pipelines.training.generic_trainer import set_seed

    dims = keywords.get("dims")
    sample_batch = keywords.get("sample_batch")
    reason = no_batch_reason(task, dims, sample_batch)
    if reason is not None:
        return {
            "passed": False,
            "losses": [],
            "initial": None,
            "final": None,
            "issue": reason,
        }
    set_seed(seed, deterministic=deterministic)
    dev = torch.device(keywords.get("device", "cpu"))
    model.to(dev)

    try:
        batch = _driving_batch(task, dims, sample_batch, dev)
    except ValueError as exc:  # report, never raise: this returns a dict
        return {
            "passed": False,
            "losses": [],
            "initial": None,
            "final": None,
            "issue": str(exc),
        }
    losses: list[float] = []
    issue: str | None = None
    try:
        # Built inside the try: a model with no trainable parameter (every parameter frozen, a
        # legitimate stage-0 configuration) makes Adam raise, and that belongs in the report.
        optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=keywords.get("lr", 1e-2),
        )
        model.train()
        for _ in range(max(1, keywords.get("steps", 20))):
            optimizer.zero_grad()
            loss = _forward_loss(model, batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    except Exception as exc:  # noqa: BLE001
        issue = f"optimization step failed: {exc}"

    finite = bool(losses) and all(
        torch.isfinite(torch.tensor(v)) for v in losses
    )
    decreased = bool(losses) and losses[-1] < losses[0]
    if issue is None:
        if not finite:
            issue = "loss became non-finite during overfitting"
        elif not decreased:
            issue = f"loss did not decrease (initial={losses[0]:.4f}, final={losses[-1]:.4f})"
    return {
        "passed": issue is None,
        "losses": losses,
        "initial": losses[0] if losses else None,
        "final": losses[-1] if losses else None,
        "issue": issue,
    }


def render_overfit_report(report: OverfitReport) -> dict[str, JsonValue]:
    """Render an :func:`overfit_check` report in the form a record can carry.

    A diverging model's raw report may hold ``nan`` or ``inf``, values the canonical JSON codec
    refuses outright. ``initial`` and ``final`` are named scalars, so each goes through
    :func:`tcip_store.values.stored_number`, which keeps a null beside a sibling field naming the
    non-finite state; ``losses`` is a positional list with no sibling slot per entry, so each goes
    through :func:`tcip_store.values.finite_or_none`, which drops the state and keeps the null.
    """
    rendered: dict[str, JsonValue] = {
        "passed": report["passed"],
        "issue": report["issue"],
    }
    rendered.update(stored_number("initial", report["initial"]))
    rendered.update(stored_number("final", report["final"]))
    rendered["losses"] = [finite_or_none(v) for v in report["losses"]]
    return rendered
