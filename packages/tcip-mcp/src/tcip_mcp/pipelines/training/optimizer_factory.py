"""Optimizer factory with differential learning rate support.

Registered optimizers: SGD, Adam, AdamW, LAMB, each over a model's named parameter groups
(:func:`param_groups`), one of which is the head group (:func:`head_group`).
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any, cast

import torch
from torch import nn

from tcip_mcp.pipelines.model_build import STATE_DICT_KEY
from tcip_mcp.pipelines.schemas import OptimizerSpec

HEAD_GROUP = "head"
"""The name of the param group ``head_lr`` places, the rate a run's epoch rows report."""
BACKBONE_GROUP = "backbone"
"""The name of the param group a model's ``backbone`` module trains in at ``backbone_lr``."""


def _build_sgd(
    params: list[dict], *, weight_decay: float, momentum: float
) -> torch.optim.SGD:
    return torch.optim.SGD(
        params, momentum=momentum, weight_decay=weight_decay
    )


def _build_adam(
    params: list[dict], *, weight_decay: float
) -> torch.optim.Adam:
    return torch.optim.Adam(params, weight_decay=weight_decay)


def _build_adamw(
    params: list[dict], *, weight_decay: float
) -> torch.optim.AdamW:
    return torch.optim.AdamW(params, weight_decay=weight_decay)


def _build_lamb(
    params: list[dict], *, weight_decay: float
) -> torch.optim.Optimizer:
    """Build a LAMB optimizer, refusing when the optional ``torch_optimizer`` is absent."""
    try:
        from torch_optimizer import Lamb
    except ImportError as exc:
        raise ImportError(
            "optimizer: lamb requires the 'torch_optimizer' package, which is not "
            "installed. Install it with `pip install torch_optimizer`, or choose a "
            "different optimizer (sgd, adam, adamw)."
        ) from exc
    return Lamb(params, weight_decay=weight_decay)


_OPTIMIZER_BUILDERS: dict[str, Callable[..., torch.optim.Optimizer]] = {
    "sgd": _build_sgd,
    "adam": _build_adam,
    "adamw": _build_adamw,
    "lamb": _build_lamb,
}


def build_optimizer(
    spec: OptimizerSpec,
    model: nn.Module,
    *,
    backbone_lr: float,
    head_lr: float,
) -> torch.optim.Optimizer:
    """Build the optimizer ``spec`` names over ``model``'s parameter groups.

    The groups train at the rates :func:`param_groups` places, with the block's weight decay
    and, for ``sgd``, its momentum; each registered builder takes the settings it reads and no
    other. :func:`param_groups`'s refusals propagate.
    """
    from tcip_mcp.pipelines.model_build import resolve_named

    factory = resolve_named(spec.name, _OPTIMIZER_BUILDERS, kind="optimizer")
    settings = {} if spec.momentum is None else {"momentum": spec.momentum}
    return factory(
        param_groups(model, backbone_lr, head_lr),
        weight_decay=spec.weight_decay,
        **settings,
    )


def param_groups(
    model: nn.Module, backbone_lr: float, head_lr: float
) -> list[dict]:
    """Return ``model``'s named parameter groups at ``backbone_lr`` and ``head_lr``.

    The groups are the model's own ``get_param_groups(backbone_lr, head_lr)`` when it defines
    one, else the parameters of its ``backbone`` module at ``backbone_lr`` in
    :data:`BACKBONE_GROUP` and every other at ``head_lr`` in :data:`HEAD_GROUP`. A model exposing
    neither trains every parameter in :data:`HEAD_GROUP` when the two rates are equal, and
    refuses (``ValueError``) naming both when they differ. A model's own groups are refused
    (``ValueError``) unless each states its own ``lr`` and a ``name`` no other group holds, and
    exactly one is named :data:`HEAD_GROUP`.
    """
    if hasattr(model, "get_param_groups"):
        # A bespoke model's own opt-in method, not part of nn.Module's stub.
        groups = cast(Any, model).get_param_groups(backbone_lr, head_lr)
        names = [group.get("name") for group in groups]
        unrated = [name for name, g in zip(names, groups) if "lr" not in g]
        if (
            unrated
            or None in names
            or len(set(names)) != len(names)
            or names.count(HEAD_GROUP) != 1
        ):
            raise ValueError(
                f"{type(model).__name__}.get_param_groups returned groups named {names}, "
                f"{unrated} of them stating no lr: each group states its own lr and a name no "
                f"other group holds, and exactly one is named {HEAD_GROUP!r}, the group "
                "head_lr places."
            )
        return groups
    backbone = getattr(model, "backbone", None)
    if isinstance(backbone, nn.Module):
        held = {id(p) for p in backbone.parameters()}
        return [
            {
                "name": BACKBONE_GROUP,
                "params": list(backbone.parameters()),
                "lr": backbone_lr,
            },
            {
                "name": HEAD_GROUP,
                "params": [p for p in model.parameters() if id(p) not in held],
                "lr": head_lr,
            },
        ]
    if backbone_lr != head_lr:
        raise ValueError(
            f"optimizer.backbone_lr {backbone_lr} and head_lr {head_lr} differ, and this "
            f"{type(model).__name__} exposes neither get_param_groups nor a backbone module "
            "to train at backbone_lr; state the two rates equal, or build a model exposing "
            "one."
        )
    return [
        {"name": HEAD_GROUP, "params": list(model.parameters()), "lr": head_lr}
    ]


def head_group(optimizer: torch.optim.Optimizer) -> dict:
    """Return ``optimizer``'s param group named :data:`HEAD_GROUP`.

    Refuses (``ValueError``) an optimizer holding none or several, which :func:`build_optimizer`
    never builds.
    """
    named = [
        group
        for group in optimizer.param_groups
        if group.get("name") == HEAD_GROUP
    ]
    if len(named) != 1:
        raise ValueError(
            f"the optimizer holds {len(named)} param groups named {HEAD_GROUP!r}, not one; "
            "build it through optimizer_factory.build_optimizer."
        )
    return named[0]


def compute_lr_scale(
    effective_batch: int, reference_batch: int, power: float
) -> float:
    """Return the LR multiplier ``(effective_batch / reference_batch) ** power``.

    Both batches are positive counts.
    """
    return (effective_batch / reference_batch) ** power


OPTIMIZER_STATE_KEY = "optimizer_state_by_name"
GROUPS_KEY = "param_groups"
TRAINING_STATE_KEYS = (STATE_DICT_KEY, OPTIMIZER_STATE_KEY, GROUPS_KEY)
"""The keys of a :func:`capture_training_state` result."""


def _copied(value: object) -> object:
    """Return an independent copy of one optimizer-state value.

    A tensor is cloned, anything else deep copied.
    """
    return (
        value.detach().clone() if torch.is_tensor(value) else deepcopy(value)
    )


def capture_training_state(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> dict:
    """Return the model's and the optimizer's state at this moment, every value a copy.

    The result holds the model's whole ``state_dict`` (weights and buffers) on the CPU under
    ``STATE_DICT_KEY``; the optimizer's per-parameter state by parameter name under
    :data:`OPTIMIZER_STATE_KEY`, each tensor a state entry holds directly moved to the CPU and a
    tensor nested inside a container entry deep copied where it lies; and the optimizer's param
    groups under :data:`GROUPS_KEY`, each once, as
    ``{"members": [parameter names], "settings": {every key but params}}``, its settings (its
    learning rate among them) copied as they are. Refuses (``KeyError``) an optimizer holding a
    parameter that is not one of ``model``'s named parameters.
    """
    name_of: dict[torch.Tensor, str] = {
        p: n for n, p in model.named_parameters()
    }
    return {
        STATE_DICT_KEY: {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        },
        OPTIMIZER_STATE_KEY: {
            name_of[p]: {
                k: _copied(v.cpu() if torch.is_tensor(v) else v)
                for k, v in buf.items()
            }
            for p, buf in optimizer.state.items()
        },
        GROUPS_KEY: [
            {
                "members": [name_of[p] for p in group["params"]],
                "settings": {
                    k: _copied(v) for k, v in group.items() if k != "params"
                },
            }
            for group in optimizer.param_groups
        ],
    }


def captured_sources(
    model: nn.Module, optimizer: torch.optim.Optimizer, groups: list[dict]
) -> list[set[int | None]]:
    """Return, per ``optimizer`` param group, the captured groups its parameters came from.

    Each entry holds the indices into ``groups`` (a capture's :data:`GROUPS_KEY`) of the captured
    groups the group's parameters were members of, ``None`` standing for a parameter no captured
    group held. Membership is read by parameter name; an empty group has no source.
    """
    name_of = {p: n for n, p in model.named_parameters()}
    source_of = {
        name: index
        for index, group in enumerate(groups)
        for name in group["members"]
    }
    return [
        {source_of.get(name_of[p]) for p in group["params"]}
        for group in optimizer.param_groups
    ]


def restore_training_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: dict,
    *,
    group_settings: bool,
) -> None:
    """Load a :func:`capture_training_state` result of ``model`` into it and into ``optimizer``.

    The load goes through the optimizer's own ``load_state_dict``, from fresh copies, so the
    capture stays as it was taken and every tensor lands on its parameter's device and dtype by
    the optimizer's own rule: the model's whole ``state_dict``, then each captured parameter's
    optimizer state, matched by name. A parameter with no captured state (frozen at the capture,
    unfrozen since) starts with none. ``group_settings`` also puts back, into each param group,
    the settings of the one captured group its members came from (:func:`captured_sources`);
    without it the groups keep the settings they were built with. Refuses (``ValueError``) a
    capture with a member ``optimizer`` does not hold, and with ``group_settings`` a non-empty
    group whose members did not all come from one captured group.
    """
    model.load_state_dict(state[STATE_DICT_KEY])
    params = dict(model.named_parameters())
    packed = optimizer.state_dict()
    pid_of = {
        param: pid
        for group, saved in zip(optimizer.param_groups, packed["param_groups"])
        for param, pid in zip(group["params"], saved["params"])
    }
    missing = sorted(
        name
        for group in state[GROUPS_KEY]
        for name in group["members"]
        if params[name] not in pid_of
    )
    if missing:
        raise ValueError(
            f"the optimizer holds no parameter named {missing}, which the captured optimizer "
            "held; a stage's optimizer must hold every parameter the previous stage trained."
        )
    packed["state"] = {
        pid_of[params[name]]: {k: _copied(v) for k, v in buf.items()}
        for name, buf in state[OPTIMIZER_STATE_KEY].items()
    }
    if group_settings:
        for index, (saved, sources) in enumerate(
            zip(
                packed["param_groups"],
                captured_sources(model, optimizer, state[GROUPS_KEY]),
            )
        ):
            if not sources:
                continue
            if len(sources) != 1 or None in sources:
                raise ValueError(
                    f"param group {index}'s members come from captured groups "
                    f"{sorted(map(str, sources))}, not one, so no one setting of it is on "
                    "record."
                )
            saved.update(_copied(state[GROUPS_KEY][sources.pop()]["settings"]))
    optimizer.load_state_dict(packed)
