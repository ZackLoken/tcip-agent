"""Optimizer factory with differential learning rate support.

Registered optimizers: SGD, Adam, AdamW, LAMB.
Supports `model.get_param_groups(backbone_lr, head_lr)` for
differential LR between backbone and heads.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, cast

import torch
from torch import nn


def _build_sgd(params, lr: float = 1e-3, momentum: float = 0.9, weight_decay: float = 1e-4, **kw):
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)


def _build_adam(params, lr: float = 1e-3, weight_decay: float = 0, **kw):
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def _build_adamw(params, lr: float = 1e-3, weight_decay: float = 1e-2, **kw):
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _build_lamb(params, lr: float = 1e-3, weight_decay: float = 1e-2, **kw):
    """LAMB optimizer, requires the optional ``torch_optimizer`` package."""
    try:
        from torch_optimizer import Lamb
    except ImportError as exc:
        raise ImportError(
            "optimizer: lamb requires the 'torch_optimizer' package, which is not "
            "installed. Install it with `pip install torch_optimizer`, or choose a "
            "different optimizer (sgd, adam, adamw)."
        ) from exc
    return Lamb(params, lr=lr, weight_decay=weight_decay)


# The four builders don't share one signature (sgd alone takes momentum), so the dict's value
# type is stated explicitly rather than left to a join across mismatched signatures.
_OPTIMIZER_BUILDERS: dict[str, Callable[..., torch.optim.Optimizer]] = {
    "sgd": _build_sgd,
    "adam": _build_adam,
    "adamw": _build_adamw,
    "lamb": _build_lamb,
}


def build_optimizer(
    name: str,
    model: nn.Module,
    backbone_lr: float = 1e-4,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Build an optimizer with differential LR.

    If model has `get_param_groups(backbone_lr, head_lr)`, uses those
    param groups. Otherwise gives all params the head_lr.
    """
    if hasattr(model, "get_param_groups"):
        # A bespoke model's own opt-in method, not part of nn.Module's stub.
        param_groups = cast(Any, model).get_param_groups(backbone_lr, head_lr)
    else:
        param_groups = [{"params": model.parameters(), "lr": head_lr}]

    try:
        factory = _OPTIMIZER_BUILDERS[name]
    except KeyError:
        raise KeyError(f"Unknown optimizer '{name}'. Available: {sorted(_OPTIMIZER_BUILDERS)}") from None
    return factory(param_groups, lr=head_lr, weight_decay=weight_decay)


# ====================================================================
# Progressive-unfreezing helpers: effective-batch LR scaling +
# name-keyed optimizer-state handoff across stages.
# ====================================================================

def compute_lr_scale(effective_batch: int, reference_batch: int, power: float) -> float:
    """Scale LR by ``(effective_batch / reference_batch) ** power``.

    ``power=0.5`` gives sqrt scaling. Returns ``1.0`` when ``reference_batch <= 0``.
    """
    if reference_batch <= 0:
        return 1.0
    return (effective_batch / reference_batch) ** power


def _flatten_param_ids(optimizer: torch.optim.Optimizer) -> dict[int, torch.nn.Parameter]:
    """Map the integer state-dict ``pid`` to its live Parameter.

    ``optimizer.state_dict()['state']`` is keyed by an integer assigned in the
    flattened ``param_groups`` iteration order; this reproduces that order so a
    snapshot can be re-keyed by parameter *name* (stable across stages) rather
    than by ``pid`` (which shifts as the trainable set grows).
    """
    pid_to_param: dict[int, torch.nn.Parameter] = {}
    pid = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            pid_to_param[pid] = p
            pid += 1
    return pid_to_param


def snapshot_optimizer_state(optimizer: torch.optim.Optimizer, model: nn.Module) -> dict:
    """Snapshot optimizer momentum buffers keyed by parameter name (CPU clones).

    Returns ``{'state_by_name': {param_name: {buf_key: cpu_tensor|value}},
    'end_lrs': [lr, ...]}``. Empty dict when ``optimizer`` or ``model`` is None.
    """
    if optimizer is None or model is None:
        return {}
    sd = optimizer.state_dict()
    pid_to_param = _flatten_param_ids(optimizer)
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    state_by_name: dict[str, dict] = {}
    for pid, buf in sd["state"].items():
        p = pid_to_param.get(pid)
        name = id_to_name.get(id(p)) if p is not None else None
        if name is None:
            continue
        state_by_name[name] = {
            k: (v.detach().clone().cpu() if torch.is_tensor(v) else deepcopy(v))
            for k, v in buf.items()
        }
    return {
        "state_by_name": state_by_name,
        "end_lrs": [float(g["lr"]) for g in optimizer.param_groups],
    }


def _buffer_shapes_ok(buf: dict, param: torch.nn.Parameter) -> bool:
    """A buffer is restorable if every non-scalar tensor matches the param shape."""
    for v in buf.values():
        if torch.is_tensor(v) and v.numel() > 1 and tuple(v.shape) != tuple(param.shape):
            return False
    return True


def restore_optimizer_state(
    optimizer: torch.optim.Optimizer, model: nn.Module, snapshot: dict
) -> int:
    """Inject snapshot buffers into a freshly built optimizer, keyed by name.

    Restores momentum for name-overlapping, shape-compatible params; newly
    unfrozen params keep their empty state. Param-group LRs are left untouched
    (the new stage's target LRs stand). Returns the number of params restored.
    """
    if not snapshot or optimizer is None or model is None:
        return 0
    state_by_name = snapshot.get("state_by_name") or {}
    if not state_by_name:
        return 0
    sd = optimizer.state_dict()  # fresh optimizer -> sd['state'] starts empty
    pid_to_param = _flatten_param_ids(optimizer)
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    restored = 0
    for pid, p in pid_to_param.items():
        name = id_to_name.get(id(p))
        buf = state_by_name.get(name) if name else None
        if buf is None or not _buffer_shapes_ok(buf, p):
            continue
        sd["state"][pid] = {
            k: (v.to(device=p.device, dtype=p.dtype)
                if (torch.is_tensor(v) and v.numel() > 1) else v)
            for k, v in buf.items()
        }
        restored += 1
    if restored:
        optimizer.load_state_dict(sd)
    return restored
