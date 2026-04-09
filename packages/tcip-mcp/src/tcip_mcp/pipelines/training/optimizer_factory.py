"""Optimizer factory with differential learning rate support.

Registered optimizers: SGD, Adam, AdamW, LAMB.
Supports `model.get_param_groups(backbone_lr, head_lr)` for
differential LR between backbone and heads.
"""

from __future__ import annotations

import torch
from torch import nn

from tcip_mcp.pipelines.registry import OPTIMIZERS


def _build_sgd(params, lr: float = 1e-3, momentum: float = 0.9, weight_decay: float = 1e-4, **kw):
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)


def _build_adam(params, lr: float = 1e-3, weight_decay: float = 0, **kw):
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def _build_adamw(params, lr: float = 1e-3, weight_decay: float = 1e-2, **kw):
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _build_lamb(params, lr: float = 1e-3, weight_decay: float = 1e-2, **kw):
    """LAMB optimizer — falls back to AdamW if torch_optimizer not installed."""
    try:
        from torch_optimizer import Lamb
        return Lamb(params, lr=lr, weight_decay=weight_decay)
    except ImportError:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


OPTIMIZERS.register_factory("sgd", _build_sgd, category="optimizer", metadata={
    "description": "SGD with momentum", "use_when": "Large datasets, well-tuned LR"})
OPTIMIZERS.register_factory("adam", _build_adam, category="optimizer", metadata={
    "description": "Adam", "use_when": "Quick experiments, default choice"})
OPTIMIZERS.register_factory("adamw", _build_adamw, category="optimizer", metadata={
    "description": "AdamW with decoupled weight decay", "use_when": "Default for transformers/timm"})
OPTIMIZERS.register_factory("lamb", _build_lamb, category="optimizer", metadata={
    "description": "LAMB (large-batch optimizer)", "use_when": "Large batch training"})


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
        param_groups = model.get_param_groups(backbone_lr, head_lr)
    else:
        param_groups = [{"params": model.parameters(), "lr": head_lr}]

    factory = OPTIMIZERS.get(name)
    return factory(param_groups, lr=head_lr, weight_decay=weight_decay)
