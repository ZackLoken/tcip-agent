"""The one model-side contract: the measurement boundary — as a behavioral check, not a mold.

TCIP Agent owns architecture and the training loop; the platform does not dictate a model's
internals. The single thing a model must honor is that it *trains* and its *inference output is
something the library scorers can measure*:

* ``TCIPModel`` — a ``runtime_checkable`` Protocol naming the minimal surface (any ``nn.Module``
  satisfies it). It is a duck-type marker, not an architecture requirement — no ``freeze_backbone``
  / ``head0_*`` here (those are optional conveniences the default trainer uses if present).
* ``check_model_contract(model, task)`` — a behavioral smoke: a train-mode forward yields a finite
  loss with a gradient, and an eval-mode forward yields the documented shape for the task
  (``list[dict]`` per image for detection/instance_seg, a ``dict`` of tensors otherwise).
* ``overfit_check(model, task, steps)`` — the cheap proof a from-scratch model actually learns:
  a few optimizer steps on one fixed tiny batch (seeded, CPU) must drive the loss down.

All torch use is lazy (inside the functions) so importing this module stays cheap.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

_DETECTION_TASKS = {"detection", "instance_seg"}


@runtime_checkable
class TCIPModel(Protocol):
    """Minimal surface a trainable, measurable model exposes (an ``nn.Module`` already does).

    Train-mode ``forward(images, targets)`` returns a loss dict (or scalar loss); eval-mode
    ``forward(images)`` returns predictions the library scorers consume. This is a marker, not
    an architecture contract.
    """

    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    def parameters(self, recurse: bool = True) -> Any: ...
    def train(self, mode: bool = True) -> Any: ...
    def eval(self) -> Any: ...


def _synth_batch(task: str, *, in_chans: int, num_classes: int, img_size: int, device: Any):
    """A minimal batch in the exact shape ``generic_trainer`` feeds ``model.forward`` for ``task``."""
    import torch

    if task in _DETECTION_TASKS:
        img = torch.rand(in_chans, img_size, img_size, device=device)
        box = [img_size * 0.2, img_size * 0.2, img_size * 0.7, img_size * 0.7]
        target = {
            "boxes": torch.tensor([box], device=device),
            "labels": torch.ones((1,), dtype=torch.long, device=device),  # 1-indexed foreground
        }
        if task == "instance_seg":
            mask = torch.zeros((1, img_size, img_size), dtype=torch.uint8, device=device)
            lo, hi = int(img_size * 0.2), int(img_size * 0.7)
            mask[0, lo:hi, lo:hi] = 1
            target["masks"] = mask
        return [img], [target]

    images = torch.rand(2, in_chans, img_size, img_size, device=device)
    if task == "ordinal":
        targets = {"ranks": torch.randint(0, max(num_classes, 2), (2,), device=device)}
    elif task == "regression":
        targets = {"values": torch.rand(2, device=device)}
    elif task == "semantic_seg":
        targets = {"masks": torch.randint(0, max(num_classes, 2), (2, img_size, img_size), device=device)}
    else:  # classification (and any dict-target head)
        targets = {"labels": torch.randint(0, max(num_classes, 2), (2,), device=device)}
    return images, targets


def _forward_loss(model: Any, images: Any, targets: Any):
    """Train-mode forward → a single scalar loss tensor (sums a returned loss dict)."""
    out = model(images, targets)
    if isinstance(out, dict):
        terms = [v for v in out.values() if hasattr(v, "requires_grad")]
        if not terms:
            raise ValueError("train-mode forward returned a dict with no tensor loss terms")
        loss = terms[0]
        for t in terms[1:]:
            loss = loss + t
        return loss
    return out  # already a scalar loss tensor


def check_model_contract(
    model: TCIPModel, task: str, *, in_chans: int = 3, num_classes: int = 1,
    img_size: int = 64, device: str = "cpu",
) -> dict:
    """Behavioral smoke test of the measurement boundary. Returns a report; never raises.

    ``{"ok": bool, "issues": [...], "train_loss": float|None, "eval_output_type": str|None}``.
    """
    import torch

    issues: list[str] = []
    report: dict[str, Any] = {"ok": False, "issues": issues, "train_loss": None, "eval_output_type": None}
    dev = torch.device(device)
    try:
        model.to(dev)
    except Exception:  # noqa: BLE001 — a model without .to still may be usable on cpu
        pass

    # Train pass — finite loss with a gradient.
    try:
        model.train()
        images, targets = _synth_batch(task, in_chans=in_chans, num_classes=num_classes,
                                        img_size=img_size, device=dev)
        loss = _forward_loss(model, images, targets)
        if not (hasattr(loss, "requires_grad") and loss.requires_grad):
            issues.append("train-mode loss does not require grad (no learnable path)")
        lv = float(loss.detach())
        report["train_loss"] = lv
        if not torch.isfinite(torch.tensor(lv)):
            issues.append(f"train-mode loss is not finite ({lv})")
        loss.backward()
        grads = [p.grad for p in model.parameters() if getattr(p, "grad", None) is not None]
        if not grads:
            issues.append("no parameter received a gradient from the train-mode loss")
        elif not all(torch.isfinite(g).all() for g in grads):
            issues.append("some parameter gradients are not finite")
    except Exception as exc:  # noqa: BLE001
        issues.append(f"train-mode forward/backward failed: {exc}")

    # Eval pass — documented output shape.
    try:
        model.eval()
        images, _ = _synth_batch(task, in_chans=in_chans, num_classes=num_classes,
                                 img_size=img_size, device=dev)
        with torch.no_grad():
            out = model(images)
        if task in _DETECTION_TASKS:
            report["eval_output_type"] = "list[dict]"
            if not (isinstance(out, list) and out and isinstance(out[0], dict)
                    and {"boxes", "scores", "labels"} <= set(out[0])):
                issues.append("detection eval output is not list[dict] with boxes/scores/labels")
        else:
            report["eval_output_type"] = "dict"
            if not isinstance(out, dict):
                issues.append(f"non-detection eval output is not a dict (got {type(out).__name__})")
    except Exception as exc:  # noqa: BLE001
        issues.append(f"eval-mode forward failed: {exc}")

    report["ok"] = not issues
    return report


def overfit_check(
    model: TCIPModel, task: str, *, steps: int = 20, in_chans: int = 3, num_classes: int = 1,
    img_size: int = 64, seed: int = 0, lr: float = 1e-2, device: str = "cpu",
) -> dict:
    """Drive ``steps`` optimizer updates on ONE fixed tiny batch; the loss must fall.

    Seeded + CPU by default so the result is reproducible. Returns
    ``{"passed": bool, "losses": [...], "initial": float, "final": float, "issue": str|None}``.
    ``passed`` iff every loss is finite and the final loss is strictly below the initial one —
    the minimal evidence a bespoke model with a real learnable path actually optimizes.
    """
    import torch

    from tcip_mcp.pipelines.training.generic_trainer import set_seed

    set_seed(seed)
    dev = torch.device(device)
    try:
        model.to(dev)
    except Exception:  # noqa: BLE001
        pass

    images, targets = _synth_batch(task, in_chans=in_chans, num_classes=num_classes,
                                   img_size=img_size, device=dev)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)

    losses: list[float] = []
    issue: str | None = None
    try:
        model.train()
        for _ in range(max(1, steps)):
            optimizer.zero_grad()
            loss = _forward_loss(model, images, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    except Exception as exc:  # noqa: BLE001
        issue = f"optimization step failed: {exc}"

    finite = bool(losses) and all(torch.isfinite(torch.tensor(v)) for v in losses)
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
