"""Active-learning helpers: scorer lookup by method name and the composed-detector precondition."""

from __future__ import annotations


def build_scorer(method: str, task: str):
    """Return the active-learning scorer for a method name.

    Resolves through the scorer registry (``scorer.resolve_scorer``): the built-in 'uncertainty' |
    'diversity' | 'combined', any acquisition function registered with ``register_scorer``, or a
    dotted ``module:factory`` you wrote. An unresolvable name raises ``ValueError`` (including a
    dotted name that fails to import). ``task`` is threaded to the logit-reading scorers.
    """
    from tcip_mcp.pipelines.active_learning.scorer import resolve_scorer

    return resolve_scorer(method, task)


def require_composed_detector(predictor, *, purpose: str = "uncertainty scoring") -> str | None:
    """Return an error string if ``predictor`` isn't a bespoke tcip nn.Module detector, else None.

    The uncertainty/diversity scorers read logits from ``predictor.model`` as an ``nn.Module``;
    a non-composed predictor kind's ``.model`` may not be one, so scoring it would be invalid.
    """
    from tcip_mcp.pipelines.inference.predictor import KIND_TCIP_MODULE

    kind = getattr(predictor, "kind", None)
    if kind != KIND_TCIP_MODULE:
        return (f"{purpose} needs a bespoke tcip detector, not a '{kind or 'unknown'}' "
                f"model (its .model is not an nn.Module the scorer can read logits from)")
    return None
