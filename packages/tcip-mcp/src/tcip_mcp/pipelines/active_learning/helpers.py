"""Active-learning helpers: scorer lookup by method name."""

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
