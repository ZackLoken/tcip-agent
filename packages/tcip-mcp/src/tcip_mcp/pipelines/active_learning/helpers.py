"""Shared active-learning helpers used by the AL MCP tools.

``prioritize_review_queue`` builds a method→scorer mapping and enforces a composed-detector
precondition; factoring both here keeps a future second logit-reading entry point from drifting.
(Its ``confidence_triage`` strategy deliberately does not use these — it partitions by prediction
confidence via ``predict_batch``, which is kind-agnostic and reads no logits.)
"""

from __future__ import annotations


def build_scorer(method: str, task: str):
    """Return the active-learning scorer for a method name.

    ``method`` is 'uncertainty' | 'diversity' | 'combined' (anything else → combined, matching
    the prior inline behaviour). ``task`` is threaded to the logit-reading scorers.
    """
    from tcip_mcp.pipelines.active_learning.scorer import (
        CombinedScorer,
        DiversityScorer,
        UncertaintyScorer,
    )

    if method == "uncertainty":
        return UncertaintyScorer(task=task)
    if method == "diversity":
        return DiversityScorer()
    return CombinedScorer(task=task)


def require_composed_detector(predictor, *, purpose: str = "uncertainty scoring") -> str | None:
    """Return an error string if ``predictor`` isn't a bespoke tcip nn.Module detector, else None.

    The uncertainty/diversity scorers read logits from ``predictor.model`` as an ``nn.Module``;
    a YOLO/ultralytics predictor's ``.model`` is not one, so scoring it is invalid.
    """
    from tcip_mcp.pipelines.inference.predictor import KIND_TCIP_MODULE

    kind = getattr(predictor, "kind", None)
    if kind != KIND_TCIP_MODULE:
        return (f"{purpose} needs a bespoke tcip detector, not a '{kind or 'unknown'}' "
                f"model (its .model is not an nn.Module the scorer can read logits from)")
    return None
