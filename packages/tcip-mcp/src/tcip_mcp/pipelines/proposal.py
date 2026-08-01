"""Annotation-proposal engines: a method-neutral seam for auto-labeling.

An auto-labeling *engine* turns an image into candidate shapes a human then reviews. This seam is
deliberately as open as ``model_source``: name a built-in engine ('sam') or bring your own by dotted
``module:factory`` path, so the agent can wire, trial, and compare techniques (Grounding DINO,
open-vocab, a bespoke proposer it writes) and deduce which serves a task best by how well each
engine's high-conf proposals survive breeder review. No engine is privileged; SAM is the one that
ships as a runnable reference.

An engine implements :class:`Proposer`: ``propose`` for whole-image candidates (auto-labeling) and
``segment`` for a prompted single mask; an engine may supply either, and the dispatch checks for the
method it needs. Candidates use a neutral schema (``candidate_id`` / ``bbox`` / ``area`` /
``rings`` / ``score`` / ``engine`` / ``engine_meta``); engine-specific signals (SAM's stability
and predicted-IoU scores) live under ``engine_meta`` so the shared review/staging path stays
method-agnostic.

``rings`` is a candidate's geometry as ``Polygon.rings``: one closed contour per connected region of
the proposed mask. An engine that only ever finds whole objects yields one ring per candidate; one
that proposes an occlusion-split object yields several, and the staging path keeps all of them.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Proposer(Protocol):
    """A source of annotation proposals for a human to review (see module docstring)."""

    def propose(self, image_path: str, **params: Any) -> list[dict]:
        """Whole-image candidate shapes in the neutral schema, for review."""
        ...

    def segment(
        self,
        image_path: str,
        *,
        points: list[dict] | None = None,
        box: dict | None = None,
        **params: Any,
    ) -> list[list[tuple[float, float]]]:
        """One mask's polygon rings (pixel coords) from an interactive point/box prompt."""
        ...


def neutral_candidate(raw: dict, *, engine: str, score_key: str, meta_keys: tuple[str, ...]) -> dict:
    """Map an engine's raw candidate dict onto the neutral schema, engine extras under engine_meta."""
    return {
        "candidate_id": raw["candidate_id"],
        "bbox": raw["bbox"],
        "area": raw["area"],
        "rings": raw["rings"],
        "score": float(raw.get(score_key, 0.0)),
        "engine": engine,
        "engine_meta": {k: raw[k] for k in meta_keys if k in raw},
    }


class SamProposer:
    """The built-in SAM2 reference engine: auto-mask proposal + prompted segmentation.

    A thin adapter over ``tcip_annotation.sam_wrapper``; SAM's mask-quality signals (stability score,
    predicted IoU) are carried through under ``engine_meta`` and its predicted IoU is the neutral
    ``score``. Engine knobs (``model_type``, ``points_per_side``, thresholds) arrive as ``params`` so
    they stay off the public tool signature.
    """

    name = "sam"

    def propose(self, image_path: str, **params: Any) -> list[dict]:
        from tcip_annotation.sam_wrapper import auto_mask

        raw = auto_mask(image_path, **params)
        return [
            neutral_candidate(c, engine=self.name, score_key="predicted_iou",
                              meta_keys=("stability_score", "predicted_iou"))
            for c in raw
        ]

    def segment(
        self,
        image_path: str,
        *,
        points: list[dict] | None = None,
        box: dict | None = None,
        **params: Any,
    ) -> list[list[tuple[float, float]]]:
        from tcip_annotation.sam_wrapper import (
            predict_from_box,
            predict_from_point,
            predict_from_points,
        )

        if box is not None:
            return predict_from_box(image_path, box["x1"], box["y1"], box["x2"], box["y2"], **params)
        if points is not None and len(points) == 1:
            p = points[0]
            return predict_from_point(image_path, p["x"], p["y"], label=p.get("label", 1), **params)
        pts = [(p["x"], p["y"]) for p in (points or [])]
        lbls = [p.get("label", 1) for p in (points or [])]
        return predict_from_points(image_path, pts, lbls, **params)


_ENGINES: dict[str, Proposer] = {}
_builtins_loaded = False


def _ensure_builtins() -> None:
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True
    register_proposal_engine("sam", SamProposer())


def register_proposal_engine(name: str, engine: Proposer) -> None:
    """Register an auto-labeling engine under ``name`` so ``engine=<name>`` resolves to it."""
    _ENGINES[name] = engine


def available_engines() -> list[str]:
    """Registered built-in engine names (a dotted ``module:factory`` also resolves)."""
    _ensure_builtins()
    return sorted(_ENGINES)


def resolve_proposer(engine: str) -> Proposer:
    """Resolve an engine: a registered built-in name, else a dotted ``module:factory`` the agent brings.

    Mirrors ``model_source``: a bring-your-own engine is imported (never ``exec``'d) and, if the
    imported target is callable (a class or factory), instantiated. A dotted name that fails to
    import raises ``ValueError`` like an unknown one, so one ``except`` covers both.
    """
    _ensure_builtins()
    if engine in _ENGINES:
        return _ENGINES[engine]
    if ":" in engine or "." in engine:
        from tcip_mcp.pipelines.model_build import _import_dotted

        try:
            target = _import_dotted(engine)
        except Exception as exc:  # noqa: BLE001 (any import failure is an unresolvable name)
            raise ValueError(f"Could not import proposal engine {engine!r}: {exc}") from exc
        return target() if callable(target) else target
    raise ValueError(
        f"Unknown proposal engine {engine!r}. Use a built-in ({sorted(_ENGINES)}), register one with "
        f"register_proposal_engine, or pass a dotted 'module:factory' you wrote."
    )
