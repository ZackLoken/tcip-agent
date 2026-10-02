"""Annotation-proposal engines: a method-neutral seam for auto-labeling.

An auto-labeling engine turns an image into candidate shapes a human then reviews. Register one
(``register_proposal_engine``) or bring it by dotted ``module:factory`` path (Grounding DINO,
open-vocab, a bespoke proposer); how well each engine's high-conf proposals survive breeder review
compares them.

An engine implements :class:`Proposer`: ``propose`` for candidates over an image. Candidates use a
neutral schema (``candidate_id`` / ``bbox`` / ``area`` / ``rings`` / ``score`` / ``engine`` /
``engine_meta``); engine-specific signals live under ``engine_meta``.

``rings`` is a candidate's geometry as ``Polygon.rings``: one closed contour per connected region
of the proposed mask; the staging path keeps all of them.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Proposer(Protocol):
    """A source of annotation proposals for a human to review (see module docstring)."""

    def propose(self, image_path: str, **params: Any) -> list[dict]:
        """Candidate shapes over the image in the neutral schema, for review."""
        ...


_ENGINES: dict[str, Proposer] = {}


def register_proposal_engine(name: str, engine: Proposer) -> None:
    """Register an auto-labeling engine under ``name`` so ``engine=<name>`` resolves to it."""
    _ENGINES[name] = engine


def resolve_proposer(engine: str) -> Proposer:
    """The engine registered under ``engine``, else the one a dotted ``module:factory`` names,
    instantiated when the imported target is callable
    (:func:`~tcip_mcp.pipelines.model_build.resolve_named`, whose refusals propagate)."""
    from tcip_mcp.pipelines.model_build import resolve_named

    target = resolve_named(engine, _ENGINES, kind="proposal engine",
                           register="register_proposal_engine")
    return target() if engine not in _ENGINES and callable(target) else target
