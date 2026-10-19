"""Annotation-proposal engines: a method-neutral seam for auto-labeling.

An auto-labeling engine turns an image into candidate shapes. Register one
(``register_proposal_engine``) or bring it by dotted ``module:factory`` path (Grounding DINO,
open-vocab, a bespoke proposer).

An engine implements :class:`Proposer`: ``propose`` for candidates over an image. Candidates use a
neutral schema (``candidate_id`` / ``bbox`` / ``area`` / ``rings`` / ``score`` / ``engine`` /
``engine_meta``); engine-specific signals live under ``engine_meta``.

``rings`` is a candidate's geometry as ``Polygon.rings``: one closed contour per connected region
of the proposed mask.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import JsonValue


@runtime_checkable
class Proposer(Protocol):
    """A source of annotation proposals for a human to review (see module docstring)."""

    def propose(self, image_path: str, **params: JsonValue) -> list[dict]:
        """Return candidate shapes over the image in the neutral schema."""
        ...


_ENGINES: dict[str, Callable[[], Proposer]] = {}
"""Each registered engine, as the factory that answers it."""


def register_proposal_engine(name: str, engine: Proposer) -> None:
    """Register an auto-labeling engine under ``name`` so ``engine=<name>`` resolves to it."""
    _ENGINES[name] = lambda: engine


def resolve_proposer(engine: str) -> Proposer:
    """Return the engine registered under ``engine``, else the one a dotted factory builds.

    The engine is resolved by :func:`~tcip_mcp.pipelines.model_build.resolve_named`, whose
    refusals propagate, and a dotted ``module:factory`` target is called with no argument; one
    that builds anything without ``propose`` refuses (``TypeError``) naming ``engine``.
    """
    from tcip_mcp.pipelines.model_build import resolve_named

    built = resolve_named(
        engine,
        _ENGINES,
        kind="proposal engine",
        register="register_proposal_engine",
    )()
    if not isinstance(built, Proposer):
        raise TypeError(
            f"proposal engine {engine!r} built a {type(built).__name__}, which has no "
            "propose: a dotted factory returns the Proposer it builds."
        )
    return built
