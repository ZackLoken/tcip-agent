"""A dotted-name proposal engine fixture.

:data:`STUB_ENGINE` names it as a ``module:factory`` engine; the factory returns a
:class:`StubProposer` of one candidate, and a test that installs its own candidates builds the
class with them. The module imports no torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tcip_mcp.pipelines.proposal import Proposal

STUB_ENGINE = "tests.proposal_stub:factory"
"""The dotted engine name of :func:`factory`, an engine proposing one candidate box."""


class StubProposer:
    """A proposer handing back the candidates it is built with, whatever the image."""

    def __init__(self, candidates: list[dict]) -> None:
        self._candidates = candidates

    def propose(self, image_path: str, **params: object) -> Proposal:
        from tcip_mcp.pipelines.execution import NoPass
        from tcip_mcp.pipelines.proposal import Proposal

        return Proposal(self._candidates, NoPass())


def factory() -> StubProposer:
    """Return a :class:`StubProposer` of one candidate box, in the neutral proposal schema."""
    return StubProposer(
        [
            {
                "candidate_id": 0,
                "bbox": [5.0, 10.0, 25.0, 30.0],
                "area": 400,
                "score": 0.9,
                "engine_meta": {},
                "rings": [
                    [(5.0, 10.0), (25.0, 10.0), (25.0, 30.0), (5.0, 30.0)]
                ],
            }
        ]
    )
