"""A dotted-name proposal engine fixture.

``tests.proposal_stub:factory`` names it as a ``module:factory`` engine; the factory returns a
:class:`StubProposer`. The module imports no torch.
"""

from __future__ import annotations


class StubProposer:
    """A proposer handing back one fixed candidate box, in the neutral proposal schema."""

    def propose(self, image_path: str, **params: object) -> list[dict]:
        return [
            {
                "candidate_id": 0,
                "bbox": [5.0, 10.0, 25.0, 30.0],
                "area": 400,
                "score": 0.9,
                "engine": "stub",
                "engine_meta": {},
                "rings": [
                    [(5.0, 10.0), (25.0, 10.0), (25.0, 30.0), (5.0, 30.0)]
                ],
            }
        ]


def factory() -> StubProposer:
    return StubProposer()
