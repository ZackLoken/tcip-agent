"""A dotted-name proposal engine fixture: importable by ``resolve_proposer`` as a bring-your-own
engine (``tests.proposal_stub:factory``), so a test can exercise the real ``module:factory`` import
path (``pipelines.proposal.resolve_proposer``) rather than a built-in name or a monkeypatch. Not a
``test_*`` module: it is imported by its dotted name, mirroring ``tests/bespoke_models.py`` for
``model_source``/``training_source``. Torch-free, since ``propose_annotations`` needs none of it.
"""

from __future__ import annotations


class StubProposer:
    """Hands back one fixed candidate box, in the neutral proposal schema."""

    def propose(self, image_path: str, **params: object) -> list[dict]:
        return [{
            "candidate_id": 0,
            "bbox": [5.0, 10.0, 25.0, 30.0],
            "area": 400,
            "score": 0.9,
            "engine": "stub",
            "engine_meta": {},
            "rings": [[(5.0, 10.0), (25.0, 10.0), (25.0, 30.0), (5.0, 30.0)]],
        }]


def factory() -> StubProposer:
    return StubProposer()
