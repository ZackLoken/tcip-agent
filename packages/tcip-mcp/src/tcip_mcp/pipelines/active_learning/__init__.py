"""Active learning pipeline: scorer and selector modules."""

DEFAULT_SCORER = "combined"
"""The scorer a review queue ranks with when none is named: the one ``scorer.SCORER_REGISTRY``
registers combining every signal."""

DEFAULT_REVIEW_BUDGET = 50
"""How many images a review queue returns when no budget is named. Provisional: a working queue
length, derived from no data."""
