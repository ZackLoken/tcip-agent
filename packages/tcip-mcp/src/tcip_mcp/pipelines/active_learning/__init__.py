"""Active learning pipeline: scorer and selector modules.

The package also holds the scorer block a review queue states.
"""

from pydantic import BaseModel, ConfigDict


class ScorerSpec(BaseModel):
    """A review queue's scorer: the factory ``name`` names (``scorer.resolve_scorer``).

    Its own values are kept as stated and handed to the factory beside the checkpoint's task.
    """

    model_config = ConfigDict(extra="allow", frozen=True)
    name: str
