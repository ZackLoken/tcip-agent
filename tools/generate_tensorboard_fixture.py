"""Write the TensorBoard serving answers of two runs as a fixture file, produced by the platform.

In a scratch session (``tools/fixture_session.py``), a project created through
``tests._web_fixtures.named_project`` and opened through the picker's route holds the runs
:data:`RUNS` names, each opened through the launcher's own producer and writer
(``tests._verified_checkpoint_fixtures.opened_run``) with one scalar a ``SummaryWriter`` wrote on
its board. ``POST /api/training/runs/<id>/tensorboard`` starts a TensorBoard over each board and
answers it; the answers are written, :func:`normalized`, to
``frontend/src/test/tensorboardServings.json``, keyed by run id. Each TensorBoard is stopped
before the session ends.

    python -m tools.generate_tensorboard_fixture
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests import FRONTEND_SRC
from tools.fixture_session import opened, scratch_session, write_fixture

if TYPE_CHECKING:
    from pydantic import JsonValue

GENERATED_PATH = FRONTEND_SRC / "test" / "tensorboardServings.json"

RUNS = ("train-x", "train-y")
"""The ids of the runs whose TensorBoards are served, two so a reader can tell their answers
apart."""


def normalized(served: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Return a serving answer with the port its TensorBoard took, which varies by run, as ``0``."""
    return {**served, "port": 0}


def produce() -> dict[str, JsonValue]:
    """Open the runs, launch a TensorBoard over each through the route; return the answers.

    The answers are :func:`normalized` and keyed by run id. A launch the route refuses raises,
    and so does a TensorBoard whose stop ``stop_tensorboard`` answers unconfirmed, naming its
    board and pid, whether or not the production raised first; a production's own exception is
    then the refusal's context.
    """
    from torch.utils.tensorboard import SummaryWriter

    from tcip_mcp.experiments import board_of
    from tcip_mcp.pipelines.training.tensorboard_manager import (
        stop_tensorboard,
    )
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        opened_run,
    )
    from tests._web_fixtures import named_project

    with scratch_session() as (workspace, client):
        root = workspace / "boards"
        opened(client, named_project(root, "Boards"))
        served: dict[str, JsonValue] = {}
        boards = []
        try:
            for experiment_id in RUNS:
                config = detection_config(
                    workspace / f"{experiment_id}-data", project=root
                )
                board = board_of(
                    opened_run(root, config, experiment_id=experiment_id)
                )
                with SummaryWriter(str(board)) as writer:
                    writer.add_scalar("loss", 0.5, 1)
                boards.append(board)
                answer = client.post(
                    f"/api/training/runs/{experiment_id}/tensorboard", json={}
                )
                answer.raise_for_status()
                served[experiment_id] = normalized(answer.json())
        finally:
            stops = {board: stop_tensorboard(str(board)) for board in boards}
            unconfirmed = {
                str(board): stop["pid"]
                for board, stop in stops.items()
                if stop["status"] == "kill_unconfirmed"
            }
            if unconfirmed:
                raise SystemExit(
                    f"TensorBoards left with their kills unconfirmed: {unconfirmed}"
                )
        return served


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))
