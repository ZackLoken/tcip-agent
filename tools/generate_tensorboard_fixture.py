"""Write the TensorBoard launch answers of four runs as a fixture file, produced by the platform.

In a scratch session (``tools/fixture_session.py``), a project created through
``tests._web_fixtures.named_project`` and opened through the picker's route holds the runs
:data:`RUNS` and :data:`FAULTS` name, each opened through the launcher's own producer and writer
(``tests._verified_checkpoint_fixtures.opened_run``) with one scalar a ``SummaryWriter`` wrote on
its board. ``POST /api/training/runs/<id>/tensorboard`` starts a TensorBoard over each board and
answers it, a fault's run under its fault so the launch answers its failure; the answers are
written, normalized by :data:`STAMPED` (``fixture_session.normalized``), to
:data:`GENERATED_PATH`, keyed by run id. Each TensorBoard is stopped before the session ends.

    python -m tools.generate_tensorboard_fixture
"""

from __future__ import annotations

import socket
import subprocess
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING
from unittest import mock

from tcip_mcp.web_client import free_port
from tests import FRONTEND_SRC
from tests._tensorboard_parent import DyingChild
from tools.fixture_session import (
    answered,
    normalized,
    opened,
    scratch_session,
    write_fixture,
)

if TYPE_CHECKING:
    from pydantic import JsonValue

    from tools.fixture_session import Stamp

GENERATED_PATH = FRONTEND_SRC / "test" / "tensorboardServings.json"

RUNS = ("train-x", "train-y")
"""The ids of the runs whose TensorBoards are served, two so a reader can tell their answers
apart."""


def _portless(requested: int) -> int:
    """Return what :func:`~tcip_mcp.web_client.free_port` answers when every bind is refused."""
    with mock.patch.object(
        socket.socket, "bind", side_effect=OSError("bind refused")
    ):
        return free_port(requested)


FAULTS: dict[str, AbstractContextManager[object]] = {
    "train-portless": mock.patch("tcip_mcp.web_client.free_port", _portless),
    "train-dying": mock.patch.object(subprocess, "Popen", DyingChild),
}
"""The ids of the runs whose launches fail, each with its fault: every port the launch tries
refusing its bind, and a child that writes one line and exits during startup."""

STAMPED: dict[str, Stamp] = {"port": (int, 0)}
"""The port a TensorBoard took, which varies by run, read as ``0``."""


def produce() -> dict[str, JsonValue]:
    """Open the runs, launch a TensorBoard over each through the route; return the answers.

    The answers are normalized and keyed by run id. A launch the route refuses, a run of
    :data:`RUNS` answered with no TensorBoard serving and a fault's run answered with no failure
    each raise, and so does a TensorBoard whose stop ``stop_tensorboard`` answers unconfirmed,
    naming its board and pid, whether or not the production raised first; a production's own
    exception is then the refusal's context.
    """
    from torch.utils.tensorboard import SummaryWriter

    from tcip_mcp.experiments import board_of
    from tcip_mcp.pipelines.training.tensorboard_manager import (
        TensorboardFailure,
        TensorboardServing,
        stop_tensorboard,
    )
    from tcip_web.routes._body_common import EmptyBodyPayload
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
            for experiment_id in (*RUNS, *FAULTS):
                config = detection_config(
                    workspace / f"{experiment_id}-data", project=root
                )
                board = board_of(
                    opened_run(root, config, experiment_id=experiment_id)
                )
                with SummaryWriter(str(board)) as writer:
                    writer.add_scalar("loss", 0.5, 1)
                if experiment_id in RUNS:
                    boards.append(board)
                with FAULTS.get(experiment_id, nullcontext()):
                    served[experiment_id] = answered(
                        client,
                        "launch_run_tensorboard",
                        200,
                        EmptyBodyPayload(),
                        path={"experiment_id": experiment_id},
                    )
                (
                    TensorboardFailure
                    if experiment_id in FAULTS
                    else TensorboardServing
                ).model_validate(served[experiment_id])
            return normalized(served, workspace, stamped=STAMPED)
        finally:
            stops = {board: stop_tensorboard(str(board)) for board in boards}
            unconfirmed = {
                str(board): stop["pid"]
                for board, stop in stops.items()
                if stop["status"] == "kill_unconfirmed"
            }
            assert not unconfirmed, (
                f"TensorBoards left with their kills unconfirmed: {unconfirmed}"
            )


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))
