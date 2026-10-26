"""Write the training routes' answers as a fixture file, produced by the platform itself.

In a scratch session (``tools/fixture_session.py``), a project created through
``tests._web_fixtures.named_project`` and opened through the picker's route holds a dataset of
labeled frames (``tests._verified_checkpoint_fixtures.detection_images``) with one partition
``draw_splits`` drew under its ``splits`` directory (``tests.test_selection_binding._draw``), the
runs ``tools.generate_tensorboard_fixture.RUNS`` names over it, each opened by the launcher's own
producer and writer (``tests._verified_checkpoint_fixtures.opened_run``), one sweep opened by the
sweep door's own producer (``opened_sweep``) holding one trial (``training_tools.open_trial``), a
run whose builder no import resolves (``tests.test_tcip_web_training_routes._opened``), and a run
the child's own entry completes taking no optimizer step (``finished_run``), registering its
checkpoint. Each request is built through its route's request model. The routes' answers are
written, normalized by :data:`STAMPED` (``fixture_session.normalized``), to
``frontend/src/test/trainingAnswers.json``, keyed by answer: the listing, the first run's detail,
data choices and cancellation, the comparison of the first run and the completed one, that
comparison's ranking and two refusals of it, the metric directions, a relaunch of the first run and
of the sweep, each spawn recorded by ``tests._producer_fixtures.fake_popen`` and starting nothing,
the refusal of the unimportable run's relaunch, the serving route's two no-logs refusals, and the
frames the metrics stream sends for the second run once it has logged an epoch and a batch row and
failed, and for an id naming no run. A refusal is kept with its status (:func:`refusal`).

    python -m tools.generate_training_fixture
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from tests import FRONTEND_SRC
from tools.fixture_session import (
    SCRATCH_ROOT,
    normalized,
    opened,
    scratch_session,
    write_fixture,
)
from tools.generate_tensorboard_fixture import RUNS

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi.testclient import TestClient
    from pydantic import BaseModel, JsonValue

    from tools.fixture_session import Stamp

GENERATED_PATH = FRONTEND_SRC / "test" / "trainingAnswers.json"

DATASET_FRAMES = 8
"""The labeled frames of the fixture's dataset, enough for ``_draw``'s four sides to each hold
one."""

REFUSED_RUN = "train-refused"
"""The id of the run whose relaunch the route refuses."""

UNIMPORTABLE_BUILDER = "not.a:module"
"""The refused run's builder, whose module path names the keyword ``not``, which no import
resolves."""

COMPLETED_RUN = "train-done"
"""The id of the run the child's own entry completes."""

CHECKPOINT_METRICS = {"val_map50": 0.5}
"""The metrics the completed run's checkpoint carries, under a metric whose direction the platform
declares."""

ABSENT_METRIC = "val_map99"
"""A metric no registered checkpoint carries."""

FAILURE = "a sample failure the scene records for this run"
"""The error the scene's failed runs end with."""

UNKNOWN_RUN = "train-unknown"
"""An id naming no run, whose stream answers so."""

STAMPED: dict[str, Stamp] = {
    **{
        key: (str, f"<{key}>")
        for key in (
            "created",
            "heartbeat",
            "registered_at",
            "timestamp",
            "dataset_fingerprint",
        )
    },
    "pid": (int, 0),
}
"""The served values that differ between runs or platforms, each read as what it stands for: the
instants a run stamps, as their key in angle brackets that no date parser reads as an instant; the
digest of the image bytes a platform's encoder wrote; and the worker pid a test double fills, as
``0``."""


def refusal(
    client: TestClient,
    url: str,
    payload: BaseModel,
    status: int,
    refuses: Callable[[dict[str, JsonValue]], bool],
) -> dict[str, JsonValue]:
    """Post ``payload`` to ``url`` and return the route's refusal, its status and its body.

    Raises (``AssertionError``) an answer of another status than ``status``, or one whose detail
    is not a mapping ``refuses`` holds of, as a request the route's own model refuses answers a
    list of its issues.
    """
    response = client.post(url, json=payload.model_dump(mode="json"))
    body = response.json()
    assert (
        response.status_code == status
        and isinstance(body["detail"], dict)
        and refuses(body["detail"])
    ), response.text
    return {"status": response.status_code, "body": body}


def produce() -> dict[str, JsonValue]:
    """Lay down the runs and the sweep in a scratch session; return the routes' answers.

    A route refusing what it should admit raises, and so does a refusal other than the one its
    scenario names (:func:`refusal`).
    """
    import pytest

    from tcip_mcp.experiments import observe_run, write_final_status
    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tcip_mcp.subject_registry import Subject, SubjectRegistry
    from tcip_mcp.tools.training_tools import open_trial
    from tcip_web.routes._body_common import EmptyBodyPayload, PersonPayload
    from tcip_web.routes.training import (
        CompareBestPayload,
        ExperimentComparePayload,
        RelaunchPayload,
    )
    from tests._chain_fixtures import BESPOKE_MODELS, training_config
    from tests._producer_fixtures import fake_popen
    from tests._verified_checkpoint_fixtures import (
        BUILT_DETECTOR,
        SCOPED_DATA,
        detection_images,
        finished_run,
        opened_run,
        opened_sweep,
    )
    from tests._web_fixtures import named_project
    from tests.test_selection_binding import _draw
    from tests.test_tcip_web_training_routes import _opened

    with scratch_session() as (workspace, client):

        def answer(
            url: str, payload: BaseModel | None = None
        ) -> dict[str, JsonValue]:
            response = (
                client.get(url)
                if payload is None
                else client.post(url, json=payload.model_dump(mode="json"))
            )
            response.raise_for_status()
            return response.json()

        def frames(experiment_id: str) -> list[dict[str, JsonValue]]:
            stream = client.base_url.join(
                f"/api/training/runs/{experiment_id}/stream"
            )
            sent: list[dict[str, JsonValue]] = []
            with client.websocket_connect(
                str(stream.copy_with(scheme="ws"))
            ) as ws:
                while not sent or sent[-1]["type"] != "status":
                    sent.append(ws.receive_json())
            return sent

        root = workspace / "training"
        opened(client, named_project(root, "Training"))
        dataset = root / "dataset"
        scope = cast("dict[str, str]", SCOPED_DATA["scope"])
        images = detection_images(
            dataset,
            scope,
            n=DATASET_FRAMES,
            registry=SubjectRegistry(
                subjects=(Subject(name=scope["subject"], description=""),)
            ),
        )
        _draw(root, dataset, dataset / "splits", subject=scope["subject"])
        config = training_config(
            root, BUILT_DETECTOR, {**images, **SCOPED_DATA}
        )
        run_dirs = {
            experiment_id: opened_run(
                root, config, experiment_id=experiment_id
            )
            for experiment_id in RUNS
        }
        sweep_dir = opened_sweep(root, config)
        open_trial(sweep_dir, "t0", {"optimizer.head_lr": 0.001})
        sweep = sweep_dir.name

        run, other = RUNS
        served: dict[str, JsonValue] = {
            "listing": answer("/api/training/runs"),
            "detail": answer(f"/api/training/runs/{run}"),
            "split_choices": answer(f"/api/training/runs/{run}/splits"),
            "cancel": answer(
                f"/api/training/runs/{run}/cancel",
                PersonPayload(user="grower"),
            ),
        }

        finished_run(
            root,
            experiment_id=COMPLETED_RUN,
            metrics=CHECKPOINT_METRICS,
            rows=[{"epoch": 1, "loss": 0.5}],
        )
        compared = [run, COMPLETED_RUN]
        (metric,) = CHECKPOINT_METRICS
        served |= {
            "compare": answer(
                "/api/training/compare",
                ExperimentComparePayload(experiment_ids=compared),
            ),
            "compare_best": answer(
                "/api/training/compare/best",
                CompareBestPayload(
                    experiment_ids=compared,
                    metric=metric,
                    include_unverified=True,
                ),
            ),
            "unverified_refusal": refusal(
                client,
                "/api/training/compare/best",
                CompareBestPayload(experiment_ids=compared, metric=metric),
                422,
                lambda detail: detail["all_unverified"] is True,
            ),
            "absent_metric_refusal": refusal(
                client,
                "/api/training/compare/best",
                CompareBestPayload(
                    experiment_ids=compared,
                    metric=ABSENT_METRIC,
                    higher_is_better=True,
                ),
                422,
                lambda detail: ABSENT_METRIC in str(detail["message"]),
            ),
            "metric_directions": answer("/api/training/metric-directions"),
            "no_logs_refusal": refusal(
                client,
                f"/api/training/runs/{run}/tensorboard",
                EmptyBodyPayload(),
                404,
                lambda detail: detail["no_logs"] is True,
            ),
        }

        refused = _opened(REFUSED_RUN, root, builder=UNIMPORTABLE_BUILDER)
        with pytest.MonkeyPatch.context() as patch:
            fake_popen(patch, [])
            run_relaunch, sweep_relaunch = [
                answer(
                    "/api/training/runs",
                    RelaunchPayload(relaunched_from=source, user="grower"),
                )
                for source in (run, sweep)
            ]
            served["refused_relaunch"] = refusal(
                client,
                "/api/training/runs",
                RelaunchPayload(relaunched_from=refused.name, user="grower"),
                422,
                lambda detail: bool(detail["issues"]),
            )
        write_final_status(refused, "failed", FAILURE, checkpoint=None)

        streamed = run_dirs[other]
        context = TrainContext(
            run=observed_run(observe_run(streamed)), train_loader=None
        )
        context.log_metrics(1, {"loss": 0.5})
        context.log_batch(2, 2, {"batch_loss": 0.25})
        write_final_status(streamed, "failed", FAILURE, checkpoint=None)
        by_type = {frame["type"]: frame for frame in frames(other)}
        (unknown,) = frames(UNKNOWN_RUN)
        served |= {
            "run_relaunch": run_relaunch,
            "sweep_relaunch": sweep_relaunch,
            "crashed_no_logs_refusal": refusal(
                client,
                f"/api/training/runs/{refused.name}/tensorboard",
                EmptyBodyPayload(),
                404,
                lambda detail: (
                    detail["message"] == FAILURE and detail["no_logs"] is True
                ),
            ),
            "metric_frame": by_type["metric"],
            "batch_frame": by_type["batch"],
            "status_frame": by_type["status"],
            "unknown_frame": unknown,
        }
        return normalized(
            served,
            stamped=STAMPED,
            rounded=True,
            minted={
                sweep: "<sweep id>",
                cast("str", run_relaunch["experiment_id"]): (
                    "<relaunched run id>"
                ),
                cast(
                    "str", sweep_relaunch["sweep_id"]
                ): "<relaunched sweep id>",
            },
            places={
                workspace.parent: SCRATCH_ROOT,
                Path(BESPOKE_MODELS).parent: "<tests>",
            },
        )


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))
