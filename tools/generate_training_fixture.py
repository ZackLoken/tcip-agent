"""Write the training routes' answers as a fixture file, produced by the platform itself.

In a scratch session (``tools/fixture_session.py``), a project created through
``tests._web_fixtures.named_project`` and opened through the picker's route holds a dataset of
labeled frames (``tests._verified_checkpoint_fixtures.detection_images``) with one partition
``draw_splits`` drew under its ``splits`` directory (``tests.test_selection_binding._draw``) and
a second under :data:`UNREADABLE_SELECTION` whose record is then damaged, the
runs ``tools.generate_tensorboard_fixture.RUNS`` names over it, each opened by the launcher's own
producer and writer (``tests._verified_checkpoint_fixtures.opened_run``), one sweep opened by the
sweep door's own producer (``opened_sweep``) holding a trial that logs an epoch and completes and
one still running (``training_tools.open_trial``), a run whose builder no import resolves
(``tests.test_tcip_web_training_routes._opened``), the runs :data:`FINISHED_RUNS` names, which the
child's own entry completes taking no optimizer step (``finished_run``), registering each
checkpoint, a run that completes having logged nothing (:data:`EMPTY_RUN`), a canceled run over
images under no dataset root (:data:`LOOSE_RUN`), and a run
:data:`DECLARED_CLIENT` launches as a declared agent (``launch_training``). Each request is built
through its route's request model. The routes' answers are written, normalized by
:data:`NUMBERED` and :data:`STAMPED` (``fixture_session.normalized``), to
:data:`GENERATED_PATH`,
keyed by answer: the first run's data choices and cancellation; comparisons of the first run and
a completed one (``compare``), of the two opened runs (``compare_same``), of the first run alone
(``compare_one``), of it and an id naming no run (``compare_unread``), of it and the loose run
(``compare_unset``), and of the finished runs'
pairs (``compare_rich``, ``compare_unstamped``, ``compare_unranked``), with the rankings of three
of them and two refusals of the first; a relaunch of the first run bound to
its selection and of the sweep, each spawn recorded by ``tests._producer_fixtures.fake_popen`` and
starting nothing, and the comparison of the bound relaunch with its source (``compare_bound``); the
refusal of the unimportable run's relaunch; the serving route's two no-logs refusals; the frames
the metrics stream sends, live, while the second run logs :data:`STREAMED_ROWS` and then fails
(``frames``), and for an id naming no run; the listing taken before that failure and the one taken
once every scene ran; the detail of every listed run, sweep and trial; once a run over a
dataset of its own is opened and that dataset's images removed (:data:`MOVED_RUN`), its data
choices (``moved_split_choices``); and, once the default trainer completes a regression run over
:data:`VERIFIED_FRAMES` (:data:`VERIFIED_RUN`, ``tests._verified_checkpoint_fixtures.worker_run``),
the comparison of it and the unstamped run (``compare_verified``) and their ranking by
:data:`VERIFIED_METRIC` with unverified checkpoints left out (``verified_best``), the one ranking
that does not include them. A refusal is kept with its status.

    python -m tools.generate_training_fixture
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, cast

from tests import FRONTEND_SRC
from tools.fixture_session import (
    ACTOR,
    answered,
    detail_of,
    normalized,
    opened,
    scratch_session,
    socket_url,
    write_fixture,
)
from tools.generate_tensorboard_fixture import RUNS

if TYPE_CHECKING:
    from pydantic import JsonValue

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

CHECKPOINT_METRICS: dict[str, object] = {"val_map50": 0.5}
"""The metrics the completed run's checkpoint carries, under a metric whose direction the platform
declares."""

COMPLETED_ROW: dict[str, object] = {
    "epoch": 1,
    "loss": 0.5,
    "selection": 0.4130041,
}
"""The completed run's one logged epoch row, its selection value a sample carrying more than four
significant figures."""

STREAMED_ROWS: tuple[tuple, ...] = (
    ("batch", 1, 3, {"batch_loss": 0.25}),
    ("metric", 1, {"loss": 0.5, "val_map50": 0.4}),
    ("metric", 2, {"loss": float("nan")}),
    ("metric", 3, {"loss": 0.2}),
    ("batch", 4, 7, {"batch_loss": 0.25}),
)
"""What the streamed run logs before it fails, in order, each a batch row (its epoch, its step
and its values) or an epoch row (its epoch and its values): a batch before its epoch's row, three
epoch rows the second of them not finite, and a batch of an epoch the run never finished. Sample
values."""

EMPTY_RUN = "train-empty"
"""The id of a run that completed having logged nothing."""

UNREADABLE_SELECTION = "unreadable"
"""The directory, one below the dataset's splits, of a drawn selection whose record is then
damaged, so the data choices offer it disabled with its read error."""
LOOSE_RUN = "train-loose"
"""The id of a run over images under no dataset root, so its launch records no fingerprint, then
canceled."""

MOVED_RUN = "train-moved"
"""The id of a run whose images directory is removed once it is opened."""

RICH_RUN = "train-rich"
"""The id of a completed run whose checkpoint carries a declared metric, an undeclared one and a
string."""

UNDECLARED_RUN = "train-undeclared"
"""The id of a completed run whose checkpoint carries only a metric no direction is declared for."""

UNSTAMPED_RUN = "train-unstamped"
"""The id of a completed run whose checkpoint carries no metric while its log carries one."""

VERIFIED_RUN = "train-verified"
"""The id of a regression run the default trainer completes, so its checkpoint's metrics are the
trainer's."""

VERIFIED_FRAMES = ((0.1, 0.2), (0.3, 0.6), (0.5, 1.0), (0.7, 1.4))
"""The verified run's frames, each a mean intensity and the value it is labeled with, sample
values."""

VERIFIED_METRIC = "val_loss"
"""The metric the verified run is ranked by: the validation loss its trainer stamps, whose direction
the platform declares."""

FINISHED_RUNS: dict[str, tuple[dict[str, object], list[dict[str, object]]]] = {
    COMPLETED_RUN: (CHECKPOINT_METRICS, [COMPLETED_ROW]),
    RICH_RUN: (
        {"val_map50": 0.71234, "val_map99": 0.7, "val_selection": "held-out"},
        [{"epoch": 1, "loss": 0.5}],
    ),
    UNDECLARED_RUN: ({"val_map99": 0.2}, [{"epoch": 1, "loss": 0.5}]),
    UNSTAMPED_RUN: ({}, [{"epoch": 1, "loss": 0.5, "val_map50": 0.712345}]),
}
"""Each completed run the child's own entry finishes, by id: its checkpoint's metrics and its
logged epoch rows, sample values, those a reader rounds carrying more than four significant
figures."""

TRIAL_ROW = {"loss": 0.75, "selection": 0.7}
"""The one epoch row the sweep's trial logs before it completes, sample values."""

DECLARED_CLIENT = ("claude-code", "2.1.238")
"""The client name and version a sample agent declares before launching a run."""

ABSENT_METRIC = "val_map99"
"""A metric no registered checkpoint carries."""

FAILURE = "a sample failure the scene records for this run"
"""The error the scene's failed runs end with."""

UNKNOWN_RUN = "train-unknown"
"""An id naming no run, whose stream answers so."""

NUMBERED = frozenset({"dataset_fingerprint", "sha256", "agent_session"})
"""The served strings that differ between runs or platforms: the digest of the image bytes a
platform's encoder wrote and of the checkpoint bytes a run saved, and the agent session minted."""

STAMPED: dict[str, Stamp] = {"pid": (int, 0)}
"""The worker pid a test double fills, read as ``0``."""


def produce() -> dict[str, JsonValue]:
    """Lay down the runs and the sweep in a scratch session; return the routes' answers.

    A route refusing what it should admit raises, and so does a refusal of another status than its
    scenario names (:func:`~tools.fixture_session.answered`) or, for the serving route, a no-logs
    detail other than its scenario's (``NoLogsRefusal``).
    """
    import pytest

    from tcip_mcp import agent_identity
    from tcip_mcp.audit import now
    from tcip_mcp.experiments import (
        TrainingListing,
        observe_run,
        write_final_status,
    )
    from tcip_mcp.pipelines.data.selection import selection_key
    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tcip_mcp.subject_registry import Subject, SubjectRegistry
    from tcip_mcp.tools.training_tools import (
        RunLaunched,
        SplitChoices,
        SweepLaunched,
        launch_training,
        open_trial,
    )
    from tcip_web.routes._body_common import EmptyBodyPayload, PersonPayload
    from tcip_web.routes.training import (
        CompareBestPayload,
        ExperimentComparePayload,
        NoLogsRefusal,
        RelaunchPayload,
    )
    from tests._chain_fixtures import (
        BESPOKE_MODELS,
        FRAME_STATISTICS,
        training_config,
    )
    from tests._producer_fixtures import fake_popen
    from tests._record_damage_fixtures import damage_record
    from tests._training_values import drawn_split
    from tests._verified_checkpoint_fixtures import (
        BUILT_DETECTOR,
        SCOPED_DATA,
        detection_config,
        detection_images,
        finished_run,
        opened_run,
        opened_sweep,
        table_images,
        worker_run,
    )
    from tests._web_fixtures import named_project
    from tests.test_selection_binding import _draw
    from tests.test_tcip_web_training_routes import _opened
    from tests.tiny_trainer_fixtures import (
        MEAN_INTENSITY_CLASSIFIER,
        TINY_TRAINER_FILE,
        regressor_config,
        write_regression_dataset,
    )

    with scratch_session() as (workspace, client):

        def frames(experiment_id: str) -> list[dict[str, JsonValue]]:
            sent: list[dict[str, JsonValue]] = []
            with client.websocket_connect(
                socket_url(
                    client, "training_stream_ws", experiment_id=experiment_id
                )
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
        unreadable = dataset / "splits" / UNREADABLE_SELECTION
        _draw(root, dataset, unreadable, subject=scope["subject"])
        config = training_config(
            root, BUILT_DETECTOR, {**images, **SCOPED_DATA}
        )
        damage_record(selection_key(unreadable), b"[]")
        run_dirs = {
            experiment_id: opened_run(
                root, config, experiment_id=experiment_id
            )
            for experiment_id in RUNS
        }
        sweep_dir = opened_sweep(root, config)
        trial = open_trial(sweep_dir, "t0", {"optimizer.head_lr": 0.001})
        TrainContext(
            run=observed_run(observe_run(trial)), train_loader=None
        ).log_metrics(1, TRIAL_ROW)
        write_final_status(trial, "completed", None, ended=now())
        open_trial(sweep_dir, "t1", {"optimizer.head_lr": 0.01})
        sweep = sweep_dir.name

        def split_choices(experiment_id: str) -> JsonValue:
            return answered(
                client,
                "list_split_choices_route",
                200,
                path={"experiment_id": experiment_id},
            )

        run, other = RUNS
        of_run = {"experiment_id": run}
        served: dict[str, JsonValue] = {
            "split_choices": split_choices(run),
            "cancel": answered(
                client,
                "cancel_run_route",
                200,
                PersonPayload(user=ACTOR),
                path=of_run,
            ),
        }

        for experiment_id, (metrics, rows) in FINISHED_RUNS.items():
            finished_run(
                root, experiment_id=experiment_id, metrics=metrics, rows=rows
            )
        write_final_status(
            opened_run(root, config, experiment_id=EMPTY_RUN),
            "completed",
            None,
            ended=now(),
        )
        loose = opened_run(
            root,
            training_config(
                root,
                {
                    "builder": MEAN_INTENSITY_CLASSIFIER,
                    "source_files": [TINY_TRAINER_FILE],
                    "task": "classification",
                    "normalization": FRAME_STATISTICS,
                },
                {
                    **table_images(workspace / "loose"),
                    "split": drawn_split(
                        val_ratio=0.15, grouping="tile_prefix"
                    ),
                },
            ),
            experiment_id=LOOSE_RUN,
        )
        write_final_status(loose, "canceled", None, ended=now())
        compared = [run, COMPLETED_RUN]
        (metric,) = CHECKPOINT_METRICS

        def comparison(*ids: str) -> JsonValue:
            return answered(
                client,
                "compare_runs_route",
                200,
                ExperimentComparePayload(experiment_ids=list(ids)),
            )

        def ranking(
            *ids: str, by: str = metric, include_unverified: bool = True
        ) -> JsonValue:
            return answered(
                client,
                "compare_best_route",
                200,
                CompareBestPayload(
                    experiment_ids=list(ids),
                    metric=by,
                    include_unverified=include_unverified,
                ),
            )

        served |= {
            "compare": comparison(*compared),
            "compare_same": comparison(run, other),
            "compare_one": comparison(run),
            "compare_unread": comparison(run, UNKNOWN_RUN),
            "compare_unset": comparison(run, LOOSE_RUN),
            "compare_rich": comparison(run, RICH_RUN),
            "compare_unstamped": comparison(UNSTAMPED_RUN, run),
            "compare_unranked": comparison(UNDECLARED_RUN, COMPLETED_RUN),
            "compare_best": ranking(*compared),
            "rich_best": ranking(run, RICH_RUN),
            "unranked_best": ranking(UNDECLARED_RUN, COMPLETED_RUN),
            "unverified_refusal": answered(
                client,
                "compare_best_route",
                422,
                CompareBestPayload(experiment_ids=compared, metric=metric),
            ),
            "absent_metric_refusal": answered(
                client,
                "compare_best_route",
                422,
                CompareBestPayload(
                    experiment_ids=compared,
                    metric=ABSENT_METRIC,
                    higher_is_better=True,
                ),
            ),
            "no_logs_refusal": answered(
                client,
                "launch_run_tensorboard",
                404,
                EmptyBodyPayload(),
                path={"experiment_id": EMPTY_RUN},
            ),
        }
        NoLogsRefusal.model_validate(detail_of(served["no_logs_refusal"]))

        refused = _opened(REFUSED_RUN, root, builder=UNIMPORTABLE_BUILDER)
        (bound,) = (
            choice
            for choice in SplitChoices.model_validate(
                served["split_choices"]
            ).selections
            if choice.reason is None
        )
        with pytest.MonkeyPatch.context() as patch:
            fake_popen(patch, [])
            run_relaunch, sweep_relaunch = [
                answered(
                    client,
                    "relaunch_route",
                    200,
                    RelaunchPayload(
                        relaunched_from=source,
                        selection_dir=chosen,
                        user=ACTOR,
                    ),
                )
                for source, chosen in (
                    (run, bound.selection_dir),
                    (sweep, None),
                )
            ]
            agent_identity.begin(*DECLARED_CLIENT)
            try:
                agent_run = RunLaunched.model_validate(
                    launch_training(root, config, actor=None)
                ).experiment_id
            finally:
                agent_identity.end()
            served["refused_relaunch"] = answered(
                client,
                "relaunch_route",
                422,
                RelaunchPayload(relaunched_from=refused.name, user=ACTOR),
            )
        write_final_status(refused, "failed", FAILURE, ended=now())

        streamed = run_dirs[other]
        context = TrainContext(
            run=observed_run(observe_run(streamed)), train_loader=None
        )
        streamed_frames: list[JsonValue] = []
        with client.websocket_connect(
            socket_url(client, "training_stream_ws", experiment_id=other)
        ) as ws:
            for kind, epoch, *logged in STREAMED_ROWS:
                if kind == "batch":
                    step, row = logged
                    context.log_batch(step, epoch, row)
                else:
                    (row,) = logged
                    context.log_metrics(epoch, row)
                streamed_frames.append(ws.receive_json())
            served["listing_before_failure"] = answered(
                client, "list_runs_route", 200
            )
            write_final_status(streamed, "failed", FAILURE, ended=now())
            streamed_frames.append(ws.receive_json())
        (unknown,) = frames(UNKNOWN_RUN)
        relaunched = RunLaunched.model_validate(run_relaunch).experiment_id
        listing = answered(client, "list_runs_route", 200)
        listed = TrainingListing.model_validate(listing)
        served |= {
            "listing": listing,
            "details": [
                answered(client, "get_run", 200, path={"experiment_id": i})
                for i in (
                    *(row.experiment_id for row in listed.runs),
                    *(s.sweep_id for s in listed.sweeps),
                    *(
                        t.experiment_id
                        for s in listed.sweeps
                        for t in s.trials
                    ),
                )
            ],
            "compare_bound": comparison(relaunched, run),
            "run_relaunch": run_relaunch,
            "sweep_relaunch": sweep_relaunch,
            "crashed_no_logs_refusal": answered(
                client,
                "launch_run_tensorboard",
                404,
                EmptyBodyPayload(),
                path={"experiment_id": refused.name},
            ),
            "frames": streamed_frames,
            "unknown_frame": unknown,
        }
        crashed = NoLogsRefusal.model_validate(
            detail_of(served["crashed_no_logs_refusal"])
        )
        assert FAILURE in crashed.message, crashed

        moved = detection_config(workspace / "moved-data", project=root)
        opened_run(root, moved, experiment_id=MOVED_RUN)
        shutil.rmtree(Path(moved["data"]["images_dir"]))
        served["moved_split_choices"] = split_choices(MOVED_RUN)

        verified_images, verified_values = write_regression_dataset(
            workspace / "verified", *zip(*VERIFIED_FRAMES, strict=True)
        )
        verified = regressor_config(
            data={
                "images_dir": str(verified_images),
                "labels_dir": str(verified_values),
                "split": drawn_split(val_ratio=0.25),
            },
            project=root,
        )
        worker_run(root, verified, experiment_id=VERIFIED_RUN)
        served |= {
            "compare_verified": comparison(VERIFIED_RUN, UNSTAMPED_RUN),
            "verified_best": ranking(
                VERIFIED_RUN,
                UNSTAMPED_RUN,
                by=VERIFIED_METRIC,
                include_unverified=False,
            ),
        }
        return normalized(
            served,
            workspace,
            stamped=STAMPED,
            numbered=NUMBERED,
            rounded=True,
            minted={
                sweep: "<sweep id>",
                agent_run: "<agent run id>",
                relaunched: "<relaunched run id>",
                SweepLaunched.model_validate(
                    sweep_relaunch
                ).sweep_id: "<relaunched sweep id>",
            },
            places={Path(BESPOKE_MODELS).parent: "<tests>"},
        )


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))
