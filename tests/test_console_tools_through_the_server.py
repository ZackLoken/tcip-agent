"""Each tool called through the real server's handshake and through its ``tcip`` command.

For one input the two answers are compared, and a value outside a typed parameter's vocabulary is
checked to be refused by the schema the server advertises, before the body runs.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from tcip_annotation.state import Annotation, BBox
from tcip_mcp.cli import hyphenated
from tcip_mcp.dataset_layout import image_dir
from tests._producer_fixtures import label_image, write_image
from tests._training_values import RATIONALE, TRIAGE_WEIGHTS

if TYPE_CHECKING:
    import mcp.types as mcp_types

    from tcip_mcp.workspace import BoundProject

DATE = "2024-01-01"
IMAGE = (image_dir("ds", DATE) / "a.png").as_posix()
"""The project-relative spelling of the one labeled, predicted image every render reads."""

BY = ["--user", "tester"]
"""The command's statement of the person a sample console call is made by."""

PLANT_DRAW = {
    "subject": "leaf",
    "seed": 3,
    "val_ratio": 0.15,
    "calibration_ratio": 0.1,
    "holdout_ratio": 0.05,
    "stratify_foreground": True,
    "rationale": RATIONALE,
}
"""A sample plant-grouped draw's statement, as the tool takes it."""


@dataclass
class Case:
    """One tool's call, its command's argv for the same input, and the answer's compared view."""

    tool: Callable[..., object]
    arguments: dict
    command: list[str]
    view: Callable[[dict], object] = lambda answer: answer


def _without(*keys: str) -> Callable[[dict], object]:
    return lambda answer: {k: v for k, v in answer.items() if k not in keys}


def _headroom(answer: dict) -> object:
    """Return what a headroom report states that does not move between two reads."""
    return (
        answer["cpu"]["logical_count"],
        answer["memory"]["total_bytes"],
        [gpu["total_bytes"] for gpu in answer["gpus"]],
        answer["active_training_runs"],
    )


def console_cases(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> list[Case]:
    """Return one case per console tool, its inputs written into ``project`` by their producers.

    The dataset under ``ds`` holds one labeled image, the bucket ``baseline`` published over it
    and the dataset registered; a registered checkpoint triages through a stubbed predictor; four
    georeferenced plants and a point shapefile sit at the project's root.
    """
    pytest.importorskip("torch")
    import tcip_mcp.pipelines.inference.generic_predictor as predmod
    from tcip_mcp.experiments import compare_experiments, run_observations
    from tcip_mcp.pipelines.training.hpo import (
        available_schedulers,
        available_search_algs,
    )
    from tcip_mcp.tools.annotation_tools import (
        read_annotations,
        score_predictions,
    )
    from tcip_mcp.tools.data_tools import (
        plant_aware_group_splits,
        scan_dataset,
    )
    from tcip_mcp.tools.feedback_tools import triage_predictions
    from tcip_mcp.tools.phenology_tools import shp_to_plant_csv
    from tcip_mcp.tools.project_tools import (
        check_dataset_identity,
        register_dataset,
    )
    from tcip_mcp.tools.training_tools import (
        inspect_compute_resources,
        preflight_config,
    )
    from tcip_mcp.tools.vision_tools import (
        overlay_reference_grid,
        render_failure_cases,
        visualize,
    )
    from tests._chain_fixtures import published
    from tests._mapping_fixtures import write_plant_csv
    from tests._verified_checkpoint_fixtures import registered_checkpoint
    from tests.test_plant_aware_group_splits_script import (
        P1_TIEPOINT,
        P2_TIEPOINT,
        P3_TIEPOINT,
        P4_TIEPOINT,
        _four_plant_dataset,
        _plant_at,
    )
    from tests.test_preflight_config_script import _fixture_config
    from tests.test_shp_to_plant_csv_script import _point_shapefile

    image = write_image(project / IMAGE)
    label_image(
        image,
        [Annotation(subject="bud", geometry=BBox(1, 1, 40, 30))],
        100,
        80,
    )
    published(
        project,
        "baseline",
        [
            {
                "image": str(image),
                "width": 100,
                "height": 80,
                "boxes": [[1.0, 1.0, 40.0, 30.0], [50.0, 50.0, 60.0, 60.0]],
                "scores": [0.9, 0.6],
                "labels": [1, 1],
                "cap": 3,
            }
        ],
        scope={"subject": "bud"},
    )
    assert "error" not in register_dataset(
        project, "ds", "chestnut", actor=None
    )
    checkpoint = registered_checkpoint(project)
    candidates = project / "candidates"
    candidates.mkdir()
    (candidates / "c.jpg").write_bytes(b"x")
    monkeypatch.setattr(
        predmod,
        "GenericPredictor",
        lambda *a, **k: SimpleNamespace(
            predict_batch=lambda sources, **kw: [
                {"image": "c.jpg", "scores": [0.5]}
            ]
        ),
    )
    _four_plant_dataset(project)
    plants = [
        _plant_at(f"P{k}", f"acc-{k}", tiepoint)
        for k, tiepoint in enumerate(
            (P1_TIEPOINT, P2_TIEPOINT, P3_TIEPOINT, P4_TIEPOINT), 1
        )
    ]
    write_plant_csv(project / "west.csv", plants[:2])
    write_plant_csv(project / "east.csv", plants[2:])
    _point_shapefile(project)
    _fixture_config(project)
    runs = sorted(obs.directory.name for obs in run_observations(project))
    named = ["--project", str(project), *BY]
    weights = [
        arg
        for key, weight in TRIAGE_WEIGHTS.items()
        for arg in (f"--{hyphenated(key)}", str(weight))
    ]
    return [
        Case(
            scan_dataset,
            {"folder_path": "ds"},
            ["--folder-path", "ds", *named],
        ),
        Case(
            check_dataset_identity,
            {"dataset_root": "ds"},
            ["--dataset-root", "ds", *named],
        ),
        Case(
            score_predictions,
            {
                "path": IMAGE,
                "bucket": "baseline",
                "iou_threshold": 0.5,
                "conf_threshold": 0.25,
                "detail": True,
            },
            [
                *("--path", IMAGE, "--bucket", "baseline", *named),
                *("--iou-threshold", "0.5", "--conf-threshold", "0.25"),
                *("--detail", "true"),
            ],
        ),
        Case(
            read_annotations,
            {"image_path": IMAGE, "bucket": "baseline"},
            ["--image-path", IMAGE, "--bucket", "baseline", *named],
        ),
        *(
            Case(
                visualize,
                {"render": render},
                ["--render", json.dumps(render), *named],
                _without("image_path"),
            )
            for render in (
                {"source": "annotations", "path": IMAGE},
                {"source": "predictions", "path": IMAGE, "bucket": "baseline"},
                {
                    "source": "comparison",
                    "path": IMAGE,
                    "bucket": "baseline",
                    "iou_threshold": 0.5,
                },
                {"source": "dataset", "path": "ds", "n": 1},
            )
        ),
        Case(
            overlay_reference_grid,
            {"image_path": IMAGE, "overlap": 0.0, "tile_size": 20},
            [
                *("--image-path", IMAGE, *named),
                *("--overlap", "0", "--tile-size", "20"),
            ],
            _without("image_path"),
        ),
        Case(
            render_failure_cases,
            {"dataset_root": "ds", "bucket": "baseline", "top_k": 1}
            | TRIAGE_WEIGHTS,
            [
                *("--dataset-root", "ds", "--bucket", "baseline", *named),
                *("--top-k", "1", *weights),
            ],
            _without("image_path", "case_images"),
        ),
        Case(inspect_compute_resources, {}, named, _headroom),
        Case(available_search_algs, {}, BY),
        Case(available_schedulers, {}, BY),
        Case(
            compare_experiments,
            {"experiment_ids": runs},
            ["--experiment-ids", json.dumps(runs), *named],
        ),
        Case(
            preflight_config,
            {
                "config": json.loads(
                    (project / "config.json").read_text(encoding="utf-8")
                )
            },
            ["--config", "@config.json", *named],
        ),
        Case(
            triage_predictions,
            {
                "checkpoint_path": checkpoint,
                "images_dir": "candidates",
                "low": 0.3,
                "high": 0.8,
                "rationale": RATIONALE,
            },
            [
                *(
                    "--checkpoint-path",
                    checkpoint,
                    "--images-dir",
                    "candidates",
                ),
                *(*named, "--low", "0.3", "--high", "0.8"),
                *("--rationale", RATIONALE),
            ],
        ),
        Case(
            plant_aware_group_splits,
            {
                "folder_path": "dataset",
                "plant_csv_paths": ["west.csv", "east.csv"],
                **PLANT_DRAW,
            },
            [
                *("--folder-path", "dataset", *named),
                *("--plant-csv-paths", "west.csv"),
                *("--plant-csv-paths", "east.csv"),
                *("--subject", "leaf", "--seed", "3", "--val-ratio", "0.15"),
                *("--calibration-ratio", "0.1", "--holdout-ratio", "0.05"),
                *("--stratify-foreground", "true", "--rationale", RATIONALE),
            ],
        ),
        Case(
            shp_to_plant_csv,
            {"shp_path": "points.shp", "csv_path": "tool.csv"},
            ["--shp-path", "points.shp", "--csv-path", "command.csv", *named],
            _without("csv_path"),
        ),
    ]


def read(text: str) -> object:
    """Return a printed answer as the JSON document it is, or the lines it holds."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text.splitlines()


def _body(result: mcp_types.CallToolResult) -> dict:
    """Return the tool's own answer to a call the server did not refuse."""
    from tests.test_agent_identity_records import _body as body

    assert not result.is_error, result
    return body(result)


def test_every_console_tool_answers_through_the_server_as_its_command_prints(
    project: Path,
    bound: BoundProject,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each tool answers through the real handshake what its command prints for one input.

    The command is the tool's own name :func:`~tcip_mcp.cli.hyphenated`, run through the
    console's dispatcher in this process, and prints the text the client reads; every answer is a
    success, never a refusal.
    """
    from mcp.types import TextContent

    from tcip_web.cli import main
    from tests.test_agent_identity_records import results_through_handshake

    cases = console_cases(project, monkeypatch)
    answers = results_through_handshake(
        [(case.tool.__name__, case.arguments) for case in cases], bound=bound
    )

    for case, result in zip(cases, answers, strict=True):
        name = case.tool.__name__
        assert "error" not in _body(result), name
        told = "".join(
            f"{block.text}\n"
            for block in result.content
            if isinstance(block, TextContent)
        )
        assert main([hyphenated(name), *case.command]) == 0, name
        printed = capsys.readouterr().out
        assert case.view(read(printed)) == case.view(read(told)), name


def test_the_usage_names_what_each_command_does(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``tcip --help`` lists each command beside the first line of what it states it does.

    A tool's line is the first of the description it is registered with, an operator command's
    the first of its module's docstring.
    """
    import importlib

    from tcip_web.cli import COMMANDS, main, tool_listing

    assert main(["--help"]) == 0
    listing = capsys.readouterr().out.partition("commands:\n")[2]
    listed = dict(
        line.split(maxsplit=1) for line in listing.splitlines() if line
    )
    stated = {
        hyphenated(tool.name): tool.description for tool in tool_listing()
    } | {
        name: importlib.import_module(module).__doc__
        for name, module in COMMANDS.items()
    }

    assert listed == {
        name: str(said).strip().splitlines()[0]
        for name, said in stated.items()
    }


def test_a_tool_command_prints_its_help_without_a_person_and_runs_only_with_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A tool's ``--help`` needs no ``--user``; the same command executed without one is refused."""
    from tcip_web.cli import main

    with pytest.raises(SystemExit) as helped:
        main(["available-schedulers", "--help"])
    assert helped.value.code == 0
    assert "--user" in capsys.readouterr().out

    with pytest.raises(SystemExit) as refused:
        main(["available-schedulers"])
    assert refused.value.code == 2
    assert "--user" in capsys.readouterr().err


def test_one_server_answers_a_tool_invocation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One tool command builds one server, its session then bound to the person; so does an unknown.

    The recognized command's server routes, lists and calls; the unknown command's usage is
    rendered from the listing its one server gave, its session never bound.
    """
    import tcip_mcp.server as server
    from tcip_web.cli import main

    built: list[server.Session] = []
    build = server.build_server

    def counted(session: server.Session) -> object:
        built.append(session)
        return build(session)

    monkeypatch.setattr(server, "build_server", counted)

    assert main(["available-schedulers", *BY]) == 0
    assert [session.actor for session in built] == ["user:tester"]

    built.clear()
    assert main(["no-such-command", *BY]) == 2
    assert [session.actor for session in built] == [None]
    assert "available-schedulers" in capsys.readouterr().err


def test_a_tool_command_reads_its_project_and_person_once(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The invocation's ``--project`` and ``--user`` are each read by one parse of the command."""
    import argparse

    from tcip_web.cli import main

    read: list[str] = []
    store = argparse._StoreAction.__call__

    def counted(
        action: argparse._StoreAction, *args: object, **kwargs: object
    ) -> None:
        read.append(action.dest)
        store(action, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(argparse._StoreAction, "__call__", counted)

    assert main(["available-schedulers", "--project", str(project), *BY]) == 0
    assert sorted(read) == ["project", "user"]


def test_a_value_outside_a_typed_parameter_is_refused_by_the_advertised_schema(
    bound: BoundProject,
) -> None:
    """A value outside a parameter's type is refused by the schema, never by the body.

    Each refused call's arguments are ones the registered tool's advertised input schema admits
    no value for, and the server's answer is its argument validation naming that parameter.
    """
    import anyio
    from jsonschema import Draft202012Validator

    from tests._server_calls import client_server
    from tests.test_agent_identity_records import results_through_handshake

    plant_draw = {
        "folder_path": "dataset",
        "plant_csv_paths": ["plants.csv"],
        **PLANT_DRAW,
    }
    refusals = [
        (
            "visualize",
            "render",
            {"render": {"source": "outline", "path": IMAGE}},
        ),
        (
            "visualize",
            "render.annotations.task",
            {
                "render": {
                    "source": "annotations",
                    "path": IMAGE,
                    "task": "keypoint",
                }
            },
        ),
        (
            "visualize",
            "render.predictions.bucket",
            {"render": {"source": "predictions", "path": IMAGE}},
        ),
        (
            "visualize",
            "render.comparison.iou_threshold",
            {
                "render": {
                    "source": "comparison",
                    "path": IMAGE,
                    "bucket": "baseline",
                }
            },
        ),
        (
            "visualize",
            "render.dataset.bucket",
            {
                "render": {
                    "source": "dataset",
                    "path": "ds",
                    "bucket": "baseline",
                }
            },
        ),
        (
            "plant_aware_group_splits",
            "plant_csv_paths",
            plant_draw | {"plant_csv_paths": "plants.csv"},
        ),
        (
            "plant_aware_group_splits",
            "stratify_foreground",
            plant_draw | {"stratify_foreground": "sometimes"},
        ),
        (
            "shp_to_plant_csv",
            "field_map",
            {
                "shp_path": "points.shp",
                "csv_path": "x.csv",
                "field_map": ["plant_id"],
            },
        ),
        (
            "shp_to_plant_csv",
            "field_map.plantid.[key]",
            {
                "shp_path": "points.shp",
                "csv_path": "x.csv",
                "field_map": {"plantid": "PID"},
            },
        ),
        (
            "render_failure_cases",
            "top_k",
            {"dataset_root": "ds", "bucket": "baseline", "top_k": "many"}
            | TRIAGE_WEIGHTS,
        ),
        (
            "score_predictions",
            "iou_threshold",
            {
                "path": IMAGE,
                "bucket": "baseline",
                "iou_threshold": "high",
                "conf_threshold": 0.25,
            },
        ),
    ]
    schemas = {
        listed.name: Draft202012Validator(listed.input_schema)
        for listed in anyio.run(client_server(bound).list_tools)
    }
    refused = results_through_handshake(
        [(name, arguments) for name, _, arguments in refusals], bound=bound
    )

    for (name, parameter, arguments), result in zip(
        refusals, refused, strict=True
    ):
        assert not schemas[name].is_valid(arguments), (name, arguments)
        (answer,) = result.content
        assert result.is_error, (name, answer)
        assert f"validation error for {name}Arguments\n{parameter}\n" in (
            answer.text
        ), answer.text


def test_a_trait_named_for_scoring_is_resolved_by_the_tool_itself(
    project: Path, bound: BoundProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A score naming a trait is governed by its confirmed revision, resolved by the tool.

    The confirmed count trait's criterion governs where the comparability convention governs
    the same score stated without one, and a name the project holds no trait for answers that
    trait's refusal.
    """
    from tests import _trait_fixtures as fx
    from tests.test_agent_identity_records import results_through_handshake

    (scored,) = [
        case.arguments
        for case in console_cases(project, monkeypatch)
        if case.tool.__name__ == "score_predictions"
    ]
    fx.seed_confirmed_count(project)

    by_trait, by_convention, unknown = (
        _body(result)
        for result in results_through_handshake(
            [
                ("score_predictions", scored | {"trait": fx.COUNT_TRAIT}),
                ("score_predictions", scored),
                ("score_predictions", scored | {"trait": "no_such_trait"}),
            ],
            bound=bound,
        )
    )

    assert "error" not in by_trait, by_trait
    assert (
        by_trait["governing_criterion"] != by_convention["governing_criterion"]
    )
    assert "no_such_trait" in unknown["error"]


def _printed(
    argv: list[str], project: Path, capsys: pytest.CaptureFixture[str]
) -> tuple[int, object, str]:
    """Run one command against ``project`` by ``BY``; return its exit code, answer and stderr."""
    from tcip_web.cli import main

    code = main([*argv, "--project", str(project), *BY])
    said = capsys.readouterr()
    return code, read(said.out), said.err


def test_a_value_reaches_the_tool_as_its_json_unless_the_argument_is_text(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stated value is its JSON, and stays the text itself where the schema admits a string.

    A whole number the schema admits strictly and a list of whole numbers reach the body, and a
    detail written as a numeral is stored as that text.
    """
    friction = ["report-friction", "--category", "missing_tool"]
    assert _printed([*friction, "--detail", "123"], project, capsys)[0] == 0
    code, page, _ = _printed(
        ["load-project-memory", "--kind", "reports", "--limit", "1"],
        project,
        capsys,
    )
    assert code == 0, page
    assert isinstance(page, dict), page
    assert [entry["record"]["detail"] for entry in page["entries"]] == ["123"]

    code, swept, said = _printed(
        [
            *("run-hyperparameter-search", "--base-config", "{}"),
            *("--param-space", "{}", "--n-trials", "1", "--search-seed", "0"),
            *("--search-alg", "random", "--scheduler", '{"name": "no-such"}'),
            *("--split-draw-seeds", "3", "--split-draw-seeds", "4"),
        ],
        project,
        capsys,
    )
    assert code == 0, said
    assert isinstance(swept, dict), swept
    assert "no-such" in swept["error"]


def test_a_file_value_is_read_against_the_project_and_one_unread_is_refused(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``@<path>`` is the file's text, so a value beginning with ``@`` is given through a file.

    A path no file is at is a refused call naming the path, never a traceback.
    """
    (project / "note.txt").write_text("@mention", encoding="utf-8")
    friction = ["report-friction", "--category", "missing_tool"]

    assert (
        _printed([*friction, "--detail", "@note.txt"], project, capsys)[0] == 0
    )
    code, answer, said = _printed(
        [*friction, "--detail", "@absent.txt"], project, capsys
    )
    assert code == 2, answer
    assert "absent.txt" in said
    page = _printed(
        ["load-project-memory", "--kind", "reports"], project, capsys
    )[1]
    assert isinstance(page, dict), page
    assert [entry["record"]["detail"] for entry in page["entries"]] == [
        "@mention"
    ]
