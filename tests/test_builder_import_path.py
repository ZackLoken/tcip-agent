"""A bespoke builder outside the interpreter's search path imports through its run's layout.

The declared files, staged into a layout, lead ``sys.path`` when the builder imports, no child
process inherits that root, and a run naming such a builder launches in a process of its own and
completes.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests._chain_fixtures import (
    FRAME_STATISTICS,
    training_config,
    weights_normalization,
)
from tests._training_values import drawn_split, evaluation_block

if TYPE_CHECKING:
    from tcip_mcp.pipelines.model_build import SourceLayout

TOP_LEVEL_BUILDER = """
def build_probe(**kwargs):
    return {"built": True, "kwargs": kwargs}
"""

PACKAGED_BUILDER = """
from tests.tiny_trainer_fixtures import build_mean_intensity_classifier


def build(**kwargs):
    return build_mean_intensity_classifier(**kwargs)
"""

SETTINGS_BUILDER = """
import json
from pathlib import Path

from tests.tiny_trainer_fixtures import build_mean_intensity_classifier


def build(**kwargs):
    settings = json.loads(Path(__file__).with_name("run.json").read_text(encoding="utf-8"))
    return build_mean_intensity_classifier(**kwargs, **settings)
"""
"""A packaged builder reading its settings from a declared ``run.json`` beside it."""

PROBE_MODULE = "probe_builder"
PROBE_BUILDER = f"{PROBE_MODULE}:build_probe"
"""The builder :data:`TOP_LEVEL_BUILDER` defines, as a ``model_source`` names it."""


def _builder_dir(tmp_path: Path) -> Path:
    """Return a directory outside the interpreter's path holding :data:`PROBE_MODULE`."""
    src = tmp_path / "agent_project" / "models"
    src.mkdir(parents=True)
    (src / f"{PROBE_MODULE}.py").write_text(
        TOP_LEVEL_BUILDER, encoding="utf-8", newline="\n"
    )
    return src


def _packaged_builder(tmp_path: Path, package: str) -> tuple[Path, list[str]]:
    """Write ``<tmp>/agent_project/<package>/model.py`` holding ``build`` beside its initializer.

    The project directory is the import root, one level above the file. Returns the project and
    the two files a run declares.
    """
    project = tmp_path / "agent_project"
    pkg = project / package
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8", newline="\n")
    model = pkg / "model.py"
    model.write_text(PACKAGED_BUILDER, encoding="utf-8", newline="\n")
    return project, [str(model), str(pkg / "__init__.py")]


def _layout(builder: str, files: list[str], project: Path) -> SourceLayout:
    """Return the layout a classification run's admission stages for ``builder`` and ``files``.

    The run is ``project``'s, over no data, staged by ``model_build.staged_sources``.
    """
    from tcip_mcp.pipelines.model_build import staged_sources
    from tcip_mcp.pipelines.schemas import train_config

    return staged_sources(
        train_config(
            training_config(
                project,
                {
                    "builder": builder,
                    "task": "classification",
                    "source_files": files,
                    "normalization": weights_normalization(),
                },
                {},
            )
        ),
        project,
    ).layout


def _child_imports(
    module: str, env: dict[str, str]
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", f"import {module}; print({module}.__name__)"],
        capture_output=True,
        text=True,
        env=env,
    )


def test_a_builder_outside_the_interpreters_path_imports_from_its_layout_no_child_inherits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A builder outside the interpreter's path imports from its layout, which no child inherits.

    The builder imports from its layout's root, first on ``sys.path``; a child process launched
    with ``child_pythonpath`` inherits no layout root, whether the inherited ``PYTHONPATH``
    spells it as placed, with forward slashes or, on Windows, with its case swapped, and the
    module does not import there.
    """
    from tcip_mcp.pipelines.model_build import (
        child_pythonpath,
        import_source_builder,
    )

    src = _builder_dir(tmp_path)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(src)])
    layout = _layout(
        PROBE_BUILDER, [str(src / f"{PROBE_MODULE}.py")], tmp_path
    )

    fn = import_source_builder(PROBE_BUILDER, layout)
    assert fn(width=4) == {"built": True, "kwargs": {"width": 4}}
    assert sys.path[0] == str(layout.root)
    assert (layout.root / f"{PROBE_MODULE}.py").is_file()

    inherited = os.environ.get("PYTHONPATH", "")
    spellings = [str(layout.root), layout.root.as_posix()]
    if sys.platform == "win32":
        spellings.append(str(layout.root).swapcase())
    for spelling in spellings:
        monkeypatch.setenv(
            "PYTHONPATH", os.pathsep.join([spelling, inherited])
        )
        assert spelling not in child_pythonpath().split(os.pathsep)
        child = _child_imports(
            PROBE_MODULE, {**os.environ, "PYTHONPATH": child_pythonpath()}
        )
        assert child.returncode != 0, spelling
        assert "ModuleNotFoundError" in child.stderr


def test_a_builder_imports_in_a_fresh_interpreter(tmp_path: Path) -> None:
    """A fresh interpreter imports and builds the builder from the layout laid out here."""
    import tcip_mcp

    layout = _layout(
        PROBE_BUILDER,
        [str(_builder_dir(tmp_path) / f"{PROBE_MODULE}.py")],
        tmp_path,
    )
    script = (
        "import sys\n"
        "from pathlib import Path, PurePosixPath\n"
        "from tcip_mcp.pipelines.model_build import SourceLayout, import_source_builder\n"
        "layout = SourceLayout(Path(sys.argv[1]), tuple(map(PurePosixPath, sys.argv[3:])),\n"
        "                      sys.argv[2])\n"
        f"print(import_source_builder({PROBE_BUILDER!r}, layout)(width=7))\n"
    )

    child = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(layout.root),
            layout.snapshot,
            *map(str, layout.places),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(tcip_mcp.__file__).parents[1]),
        },
    )

    assert child.returncode == 0, child.stderr
    assert child.stdout.splitlines() == [
        "{'built': True, 'kwargs': {'width': 7}}"
    ]


def test_a_builder_imports_each_declared_module_from_its_own_tree_however_deep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A builder, its helper and the helper's dependency import from the layout being loaded.

    The three modules, all declared, are loaded from one tree and then from another; the second
    load builds the value the second tree's dependency holds.
    """
    import uuid

    from tcip_mcp.pipelines.model_build import import_source_builder

    monkeypatch.setattr(sys, "path", list(sys.path))
    tag = uuid.uuid4().hex[:8]
    builder, helper, dependency = (
        f"{role}_{tag}" for role in ("builder", "helper", "dep")
    )
    plans = []
    for value in (11, 22):
        tree = tmp_path / f"tree{value}"
        tree.mkdir()
        (tree / f"{dependency}.py").write_text(
            f"VALUE = {value}\n", encoding="utf-8"
        )
        (tree / f"{helper}.py").write_text(
            f"import {dependency}\nVALUE = {dependency}.VALUE\n",
            encoding="utf-8",
        )
        (tree / f"{builder}.py").write_text(
            f"import {helper}\n\n\ndef build():\n    return {helper}.VALUE\n",
            encoding="utf-8",
        )
        plans.append(
            _layout(
                f"{builder}:build",
                [
                    str(tree / f"{name}.py")
                    for name in (builder, helper, dependency)
                ],
                tree,
            )
        )

    assert import_source_builder(f"{builder}:build", plans[0])() == 11
    assert import_source_builder(f"{builder}:build", plans[1])() == 22


def test_a_declared_file_edited_between_two_loads_imports_as_edited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declared builder rewritten between two loads in one process builds as rewritten."""
    import uuid

    from tcip_mcp.pipelines.model_build import import_source_builder

    monkeypatch.setattr(sys, "path", list(sys.path))
    module = f"edited_{uuid.uuid4().hex[:8]}"
    builder = tmp_path / f"{module}.py"
    values = []
    for value in (11, 2222):
        builder.write_text(
            f"def build():\n    return {value}\n", encoding="utf-8"
        )
        values.append(
            import_source_builder(
                f"{module}:build",
                _layout(f"{module}:build", [str(builder)], tmp_path),
            )()
        )

    assert values == [11, 2222]


def test_the_last_plan_imported_leads_the_path_and_no_other_plan_stays_on_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last plan imported leads ``sys.path``, and no other plan's root stays on it.

    Plans A, B, A imported in turn leave A's root first on ``sys.path`` and B's root off it.
    """
    import uuid

    from tcip_mcp.pipelines.model_build import import_source_builder

    monkeypatch.setattr(sys, "path", list(sys.path))
    module = f"rooted_{uuid.uuid4().hex[:8]}"
    plans = {}
    for name in ("a", "b"):
        tree = tmp_path / name
        tree.mkdir()
        (tree / f"{module}.py").write_text(
            f"def build():\n    return {name!r}\n", encoding="utf-8"
        )
        plans[name] = _layout(
            f"{module}:build", [str(tree / f"{module}.py")], tree
        )

    roots = {}
    for name in ("a", "b", "a"):
        assert import_source_builder(f"{module}:build", plans[name])() == name
        roots[name] = sys.path[0]

    assert sys.path[0] == roots["a"]
    assert roots["b"] not in sys.path


def test_an_import_takes_off_the_path_only_the_layout_roots_imports_put_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second layout's import removes the first layout's root and keeps the caller's entry.

    The second layout's root leads ``sys.path``, the first layout's root is gone from it, and the
    caller's own directory named like a snapshot stays on it.
    """
    import uuid

    from tcip_mcp.pipelines.model_build import (
        SNAPSHOT_DIR,
        import_source_builder,
    )

    own = tmp_path / "own" / SNAPSHOT_DIR
    own.mkdir(parents=True)
    monkeypatch.setattr(sys, "path", [str(own), *sys.path])
    module = f"pathed_{uuid.uuid4().hex[:8]}"
    layouts = []
    for name in ("a", "b"):
        tree = tmp_path / name
        tree.mkdir()
        (tree / f"{module}.py").write_text(
            "def build():\n    return 1\n", encoding="utf-8"
        )
        layouts.append(
            _layout(f"{module}:build", [str(tree / f"{module}.py")], tree)
        )
        import_source_builder(f"{module}:build", layouts[-1])

    assert sys.path[0] == str(layouts[1].root)
    assert str(layouts[0].root) not in sys.path
    assert str(own) in sys.path


def test_a_plan_imports_the_declared_helper_never_one_beside_the_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan imports the declared helper, never an undeclared one beside the builder.

    A builder from another tree importing ``helper`` imports the project's declared ``helper``,
    as the run's snapshot lays them out, never an undeclared one beside the builder in its own
    tree.
    """
    import uuid

    from tcip_mcp.pipelines.model_build import import_source_builder

    monkeypatch.setattr(sys, "path", list(sys.path))
    tag = uuid.uuid4().hex[:8]
    builder, helper = f"external_{tag}", f"helper_{tag}"
    external, project = tmp_path / "external", tmp_path / "project"
    external.mkdir()
    project.mkdir()
    (external / f"{builder}.py").write_text(
        f"import {helper}\n\n\ndef build():\n    return {helper}.VALUE\n",
        encoding="utf-8",
    )
    (external / f"{helper}.py").write_text("VALUE = 11\n", encoding="utf-8")
    (project / f"{helper}.py").write_text("VALUE = 22\n", encoding="utf-8")
    plan = _layout(
        f"{builder}:build",
        [str(external / f"{builder}.py"), str(project / f"{helper}.py")],
        project,
    )

    assert import_source_builder(f"{builder}:build", plan)() == 22


def _resaved(
    project: Path, checkpoint: str, source_files: list[str] | None, name: str
) -> str:
    """Save the checkpoint at ``checkpoint`` again under ``project`` and return its new path.

    Its config's ``model_source.source_files`` is replaced by ``source_files``, and the copy is
    registered under ``name`` (``register_model``).
    """
    import torch

    from tcip_mcp.pipelines.model_build import CONFIG_KEY
    from tests._verified_checkpoint_fixtures import register_checkpoint

    payload = torch.load(checkpoint, weights_only=False)
    payload[CONFIG_KEY]["model_source"]["source_files"] = source_files
    path = project / "elsewhere" / f"{name}.pt"
    path.parent.mkdir(exist_ok=True)
    torch.save(payload, path)
    register_checkpoint(project, str(path), name=name)
    return str(path)


def test_a_checkpoints_layout_is_its_runs_record_whatever_its_files_are_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint's layout is its run's recorded layout, with a declared data file beside it.

    A checkpoint of a run whose packaged builder reads a declared data file named ``run.json``
    beside it loads with its run's recorded layout, and its predictor builds with the settings
    that file states.
    """
    pytest.importorskip("torch")
    from tcip_mcp.experiments import observe
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tests._chain_fixtures import BESPOKE_MODELS
    from tests._verified_checkpoint_fixtures import finished_run

    monkeypatch.setattr(sys, "path", list(sys.path))
    project, files = _packaged_builder(tmp_path, "agentpkg_owned")
    (project / "agentpkg_owned" / "model.py").write_text(
        SETTINGS_BUILDER, encoding="utf-8"
    )
    settings = project / "agentpkg_owned" / "run.json"
    settings.write_text('{"init_weight": 0.5}', encoding="utf-8")
    observation = observe(
        finished_run(
            project,
            model_source={
                "builder": "agentpkg_owned.model:build",
                "task": "classification",
                "normalization": FRAME_STATISTICS,
                "source_files": [*files, str(settings), BESPOKE_MODELS],
            },
        )
    )
    assert observation.checkpoint is not None, observation.final
    loaded = load_registered_checkpoint(
        observation.checkpoint.path, project=project
    )

    assert loaded.layout == observed_run(observation).layout
    assert GenericPredictor(loaded, device="cpu").model.weight.item() == 0.5


MARKED_BUILDER = """
from tests.tiny_trainer_fixtures import build_mean_intensity_classifier

MARK = {mark}


def build(**kwargs):
    return build_mean_intensity_classifier(**kwargs)
"""
"""A builder module's source stating ``MARK``, filled in per copy."""


def test_a_checkpoint_moved_alone_beside_a_run_of_its_paths_but_not_its_sources_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint copied alone beside a run of its paths but not its sources refuses to build.

    Projects A and B each hold a run of one name declaring one builder file name, the two
    builders' bytes differing. A's checkpoint builds in A; its bytes copied alone into B and
    registered there refuse at the predictor with an error naming ``launch_training``.
    """
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tests._chain_fixtures import BESPOKE_MODELS
    from tests._verified_checkpoint_fixtures import (
        register_checkpoint,
        registered_checkpoint,
    )

    monkeypatch.setattr(sys, "path", list(sys.path))
    checkpoints = {}
    for name, mark in (("a", 11), ("b", 22)):
        project = tmp_path / name
        project.mkdir()
        (project / "marked.py").write_text(
            MARKED_BUILDER.format(mark=mark), encoding="utf-8"
        )
        checkpoints[name] = Path(
            registered_checkpoint(
                project,
                experiment_id="one_run_name",
                model_source={
                    "builder": "marked:build",
                    "task": "classification",
                    "normalization": FRAME_STATISTICS,
                    "source_files": [
                        str(project / "marked.py"),
                        BESPOKE_MODELS,
                    ],
                },
            )
        )
    GenericPredictor(
        load_registered_checkpoint(checkpoints["a"], project=tmp_path / "a"),
        device="cpu",
    )

    moved = tmp_path / "b" / "elsewhere" / "from_a.pt"
    moved.parent.mkdir()
    moved.write_bytes(checkpoints["a"].read_bytes())
    register_checkpoint(tmp_path / "b", str(moved), name="from-a")

    with pytest.raises(ValueError, match="launch_training"):
        GenericPredictor(
            load_registered_checkpoint(moved, project=tmp_path / "b"),
            device="cpu",
        )


def test_a_checkpoint_registered_by_name_builds_beside_its_run_and_where_the_run_is_carried(
    tmp_path: Path,
) -> None:
    """A checkpoint registered by name builds beside its run and where the run is carried.

    A checkpoint registered by name (``foreign_checkpoint``) builds in the project holding the
    run that saved it, and again in a fresh project after ``archive_project`` then
    ``import_project`` carries that project there whole.
    """
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tcip_mcp.tools.project_tools import (
        ZipBundle,
        archive_project,
        import_project,
        initialize_project,
    )
    from tests._verified_checkpoint_fixtures import foreign_checkpoint
    from tests._web_fixtures import INVARIANCES

    source, carried = tmp_path / "source", tmp_path / "carried"
    assert "error" not in initialize_project(
        str(source), "Source project", "north orchard", INVARIANCES
    )
    checkpoint = Path(foreign_checkpoint(source))
    loaded = load_registered_checkpoint(checkpoint, project=source)
    assert loaded.experiment_id is None
    GenericPredictor(loaded, device="cpu")

    archive = tmp_path / "source.zip"
    assert "error" not in archive_project(
        source, ZipBundle(output_path=str(archive)), include_models=True
    )
    assert "error" not in import_project(str(archive), str(carried))
    GenericPredictor(
        load_registered_checkpoint(
            carried / checkpoint.relative_to(source), project=carried
        ),
        device="cpu",
    )


def test_a_checkpoint_whose_source_files_lie_under_no_run_refuses_to_build(
    tmp_path: Path,
) -> None:
    """A checkpoint whose source files lie under no run refuses to build.

    A produced checkpoint saved again naming copies of its declared files laid out in a
    directory of the project no run holds (a mutated location) refuses at its predictor, naming
    how a checkpoint comes to have a run.
    """
    import shutil

    from tcip_mcp.experiments import observe
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tcip_mcp.pipelines.model_build import SNAPSHOT_DIR
    from tests._verified_checkpoint_fixtures import finished_run

    observation = observe(finished_run(tmp_path))
    assert observation.checkpoint is not None, observation.final
    snapshot, loose = observation.directory / SNAPSHOT_DIR, tmp_path / "loose"
    shutil.copytree(snapshot, loose)
    produced = load_registered_checkpoint(
        observation.checkpoint.path, project=tmp_path
    )
    relocated = _resaved(
        tmp_path,
        observation.checkpoint.path,
        [
            str(loose / Path(file).relative_to(snapshot))
            for file in produced.spec.model_source.source_files or []
        ],
        "loose",
    )

    with pytest.raises(ValueError, match="launch_training"):
        GenericPredictor(
            load_registered_checkpoint(relocated, project=tmp_path),
            device="cpu",
        )


@pytest.mark.parametrize("source_files", [[], None])
def test_a_checkpoint_declaring_no_source_file_has_no_run_and_refuses(
    tmp_path: Path, source_files: list[str] | None
) -> None:
    """A checkpoint declaring no source file has no run, and its predictor refuses.

    A produced checkpoint saved again declaring no source file (a mutated declaration) has no
    owning run, so its predictor refuses naming how a checkpoint comes to have one.
    """
    from tcip_mcp.experiments import observe
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tests._verified_checkpoint_fixtures import finished_run

    observation = observe(finished_run(tmp_path))
    assert observation.checkpoint is not None, observation.final
    path = _resaved(
        tmp_path, observation.checkpoint.path, source_files, "undeclared"
    )

    with pytest.raises(ValueError, match="launch_training"):
        GenericPredictor(
            load_registered_checkpoint(path, project=tmp_path), device="cpu"
        )


HELPER_SHAPES = {
    "module": ("{helper}.py", "import {helper} as part"),
    "namespace package": ("{helper}/part.py", "from {helper} import part"),
}
"""A declared helper as a top-level module and as a module of a namespace package (a directory
with no ``__init__.py``): the file a tree holds it in and the line a builder imports it by."""


def _trees_with_helper(
    tmp_path: Path, shape: str, values: dict[str, int | None]
) -> tuple[str, str, dict[str, Path], dict[str, SourceLayout]]:
    """Write a builder module of one fresh name in a tree per ``values`` key.

    Each builder imports a helper of ``shape`` (:data:`HELPER_SHAPES`) whose ``VALUE`` it
    builds; a tree whose value is not ``None`` also holds that helper. Returns the builder, the
    helper's name, its trees, and each tree's layout declaring the files it holds.
    """
    import uuid

    tag = uuid.uuid4().hex[:8]
    builder, helper = f"earlier_{tag}", f"helping_{tag}"
    file, line = (
        spelled.format(helper=helper) for spelled in HELPER_SHAPES[shape]
    )
    trees, layouts = {}, {}
    for name, value in values.items():
        trees[name] = tree = tmp_path / name
        tree.mkdir()
        (tree / f"{builder}.py").write_text(
            f"{line}\n\n\ndef build():\n    return part.VALUE\n",
            encoding="utf-8",
        )
        declared = [str(tree / f"{builder}.py")]
        if value is not None:
            (tree / file).parent.mkdir(exist_ok=True)
            (tree / file).write_text(f"VALUE = {value}\n", encoding="utf-8")
            declared.append(str(tree / file))
        layouts[name] = _layout(f"{builder}:build", declared, tree)
    return f"{builder}:build", helper, trees, layouts


@pytest.mark.parametrize("shape", HELPER_SHAPES)
def test_a_module_an_earlier_layout_loaded_is_not_reachable_from_a_later_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    """A module an earlier layout loaded is not reachable from a later layout.

    Layout A declares its builder and helper; layout B declares only a builder that imports that
    helper when it is imported. B refuses, at the import and at a real preflight of its config,
    though A loaded the helper.
    """
    from tcip_mcp.pipelines.model_build import import_source_builder
    from tcip_mcp.tools.training_tools import preflight_config

    monkeypatch.setattr(sys, "path", list(sys.path))
    builder, helper, trees, layouts = _trees_with_helper(
        tmp_path, shape, {"a": 11, "b": None}
    )
    assert import_source_builder(builder, layouts["a"])() == 11

    with pytest.raises(ModuleNotFoundError, match=helper):
        import_source_builder(builder, layouts["b"])

    import_source_builder(builder, layouts["a"])
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    issues = preflight_config(
        trees["b"],
        training_config(
            trees["b"],
            {
                "builder": builder,
                "task": "classification",
                "normalization": weights_normalization(),
                "source_files": [
                    str(trees["b"] / f"{builder.partition(':')[0]}.py")
                ],
            },
            {"images_dir": str(images_dir)},
        ),
    )["issues"]
    assert any(
        i.startswith("model_source.builder not importable") and helper in i
        for i in issues
    ), issues


@pytest.mark.parametrize("shape", HELPER_SHAPES)
def test_layouts_declaring_one_helper_each_import_their_own_in_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    """Layouts declaring one helper name each import their own helper in turn.

    Layouts A and B each declare their own helper under one name; importing A, B, then A builds
    each layout's own helper every time.
    """
    from tcip_mcp.pipelines.model_build import import_source_builder

    monkeypatch.setattr(sys, "path", list(sys.path))
    builder, _helper, _trees, layouts = _trees_with_helper(
        tmp_path, shape, {"a": 11, "b": 22}
    )

    assert [
        import_source_builder(builder, layouts[name])()
        for name in ("a", "b", "a")
    ] == [11, 22, 11]


def test_a_project_directory_named_like_a_snapshot_imports_only_what_it_declares(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A project directory named like a snapshot imports no undeclared helper beside the builder.

    In a project keeping its sources in a directory of its own named ``model_src``, an
    undeclared helper next to the declared builder refuses at the admission's import.
    """
    import uuid

    from tcip_mcp.pipelines.model_build import (
        SNAPSHOT_DIR,
        import_source_builder,
    )

    monkeypatch.setattr(sys, "path", list(sys.path))
    tag = uuid.uuid4().hex[:8]
    builder, helper = f"netp_{tag}", f"helperp_{tag}"
    project = tmp_path / "project"
    src = project / SNAPSHOT_DIR
    src.mkdir(parents=True)
    (src / f"{builder}.py").write_text(
        f"import {helper}\n\n\ndef build():\n    return {helper}.VALUE\n",
        encoding="utf-8",
    )
    (src / f"{helper}.py").write_text(
        "VALUE = 'undeclared'\n", encoding="utf-8"
    )

    with pytest.raises(ModuleNotFoundError, match=helper):
        import_source_builder(
            f"{builder}:build",
            _layout(f"{builder}:build", [str(src / f"{builder}.py")], project),
        )


def test_the_admissions_import_reads_the_files_its_run_holds(
    tmp_path: Path,
) -> None:
    """The admission's import layout and its run's layout agree and hold the same files.

    With its root and directory set to the run's, the admission's layout equals the layout of
    the run opened from it, and the two roots hold the same ``.py`` files byte for byte, the
    declared helper among them.
    """
    import dataclasses

    from tcip_mcp import experiments
    from tcip_mcp.pipelines.data.split_construction import resolve_run
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tcip_mcp.tools.training_tools import _admitted, open_run
    from tests._verified_checkpoint_fixtures import (
        BUILT_DETECTOR,
        detection_config,
    )

    project = tmp_path / "project"
    pkg = project / "agentpkg_agree"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "helper.py").write_text("SCALE = 2\n", encoding="utf-8")
    config = detection_config(
        tmp_path / "data",
        project=tmp_path,
        model_source={
            **BUILT_DETECTOR,
            "source_files": [
                *BUILT_DETECTOR["source_files"],
                str(pkg / "__init__.py"),
                str(pkg / "helper.py"),
            ],
        },
    )
    spec, sources, issues = _admitted(config, {}, project)
    assert spec is not None and sources is not None and not issues, issues
    run_dir = experiments.experiment_dir("agree", project=project)
    open_run(
        run_dir,
        spec,
        sources,
        resolve_run(spec, sources.layout, project=project).record,
    )

    def laid_out(root: Path) -> dict:
        return {
            p.relative_to(root).as_posix(): p.read_bytes()
            for p in root.rglob("*.py")
        }

    run = observed_run(experiments.observe(run_dir)).layout
    assert (
        dataclasses.replace(sources.layout, root=run.root, directory=None)
        == run
    )
    staged, held = laid_out(sources.layout.root), laid_out(run.root)
    assert staged == held
    assert "agentpkg_agree/helper.py" in staged


def test_a_model_built_from_one_layout_still_answers_after_another_layout_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model built from one layout still answers after another layout imports.

    Layouts A and B each hold a module of one name scaling by 3 and by 5; after B's import, A's
    model answers 6 for 2 and B's answers 10.
    """
    import uuid

    from tcip_mcp.pipelines.model_build import import_source_builder

    monkeypatch.setattr(sys, "path", list(sys.path))
    module = f"answering_{uuid.uuid4().hex[:8]}"
    layouts = {}
    for name in ("a", "b"):
        tree = tmp_path / name
        tree.mkdir()
        (tree / f"{module}.py").write_text(
            f"SCALE = {3 if name == 'a' else 5}\n\n\nclass Model:\n"
            "    def __call__(self, x):\n        return x * SCALE\n\n\n"
            "def build():\n    return Model()\n",
            encoding="utf-8",
        )
        layouts[name] = _layout(
            f"{module}:build", [str(tree / f"{module}.py")], tree
        )

    model_a = import_source_builder(f"{module}:build", layouts["a"])()
    model_b = import_source_builder(f"{module}:build", layouts["b"])()

    assert (model_a(2), model_b(2)) == (6, 10)


def test_a_module_inside_an_undeclared_regular_package_refuses_at_the_plan(
    tmp_path: Path,
) -> None:
    """A module inside an undeclared regular package refuses at the plan, naming the initializer.

    Declaring ``pkgundeclared/child.py`` alone, its ``__init__.py`` beside it, raises
    ``ValueError`` naming ``__init__.py``; declaring the initializer too builds a layout.
    """
    pkg = tmp_path / "pkgundeclared"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "child.py").write_text(
        "def build():\n    return 1\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="__init__.py"):
        _layout("pkgundeclared.child:build", [str(pkg / "child.py")], tmp_path)
    assert _layout(
        "pkgundeclared.child:build",
        [str(pkg / "child.py"), str(pkg / "__init__.py")],
        tmp_path,
    )


def test_a_builder_its_source_files_do_not_hold_refuses_though_it_imports_by_name() -> (
    None
):
    """A builder its source files do not hold refuses, naming source_files, though it imports.

    ``tests.bespoke_models:build_bespoke_detection``, importable by name, with no source file
    declared raises ``ValueError`` naming ``source_files``.
    """
    from tests import REPO_ROOT

    with pytest.raises(ValueError, match="source_files"):
        _layout("tests.bespoke_models:build_bespoke_detection", [], REPO_ROOT)


def test_a_packaged_builder_imports_from_its_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A packaged builder imports from its project root.

    ``pkg.model:build`` at ``project/pkg/model.py`` imports from the layout root holding
    ``pkg``, and the file's own directory is not on ``sys.path``.
    """
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.model_build import import_source_builder

    project, files = _packaged_builder(tmp_path, "agentpkg_import")
    monkeypatch.setattr(sys, "path", list(sys.path))

    layout = _layout("agentpkg_import.model:build", files, project)
    build = import_source_builder("agentpkg_import.model:build", layout)

    assert build().num_classes == 2
    assert sys.path[0] == str(layout.root)
    assert (layout.root / "agentpkg_import" / "model.py").is_file()
    assert str(Path(files[0]).parent) not in sys.path


def test_a_packaged_builder_outside_the_path_launches_and_trains_in_the_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A packaged builder outside the path launches and trains to completion in the worker.

    The run names only the builder and its package's files; it launches in a process of its own
    and completes.
    """
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import launch_training
    from tests._verified_checkpoint_fixtures import run_to_end
    from tests.tiny_trainer_fixtures import write_regression_dataset

    project, files = _packaged_builder(tmp_path, "agentpkg_launch")
    monkeypatch.setattr(
        sys, "path", [p for p in sys.path if p != str(project)]
    )
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard",
        lambda *a, **k: {},
    )
    images_dir, csv_path = write_regression_dataset(
        tmp_path / "ds", intensities=[0.1, 0.9] * 4, values=[0, 1] * 4
    )
    cfg = training_config(
        tmp_path,
        {
            "builder": "agentpkg_launch.model:build",
            "task": "classification",
            "normalization": FRAME_STATISTICS,
            "source_files": files,
        },
        {
            "images_dir": str(images_dir),
            "labels_dir": str(csv_path),
            "split": drawn_split(val_ratio=0.15),
        },
        batch_size=4,
        checkpoint_every_n_epochs=0,
        evaluation=evaluation_block(selection_metric="loss"),
    )

    res = launch_training(tmp_path, cfg, actor=None)

    assert "error" not in res, res
    assert res["pid"] != os.getpid()
    status = run_to_end(tmp_path, res["experiment_id"])
    assert status["state"] == "completed", status


def test_preflight_imports_a_builder_through_its_source_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight imports a builder through its declared source files.

    A config naming its builder's file passes preflight's importability check without the
    directory put on the path by hand.
    """
    from tcip_mcp.tools.training_tools import preflight_config

    src = _builder_dir(tmp_path)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(src)])
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    cfg = training_config(
        tmp_path,
        {
            "builder": PROBE_BUILDER,
            "task": "classification",
            "normalization": weights_normalization(),
            "source_files": [str(src / f"{PROBE_MODULE}.py")],
        },
        {"images_dir": str(images_dir)},
    )

    result = preflight_config(tmp_path, cfg)

    assert not any(PROBE_MODULE in issue for issue in result["issues"]), (
        result["issues"]
    )
