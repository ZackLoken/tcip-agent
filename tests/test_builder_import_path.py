"""A bespoke builder that lives outside the interpreter's search path imports through its run's
layout: the declared files, staged until a run holds them, lead ``sys.path`` at the one import
site, no child process inherits that root, and a run naming such a builder launches and trains in
the platform's training worker from its own snapshot.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests._chain_fixtures import training_config

TOP_LEVEL_BUILDER = '''
def build_probe(**kwargs):
    return {"built": True, "kwargs": kwargs}
'''

PACKAGED_BUILDER = '''
from tests.tiny_trainer_fixtures import build_mean_intensity_classifier


def build(**kwargs):
    return build_mean_intensity_classifier(**kwargs)
'''

SETTINGS_BUILDER = '''
import json
from pathlib import Path

from tests.tiny_trainer_fixtures import build_mean_intensity_classifier


def build(**kwargs):
    settings = json.loads(Path(__file__).with_name("run.json").read_text(encoding="utf-8"))
    return build_mean_intensity_classifier(**kwargs, **settings)
'''
"""A packaged builder reading its settings from a declared ``run.json`` beside it."""

PROBE_MODULE = "probe_builder"
PROBE_BUILDER = f"{PROBE_MODULE}:build_probe"
"""The builder :data:`TOP_LEVEL_BUILDER` defines, as a ``model_source`` names it."""


def _builder_dir(tmp_path: Path) -> Path:
    """A directory outside the interpreter's path holding :data:`PROBE_MODULE`; the directory."""
    src = tmp_path / "agent_project" / "models"
    src.mkdir(parents=True)
    (src / f"{PROBE_MODULE}.py").write_text(TOP_LEVEL_BUILDER, encoding="utf-8", newline="\n")
    return src


def _packaged_builder(tmp_path: Path, package: str) -> tuple[Path, list[str]]:
    """``<tmp>/agent_project/<package>/model.py`` holding ``build`` beside its package's
    initializer: the project directory is the import root, one level above the file. The project
    and the two files a run declares."""
    project = tmp_path / "agent_project"
    pkg = project / package
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8", newline="\n")
    model = pkg / "model.py"
    model.write_text(PACKAGED_BUILDER, encoding="utf-8", newline="\n")
    return project, [str(model), str(pkg / "__init__.py")]


def _layout(builder: str, files: list[str], project: Path):
    """The layout the admission of a classification run of ``project`` naming ``builder`` and
    declaring ``files``, over no data, stages (``model_build.staged_sources``)."""
    from tcip_mcp.pipelines.model_build import staged_sources
    from tcip_mcp.pipelines.schemas import train_config

    return staged_sources(train_config(training_config(
        {"builder": builder, "task": "classification", "source_files": files}, {})),
        project).layout


def _child_imports(module: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", f"import {module}; print({module}.__name__)"],
        capture_output=True, text=True, env=env,
    )


def test_a_builder_outside_the_interpreters_path_imports_from_its_layout_no_child_inherits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The builder imports from its layout's root, first on ``sys.path``; a child process
    launched with ``child_pythonpath`` inherits no layout root, however the inherited
    ``PYTHONPATH`` spells it (as placed, with forward slashes, or with its case swapped where the
    platform's paths ignore case), so it binds its own."""
    from tcip_mcp.pipelines.model_build import child_pythonpath, import_source_builder

    src = _builder_dir(tmp_path)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(src)])
    layout = _layout(PROBE_BUILDER, [str(src / f"{PROBE_MODULE}.py")], tmp_path)

    fn = import_source_builder(PROBE_BUILDER, layout)
    assert fn(width=4) == {"built": True, "kwargs": {"width": 4}}
    assert sys.path[0] == str(layout.root)
    assert (layout.root / f"{PROBE_MODULE}.py").is_file()

    inherited = os.environ.get("PYTHONPATH", "")
    spellings = [str(layout.root), layout.root.as_posix()]
    if sys.platform == "win32":
        spellings.append(str(layout.root).swapcase())
    for spelling in spellings:
        monkeypatch.setenv("PYTHONPATH", os.pathsep.join([spelling, inherited]))
        assert spelling not in child_pythonpath().split(os.pathsep)
        child = _child_imports(PROBE_MODULE, {**os.environ, "PYTHONPATH": child_pythonpath()})
        assert child.returncode != 0, spelling
        assert "ModuleNotFoundError" in child.stderr


def test_a_builder_imports_in_a_fresh_interpreter(tmp_path: Path) -> None:
    """The one import site serves a process that has loaded nothing of ``importlib.util`` yet:
    the layout is laid out here, and a fresh child holding only the importer imports and builds
    the declared builder from it."""
    import tcip_mcp

    layout = _layout(PROBE_BUILDER, [str(_builder_dir(tmp_path) / f"{PROBE_MODULE}.py")],
                     tmp_path)
    script = (
        "import sys\n"
        "from pathlib import Path, PurePosixPath\n"
        "from tcip_mcp.pipelines.model_build import SourceLayout, import_source_builder\n"
        "print('importlib.util' in sys.modules)\n"
        "layout = SourceLayout(Path(sys.argv[1]), tuple(map(PurePosixPath, sys.argv[3:])),\n"
        "                      sys.argv[2])\n"
        f"print(import_source_builder({PROBE_BUILDER!r}, layout)(width=7))\n")

    child = subprocess.run(
        [sys.executable, "-c", script, str(layout.root), layout.snapshot,
         *map(str, layout.places)],
        capture_output=True, text=True, cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(Path(tcip_mcp.__file__).parents[1])})

    assert child.returncode == 0, child.stderr
    assert child.stdout.splitlines() == ["False", "{'built': True, 'kwargs': {'width': 7}}"]


def test_a_builder_imports_each_declared_module_from_its_own_tree_however_deep(
        tmp_path, monkeypatch):
    """A builder importing a helper that imports a dependency, all three declared, loaded once
    from one tree and then from another: every module, the dependency the helper reaches
    included, comes from the second tree, whatever order the files are declared in."""
    import uuid

    from tcip_mcp.pipelines.model_build import import_source_builder

    monkeypatch.setattr(sys, "path", list(sys.path))
    tag = uuid.uuid4().hex[:8]
    builder, helper, dependency = (f"{role}_{tag}" for role in ("builder", "helper", "dep"))
    plans = []
    for value in (11, 22):
        tree = tmp_path / f"tree{value}"
        tree.mkdir()
        (tree / f"{dependency}.py").write_text(f"VALUE = {value}\n", encoding="utf-8")
        (tree / f"{helper}.py").write_text(f"import {dependency}\nVALUE = {dependency}.VALUE\n",
                                           encoding="utf-8")
        (tree / f"{builder}.py").write_text(
            f"import {helper}\n\n\ndef build():\n    return {helper}.VALUE\n", encoding="utf-8")
        plans.append(_layout(f"{builder}:build",
                           [str(tree / f"{name}.py") for name in (builder, helper, dependency)],
                           tree))

    assert import_source_builder(f"{builder}:build", plans[0])() == 11
    assert import_source_builder(f"{builder}:build", plans[1])() == 22


def test_a_declared_file_edited_between_two_loads_imports_as_edited(tmp_path, monkeypatch):
    """A declared builder rewritten between two loads in one process builds what the second
    load's file says."""
    import uuid

    from tcip_mcp.pipelines.model_build import import_source_builder

    monkeypatch.setattr(sys, "path", list(sys.path))
    module = f"edited_{uuid.uuid4().hex[:8]}"
    builder = tmp_path / f"{module}.py"
    values = []
    for value in (11, 2222):
        builder.write_text(f"def build():\n    return {value}\n", encoding="utf-8")
        values.append(import_source_builder(
            f"{module}:build", _layout(f"{module}:build", [str(builder)], tmp_path))())

    assert values == [11, 2222]


def test_the_last_plan_imported_leads_the_path_and_no_other_plan_stays_on_it(
        tmp_path, monkeypatch):
    """Plans A, B, A imported in turn leave A's root first on ``sys.path`` and B's root off it,
    so a module B's builder reaches undeclared is never found in A's layout."""
    import uuid

    from tcip_mcp.pipelines.model_build import import_source_builder

    monkeypatch.setattr(sys, "path", list(sys.path))
    module = f"rooted_{uuid.uuid4().hex[:8]}"
    plans = {}
    for name in ("a", "b"):
        tree = tmp_path / name
        tree.mkdir()
        (tree / f"{module}.py").write_text(f"def build():\n    return {name!r}\n",
                                           encoding="utf-8")
        plans[name] = _layout(f"{module}:build", [str(tree / f"{module}.py")], tree)

    roots = {}
    for name in ("a", "b", "a"):
        assert import_source_builder(f"{module}:build", plans[name])() == name
        roots[name] = sys.path[0]

    assert sys.path[0] == roots["a"]
    assert roots["b"] not in sys.path


def test_an_import_takes_off_the_path_only_the_layout_roots_imports_put_there(
        tmp_path, monkeypatch):
    """A layout's import takes every earlier layout root off ``sys.path`` and leaves every entry
    no import put there, a directory of the caller's own named like a snapshot included."""
    import uuid

    from tcip_mcp.pipelines.model_build import SNAPSHOT_DIR, import_source_builder

    own = tmp_path / "own" / SNAPSHOT_DIR
    own.mkdir(parents=True)
    monkeypatch.setattr(sys, "path", [str(own), *sys.path])
    module = f"pathed_{uuid.uuid4().hex[:8]}"
    layouts = []
    for name in ("a", "b"):
        tree = tmp_path / name
        tree.mkdir()
        (tree / f"{module}.py").write_text("def build():\n    return 1\n", encoding="utf-8")
        layouts.append(_layout(f"{module}:build", [str(tree / f"{module}.py")], tree))
        import_source_builder(f"{module}:build", layouts[-1])

    assert sys.path[0] == str(layouts[1].root)
    assert str(layouts[0].root) not in sys.path
    assert str(own) in sys.path


def test_a_plan_imports_the_declared_helper_never_one_beside_the_builder(tmp_path, monkeypatch):
    """A builder from another tree importing ``helper`` imports the project's declared
    ``helper``, as the run's snapshot lays them out, never an undeclared one beside the builder
    in its own tree."""
    import uuid

    from tcip_mcp.pipelines.model_build import import_source_builder

    monkeypatch.setattr(sys, "path", list(sys.path))
    tag = uuid.uuid4().hex[:8]
    builder, helper = f"external_{tag}", f"helper_{tag}"
    external, project = tmp_path / "external", tmp_path / "project"
    external.mkdir()
    project.mkdir()
    (external / f"{builder}.py").write_text(
        f"import {helper}\n\n\ndef build():\n    return {helper}.VALUE\n", encoding="utf-8")
    (external / f"{helper}.py").write_text("VALUE = 11\n", encoding="utf-8")
    (project / f"{helper}.py").write_text("VALUE = 22\n", encoding="utf-8")
    plan = _layout(f"{builder}:build", [str(external / f"{builder}.py"),
                                      str(project / f"{helper}.py")], project)

    assert import_source_builder(f"{builder}:build", plan)() == 22


def _resaved(project: Path, checkpoint: str, source_files: list[str] | None, name: str) -> str:
    """The checkpoint at ``checkpoint`` saved again under ``project`` with its config's
    ``model_source.source_files`` replaced by ``source_files``, registered under ``name``
    (``register_model``); its path."""
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
        tmp_path, monkeypatch):
    """A checkpoint of a run whose packaged builder reads a declared data file named ``run.json``
    beside it imports from its run's snapshot exactly as that run's record places it, rooted
    above the package rather than at the declared files' directory, and its predictor builds."""
    pytest.importorskip("torch")
    from tcip_mcp.experiments import observe
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tests._chain_fixtures import BESPOKE_MODELS
    from tests._verified_checkpoint_fixtures import finished_run

    monkeypatch.setattr(sys, "path", list(sys.path))
    project, files = _packaged_builder(tmp_path, "agentpkg_owned")
    (project / "agentpkg_owned" / "model.py").write_text(SETTINGS_BUILDER, encoding="utf-8")
    settings = project / "agentpkg_owned" / "run.json"
    settings.write_text('{"init_weight": 0.5}', encoding="utf-8")
    observation = observe(finished_run(project, model_source={
        "builder": "agentpkg_owned.model:build", "task": "classification",
        "source_files": [*files, str(settings), BESPOKE_MODELS]}))
    assert observation.checkpoint is not None, observation.final
    loaded = load_registered_checkpoint(observation.checkpoint["path"], project=project)

    assert loaded.layout == observed_run(observation).layout
    assert GenericPredictor(loaded, device="cpu").model.weight.item() == 0.5


MARKED_BUILDER = '''
from tests.tiny_trainer_fixtures import build_mean_intensity_classifier

MARK = {mark}


def build(**kwargs):
    return build_mean_intensity_classifier(**kwargs)
'''
"""A builder whose module states ``MARK``, so two projects' copies of one file name differ."""


def test_a_checkpoint_moved_alone_beside_a_run_of_its_paths_but_not_its_sources_refuses(
        tmp_path, monkeypatch):
    """Projects A and B each hold a run of one name declaring one builder file name, the two
    builders' bytes differing. A's checkpoint builds in A; the same bytes copied alone into B and
    registered there refuse at the predictor, naming how a checkpoint comes to have a run, though
    B's run names the same copies."""
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tests._chain_fixtures import BESPOKE_MODELS
    from tests._verified_checkpoint_fixtures import register_checkpoint, registered_checkpoint

    monkeypatch.setattr(sys, "path", list(sys.path))
    checkpoints = {}
    for name, mark in (("a", 11), ("b", 22)):
        project = tmp_path / name
        project.mkdir()
        (project / "marked.py").write_text(MARKED_BUILDER.format(mark=mark), encoding="utf-8")
        checkpoints[name] = Path(registered_checkpoint(
            project, experiment_id="one_run_name", model_source={
                "builder": "marked:build", "task": "classification",
                "source_files": [str(project / "marked.py"), BESPOKE_MODELS]}))
    GenericPredictor(load_registered_checkpoint(checkpoints["a"], project=tmp_path / "a"),
                     device="cpu")

    moved = tmp_path / "b" / "elsewhere" / "from_a.pt"
    moved.parent.mkdir()
    moved.write_bytes(checkpoints["a"].read_bytes())
    register_checkpoint(tmp_path / "b", str(moved), name="from-a")

    with pytest.raises(ValueError, match="launch_training"):
        GenericPredictor(load_registered_checkpoint(moved, project=tmp_path / "b"),
                         device="cpu")


def test_a_checkpoint_registered_by_name_builds_beside_its_run_and_where_the_run_is_carried(
        tmp_path):
    """A checkpoint registered by name (``foreign_checkpoint``) builds in the project holding
    the run that saved it, and again in a fresh project after ``archive_project`` then
    ``import_project`` carries that project there whole."""
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tcip_mcp.tools.project_tools import archive_project, import_project, initialize_project
    from tests._verified_checkpoint_fixtures import foreign_checkpoint

    source, carried = tmp_path / "source", tmp_path / "carried"
    assert "error" not in initialize_project(str(source), "Source project", "north orchard")
    checkpoint = Path(foreign_checkpoint(source))
    loaded = load_registered_checkpoint(checkpoint, project=source)
    assert loaded.experiment_id is None
    GenericPredictor(loaded, device="cpu")

    archive = tmp_path / "source.zip"
    assert "error" not in archive_project(source, str(archive), include_models=True)
    assert "error" not in import_project(str(archive), str(carried))
    GenericPredictor(load_registered_checkpoint(
        carried / checkpoint.relative_to(source), project=carried), device="cpu")


def test_a_checkpoint_whose_source_files_lie_under_no_run_refuses_to_build(tmp_path):
    """A produced checkpoint saved again naming copies of its declared files laid out in a
    directory of the project no run holds (a mutated location) refuses at its predictor, naming
    how a checkpoint comes to have a run, never importing from the copies' own directory."""
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
    produced = load_registered_checkpoint(observation.checkpoint["path"], project=tmp_path)
    relocated = _resaved(tmp_path, observation.checkpoint["path"], [
        str(loose / Path(file).relative_to(snapshot))
        for file in produced.spec.model_source.source_files or []], "loose")

    with pytest.raises(ValueError, match="launch_training"):
        GenericPredictor(load_registered_checkpoint(relocated, project=tmp_path), device="cpu")


@pytest.mark.parametrize("source_files", [[], None])
def test_a_checkpoint_declaring_no_source_file_has_no_run_and_refuses(tmp_path, source_files):
    """A produced checkpoint saved again declaring no source file (a mutated declaration) has no
    owning run, so its predictor refuses naming how a checkpoint comes to have one."""
    from tcip_mcp.experiments import observe
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tests._verified_checkpoint_fixtures import finished_run

    observation = observe(finished_run(tmp_path))
    assert observation.checkpoint is not None, observation.final
    path = _resaved(tmp_path, observation.checkpoint["path"], source_files, "undeclared")

    with pytest.raises(ValueError, match="launch_training"):
        GenericPredictor(load_registered_checkpoint(path, project=tmp_path), device="cpu")


HELPER_SHAPES = {
    "module": ("{helper}.py", "import {helper} as part"),
    "namespace package": ("{helper}/part.py", "from {helper} import part"),
}
"""A declared helper as a top-level module and as a module of a namespace package (a directory
with no ``__init__.py``): the file a tree holds it in and the line a builder imports it by."""


def _trees_with_helper(tmp_path: Path, shape: str, values: dict[str, int | None]):
    """A builder module of one fresh name in a tree per ``values`` key, importing a helper of
    ``shape`` (:data:`HELPER_SHAPES`) whose ``VALUE`` it builds; a tree whose value is not
    ``None`` also holds that helper. The builder, its trees, and each tree's layout declaring
    the files it holds."""
    import uuid

    tag = uuid.uuid4().hex[:8]
    builder, helper = f"earlier_{tag}", f"helping_{tag}"
    file, line = (spelled.format(helper=helper) for spelled in HELPER_SHAPES[shape])
    trees, layouts = {}, {}
    for name, value in values.items():
        trees[name] = tree = tmp_path / name
        tree.mkdir()
        (tree / f"{builder}.py").write_text(f"{line}\n\n\ndef build():\n    return part.VALUE\n",
                                            encoding="utf-8")
        declared = [str(tree / f"{builder}.py")]
        if value is not None:
            (tree / file).parent.mkdir(exist_ok=True)
            (tree / file).write_text(f"VALUE = {value}\n", encoding="utf-8")
            declared.append(str(tree / file))
        layouts[name] = _layout(f"{builder}:build", declared, tree)
    return f"{builder}:build", helper, trees, layouts


@pytest.mark.parametrize("shape", HELPER_SHAPES)
def test_a_module_an_earlier_layout_loaded_is_not_reachable_from_a_later_one(
        tmp_path, monkeypatch, shape):
    """Layout A declares its builder and helper; layout B declares only a builder that imports
    that helper when it is imported. B refuses, at the import and at a real preflight of its
    config, as its run's snapshot refuses in a fresh process, though A loaded the helper."""
    from tcip_mcp.pipelines.model_build import import_source_builder
    from tcip_mcp.tools.training_tools import preflight_config

    monkeypatch.setattr(sys, "path", list(sys.path))
    builder, helper, trees, layouts = _trees_with_helper(tmp_path, shape, {"a": 11, "b": None})
    assert import_source_builder(builder, layouts["a"])() == 11

    with pytest.raises(ModuleNotFoundError, match=helper):
        import_source_builder(builder, layouts["b"])

    import_source_builder(builder, layouts["a"])
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    issues = preflight_config(trees["b"], training_config(
        {"builder": builder, "task": "classification",
         "source_files": [str(trees["b"] / f"{builder.partition(':')[0]}.py")]},
        {"images_dir": str(images_dir)}))["issues"]
    assert any(i.startswith("model_source.builder not importable") and helper in i
               for i in issues), issues


@pytest.mark.parametrize("shape", HELPER_SHAPES)
def test_layouts_declaring_one_helper_each_import_their_own_in_turn(tmp_path, monkeypatch, shape):
    """Layouts A and B each declare their own helper under one name; importing A, B, then A
    builds each layout's own helper every time."""
    from tcip_mcp.pipelines.model_build import import_source_builder

    monkeypatch.setattr(sys, "path", list(sys.path))
    builder, _helper, _trees, layouts = _trees_with_helper(tmp_path, shape, {"a": 11, "b": 22})

    assert [import_source_builder(builder, layouts[name])()
            for name in ("a", "b", "a")] == [11, 22, 11]


def test_a_project_directory_named_like_a_snapshot_imports_only_what_it_declares(
        tmp_path, monkeypatch):
    """A project keeping its sources in a directory of its own named ``model_src`` imports what
    it declares and nothing beside it: an undeclared helper next to the declared builder refuses
    at the admission's import, as it does from the run's own snapshot."""
    import uuid

    from tcip_mcp.pipelines.model_build import SNAPSHOT_DIR, import_source_builder

    monkeypatch.setattr(sys, "path", list(sys.path))
    tag = uuid.uuid4().hex[:8]
    builder, helper = f"netp_{tag}", f"helperp_{tag}"
    project = tmp_path / "project"
    src = project / SNAPSHOT_DIR
    src.mkdir(parents=True)
    (src / f"{builder}.py").write_text(
        f"import {helper}\n\n\ndef build():\n    return {helper}.VALUE\n", encoding="utf-8")
    (src / f"{helper}.py").write_text("VALUE = 'undeclared'\n", encoding="utf-8")

    with pytest.raises(ModuleNotFoundError, match=helper):
        import_source_builder(f"{builder}:build",
                              _layout(f"{builder}:build", [str(src / f"{builder}.py")], project))


def test_the_admissions_import_reads_the_files_its_run_holds(tmp_path):
    """The layout an admission imports from and the layout of the run opened from that admission
    are one layout, their places and snapshot digest included, under two roots holding the same
    files: what preflight imported is what the run imports."""
    import dataclasses

    from tcip_mcp import experiments
    from tcip_mcp.pipelines.data.split_construction import resolve_run
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tcip_mcp.tools.training_tools import _admitted, open_run
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR, detection_config

    project = tmp_path / "project"
    pkg = project / "agentpkg_agree"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "helper.py").write_text("SCALE = 2\n", encoding="utf-8")
    config = detection_config(tmp_path / "data", model_source={
        **BUILT_DETECTOR, "source_files": [*BUILT_DETECTOR["source_files"],
                                           str(pkg / "__init__.py"), str(pkg / "helper.py")]})
    spec, sources, issues = _admitted(config, {}, project)
    assert spec is not None and sources is not None and not issues, issues
    run_dir = experiments.experiment_dir("agree", project=project)
    open_run(run_dir, spec, sources, resolve_run(spec, sources.layout, project=project).record)

    def laid_out(root: Path) -> dict:
        return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*.py")}

    run = observed_run(experiments.observe(run_dir)).layout
    assert dataclasses.replace(sources.layout, root=run.root, directory=None) == run
    staged, held = laid_out(sources.layout.root), laid_out(run.root)
    assert staged == held
    assert "agentpkg_agree/helper.py" in staged


def test_a_model_built_from_one_layout_still_answers_after_another_layout_imports(
        tmp_path, monkeypatch):
    """A model built from layout A answers after layout B's import evicts A's modules and takes
    A's root off ``sys.path``: what the model's module bound at import stays bound."""
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
            "def build():\n    return Model()\n", encoding="utf-8")
        layouts[name] = _layout(f"{module}:build", [str(tree / f"{module}.py")], tree)

    model_a = import_source_builder(f"{module}:build", layouts["a"])()
    model_b = import_source_builder(f"{module}:build", layouts["b"])()

    assert (model_a(2), model_b(2)) == (6, 10)


def test_a_module_inside_an_undeclared_regular_package_refuses_at_the_plan(tmp_path):
    """Declaring ``pkg/child.py`` while ``pkg/__init__.py`` lies beside it undeclared leaves the
    package the module belongs to unbound, so the plan refuses naming the initializer; declaring
    it admits the run."""
    pkg = tmp_path / "pkgundeclared"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "child.py").write_text("def build():\n    return 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="__init__.py"):
        _layout("pkgundeclared.child:build", [str(pkg / "child.py")], tmp_path)
    assert _layout("pkgundeclared.child:build",
                 [str(pkg / "child.py"), str(pkg / "__init__.py")], tmp_path)


def test_a_builder_its_source_files_do_not_hold_refuses_though_it_imports_by_name():
    """A builder binds only through the file its source declares for its module: one the
    environment imports by name, its module declared nowhere, refuses naming source_files."""
    from tests import REPO_ROOT

    with pytest.raises(ValueError, match="source_files"):
        _layout("tests.bespoke_models:build_bespoke_detection", [], REPO_ROOT)


def test_a_packaged_builder_imports_from_its_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pkg.model:build`` at ``project/pkg/model.py`` imports from ``project``, never from the
    file's own directory, which would make ``pkg`` itself unimportable."""
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run names only the builder and its package's files; the launcher's preflight imports
    it, the worker process builds it, and the run completes."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import launch_training
    from tests._verified_checkpoint_fixtures import run_to_end
    from tests.tiny_trainer_fixtures import write_regression_dataset

    project, files = _packaged_builder(tmp_path, "agentpkg_launch")
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(project)])
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})
    images_dir, csv_path = write_regression_dataset(
        tmp_path / "ds", intensities=[0.1, 0.9] * 4, values=[0, 1] * 4)
    cfg = training_config(
        {"builder": "agentpkg_launch.model:build", "task": "classification",
         "source_files": files},
        {"images_dir": str(images_dir), "labels_dir": str(csv_path),
         "split": {"seed": 0, "val_ratio": 0.15}},
        batch_size=4, stages=[{"freeze_to": 0, "epochs": 1}], checkpoint_every_n_epochs=0)

    res = launch_training(tmp_path, cfg, actor=None)

    assert "error" not in res, res
    assert res["pid"] != os.getpid()
    status = run_to_end(tmp_path, res["experiment_id"])
    assert status["state"] == "completed", status


def test_preflight_imports_a_builder_through_its_source_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one import site serves preflight too: a config naming its builder's file passes the
    importability check without the agent putting the directory on the path by hand."""
    from tcip_mcp.tools.training_tools import preflight_config

    src = _builder_dir(tmp_path)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(src)])
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    cfg = training_config({"builder": PROBE_BUILDER, "task": "classification",
                           "source_files": [str(src / f"{PROBE_MODULE}.py")]},
                          {"images_dir": str(images_dir)})

    result = preflight_config(tmp_path, cfg)

    assert not any(PROBE_MODULE in issue for issue in result["issues"]), result["issues"]
