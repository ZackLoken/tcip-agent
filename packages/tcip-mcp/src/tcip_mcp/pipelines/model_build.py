"""``build_from_model_source``: a validated run config's ``model_source``
(``schemas.ModelSourceSchema``) to an ``nn.Module``, by importing the dotted builder it names and
calling it.

``model_source`` schema::

    {"builder": "my_module:build_net",     # required, 'module:function' (or 'module.function')
     "builder_kwargs": {...},              # optional, passed to the builder
     "source_files": [...],                # the files the run imports, its builder's module among
                                           # them; a run binds its own snapshot copies
     "task": "detection"}                  # required, the run's task

The builder is also handed the run's width, count and attributes (:func:`model_dims`), never
stated here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, NamedTuple

from tcip_store import canonical_path

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.schemas import ModelSourceSchema, TrainConfigSchema

STATE_DICT_KEY = "model_state_dict"
CONFIG_KEY = "config"
METRICS_KEY = "metrics"
SNAPSHOT_KEY = "source_snapshot"
"""A checkpoint's keys holding its weights, the config its model builds from, the metrics it
was selected on and the digest of the source snapshot its run took
(:attr:`SourceLayout.snapshot`)."""

RESERVED_DIMS = ("in_chans", "num_classes", "num_ranks", "attributes")
"""The dimensions the platform hands a model builder (:func:`model_dims`), never its
``builder_kwargs``."""


def _split_dotted(target: str) -> tuple[str, str]:
    """Split ``'module.path:function'`` (or ``'module.path.function'``) into ``(module, attr)``,
    with no resolution or validation of either half."""
    if ":" in target:
        mod_name, _, attr = target.partition(":")
    else:
        mod_name, _, attr = target.rpartition(".")
    return mod_name, attr


def _import_dotted(target: object) -> Any:
    """Resolve a ``'module.path:function'`` (or ``'module.path.function'``) string to the callable;
    a non-string or empty ``builder`` refuses.
    """
    if not isinstance(target, str) or not target:
        raise ValueError(f"builder must be a non-empty 'module:function' string, got {target!r}")
    mod_name, attr = _split_dotted(target)
    if not mod_name or not attr:
        raise ValueError(f"Invalid dotted builder {target!r}; expected 'module:function'.")

    import importlib

    module = importlib.import_module(mod_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(f"Builder {attr!r} not found in module {mod_name!r}.") from exc


def resolve_named(name: str, registry: "Mapping[str, Any]", *, kind: str,
                  register: str | None = None) -> Any:
    """The entry ``registry`` holds under ``name``. Where ``register`` names the call that adds
    an entry to ``registry``, a dotted ``module:factory`` ``name`` imports its target instead.
    Refuses (``ValueError``) a name that is neither, naming the ``kind``, the registered names
    and ``register``, and a dotted name that will not import."""
    if name in registry:
        return registry[name]
    if register is None:
        raise ValueError(f"Unknown {kind} {name!r}. Name one of {sorted(registry)}.")
    if ":" in name or "." in name:
        try:
            return _import_dotted(name)
        except Exception as exc:  # noqa: BLE001 (any import failure is an unresolvable name)
            raise ValueError(f"Could not import {kind} {name!r}: {exc}") from exc
    raise ValueError(
        f"Unknown {kind} {name!r}. Name a registered one ({sorted(registry)}), register one with "
        f"{register}, or pass a dotted 'module:factory' you wrote."
    )


def keyword_parameters(fn: Any) -> tuple[set[str], bool]:
    """The keyword names ``fn``'s signature names (``self`` and variadics excluded), and whether
    it also takes any keyword (``**kwargs``)."""
    import inspect

    params = inspect.signature(fn).parameters.values()
    named = {p.name for p in params
             if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) and p.name != "self"}
    return named, any(p.kind is p.VAR_KEYWORD for p in params)


def _import_root(file: Path, module: str) -> Path | None:
    """The directory ``module`` imports from when ``file`` is that module's own ``.py`` source,
    or ``None`` when ``file`` is not it.

    ``mypkg.model`` at ``project/mypkg/model.py`` (or a package ``mypkg`` at
    ``project/mypkg/__init__.py``) imports from ``project``: one directory up per dotted component,
    matched against the file's own path so an unrelated file of the same stem is never taken for it.
    """
    parts = tuple(module.split("."))
    module_path = file.parent if file.name == "__init__.py" else file.with_suffix("")
    if file.suffix != ".py" or module_path.parts[-len(parts):] != parts:
        return None
    return module_path.parents[len(parts) - 1]


@dataclass(frozen=True)
class SourceLayout:
    """The files a run declares laid out under ``root``, each at its place there (its module
    path), and nothing else: a run's snapshot (:func:`run_layout`) or a layout written for an
    admission no run holds yet (:func:`staged_sources`), whose temporary ``directory`` is removed
    once nothing holds the layout. ``snapshot`` is the digest of the snapshot record it lays out,
    every copy's place and content digest."""

    root: Path
    places: tuple[PurePosixPath, ...]
    snapshot: str
    directory: Any = None


def module_root(dotted: str, source_files: list[str] | None) -> Path:
    """The directory the module ``dotted`` (``'module:function'``) names imports from, off its
    own file among ``source_files`` (:func:`_import_root`). Refuses (``ValueError``) naming both
    when none is that module's file: a module a run imports is a file it declares."""
    module = _split_dotted(dotted)[0]
    for file in source_files or []:
        root = _import_root(Path(file), module)
        if root is not None:
            return root
    raise ValueError(
        f"{dotted!r} imports the module {module!r}, and none of the source_files "
        f"{list(source_files or [])} is that module's own file: name the file it is defined in "
        "under source_files, so the run snapshots it and every checkpoint binds that copy")


def lay_out(directory: Path, copies: Mapping[str, bytes]) -> None:
    """Write each of a snapshot's ``copies`` (:func:`snapshot_model_source`) at its path under
    ``directory``, the layout :func:`run_layout` states there."""
    for name, data in copies.items():
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(data)


class StagedSources(NamedTuple):
    """A run's source snapshot taken once at its admission (:func:`snapshot_model_source`'s
    ``record`` and ``copies``, which its run directory then holds) and those copies laid out
    (:func:`lay_out`) in a temporary directory the ``layout`` holds, for the admission to import
    from before any run holds them."""

    record: dict
    copies: dict[str, bytes]
    layout: SourceLayout


def staged_sources(spec: "TrainConfigSchema", project: Path) -> StagedSources:
    """The :class:`StagedSources` of a run of ``project`` over ``spec``; refuses as
    :func:`snapshot_model_source` does."""
    import dataclasses
    import tempfile

    record, copies = snapshot_model_source(spec, project)
    held = tempfile.TemporaryDirectory(prefix="tcip-source-")
    lay_out(Path(held.name), copies)
    return StagedSources(record, copies, dataclasses.replace(
        run_layout(Path(held.name), record), directory=held))


def run_layout(run_dir: Path, source: dict) -> SourceLayout:
    """The layout a snapshot's copies make under ``run_dir`` (:func:`lay_out`): its
    :data:`SNAPSHOT_DIR`, each copy at the place the snapshot's record ``source``
    (:func:`snapshot_model_source`'s record, a run's own) states, and the digest of every copy's
    place and ``sha256`` that record states."""
    import hashlib
    import json

    copies = sorted((entry["file"], entry["sha256"]) for entry in source["files"].values())
    return SourceLayout(
        run_dir / SNAPSHOT_DIR, tuple(PurePosixPath(file).relative_to(SNAPSHOT_DIR)
                                      for file, _sha256 in copies),
        hashlib.sha256(json.dumps(copies).encode()).hexdigest())


def owning_run_layout(spec: "TrainConfigSchema", snapshot: str | None,
                      project: Path) -> SourceLayout:
    """The layout (:func:`run_layout`) of the run of ``project`` (``experiments.run_dirs``) that
    took the source snapshot whose digest is ``snapshot`` (:attr:`SourceLayout.snapshot`, a
    checkpoint's :data:`SNAPSHOT_KEY`) and whose layout holds every file ``spec`` declares.
    Refuses (``ValueError``) a spec declaring no file, or a snapshot and files no run of
    ``project`` took, naming how a checkpoint comes to have a run."""
    from tcip_mcp.experiments import RUN_FILE, read_run_record, run_dirs

    declared = {Path(file) for _field, _dotted, files in source_seams(spec)
                for file in files or []}
    for run_dir in run_dirs(project):
        layout = run_layout(run_dir, read_run_record(run_dir / RUN_FILE)["source"])
        if declared and layout.snapshot == snapshot and declared <= {
                layout.root / place for place in layout.places}:
            return layout
    raise ValueError(
        f"No run of {project} took {sorted(map(str, declared))} as its source snapshot, so "
        "nothing states the layout a model built from them imports: a checkpoint builds from the "
        "snapshot its run took. Train it through a run of this project (launch_training), or "
        "carry the run that produced it whole (archive_project, then import_project).")


_LAYOUT_ROOTS: set[str] = set()
"""Every layout root this process's imports put on ``sys.path`` (:func:`import_source_builder`),
in its one spelling (``tcip_store.canonical_path``): the next import takes each off, together
with the modules loaded from it, and :func:`child_pythonpath` forwards none, however an entry
spells it."""


def import_source_builder(dotted: str, layout: SourceLayout) -> Any:
    """The callable ``dotted`` (``'module:function'``) names, imported from ``layout``.

    Every module loaded from another layout root still on ``sys.path`` (its file, or for a
    package with no file, any of its ``__path__`` locations, lies under it), every module the
    layout places and every submodule under one is evicted from ``sys.modules``; then the
    layout's root becomes the first ``sys.path`` entry, every entry naming another layout root
    leaves it, and the builder's module is imported by name once. Paths compare in their one
    spelling (``tcip_store.canonical_path``). A module the import reaches is the layout's file,
    or one found past every layout root; an object already built from another layout keeps the
    modules it bound.
    """
    import importlib
    import sys

    root = canonical_path(layout.root)
    left = {gone for gone in map(canonical_path, sys.path) if gone in _LAYOUT_ROOTS - {root}}
    names = {".".join(place.parent.parts if place.name == "__init__.py"
                      else place.with_suffix("").parts) for place in layout.places}
    for loaded, module in list(sys.modules.items()):
        own = getattr(module, "__dict__", {})
        if any(loaded == name or loaded.startswith(f"{name}.") for name in names) or left and any(
                Path(canonical_path(where)).is_relative_to(gone) for gone in left
                for where in (own.get("__file__"), *(own.get("__path__") or ())) if where):
            del sys.modules[loaded]
    _LAYOUT_ROOTS.add(root)
    sys.path[:] = [str(layout.root),
                   *(p for p in sys.path if canonical_path(p) not in _LAYOUT_ROOTS)]
    importlib.invalidate_caches()
    return _import_dotted(dotted)


def child_pythonpath() -> str:
    """The ``PYTHONPATH`` string that makes this process's extra import path entries importable in
    a child process or Ray worker: every non-empty ``sys.path`` entry, then every entry of the
    existing ``PYTHONPATH`` env value, each but one naming a layout root this process's imports
    put on the path (:data:`_LAYOUT_ROOTS`, however the entry spells it, so a child imports from
    the layout it binds itself), joined with ``os.pathsep``. A child process appends its own
    leading ``sys.path`` entries first, so this string lands after them.

    Ray workers apply environment-variable expansion to ``env_vars`` values (a ``${NAME}`` or
    ``%NAME%`` pattern inside an entry is substituted or stripped), while the subprocess launch
    path passes the string literally.
    """
    import os
    import sys

    entries = [*sys.path, *os.environ.get("PYTHONPATH", "").split(os.pathsep)]
    return os.pathsep.join(p for p in entries if p and canonical_path(p) not in _LAYOUT_ROOTS)


def model_dims(scope: "ClassScope", sizes: "Mapping[str, int]") -> dict[str, Any]:
    """The dimensions a run's model is built at, each handed to its builder under its own name.

    ``in_chans`` is the band count the run's sources are read at, ``sizes["num_channels"]``
    (:func:`~tcip_mcp.pipelines.data.datasets.resolve_sizes`). For a run over label documents
    (``scope`` naming a subject), ``num_classes`` is 1, since a ``ClassScope`` isolates one subject,
    and ``attributes``, present when the scope declares any, is its tuple of
    :class:`~tcip_mcp.subject_registry.Attribute` records, one head per record sized by its
    values; a ``sizes`` stating a count beside it refuses by name. Otherwise the ``num_classes``
    or ``num_ranks`` ``sizes`` carries, and none for a run whose ground truth carries no count.
    Refuses by name a ``sizes`` recording no band count, and a document scope the admission did
    not read attributes for (:meth:`~tcip_mcp.pipelines.data.selection.ClassScope.admitted_for`).
    """
    from tcip_mcp.pipelines.data.datasets import GROUND_TRUTH_COUNTS
    from tcip_mcp.pipelines.data.selection import DOCUMENT

    if sizes.get("num_channels") is None:
        raise ValueError(
            "this run records no band count (data.num_channels), so the width its model reads at "
            "is unknown, and building at a guess would feed it something other than what it "
            "trained on. Build from a run this platform trained, or state data.num_channels."
        )
    counts = {name: int(sizes[name]) for name in GROUND_TRUTH_COUNTS
              if sizes.get(name) is not None}
    if scope.subject is None:
        return {"in_chans": int(sizes["num_channels"]), **counts}
    if counts:
        raise ValueError(
            f"this run over label documents records {sorted(counts)}: its head sizes are the "
            "subject its scope isolates and the attributes the registry declares for it, so a "
            f"stated count would be a second one. Drop {sorted(counts)} from data."
        )
    attributes = scope.admitted_for(DOCUMENT, "this run's data.scope").attributes
    return {"in_chans": int(sizes["num_channels"]), "num_classes": 1,
            **({"attributes": attributes} if attributes else {})}


def recorded_model_dims(spec: "TrainConfigSchema") -> dict[str, Any]:
    """:func:`model_dims` over what a validated run config records on its data block: its
    ``scope`` and its sizes. Refuses by name a run of a built-in loader's task that records no
    count its ground truth derives (``num_ranks`` for an ordinal run)."""
    from tcip_mcp.pipelines.data.datasets import builtin_loader, stated_sizes

    data, task = spec.data, spec.model_source.task
    dims = model_dims(data.recorded_scope, stated_sizes(data))
    loader = builtin_loader(task, data.dataset_source)
    count = loader.ground_truth_count if loader is not None else None
    if count is not None and count not in dims:
        raise ValueError(
            f"this {task} run records no {count} (data.{count}), so the head its "
            "model was built at is unknown. Build from a run this platform trained, whose "
            "admission records it."
        )
    return dims


def build_from_model_source(source: "ModelSourceSchema", layout: SourceLayout,
                            dims: "Mapping[str, Any]") -> Any:
    """Import the builder a validated ``model_source`` (``schemas.TrainConfigSchema``'s) names
    from its run's ``layout`` (:func:`import_source_builder`) and call it with its
    ``builder_kwargs`` and ``dims`` (:func:`model_dims`). Refuses (``ValueError``) by name a
    ``builder_kwargs`` naming a dimension (:data:`RESERVED_DIMS`), whether or not this run
    resolved it.
    """
    fn = import_source_builder(source.builder, layout)
    kwargs = source.builder_kwargs or {}
    restated = sorted(set(RESERVED_DIMS) & set(kwargs))
    if restated:
        raise ValueError(
            f"model_source.builder_kwargs restates {restated}: the band count a run reads its "
            f"sources at and the count its ground truth derives are the platform's to hand the "
            f"builder, and a second value for one would build a model the run's own record does "
            f"not describe. Drop {restated} from builder_kwargs."
        )
    return fn(**kwargs, **dims)


def resolve_contract_dims(spec: "TrainConfigSchema", dims: "Mapping[str, Any]") -> dict:
    """The dimensions a synthetic smoke batch of a validated run config is shaped at: the width,
    count and attributes the model is built at (``dims``, :func:`model_dims`, a rank count
    carried as ``num_classes``) and an ``img_size``.

    ``img_size`` is the tile edge when detection tiling is on (the real training input), else a
    safe non-tiny fallback that clears typical stride-32 backbones. The count is the one ``dims``
    states.
    """
    from tcip_mcp.pipelines.data.datasets import run_tiling

    count = dims.get("num_classes", dims.get("num_ranks"))
    tiler = run_tiling(spec.model_source.task, spec.data.tiling)
    # 224 clears typical stride-32 backbones at 7x7; a tiled run's real tile edge replaces it.
    img_size = (tiler.tile_size if tiler is not None else None) or 224
    return {"in_chans": dims["in_chans"], "img_size": img_size,
            **({} if count is None else {"num_classes": count}),
            **({"attributes": dims["attributes"]} if "attributes" in dims else {})}


# The checkout's commit can't change within a process, resolve it once (a subprocess per training
# run is wasteful and its latency widens audit races between concurrent runs). Sentinel: unset.
_GIT_COMMIT: str | None = ""


def _tcip_git_commit() -> str | None:
    """Best-effort short git commit of the tcip-mcp checkout (``None`` if unavailable), cached."""
    global _GIT_COMMIT
    if _GIT_COMMIT != "":
        return _GIT_COMMIT
    import subprocess
    from pathlib import Path

    try:
        repo = Path(__file__).resolve().parents[4]
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        _GIT_COMMIT = (out.stdout.strip() or None) if out.returncode == 0 else None
    except Exception:
        _GIT_COMMIT = None
    return _GIT_COMMIT


def capture_env() -> dict:
    """The platform git commit, the Python and ML library versions and the CUDA version this
    process runs, each ``None`` when it cannot be resolved."""
    import sys

    env: dict[str, Any] = {"python": sys.version.split()[0], "tcip_git_commit": _tcip_git_commit()}
    for name in ("torch", "torchvision", "timm", "numpy"):
        try:
            env[name] = getattr(__import__(name), "__version__", "unknown")
        except Exception:
            env[name] = None
    # CUDA/driver fingerprint, a run's numerics depend on it; null on a CPU-only or torch-less env.
    try:
        import torch

        env["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
    except Exception:
        env["cuda"] = None
    return env


SNAPSHOT_DIR = "model_src"
"""The run-directory subdirectory a bespoke run's copied source files land in."""


def source_seams(spec: "TrainConfigSchema") -> list[tuple[str, str, list[str] | None]]:
    """Each module a run of ``spec`` imports from its declared files: the config field naming
    it, its dotted name, and the files declared for it (a ``training_source`` is one of the
    ``model_source``'s)."""
    dataset_source = spec.data.dataset_source
    return [(field, dotted, files) for field, dotted, files in (
        ("model_source.builder", spec.model_source.builder, spec.model_source.source_files),
        ("training_source", spec.training_source, spec.model_source.source_files),
        ("data.dataset_source.builder", dataset_source and dataset_source.builder,
         dataset_source.source_files if dataset_source else None)) if dotted]


def module_plan(spec: "TrainConfigSchema", project: Path) -> dict[Path, PurePosixPath]:
    """Where each file a run of ``project`` declares (its :func:`source_seams`' files, once)
    lies under the run's one snapshot import root (:data:`SNAPSHOT_DIR`), which is also the
    module path it imports as there: its path under the import root (:func:`module_root`) of the
    first seam whose root holds it, else under ``project``. Its files are laid out there
    (:func:`lay_out`) and imported from nowhere else, so its modules reach each other by these
    paths alone.

    Refuses (``ValueError``) a module none of the declared files is, a file outside ``project``
    under none of those roots, naming it and the roots, one whose top-level name there is a
    standard-library module's, two files on one path, and a file inside a regular package (a
    directory holding ``__init__.py``) whose initializer is not declared, naming it.
    """
    import sys

    seams = source_seams(spec)
    roots = [module_root(dotted, files) for _field, dotted, files in seams]
    plan: dict[Path, PurePosixPath] = {}
    for p in dict.fromkeys(Path(file) for _field, _dotted, files in seams for file in files or []):
        base = next((root for root in (*roots, project) if p.is_relative_to(root)), None)
        if base is None:
            raise ValueError(
                f"{p} lies outside {project} and under none of the declared modules' import "
                f"roots ({', '.join(map(str, dict.fromkeys(roots)))}), so a run's snapshot has "
                "no place for it: declare it inside the project or beside the module that "
                "imports it")
        plan[p] = PurePosixPath(p.relative_to(base).as_posix())
        if (top := plan[p].parts[0].removesuffix(".py")) in sys.stdlib_module_names:
            raise ValueError(f"{p} would import as the standard-library module {top}")
        if list(plan.values()).count(plan[p]) > 1:
            raise ValueError(f"{p} and another declared file would both lie at {plan[p]}: a "
                             "run's snapshot holds one file at each path")
    for p, place in plan.items():
        for depth in range(1, len(place.parts)):
            init = p.parents[len(place.parts) - 1].joinpath(*place.parts[:depth], "__init__.py")
            if init.is_file() and init not in plan:
                raise ValueError(
                    f"{p} imports as a module of the package {'.'.join(place.parts[:depth])}, "
                    f"whose initializer {init} is not declared: declare it in source_files, so "
                    "the run binds the package its module belongs to")
    return plan


def snapshot_model_source(spec: "TrainConfigSchema",
                          project: Path) -> tuple[dict, dict[str, bytes]]:
    """The snapshot of the files a run of ``project`` declares, placed by its
    :func:`module_plan` (which refuses as it does): its record, ``{"files": {...}}``, one entry
    per declared file keyed by its stored path (``registry_paths.stored_path`` against
    ``project``) naming its copy's path in the run directory (``file``, under
    :data:`SNAPSHOT_DIR`), its ``sha256`` and ``bytes``; and each copy's bytes by that path, for
    :func:`lay_out` to write. Refuses a file that cannot be read."""
    import hashlib

    from tcip_mcp.registry_paths import stored_path

    entries: dict[str, dict] = {}
    copies: dict[str, bytes] = {}
    for p, place in module_plan(spec, project).items():
        copy = f"{SNAPSHOT_DIR}/{place}"
        copies[copy] = data = p.read_bytes()
        entries[stored_path(p, project)] = {
            "file": copy, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    return {"files": entries}, copies


def snapshot_bound(spec: "TrainConfigSchema", source: dict, run_dir: Path) -> "TrainConfigSchema":
    """``spec``, a run's config, with each file it declares (:func:`source_seams`) replaced by
    its copy in ``run_dir``'s snapshot (``source``, :func:`snapshot_model_source`'s record of that
    run, which names each copy by the declared file's stored path), so the run, and every
    checkpoint config it writes, names the copies its imports read."""
    from tcip_mcp.experiments import project_of_run
    from tcip_mcp.registry_paths import stored_path

    def bound(files: list[str] | None) -> list[str]:
        return [str(run_dir / source["files"][stored_path(file, project_of_run(run_dir))]["file"])
                for file in files or []]

    model_source = spec.model_source.model_copy(
        update={"source_files": bound(spec.model_source.source_files)})
    dataset_source = spec.data.dataset_source
    data = spec.data if dataset_source is None else spec.data.model_copy(update={
        "dataset_source": dataset_source.model_copy(
            update={"source_files": bound(dataset_source.source_files)})})
    return spec.model_copy(update={"model_source": model_source, "data": data})
