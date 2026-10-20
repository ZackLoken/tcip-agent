"""``build_from_model_source``: a validated run config's ``model_source`` to an ``nn.Module``.

The ``model_source`` is a ``schemas.ModelSourceSchema``; the module is built by importing the
dotted builder it names and calling it.

``model_source`` schema::

    {
        "builder": "my_module:build_net",  # required, 'module:function' (or 'module.function')
        "builder_kwargs": {...},  # optional, passed to the builder
        "source_files": [
            ...
        ],  # the files the run imports, its builder's module among
        # them; a run binds its own snapshot copies
        "task": "detection",  # required, the run's task
        "normalization": {
            ...
        },  # required, the statistics handed as image_mean/image_std, and any
        # pretrained_weights its source names
        "detector_geometry": {
            ...
        },  # a detector's geometry stated where the run resolves no frame
    }

The builder is also handed the run's width, count and attributes (:func:`model_dims`) and a
detector's geometry its resolution recorded (``schemas.TrainGeometry.detector``), never stated
here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Required,
    TypedDict,
    overload,
)

from tcip_annotation.state import Attribute
from tcip_store import canonical_path

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.schemas import (
        ModelDims,
        ModelSourceSchema,
        TrainConfigSchema,
        TrainGeometry,
    )

STATE_DICT_KEY = "model_state_dict"
CONFIG_KEY = "config"
METRICS_KEY = "metrics"
SNAPSHOT_KEY = "source_snapshot"
GEOMETRY_KEY = "geometry"
"""A checkpoint's keys holding its weights, the config its model builds from, the metrics it
was selected on, the digest of the source snapshot its run took
(:attr:`SourceLayout.snapshot`) and the geometry its run's resolution read off its training
frames (``schemas.TrainGeometry``)."""


def _split_dotted(target: str) -> tuple[str, str]:
    """Split ``'module.path:function'`` (or ``'module.path.function'``) into ``(module, attr)``.

    Neither half is resolved or validated.
    """
    if ":" in target:
        mod_name, _, attr = target.partition(":")
    else:
        mod_name, _, attr = target.rpartition(".")
    return mod_name, attr


def _import_dotted(target: object) -> Callable[..., object]:
    """Resolve a ``'module.path:function'`` (or ``'module.path.function'``) string to the callable.

    A non-string or empty ``target``, and a name whose object is not callable, refuse
    (``ValueError``).
    """
    if not isinstance(target, str) or not target:
        raise ValueError(
            f"builder must be a non-empty 'module:function' string, got {target!r}"
        )
    mod_name, attr = _split_dotted(target)
    if not mod_name or not attr:
        raise ValueError(
            f"Invalid dotted builder {target!r}; expected 'module:function'."
        )

    import importlib

    module = importlib.import_module(mod_name)
    try:
        found: object = getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(
            f"Builder {attr!r} not found in module {mod_name!r}."
        ) from exc
    if not callable(found):
        raise ValueError(
            f"{target!r} names a {type(found).__name__}, which is not callable: name the "
            "function or class in that module that builds it."
        )
    return found


@overload
def resolve_named[T](
    name: str,
    registry: Mapping[str, T],
    *,
    kind: str,
    register: None = None,
) -> T: ...


@overload
def resolve_named[T](
    name: str,
    registry: Mapping[str, T],
    *,
    kind: str,
    register: str,
) -> T | Callable[..., object]: ...


def resolve_named[T](
    name: str,
    registry: Mapping[str, T],
    *,
    kind: str,
    register: str | None = None,
) -> T | Callable[..., object]:
    """Return the entry ``registry`` holds under ``name``.

    Where ``register`` names the call that adds an entry to ``registry``, a dotted
    ``module:factory`` ``name`` imports its callable target instead. Refuses (``ValueError``) a
    name that is neither, naming the ``kind``, the registered names and ``register``, and a dotted
    name that will not import or names nothing callable.
    """
    if name in registry:
        return registry[name]
    if register is None:
        raise ValueError(
            f"Unknown {kind} {name!r}. Name one of {sorted(registry)}."
        )
    if ":" in name or "." in name:
        try:
            return _import_dotted(name)
        except Exception as exc:  # noqa: BLE001 (any import failure is an unresolvable name)
            raise ValueError(
                f"Could not import {kind} {name!r}: {exc}"
            ) from exc
    raise ValueError(
        f"Unknown {kind} {name!r}. Name a registered one ({sorted(registry)}), register one with "
        f"{register}, or pass a dotted 'module:factory' you wrote."
    )


def keyword_parameters(fn: Callable[..., object]) -> tuple[set[str], bool]:
    """Return the keyword names ``fn``'s signature names and whether it takes any keyword.

    ``self`` and variadics are excluded from the names; any keyword is a ``**kwargs``.
    """
    import inspect

    params = inspect.signature(fn).parameters.values()
    named = {
        p.name
        for p in params
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) and p.name != "self"
    }
    return named, any(p.kind is p.VAR_KEYWORD for p in params)


def _import_root(file: Path, module: str) -> Path | None:
    """Return the directory ``module`` imports from when ``file`` is its own ``.py`` source.

    ``None`` when ``file`` is not it. ``mypkg.model`` at ``project/mypkg/model.py`` (or a package
    ``mypkg`` at ``project/mypkg/__init__.py``) imports from ``project``: one directory up per
    dotted component, matched against the file's own path so an unrelated file of the same stem
    is never taken for it.
    """
    parts = tuple(module.split("."))
    module_path = (
        file.parent if file.name == "__init__.py" else file.with_suffix("")
    )
    if file.suffix != ".py" or module_path.parts[-len(parts) :] != parts:
        return None
    return module_path.parents[len(parts) - 1]


@dataclass(frozen=True)
class SourceLayout:
    """The files a run declares laid out under ``root``, each at its place there, nothing else.

    A file's place is its module path. The layout is a run's snapshot (:func:`run_layout`) or a
    layout written for an admission no run holds yet (:func:`staged_sources`), whose temporary
    ``directory`` is removed once nothing holds the layout. ``snapshot`` is the digest of the
    snapshot record it lays out, every copy's place and content digest.
    """

    root: Path
    places: tuple[PurePosixPath, ...]
    snapshot: str
    directory: Any = None


def module_root(dotted: str, source_files: list[str] | None) -> Path:
    """Return the directory the module ``dotted`` (``'module:function'``) names imports from.

    The directory is read off the module's own file among ``source_files``
    (:func:`_import_root`). Refuses (``ValueError``) naming both when none is that module's file: a
    module a run imports is a file it declares.
    """
    module = _split_dotted(dotted)[0]
    for file in source_files or []:
        root = _import_root(Path(file), module)
        if root is not None:
            return root
    raise ValueError(
        f"{dotted!r} imports the module {module!r}, and none of the source_files "
        f"{list(source_files or [])} is that module's own file: name the file it is defined in "
        "under source_files, so the run snapshots it and every checkpoint binds that copy"
    )


def lay_out(directory: Path, copies: Mapping[str, bytes]) -> None:
    """Write each of a snapshot's ``copies`` at its path under ``directory``.

    The copies are :func:`snapshot_model_source`'s; the result is the layout :func:`run_layout`
    states there.
    """
    for name, data in copies.items():
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(data)


class SnapshotCopy(TypedDict):
    """One declared file's copy in a run's source snapshot.

    ``file`` is its path in the run directory (under :data:`SNAPSHOT_DIR`), ``sha256`` its content
    digest and ``bytes`` its size.
    """

    file: str
    sha256: str
    bytes: int


class SourceSnapshot(TypedDict):
    """A run's source snapshot record (:func:`snapshot_model_source`).

    ``files`` holds each declared file's copy, keyed by the file's stored path
    (``registry_paths.stored_path`` against the run's project).
    """

    files: dict[str, SnapshotCopy]


class StagedSources(NamedTuple):
    """A run's source snapshot taken once at its admission, and those copies laid out.

    The snapshot is :func:`snapshot_model_source`'s ``record`` and ``copies``; the copies are laid
    out (:func:`lay_out`) in a temporary directory the ``layout`` holds.
    """

    record: SourceSnapshot
    copies: dict[str, bytes]
    layout: SourceLayout


def staged_sources(spec: TrainConfigSchema, project: Path) -> StagedSources:
    """Stage the :class:`StagedSources` of a run of ``project`` over ``spec``.

    Refuses as :func:`snapshot_model_source` does.
    """
    import dataclasses
    import tempfile

    record, copies = snapshot_model_source(spec, project)
    held = tempfile.TemporaryDirectory(prefix="tcip-source-")
    lay_out(Path(held.name), copies)
    return StagedSources(
        record,
        copies,
        dataclasses.replace(
            run_layout(Path(held.name), record), directory=held
        ),
    )


def run_layout(run_dir: Path, source: SourceSnapshot) -> SourceLayout:
    """Return the layout a snapshot's copies make under ``run_dir`` (:func:`lay_out`).

    It is ``run_dir``'s :data:`SNAPSHOT_DIR`, each copy at the place the snapshot's record
    ``source`` (:func:`snapshot_model_source`'s record, a run's own) states, and the digest of
    every copy's place and ``sha256`` that record states.
    """
    import hashlib
    import json

    copies = sorted(
        (entry["file"], entry["sha256"]) for entry in source["files"].values()
    )
    return SourceLayout(
        run_dir / SNAPSHOT_DIR,
        tuple(
            PurePosixPath(file).relative_to(SNAPSHOT_DIR)
            for file, _sha256 in copies
        ),
        hashlib.sha256(json.dumps(copies).encode()).hexdigest(),
    )


def owning_run_layout(
    spec: TrainConfigSchema, snapshot: str | None, project: Path
) -> SourceLayout:
    """Return the layout (:func:`run_layout`) of the run of ``project`` that took ``snapshot``.

    The runs are ``experiments.run_dirs``'; the one returned took the source snapshot whose digest
    is ``snapshot`` (:attr:`SourceLayout.snapshot`, a checkpoint's :data:`SNAPSHOT_KEY`) and its
    layout holds every file ``spec`` declares. Refuses (``ValueError``) a spec declaring no file,
    or a snapshot and files no run of ``project`` took, naming how a checkpoint comes to have a
    run.
    """
    from tcip_mcp.experiments import read_run, run_dirs

    declared = {
        Path(file)
        for _field, _dotted, files in source_seams(spec)
        for file in files or []
    }
    for run_dir in run_dirs(project):
        layout = run_layout(run_dir, read_run(run_dir).source)
        if (
            declared
            and layout.snapshot == snapshot
            and declared <= {layout.root / place for place in layout.places}
        ):
            return layout
    raise ValueError(
        f"No run of {project} took {sorted(map(str, declared))} as its source snapshot, so "
        "nothing states the layout a model built from them imports: a checkpoint builds from the "
        "snapshot its run took. Train it through a run of this project (launch_training), or "
        "carry the run that produced it whole (archive_project, then import_project)."
    )


_LAYOUT_ROOTS: set[str] = set()
"""Every layout root this process's imports put on ``sys.path``, in its one spelling
(``tcip_store.canonical_path``)."""


def import_source_builder(
    dotted: str, layout: SourceLayout
) -> Callable[..., object]:
    """Import the callable ``dotted`` (``'module:function'``) names from ``layout``.

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
    left = {
        gone
        for gone in map(canonical_path, sys.path)
        if gone in _LAYOUT_ROOTS - {root}
    }
    names = {
        ".".join(
            place.parent.parts
            if place.name == "__init__.py"
            else place.with_suffix("").parts
        )
        for place in layout.places
    }
    for loaded, module in list(sys.modules.items()):
        own = getattr(module, "__dict__", {})
        if (
            any(
                loaded == name or loaded.startswith(f"{name}.")
                for name in names
            )
            or left
            and any(
                Path(canonical_path(where)).is_relative_to(gone)
                for gone in left
                for where in (
                    own.get("__file__"),
                    *(own.get("__path__") or ()),
                )
                if where
            )
        ):
            del sys.modules[loaded]
    _LAYOUT_ROOTS.add(root)
    sys.path[:] = [
        str(layout.root),
        *(p for p in sys.path if canonical_path(p) not in _LAYOUT_ROOTS),
    ]
    importlib.invalidate_caches()
    return _import_dotted(dotted)


def child_pythonpath() -> str:
    """Return the ``PYTHONPATH`` making this process's import path entries importable in a child.

    A child is a child process or Ray worker. The string is every non-empty ``sys.path`` entry,
    then every entry of the existing ``PYTHONPATH`` env value, each but one naming a layout root
    this process's imports put on the path (:data:`_LAYOUT_ROOTS`, however the entry spells it),
    joined with ``os.pathsep``.
    """
    import os
    import sys

    entries = [*sys.path, *os.environ.get("PYTHONPATH", "").split(os.pathsep)]
    return os.pathsep.join(
        p for p in entries if p and canonical_path(p) not in _LAYOUT_ROOTS
    )


def model_dims(scope: ClassScope, sizes: Mapping[str, int]) -> ModelDims:
    """Return the dimensions a run's model is built at (``schemas.ModelDims``).

    ``in_chans`` is the band count the run's sources are read at, ``sizes["num_channels"]``
    (:func:`~tcip_mcp.pipelines.data.datasets.resolve_sizes`). For a run over label documents
    (``scope`` naming a subject), ``num_classes`` is 1 and ``attributes``, present when the scope
    declares any, is its tuple of :class:`~tcip_annotation.state.Attribute` records; a
    ``sizes`` stating a count beside it refuses by name.
    Otherwise the ``num_classes`` or ``num_ranks`` ``sizes`` carries, and none for a run whose
    ground truth carries no count.
    Refuses by name a ``sizes`` recording no band count, and a document scope the admission did
    not read attributes for
    (:meth:`~tcip_mcp.pipelines.data.selection.ClassScope.document_attributes`).
    """
    from tcip_mcp.pipelines.data.datasets import GROUND_TRUTH_COUNTS
    from tcip_mcp.pipelines.schemas import ModelDims

    if sizes.get("num_channels") is None:
        raise ValueError(
            "this run records no band count (data.num_channels), so the width its model reads at "
            "is unknown, and building at a guess would feed it something other than what it "
            "trained on. Build from a run this platform trained, or state data.num_channels."
        )
    dims = ModelDims(in_chans=int(sizes["num_channels"]))
    counts = [
        name for name in GROUND_TRUTH_COUNTS if sizes.get(name) is not None
    ]
    if scope.subject is None:
        for name in counts:
            dims[name] = int(sizes[name])
        return dims
    if counts:
        raise ValueError(
            f"this run over label documents records {sorted(counts)}: its head sizes are the "
            "subject its scope isolates and the attributes the registry declares for it, so a "
            f"stated count would be a second one. Drop {sorted(counts)} from data."
        )
    dims["num_classes"] = 1
    if attributes := scope.document_attributes(
        "this run's data.scope"
    ).attributes:
        dims["attributes"] = attributes
    return dims


def recorded_model_dims(spec: TrainConfigSchema) -> ModelDims:
    """Return :func:`model_dims` over what a validated run config records on its data block.

    That is its ``scope`` and its sizes. Refuses by name a run of a built-in loader's task that
    records no count its ground truth derives (``num_ranks`` for an ordinal run).
    """
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


def build_from_model_source(
    source: ModelSourceSchema,
    layout: SourceLayout,
    dims: ModelDims,
    geometry: TrainGeometry,
) -> torch.nn.Module:
    """Import the builder a validated ``model_source`` names from ``layout`` and call it.

    The ``model_source`` is a ``schemas.TrainConfigSchema``'s; the import is
    :func:`import_source_builder`'s. The builder is called with its ``builder_kwargs``, its
    normalization's values and weights (``schemas.Normalization.handed``), ``dims``
    (:func:`model_dims`) and, for a detector, the geometry its run's resolution recorded
    (``geometry.detector``). Refuses (``TypeError``) a builder that returns anything but a
    ``torch.nn.Module``, naming ``model_source.builder``.
    """
    import torch

    fn = import_source_builder(source.builder, layout)
    detector = {} if geometry.detector is None else geometry.detector.handed()
    model = fn(
        **(source.builder_kwargs or {}),
        **source.normalization.handed(),
        **detector,
        **dims,
    )
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"model_source.builder {source.builder!r} returned a {type(model).__name__}, "
            "not a torch.nn.Module: the builder returns the module it builds."
        )
    return model


class ContractDims(TypedDict, total=False):
    """The dimensions a synthetic smoke batch of a run is shaped at.

    They are :func:`resolve_contract_dims`': ``in_chans``, the band count; ``img_size``, the
    ``(width, height)`` of the frame it trains on; ``num_classes``, its class or rank count;
    ``attributes``, the attribute records its detector heads read. A run has no key for a
    dimension it has none of.
    """

    in_chans: Required[int]
    img_size: tuple[int, int]
    num_classes: int
    attributes: tuple[Attribute, ...]


def resolve_contract_dims(
    spec: TrainConfigSchema,
    dims: ModelDims,
    geometry: TrainGeometry,
) -> ContractDims:
    """Return the dimensions a smoke batch of a run resolved to ``spec`` and ``geometry`` takes.

    The batch is synthetic, shaped at the width, count and attributes the model is built at
    (``dims``, :func:`model_dims`, a rank count carried as ``num_classes``) and the ``img_size``
    of the frame it trains on (``schemas.run_frame``). A run that resolved none carries no
    ``img_size``.
    """
    from tcip_mcp.pipelines.schemas import run_frame

    resolved = ContractDims(in_chans=dims["in_chans"])
    frame = run_frame(
        spec.model_source.task, spec.data.tiling, geometry.native_size
    )
    if frame is not None:
        resolved["img_size"] = frame
    count = dims.get("num_classes", dims.get("num_ranks"))
    if count is not None:
        resolved["num_classes"] = count
    if "attributes" in dims:
        resolved["attributes"] = dims["attributes"]
    return resolved


# The checkout's commit can't change within a process, resolve it once (a subprocess per training
# run is wasteful and its latency widens audit races between concurrent runs). Sentinel: unset.
_GIT_COMMIT: str | None = ""


def _tcip_git_commit() -> str | None:
    """Return the cached short git commit of the tcip-mcp checkout, ``None`` if unavailable."""
    global _GIT_COMMIT
    if _GIT_COMMIT != "":
        return _GIT_COMMIT
    import subprocess
    from pathlib import Path

    try:
        repo = Path(__file__).resolve().parents[4]
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        _GIT_COMMIT = (
            (out.stdout.strip() or None) if out.returncode == 0 else None
        )
    except Exception:
        _GIT_COMMIT = None
    return _GIT_COMMIT


class Environment(TypedDict):
    """The platform git commit and the Python, ML library and CUDA versions a run ran under.

    Each but ``python`` is ``None`` when it cannot be resolved; ``cuda`` is ``None`` on a CPU-only
    or torch-less environment.
    """

    python: str
    tcip_git_commit: str | None
    torch: str | None
    torchvision: str | None
    timm: str | None
    numpy: str | None
    cuda: str | None


def _version(module: str) -> str | None:
    """Return ``module``'s ``__version__``, ``"unknown"`` if it states none, ``None`` unimported."""
    try:
        return getattr(__import__(module), "__version__", "unknown")
    except Exception:
        return None


def capture_env() -> Environment:
    """Return the :class:`Environment` this process runs."""
    import sys

    # CUDA/driver fingerprint, a run's numerics depend on it.
    try:
        import torch

        cuda = torch.version.cuda if torch.cuda.is_available() else None
    except Exception:
        cuda = None
    return Environment(
        python=sys.version.split()[0],
        tcip_git_commit=_tcip_git_commit(),
        torch=_version("torch"),
        torchvision=_version("torchvision"),
        timm=_version("timm"),
        numpy=_version("numpy"),
        cuda=cuda,
    )


def placement(device: str | None) -> torch.device:
    """Return the torch device a model runs on: ``device`` as stated, else ``cuda`` or ``cpu``.

    An unstated ``device`` (``None``) places on ``cuda`` when torch sees one and on ``cpu``
    otherwise. A stated one, empty included, is ``torch.device``'s as given, refused as torch
    refuses it (``RuntimeError``).
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


SNAPSHOT_DIR = "model_src"
"""The run-directory subdirectory a bespoke run's copied source files land in."""


def source_seams(
    spec: TrainConfigSchema,
) -> list[tuple[str, str, list[str] | None]]:
    """Return each module a run of ``spec`` imports from its declared files.

    Each is the config field naming it, its dotted name, and the files declared for it (a
    ``training_source`` is one of the ``model_source``'s).
    """
    dataset_source = spec.data.dataset_source
    return [
        (field, dotted, files)
        for field, dotted, files in (
            (
                "model_source.builder",
                spec.model_source.builder,
                spec.model_source.source_files,
            ),
            (
                "training_source",
                spec.training_source,
                spec.model_source.source_files,
            ),
            (
                "data.dataset_source.builder",
                dataset_source and dataset_source.builder,
                dataset_source.source_files if dataset_source else None,
            ),
        )
        if dotted
    ]


def module_plan(
    spec: TrainConfigSchema, project: Path
) -> dict[Path, PurePosixPath]:
    """Place each file a run of ``project`` declares under the run's one snapshot import root.

    The files are its :func:`source_seams`' files, once; the root is :data:`SNAPSHOT_DIR`. A
    file's place is also the module path it imports as there: its path under the import root
    (:func:`module_root`) of the first seam whose root holds it, else under ``project``.

    Refuses (``ValueError``) a module none of the declared files is, a file outside ``project``
    under none of those roots, naming it and the roots, one whose top-level name there is a
    standard-library module's, two files on one path, and a file inside a regular package (a
    directory holding ``__init__.py``) whose initializer is not declared, naming it.
    """
    import sys

    seams = source_seams(spec)
    roots = [module_root(dotted, files) for _field, dotted, files in seams]
    plan: dict[Path, PurePosixPath] = {}
    for p in dict.fromkeys(
        Path(file) for _field, _dotted, files in seams for file in files or []
    ):
        base = next(
            (root for root in (*roots, project) if p.is_relative_to(root)),
            None,
        )
        if base is None:
            raise ValueError(
                f"{p} lies outside {project} and under none of the declared modules' import "
                f"roots ({', '.join(map(str, dict.fromkeys(roots)))}), so a run's snapshot has "
                "no place for it: declare it inside the project or beside the module that "
                "imports it"
            )
        plan[p] = PurePosixPath(p.relative_to(base).as_posix())
        if (
            top := plan[p].parts[0].removesuffix(".py")
        ) in sys.stdlib_module_names:
            raise ValueError(
                f"{p} would import as the standard-library module {top}"
            )
        if list(plan.values()).count(plan[p]) > 1:
            raise ValueError(
                f"{p} and another declared file would both lie at {plan[p]}: a "
                "run's snapshot holds one file at each path"
            )
    for p, place in plan.items():
        for depth in range(1, len(place.parts)):
            init = p.parents[len(place.parts) - 1].joinpath(
                *place.parts[:depth], "__init__.py"
            )
            if init.is_file() and init not in plan:
                raise ValueError(
                    f"{p} imports as a module of the package {'.'.join(place.parts[:depth])}, "
                    f"whose initializer {init} is not declared: declare it in source_files, so "
                    "the run binds the package its module belongs to"
                )
    return plan


def snapshot_model_source(
    spec: TrainConfigSchema, project: Path
) -> tuple[SourceSnapshot, dict[str, bytes]]:
    """Snapshot the files a run of ``project`` declares, placed by its :func:`module_plan`.

    The plan refuses as it does. Returns the snapshot's record, ``{"files": {...}}``, one entry
    per declared file keyed by its stored path (``registry_paths.stored_path`` against
    ``project``) naming its copy's path in the run directory (``file``, under
    :data:`SNAPSHOT_DIR`), its ``sha256`` and ``bytes``; and each copy's bytes by that path.
    Refuses a file that cannot be read.
    """
    import hashlib

    from tcip_mcp.registry_paths import stored_path

    entries: dict[str, SnapshotCopy] = {}
    copies: dict[str, bytes] = {}
    for p, place in module_plan(spec, project).items():
        copy = f"{SNAPSHOT_DIR}/{place}"
        copies[copy] = data = p.read_bytes()
        entries[stored_path(p, project)] = SnapshotCopy(
            file=copy,
            sha256=hashlib.sha256(data).hexdigest(),
            bytes=len(data),
        )
    return SourceSnapshot(files=entries), copies


def snapshot_bound(
    spec: TrainConfigSchema, source: SourceSnapshot, run_dir: Path
) -> TrainConfigSchema:
    """Return ``spec``, a run's config, with each declared file replaced by its snapshot copy.

    The declared files are :func:`source_seams`'; each copy is in ``run_dir``'s snapshot
    (``source``, :func:`snapshot_model_source`'s record of that run, which names each copy by the
    declared file's stored path).
    """
    from tcip_mcp.experiments import project_of_run
    from tcip_mcp.registry_paths import stored_path

    def bound(files: list[str] | None) -> list[str]:
        return [
            str(
                run_dir
                / source["files"][stored_path(file, project_of_run(run_dir))][
                    "file"
                ]
            )
            for file in files or []
        ]

    model_source = spec.model_source.model_copy(
        update={"source_files": bound(spec.model_source.source_files)}
    )
    dataset_source = spec.data.dataset_source
    data = (
        spec.data
        if dataset_source is None
        else spec.data.model_copy(
            update={
                "dataset_source": dataset_source.model_copy(
                    update={"source_files": bound(dataset_source.source_files)}
                )
            }
        )
    )
    return spec.model_copy(update={"model_source": model_source, "data": data})
