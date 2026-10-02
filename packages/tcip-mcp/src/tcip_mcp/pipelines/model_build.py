"""``build_model``: a run config's ``model_source`` to an ``nn.Module``, by importing the dotted
builder it names and calling it.

``model_source`` schema::

    {"builder": "my_module:build_net",     # required, 'module:function' (or 'module.function')
     "builder_kwargs": {...},              # optional, passed to the builder
     "source_files": [...],                # optional: the builder's own files, joined to sys.path
                                           # for the import and snapshotted as provenance
     "task": "detection"}                  # the run's task

The builder is also handed the run's width, count and attributes (:func:`model_dims`), never
stated here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.selection import ClassScope

MODEL_SOURCE_KEY = "model_source"
TRAINING_SOURCE_KEY = "training_source"
DATASET_SOURCE_KEY = "dataset_source"
STATE_DICT_KEY = "model_state_dict"
"""The config keys naming a run's bespoke sources, and the checkpoint key holding its weights."""

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


def resolve_named(name: str, registry: "Mapping[str, Any]", *, kind: str, register: str) -> Any:
    """The entry ``registry`` holds under ``name``, else the target a dotted ``module:factory``
    ``name`` imports. Refuses (``ValueError``) a name that is neither, the registered names and
    ``register`` (the call that adds one) named, and a dotted name that will not import."""
    if name in registry:
        return registry[name]
    if ":" in name or "." in name:
        try:
            return _import_dotted(name)
        except Exception as exc:  # noqa: BLE001 (any import failure is an unresolvable name)
            raise ValueError(f"Could not import {kind} {name!r}: {exc}") from exc
    raise ValueError(
        f"Unknown {kind} {name!r}. Name a registered one ({sorted(registry)}), register one with "
        f"{register}, or pass a dotted 'module:factory' you wrote."
    )


def _import_root(file: Path, module: str) -> Path | None:
    """The directory ``module`` imports from when ``file`` is that module's own source, or
    ``None`` when ``file`` is not it.

    ``mypkg.model`` at ``project/mypkg/model.py`` (or a package ``mypkg`` at
    ``project/mypkg/__init__.py``) imports from ``project``: one directory up per dotted component,
    matched against the file's own path so an unrelated file of the same stem is never taken for it.
    """
    parts = tuple(module.split("."))
    module_path = file.parent if file.name == "__init__.py" else file.with_suffix("")
    if module_path.parts[-len(parts):] != parts:
        return None
    return module_path.parents[len(parts) - 1]


def _make_source_files_importable(source: dict) -> None:
    """Put the import root of the builder's own module on ``sys.path``, ahead of everything else,
    when that module is one of ``source``'s ``source_files``.

    The root is resolved from the dotted ``builder`` and the file's path (:func:`_import_root`), so
    a packaged builder (``mypkg.model:build`` at ``project/mypkg/model.py``) imports from
    ``project`` as a top-level one (``model:build`` at ``project/model.py``) does. A source whose
    files do not hold the builder's module changes nothing, and a root already on the path is not
    added twice.
    """
    import sys

    builder = source.get("builder")
    if not isinstance(builder, str) or not builder:
        return
    module, _attr = _split_dotted(builder)
    if not module:
        return
    for file in source.get("source_files") or []:
        root = _import_root(Path(file).resolve(), module)
        if root is not None and str(root) not in sys.path:
            sys.path.insert(0, str(root))


def import_source_builder(source: dict) -> Any:
    """Resolve a ``model_source`` or ``dataset_source`` mapping's ``builder`` to the callable,
    making its own ``source_files`` importable first.
    """
    if not isinstance(source, dict):
        raise ValueError("a builder source must be a dict carrying 'builder'")
    _make_source_files_importable(source)
    return _import_dotted(source.get("builder"))


def child_pythonpath() -> str:
    """The ``PYTHONPATH`` string that makes this process's extra import path entries importable in
    a child process or Ray worker: every non-empty ``sys.path`` entry, then the existing
    ``PYTHONPATH`` env value appended if set, joined with ``os.pathsep``. A child process appends
    its own leading ``sys.path`` entries first, so this string lands after them.

    Ray workers apply environment-variable expansion to ``env_vars`` values (a ``${NAME}`` or
    ``%NAME%`` pattern inside an entry is substituted or stripped), while the subprocess launch
    path passes the string literally.
    """
    import os
    import sys

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    path_entries = [p for p in sys.path if p]
    if existing_pythonpath:
        path_entries = path_entries + [existing_pythonpath]
    return os.pathsep.join(path_entries)


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


def recorded_model_dims(config: "Mapping[str, Any]") -> dict[str, Any]:
    """:func:`model_dims` over what a run's config records on its data section: its ``scope`` and
    its sizes. Refuses by name a run of a built-in loader's task that records no count its ground
    truth derives (``num_ranks`` for an ordinal run)."""
    from tcip_mcp.pipelines.data.datasets import _DATASET_MAP, stated_sizes
    from tcip_mcp.pipelines.data.selection import ClassScope

    data_cfg = config.get("data") or {}
    dims = model_dims(ClassScope.of(data_cfg), stated_sizes(data_cfg))
    loader = None if data_cfg.get(DATASET_SOURCE_KEY) else _DATASET_MAP.get(run_task(config))
    count = loader.ground_truth_count if loader is not None else None
    if count is not None and count not in dims:
        raise ValueError(
            f"this {run_task(config)} run records no {count} (data.{count}), so the head its "
            "model was built at is unknown. Build from a run this platform trained, whose "
            "admission records it."
        )
    return dims


def run_task(config: "Mapping[str, Any]") -> str:
    """The task a run's config names, ``model_source.task``. Raises ``ValueError`` when it states
    none."""
    task = (config.get(MODEL_SOURCE_KEY) or {}).get("task")
    if not task:
        raise ValueError(
            "this config states no task: name it as model_source.task (detection, "
            "instance_seg, classification, ...), the task the builder's model is for")
    return task


def build_from_model_source(model_source: dict, dims: "Mapping[str, Any]") -> Any:
    """Import the agent's builder and call it with its ``builder_kwargs`` and ``dims``
    (:func:`model_dims`). Only ``builder`` is required to construct the model.

    ``model_source`` is held to :class:`~tcip_mcp.pipelines.schemas.ModelSourceSchema`, so a key
    outside it (an ``in_chans`` among them) refuses by name. A ``builder_kwargs`` naming a
    dimension (:data:`RESERVED_DIMS`) refuses by name, whether or not this run resolved it.
    """
    from tcip_mcp.pipelines.schemas import ModelSourceSchema

    ModelSourceSchema.model_validate(model_source)
    fn = import_source_builder(model_source)
    kwargs = model_source.get("builder_kwargs") or {}
    restated = sorted(set(RESERVED_DIMS) & set(kwargs))
    if restated:
        raise ValueError(
            f"model_source.builder_kwargs restates {restated}: the band count a run reads its "
            f"sources at and the count its ground truth derives are the platform's to hand the "
            f"builder, and a second value for one would build a model the run's own record does "
            f"not describe. Drop {restated} from builder_kwargs."
        )
    return fn(**kwargs, **dims)


def build_model(config: "Mapping[str, Any]", dims: "Mapping[str, Any]") -> Any:
    """Build a model from a run config (a checkpoint's own ``config`` included) via its
    ``model_source`` builder, at ``dims`` (:func:`model_dims`)."""
    model_source = config.get(MODEL_SOURCE_KEY)
    if model_source:
        return build_from_model_source(model_source, dims)
    raise ValueError("Config has no 'model_source'.")


def resolve_contract_dims(config: dict, task: str, dims: "Mapping[str, Any]") -> dict:
    """The dimensions a synthetic smoke batch is shaped at: the width, count and attributes the
    model is built at (``dims``, :func:`model_dims`, a rank count carried as ``num_classes``) and
    an ``img_size``.

    ``img_size`` is the tile edge when detection tiling is on (the real training input), else a
    safe non-tiny fallback that clears typical stride-32 backbones. The count is the one ``dims``
    states.
    """
    count = dims.get("num_classes", dims.get("num_ranks"))
    img_size = 224  # safe non-tiny default (7x7 at stride 32); overridden by the real tile edge below
    tiling = (config.get("data") or {}).get("tiling")
    if task == "detection" and isinstance(tiling, dict) and tiling.get("enabled", True) and tiling.get("tile_size"):
        img_size = int(tiling["tile_size"])
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


def snapshot_model_source(config: dict, run_dir: Path) -> dict | None:
    """Copy a bespoke run's model, training and dataset source into ``<run_dir>/model_src/``,
    each file content-addressed as ``<sha256[:8]>/<basename>``, and return what was copied.

    Covers each source's ``source_files`` and the module files of ``model_source``'s builder,
    ``training_source`` and ``data.dataset_source``'s builder. A missing file is listed under
    ``missing`` and a module that will not import under ``snapshot_errors``, never raised.
    Returns ``None`` when nothing bespoke is named.
    """
    import hashlib

    model_source = config.get(MODEL_SOURCE_KEY)
    training_source = config.get(TRAINING_SOURCE_KEY)
    dataset_source = (config.get("data") or {}).get(DATASET_SOURCE_KEY)
    if not model_source and not training_source and not dataset_source:
        return None

    files: list[str] = []
    builder = None
    if isinstance(model_source, dict):
        builder = model_source.get("builder")
        files.extend(model_source.get("source_files") or [])
    dataset_builder = None
    if isinstance(dataset_source, dict):
        dataset_builder = dataset_source.get("builder")
        files.extend(dataset_source.get("source_files") or [])
    snapshot_errors: list[str] = []
    # Snapshot the agent's training-loop + dataset modules too (best-effort, resolve mod:fn -> file).
    for dotted in (builder, training_source, dataset_builder):
        if isinstance(dotted, str) and dotted:
            mod_name, _ = _split_dotted(dotted)
            try:
                import importlib

                mod_file = getattr(importlib.import_module(mod_name), "__file__", None)
                if mod_file:
                    files.append(mod_file)
                else:
                    snapshot_errors.append(
                        f"{dotted!r} imported but its module has no __file__ (namespace/frozen "
                        "module?), cannot snapshot its source")
            except Exception as exc:
                snapshot_errors.append(f"could not import {dotted!r}: {exc}")

    entries: list[dict] = []
    seen_content: set[str] = set()
    missing: list[str] = []
    for f in files:
        p = Path(f)
        if not p.is_file():
            missing.append(f)
            continue
        data = p.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha in seen_content:
            continue
        seen_content.add(sha)
        destination = run_dir / SNAPSHOT_DIR / sha[:8] / p.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        entries.append({"file": f"{sha[:8]}/{p.name}", "src": str(p),
                        "sha256": sha, "bytes": len(data)})

    return {
        "builder": builder,
        "training_source": training_source,
        "dataset_builder": dataset_builder,
        "declared_files": files,
        "files": entries,
        "missing": missing,
        "snapshot_errors": snapshot_errors,
    }
