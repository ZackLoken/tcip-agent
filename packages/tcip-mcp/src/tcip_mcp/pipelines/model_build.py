"""``build_model``: a run config's ``model_source`` to an ``nn.Module``, by importing the dotted
builder it names and calling it.

``model_source`` schema::

    {"builder": "my_module:build_net",     # required, 'module:function' (or 'module.function')
     "builder_kwargs": {...},              # optional, passed to the builder
     "source_files": [...],                # optional: the builder's own files, joined to sys.path
                                           # for the import and snapshotted as provenance
     "task": "detection"}                  # the run's task

The builder is also handed the run's width and count (:func:`model_dims`), never stated here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.selection import ClassScope

from tcip_store import RECORD_JSON, Key, StoreDescriptor, register_store, store
from tcip_store.file_backend import RootedFileLocator

MODEL_SOURCE_KEY = "model_source"
TRAINING_SOURCE_KEY = "training_source"
DATASET_SOURCE_KEY = "dataset_source"
STATE_DICT_KEY = "model_state_dict"
"""The config keys naming a run's bespoke sources, and the checkpoint key holding its weights."""

RESERVED_DIMS = ("in_chans", "num_classes", "num_ranks")
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


def model_dims(scope: "ClassScope", sizes: "Mapping[str, int]") -> dict[str, int]:
    """The dimensions a run's model is built at, each handed to its builder under its own name.

    ``in_chans`` is the band count the run's sources are read at, ``sizes["num_channels"]``
    (:func:`~tcip_mcp.pipelines.data.datasets.resolve_sizes`). The one count is the one the ground
    truth derives: ``num_classes``, the length of ``scope``'s map, for a scoped run; otherwise the
    ``num_classes`` or ``num_ranks`` ``sizes`` carries, and none for a run whose ground truth
    carries no count. Refuses by name a ``sizes`` recording no band count, and a second count.
    """
    from tcip_mcp.pipelines.data.datasets import GROUND_TRUTH_COUNTS

    if sizes.get("num_channels") is None:
        raise ValueError(
            "this run records no band count (data.num_channels), so the width its model reads at "
            "is unknown, and building at a guess would feed it something other than what it "
            "trained on. Build from a run this platform trained, or state data.num_channels."
        )
    counts = [(name, int(sizes[name])) for name in GROUND_TRUTH_COUNTS
              if sizes.get(name) is not None]
    if scope.id_map:
        counts.append(("num_classes", len(scope.id_map)))
    if len(counts) > 1:
        raise ValueError(
            f"this run records two counts ({counts}): a model has one head size, the class map's "
            "length for a scoped run and the one count its ground truth derives otherwise. Drop "
            "the count the run's ground truth does not derive."
        )
    return {"in_chans": int(sizes["num_channels"]), **dict(counts)}


def recorded_model_dims(config: "Mapping[str, Any]") -> dict[str, int]:
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


def build_from_model_source(model_source: dict, dims: "Mapping[str, int]") -> Any:
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


def build_model(config: "Mapping[str, Any]", dims: "Mapping[str, int]") -> Any:
    """Build a model from a run config (a checkpoint's own ``config`` included) via its
    ``model_source`` builder, at ``dims`` (:func:`model_dims`)."""
    model_source = config.get(MODEL_SOURCE_KEY)
    if model_source:
        return build_from_model_source(model_source, dims)
    raise ValueError("Config has no 'model_source'.")


def resolve_contract_dims(config: dict, task: str, dims: "Mapping[str, int]") -> dict:
    """The dimensions a synthetic smoke batch is shaped at: the width and count the model is built
    at (``dims``, :func:`model_dims`, a rank count carried as ``num_classes``) and an ``img_size``.

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
            **({} if count is None else {"num_classes": count})}


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
    """Best-effort snapshot of the code + library versions a run's reproducibility depends on.

    Records the platform git commit (the decisive code) alongside the ML library versions; each
    field is null rather than fatal when unresolvable, so provenance never sinks a run.
    """
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


_SNAPSHOT_DIR = ("model_src",)
_SNAPSHOT_MANIFEST_SUFFIX = ".json"


@dataclass(frozen=True)
class _SnapshotManifestLocator:
    """Places one experiment's snapshot manifest under that experiment's own directory.

    The store is keyed off the experiments root so every experiment's members share one scope,
    while the file still lands at ``<experiment_id>/model_src/<document>.json``. The generic
    rooted locator cannot spell that: its prefix precedes every part, and here the first part
    precedes the prefix.
    """

    def relative_path(self, scope: str, parts: tuple[str, ...]) -> PurePosixPath:
        experiment_id, document = parts
        return PurePosixPath(
            experiment_id, *_SNAPSHOT_DIR, f"{document}{_SNAPSHOT_MANIFEST_SUFFIX}"
        )

    def parts_from(self, relative_path: PurePosixPath) -> tuple[str, ...] | None:
        segments = relative_path.parts
        if len(segments) != len(_SNAPSHOT_DIR) + 2:
            return None
        if segments[1:-1] != _SNAPSHOT_DIR:
            return None
        if not segments[-1].endswith(_SNAPSHOT_MANIFEST_SUFFIX):
            return None
        document = segments[-1][: -len(_SNAPSHOT_MANIFEST_SUFFIX)]
        if not document:
            return None
        return (segments[0], document)


SNAPSHOT_MANIFEST_STORE = "model_snapshot_manifest"
register_store(
    StoreDescriptor(
        name=SNAPSHOT_MANIFEST_STORE,
        kind="record",
        key_fields=("experiment_id", "document"),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        locator=_SnapshotManifestLocator(),
    )
)

SNAPSHOT_FILE_STORE = "model_snapshot_file"
register_store(
    StoreDescriptor(
        name=SNAPSHOT_FILE_STORE,
        kind="blob",
        key_fields=("content", "filename"),
        frozen=True,
        cannot_carry_field="a source file's raw bytes",
        locator=RootedFileLocator(prefix=_SNAPSHOT_DIR),
    )
)


def snapshot_manifest_key(exp_dir: Path | str) -> Key:
    """What one run's source snapshot claims to hold: the files, the env, what was missed. A
    record, keyed off the directory holding the experiment; ``last_writer_wins``.
    """
    directory = Path(exp_dir).resolve()
    return Key(SNAPSHOT_MANIFEST_STORE, str(directory.parent), (directory.name, "manifest"))


def snapshot_file_key(exp_dir: Path | str, content: str, filename: str) -> Key:
    """One copied source file, addressed by its content and its own name.

    Content-addressed so two distinct files sharing a basename never clobber each other, and
    so the same file reached by two path spellings lands once.
    """
    return Key(SNAPSHOT_FILE_STORE, str(Path(exp_dir).resolve()), (content, filename))


def snapshot_model_source(config: dict, exp_dir: Any) -> dict | None:
    """Copy a bespoke run's model + training + dataset source into ``<exp>/model_src/`` with sha256
    + env.

    Records the agent-written source files (each source's ``source_files`` + the builder/loop
    module files) of ``model_source`` / ``training_source`` / ``data.dataset_source``. Best-effort:
    a missing file is skipped and any failure returns without raising, and the manifest records
    what it failed to capture (``missing``/``snapshot_errors``). Destination files are
    content-addressed (``<sha256[:8]>/<basename>``). Returns the manifest, or ``None`` when there
    is nothing bespoke to snapshot.
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
        store.put_blob(snapshot_file_key(exp_dir, sha[:8], p.name), data)
        entries.append({"file": f"{sha[:8]}/{p.name}", "src": str(p),
                        "sha256": sha, "bytes": len(data)})

    manifest = {
        "builder": builder,
        "training_source": training_source,
        "dataset_builder": dataset_builder,
        "declared_files": files,
        "files": entries,
        "missing": missing,
        "snapshot_errors": snapshot_errors,
        "env": capture_env(),
    }
    store.replace(snapshot_manifest_key(exp_dir), manifest)
    return manifest


def stamp_model_ref(payload: dict, *, experiment_id: str | None = None) -> dict:
    """Stamp a checkpoint payload with its kind and experiment id, read off the run config the
    payload carries under ``config``, the one place its ``model_source`` is recorded.

    Uses ``setdefault``, so an explicit value the caller already put in ``payload`` wins.
    ``experiment_id`` is stamped only when known.

    Refuses to stamp ``kind`` onto a payload with no ``STATE_DICT_KEY``.
    """
    from tcip_mcp.pipelines.inference.predictor import KIND_TCIP_MODULE

    config = payload["config"]
    if config.get(MODEL_SOURCE_KEY):
        if STATE_DICT_KEY not in payload:
            raise ValueError(
                f"stamp_model_ref refuses to stamp 'kind' onto a payload with no "
                f"{STATE_DICT_KEY!r}: a checkpoint sniffed as a loadable tcip module must carry "
                "its weights."
            )
        payload.setdefault("kind", KIND_TCIP_MODULE)
    eid = experiment_id if experiment_id is not None else config.get("experiment_id")
    if eid is not None:
        payload.setdefault("experiment_id", eid)
    return payload
