"""``build_model``, the one indirection between a config/checkpoint and an ``nn.Module``.

The single build path is ``model_source``: import a dotted builder the agent wrote and call it.
No ``exec``; the builder is imported like any module. The CV-scientist agent supplies an arbitrary
architecture (from scratch or importing plain PyTorch blocks) through this seam. The only
model-side contract is the measurement boundary (see ``model_contract``): whatever the model is,
its inference output must be something the platform's library scorers can consume.

``model_source`` schema::

    {"builder": "my_module:build_net",     # required, 'module:function' (or 'module.function')
     "builder_kwargs": {...},              # optional, passed to the builder
     "source_files": [...],                # optional, provenance (snapshot_model_source copies these)
     "task": "detection",                  # optional, measurement/eval routing
     "in_chans": 3}                        # optional, channel-compat check

At import time this reaches no further than the standard library and the storage seam; torch
imports lazily inside the builder so MCP-server startup stays fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from tcip_store import RECORD_JSON, Key, StoreDescriptor, register_store, store
from tcip_store.file_backend import RootedFileLocator

MODEL_SOURCE_KEY = "model_source"
TRAINING_SOURCE_KEY = "training_source"
DATASET_SOURCE_KEY = "dataset_source"
STATE_DICT_KEY = "model_state_dict"
"""The checkpoint keys this platform's own payloads carry: the importable model reference
that rebuilds the module, and the weights that go into it. Both ends of a checkpoint, the
writer and the reader, import these rather than spelling them, so the payload's vocabulary
is stated once."""


def _split_dotted(target: str) -> tuple[str, str]:
    """Split ``'module.path:function'`` (or ``'module.path.function'``) into ``(module, attr)``,
    with no resolution or validation of either half; the one place that grammar is spelled."""
    if ":" in target:
        mod_name, _, attr = target.partition(":")
    else:
        mod_name, _, attr = target.rpartition(".")
    return mod_name, attr


def _import_dotted(target: str) -> Any:
    """Resolve a ``'module.path:function'`` (or ``'module.path.function'``) string to the callable."""
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


def declared_in_chans(model_source: dict | None) -> int | None:
    """The channel count ``model_source`` declares: its own ``in_chans``, falling back to
    ``builder_kwargs.in_chans``. ``None`` when neither declares it (the caller's own default,
    never baked in here), so every reader of this fact (``resolve_contract_dims`` in this module,
    ``generic_trainer._expected_in_chans``, ``GenericPredictor.__init__`` and
    ``training_tools.preflight_config``'s channel firewall) agrees on where it lives.
    """
    if not isinstance(model_source, dict):
        return None
    bk = model_source.get("builder_kwargs")
    bk = bk if isinstance(bk, dict) else {}
    value = model_source.get("in_chans", bk.get("in_chans"))
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_from_model_source(model_source: dict) -> Any:
    """Import the agent's builder and call it. Registry-free; no ``exec``.

    Only ``builder`` is required to construct the model; the rest of the schema is
    provenance / measurement metadata consumed elsewhere (the envelope snapshot, the
    predictor's channel check, eval routing).
    """
    if not isinstance(model_source, dict):
        raise ValueError("model_source must be a dict")
    builder = model_source.get("builder")
    fn = _import_dotted(builder)
    kwargs = model_source.get("builder_kwargs") or {}
    if not isinstance(kwargs, dict):
        raise ValueError("model_source.builder_kwargs must be a dict")
    return fn(**kwargs)


def build_model(config_or_ckpt: dict) -> Any:
    """Build a model from a training-config or checkpoint dict via its ``model_source`` builder."""
    if not isinstance(config_or_ckpt, dict):
        raise ValueError("build_model expects a config/checkpoint dict")
    model_source = config_or_ckpt.get(MODEL_SOURCE_KEY)
    if model_source:
        return build_from_model_source(model_source)
    raise ValueError("Config has no 'model_source'.")


def resolve_contract_dims(config: dict, task: str) -> dict:
    """Resolve the ``(in_chans, num_classes, img_size)`` the smoke contract must forward at.

    Read from the same config the builder reads, never the contract's tiny 64px default: a model
    with a minimum-spatial-size assumption must be smoked at the size it will actually see, or a
    valid model false-fails. ``img_size`` is the tile edge when detection tiling is on (the real
    training input), else a safe non-tiny fallback that clears typical stride-32 backbones.
    ``in_chans`` comes from ``model_source`` / ``builder_kwargs``. ``num_classes`` is reconciled with
    the dataset's ``classes.json``: a detection/instance_seg scope resolves it through the same
    ``assign_class_ids`` map the loader uses (so the smoke forwards at the count that will actually
    train), and fails open to the head's ``builder_kwargs`` count when no registry/subject is in
    scope (a bespoke ``dataset_source`` or a registry-less build). The +1 background offset lives
    only in the loader, never here.
    """
    ms = config.get(MODEL_SOURCE_KEY) or {}
    bk = ms.get("builder_kwargs") if isinstance(ms, dict) else None
    bk = bk if isinstance(bk, dict) else {}

    def _int(value: Any, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    in_chans = declared_in_chans(ms) if isinstance(ms, dict) else None
    in_chans = in_chans if in_chans is not None else 3
    num_classes = _int(bk.get("num_classes"), 1)

    data = config.get("data") or {}
    # Precondition check, not a broad except: a config with no subject in scope (a bespoke
    # dataset_source, or a registry-less build) legitimately fails open to the head's declared
    # count below, that is the one real "no registry in scope" case. A subject that is given but
    # whose read fails for a real reason (corrupted classes.json, an attribute needing a registry
    # that isn't there) must not be silently swallowed into the same fallback.
    if task in ("detection", "instance_seg") and data.get("subject"):
        from tcip_mcp.pipelines.data.datasets import _resolve_registry_id_map

        _reg, id_map = _resolve_registry_id_map(
            data.get("labels_dir", ""), data.get("subject"), data.get("attribute"))
        num_classes = len(id_map)

    img_size = 224  # safe non-tiny default (7x7 at stride 32); overridden by the real tile edge below
    tiling = (config.get("data") or {}).get("tiling")
    if task == "detection" and isinstance(tiling, dict) and tiling.get("enabled", True) and tiling.get("tile_size"):
        img_size = _int(tiling.get("tile_size"), img_size)
    return {"in_chans": in_chans, "num_classes": num_classes, "img_size": img_size}


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
        locator=RootedFileLocator(prefix=_SNAPSHOT_DIR),
    )
)


def snapshot_manifest_key(exp_dir: Path | str) -> Key:
    """What one run's source snapshot claims to hold: the files, the env, what was missed.

    A record and not a blob because it is a document the platform reads back and reasons
    about, while the files it lists are opaque bytes. ``last_writer_wins``: one snapshot pass
    composes the whole manifest and writes it once.

    Keyed off the directory holding the experiment rather than the experiment's own directory,
    so every experiment's records hang off the one experiments-root scope its other members
    already use. The file lands exactly where it always did.
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
    """Copy a bespoke run's model + training + dataset source into ``<exp>/model_src/`` with sha256 + env.

    Called by the training envelope when ``model_source`` / ``training_source`` / ``data.dataset_source``
    is set. Records the agent-written source files (each source's ``source_files`` + the builder/loop
    module files) so the run is reproducible from importable builders, never ``exec``. Best-effort: a
    missing file is skipped and any failure returns without raising (provenance must not sink a run),
    but the manifest is self-describing about what it failed to capture (``missing``/``snapshot_errors``)
    rather than silently indistinguishable from a complete one. Destination files are
    content-addressed (``<sha256[:8]>/<basename>``), so two distinct source files sharing a basename never
    clobber each other, and the same file reached via two different path spellings dedups to one entry
    rather than two rows claiming the same basename with different hashes.
    Returns the manifest, or ``None`` when there is nothing bespoke to snapshot.
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
        "seed": config.get("seed", config.get("training", {}).get("seed")),
    }
    store.replace(snapshot_manifest_key(exp_dir), manifest)
    return manifest


def stamp_model_ref(payload: dict, config: dict, *, experiment_id: str | None = None) -> dict:
    """Stamp a checkpoint payload with its ``model_source`` reference, kind, and experiment id.

    So a hand-rolled loop's checkpoint (via ``ctx.save_checkpoint``) and the default trainer's are
    both reproducible, kind-routable, and traceable back to the run that produced them. Uses
    ``setdefault``, an explicit value the caller already put in ``payload`` wins. ``experiment_id``
    is optional: a raw/foreign checkpoint legitimately has none, so it is stamped only when known.

    Refuses to stamp ``kind``/``model_source`` onto a payload with no ``STATE_DICT_KEY``: that
    stamp is what a predictor sniffs to load the checkpoint's weights, and a payload with none
    would fail at inference with a bare ``KeyError`` naming no contract.
    """
    from tcip_mcp.pipelines.inference.predictor import KIND_TCIP_MODULE

    if config.get(MODEL_SOURCE_KEY):
        if STATE_DICT_KEY not in payload:
            raise ValueError(
                f"stamp_model_ref refuses to stamp {MODEL_SOURCE_KEY!r}/'kind' onto a payload "
                f"with no {STATE_DICT_KEY!r}: a checkpoint sniffed as a loadable tcip module must "
                "carry its weights."
            )
        payload.setdefault(MODEL_SOURCE_KEY, config[MODEL_SOURCE_KEY])
        payload.setdefault("kind", KIND_TCIP_MODULE)
    eid = experiment_id if experiment_id is not None else config.get("experiment_id")
    if eid is not None:
        payload.setdefault("experiment_id", eid)
    return payload
