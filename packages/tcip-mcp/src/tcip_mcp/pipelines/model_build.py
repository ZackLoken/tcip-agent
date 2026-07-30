"""``build_model`` — the one indirection between a config/checkpoint and an ``nn.Module``.

The single build path is ``model_source``: import a dotted builder the agent wrote and call it.
No ``exec``; the builder is imported like any module. The CV-scientist agent supplies an arbitrary
architecture (from scratch or importing plain PyTorch blocks) through this seam. The only
model-side contract is the measurement boundary (see ``model_contract``): whatever the model is,
its inference output must be something the platform's library scorers can consume.

``model_source`` schema::

    {"builder": "my_module:build_net",     # required — 'module:function' (or 'module.function')
     "builder_kwargs": {...},              # optional — passed to the builder
     "source_files": [...],                # optional — provenance (snapshot_model_source copies these)
     "task": "detection",                  # optional — measurement/eval routing
     "in_chans": 3}                        # optional — channel-compat check

Pure-stdlib at import time (importlib only); torch imports lazily inside the builder so
MCP-server startup stays fast.
"""

from __future__ import annotations

from typing import Any

MODEL_SOURCE_KEY = "model_source"


def _import_dotted(target: str) -> Any:
    """Resolve a ``'module.path:function'`` (or ``'module.path.function'``) string to the callable."""
    if not isinstance(target, str) or not target:
        raise ValueError(f"builder must be a non-empty 'module:function' string, got {target!r}")
    if ":" in target:
        mod_name, _, attr = target.partition(":")
    else:
        mod_name, _, attr = target.rpartition(".")
    if not mod_name or not attr:
        raise ValueError(f"Invalid dotted builder {target!r}; expected 'module:function'.")

    import importlib

    module = importlib.import_module(mod_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(f"Builder {attr!r} not found in module {mod_name!r}.") from exc


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

    in_chans = _int(ms.get("in_chans", bk.get("in_chans")), 3)
    num_classes = _int(bk.get("num_classes"), 1)

    data = config.get("data") or {}
    if task in ("detection", "instance_seg"):
        try:
            from tcip_mcp.pipelines.data.datasets import _resolve_registry_id_map

            _reg, id_map = _resolve_registry_id_map(
                data.get("labels_dir", ""), data.get("subject"), data.get("attribute"))
            num_classes = len(id_map)
        except Exception:  # noqa: BLE001 — fail open to the head's declared count
            pass

    img_size = 224  # safe non-tiny default (7x7 at stride 32); overridden by the real tile edge below
    tiling = (config.get("data") or {}).get("tiling")
    if task == "detection" and isinstance(tiling, dict) and tiling.get("enabled", True) and tiling.get("tile_size"):
        img_size = _int(tiling.get("tile_size"), img_size)
    return {"in_chans": in_chans, "num_classes": num_classes, "img_size": img_size}


# The checkout's commit can't change within a process — resolve it once (a subprocess per training
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
    # CUDA/driver fingerprint — a run's numerics depend on it; null on a CPU-only or torch-less env.
    try:
        import torch

        env["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
    except Exception:
        env["cuda"] = None
    return env


def snapshot_model_source(config: dict, exp_dir: Any) -> dict | None:
    """Copy a bespoke run's model + training + dataset source into ``<exp>/model_src/`` with sha256 + env.

    Called by the training envelope when ``model_source`` / ``training_source`` / ``data.dataset_source``
    is set. Records the agent-written source files (each source's ``source_files`` + the builder/loop
    module files) so the run is reproducible from importable builders — never ``exec``. Best-effort: a
    missing file is skipped and any failure returns without raising (provenance must not sink a run) —
    but the manifest is self-describing about what it failed to capture (``missing``/``snapshot_errors``)
    rather than silently indistinguishable from a complete one (K12 finding 1). Destination files are
    content-addressed (``<sha256[:8]>/<basename>``), so two distinct source files sharing a basename never
    clobber each other, and the same file reached via two different path spellings dedups to one entry
    rather than two rows claiming the same basename with different hashes (K12 finding 2).
    Returns the manifest, or ``None`` when there is nothing bespoke to snapshot.
    """
    import hashlib
    from pathlib import Path

    from tcip_mcp.utils.atomic_io import atomic_write_json

    model_source = config.get(MODEL_SOURCE_KEY)
    training_source = config.get("training_source")
    dataset_source = (config.get("data") or {}).get("dataset_source")
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
    # Snapshot the agent's training-loop + dataset modules too (best-effort — resolve mod:fn -> file).
    for dotted in (builder, training_source, dataset_builder):
        if isinstance(dotted, str) and dotted:
            mod_name = dotted.partition(":")[0] if ":" in dotted else dotted.rpartition(".")[0]
            try:
                import importlib

                mod_file = getattr(importlib.import_module(mod_name), "__file__", None)
                if mod_file:
                    files.append(mod_file)
                else:
                    snapshot_errors.append(
                        f"{dotted!r} imported but its module has no __file__ (namespace/frozen "
                        "module?) — cannot snapshot its source")
            except Exception as exc:
                snapshot_errors.append(f"could not import {dotted!r}: {exc}")

    dst = Path(exp_dir) / "model_src"
    dst.mkdir(parents=True, exist_ok=True)
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
        key = f"{sha[:8]}/{p.name}"
        dst_file = dst / key
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        dst_file.write_bytes(data)
        entries.append({"file": key, "src": str(p), "sha256": sha, "bytes": len(data)})

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
    atomic_write_json(dst / "manifest.json", manifest)
    return manifest


def stamp_model_ref(payload: dict, config: dict, *, experiment_id: str | None = None) -> dict:
    """Stamp a checkpoint payload with its ``model_source`` reference, kind, and experiment id.

    So a hand-rolled loop's checkpoint (via ``ctx.save_checkpoint``) and the default trainer's are
    both reproducible, kind-routable, and traceable back to the run that produced them. Uses
    ``setdefault`` — an explicit value the caller already put in ``payload`` wins. ``experiment_id``
    is optional: a raw/foreign checkpoint legitimately has none, so it is stamped only when known.
    """
    from tcip_mcp.pipelines.inference.predictor import KIND_TCIP_MODULE

    if config.get(MODEL_SOURCE_KEY):
        payload.setdefault(MODEL_SOURCE_KEY, config[MODEL_SOURCE_KEY])
        payload.setdefault("kind", KIND_TCIP_MODULE)
    eid = experiment_id if experiment_id is not None else config.get("experiment_id")
    if eid is not None:
        payload.setdefault("experiment_id", eid)
    return payload
