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


def capture_env() -> dict:
    """Best-effort snapshot of the library versions a run's reproducibility depends on."""
    import sys

    env: dict[str, Any] = {"python": sys.version.split()[0]}
    for name in ("torch", "torchvision", "timm"):
        try:
            env[name] = getattr(__import__(name), "__version__", "unknown")
        except Exception:
            env[name] = None
    return env


def snapshot_model_source(config: dict, exp_dir: Any) -> dict | None:
    """Copy a bespoke run's model + training source into ``<exp>/model_src/`` with sha256 + env.

    Called by the training envelope when ``model_source`` / ``training_source`` is set. Records the
    agent-written source files (``model_source.source_files`` + the ``training_source`` module file)
    so the run is reproducible from an importable builder — never ``exec``. Best-effort: a missing
    file is skipped, and any failure returns without raising (provenance must not sink a run).
    Returns the manifest, or ``None`` when there is nothing bespoke to snapshot.
    """
    import hashlib
    from pathlib import Path

    from tcip_mcp.utils.atomic_io import atomic_write_json

    model_source = config.get(MODEL_SOURCE_KEY)
    training_source = config.get("training_source")
    if not model_source and not training_source:
        return None

    files: list[str] = []
    builder = None
    if isinstance(model_source, dict):
        builder = model_source.get("builder")
        files.extend(model_source.get("source_files") or [])
    # Snapshot the agent's training-loop module too (best-effort — resolve mod:fn -> module file).
    for dotted in (builder, training_source):
        if isinstance(dotted, str) and dotted:
            mod_name = dotted.partition(":")[0] if ":" in dotted else dotted.rpartition(".")[0]
            try:
                import importlib

                mod_file = getattr(importlib.import_module(mod_name), "__file__", None)
                if mod_file:
                    files.append(mod_file)
            except Exception:
                pass

    dst = Path(exp_dir) / "model_src"
    dst.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    seen: set[str] = set()
    for f in files:
        p = Path(f)
        if not p.is_file() or str(p) in seen:
            continue
        seen.add(str(p))
        data = p.read_bytes()
        (dst / p.name).write_bytes(data)
        entries.append({"file": p.name, "src": str(p),
                        "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})

    manifest = {
        "builder": builder,
        "training_source": training_source,
        "files": entries,
        "env": capture_env(),
        "seed": config.get("seed", config.get("training", {}).get("seed")),
    }
    atomic_write_json(dst / "manifest.json", manifest)
    return manifest


def stamp_model_ref(payload: dict, config: dict) -> dict:
    """Stamp a checkpoint payload with its ``model_source`` reference + kind.

    So a hand-rolled loop's checkpoint (via ``ctx.save_checkpoint``) and the default trainer's are
    both reproducible and kind-routable at inference. Uses ``setdefault`` — an explicit value the
    caller already put in ``payload`` wins.
    """
    from tcip_mcp.pipelines.inference.predictor import KIND_TCIP_MODULE

    if config.get(MODEL_SOURCE_KEY):
        payload.setdefault(MODEL_SOURCE_KEY, config[MODEL_SOURCE_KEY])
        payload.setdefault("kind", KIND_TCIP_MODULE)
    return payload
