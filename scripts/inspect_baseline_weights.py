"""Print framework / model metadata from the baseline weights.pt.

Tells us what we're dealing with before we decide how to use it.
"""
from __future__ import annotations

import torch

from _paths import vf_root

WEIGHTS = vf_root() / "models" / "baseline" / "weights.pt"


def main() -> None:
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    keys = sorted(ckpt.keys()) if isinstance(ckpt, dict) else None
    print(f"top-level keys: {keys}")
    if isinstance(ckpt, dict):
        for k in ("epoch", "best_fitness", "date", "version", "license", "docs", "train_args", "train_results"):
            if k in ckpt:
                print(f"{k}: {ckpt[k]}")
        model = ckpt.get("model")
        if model is not None:
            print(f"model type: {type(model).__name__}")
            for attr in ("yaml_file", "task", "nc", "names", "args", "stride"):
                val = getattr(model, attr, None)
                if val is not None:
                    print(f"model.{attr}: {val}")


if __name__ == "__main__":
    main()
