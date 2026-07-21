"""Data-state doctor: scan a live project for state inconsistencies code audits can't see.

Checks the bug family found in field sessions (2026-07-16): status-store vs disk disagreements
on negatives, registry entries pointing at missing/test-fixture checkpoints, provenance smells,
and orphaned labels. Read-only. Run at session start:

    python scripts/doctor.py <project_root>

Exit codes: 0 clean, 1 warnings only, 2 errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _image_stems(root: Path) -> dict[str, str]:
    """stem -> file name for every image under images/ (any date bucket)."""
    out: dict[str, str] = {}
    images = root / "images"
    if images.is_dir():
        for p in images.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                out[p.stem] = p.name
    return out


def check_negatives(root: Path, findings: list) -> None:
    """A negative is empty labels + human Complete — flag every disk/status disagreement."""
    from tcip_mcp.dataset_layout import (
        normalize_status_store, parse_annotation_dir, status_bucket,
    )

    # Through the same normalizer the web layer reads with, so the two can never disagree about
    # what the store says.
    by_bucket = normalize_status_store(_load(root / ".tcip" / "state" / "image_status.json"))
    stems = _image_stems(root)

    def _negatives_for(label_path: Path) -> set[str]:
        """Negatives confirmed for *this* label file's own campaign.

        A negative in one campaign says nothing about another — that is what scoping means — so
        comparing against every bucket would resurrect the cross-campaign leak as a false alarm.
        """
        parsed = parse_annotation_dir(label_path.parent)
        if parsed is None:
            return set()
        campaign, date, _task = parsed
        bucket = by_bucket.get(status_bucket(campaign, date), {})
        return {n for n, s in bucket.items() if s == "negative"}

    neg_names = {n for b in by_bucket.values() for n, s in b.items() if s == "negative"}

    for label in (root / "annotations").rglob("*.json") if (root / "annotations").is_dir() else []:
        if ".original" in label.parts:
            continue
        data = _load(label)
        if not isinstance(data, dict) or "objects" not in data:
            continue
        name = stems.get(label.stem, f"{label.stem}.JPG")
        own = _negatives_for(label)  # this campaign's confirmations, not every campaign's
        if data["objects"] == [] and name not in own:
            findings.append(("warn", f"{label.relative_to(root)}: empty label but not a confirmed "
                            "negative for this campaign; excluded from training (delete the file, "
                            "or mark the image Complete while this campaign is selected)"))
        elif data["objects"] and name in own:
            findings.append(("error", f"{label.relative_to(root)}: {len(data['objects'])} objects "
                            "but the status store says 'negative' — contradictory; re-review"))
        if label.stem not in stems:
            findings.append(("warn", f"{label.relative_to(root)}: label has no matching image"))

    # The dominant case under "label a few examples": images with no label record at all. They are
    # excluded from training, so a breeder who labelled 30 of 400 trains on 30 — worth seeing at a
    # glance. Reported as one line, not one per image.
    labelled = {p.stem for p in (root / "annotations").rglob("*.json")
                if ".original" not in p.parts} if (root / "annotations").is_dir() else set()
    unannotated = sorted(set(stems) - labelled)
    if unannotated:
        shown = ", ".join(unannotated[:5]) + ("…" if len(unannotated) > 5 else "")
        findings.append(("info", f"{len(unannotated)} of {len(stems)} image(s) have no label "
                        f"record and are excluded from training ({shown}). Annotate them, or mark "
                        "the genuinely-empty ones Complete to train them as negatives."))

    for name in neg_names:
        stem = Path(name).stem
        det = list((root / "annotations").rglob(f"{stem}.json")) if (root / "annotations").is_dir() else []
        if not det:
            findings.append(("warn", f"status says {name} is negative but no label file exists "
                            "(a confirmed negative should have an empty objects file)"))


def check_registry(root: Path, findings: list) -> None:
    entries = _load(root / ".tcip" / "models" / "registry.json") or []
    for m in entries if isinstance(entries, list) else []:
        ckpt = m.get("checkpoint_path", "")
        if "pytest-of-" in ckpt or "\\Temp\\" in ckpt or "/Temp/" in ckpt:
            findings.append(("error", f"registry entry {m.get('name')!r} points at a test/temp "
                            f"checkpoint: {ckpt}"))
        elif ckpt and not Path(ckpt).is_file():
            findings.append(("error", f"registry entry {m.get('name')!r} checkpoint missing: {ckpt}"))


def check_provenance(root: Path, findings: list) -> None:
    unstamped = 0
    for label in (root / "annotations").rglob("*.json") if (root / "annotations").is_dir() else []:
        if ".original" in label.parts:
            continue
        data = _load(label)
        for o in (data or {}).get("objects", []) if isinstance(data, dict) else []:
            if not isinstance(o, dict):
                continue
            if o.get("accepted_by") and not o.get("created_by"):
                findings.append(("warn", f"{label.relative_to(root)}: object has accepted_by "
                                "without created_by (acceptance without origin)"))
            if not o.get("created_by"):
                unstamped += 1
    if unstamped:
        findings.append(("info", f"{unstamped} GT objects carry no created_by (pre-provenance "
                        "data; fine, but new writes should always stamp)"))


def check_state(root: Path, findings: list) -> None:
    state = root / ".tcip" / "state"
    stems = _image_stems(root)
    shard_dir = state / "review"
    if shard_dir.is_dir():
        for shard in shard_dir.glob("*.json"):
            payload = _load(shard) or {}
            img = payload.get("img_name", shard.stem)
            if Path(img).stem not in stems:
                findings.append(("warn", f"review shard {shard.name} references unknown image {img!r}"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("project_root", help="project directory holding images/ annotations/ .tcip/")
    args = ap.parse_args()
    root = Path(args.project_root)
    if not root.is_dir():
        print(f"error: not a directory: {root}")
        return 2

    findings: list[tuple[str, str]] = []
    for check in (check_negatives, check_registry, check_provenance, check_state):
        check(root, findings)

    rank = {"error": 0, "warn": 1, "info": 2}
    findings.sort(key=lambda f: rank[f[0]])
    for level, msg in findings:
        print(f"[{level.upper():5}] {msg}")
    errors = sum(1 for level, _ in findings if level == "error")
    warns = sum(1 for level, _ in findings if level == "warn")
    print(f"\ndoctor: {errors} error(s), {warns} warning(s), "
          f"{len(findings) - errors - warns} info")
    return 2 if errors else (1 if warns else 0)


if __name__ == "__main__":
    sys.exit(main())
