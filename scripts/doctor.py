"""Data-state doctor: scan a live project for state inconsistencies code audits can't see.

Checks the bug family found in field sessions: status-store vs disk disagreements
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
    from tcip_mcp.dataset_layout import image_root

    out: dict[str, str] = {}
    images = image_root(root)
    if images.is_dir():
        for p in images.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                out[p.stem] = p.name
    return out


def check_negatives(root: Path, findings: list) -> None:
    """A negative is empty labels + human Complete, per subject: flag every disk/status disagreement.

    Labels are one name-based file per image (all subjects); a confirmed negative is scoped to a
    subject and date, so the disagreement is checked per subject present in the file.
    """
    from tcip_annotation import json_io
    from tcip_mcp.dataset_layout import (
        annotation_date, annotation_root, bucket_subject_date, image_status_path,
        is_confirmed_negative, normalize_status_store,
    )

    # Confirmations are dataset-native, and this check already assumes root == dataset_root.
    by_bucket = normalize_status_store(_load(image_status_path(root)))
    stems = _image_stems(root)
    ann_root = annotation_root(root)
    neg_names = {n for b in by_bucket.values() for n, s in b.items() if is_confirmed_negative(s)}

    for label in ann_root.rglob("*.json") if ann_root.is_dir() else []:
        if ".original" in label.parts:
            continue
        anns = json_io.read_annotations(str(label))
        name = stems.get(label.stem, f"{label.stem}.JPG")
        date = annotation_date(label)
        subjects_here = {a.subject for a in anns if a.geometry is not None}
        for key, bucket in by_bucket.items():
            if not is_confirmed_negative(bucket.get(name)):
                continue
            subj, bdate = bucket_subject_date(key)
            if bdate != date:
                continue
            if subj in subjects_here:
                findings.append(("error", f"{label.relative_to(root)}: has {subj!r} annotations but "
                                f"the status store says 'negative' for {subj!r}, contradictory; re-review"))
        if not anns and name not in neg_names:
            findings.append(("warn", f"{label.relative_to(root)}: empty label but not a confirmed "
                            "negative for any subject; excluded from training (delete the file, or "
                            "mark the image Complete for the subject it should be a negative of)"))
        if label.stem not in stems:
            findings.append(("warn", f"{label.relative_to(root)}: label has no matching image"))

    # The dominant case under "label a few examples": images with no label record at all. They are
    # excluded from training, so a breeder who labelled 30 of 400 trains on 30; worth seeing at a
    # glance. Reported as one line, not one per image.
    labelled = {p.stem for p in ann_root.rglob("*.json")
                if ".original" not in p.parts} if ann_root.is_dir() else set()
    unannotated = sorted(set(stems) - labelled)
    if unannotated:
        shown = ", ".join(unannotated[:5]) + ("…" if len(unannotated) > 5 else "")
        findings.append(("info", f"{len(unannotated)} of {len(stems)} image(s) have no label "
                        f"record and are excluded from training ({shown}). Annotate them, or mark "
                        "the genuinely-empty ones Complete to train them as negatives."))

    from tcip_mcp.dataset_layout import label_filename

    for name in neg_names:
        stem = Path(name).stem
        det = list(ann_root.rglob(label_filename(stem))) if ann_root.is_dir() else []
        if not det:
            findings.append(("warn", f"status says {name} is negative but no label file exists "
                            "(a confirmed negative should have an empty label file)"))


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
    from tcip_annotation.json_io import ANNOTATIONS_KEY
    from tcip_mcp.dataset_layout import annotation_root

    ann_root = annotation_root(root)
    unstamped = 0
    for label in ann_root.rglob("*.json") if ann_root.is_dir() else []:
        if ".original" in label.parts:
            continue
        data = _load(label)
        for o in (data or {}).get(ANNOTATIONS_KEY, []) if isinstance(data, dict) else []:
            if not isinstance(o, dict):
                continue
            if o.get("accepted_by") and not o.get("created_by"):
                findings.append(("warn", f"{label.relative_to(root)}: annotation has accepted_by "
                                "without created_by (acceptance without origin)"))
            if not o.get("created_by"):
                unstamped += 1
    if unstamped:
        findings.append(("info", f"{unstamped} GT annotations carry no created_by (pre-provenance "
                        "data; fine, but new writes should always stamp)"))

    # Bespoke-run source snapshots: a manifest that failed to capture a declared
    # file is now self-describing rather than silently indistinguishable from a complete one.
    experiments_dir = root / ".tcip" / "experiments"
    if experiments_dir.is_dir():
        for exp_dir in experiments_dir.iterdir():
            manifest_path = exp_dir / "model_src" / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = _load(manifest_path)
            if not isinstance(manifest, dict):
                continue
            missing = manifest.get("missing") or []
            errors = manifest.get("snapshot_errors") or []
            if missing or errors:
                findings.append(("warn", f"{manifest_path.relative_to(root)}: source snapshot "
                                f"incomplete: {len(missing)} missing file(s), "
                                f"{len(errors)} import error(s)"))


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


def check_region_completeness(root: Path, findings: list) -> None:
    """A region-completeness attestation whose cell content has since been edited or deleted is a
    stale claim block calibration could otherwise trust silently; flag every disagreement between
    an attested cell's stamped digest and its current annotation content."""
    from tcip_mcp.dataset_layout import (
        bucket_subject_date, normalize_region_completeness_store,
        region_completeness_digest_path, region_completeness_path,
    )
    from tcip_mcp.pipelines.region_completeness import stale_cells

    store = normalize_region_completeness_store(_load(region_completeness_path(root)))
    if not store:
        return
    digests = _load(region_completeness_digest_path(root))
    if not isinstance(digests, dict):
        digests = {}
    for bucket, record in store.items():
        subject = record.get("subject")
        if not isinstance(subject, str) or not subject:
            subject, _ = bucket_subject_date(bucket)
        stamped = digests.get(bucket)
        stale = stale_cells(root, record, stamped if isinstance(stamped, dict) else {}, subject)
        if stale:
            findings.append(("error", f"region completeness for {subject!r} on "
                            f"{record.get('stem')!r}: cell(s) {stale} are attested complete but "
                            "either carry no stamped digest or their annotation content has "
                            "changed since attestation; re-attest"))


def check_trait_specs(root: Path, findings: list) -> None:
    from tcip_mcp.traits import load_trait_specs_with_errors

    # Diagnosing a project must leave it as it was found, so a stray spelling is named, not renamed.
    _specs, errors = load_trait_specs_with_errors(project_root=root, adopt_canonical_suffix=False)
    for e in errors:
        findings.append(("error", f"trait spec {e['file']} failed to load: {e['reason']}"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("project_root", help="project directory holding images/ annotations/ .tcip/")
    args = ap.parse_args()
    root = Path(args.project_root)
    if not root.is_dir():
        print(f"error: not a directory: {root}")
        return 2

    # Its own process entry point, so it binds the storage backend the seam has no default for.
    from tcip_store.file_backend import bind_default

    bind_default()

    findings: list[tuple[str, str]] = []
    for check in (check_negatives, check_registry, check_provenance, check_state,
                 check_region_completeness, check_trait_specs):
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
