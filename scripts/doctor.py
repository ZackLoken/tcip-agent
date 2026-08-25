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
    subject and date, so the disagreement is checked per subject present in the file. An
    image-level record (a subject with no geometry) counts as content for that subject, the same
    rule ``annotations_hold_subject`` applies everywhere else this question is asked.
    """
    from tcip_annotation import json_io
    from tcip_mcp.dataset_layout import (
        annotation_date, annotation_root, annotations_hold_subject, bucket_subject_date,
        image_status_path, is_confirmed_negative, normalize_status_store,
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
        for key, bucket in by_bucket.items():
            if not is_confirmed_negative(bucket.get(name)):
                continue
            subj, bdate = bucket_subject_date(key)
            if bdate != date:
                continue
            if annotations_hold_subject(anns, subj):
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


def check_status_tokens(root: Path, findings: list) -> None:
    """Flag two ways the status store can no longer be trusted at face value: an entry a reader
    doesn't recognize (dropped silently by any merge), and a stored ``"complete"`` whose label
    file holds no annotation of the confirmed subject (a stale token, possibly reconciled since by
    the GUI's own hydrate, possibly not). A report, not a rewrite: the record names who confirmed
    it, and this doctor is not that person.
    """
    from tcip_annotation import json_io
    from tcip_mcp.dataset_layout import (
        annotation_path, annotations_hold_subject, bucket_subject_date, image_status_path,
        status_confirmations, unreadable_status_entries,
    )

    raw = _load(image_status_path(root))
    unreadable = unreadable_status_entries(raw)
    if unreadable:
        findings.append(("warn", f"{len(unreadable)} status entr{'y is' if len(unreadable) == 1 else 'ies are'} "
                        f"in a shape this reader does not recognize, starting with {unreadable[:3]}"))

    for bucket, records in status_confirmations(raw).items():
        subject, date = bucket_subject_date(bucket)
        for name, record in records.items():
            if record.get("status") != "complete":
                continue
            label = annotation_path(root, date, Path(name).stem)
            anns = json_io.read_annotations(str(label)) if label.is_file() else []
            if not annotations_hold_subject(anns, subject):
                findings.append(("warn", f"{bucket}/{name}: status says 'complete' but the label "
                                f"file holds no {subject!r} annotation; re-confirm"))


TEMP_TREE_MARKERS = ("pytest-of-", "\\Temp\\", "/Temp/")
"""Path fragments that place a registered checkpoint inside a test or temp tree."""


def check_registry(root: Path, findings: list) -> None:
    """Flag registered models whose checkpoint is missing or points into a test/temp tree.

    Every way the registry index can refuse to be read comes out as a finding, not as a
    traceback: the doctor's contract is an exit code and a list, and a check that dies takes the
    whole run's findings and exit code with it. A store that will not decode and a database file
    that will not open are both "this could not be checked", which is not the same answer as clean.
    """
    from tcip_store import StoreError

    from tcip_mcp.model_registry import read_registry_index

    try:
        entries = read_registry_index(root)
    except StoreError as exc:
        findings.append(("error", "the model registry index will not decode or read, so this "
                        f"project's registered models could not be checked at all: {exc}"))
        return
    for m in entries if isinstance(entries, list) else []:
        ckpt = m.get("checkpoint_path", "")
        if "metrics_source" not in m:
            findings.append(("warn", f"{m.get('name')!r} in the model registry carries no "
                            "metrics_source (predates the field); conform it with "
                            "scripts/conform_registry_metrics_source.py"))
        if any(marker in ckpt for marker in TEMP_TREE_MARKERS):
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
        # Shards sit one directory deep per prediction bucket, and directly here for a review
        # that named no bucket.
        for shard in shard_dir.rglob("*.json"):
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
    """Deliberately absent from ``gated_stores()``: this reads entirely through the storage seam
    (``load_trait_specs_with_errors``), which resolves to whichever backend the process is bound
    to and so can never be stale relative to itself, unlike a check that reads a raw file off
    disk."""
    from tcip_mcp.traits import load_trait_specs_with_errors

    _specs, errors = load_trait_specs_with_errors(project_root=root)
    for e in errors:
        findings.append(("error", f"trait spec {e['file']} failed to load: {e['reason']}"))


def check_trait_spec_statements(root: Path, findings: list) -> None:
    """A registered trait spec with no authoring statement behind it: the recoverable gap
    author_trait_spec's own second write can leave when it fails partway. The breeder's new
    confirmation panel would show no row for a trait that plainly exists, so this is worth
    surfacing rather than leaving to be found only when the breeder asks why.

    Deliberately absent from ``gated_stores()`` for the same reason as ``check_trait_specs``:
    it reads only through the storage seam (``load_trait_specs``/``ts.keys``), never a raw file,
    so it cannot be stale relative to the backend it is reading from."""
    import tcip_store as ts

    from tcip_mcp.traits import (
        TRAIT_SPEC_STATEMENTS_STORE,
        load_trait_specs,
        trait_spec_statements_scope,
    )

    specs = load_trait_specs(project_root=root)
    if not specs:
        return
    scope = trait_spec_statements_scope(root)
    stated = {key.parts[0] for key in ts.keys(TRAIT_SPEC_STATEMENTS_STORE, str(scope))}
    for spec in specs:
        if spec.name not in stated:
            findings.append(("warn", f"trait spec {spec.name!r} has no authoring statement on "
                            "record; author it with author_trait_spec so the breeder has "
                            "something to confirm"))


def gated_stores(root: Path) -> dict[str, tuple[tuple[Path, str], ...]]:
    """Which database-held store each file-reading check depends on, and under which root.

    Exactly what the checks above read off disk, no wider: a store the doctor never reads
    cannot make a check it does not run report anything. Live-state stores are not doctor
    inputs and are not here.
    """
    return {
        "check_negatives": ((root, "image_status"),),
        "check_status_tokens": ((root, "image_status"),),
        "check_region_completeness": (
            (root, "region_completeness"),
            (root, "region_completeness_digest"),
        ),
        "check_state": ((root / ".tcip" / "state", "review_verdicts"),),
        "check_provenance": ((root / ".tcip" / "experiments", "model_snapshot_manifest"),),
    }


def staleness_findings(root: Path) -> dict[str, str]:
    """Per check, why its files cannot be trusted, for the checks whose stores are behind.

    A root with no database is on the file layout and every check reads the authority
    directly. A store with no counter row was never written in its database and reads current.
    Anything else stale is reported as this check being invalid rather than as clean, because a
    check that read files older than the database found the state of an earlier session.
    """
    from tcip_store.errors import StoreError
    from tcip_store.export import stale_stores
    from tcip_store.file_backend import database_file

    invalid: dict[str, str] = {}
    for check, gated in gated_stores(root).items():
        by_root: dict[Path, list[str]] = {}
        for store_root, store in gated:
            by_root.setdefault(store_root, []).append(store)
        reasons: list[str] = []
        for store_root, stores in by_root.items():
            db_path = database_file(str(store_root.absolute()))
            if not db_path.is_file():
                continue
            try:
                stale = stale_stores(db_path, tuple(stores))
            except StoreError as exc:
                reasons.append(f"{db_path} could not be read: {exc}")
                continue
            if stale:
                reasons.append(f"{', '.join(stale)} in {db_path}")
        if reasons:
            invalid[check] = "; ".join(reasons)
    return invalid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("project_root", help="project directory holding images/ annotations/ .tcip/")
    args = ap.parse_args()
    root = Path(args.project_root)
    if not root.is_dir():
        print(f"error: not a directory: {root}")
        return 2

    # Its own process entry point, so it binds the storage backend the seam has no default for.
    from tcip_store.binding import bind_default

    bind_default()

    findings: list[tuple[str, str]] = []
    invalid = staleness_findings(root)
    for check in (check_negatives, check_status_tokens, check_registry, check_provenance,
                 check_state, check_region_completeness, check_trait_specs,
                 check_trait_spec_statements):
        reason = invalid.get(check.__name__)
        if reason:
            findings.append(("error", f"{check.__name__} reads state as files and those files "
                            f"are behind the database that holds it ({reason}). This check is "
                            "invalid, not clean: write the files out with "
                            "'python scripts/export_store.py' and run the doctor again."))
            continue
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
