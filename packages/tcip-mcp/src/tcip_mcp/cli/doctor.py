"""Data-state doctor: scan a live project for state inconsistencies code audits can't see.

Checks empty label documents nobody marked complete, registry entries pointing at
missing/test-fixture checkpoints, provenance smells, orphaned labels, verdict shards that will not
read, and a stray file under ``.tcip/state`` no store claims. Read-only. Run at session start:

    tcip doctor <project_root>

Exit codes: 0 clean, 1 warnings only, 2 errors.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from tcip_mcp.project_paths import project_state_dir


def _census(root: Path, findings: list, seen: "set[str]") -> dict | None:
    """The one dataset census a doctor run reads (``data_tools._scan_dataset``), or ``None`` with
    its failure reported: an ``images/`` stem collision or an unreadable ``.bandgroup``
    manifest.

    ``label_reads`` maps each census label's resolved path to its document as the one per-image
    reader (``json_io.read_label_document``) returns it, or to the ``UnreadableLabelDocument`` it
    raised: every check that reads a label reads it from here, so a run reads each label once, and
    only :func:`check_data_quality` reports an unreadable one.
    """
    from tcip_annotation.json_io import UnreadableLabelDocument, read_label_document
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem
    from tcip_mcp.tools.data_tools import _scan_dataset
    from tcip_store import SchemaVersionRefused

    try:
        scan = _scan_dataset(str(root))
    except AmbiguousImageStem as exc:
        _report_stem_collision(findings, exc, seen)
        return None
    except SchemaVersionRefused as exc:
        _report_band_group_version_refusal(findings, root, exc)
        return None
    label_reads: dict[Path, object] = {}
    for label in scan["labels"]:
        try:
            label_reads[Path(label).resolve()] = read_label_document(label)
        except UnreadableLabelDocument as exc:
            label_reads[Path(label).resolve()] = exc
    return {**scan, "label_reads": label_reads}


def _report_stem_collision(findings: list, exc: Exception, seen: "set[str] | None") -> None:
    """Append one collided-bucket finding, skipping a message this run already reported; ``seen``
    is ``None`` for a check run standalone.
    """
    message = str(exc)
    if seen is not None:
        if message in seen:
            return
        seen.add(message)
    findings.append(("error", message))


def _report_band_group_version_refusal(findings: list, root: Path, exc: Exception) -> None:
    """Report a ``.bandgroup`` manifest whose ``schema_version`` this reader does not accept as a
    finding naming the images tree it sits under.
    """
    from tcip_mcp.dataset_layout import image_root

    findings.append(("warn", f"{image_root(root)}: a .bandgroup manifest could not be read: {exc}"))


def _image_stems(root: Path) -> dict[str, str]:
    """stem -> file name for every image under images/ (the flat form and every date bucket),
    through the platform's own bucket enumeration, so a stem collision within one bucket refuses
    here too. A stem present in more than one bucket keeps the latest bucket's file, buckets
    visited in sorted (chronological, for ISO dates) order.
    """
    from tcip_mcp.dataset_layout import image_root
    from tcip_mcp.pipelines.image_utils import list_logical_images, logical_image_name

    out: dict[str, str] = {}
    images = image_root(root)
    if not images.is_dir():
        return out
    buckets = [images] + sorted(p for p in images.iterdir() if p.is_dir())
    for bucket in buckets:
        for stem, source in list_logical_images(bucket).items():
            out[stem] = logical_image_name(source)
    return out


def check_data_quality(root: Path, findings: list, *, census: dict | None) -> None:
    """Per-file annotation quality, each file read on its own so one bad document never hides the
    findings about the rest: a label with no matching image, a label the one per-image reader
    refuses (undecodable, a dataset-level COCO, an unrecognized shape, a malformed mark), an empty
    label no live mark finishes for any subject, and one line counting the images with no label
    record. Reads the run's one dataset census (``census``, ``None`` when it could not be taken).
    """
    from tcip_annotation.json_io import LabelDocument

    if census is None:
        return
    scan = census
    image_stems = {Path(p).stem for p in scan["images"]}
    labeled: set[str] = set()
    for label_path in scan["labels"]:
        label = Path(label_path)
        rel = label.relative_to(root) if root in label.parents else label
        labeled.add(label.stem)
        if label.stem not in image_stems:
            findings.append(("error", f"{rel}: no matching image"))
        doc = scan["label_reads"][label.resolve()]
        if not isinstance(doc, LabelDocument):
            findings.append(("error", f"{rel}: label file will not read: {doc}"))
        elif not doc.annotations and "negative" not in {doc.state(s) for s in doc.marks}:
            findings.append(("error", f"{rel}: empty label file, not marked complete for any "
                            "subject; excluded from training"))

    # Images with no label record at all: excluded from training, so a breeder who labeled 30 of
    # 400 trains on 30. Reported as one line, not one per image (the dominant case at small scale).
    unannotated = sorted(image_stems - labeled)
    if unannotated:
        shown = ", ".join(unannotated[:5]) + ("…" if len(unannotated) > 5 else "")
        findings.append(("info", f"{len(unannotated)} of {len(image_stems)} image(s) have no "
                        f"label record and are excluded from training ({shown}). Annotate them, "
                        "or mark the genuinely-empty ones Complete to train them as negatives."))


def check_reserved_names(root: Path, findings: list, *, census: dict | None) -> None:
    """Flag every image and label document whose stem is reserved for a prediction bucket's own
    record (``tcip_annotation.json_io.is_bucket_record``), as the dataset census names them."""
    if census is None:
        return
    scan = census
    for p in scan["reserved_name_images"]:
        findings.append(("error", f"{Path(p).relative_to(root)}: image stem is reserved for a "
                        "prediction bucket's own record; its label can never be read "
                        "through any bucket walk"))
    for p in scan["reserved_name_labels"]:
        findings.append(("error", f"{Path(p).relative_to(root)}: label filename is reserved for "
                        "a prediction bucket's own record; it is excluded from every "
                        "bucket walk and its annotations are unreadable through them"))


TEMP_TREE_MARKERS = ("pytest-of-", "\\Temp\\", "/Temp/")
"""Path fragments that place a registered checkpoint inside a test or temp tree."""


def check_registry(root: Path, findings: list) -> None:
    """Flag registered models whose checkpoint is missing or points into a test/temp tree, and
    every published bucket whose record names a checkpoint digest no registry entry carries.

    A checkpoint resolving under the project root never triggers the temp-tree marker scan, even
    when the root itself sits under one; only a checkpoint the root does not contain is scanned.

    Every way the registry can refuse to be read (a store that will not decode, a database file
    that will not open) comes out as a finding, not as a traceback.
    """
    from tcip_store import StoreError

    from tcip_mcp.buckets import bucket_dirs, read_bucket
    from tcip_mcp.model_registry import RegistryVersionRefused, registered_entries
    from tcip_mcp.registry_paths import (
        RegistryPathEmpty, RegistryPathTraversal, is_at_or_under, resolved_registry_path,
    )

    try:
        entries = registered_entries(root)
    except (StoreError, RegistryVersionRefused) as exc:
        findings.append(("error", "the model registry index will not decode or read, so this "
                        f"project's registered models could not be checked at all: {exc}"))
        return
    root_resolved = Path(root).resolve()
    for m in entries:
        # Existence resolves first; the temp-tree marker scan runs over the resolved string.
        try:
            resolved = resolved_registry_path(root, m["checkpoint_path"])
        except (RegistryPathEmpty, RegistryPathTraversal) as exc:
            findings.append(("error", f"registry entry {m['name']!r} checkpoint_path "
                            f"could not be resolved: {exc}"))
            continue
        ckpt = str(resolved)
        stray = not is_at_or_under(resolved, root_resolved) and any(
            marker in ckpt for marker in TEMP_TREE_MARKERS)
        if stray:
            findings.append(("error", f"registry entry {m['name']!r} points at a test/temp "
                            f"checkpoint: {ckpt}"))
        elif not Path(ckpt).is_file():
            findings.append(("error", f"registry entry {m['name']!r} checkpoint missing: {ckpt}"))

    registered_shas = {m["sha256"] for m in entries}
    for bucket in bucket_dirs(root):
        try:
            sha = read_bucket(bucket).producer.get("checkpoint_sha256")
        except ValueError as exc:
            findings.append(("error", f"{bucket.relative_to(root)}: bucket record will not "
                            f"read: {exc}"))
            continue
        if sha is not None and sha not in registered_shas:
            findings.append(("warn", f"{bucket.relative_to(root)}: prediction bucket's "
                            f"record names checkpoint {sha}, which no registry entry "
                            "names; register the checkpoint to make this bucket's "
                            "provenance verifiable going forward."))


def check_provenance(root: Path, findings: list, *, census: dict | None) -> None:
    from tcip_annotation.json_io import LabelDocument

    from tcip_mcp.experiments import run_observations

    unstamped = 0
    for label in map(Path, census["labels"] if census is not None else []):
        doc = census["label_reads"][label.resolve()] if census is not None else None
        for a in doc.annotations if isinstance(doc, LabelDocument) else []:
            if a.accepted_by and not a.created_by:
                findings.append(("warn", f"{label.relative_to(root)}: annotation has accepted_by "
                                "without created_by (acceptance without origin)"))
            if not a.created_by:
                unstamped += 1
    if unstamped:
        findings.append(("info", f"{unstamped} GT annotations carry no created_by"))

    # A bespoke run's source snapshot names what it failed to capture in its run.json.
    for run in run_observations(root):
        source = run.record["source"]
        if source is not None and (source["missing"] or source["snapshot_errors"]):
            findings.append(("warn", f"{run.directory.relative_to(root)}: source snapshot incomplete: "
                            f"{len(source['missing'])} missing file(s), "
                            f"{len(source['snapshot_errors'])} import error(s)"))


def check_state(root: Path, findings: list, *, seen: "set[str] | None" = None) -> None:
    """Warn of every verdict shard that will not read through the one decoder
    (:func:`~tcip_annotation.verdicts.read_verdicts`) and every one naming an image the dataset
    does not hold. ``seen`` carries a stem-collision message across the checks that enumerate the
    same ``images/`` tree, so one collision is reported once per run."""
    import tcip_store
    from tcip_annotation.verdicts import REVIEW_VERDICTS_STORE, read_verdicts
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem
    from tcip_store import SchemaVersionRefused, StoreError

    try:
        stems = _image_stems(root)
    except AmbiguousImageStem as exc:
        _report_stem_collision(findings, exc, seen)
        return
    except SchemaVersionRefused as exc:
        _report_band_group_version_refusal(findings, root, exc)
        return
    for key in tcip_store.keys(REVIEW_VERDICTS_STORE, str(project_state_dir(root))):
        bucket, image = key.parts
        try:
            read_verdicts(key)
        except (ValueError, StoreError) as exc:
            findings.append(("warn", f"verdict shard {bucket}/{image} will not read: {exc}"))
        if Path(image).stem not in stems:
            findings.append(("warn", f"verdict shard {bucket}/{image} names unknown image "
                            f"{image!r}"))


def check_traits(root: Path, findings: list) -> None:
    """An error for every trait record the store or its schema refuses to read (a warning when
    the refusal is the record's ``schema_version``), and a warning for every trait whose latest
    revision the breeder has not confirmed."""
    from pydantic import ValidationError
    from tcip_store import SchemaVersionRefused, StoreError

    from tcip_mcp.traits import read_trait, trait_names

    for name in trait_names(root):
        try:
            record = read_trait(name, root)
        except (StoreError, ValidationError) as exc:
            level = "warn" if isinstance(exc, SchemaVersionRefused) else "error"
            findings.append((level, f"trait {name!r} will not read: {exc}"))
            continue
        if not record.latest.confirmed:
            findings.append(("warn", f"the latest revision ({record.latest.number}) of trait "
                            f"{name!r} is not confirmed; ask the breeder to confirm it in the "
                            "Setup tab"))


def check_project_record(root: Path, findings: list) -> None:
    """An error for a project record that is absent, damaged, or on a root the store refuses to
    read."""
    from tcip_mcp.project_record import record_fields

    problem = record_fields(root)["record_problem"]
    if problem is not None:
        findings.append(("error", problem))


def check_stray_state_files(root: Path, findings: list) -> None:
    """A file under ``.tcip/state`` no store claims (``tcip_mcp.stray_state.stray_state_files``) is
    an info finding naming ``delete_stray_state_file`` as the remedy, never a warn or error. The
    predicate matches path templates and never consults a database, so on a behind-database root
    it can only under-report.
    """
    from tcip_mcp.stray_state import stray_state_files
    from tcip_mcp.tools.bundle import AnchorMisplaced
    from tcip_store.errors import StoreError

    try:
        strays = stray_state_files(root)
    except (AnchorMisplaced, StoreError) as exc:
        findings.append(("warn", f"the state root's accounting refused: {exc}"))
        return
    state_root = project_state_dir(Path(root).resolve())
    for path in strays:
        findings.append((
            "info",
            f".tcip/state/{path.relative_to(state_root).as_posix()}: a stray file under "
            ".tcip/state that no store claims; delete it with delete_stray_state_file if it is "
            "not needed"))


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], prog=prog)
    ap.add_argument("project_root", help="project directory holding images/ annotations/ .tcip/")
    args = ap.parse_args(argv)
    root = Path(args.project_root)
    if not root.is_dir():
        print(f"error: not a directory: {root}")
        return 2

    # Its own process entry point, so it binds the storage backend the seam has no default for.
    from tcip_store.binding import bind_default

    bind_default()

    findings: list[tuple[str, str]] = []
    # Shared across the checks that independently enumerate images/, so an images/ tree
    # collision or an unreadable .bandgroup manifest is reported once, not once per check.
    ambiguous_seen: set[str] = set()
    census = _census(root, findings, ambiguous_seen)
    checks_taking_census = (check_data_quality, check_reserved_names, check_provenance)
    for check in (check_data_quality, check_reserved_names, check_registry, check_provenance,
                  check_state, check_traits, check_project_record, check_stray_state_files):
        run: Callable[..., None] = check
        if check is check_state:
            run(root, findings, seen=ambiguous_seen)
        elif check in checks_taking_census:
            run(root, findings, census=census)
        else:
            run(root, findings)

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
