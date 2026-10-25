"""Data-state doctor: scan a live project for state inconsistencies code audits can't see.

Checks empty label documents nobody marked complete, registry entries pointing at
missing/test-fixture checkpoints, provenance smells, orphaned labels, and verdict shards that will
not read. Read-only:

    tcip doctor <project_root>

Exit codes: 0 clean, 1 warnings only, 2 errors.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tcip_store import Key


def _named(key: Key) -> str:
    """Return a label document's key as a finding names it, ``<capture>/<stem>``."""
    return "/".join(key.parts)


def _census(root: Path, findings: list) -> dict | None:
    """Take the dataset census with every label document read.

    Returns the census (``data_tools._scan_dataset``) with ``label_reads``, each label key's
    document or the ``UnreadableLabelDocumentError`` it raised; ``None`` with an error finding for
    an ``images/`` stem collision.
    """
    from tcip_annotation.json_io import (
        UnreadableLabelDocumentError,
        read_label_document,
    )
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStemError
    from tcip_mcp.tools.data_tools import _scan_dataset

    try:
        scan = _scan_dataset(str(root))
    except AmbiguousImageStemError as exc:
        findings.append(("error", str(exc)))
        return None
    label_reads: dict = {}
    for key in scan["labels"]:
        try:
            label_reads[key] = read_label_document(key)
        except UnreadableLabelDocumentError as exc:
            label_reads[key] = exc
    return {**scan, "label_reads": label_reads}


def check_data_quality(
    root: Path, findings: list, *, census: dict | None
) -> None:
    """Append the annotation quality findings, one document at a time.

    Each document is read on its own so one bad document never hides the findings about the
    rest, each named by its capture and stem: a label with no matching image, a label the
    per-image reader refuses (undecodable, an unrecognized shape, a malformed mark), an empty
    label no live mark finishes for any subject, and one line counting the images with no label
    record. Reads the dataset census ``census`` (``None`` when it could not be taken).
    """
    from tcip_annotation.json_io import LabelDocument
    from tcip_mcp.pipelines.data.label_queries import admits

    if census is None:
        return
    scan = census
    imaged = set(scan["images"].values())
    for key in scan["labels"]:
        named = _named(key)
        if key not in imaged:
            findings.append(("error", f"{named}: no matching image"))
        doc = scan["label_reads"][key]
        if not isinstance(doc, LabelDocument):
            findings.append(
                ("error", f"{named}: label document will not read: {doc}")
            )
        elif not doc.annotations and not any(
            admits(doc.state(s)) for s in doc.marks
        ):
            findings.append(
                (
                    "error",
                    f"{named}: empty label document, not marked complete for "
                    "any subject; excluded from training",
                )
            )

    # Images with no label record at all: excluded from training, so a breeder who labeled 30 of
    # 400 trains on 30. Reported as one line, not one per image (the dominant case at small scale).
    labeled = set(scan["labels"])
    unannotated = sorted(
        _named(key) for key in scan["images"].values() if key not in labeled
    )
    if unannotated:
        shown = ", ".join(unannotated[:5]) + (
            "…" if len(unannotated) > 5 else ""
        )
        findings.append(
            (
                "info",
                f"{len(unannotated)} of {len(scan['images'])} image(s) have no "
                f"label record and are excluded from training ({shown}); an image marked "
                "Complete with no annotation trains as a negative.",
            )
        )


TEMP_TREE_MARKERS = ("pytest-of-", "\\Temp\\", "/Temp/")
"""Path fragments that place a registered checkpoint inside a test or temp tree."""


def check_registry(root: Path, findings: list) -> None:
    """Flag registered models whose checkpoint is missing or points into a test or temp tree.

    Also flags every published bucket whose record names a checkpoint digest no registry entry
    carries.

    A checkpoint resolving under the project root never triggers the temp-tree marker scan, even
    when the root itself sits under one; only a checkpoint the root does not contain is scanned.

    Every way the registry can refuse to be read (a store that will not decode, a database file
    that will not open, an index that does not validate) comes out as a finding, not as a
    traceback.
    """
    from pydantic import ValidationError

    import tcip_store
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.dataset_layout import PREDICTION_BUCKETS
    from tcip_mcp.model_registry import Producer, registered_entries
    from tcip_mcp.registry_paths import (
        RegistryPathEmptyError,
        RegistryPathTraversalError,
        resolved_registry_path,
    )
    from tcip_store import StoreError

    try:
        entries = registered_entries(root)
    except (StoreError, ValidationError) as exc:
        findings.append(
            (
                "error",
                "the model registry index will not decode or read, so this "
                f"project's registered models could not be checked at all: {exc}",
            )
        )
        return
    for m in entries:
        # Existence resolves first; the temp-tree marker scan runs over the resolved string.
        try:
            resolved = resolved_registry_path(root, m.checkpoint_path)
        except (RegistryPathEmptyError, RegistryPathTraversalError) as exc:
            findings.append(
                (
                    "error",
                    f"registry entry {m.name!r} checkpoint_path "
                    f"could not be resolved: {exc}",
                )
            )
            continue
        ckpt = str(resolved)
        stray = not resolved.is_relative_to(root) and any(
            marker in ckpt for marker in TEMP_TREE_MARKERS
        )
        if stray:
            findings.append(
                (
                    "error",
                    f"registry entry {m.name!r} points at a test/temp "
                    f"checkpoint: {ckpt}",
                )
            )
        elif not Path(ckpt).is_file():
            findings.append(
                (
                    "error",
                    f"registry entry {m.name!r} checkpoint missing: {ckpt}",
                )
            )

    registered_shas = {m.sha256 for m in entries}
    for key in tcip_store.keys(PREDICTION_BUCKETS, str(root)):
        name = key.parts[0]
        try:
            producer = read_bucket(root, name).producer
        except ValueError as exc:
            findings.append(
                (
                    "error",
                    f"bucket {name!r}: bucket record will not read: {exc}",
                )
            )
            continue
        if (
            isinstance(producer, Producer)
            and producer.checkpoint_sha256 not in registered_shas
        ):
            findings.append(
                (
                    "warn",
                    f"bucket {name!r}: prediction bucket's record names checkpoint "
                    f"{producer.checkpoint_sha256}, which no registry entry names, so the "
                    "bucket's provenance is not verifiable against the registry.",
                )
            )


def check_provenance(
    root: Path, findings: list, *, census: dict | None
) -> None:
    """Append the provenance findings over the census's readable label documents.

    Warns of every annotation carrying ``accepted_by`` without ``created_by``, and, when any
    annotation carries no ``created_by``, adds one info finding counting them. Appends nothing
    when ``census`` is ``None``.
    """
    from tcip_annotation.json_io import LabelDocument

    unstamped = 0
    for key, doc in (
        census["label_reads"] if census is not None else {}
    ).items():
        for a in doc.annotations if isinstance(doc, LabelDocument) else []:
            if a.accepted_by and not a.created_by:
                findings.append(
                    (
                        "warn",
                        f"{_named(key)}: annotation has accepted_by without "
                        "created_by (acceptance without origin)",
                    )
                )
            if not a.created_by:
                unstamped += 1
    if unstamped:
        findings.append(
            ("info", f"{unstamped} GT annotations carry no created_by")
        )


def check_state(root: Path, findings: list) -> None:
    """Warn of every unreadable flags record or verdict shard and every shard naming no document.

    Warns of every verdict shard that will not read through
    :func:`~tcip_annotation.verdicts.read_verdicts`, every one naming an image its bucket's
    record names no document for (:func:`~tcip_mcp.buckets.read_bucket`), and every flags record
    that will not read (:func:`~tcip_annotation.flags.read_flags`).
    """
    import tcip_store
    from tcip_annotation.flags import REVIEW_FLAGS_STORE, read_flags
    from tcip_annotation.verdicts import REVIEW_VERDICTS_STORE, read_verdicts
    from tcip_mcp.buckets import read_bucket
    from tcip_store import StoreError

    for key in tcip_store.keys(REVIEW_FLAGS_STORE, str(root)):
        try:
            read_flags(key)
        except (ValueError, StoreError) as exc:
            findings.append(
                (
                    "warn",
                    f"flags record {'/'.join(key.parts)} will not read: {exc}",
                )
            )
    for key in tcip_store.keys(REVIEW_VERDICTS_STORE, str(root)):
        bucket, stem = key.parts
        try:
            read_verdicts(key)
            known = stem in read_bucket(root, bucket).documents
        except (ValueError, StoreError) as exc:
            findings.append(
                ("warn", f"verdict shard {bucket}/{stem} will not read: {exc}")
            )
            continue
        if not known:
            findings.append(
                (
                    "warn",
                    f"verdict shard {bucket}/{stem} names image {stem!r}, "
                    "which its bucket's record names no document for",
                )
            )


def check_traits(root: Path, findings: list) -> None:
    """Append an error for every unreadable trait record and a warning for every unconfirmed one.

    An unreadable record is one the store or its schema refuses to read; an unconfirmed one is a
    trait whose latest revision the breeder has not confirmed.
    """
    from pydantic import ValidationError

    from tcip_mcp.traits import read_trait, trait_names
    from tcip_store import StoreError

    for name in trait_names(root):
        try:
            record = read_trait(name, root)
        except (StoreError, ValidationError) as exc:
            findings.append(("error", f"trait {name!r} will not read: {exc}"))
            continue
        if not record.latest.confirmed:
            findings.append(
                (
                    "warn",
                    f"the latest revision ({record.latest.number}) of trait "
                    f"{name!r} is not confirmed; the breeder confirms a revision in the "
                    "Setup tab",
                )
            )


def check_project_record(root: Path, findings: list) -> None:
    """Append an error when the project record is absent, damaged, or on an unreadable root."""
    from tcip_mcp.project_record import record_fields

    problem = record_fields(root)["record_problem"]
    if problem is not None:
        findings.append(("error", problem))


def findings(root: Path) -> list[tuple[str, str]]:
    """Return every check's findings over the project at ``root``, errors first, then warnings.

    Each finding is its level (``error``, ``warn`` or ``info``) and its message, read through the
    store this process already bound.
    """
    found: list[tuple[str, str]] = []
    census = _census(root, found)
    checks_taking_census = (check_data_quality, check_provenance)
    for check in (
        check_data_quality,
        check_registry,
        check_provenance,
        check_state,
        check_traits,
        check_project_record,
    ):
        run: Callable[..., None] = check
        if check in checks_taking_census:
            run(root, found, census=census)
        else:
            run(root, found)
    rank = {"error": 0, "warn": 1, "info": 2}
    return sorted(found, key=lambda f: rank[f[0]])


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    """Run every check over the project named in ``argv`` and print the findings, errors first.

    Returns 2 when any finding is an error or the root is not a directory, 1 when the worst is a
    warning, and 0 otherwise.
    """
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], prog=prog
    )
    ap.add_argument(
        "project_root", help="project directory holding images/ and .tcip/"
    )
    args = ap.parse_args(argv)
    root = Path(args.project_root).resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}")
        return 2

    from tcip_store import bind

    bind()
    found = findings(root)
    for level, msg in found:
        print(f"[{level.upper():5}] {msg}")
    errors = sum(1 for level, _ in found if level == "error")
    warns = sum(1 for level, _ in found if level == "warn")
    print(
        f"\ndoctor: {errors} error(s), {warns} warning(s), "
        f"{len(found) - errors - warns} info"
    )
    return 2 if errors else (1 if warns else 0)


if __name__ == "__main__":
    sys.exit(main())
