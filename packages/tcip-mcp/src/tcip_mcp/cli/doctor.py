"""Data-state doctor: scan a live project for state inconsistencies code audits can't see.

Checks empty label documents nobody marked complete, registry entries pointing at
missing/test-fixture checkpoints, provenance smells, orphaned labels, and verdict shards that will
not read. Read-only. Run at session start:

    tcip doctor <project_root>

Exit codes: 0 clean, 1 warnings only, 2 errors.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path


def _named(key) -> str:
    """A label document's key as a finding names it: ``<capture>/<stem>``."""
    return "/".join(key.parts)


def _census(root: Path, findings: list) -> dict | None:
    """The dataset census (``data_tools._scan_dataset``) with ``label_reads``, each label key's
    document or the ``UnreadableLabelDocumentError`` it raised; ``None`` with an error finding for
    an ``images/`` stem collision or an unreadable ``.bandgroup`` manifest."""
    from tcip_annotation.json_io import UnreadableLabelDocumentError, read_label_document
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


def check_data_quality(root: Path, findings: list, *, census: dict | None) -> None:
    """Per-document annotation quality, each document read on its own so one bad document never
    hides the findings about the rest, each named by its capture and stem: a label with no
    matching image, a label the one per-image reader refuses (undecodable, an unrecognized shape,
    a malformed mark), an empty label no live mark finishes for any subject, and one line counting
    the images with no label record. Reads the run's one dataset census (``census``, ``None`` when
    it could not be taken).
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
            findings.append(("error", f"{named}: label document will not read: {doc}"))
        elif not doc.annotations and not any(admits(doc.state(s)) for s in doc.marks):
            findings.append(("error", f"{named}: empty label document, not marked complete for "
                            "any subject; excluded from training"))

    # Images with no label record at all: excluded from training, so a breeder who labeled 30 of
    # 400 trains on 30. Reported as one line, not one per image (the dominant case at small scale).
    labeled = set(scan["labels"])
    unannotated = sorted(_named(key) for key in scan["images"].values() if key not in labeled)
    if unannotated:
        shown = ", ".join(unannotated[:5]) + ("…" if len(unannotated) > 5 else "")
        findings.append(("info", f"{len(unannotated)} of {len(scan['images'])} image(s) have no "
                        f"label record and are excluded from training ({shown}). Annotate them, "
                        "or mark the genuinely-empty ones Complete to train them as negatives."))


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
    import tcip_store
    from tcip_store import StoreError

    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.dataset_layout import PREDICTION_BUCKETS
    from tcip_mcp.model_registry import registered_entries
    from tcip_mcp.registry_paths import (
        RegistryPathEmptyError, RegistryPathTraversalError, resolved_registry_path,
    )

    try:
        entries = registered_entries(root)
    except StoreError as exc:
        findings.append(("error", "the model registry index will not decode or read, so this "
                        f"project's registered models could not be checked at all: {exc}"))
        return
    root_resolved = Path(root).resolve()
    for m in entries:
        # Existence resolves first; the temp-tree marker scan runs over the resolved string.
        try:
            resolved = resolved_registry_path(root, m["checkpoint_path"])
        except (RegistryPathEmptyError, RegistryPathTraversalError) as exc:
            findings.append(("error", f"registry entry {m['name']!r} checkpoint_path "
                            f"could not be resolved: {exc}"))
            continue
        ckpt = str(resolved)
        stray = not resolved.is_relative_to(root_resolved) and any(
            marker in ckpt for marker in TEMP_TREE_MARKERS)
        if stray:
            findings.append(("error", f"registry entry {m['name']!r} points at a test/temp "
                            f"checkpoint: {ckpt}"))
        elif not Path(ckpt).is_file():
            findings.append(("error", f"registry entry {m['name']!r} checkpoint missing: {ckpt}"))

    registered_shas = {m["sha256"] for m in entries}
    for key in tcip_store.keys(PREDICTION_BUCKETS, str(root_resolved)):
        name = key.parts[0]
        try:
            sha = read_bucket(root_resolved, name).producer.get("checkpoint_sha256")
        except ValueError as exc:
            findings.append(("error", f"bucket {name!r}: bucket record will not read: {exc}"))
            continue
        if sha is not None and sha not in registered_shas:
            findings.append(("warn", f"bucket {name!r}: prediction bucket's record names "
                            f"checkpoint {sha}, which no registry entry names; register the "
                            "checkpoint to make this bucket's provenance verifiable going "
                            "forward."))


def check_provenance(root: Path, findings: list, *, census: dict | None) -> None:
    from tcip_annotation.json_io import LabelDocument

    from tcip_mcp.experiments import run_observations

    unstamped = 0
    for key, doc in (census["label_reads"] if census is not None else {}).items():
        for a in doc.annotations if isinstance(doc, LabelDocument) else []:
            if a.accepted_by and not a.created_by:
                findings.append(("warn", f"{_named(key)}: annotation has accepted_by without "
                                "created_by (acceptance without origin)"))
            if not a.created_by:
                unstamped += 1
    if unstamped:
        findings.append(("info", f"{unstamped} GT annotations carry no created_by"))

    # A bespoke run's source snapshot names what it failed to capture in its run.json.
    for run in run_observations(root):
        source = run.record["source"]
        if source is not None and (source["missing"] or source["snapshot_errors"]):
            findings.append(("warn",
                            f"{run.directory.relative_to(root)}: source snapshot incomplete: "
                            f"{len(source['missing'])} missing file(s), "
                            f"{len(source['snapshot_errors'])} import error(s)"))


def check_state(root: Path, findings: list) -> None:
    """Warn of every verdict shard that will not read through the one decoder
    (:func:`~tcip_annotation.verdicts.read_verdicts`) and every one naming an image its bucket's
    record names no document for (:func:`~tcip_mcp.buckets.read_bucket`)."""
    import tcip_store
    from tcip_annotation.verdicts import REVIEW_VERDICTS_STORE, read_verdicts
    from tcip_store import StoreError

    from tcip_mcp.buckets import read_bucket

    resolved = Path(root).resolve()
    for key in tcip_store.keys(REVIEW_VERDICTS_STORE, str(resolved)):
        bucket, stem = key.parts
        try:
            read_verdicts(key)
            known = stem in read_bucket(resolved, bucket).documents
        except (ValueError, StoreError) as exc:
            findings.append(("warn", f"verdict shard {bucket}/{stem} will not read: {exc}"))
            continue
        if not known:
            findings.append(("warn", f"verdict shard {bucket}/{stem} names image {stem!r}, "
                            "which its bucket's record names no document for"))


def check_traits(root: Path, findings: list) -> None:
    """An error for every trait record the store or its schema refuses to read, and a warning for
    every trait whose latest revision the breeder has not confirmed."""
    from pydantic import ValidationError
    from tcip_store import StoreError

    from tcip_mcp.traits import read_trait, trait_names

    for name in trait_names(root):
        try:
            record = read_trait(name, root)
        except (StoreError, ValidationError) as exc:
            findings.append(("error", f"trait {name!r} will not read: {exc}"))
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


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], prog=prog)
    ap.add_argument("project_root", help="project directory holding images/ and .tcip/")
    args = ap.parse_args(argv)
    root = Path(args.project_root)
    if not root.is_dir():
        print(f"error: not a directory: {root}")
        return 2

    from tcip_store import bind

    bind()

    findings: list[tuple[str, str]] = []
    census = _census(root, findings)
    checks_taking_census = (check_data_quality, check_provenance)
    for check in (check_data_quality, check_registry, check_provenance, check_state,
                  check_traits, check_project_record):
        run: Callable[..., None] = check
        if check in checks_taking_census:
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
