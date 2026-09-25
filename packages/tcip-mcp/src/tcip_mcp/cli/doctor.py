"""Data-state doctor: scan a live project for state inconsistencies code audits can't see.

Checks status-store vs disk disagreements on negatives, registry entries pointing at
missing/test-fixture checkpoints, provenance smells, orphaned labels, and a stray file under
``.tcip/state`` no store claims. Read-only. Run at session start:

    tcip doctor <project_root>

Exit codes: 0 clean, 1 warnings only, 2 errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _note_version(findings: list, where: str, store: str, doc) -> None:
    """Report, never refuse, a document whose schema_version this reader does not accept."""
    from tcip_store import SchemaVersionRefused, check_schema_version, get_descriptor

    try:
        check_schema_version(get_descriptor(store), doc)
    except SchemaVersionRefused as exc:
        findings.append(("warn", f"{where}: {exc}"))


def _read_reporting_version(findings: list, root: Path, key, path: Path) -> dict | None:
    """One store document read through the seam, ``{}`` when none is stored, or ``None`` with a
    warning finding when its schema version is above this reader's."""
    import tcip_store
    from tcip_store import SchemaVersionRefused

    try:
        return tcip_store.read(key, default={})
    except SchemaVersionRefused as exc:
        findings.append(("warn", f"{path.relative_to(root)}: {exc}"))
        return None


def _census(root: Path, findings: list, seen: "set[str]") -> dict | None:
    """The one dataset census a doctor run reads (``data_tools._scan_dataset``), or ``None`` with
    its failure reported: an ``images/`` stem collision or an unreadable ``.bandgroup``
    manifest.

    ``label_reads`` maps each census label's resolved path to its annotations as the one
    per-image reader (``json_io.read_annotations``) returns them, or to the
    ``UnreadableLabelDocument`` it raised: every check that reads a label reads it from here, so
    a run reads each label once, and only :func:`check_data_quality` reports an unreadable one.
    """
    from tcip_annotation.json_io import UnreadableLabelDocument, read_annotations
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
            label_reads[Path(label).resolve()] = read_annotations(label)
        except UnreadableLabelDocument as exc:
            label_reads[Path(label).resolve()] = exc
    return {**scan, "label_reads": label_reads}


def _readable_annotations(census: dict, label: Path) -> list | None:
    """``label``'s annotations from the census's one read, ``[]`` for a label the census holds no
    file for, ``None`` for one that will not read (:func:`check_data_quality`'s finding)."""
    from tcip_annotation.json_io import UnreadableLabelDocument

    read = census["label_reads"].get(Path(label).resolve(), [])
    return None if isinstance(read, UnreadableLabelDocument) else read


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


def check_negatives(root: Path, findings: list, *, census: dict | None,
                    seen: "set[str] | None" = None) -> None:
    """A negative is empty labels + human Complete, per subject: flag every disk/status
    disagreement.

    Labels are one name-based file per image (all subjects); a confirmed negative is scoped to a
    subject and date, so the disagreement is checked per subject present in the file. An
    image-level record (a subject with no geometry) counts as content for that subject. Reads the
    status store through the storage seam.

    ``census`` is the run's one dataset census (:func:`_census`), ``None`` when it could not be
    taken; ``seen`` carries a stem-collision message across the checks that enumerate the same
    ``images/`` tree, so one collision is reported once per run. An unreadable label is
    :func:`check_data_quality`'s finding.
    """
    from tcip_mcp.dataset_layout import (
        annotation_date, annotations_hold_subject, bucket_subject_date,
        confirmed_negative_names_any_subject, is_confirmed_negative, status_tokens,
        read_image_status_store, resolve_image_name,
    )
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem
    from tcip_store import StoreError

    if census is None:
        return

    # Confirmations are dataset-native, and this check already assumes root == dataset_root.
    try:
        by_bucket = status_tokens(read_image_status_store(root))
    except StoreError as exc:
        # The same soft-rail posture check_status_tokens already takes on this file: a reporter
        # names what it could not verify rather than blocking the whole run over it.
        findings.append(("warn", f"the image status store will not read ({exc}); negatives "
                         "cannot be verified against it"))
        return
    try:
        stems = _image_stems(root)
    except AmbiguousImageStem as exc:
        _report_stem_collision(findings, exc, seen)
        return
    labels = [Path(p) for p in census["labels"]]
    neg_names = confirmed_negative_names_any_subject(by_bucket)
    ambiguous_reported: set[str] = set()

    for label in labels:
        anns = _readable_annotations(census, label)
        if anns is None:
            continue
        date = annotation_date(label)
        try:
            name = resolve_image_name(root, date, label.stem)
        except AmbiguousImageStem as exc:
            message = str(exc)
            if message not in ambiguous_reported:
                findings.append(("error", message))
                ambiguous_reported.add(message)
            continue
        if name is not None:
            for key, bucket in by_bucket.items():
                if not is_confirmed_negative(bucket.get(name)):
                    continue
                subj, bdate = bucket_subject_date(key)
                if bdate != date:
                    continue
                if annotations_hold_subject(anns, subj):
                    findings.append(("error", f"{label.relative_to(root)}: has {subj!r} annotations "
                                    f"but the status store says 'negative' for {subj!r}, "
                                    "contradictory; re-review"))

    # Images with no label record at all: excluded from training, so a breeder who labeled 30 of
    # 400 trains on 30. Reported as one line, not one per image (the dominant case at small scale).
    labeled = {p.stem for p in labels}
    unannotated = sorted(set(stems) - labeled)
    if unannotated:
        shown = ", ".join(unannotated[:5]) + ("…" if len(unannotated) > 5 else "")
        findings.append(("info", f"{len(unannotated)} of {len(stems)} image(s) have no label "
                        f"record and are excluded from training ({shown}). Annotate them, or mark "
                        "the genuinely-empty ones Complete to train them as negatives."))

    for name in neg_names:
        if Path(name).stem not in labeled:
            findings.append(("warn", f"status says {name} is negative but no label file exists "
                            "(a confirmed negative should have an empty label file)"))


def check_data_quality(root: Path, findings: list, *, census: dict | None) -> None:
    """Per-file annotation quality: stem matching between images and labels, an empty per-image
    label with no human confirmation the image is a negative, and a file the one per-image reader
    refuses (undecodable, a dataset-level COCO, an unrecognized shape), each file read on its own
    so one bad document never hides the findings about the rest.

    Reads the run's one dataset census (``census``, ``None`` when it could not be taken), and
    the status store through the storage seam.
    """
    from tcip_mcp.dataset_layout import (
        annotation_date, confirmed_negative_names_any_subject, status_tokens,
        read_image_status_store, resolve_image_name,
    )
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem
    from tcip_store import StoreError

    if census is None:
        return
    scan = census
    image_stems = {Path(p).stem for p in scan["images"]}
    ambiguous_reported: set[str] = set()

    try:
        negatives = confirmed_negative_names_any_subject(
            status_tokens(read_image_status_store(str(root)))
        )
    except StoreError as exc:
        # The same soft-rail posture check_negatives already takes on this file: a reporter
        # names what it could not verify rather than blocking the whole run over it.
        findings.append(("warn", f"the image status store will not read ({exc}); confirmed "
                         "negatives cannot be verified against it"))
        return

    for label_path in scan["labels"]:
        label = Path(label_path)
        rel = label.relative_to(root) if root in label.parents else label
        stem = label.stem
        if stem not in image_stems:
            findings.append(("error", f"{rel}: no matching image"))
        anns = scan["label_reads"][label.resolve()]
        if not isinstance(anns, list):
            findings.append(("error", f"{rel}: label file will not read: {anns}"))
            continue
        if not anns:
            try:
                name = resolve_image_name(str(root), annotation_date(label_path), stem)
            except AmbiguousImageStem as exc:
                message = str(exc)
                if message not in ambiguous_reported:
                    findings.append(("error", message))
                    ambiguous_reported.add(message)
                continue
            if name is None or name not in negatives:
                findings.append(("error", f"{rel}: empty label file, not a confirmed "
                                "negative for any subject; excluded from training"))


def check_reserved_names(root: Path, findings: list, *, census: dict | None) -> None:
    """Flag every image and label document whose stem is reserved for a prediction bucket's own
    provenance stamp (``tcip_annotation.json_io.is_sidecar_name``), as the dataset census names
    them."""
    if census is None:
        return
    scan = census
    for p in scan["reserved_name_images"]:
        findings.append(("error", f"{Path(p).relative_to(root)}: image stem is reserved for a "
                        "prediction bucket's own provenance stamp; its label can never be read "
                        "through any bucket walk"))
    for p in scan["reserved_name_labels"]:
        findings.append(("error", f"{Path(p).relative_to(root)}: label filename is reserved for "
                        "a prediction bucket's own provenance stamp; it is excluded from every "
                        "bucket walk and its annotations are unreadable through them"))


def check_status_tokens(root: Path, findings: list, *, census: dict | None) -> None:
    """Flag a stored ``"complete"`` whose label file holds no annotation of the confirmed subject
    (a stale token), and a status store above this reader's schema version. A report, never a
    rewrite. Labels are read from the run's one census (``None`` when it could not be taken); an
    unreadable one is :func:`check_data_quality`'s finding.
    """
    from tcip_mcp.dataset_layout import (
        annotation_path, annotations_hold_subject, bucket_subject_date, image_status_key,
        image_status_path, status_confirmations,
    )

    raw = _read_reporting_version(findings, root, image_status_key(root), image_status_path(root))
    if raw is None or census is None:
        return

    for bucket, records in status_confirmations(raw).items():
        subject, date = bucket_subject_date(bucket)
        for name, record in records.items():
            if record["status"] != "complete":
                continue
            anns = _readable_annotations(census, annotation_path(root, date, Path(name).stem))
            if anns is None:
                continue
            if not annotations_hold_subject(anns, subject):
                findings.append(("warn", f"{bucket}/{name}: status says 'complete' but the label "
                                f"file holds no {subject!r} annotation; re-confirm"))


TEMP_TREE_MARKERS = ("pytest-of-", "\\Temp\\", "/Temp/")
"""Path fragments that place a registered checkpoint inside a test or temp tree."""


def check_registry(root: Path, findings: list) -> None:
    """Flag registered models whose checkpoint is missing or points into a test/temp tree, and
    every prediction bucket whose stamp names a checkpoint digest no registry entry carries.

    The bucket walk includes the cleared archive (``include_cleared=True``).

    A checkpoint resolving under the project root never triggers the temp-tree marker scan, even
    when the root itself sits under one; only a checkpoint the root does not contain is scanned.

    Every way the registry index can refuse to be read (a store that will not decode, a database
    file that will not open) comes out as a finding, not as a traceback.
    """
    from tcip_store import StoreError

    from tcip_mcp.dataset_layout import prediction_bucket_dirs
    from tcip_mcp.model_registry import RegistryVersionRefused, read_registry_index
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar
    from tcip_mcp.registry_paths import (
        RegistryPathEmpty, RegistryPathTraversal, is_at_or_under, resolved_registry_path,
    )

    try:
        entries = read_registry_index(root)
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
    for bucket in prediction_bucket_dirs(root, include_cleared=True):
        sha = (read_operating_point_sidecar(bucket) or {}).get("checkpoint_sha256")
        if sha and sha not in registered_shas:
            findings.append(("warn", f"{bucket.relative_to(root)}: prediction bucket's "
                            f"stamp names checkpoint {sha}, which no registry entry "
                            "names; register the checkpoint to make this bucket's "
                            "provenance verifiable going forward."))


def check_provenance(root: Path, findings: list, *, census: dict | None) -> None:
    from tcip_mcp.pipelines.model_build import SNAPSHOT_MANIFEST_STORE

    unstamped = 0
    for label in map(Path, census["labels"] if census is not None else []):
        for a in (_readable_annotations(census, label) if census is not None else None) or []:
            if a.accepted_by and not a.created_by:
                findings.append(("warn", f"{label.relative_to(root)}: annotation has accepted_by "
                                "without created_by (acceptance without origin)"))
            if not a.created_by:
                unstamped += 1
    if unstamped:
        findings.append(("info", f"{unstamped} GT annotations carry no created_by"))

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
            _note_version(
                findings, str(manifest_path.relative_to(root)), SNAPSHOT_MANIFEST_STORE, manifest
            )
            missing = manifest.get("missing") or []
            errors = manifest.get("snapshot_errors") or []
            if missing or errors:
                findings.append(("warn", f"{manifest_path.relative_to(root)}: source snapshot "
                                f"incomplete: {len(missing)} missing file(s), "
                                f"{len(errors)} import error(s)"))


def check_state(root: Path, findings: list, *, seen: "set[str] | None" = None) -> None:
    """``seen`` is the same cross-check stem-collision set ``check_negatives`` takes, so the
    three checks that enumerate ``images/`` report one collision once, not once each."""
    from tcip_annotation.review_engine import REVIEW_VERDICTS_STORE
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem
    from tcip_store import SchemaVersionRefused

    state = root / ".tcip" / "state"
    try:
        stems = _image_stems(root)
    except AmbiguousImageStem as exc:
        _report_stem_collision(findings, exc, seen)
        return
    except SchemaVersionRefused as exc:
        _report_band_group_version_refusal(findings, root, exc)
        return
    shard_dir = state / "review"
    if shard_dir.is_dir():
        # Shards sit one directory deep per prediction bucket, and directly here for a review
        # that named no bucket.
        for shard in shard_dir.rglob("*.json"):
            payload = _load(shard) or {}
            _note_version(findings, str(shard.relative_to(root)), REVIEW_VERDICTS_STORE, payload)
            img = payload.get("img_name", shard.stem)
            if Path(img).stem not in stems:
                findings.append(("warn", f"review shard {shard.name} references unknown image {img!r}"))


def check_region_completeness(root: Path, findings: list, *, census: dict | None) -> None:
    """A region-completeness attestation whose cell content has since been edited or deleted is a
    stale claim block calibration could otherwise trust silently; flag every disagreement between
    an attested cell's stamped digest and its current annotation content. An attestation whose
    label the run's census (``None`` when it could not be taken) read as unreadable is skipped:
    that is :func:`check_data_quality`'s finding."""
    from tcip_mcp.dataset_layout import (
        annotation_path, region_completeness_digest_key, region_completeness_digest_path,
        region_completeness_key, region_completeness_path,
    )
    from tcip_mcp.pipelines.region_completeness import stale_cells

    # Both read before either is judged, so a digest store above this reader's version is reported
    # even when the attestations hold no bucket.
    store = _read_reporting_version(
        findings, root, region_completeness_key(root), region_completeness_path(root))
    digests = _read_reporting_version(
        findings, root, region_completeness_digest_key(root),
        region_completeness_digest_path(root))
    if not store or digests is None:
        return
    for bucket, record in store.items():
        subject = record["subject"]
        label = annotation_path(root, record["date"], record["stem"])
        annotations = _readable_annotations(census, label) if census is not None else None
        if annotations is None:
            continue
        stale = stale_cells(record, annotations, digests.get(bucket, {}), subject)
        if stale:
            findings.append(("error", f"region completeness for {subject!r} on "
                            f"{record['stem']!r}: cell(s) {stale} are attested complete but "
                            "either carry no stamped digest or their annotation content has "
                            "changed since attestation; re-attest"))


def check_trait_specs(root: Path, findings: list) -> None:
    """Flag every trait spec the store fails to read, through ``load_trait_specs_with_errors``."""
    from tcip_mcp.traits import load_trait_specs_with_errors

    _specs, errors = load_trait_specs_with_errors(project_root=root)
    for e in errors:
        level = "warn" if e.get("kind") == "version_refused" else "error"
        findings.append((level, f"trait spec {e['file']} failed to load: {e['reason']}"))


def check_trait_spec_statements(root: Path, findings: list) -> None:
    """Every registered trait spec whose own trait-spec statement is not both confirmed and
    current, one of three states: absent (the recoverable gap ``author_trait_spec``'s own second
    write can leave when it fails partway), stale (the spec moved past what the statement
    recorded), or current but never confirmed by the breeder. Reads only through the storage seam.
    """
    import tcip_store as ts
    from tcip_store import DecodeError, SchemaVersionRefused

    from tcip_mcp.traits import (
        load_trait_specs_with_errors,
        trait_spec_statement_key,
        trait_spec_statement_stale,
        trait_spec_statements_scope,
    )

    # A spec that did not load is check_trait_specs' own finding.
    specs, _errors = load_trait_specs_with_errors(project_root=root)
    if not specs:
        return
    scope = trait_spec_statements_scope(root)
    for spec in specs:
        key = trait_spec_statement_key(scope, spec.name)
        try:
            statement = ts.read_versioned(key, default=None).value
        except DecodeError as exc:
            findings.append(("error", f"trait spec {spec.name!r}'s authoring statement will "
                            f"not read: {exc}"))
            continue
        except SchemaVersionRefused as exc:
            findings.append(("warn", f"trait spec {spec.name!r}'s authoring statement: {exc}"))
            continue
        if not statement:
            findings.append(("warn", f"trait spec {spec.name!r} has no authoring statement on "
                            "record; state it with revise_trait_spec(project_root=..., "
                            f"trait_name={spec.name!r}, fields={{}}, rationale=...) so the "
                            "breeder has something to confirm"))
        elif trait_spec_statement_stale(spec, statement):
            findings.append(("warn", f"trait spec {spec.name!r}'s authoring statement no longer "
                            "matches its live spec; restate it with "
                            f"revise_trait_spec(project_root=..., trait_name={spec.name!r}, "
                            "fields=..., rationale=...) and have the breeder confirm it again"))
        elif not statement.get("confirmed_by"):
            findings.append(("warn", f"trait spec {spec.name!r}'s authoring statement is current "
                            "but the breeder has not confirmed it; ask them to confirm it in the "
                            "Results tab"))


def check_project_record(root: Path, findings: list) -> None:
    """An error for a project record that is absent, damaged, or on a root the store refuses to
    read."""
    from tcip_store import StoreError

    from tcip_mcp.project_record import ProjectRecordInvalid, ProjectRecordMissing, read_record

    try:
        read_record(root)
    except (ProjectRecordMissing, ProjectRecordInvalid, StoreError, OSError) as exc:
        findings.append(("error", str(exc)))


def check_stray_state_files(root: Path, findings: list) -> None:
    """A file under ``.tcip/state`` no store claims (``tcip_mcp.stray_state.stray_state_files``) is
    an info finding naming ``delete_stray_state_file`` as the remedy, never a warn or error.

    Absent from :func:`gated_stores`: the predicate matches path templates and never consults a
    database, so on a behind-database root it can only under-report.
    """
    from tcip_mcp.stray_state import stray_state_files
    from tcip_mcp.tools.bundle import AnchorMisplaced
    from tcip_store.errors import StoreError

    try:
        strays = stray_state_files(root)
    except (AnchorMisplaced, StoreError) as exc:
        findings.append(("warn", f"the state root's accounting refused: {exc}"))
        return
    state_root = Path(root).resolve() / ".tcip" / "state"
    for path in strays:
        findings.append((
            "info",
            f".tcip/state/{path.relative_to(state_root).as_posix()}: a stray file under "
            ".tcip/state that no store claims; delete it with delete_stray_state_file if it is "
            "not needed"))


def gated_stores(root: Path) -> dict[str, tuple[tuple[Path, str], ...]]:
    """Which database-held store each file-reading check depends on, and under which root: exactly
    what the checks above read off disk, no wider.
    """
    return {
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

    A root with no database is on the file layout and every check reads the authority directly. A
    store with no counter row was never written in its database and reads current. Anything else
    stale is reported as this check being invalid rather than as clean.
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
    invalid = staleness_findings(root)
    # Shared across the three checks that independently enumerate images/, so an images/ tree
    # collision or an unreadable .bandgroup manifest is reported once, not once per check.
    ambiguous_seen: set[str] = set()
    census = _census(root, findings, ambiguous_seen)
    checks_taking_census = (check_negatives, check_data_quality, check_status_tokens,
                            check_reserved_names, check_provenance, check_region_completeness)
    for check in (check_negatives, check_data_quality, check_status_tokens, check_reserved_names,
                 check_registry, check_provenance, check_state, check_region_completeness,
                 check_trait_specs, check_trait_spec_statements,
                 check_project_record, check_stray_state_files):
        reason = invalid.get(check.__name__)
        if reason:
            findings.append(("error", f"{check.__name__} reads state as files and those files "
                            f"are behind the database that holds it ({reason}). This check is "
                            "invalid, not clean: write the files out with "
                            "'tcip export-store' and run the doctor again."))
            continue
        run: Callable[..., None] = check
        if check is check_negatives:
            run(root, findings, census=census, seen=ambiguous_seen)
        elif check is check_state:
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
