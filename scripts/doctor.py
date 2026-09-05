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

def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _note_version(findings: list, where: str, store: str, doc) -> None:
    """Report, never refuse, a document whose schema_version this reader does not accept.

    The doctor reads raw files and reports findings; it never acts on a document's content, so
    an unsupported version is a finding here, not the hard refusal a reader about to train or
    measure on the document would raise.
    """
    from tcip_store import SchemaVersionRefused, check_schema_version, get_descriptor

    try:
        check_schema_version(get_descriptor(store), doc)
    except SchemaVersionRefused as exc:
        findings.append(("warn", f"{where}: {exc}"))


def _image_stems(root: Path) -> dict[str, str]:
    """stem -> file name for every image under images/ (the flat form and every date bucket),
    through the platform's own bucket enumeration, so a stem collision within one bucket refuses
    here too rather than this walk silently keeping one raw file of the pair. A stem present in
    more than one bucket keeps the latest bucket's file, buckets visited in sorted (chronological,
    for ISO dates) order."""
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


def check_negatives(root: Path, findings: list) -> None:
    """A negative is empty labels + human Complete, per subject: flag every disk/status disagreement.

    Labels are one name-based file per image (all subjects); a confirmed negative is scoped to a
    subject and date, so the disagreement is checked per subject present in the file. An
    image-level record (a subject with no geometry) counts as content for that subject, the same
    rule ``annotations_hold_subject`` applies everywhere else this question is asked.

    Reads the status store through the same seam ``check_data_quality`` reads it through, so
    the two agree on one root under whichever backend the process is bound to. Not gated in
    ``gated_stores`` for that reason: the label files this check also walks are blobs, a real
    file under both backends, so neither read can be stale relative to the bound backend the way
    a raw document read would be behind the database.
    """
    from tcip_annotation import json_io
    from tcip_annotation.json_io import UnreadableLabelDocument
    from tcip_annotation.review_engine import BASELINE_DIRNAME
    from tcip_mcp.dataset_layout import (
        annotation_date, annotation_root, annotations_hold_subject, bucket_subject_date,
        confirmed_negative_names_any_subject, is_confirmed_negative, normalize_status_store,
        read_image_status_store, resolve_image_name,
    )
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem
    from tcip_store import StoreError

    # Confirmations are dataset-native, and this check already assumes root == dataset_root.
    try:
        by_bucket = normalize_status_store(read_image_status_store(root))
    except StoreError as exc:
        # The same soft-rail posture check_status_tokens already takes on this file: a reporter
        # names what it could not verify rather than blocking the whole run over it.
        findings.append(("warn", f"the image status store will not read ({exc}); negatives "
                         "cannot be verified against it"))
        return
    try:
        stems = _image_stems(root)
    except AmbiguousImageStem as exc:
        findings.append(("error", str(exc)))
        return
    ann_root = annotation_root(root)
    neg_names = confirmed_negative_names_any_subject(by_bucket)
    ambiguous_reported: set[str] = set()

    for label in ann_root.rglob("*.json") if ann_root.is_dir() else []:
        if BASELINE_DIRNAME in label.parts:
            continue
        try:
            anns = json_io.read_annotations(str(label))
        except UnreadableLabelDocument as exc:
            findings.append(("error", f"{label.relative_to(root)}: label file will not read: {exc}"))
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
        if not anns and (name is None or name not in neg_names):
            findings.append(("warn", f"{label.relative_to(root)}: empty label but not a confirmed "
                            "negative for any subject; excluded from training (delete the file, or "
                            "mark the image Complete for the subject it should be a negative of)"))
        if label.stem not in stems:
            findings.append(("warn", f"{label.relative_to(root)}: label has no matching image"))

    # Images with no label record at all: excluded from training, so a breeder who labelled 30 of
    # 400 trains on 30. Reported as one line, not one per image (the dominant case at small scale).
    labelled = {p.stem for p in ann_root.rglob("*.json")
                if BASELINE_DIRNAME not in p.parts} if ann_root.is_dir() else set()
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


def check_data_quality(root: Path, findings: list) -> None:
    """Per-file annotation quality, any supported format, folded in from the retired per-file
    quality tool: stem matching between images and labels, an empty per-image
    label with no human confirmation the image is a negative, a file whose format cannot be
    determined, and a file present but unreadable. Format is decided per label file, never once
    for the whole dataset, so a store mixing shapes cannot report clean because one file's shape
    happened to be detected first.

    Reuses ``data_tools._scan_dataset``'s own image/label census and root-candidate walk rather
    than re-deriving them, so this check and ``scan_dataset`` can never silently disagree on what
    counts as a label. Some overlap with ``check_negatives`` is expected (an orphan or
    unconfirmed-empty per-image label can be named by both); this check's own value is the
    per-file format decision and COCO-aware validation neither ``check_negatives`` nor
    ``check_reserved_names`` carries.

    Reads the status store through the same seam ``check_negatives`` reads it through, so the
    two agree on one root under whichever backend the process is bound to. Not gated in
    ``gated_stores`` for that reason: the label files this check reads through ``_scan_dataset``
    are blobs, a real file under both backends, so neither read can be stale relative to the
    bound backend the way a raw document read would be behind the database.
    """
    from tcip_annotation.format_io import detect_format
    from tcip_annotation.json_io import (
        UnreadableLabelDocument, load_json_document, read_annotations as read_labels,
    )
    from tcip_mcp.dataset_layout import (
        annotation_date, confirmed_negative_names_any_subject, normalize_status_store,
        read_image_status_store, resolve_image_name,
    )
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem
    from tcip_mcp.tools.data_tools import _scan_dataset
    from tcip_store import StoreError

    try:
        scan = _scan_dataset(str(root))
    except UnreadableLabelDocument as exc:
        findings.append(("error", f"data quality scan: {exc}"))
        return
    except AmbiguousImageStem as exc:
        findings.append(("error", str(exc)))
        return

    image_stems = {Path(p).stem for p in scan["images"]}
    ambiguous_reported: set[str] = set()

    try:
        negatives = confirmed_negative_names_any_subject(
            normalize_status_store(read_image_status_store(str(root)))
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
        try:
            file_fmt = detect_format(label_path)
        except ValueError as exc:
            findings.append(("error", f"{rel}: cannot determine annotation format: {exc}"))
            continue
        except UnreadableLabelDocument as exc:
            findings.append(("error", f"{rel}: label file will not read: {exc}"))
            continue

        if file_fmt == "json":
            stem = label.stem
            if stem not in image_stems:
                findings.append(("error", f"{rel}: no matching image"))
            try:
                anns = read_labels(label_path)
            except UnreadableLabelDocument as exc:
                findings.append(("error", f"{rel}: label file will not read: {exc}"))
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
        elif file_fmt == "coco":
            try:
                coco = load_json_document(label_path)
                coco_fnames = {img.get("file_name", "") for img in coco.get("images", [])}
                for fn in coco_fnames:
                    if Path(fn).stem not in image_stems:
                        findings.append(("warn", f"{rel}: COCO image {fn!r} not found in "
                                        "images dir"))
            except Exception as exc:  # noqa: BLE001 - a malformed COCO document, named and moved past
                findings.append(("error", f"{rel}: COCO parse error: {exc}"))


def check_reserved_names(root: Path, findings: list) -> None:
    """Flag every image and label document whose stem is reserved for a prediction bucket's own
    provenance stamp (``tcip_annotation.json_io.is_sidecar_name``).

    Ingest and band grouping both refuse to mint one, but data not brought in through the
    platform (a hand-placed file, an older export) can still carry one, and every walk that
    enumerates a bucket through ``prediction_documents`` silently excludes it rather than raising.
    This check walks with ``rglob``, so it sees exactly what those walks hide.
    """
    from tcip_annotation.json_io import is_sidecar_name
    from tcip_mcp.dataset_layout import annotation_root, image_root
    from tcip_mcp.pipelines.image_utils import IMAGE_EXTS

    images = image_root(root)
    if images.is_dir():
        for p in sorted(images.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and is_sidecar_name(f"{p.stem}.json"):
                findings.append(("error", f"{p.relative_to(root)}: image stem is reserved for a "
                                "prediction bucket's own provenance stamp; its label can never be "
                                "read through any bucket walk"))

    ann_root = annotation_root(root)
    if ann_root.is_dir():
        for p in sorted(ann_root.rglob("*.json")):
            if p.is_file() and is_sidecar_name(p.name):
                findings.append(("error", f"{p.relative_to(root)}: label filename is reserved for "
                                "a prediction bucket's own provenance stamp; it is excluded from "
                                "every bucket walk and its annotations are unreadable through them"))


def check_status_tokens(root: Path, findings: list) -> None:
    """Flag two ways the status store can no longer be trusted at face value: an entry a reader
    doesn't recognize (dropped silently by any merge), and a stored ``"complete"`` whose label
    file holds no annotation of the confirmed subject (a stale token). The GUI's own hydrate
    surfaces the same disagreement for re-confirmation but never rewrites it, so this finding may
    already be one the breeder has seen and not yet acted on, not a new discovery. A report, not a
    rewrite either way: the record names who confirmed it, and this doctor is not that person.
    """
    from tcip_annotation import json_io
    from tcip_annotation.json_io import UnreadableLabelDocument
    from tcip_mcp.dataset_layout import (
        IMAGE_STATUS_STORE, annotation_path, annotations_hold_subject, bucket_subject_date,
        image_status_path, status_confirmations, unreadable_status_entries,
    )

    raw = _load(image_status_path(root))
    _note_version(findings, str(image_status_path(root).relative_to(root)), IMAGE_STATUS_STORE, raw)
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
            try:
                anns = json_io.read_annotations(str(label)) if label.is_file() else []
            except UnreadableLabelDocument as exc:
                findings.append(("error", f"{bucket}/{name}: label file will not read: {exc}"))
                continue
            if not annotations_hold_subject(anns, subject):
                findings.append(("warn", f"{bucket}/{name}: status says 'complete' but the label "
                                f"file holds no {subject!r} annotation; re-confirm"))


TEMP_TREE_MARKERS = ("pytest-of-", "\\Temp\\", "/Temp/")
"""Path fragments that place a registered checkpoint inside a test or temp tree."""


def check_registry(root: Path, findings: list) -> None:
    """Flag registered models whose checkpoint is missing or points into a test/temp tree, and
    every prediction bucket whose stamp names a checkpoint digest no registry entry carries.

    A checkpoint resolving under the project root never triggers the temp-tree marker scan, even
    when the root itself sits under one (a fixture, a sandboxed workspace): a marker anywhere in
    that path is then a fact about the root's own location, not the checkpoint's, and the project
    is legitimate work, not pollution. Only a checkpoint the root does not contain is scanned, the
    genuinely stray case the marker exists to catch.

    Every way the registry index can refuse to be read comes out as a finding, not as a
    traceback: the doctor's contract is an exit code and a list, and a check that dies takes the
    whole run's findings and exit code with it. A store that will not decode and a database file
    that will not open are both "this could not be checked", which is not the same answer as clean.
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
    entry_list = entries if isinstance(entries, list) else []
    root_resolved = Path(root).resolve()
    for m in entry_list:
        ckpt_raw = m.get("checkpoint_path", "")
        if "metrics_source" not in m:
            findings.append(("warn", f"{m.get('name')!r} in the model registry carries no "
                            "metrics_source (predates the field, and experiment_id with it); "
                            "conform it with scripts/conform_registry_experiment_id.py, then "
                            "re-register it through register_model"))
        if "experiment_id" not in m:
            findings.append(("warn", f"{m.get('name')!r} in the model registry carries no "
                            "experiment_id (predates the producer-binding field); conform it "
                            "with scripts/conform_registry_experiment_id.py"))
        if not ckpt_raw:
            continue
        # Existence resolves first; the temp-tree marker scan runs over the resolved string.
        try:
            resolved = resolved_registry_path(root, ckpt_raw)
        except (RegistryPathEmpty, RegistryPathTraversal) as exc:
            findings.append(("error", f"registry entry {m.get('name')!r} checkpoint_path "
                            f"could not be resolved: {exc}"))
            continue
        ckpt = str(resolved)
        stray = not is_at_or_under(resolved, root_resolved) and any(
            marker in ckpt for marker in TEMP_TREE_MARKERS)
        if stray:
            findings.append(("error", f"registry entry {m.get('name')!r} points at a test/temp "
                            f"checkpoint: {ckpt}"))
        elif not Path(ckpt).is_file():
            findings.append(("error", f"registry entry {m.get('name')!r} checkpoint missing: {ckpt}"))

    # A bucket predating the checkpoint-digest rail may name a digest no entry carries; visible
    # here, never floored. The sidecar is read through the store seam, not a plain-file glob.
    registered_shas = {m.get("sha256") for m in entry_list}
    for bucket in prediction_bucket_dirs(root):
        sha = (read_operating_point_sidecar(bucket) or {}).get("checkpoint_sha256")
        if sha and sha not in registered_shas:
            findings.append(("warn", f"{bucket.relative_to(root)}: prediction bucket's "
                            f"stamp names checkpoint {sha}, which no registry entry "
                            "names; register the checkpoint to make this bucket's "
                            "provenance verifiable going forward."))


def check_provenance(root: Path, findings: list) -> None:
    from tcip_annotation.json_io import ANNOTATIONS_KEY, UnreadableLabelDocument, load_label_document
    from tcip_annotation.review_engine import BASELINE_DIRNAME
    from tcip_mcp.dataset_layout import annotation_root
    from tcip_mcp.pipelines.model_build import SNAPSHOT_MANIFEST_STORE

    ann_root = annotation_root(root)
    unstamped = 0
    for label in ann_root.rglob("*.json") if ann_root.is_dir() else []:
        if BASELINE_DIRNAME in label.parts:
            continue
        try:
            data = load_label_document(label)
        except UnreadableLabelDocument as exc:
            findings.append(("error", f"{label.relative_to(root)}: label file will not read: {exc}"))
            continue
        for o in data.get(ANNOTATIONS_KEY, []) if isinstance(data.get(ANNOTATIONS_KEY), list) else []:
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
            _note_version(
                findings, str(manifest_path.relative_to(root)), SNAPSHOT_MANIFEST_STORE, manifest
            )
            missing = manifest.get("missing") or []
            errors = manifest.get("snapshot_errors") or []
            if missing or errors:
                findings.append(("warn", f"{manifest_path.relative_to(root)}: source snapshot "
                                f"incomplete: {len(missing)} missing file(s), "
                                f"{len(errors)} import error(s)"))


def check_state(root: Path, findings: list) -> None:
    from tcip_annotation.review_engine import REVIEW_VERDICTS_STORE
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem

    state = root / ".tcip" / "state"
    try:
        stems = _image_stems(root)
    except AmbiguousImageStem as exc:
        findings.append(("error", str(exc)))
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


def check_region_completeness(root: Path, findings: list) -> None:
    """A region-completeness attestation whose cell content has since been edited or deleted is a
    stale claim block calibration could otherwise trust silently; flag every disagreement between
    an attested cell's stamped digest and its current annotation content."""
    from tcip_annotation.json_io import UnreadableLabelDocument
    from tcip_mcp.dataset_layout import (
        REGION_COMPLETENESS_DIGEST_STORE, REGION_COMPLETENESS_STORE, bucket_subject_date,
        normalize_region_completeness_store, region_completeness_digest_path,
        region_completeness_path, unreadable_completeness_entries,
    )
    from tcip_mcp.pipelines.region_completeness import stale_cells

    raw = _load(region_completeness_path(root))
    _note_version(
        findings, str(region_completeness_path(root).relative_to(root)),
        REGION_COMPLETENESS_STORE, raw,
    )
    unreadable = unreadable_completeness_entries(raw)
    if unreadable:
        findings.append(("warn", f"{len(unreadable)} region-completeness entr"
                        f"{'y is' if len(unreadable) == 1 else 'ies are'} in a shape this reader "
                        f"does not recognize, starting with {unreadable[:3]}"))

    # Read and version-check the digest file before the no-recognized-bucket early return below,
    # so a bumped digest shape is still reported even when the main store parses to nothing.
    digests = _load(region_completeness_digest_path(root))
    _note_version(
        findings,
        str(region_completeness_digest_path(root).relative_to(root)),
        REGION_COMPLETENESS_DIGEST_STORE,
        digests,
    )

    store = normalize_region_completeness_store(raw)
    if not store:
        return
    if not isinstance(digests, dict):
        digests = {}
    for bucket, record in store.items():
        subject = record.get("subject")
        if not isinstance(subject, str) or not subject:
            subject, _ = bucket_subject_date(bucket)
        stamped = digests.get(bucket)
        try:
            stale = stale_cells(root, record, stamped if isinstance(stamped, dict) else {}, subject)
        except UnreadableLabelDocument as exc:
            findings.append(("error", f"region completeness for {subject!r} on "
                            f"{record.get('stem')!r}: label file will not read: {exc}"))
            continue
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
        # A version refusal is a soft-rail finding, the doctor's own posture; every other reason
        # (malformed JSON, an invalid config) still blocks, unchanged from before this family.
        level = "warn" if e.get("kind") == "version_refused" else "error"
        findings.append((level, f"trait spec {e['file']} failed to load: {e['reason']}"))


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


def check_project_record(root: Path, findings: list) -> None:
    """A project's own site, from ``tcip_mcp.project_record``: an absent record is the accepted
    standing state of a project that predates the field, so it warns; a damaged record or a root
    the store refuses to read is a check that could not run, the same line ``check_registry``
    draws between could-not-be-checked and clean, so it errors.

    Deliberately absent from ``gated_stores()`` for the same reason as ``check_trait_specs``: it
    reads entirely through the storage seam, so it cannot be stale relative to the backend it
    reads from and needs no export to be valid.
    """
    from tcip_store import StoreError

    from tcip_mcp.project_record import ProjectRecordInvalid, ProjectRecordMissing, read_record

    try:
        read_record(root)
    except ProjectRecordMissing as exc:
        findings.append(("warn", str(exc)))
    except (ProjectRecordInvalid, StoreError, OSError) as exc:
        findings.append(("error", str(exc)))


def gated_stores(root: Path) -> dict[str, tuple[tuple[Path, str], ...]]:
    """Which database-held store each file-reading check depends on, and under which root.

    Exactly what the checks above read off disk, no wider: a store the doctor never reads
    cannot make a check it does not run report anything. Live-state stores are not doctor
    inputs and are not here. ``check_negatives`` and ``check_data_quality`` read
    ``image_status`` through ``read_image_status_store``, the storage seam, not the raw
    document off disk, so a database-backed root's un-exported ``image_status.json`` never
    makes either of them stale and neither belongs here.
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
    for check in (check_negatives, check_data_quality, check_status_tokens, check_reserved_names,
                 check_registry, check_provenance, check_state, check_region_completeness,
                 check_trait_specs, check_trait_spec_statements, check_project_record):
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
