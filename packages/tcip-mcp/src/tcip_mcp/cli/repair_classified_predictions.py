"""Repair a project's classified prediction buckets into the writer rail's shape: an
``operating_point.json`` stamp carrying its ``(subject, attribute)`` pair, and per-image documents
that carry the decoded value under ``attributes[attribute]`` with the object class in ``subject``,
the shape ``write_predictions_json`` now writes and every reader now holds a bucket to.

A logged operator command: bind, walk, one outcome line per unit, exit 2 on any refusal. Its
units are prediction buckets. For each named project
root, every registered dataset (``read_datasets``) is walked and, under each dataset's own
prediction tree, every model directory and every directory one level below it that holds a stamp
record or prediction documents is a bucket. ``list_dates`` is not used: it lists the buckets under
``images/``, and a prediction date whose image folder was removed or renamed would fall outside
that walk and refuse at every reader with nothing in this report naming it. ``--bucket DIR``,
repeatable, names a directory outside that walk (a bespoke ``output_dir``, a hand-split copy); a
bare one (no stamp at all) is conformed only paired with ``--like DIR``, another bucket whose stamp
already carries the pair and a recorded ``id_map``, given in the same order as its ``--bucket``.

Per stamped bucket, in order:

1. A stamp already carrying both keys is reported conformed; its documents are still scanned for a
   value-keyed record under the classified pair, reported with re-inference as the remedy, never
   rewritten. A second run over an already-conformed tree rewrites and stamps nothing.
2. A stamp decoding with no usable pair sources its scope: the experiment the stamp names
   (``config_key``/``read_member``, read under the root being walked), else ``--like``, else the
   operator (``--subject``/``--attribute``, recorded as an operator statement). No source
   answering leaves the bucket exactly as it is.
3. A sourced detector pair (``attribute`` ``None``) is stamped only when
   ``scope_consistent_with_map`` (the rail's own predicate, called here rather than re-spelled)
   admits it over the bucket's recorded ``id_map``; otherwise the stated pair is refused and
   reported, never written over a map that says the bucket classified.
4. A sourced classified pair over a bucket recording no ``id_map``, or one keyed by a
   decimal-integer-string (the raw-index name, indistinguishable from a value), is reported with
   re-inference as the remedy; nothing is stamped or rewritten.
5. A sourced classified pair over a bucket carrying review verdicts is reported with the count;
   nothing is stamped or rewritten, since the platform never rewrites predictions a human reviewed.
   The verdict store is asked only when the bucket resolves under a real dataset root
   (``dataset_layout.dataset_root_of``, the canonical-segment test, not
   ``dataset_scope_of``'s own bucket-itself fallback). On the file backend, a bucket outside any
   dataset's canonical layout resolves to no root at all, so the guard is inoperative by
   construction. On the database backend, such a bucket's own stamp write plants
   ``<bucket>/.tcip/store.db``, so ``dataset_scope_of`` answers the bucket itself, a root no
   ``ReviewEngine`` ever writes to; the guard is inoperative there too, for the same reason. Either
   way the no-verdict-store note is reported rather than silently skipped, the same inoperative
   guard ``run_inference`` states for such a bucket.
6. Otherwise every document is read whole and every record classified, the object-class check made
   first: already carrying the object class, a mapped value under the attribute settles it
   conformed, no value settles it unconformable (a stale detector document, never mistaken for a
   rewrite even when the map declares a value spelled like the object class itself, the one-key
   blind spot). Everything else carrying a ``subject`` that is a key of the map is a rewrite;
   anything else (unconformable). Any unconformable record reports the whole bucket by file and
   index, re-inference as the remedy, neither rewritten nor stamped. A bucket with none is
   rewritten (or, if every record was already conformed, left as it is) and stamped.
7. After a rewrite, the new content's binding is checked (``verify_stamp_binding``): a count claim
   the bucket carried floors when its covered digest no longer matches, reported beside the
   stamp's own stored ``validated`` so a stale ``true`` is never read as still validated.
8. One audit entry per bucket whose documents or stamp were actually written, filed under the
   bucket's own resolved scope (``dataset_scope_of``): a real dataset root on the file backend, the
   bucket itself on the database backend when it sits outside any dataset's canonical layout (its
   own stamp write already planted ``<bucket>/.tcip/store.db``), or the platform log when no root
   resolves at all. The entry carries the documents rewritten, whether the stamp was written, the
   scope pair and its source, the free-text outcome line, and, for a rewrite under a stamp, the
   content digest before and after. A refusal, a no-op ("already conformed", "no stamp"), or a
   ``--plan`` preview writes no entry.

For each dataset root walked, this command also reports, never rewrites, ground-truth records whose
``subject`` is a key of a conformed bucket's ``id_map`` or a declared attribute value of the
dataset's own registry: a legitimate subject of that name is possible, and only a person can tell
one apart from an accept made through the Review tab before this platform recorded a scope.

``--plan`` previews every outcome without writing anything; a bucket that would change under a
real run counts toward the exit code the same way an unconformed one does.

    tcip repair-classified-predictions <project_root> [<project_root> ...]
    tcip repair-classified-predictions --plan <project_root>
    tcip repair-classified-predictions --bucket <dir> --like <scoped_dir>

Exit codes: 0 when every stamped bucket in every named root (and every named ``--bucket``) is
already conformed or was just conformed; 2 if any bucket refuses, cannot be conformed, or (under
``--plan``) would change.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from tcip_annotation import json_io
from tcip_annotation.json_io import UnreadableLabelDocument
from tcip_annotation.state import Annotation
from tcip_store import StoreError
from tcip_store.binding import bind_default

from tcip_mcp.audit import dataset_scope_of, record_event_or_raise
from tcip_mcp.class_registry import RegistryError, read_registry
from tcip_mcp.dataset_layout import (
    annotation_root, classes_path, dataset_root_of, is_bucket_name, prediction_bucket_dirs,
    prediction_root,
)
from tcip_mcp.experiments import config_key, read_member
from tcip_mcp.pipelines.resolution import (
    BucketScope, StampScopeUnstated, bucket_scope, read_operating_point_sidecar,
    scope_consistent_with_map, update_sidecar, verify_stamp_binding,
)
from tcip_mcp.prediction_buckets import (
    bucket_content_digest, bucket_key_of, bucket_stems, review_state_dir_of, verdict_count,
)
from tcip_mcp.tools.project_tools import dataset_entry_path, read_datasets

TOOL_NAME = "repair_classified_predictions"


def _looks_like_bucket(d: Path) -> bool:
    """Whether ``d`` holds a stamp record or prediction documents: this command's own definition
    of a bucket, checked before a directory is treated as one."""
    if not d.is_dir():
        return False
    if json_io.prediction_documents(d):
        return True
    return any(f.is_file() and json_io.is_sidecar_name(f.name) for f in d.glob("*.json"))


def bucket_dirs_under(dataset_root: Path) -> list[Path]:
    """Every bucket directory under ``dataset_root``'s own prediction tree, through the shared
    walk (``tcip_mcp.dataset_layout.prediction_bucket_dirs``) so this command and every other
    reader of the layout agree on what a bucket directory is.

    ``prediction_bucket_dirs`` admits every directory one level under a model directory
    regardless of its own name (it answers "where could a sidecar sit", not "is this a bucket
    this command can act on"), so a dot-prefixed date directory reaches it; filtered back out here
    by ``is_bucket_name``, since a hidden directory is never a bucket this command repairs.
    ``_looks_like_bucket`` then keeps only a directory actually holding a stamp or prediction
    documents, this command's own definition of a bucket.
    """
    root = prediction_root(dataset_root)
    if not root.is_dir():
        return []
    return [d for d in prediction_bucket_dirs(dataset_root)
            if is_bucket_name(d.name) and _looks_like_bucket(d)]


@dataclass
class StampState:
    """A bucket's own stamp, decoded once: ``kind`` is one of ``absent`` (no stamp at all),
    ``undecodable`` (a present stamp the seam will not decode), ``unstated`` (decodes but carries
    no usable ``(subject, attribute)`` pair) or ``scoped`` (a usable pair already)."""

    kind: str
    stamp: dict | None
    scope: BucketScope | None
    error: str | None = None


def read_stamp_state(bucket_dir: Path) -> StampState:
    stamp = read_operating_point_sidecar(bucket_dir)
    try:
        scope = bucket_scope(bucket_dir)
    except StampScopeUnstated:
        return StampState("unstated", stamp, None)
    except StoreError as exc:
        return StampState("undecodable", None, None, str(exc))
    if scope is None:
        return StampState("absent", None, None)
    return StampState("scoped", stamp, scope)


def _experiment_source(
    stamp: dict, *, root: Path | None,
) -> tuple[str, str | None, str | None] | None:
    """``('experiment (read under <root>)', subject, attribute)`` when the run's own experiment
    record answers both, else ``None``: a bespoke or COCO-sourced run's record, or one with no
    experiment_id at all, does not answer. The label names the root the record was read under,
    the walked project (``root``) rather than the process's own pinned platform root, so the
    report is explicit about which project's experiment store answered."""
    experiment_id = stamp.get("experiment_id")
    if not experiment_id:
        return None
    config = read_member(config_key(str(experiment_id), root=root))
    if not isinstance(config, dict):
        return None
    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict) or "subject" not in data_cfg or "attribute" not in data_cfg:
        return None
    root_note = root if root is not None else "this process's own pinned platform root"
    return f"experiment (read under {root_note})", data_cfg["subject"], data_cfg["attribute"]


def _like_source(like_dir: Path | None) -> tuple[str, str, str | None, dict[str, int]] | None:
    """``('--like <dir>', subject, attribute, id_map)`` from another bucket's own scope and
    recorded map, or ``None`` when ``like_dir`` is not given or does not answer both."""
    if like_dir is None:
        return None
    try:
        scope = bucket_scope(like_dir)
    except (StampScopeUnstated, StoreError):
        return None
    if scope is None or scope.subject is None:
        return None
    stamp = read_operating_point_sidecar(like_dir) or {}
    id_map = stamp.get("id_map")
    if not isinstance(id_map, dict) or not id_map:
        return None
    return f"--like {like_dir}", scope.subject, scope.attribute, dict(id_map)


def _operator_source(
    subject: str | None, attribute: str | None,
) -> tuple[str, str, str | None] | None:
    if subject is None:
        return None
    return "operator statement (--subject/--attribute)", subject, attribute


def source_scope(
    stamp: dict, *, root: Path | None, like_dir: Path | None,
    operator_subject: str | None, operator_attribute: str | None,
) -> tuple[str, str, str | None, dict | None] | None:
    """``(source, subject, attribute, like_id_map)`` from the first source that answers: the run's
    own experiment record, ``--like``, then the operator's stated pair. ``like_id_map`` is the
    ``--like`` bucket's own recorded map, used as this bucket's vocabulary only when the source is
    ``--like`` itself (a bare copy has none of its own to check against).

    Raises :class:`ValueError` naming the source when the experiment record answers with no
    subject (``data.subject`` recorded ``None``, with or without an attribute): unlike ``--like``
    and the operator statement, which never answer with one, the experiment source reads the
    launch config's own ``data`` section verbatim with no validation of its own, so a record
    naming no subject is a bad answer refused before any write, never one this resolver silently
    falls through past to try ``--like`` or the operator instead.
    """
    experiment = _experiment_source(stamp, root=root)
    if experiment is not None:
        source, subject, attribute = experiment
        if subject is None:
            if attribute is not None:
                raise ValueError(
                    f"the {source} names attribute {attribute!r} with no subject: a value under "
                    "no object class cannot be conformed"
                )
            raise ValueError(
                f"the {source} names no subject: a stamp cannot be written naming no object class"
            )
        return source, subject, attribute, None
    like = _like_source(like_dir)
    if like is not None:
        source, subject, attribute, id_map = like
        return source, subject, attribute, id_map
    operator = _operator_source(operator_subject, operator_attribute)
    if operator is not None:
        source, subject, attribute = operator
        return source, subject, attribute, None
    return None


def _decimal_key(id_map: dict) -> str | None:
    for key in id_map:
        if isinstance(key, str) and key.lstrip("-").isdigit():
            return key
    return None


@dataclass
class RecordVerdict:
    conformed: bool = False
    rewrite: str | None = None  # the old subject (the value) a rewrite would replace
    unconformable: bool = False


def classify_record(a: Annotation, *, subject: str, attribute: str, id_map: dict) -> RecordVerdict:
    """A record already carrying the object class (``a.subject == subject``) is conformed when a
    valid value sits under ``attribute``, unconformable otherwise (a stale detector document
    carrying the object class and no value, never a rewrite candidate): the object-class check is
    made first and settles the record either way, so the one-key blind spot (an attribute
    declaring a value spelled like the object class itself) can never read that stale document's
    bare object-class subject as a value to rewrite from."""
    if a.subject == subject:
        if a.attributes.get(attribute) in id_map:
            return RecordVerdict(conformed=True)
        return RecordVerdict(unconformable=True)
    if a.subject in id_map:
        return RecordVerdict(rewrite=a.subject)
    return RecordVerdict(unconformable=True)


def rewritten_annotation(
    a: Annotation, *, subject: str, attribute: str, old_subject: str,
) -> Annotation:
    return replace(a, subject=subject, attributes={**a.attributes, attribute: old_subject})


def unconformable_records(
    bucket_dir: Path, *, subject: str, attribute: str, id_map: dict,
) -> list[str]:
    hits: list[str] = []
    for path in json_io.prediction_documents(bucket_dir):
        for i, a in enumerate(json_io.read_annotations(str(path))):
            v = classify_record(a, subject=subject, attribute=attribute, id_map=id_map)
            if v.unconformable:
                hits.append(f"{path} record {i}: subject {a.subject!r}")
    return hits


def rewrite_bucket(bucket_dir: Path, *, subject: str, attribute: str, id_map: dict) -> int:
    """Rewrite every value-keyed record in ``bucket_dir``'s documents into the conformed shape,
    preserving each document's own width/height. Returns the number of documents rewritten."""
    rewritten = 0
    for path in json_io.prediction_documents(bucket_dir):
        document = json_io.load_label_document(str(path))
        annotations = json_io.annotations_of_document(document)
        verdicts = [classify_record(a, subject=subject, attribute=attribute, id_map=id_map)
                   for a in annotations]
        if not any(v.rewrite is not None for v in verdicts):
            continue
        new_annotations = [
            rewritten_annotation(a, subject=subject, attribute=attribute, old_subject=v.rewrite)
            if v.rewrite is not None else a
            for a, v in zip(annotations, verdicts)
        ]
        json_io.write_annotations(
            str(path), new_annotations, int(document.get("width") or 0),
            int(document.get("height") or 0), keep_empty=True,
        )
        rewritten += 1
    return rewritten


def _stamp_completed(bucket_dir: Path, *, subject: str | None, attribute: str | None) -> None:
    update_sidecar(bucket_dir, lambda cur: {**cur, "subject": subject, "attribute": attribute})


def _emit_conform_audit(
    bucket_dir: Path, *, documents_rewritten: int, stamp_written: bool,
    subject: str | None, attribute: str | None, source: str, outcome: str, scope: Path | None,
    digest_before: str | None = None, digest_after: str | None = None,
) -> None:
    """The one audit entry a bucket write earns, filed under ``scope`` or the platform log when
    ``scope`` is ``None``, called at the exact point each write happens rather than decided again
    by a caller from the outcome text: a refusal never reaches here, so an entry is written only
    for a bucket whose documents or stamp actually changed.

    ``scope`` is the caller's own ``dataset_scope_of(bucket_dir)``, resolved once before the first
    write into the bucket and passed in rather than re-resolved here: under the database backend a
    stamp write plants ``<bucket>/.tcip/store.db`` (a document write goes through ``put_blob`` to
    the file backend and plants nothing), so a ``dataset_scope_of`` read taken after the stamp
    write answers the bucket as its own dataset root, which is the seam's own behaviour, not this
    command's to change. ``outcome`` is the free-text line the caller is about
    to return, carried on the entry beside its structured fields rather than left to reach only
    stdout. ``digest_before``/``digest_after`` are given only for the one branch that rewrites
    documents under a stamp (rule 7's binding check runs there); a stamp-only completion or a bare
    directory's document-only rewrite carries neither, since nothing there floors a claim.
    """
    extra: dict = {
        "documents_rewritten": documents_rewritten, "stamp_written": stamp_written,
        "subject": subject, "attribute": attribute, "source": source, "outcome": outcome,
    }
    if digest_before is not None:
        extra["digest_before"] = digest_before
        extra["digest_after"] = digest_after
    record_event_or_raise(
        TOOL_NAME, {"bucket": str(bucket_dir)}, status="ok", scope=scope,
        **extra,
    )


def scan_value_keyed_records(bucket_dir: Path, scope: BucketScope, id_map: dict | None) -> list[str]:
    """Every record under ``bucket_dir`` whose ``subject`` is a key of ``id_map`` rather than the
    scope's own object class: reported as a hand edit, never rewritten (rule 1)."""
    if not id_map:
        return []
    hits: list[str] = []
    for path in json_io.prediction_documents(bucket_dir):
        for i, a in enumerate(json_io.read_annotations(str(path))):
            if a.subject != scope.subject and a.subject in id_map:
                hits.append(f"{path} record {i}: subject {a.subject!r} is a value of this "
                            f"bucket's own id_map, not its object class {scope.subject!r}")
    return hits


def conform_bare_bucket(
    bucket_dir: Path, *, like_dir: Path | None, plan: bool,
) -> tuple[str, bool, bool]:
    """Conform a bare directory (no stamp at all) named with ``--bucket``, paired with ``--like``.

    A bare directory this command repairs holds a classified bucket's hand-split copy: records
    whose ``subject`` is the value, waiting to move under an attribute. ``--like`` must therefore
    name a classified bucket (its own attribute is not ``None``); a detector ``--like`` is refused
    by name, since a detector bucket's records already carry the object class in ``subject`` and
    its scope names no attribute to rewrite this copy's values under.

    Returns ``(outcome, refused, changed)``; ``changed`` is true exactly when a document was
    actually rewritten, so a second run over an already-conformed copy reports and audits nothing.
    """
    like = _like_source(like_dir)
    if like is None:
        return ("no stamp; a bare directory named with --bucket is conformed only paired with "
                "--like naming a scoped bucket with a recorded id_map", True, False)
    _source, subject, attribute, id_map = like
    if attribute is None:
        return (
            f"refused, --like {like_dir} is a detector bucket (its scope names no attribute): a "
            "bare directory this command repairs holds a classified bucket's value-in-subject "
            "records, which need a --like naming the attribute to move the value under", True,
            False,
        )
    unconformable = unconformable_records(bucket_dir, subject=subject, attribute=attribute,
                                          id_map=id_map)
    if unconformable:
        return (f"unconformable, re-infer this bucket: {'; '.join(unconformable)}", True, False)
    if plan:
        return (f"would rewrite from --like {like_dir}'s vocabulary, no stamp written (a bare "
                "directory's regime stays the caller's own statement)", True, False)
    dataset_root = dataset_scope_of(bucket_dir)
    docs_rewritten = rewrite_bucket(bucket_dir, subject=subject, attribute=attribute, id_map=id_map)
    if docs_rewritten == 0:
        return (f"conformed; already matches --like {like_dir}'s vocabulary, nothing rewritten",
                False, False)
    outcome = (f"rewrote {docs_rewritten} document(s) from --like {like_dir}'s vocabulary, no "
              "stamp written (a bare directory's regime stays the caller's own statement)")
    _emit_conform_audit(
        bucket_dir, documents_rewritten=docs_rewritten, stamp_written=False,
        subject=subject, attribute=attribute, source=f"--like {like_dir}", outcome=outcome,
        scope=dataset_root,
    )
    return (outcome, False, True)


def conform_bucket(
    bucket_dir: Path, *, root: Path | None, plan: bool, like_dir: Path | None,
    operator_subject: str | None, operator_attribute: str | None, is_bare_named: bool,
) -> tuple[str, bool, dict | None, bool]:
    """Conform one bucket. Returns ``(outcome, refused, id_map_if_conformed, changed)``.

    ``changed`` is true exactly for a bucket whose documents or stamp were actually written; a
    refusal, a no-op ("already conformed", "no stamp"), or a ``--plan`` preview is always false.
    The one audit entry a changed bucket earns (:func:`_emit_conform_audit`) is written at the
    exact point the change happens, not decided again here from the returned outcome text.
    """
    state = read_stamp_state(bucket_dir)

    if state.kind == "undecodable":
        return (f"refused, {state.error}; re-infer or hand-repair this bucket's own "
                "operating_point.json through the store's own tools", True, None, False)

    if state.kind == "absent":
        if not is_bare_named:
            return ("no stamp; not a bucket this command repairs", False, None, False)
        outcome, refused, changed = conform_bare_bucket(bucket_dir, like_dir=like_dir, plan=plan)
        return (outcome, refused, None, changed)

    if state.kind == "scoped":
        assert state.scope is not None
        hits = scan_value_keyed_records(bucket_dir, state.scope, (state.stamp or {}).get("id_map"))
        if hits:
            return (f"conformed; {len(hits)} value-keyed record(s) reported for re-inference: "
                    + "; ".join(hits), False, None, False)
        return ("conformed", False, None, False)

    assert state.kind == "unstated"
    stamp = state.stamp or {}
    try:
        sourced = source_scope(
            stamp, root=root, like_dir=like_dir,
            operator_subject=operator_subject, operator_attribute=operator_attribute,
        )
    except ValueError as exc:
        return (f"refused, {exc}", True, None, False)
    if sourced is None:
        return ("stamp decodes with no usable (subject, attribute) pair, and no source (the run's "
                "own experiment record, --like, the operator) answered; left as it is",
                True, None, False)
    # source_scope raises rather than returning a None subject, so subject is a str from here on.
    source, subject, attribute, like_id_map = sourced

    if attribute is None:
        recorded_map = stamp.get("id_map")
        reason = scope_consistent_with_map(subject, None, recorded_map)
        if reason is not None:
            return (f"refused, the {source} names a detector pair but {reason}", True, None, False)
        if plan:
            return (f"would stamp the detector pair ({subject!r}, None) from the {source}; "
                    "documents untouched", True, None, False)
        dataset_root = dataset_scope_of(bucket_dir)
        _stamp_completed(bucket_dir, subject=subject, attribute=None)
        outcome = (f"stamped the detector pair ({subject!r}, None) from the {source}; documents "
                  "untouched")
        _emit_conform_audit(
            bucket_dir, documents_rewritten=0, stamp_written=True,
            subject=subject, attribute=None, source=source, outcome=outcome, scope=dataset_root,
        )
        return (outcome, False, None, True)

    recorded_map = stamp.get("id_map")
    if recorded_map:
        if like_id_map is not None and like_id_map != recorded_map:
            return (
                f"refused, --like's id_map {like_id_map!r} disagrees with this bucket's own "
                f"recorded id_map {recorded_map!r}: a bucket whose stamp records a map is "
                "rewritten against its own, never --like's; --like supplies a vocabulary only "
                "for a bare directory or a stamp recording none", True, None, False,
            )
        id_map = recorded_map
    else:
        id_map = like_id_map
    if not isinstance(id_map, dict) or not id_map:
        return (f"refused, the {source} names a classified pair ({subject!r}, {attribute!r}) but "
                "this bucket records no id_map to decode by; re-infer this run", True, None, False)
    bad_key = _decimal_key(id_map)
    if bad_key is not None:
        return (f"refused, this bucket's own id_map carries the decimal key {bad_key!r}, "
                "indistinguishable from a raw-index name; re-infer this run", True, None, False)

    dataset_root = dataset_scope_of(bucket_dir)
    no_dataset_note = ""
    if dataset_root is not None and dataset_root_of(bucket_dir) is not None:
        vcount = verdict_count(review_state_dir_of(dataset_root), bucket_key_of(bucket_dir),
                                bucket_stems(bucket_dir))
        if vcount:
            return (f"refused, {vcount} review verdict(s) recorded against this bucket; the "
                    "platform never rewrites predictions a human reviewed. Re-run inference into "
                    "a fresh bucket and review it anew", True, None, False)
    else:
        no_dataset_note = " (this bucket sits under no dataset root, so no verdict store guards it)"

    unconformable = unconformable_records(bucket_dir, subject=subject, attribute=attribute,
                                          id_map=id_map)
    if unconformable:
        return (f"refused, {len(unconformable)} unconformable record(s), re-inference is the "
                "remedy: " + "; ".join(unconformable), True, None, False)

    if plan:
        return (f"would conform the classified pair ({subject!r}, {attribute!r}) from the "
                f"{source}{no_dataset_note}", True, None, False)

    old_digest = bucket_content_digest(bucket_dir)
    docs_rewritten = rewrite_bucket(bucket_dir, subject=subject, attribute=attribute, id_map=id_map)
    _stamp_completed(bucket_dir, subject=subject, attribute=attribute)
    if docs_rewritten == 0:
        outcome = (f"stamped the classified pair ({subject!r}, {attribute!r}) from the "
                  f"{source}{no_dataset_note}; every record was already conformed")
        _emit_conform_audit(
            bucket_dir, documents_rewritten=0, stamp_written=True,
            subject=subject, attribute=attribute, source=source, outcome=outcome,
            scope=dataset_root, digest_before=old_digest, digest_after=old_digest,
        )
        return (outcome, False, id_map, True)
    new_digest = bucket_content_digest(bucket_dir)
    new_stamp = read_operating_point_sidecar(bucket_dir) or {}
    binding = verify_stamp_binding(new_stamp, bucket_dir, document="operating_point")
    floor_note = ""
    if binding.claimed and not binding.ok:
        floor_note = (
            f"; a count claim over this bucket floors: {binding.note} (stored validated="
            f"{new_stamp.get('validated')!r}, effective binding=floored; re-earn through "
            "calibrate_count_operating_point)"
        )
    outcome = (f"rewrote {docs_rewritten} document(s) and stamped the classified pair "
              f"({subject!r}, {attribute!r}) from the {source}{no_dataset_note}; digest "
              f"{old_digest} -> {new_digest}{floor_note}")
    _emit_conform_audit(
        bucket_dir, documents_rewritten=docs_rewritten, stamp_written=True,
        subject=subject, attribute=attribute, source=source, outcome=outcome,
        scope=dataset_root, digest_before=old_digest, digest_after=new_digest,
    )
    return (outcome, False, id_map, True)


def ground_truth_candidates(dataset_root: Path, conformed_id_maps: list[dict]) -> list[str]:
    """Every ground-truth record whose ``subject`` is a key of a conformed bucket's ``id_map`` or a
    declared attribute value of the dataset's own registry: reported as a candidate, never
    rewritten (a legitimate subject of that name is possible)."""
    candidate_names: set[str] = set()
    for id_map in conformed_id_maps:
        candidate_names.update(id_map)
    cp = classes_path(dataset_root)
    if cp.is_file():
        try:
            registry = read_registry(cp)
        except (OSError, RegistryError):
            registry = None
        if registry is not None:
            for subject in registry.subjects:
                for attribute in subject.attributes:
                    candidate_names.update(attribute.values)
    if not candidate_names:
        return []
    root = annotation_root(dataset_root)
    if not root.is_dir():
        return []
    hits: list[str] = []
    for d in [root, *sorted(p for p in root.iterdir() if p.is_dir())]:
        for path in json_io.prediction_documents(d):
            try:
                annotations = json_io.read_annotations(str(path))
            except UnreadableLabelDocument:
                continue
            for i, a in enumerate(annotations):
                if a.subject in candidate_names:
                    hits.append(f"{path} record {i}: subject {a.subject!r} is a candidate "
                                "(a conformed bucket's value, or a declared attribute value)")
    return hits


def process_project_root(
    root: Path, *, plan: bool, operator_subject: str | None, operator_attribute: str | None,
) -> tuple[list[str], bool]:
    outcomes: list[str] = []
    refused = False
    changed_count = 0
    unchanged_count = 0
    if not (root / ".tcip").is_dir():
        return ([f"{root}: refused, no .tcip directory found; not a project root; no summary "
                 "applies"], True)
    datasets = read_datasets(root)
    if not datasets:
        return ([f"{root}: no registered datasets; no summary applies"], False)
    for entry in datasets:
        dataset_root = dataset_entry_path(root, entry)
        conformed_id_maps: list[dict] = []
        for bucket_dir in bucket_dirs_under(dataset_root):
            # The audit entry for a changed bucket is written inside conform_bucket itself, at
            # the exact point each write happens; this loop only collects outcomes and counts.
            outcome, this_refused, id_map, changed = conform_bucket(
                bucket_dir, root=root, plan=plan, like_dir=None,
                operator_subject=operator_subject, operator_attribute=operator_attribute,
                is_bare_named=False,
            )
            outcomes.append(f"{bucket_dir}: {outcome}")
            if changed:
                changed_count += 1
            else:
                unchanged_count += 1
            if this_refused:
                refused = True
            elif id_map is not None:
                conformed_id_maps.append(id_map)
        for hit in ground_truth_candidates(dataset_root, conformed_id_maps):
            outcomes.append(f"{dataset_root}: ground-truth candidate, {hit}")
    outcomes.append(
        f"{root}: {changed_count} bucket(s) changed, {unchanged_count} left as they were")
    return outcomes, refused


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], prog=prog)
    ap.add_argument("roots", nargs="*", type=Path)
    ap.add_argument("--plan", action="store_true", help="preview only; nothing is written")
    ap.add_argument("--bucket", action="append", default=[], type=Path,
                    help="a bucket directory outside the project walk, repeatable")
    ap.add_argument("--like", action="append", default=[], type=Path,
                    help="paired with --bucket in order, a scoped bucket a bare copy is like")
    ap.add_argument("--subject", default=None, help="operator-stated subject for an unsourced pair")
    ap.add_argument("--attribute", default=None,
                    help="operator-stated attribute for an unsourced pair (omit for a detector pair)")
    args = ap.parse_args(argv)

    if args.like and len(args.like) != len(args.bucket):
        ap.error("--like must be given once per --bucket, in the same order")

    bind_default()

    refused_any = False
    for root in args.roots:
        root = root.resolve()
        outcomes, refused = process_project_root(
            root, plan=args.plan, operator_subject=args.subject, operator_attribute=args.attribute,
        )
        if refused:
            refused_any = True
        for line in outcomes:
            print(line)

    bucket_changed_count = 0
    bucket_unchanged_count = 0
    for i, bucket_dir in enumerate(args.bucket):
        bucket_dir = bucket_dir.resolve()
        like_dir = args.like[i].resolve() if args.like else None
        # The audit entry for a changed bucket is written inside conform_bucket itself.
        outcome, this_refused, _id_map, changed = conform_bucket(
            bucket_dir, root=None, plan=args.plan, like_dir=like_dir,
            operator_subject=args.subject, operator_attribute=args.attribute, is_bare_named=True,
        )
        print(f"{bucket_dir}: {outcome}")
        if changed:
            bucket_changed_count += 1
        else:
            bucket_unchanged_count += 1
        if this_refused:
            refused_any = True
    if args.bucket:
        print(f"--bucket: {bucket_changed_count} bucket(s) changed, "
              f"{bucket_unchanged_count} left as they were")

    return 2 if refused_any else 0


if __name__ == "__main__":
    sys.exit(main())
