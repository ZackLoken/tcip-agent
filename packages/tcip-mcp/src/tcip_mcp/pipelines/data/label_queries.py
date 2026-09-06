"""The label-store and registry query library: reads a dataset's per-image JSON or assembled
COCO labels, its ``classes.json`` registry, and its confirmed-negative image-status store, and
assembles them into the boxes/labels/COCO shapes the dataset classes in ``datasets.py`` and the
outside-layer tools (calibration, evaluation, inference, training) consume. Holds no ``Dataset``
subclass and no tensor conversion; those stay in ``datasets.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from tcip_mcp.pipelines.image_utils import list_logical_images, logical_image_name


def image_name_map(images_dir) -> dict[str, str]:
    """``{stem: real on-disk filename}`` from one directory listing (``list_logical_images``).

    A ``BandGroupRef``'s "name" is its own ``.bandgroup`` manifest's filename, the file that
    stands in for the grouped capture everywhere a name is matched against a store
    (``image_status.json``, a COCO ``file_name``), never one of its sibling band files.
    """
    return {stem: logical_image_name(src) for stem, src in list_logical_images(images_dir).items()}


def authored_frame(stem: str, labels_dir, fmt: str, coco=None,
                    file_name: str = "") -> tuple[int, int] | None:
    """``(width, height)`` the labels record, or ``None`` when they record none.

    The frame the boxes were drawn in, straight from the annotation a human produced, the only
    reference that can catch a reader disagreeing with the authoring tool. The json branch reads
    through :func:`~tcip_mcp.pipelines.data.splits.image_extent_from_labels`, the same function a
    split derives its own frame from, so the two agree by construction rather than each parsing
    the label file on its own; a present, unreadable label raises
    :class:`~tcip_annotation.json_io.UnreadableLabelDocument` rather than reading as no frame.
    """
    if coco is not None:
        images = coco.get("images", []) if isinstance(coco, dict) else []
        for entry in images:
            if entry.get("file_name") in (file_name, stem) or Path(
                    str(entry.get("file_name", ""))).stem == stem:
                w, h = int(entry.get("width", 0) or 0), int(entry.get("height", 0) or 0)
                return (w, h) if w > 0 and h > 0 else None
        return None
    if fmt not in ("", "json"):
        return None  # only the canonical per-image JSON carries its own frame
    from tcip_mcp.pipelines.data.splits import image_extent_from_labels

    return image_extent_from_labels(labels_dir, stem)


def targets_registry_derived(data_cfg: dict) -> bool:
    """Whether a run's targets are registry-derived: an image folder plus ``classes.json``, the
    one shape a run's own recorded ``id_map`` can be trusted to have come from this
    ``(labels_dir, subject, attribute)`` triple.

    ``False`` for a run trained from a bespoke ``dataset_source``, a pre-built ``coco_json``, or
    ``label_format="coco"``: none of those routes guarantee their targets came from this triple at
    all, a COCO file's own category ids can be authored in any order, and a bespoke builder owns
    its class space entirely. The one predicate ``_resolve_run_id_map``
    (``pipelines/training/subprocess_worker.py``) records no map by, extracted here so the
    inference-side door remedy (``unmapped_classified_run``) reads the identical rule rather than
    a second copy of it.
    """
    from tcip_mcp.pipelines.model_build import DATASET_SOURCE_KEY

    return not (
        data_cfg.get(DATASET_SOURCE_KEY) or data_cfg.get("coco_json")
        or (data_cfg.get("label_format") or "").lower() == "coco"
    )


def resolved_classes_path(dataset_dir) -> Path | None:
    """The real ``classes.json`` path for the dataset containing ``dataset_dir``, or ``None`` if it
    doesn't exist. The one fact ``resolve_registry_id_map``'s attribute-without-registry refusal
    and any caller wanting to precheck it (``inference_tools.run_inference`` precondition-checks
    this before attempting resolution, so a legitimately absent registry degrades to an honest
    ``id_map=None`` instead of a caught-and-swallowed exception) both need, computed once, never
    two independent implementations of the same "does a registry exist" fact.
    """
    from tcip_mcp.dataset_layout import classes_path, dataset_root_of

    root = dataset_root_of(dataset_dir)
    cp = classes_path(root) if root is not None else None
    return Path(cp) if cp is not None and Path(cp).is_file() else None


def resolve_registry_id_map(labels_dir, subject: str | None, attribute: str | None):
    """``(registry, id_map)`` for a training scope from the dataset's ``classes.json``.

    The single name→id derivation is :func:`class_registry.assign_class_ids`; the loader below,
    ``assemble_coco``, and the contract dims all read *this* map, never a second one. A plain
    single-class detector (``attribute`` is ``None``) needs no registry file, the subject *is* the
    class, so it is derived from a synthesized single-subject registry through the same
    ``assign_class_ids``, not a local ``{subject: 0}`` literal. Attribute classification needs the
    registry to order its values, and refuses when there is none.
    """
    from tcip_mcp import class_registry

    if not subject:
        raise ValueError(
            "a detection/instance_seg run needs an explicit subject to read name-based labels; "
            "none was threaded through build_dataset.")
    cp = resolved_classes_path(labels_dir)
    if cp is not None:
        registry = class_registry.read_registry(cp)
    elif attribute is not None:
        raise ValueError(
            f"attribute {attribute!r} classification needs a classes.json to order its values, "
            f"but none was found for {labels_dir}.")
    else:
        registry = class_registry.ClassRegistry(subjects=(class_registry.Subject(name=subject),))
    return registry, class_registry.assign_class_ids(registry, subject, attribute)


def coco_det_targets(coco, file_name):
    """Pixel-xyxy boxes + 1-indexed labels for one image from an assembled COCO.

    ``category_id`` is the run's 0-indexed id (from ``to_coco_dataset`` over the run's id_map); +1
    applies the detector's background offset. The loader owns the +1, nothing on disk.
    """
    from tcip_annotation import format_io
    anns, _, _ = format_io._coco_image_annotations(coco, file_name=file_name)
    boxes, labels = [], []
    for a in anns:
        bb = a.get("bbox")
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        x, y, bw, bh = (float(v) for v in bb)
        boxes.append([x, y, x + bw, y + bh])
        labels.append(int(a.get("category_id", 0)) + 1)
    return boxes, labels


def json_det_targets(path, subject, attribute, id_map):
    """``(boxes, labels, n_unlabeled)`` for one image from the name-based per-image JSON.

    Filters to ``subject`` + a box-derivable geometry, then maps each kept annotation to its
    0-indexed id via ``id_map`` (the single ``assign_class_ids`` map), +1 for background. An
    annotation the registry cannot decode raises, a real label read as nothing is a measurement bug.

    ``n_unlabeled`` counts instances of ``subject`` never assessed for ``attribute`` yet (a soft,
    expected gap, not a decode bug, excluded from ``boxes``/``labels`` rather than raising).
    Returning the count, not just silently dropping it, matters because an image with any
    unlabeled instance has incomplete ground truth for this scope, and a caller scoring/training
    only the labeled subset turns its real, unlabeled objects into silent false positives or
    background noise. Every caller therefore excludes the whole image on ``n_unlabeled > 0``, the
    same precedent a missing label file already gets, never partial trust in its labeled subset.
    A caller that builds per-image records fresh each call (delivery evaluation, operating-point
    calibration) drops the record as it builds it; a caller bound to a fixed per-image dataset
    length (a ``Dataset.__getitem__``, which cannot drop a sample per call) applies the same
    verdict once, up front, in ``trainable_stems``' ``skipped_incomplete_attribute`` partition.
    """
    from tcip_annotation import json_io
    from tcip_annotation.state import Point, bbox_of

    boxes, labels = [], []
    n_unlabeled = 0
    for a in json_io.read_annotations(path):
        # allow_unlabeled=True: an instance never assessed for `attribute` yet is a soft, expected
        # gap, not a decode bug, must not raise and abort the whole read.
        cid = json_io.target_class_id(a, subject, attribute, id_map, allow_unlabeled=True)
        if cid == json_io.UNLABELED:
            n_unlabeled += 1
            continue
        if cid is None or a.geometry is None or isinstance(a.geometry, Point):
            continue
        # target_class_id returns UNLABELED (a str) only for the case handled above; a real
        # target reaching here always carries its own int class id.
        assert isinstance(cid, int)
        box = bbox_of(a.geometry)
        boxes.append([box.x1, box.y1, box.x2, box.y2])
        labels.append(cid + 1)
    return boxes, labels, n_unlabeled


def first_labels_json(labels_dir) -> Path | None:
    """The first ``.json`` file in ``labels_dir``, sorted: the file :func:`dir_label_format`
    decides this directory's shape from, exposed separately so a refusal naming a dataset-level
    COCO export can point at the actual file, not just the directory.

    Raises :class:`~tcip_annotation.json_io.UnreadableLabelDocument` when that first file is
    present but will not read: trying the next file instead would read a directory as unlabeled
    that in fact holds a document nobody can make sense of, the opposite of what "the first" here
    is supposed to name. Reads through the unchecked ``load_json_document``, not
    ``load_label_document``: this directory's shape (json or a dataset-level COCO export) is not
    yet decided here, only ``dir_label_format``'s own ``detect_json_format`` call decides it, so
    this precondition read must not apply the per-image version ceiling to a file that may turn
    out COCO-shaped.
    """
    from tcip_annotation.json_io import load_json_document, prediction_documents

    candidates = prediction_documents(labels_dir)
    if not candidates:
        return None
    jp = candidates[0]
    load_json_document(jp)
    return jp


def dir_label_format(labels_dir) -> str | None:
    """``"json"``/``"coco"`` if this dir's first ``.json`` file declares that shape, else
    ``None``.

    Used to route a JSON label store onto the COCO training path. Decides the first file's shape
    through ``format_io``'s own per-file detection (COCO markers checked first, the same priority
    a dataset-level COCO reads with everywhere else), rather than a second, disagreeing
    ``ANNOTATIONS_KEY``-only test; the old ``objects`` schema, which that detection raises on, is
    treated the same as any other unrecognized shape here: ``None``, never a raise. A ``.json``
    that is not one of these shapes is not claimed, an unrecognized store must not be read as an
    all-empty one. A present, unreadable first file raises
    :class:`~tcip_annotation.json_io.UnreadableLabelDocument`, from :func:`first_labels_json` or
    from this function's own re-check of the same file, uncaught here: an unreadable document is
    not the same fact as an unrecognized one.
    """
    from tcip_annotation.format_io import detect_json_format

    jp = first_labels_json(labels_dir)
    if jp is None:
        return None
    try:
        return detect_json_format(jp)
    except ValueError:
        return None


def trainable_stems(
    labels_dir, images_dir, stems=None, *, subject: str | None = None, date,
    coco: dict | None = None,
    attribute: str | None = None, id_map: dict[str, int] | None = None,
    contradicted_out: set[str] | None = None,
) -> tuple[list[str], dict[str, int]]:
    """The stems that may train, plus the partition that produced them.

    A sample is admitted only when the label store actually accounts for it *for this subject*:

    - it has ≥1 annotation of ``subject``, or
    - it has none and a human marked that image negative for ``subject``
      (``confirmed_negative_names``, the Complete in ``.tcip/state/image_status.json``).

    An image with no label file, or an empty label file nobody confirmed, is unannotated, not a
    negative. Enumerating samples from ``images_dir`` instead served both as zero-box samples, so a
    project where the breeder labelled 30 of 400 images trained on 370 images asserted to be empty.

    Returns ``(stems, counts)`` where counts carries ``annotated`` / ``confirmed_negative`` /
    ``skipped_unannotated`` / ``skipped_unconfirmed_empty`` / ``skipped_incomplete_attribute`` /
    ``quarantined_stale_definition`` so a run can record what it dropped.
    ``quarantined_stale_definition`` is distinct from ``skipped_unconfirmed_empty``: it means a
    human did finish the image, complete or negative alike, but the subject's attribute schema has
    since changed and the confirmation can no longer be trusted as-is, a different situation from
    nobody ever having looked, and one a reproduce-a-number chain must be able to tell apart (see
    :func:`confirmed_negative_records`'s and :func:`stale_finished_names`'s shared quarantine
    logic, :func:`_stale_finished`). A recorded count from before this widening counted negatives
    alone, so counts across that boundary are not comparable.

    ``date`` states which capture date's confirmations this partition may admit, ``None`` for a
    tree that carries no date, and is passed through to ``confirmed_negative_names`` as the bucket
    key rather than recovered from ``labels_dir``.

    A confirmed negative whose label file now holds ``subject``, contradicting the store, is
    excluded from ``negatives`` the same way a quarantined one is; the stem still trains, admitted
    by its real content through the ``has_objects`` branch below rather than as a negative, even
    when that same negative is also stale-stamped (a stale-and-contradicted negative is not the
    stale complete the quarantine below exists to catch: real content contradicts a *negative*
    claim outright, and staleness never overrides that). This function keeps its own set of
    contradicted names regardless of what the caller passed, updating the caller's ``set`` (via
    ``contradicted_out``) when given rather than replacing it, so a caller with no set of its own
    still gets the reorder below right. Pass ``contradicted_out`` to learn which names those were,
    for a caller to surface as its own warning: the exclusion here is silent about admission only,
    never about the disagreement.

    A stale-stamped *complete* confirmation is quarantined ahead of the two annotated branches
    below (``coco_annotated`` and ``has_objects``), never admitted by its real content the way an
    unconfirmed image with the same content would be: a human's assertion under a since-changed
    schema is exactly what this count exists to hold back, whether or not the label file happens
    to carry boxes. Three asymmetries follow from this and stay unresolved by design: a stale
    finished status with no label file at all counts the quarantine first on the COCO path but
    ``skipped_unannotated`` first on the direct-JSON path (see the branches below); a materialized
    split tree carries a negative's stamp and no ``complete``, so it can still train a stale
    ``complete`` its source dataset would refuse (a residual of this family, not closed here); and
    a run bound to a split manifest refuses only through its ``train``/``val`` members
    (:func:`~tcip_mcp.pipelines.data.splits.bind_manifest_stems`), never through a ``calibration``
    member, which still calibrates on a stale finished image as it always has.

    ``skipped_incomplete_attribute`` is the whole-image attribute-completeness rail: with
    ``attribute`` set, an image carrying any instance never assessed for it has incomplete ground
    truth for this scope and is dropped entirely, never trained on its labelled subset (which
    would leave its real, unlabelled objects to train as background). It lives here, in the one
    partition that already decides admission, rather than as a second filter over this function's
    output: a filter downstream cannot record *why* a stem left, and applying the rail there
    instead corrupts these counts outright. Both label paths reach the same verdict from one
    implementation:
    the COCO path reads the ``excluded_incomplete_attribute`` names ``to_coco_dataset`` already
    computed during assembly (never re-deriving them, and never mistaking that absence for
    "empty label file nobody confirmed"), and the direct-JSON path applies the rail through
    ``json_det_targets``, the same reader the loader itself uses. ``attribute``/``id_map`` unset
    (every non-attribute run) applies no such rail.
    """
    names = image_name_map(images_dir)
    candidates = list(stems) if stems is not None else sorted(names)
    quarantined: set[str] = set()
    contradicted: set[str] = set()
    negatives = confirmed_negative_names(labels_dir, subject=subject, date=date,
                                         quarantined_out=quarantined,
                                         contradicted_out=contradicted)
    if contradicted_out is not None:
        contradicted_out.update(contradicted)
    counts = {"annotated": 0, "confirmed_negative": 0, "skipped_unannotated": 0,
              "skipped_unconfirmed_empty": 0, "skipped_incomplete_attribute": 0,
              "quarantined_stale_definition": 0}

    coco_names: set[str] | None = None
    coco_annotated: set[str] = set()
    incomplete_names: set[str] = set()
    if coco is not None:
        # assemble_coco already applied this rail to build ``images``; intersecting with it *is*
        # the rail, so the two can never disagree about which samples exist.
        by_id = {e.get("id"): str(e.get("file_name", "")) for e in coco.get("images", [])}
        coco_names = set(by_id.values())
        coco_annotated = {by_id.get(a.get("image_id"), "") for a in coco.get("annotations", [])}
        # Which images to_coco_dataset dropped for attribute-incompleteness, read from its own
        # record rather than re-derived; their absence from ``images`` is otherwise
        # indistinguishable from an unconfirmed-empty one, and reporting that reason would be a
        # lie.
        incomplete_names = {str(n) for n in coco.get("excluded_incomplete_attribute", [])}
    elif attribute is not None and id_map is not None:
        # Direct-JSON path: the same rail, through the same reader the loader uses.
        for stem in candidates:
            image_name = names.get(stem)
            if image_name is None:
                continue
            label_path = Path(labels_dir) / f"{stem}.json"
            if not label_path.is_file():
                continue
            _boxes, _labels, n_unlabeled = json_det_targets(
                str(label_path), subject, attribute, id_map)
            if n_unlabeled:
                incomplete_names.add(image_name)

    keep: list[str] = []
    for stem in candidates:
        image_name = names.get(stem)
        if image_name is None:
            counts["skipped_unannotated"] += 1
            continue
        if image_name in incomplete_names:
            # Checked before every other verdict: an image with incomplete attribute GT is dropped
            # for that reason, not for whichever downstream category its absence happens to resemble.
            counts["skipped_incomplete_attribute"] += 1
            continue
        if coco_names is not None:
            if image_name not in coco_names:
                # assemble_coco already dropped it, but not why: check quarantine here too, or a
                # stale confirmation reads as "nobody ever looked" instead of "looked, but stale".
                if image_name in quarantined:
                    counts["quarantined_stale_definition"] += 1
                    continue
                # "Annotate this" and "confirm this empty one" are different jobs.
                has_record, _ = _label_record_state(stem, labels_dir, subject)
                counts["skipped_unconfirmed_empty" if has_record
                       else "skipped_unannotated"] += 1
            elif image_name in quarantined and image_name not in contradicted:
                # A stale-stamped complete or negative is quarantined even with real content,
                # unless that content also contradicts a stored negative claim outright.
                counts["quarantined_stale_definition"] += 1
            elif image_name in coco_annotated:
                keep.append(stem)
                counts["annotated"] += 1
            elif image_name in negatives:
                # Zero annotations is a negative only with a human Complete. assemble_coco
                # already enforces that, but an externally supplied coco_json never went through
                # it, so the confirmation is re-checked here rather than inferred from the
                # file's shape.
                keep.append(stem)
                counts["confirmed_negative"] += 1
            elif image_name in quarantined:
                counts["quarantined_stale_definition"] += 1
            else:
                counts["skipped_unconfirmed_empty"] += 1
            continue
        has_record, has_objects = _label_record_state(stem, labels_dir, subject)
        if not has_record:
            counts["skipped_unannotated"] += 1
        elif image_name in quarantined and image_name not in contradicted:
            counts["quarantined_stale_definition"] += 1
        elif has_objects:
            keep.append(stem)
            counts["annotated"] += 1
        elif image_name in negatives:
            keep.append(stem)
            counts["confirmed_negative"] += 1
        elif image_name in quarantined:
            counts["quarantined_stale_definition"] += 1
        else:
            counts["skipped_unconfirmed_empty"] += 1
    return keep, counts


def require_samples(stems: list[str], counts: dict[str, int], labels_dir) -> None:
    """Refuse an empty sample set, naming why each image was dropped.

    Filtering to the label store can legitimately empty a dataset, an images_dir where nothing is
    annotated yet. Building it anyway would train on nothing and report success.
    """
    if stems:
        return
    quarantined = counts.get("quarantined_stale_definition", 0)
    quarantine_note = (
        f" {quarantined} more were confirmed complete or negative but quarantined because the "
        f"subject's attribute schema changed since, re-confirm them or revert the schema edit."
        if quarantined else ""
    )
    # Read defensively: this is the refusal path, and a counts dict missing a key here would
    # replace the explanation with a bare KeyError.
    incomplete = counts.get("skipped_incomplete_attribute", 0)
    incomplete_note = (
        f" {incomplete} more carry at least one instance never assessed for this run's attribute, "
        f"so their ground truth is incomplete for this scope and the whole image is held out "
        f"rather than trained on its labelled subset, finish attributing them, or run without "
        f"an attribute scope."
        if incomplete else ""
    )
    raise ValueError(
        f"no trainable samples in {labels_dir}: {counts.get('skipped_unannotated', 0)} image(s) "
        f"have no label record and {counts.get('skipped_unconfirmed_empty', 0)} have an empty one "
        f"nobody confirmed. An empty label file is a negative only once a human marks that image "
        f"Complete; until then it reads as unannotated. Annotate some images, or mark the "
        f"genuinely-empty ones Complete.{incomplete_note}{quarantine_note}"
    )


def _label_record_state(stem: str, labels_dir, subject: str | None) -> tuple[bool, bool]:
    """``(a record exists, it has ≥1 detection/seg target of ``subject``)`` for one stem.

    ``has_objects`` is subject-scoped *and box/polygon-bearing*: the unified file holds every subject,
    so "annotated" for a given subject's run means it carries an annotation of that subject whose
    geometry is a real detection/seg target, the same membership ``to_coco_dataset``/``target_class_id``
    apply (a box/polygon is a target; a geometry-less image-level label and a ``Point`` are not). Counting an
    image whose only annotations are non-targets as annotated would keep it on the direct-json path
    and train it as a zero-object negative, diverging from the COCO path and fabricating a negative no
    human confirmed.
    """
    from tcip_annotation import json_io
    from tcip_annotation.state import Point

    path = Path(labels_dir) / f"{stem}.json"
    if not path.is_file():
        return False, False
    anns = json_io.read_annotations(str(path))

    def _is_target(a) -> bool:
        return a.geometry is not None and not isinstance(a.geometry, Point)

    if subject is None:
        return True, any(_is_target(a) for a in anns)
    return True, any(a.subject == subject and _is_target(a) for a in anns)


def _raw_status_store(labels_dir) -> object:
    """The dataset's stored image statuses, for a caller that has a labels directory rather than
    a dataset root, or ``{}`` when the directory belongs to no dataset.

    The store read itself is :func:`~tcip_mcp.dataset_layout.read_image_status_store`, shared with
    every other confirmed-negative reader, so the enumeration of a subject's buckets and the records
    taken from them come from the same document.
    """
    from tcip_mcp.dataset_layout import dataset_root_of, read_image_status_store

    root = dataset_root_of(labels_dir)
    return {} if root is None else read_image_status_store(root)


def stale_stamped_names(
    stamped_by_image: Mapping[str, object], current_digest: str, names: Iterable[str],
) -> set[str]:
    """Names among ``names`` whose recorded digest stamp positively disagrees with
    ``current_digest``.

    The one comparison a stamped confirmation is measured against a schema for: shared by
    :func:`_stale_finished` (the quarantine both :func:`confirmed_negative_records` and
    :func:`stale_finished_names` read) and
    :func:`tcip_mcp.class_registry._sweep_schema_change`'s count of confirmations a vocabulary
    change left predating the new schema, so the definition of "stale" cannot drift between the
    two readers. Absence of a stamp is never stale (a rail admits valid work, not only rejects
    it): only an explicit stamp that disagrees is grounds to call a name out.
    """
    return {
        name for name in names
        if isinstance(stamped_by_image.get(name), str) and stamped_by_image[name] != current_digest
    }


def _stale_finished(
    root: Path, bucket_key: str, records: Mapping[str, dict[str, str]], subject: str,
) -> set[str]:
    """Names among ``records`` (one bucket's ``{image_name: record}``, already loaded by the
    caller) whose stored status is finished (:func:`~tcip_mcp.dataset_layout.is_finished_status`,
    ``complete`` or ``negative``) and whose stamped digest positively disagrees with ``subject``'s
    current attribute-schema digest, over one read each of the digest store and the registry.

    Shared by :func:`confirmed_negative_records`, which calls it inside its own single read of
    the status store, and :func:`stale_finished_names`, which calls it after its own: each caller
    reads the digest store and the registry once for the bucket it already holds, never twice for
    the same read. The admit rules answer the same question :func:`stale_stamped_names` states:
    no readable digest store, no stamp for an image, no readable or existing registry, or no
    digest for ``subject`` all admit rather than quarantine, since a rail must admit valid work,
    not only reject it.
    """
    import tcip_store

    from tcip_mcp.class_registry import attribute_schema_digest, read_registry
    from tcip_mcp.dataset_layout import (
        bucket_digest_stamps, classes_path, image_status_digest_key, is_finished_status,
        status_of,
    )

    finished = {name for name, record in records.items() if is_finished_status(status_of(record))}
    if not finished:
        return set()
    try:
        stamps = tcip_store.read(image_status_digest_key(root), default={})
    except (tcip_store.DecodeError, OSError):
        stamps = {}
    stamped_by_image = bucket_digest_stamps(stamps, bucket_key)
    if not stamped_by_image:
        return set()
    cp = classes_path(root)
    if not cp.is_file():
        return set()
    try:
        current_digest = attribute_schema_digest(read_registry(cp), subject)
    except (OSError, ValueError):
        return set()
    if current_digest is None:
        return set()
    return stale_stamped_names(stamped_by_image, current_digest, finished)


def stale_finished_names(
    dataset_root: str | Path | None, *, subject: str | None, date,
) -> set[str]:
    """Names in ``status_bucket(subject, date)`` whose stored status is finished and whose
    stamped digest positively disagrees with ``subject``'s current attribute-schema digest: the
    public D1 reader over a resolved dataset root, for a caller that has one directly rather than
    a labels directory to derive it from (the status route holds ``root``;
    :func:`confirmed_negative_records` already resolves its own and calls :func:`_stale_finished`
    without going through this wrapper, so the partition it feeds
    (:func:`trainable_stems`) still reads each store once, as today).

    Reads the status store, then the digest store and the registry through
    :func:`_stale_finished`, in a call of its own: never the same snapshot
    :func:`confirmed_negative_records` reads for the same bucket elsewhere, since the two are
    separate callers at separate times.

    Answers an empty set for ``dataset_root`` unset or ``subject`` unset: no root or no subject
    names no bucket to read, and for a bucket nothing was ever written under.
    """
    if not dataset_root or not subject:
        return set()
    from tcip_mcp.dataset_layout import read_image_status_store, status_bucket, status_confirmations

    root = Path(dataset_root)
    bucket_key = status_bucket(subject, date)
    bucket = status_confirmations(read_image_status_store(root)).get(bucket_key)
    if not bucket:
        return set()
    return _stale_finished(root, bucket_key, bucket, subject)


def confirmed_negative_names(
    labels_dir, *, subject: str | None, date, quarantined_out: set[str] | None = None,
    contradicted_out: set[str] | None = None,
) -> set[str]:
    """Image names a human marked negative (empty + Complete) for this subject.

    The name projection of :func:`confirmed_negative_records`, which it calls rather than reading
    the store a second time, for the admission decisions that need only the names. A name whose
    label file holds the subject, contradicting the stored negative, is excluded from the return
    value the same way a quarantined name is; pass ``contradicted_out`` to learn which.
    """
    return set(confirmed_negative_records(
        labels_dir, subject=subject, date=date, quarantined_out=quarantined_out,
        contradicted_out=contradicted_out))


def _exclude_contradicted(
    records: dict[str, dict[str, str]], subject: str, labels_dir,
    contradicted_out: set[str] | None,
) -> dict[str, dict[str, str]]:
    """Drops a name whose label file holds ``subject``, the disagreement ``scripts.doctor``'s
    ``check_negatives`` flags, through the same ``annotations_hold_subject`` predicate. Not
    lost: its label file carries real content, so a trainable-stems enumeration over the same
    directory admits it by that content instead."""
    from tcip_annotation import json_io
    from tcip_mcp.dataset_layout import annotations_hold_subject, label_filename

    def _label_holds_subject(name: str) -> bool:
        label = Path(labels_dir) / label_filename(Path(name).stem)
        return label.is_file() and annotations_hold_subject(
            json_io.read_annotations(str(label)), subject
        )

    contradicted = [name for name in records if _label_holds_subject(name)]
    if not contradicted:
        return records
    if contradicted_out is not None:
        contradicted_out.update(contradicted)
    return {name: r for name, r in records.items() if name not in contradicted}


def confirmed_negative_records(
    labels_dir, *, subject: str | None, date, quarantined_out: set[str] | None = None,
    contradicted_out: set[str] | None = None,
) -> dict[str, dict[str, str]]:
    """Image names a human marked negative for this subject, each with the record that says so.

    A record is the store's own ``{status, recorded_by, recorded_at}``, returned whole so a caller
    copying these confirmations into another dataset carries who confirmed them rather than
    re-attributing them to itself.

    Reads the dataset-native store ``image_status_key`` names, keyed by the dataset root the way
    ``classes.json`` is, so confirmations travel with the dataset rather than living in whichever
    project's private ``.tcip/`` happened to be an ancestor, and returns only the
    ``status_bucket(subject, date)`` bucket. A confirmation is
    a human's statement about one subject on one image; a store keyed by image name alone re-applies
    it to subjects they never looked at, so an image full of bushes trains as "contains no bushes".

    ``date`` is the capture date the confirmation was recorded under, and it is stated by the
    caller, ``None`` for a tree that genuinely carries no date (a materialized split's ``labels/``,
    a curated review dataset's flat ``annotations/``). It is never recovered from ``labels_dir``:
    the key a writer stated and the date a path happens to spell are two different facts, and
    substituting one for the other reads a bucket nobody wrote under, either dropping a human's
    confirmations or answering with another date's. A bucket nothing wrote to is empty here.

    A stale stamp is computed once per call over every finished status in the bucket, complete or
    negative alike, through :func:`_stale_finished`. ``quarantined_out`` (a set, mutated in place)
    therefore carries every stale finished name the bucket holds, a stale complete confirmation
    included even though a complete is never part of this function's own return value; see
    :func:`trainable_stems`'s ``quarantined_stale_definition`` count, which counts from this wider
    set. This function's own return value excludes only the stale *negatives* among them, and the
    contradicted ones: :func:`_exclude_contradicted` runs over every original negative, stale or
    not, so a negative both stale-stamped and contradicted lands in ``contradicted_out`` too,
    never swallowed by the stale exclusion first.

    A negative is quarantined, excluded from the return value, only when the dataset's
    ``image_status_digest.json`` sidecar carries an explicit stamp for that image (not merely its
    bucket, a bucket holds every image ever touched under the subject/date, so a bucket-wide stamp
    would be silently overwritten by the next unrelated write and un-quarantine a stale confirmation
    nobody re-reviewed) and it no longer matches the subject's current
    :func:`~tcip_mcp.class_registry.attribute_schema_digest`: positive, provable evidence the
    subject's classification schema changed since that confirmation was made. Absence of a stamp,
    no sidecar, no stamp for that image, or a dataset that predates this mechanism entirely, is
    not quarantined: a rail must admit valid work, not only reject it, and treating "nobody
    stamped this yet" as "unverifiable, therefore invalid" would silently empty
    every pre-existing project's confirmed negatives.

    ``subject`` must be threaded explicitly; there is no per-subject label directory to recover
    it from. When ``subject`` is unthreaded and the dataset holds confirmed negatives, this refuses
    loudly rather than returning nothing: a silent empty would drop every hard negative the review
    loop harvested. With no locatable dataset root, no store, or no confirmations for this subject,
    it returns nothing.
    """
    from tcip_mcp.dataset_layout import (
        dataset_root_of, is_confirmed_negative, status_confirmations, status_bucket, status_of,
    )

    root = dataset_root_of(labels_dir)
    if root is None:
        return {}
    statuses = status_confirmations(_raw_status_store(labels_dir))
    if not subject:
        # Refuse only when there is something to lose: a store with confirmed negatives this
        # run might be entitled to. Silently returning none would drop the human's work.
        has_negatives = any(
            is_confirmed_negative(status_of(r)) for b in statuses.values() for r in b.values()
        )
        if not has_negatives:
            return {}
        raise ValueError(
            f"confirmed_negative_names needs an explicit subject to read the negative bucket "
            f"for {labels_dir}, and this dataset has human-confirmed negatives that would be "
            f"silently dropped. Thread the run's subject through build_dataset / assemble_coco."
        )
    bucket_key = status_bucket(subject, date)
    bucket = statuses.get(bucket_key)
    if not bucket:
        return {}  # nothing was ever written under the key this caller stated
    negatives = {name: r for name, r in bucket.items() if is_confirmed_negative(status_of(r))}
    # Computed over the whole bucket, never only ``negatives``: a stale complete confirmation
    # has to reach quarantined_out even in a bucket with no negative in it at all.
    stale = _stale_finished(root, bucket_key, bucket, subject)
    if quarantined_out is not None:
        quarantined_out.update(stale)
    if not negatives:
        return negatives
    # Runs over the original negatives, not a stale-excluded remainder, so a negative both
    # stale-stamped and contradicted is named in contradicted_out too.
    without_contradicted = _exclude_contradicted(negatives, subject, labels_dir, contradicted_out)
    return {name: r for name, r in without_contradicted.items() if name not in stale}


def assemble_coco(
    labels_dir, images_dir, stems=None, *, subject: str, attribute: str | None = None,
    id_map: dict[str, int], date,
) -> dict:
    """Assemble a dataset-level COCO dict from the name-based per-image JSON, scoped to ``subject``.

    Pairs each stem's ``<labels_dir>/<stem>.json`` with its image's on-disk file name, the same
    name the dataset resolves at read time, so the COCO ``file_name`` keys line up. ``id_map`` is
    the run's ``assign_class_ids`` map; this is the single delegation to ``json_io.to_coco_dataset``,
    so the COCO categories, the loader targets, and the contract dims all rest on one name→id map.
    Stems whose image is missing are skipped. This is how per-image JSON reaches training: a COCO the
    ``label_format='coco'`` path consumes. ``date`` is the confirmation bucket's own date, stated by
    the caller and passed straight through, so the assembled COCO's negatives and the partition's
    come from one key.
    """
    from tcip_annotation import json_io

    labels_dir = Path(labels_dir)
    images_dir = Path(images_dir)
    if stems is None:
        stems = sorted(p.stem for p in json_io.prediction_documents(labels_dir))
    # Real on-disk names: to_coco_dataset matches these against the confirmed-negative store, and a
    # constructed name would silently match nothing for an uppercase extension.
    names = image_name_map(images_dir)
    entries: list[tuple[str, str]] = []
    for stem in stems:
        file_name = names.get(stem)
        if file_name is None:
            continue
        entries.append((str(labels_dir / f"{stem}.json"), file_name))
    return json_io.to_coco_dataset(
        entries, subject=subject, id_map=id_map, attribute=attribute,
        confirmed_negative_names=confirmed_negative_names(labels_dir, subject=subject, date=date),
    )
