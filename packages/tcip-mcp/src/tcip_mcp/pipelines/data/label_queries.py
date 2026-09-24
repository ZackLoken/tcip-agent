"""The producer: the one place a directory of ground truth or a ground-truth table becomes the
samples a run trains, evaluates or calibrates over.

It reads a dataset's per-image label documents, its ``<stem>.png`` masks or its table of rows,
its ``subjects.json`` registry and its confirmed-negative image-status store, admits by the shape
the ground truth itself carries (:func:`ground_truth_shape`), and answers an :class:`Admission`
whose samples the dataset classes in ``datasets.py`` and the outside-layer tools (calibration,
evaluation, inference, training) build from. Holds no ``Dataset`` subclass and no tensor
conversion; those stay in ``datasets.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from tcip_annotation.state import BBox, Polygon, box_derivable

from tcip_mcp.pipelines.image_utils import list_logical_images, logical_image_name

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.selection import ClassScope, Sample


def authored_frame(label_path) -> tuple[int, int] | None:
    """``(width, height)`` one sample's own label document records, or ``None`` when it records
    none.

    The frame the boxes were drawn in, straight from the annotation a human produced, the only
    reference that can catch a reader disagreeing with the authoring tool. ``label_path`` is the
    sample's own document, the path it recorded, so nothing reconstructs one from a directory and
    a stem. Reads through :func:`~tcip_mcp.pipelines.data.splits.label_document_extent`, the same
    function a split derives its own frame from, so the two agree by construction rather than each
    parsing the label file on its own; a present, unreadable label raises
    :class:`~tcip_annotation.json_io.UnreadableLabelDocument` rather than reading as no frame.
    """
    from tcip_mcp.pipelines.data.splits import label_document_extent

    return label_document_extent(label_path)


def resolved_subjects_path(dataset_dir) -> Path | None:
    """The real ``subjects.json`` path for the dataset containing ``dataset_dir``, or ``None`` if it
    doesn't exist. The one fact ``resolve_registry_id_map``'s attribute-without-registry refusal
    and any caller wanting to precheck it (``inference_tools.run_inference`` precondition-checks
    this before attempting resolution, so a legitimately absent registry degrades to an honest
    ``id_map=None`` instead of a caught-and-swallowed exception) both need, computed once, never
    two independent implementations of the same "does a registry exist" fact.
    """
    from tcip_mcp.dataset_layout import dataset_root_of, subjects_path

    root = dataset_root_of(dataset_dir)
    cp = subjects_path(root) if root is not None else None
    return Path(cp) if cp is not None and Path(cp).is_file() else None


def resolve_registry_id_map(labels_dir, subject: str | None, attribute: str | None):
    """``(registry, id_map)`` for a training scope from the dataset's ``subjects.json``.

    The single name→id derivation is :func:`subject_registry.assign_class_ids`: the admission
    below resolves it once and every reader downstream, the loaders and the contract dims among
    them, reads the map it recorded rather than a second resolution of its own. A plain
    single-class detector (``attribute`` is ``None``) needs no registry file, the subject *is* the
    class, so it is derived from a synthesized single-subject registry through the same
    ``assign_class_ids``, not a local ``{subject: 0}`` literal. Attribute classification needs the
    registry to order its values, and refuses when there is none. A scope naming no subject is
    refused through :meth:`~tcip_mcp.pipelines.data.selection.ClassScope.named_subject`.
    """
    from tcip_mcp import subject_registry
    from tcip_mcp.dataset_layout import dataset_root_of
    from tcip_mcp.pipelines.data.selection import ClassScope

    scope = ClassScope.recorded_in({"subject": subject, "attribute": attribute})
    subject, attribute = scope.named_subject(f"the run over {labels_dir}"), scope.attribute
    cp = resolved_subjects_path(labels_dir)
    if cp is not None:
        registry = subject_registry.read_registry(cp)
    elif attribute is not None:
        root = dataset_root_of(labels_dir)
        stale = subject_registry.retired_document(root) if root is not None else None
        if stale is not None:
            raise ValueError(
                f"attribute {attribute!r} classification needs a subjects.json to order its "
                f"values, but {labels_dir} resolves only the retired registry at {stale}; "
                "rename it to subjects.json by hand.")
        raise ValueError(
            f"attribute {attribute!r} classification needs a subjects.json to order its values, "
            f"but none was found for {labels_dir}.")
    else:
        registry = subject_registry.SubjectRegistry(
            subjects=(subject_registry.Subject(name=subject),))
    return registry, subject_registry.assign_class_ids(registry, subject, attribute)


def json_det_targets(path, subject, attribute, id_map,
                     reads: Callable[[Any], bool] = box_derivable):
    """``(target, n_unlabeled)`` for one image from the name-based per-image JSON: the one
    builder of a target from annotations, for the detection loader, the instance loader and every
    evaluation and calibration reader alike.

    ``target`` is the detection target shape, ``{"boxes", "labels", "iscrowd"}`` as parallel lists
    (pixel xyxy, 1-indexed label, crowd flag), every reader of a box's target reading its flag
    beside it, and ``"geometry"``, each row's own geometry, which a mask is rasterized from.
    Filters to ``subject`` and the geometry the caller reads (``reads``, a loader's own
    ``reads_geometry``; :func:`~tcip_annotation.state.box_derivable` unless stated, the one
    statement of which geometry a target is read from), then maps each
    kept annotation to its 0-indexed id via ``id_map`` (the single ``assign_class_ids`` map), +1
    for background, the one place that offset is added. An annotation the registry cannot decode
    raises, a real label read as nothing is a measurement bug.

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
    verdict once, up front, in ``admitted_documents``' ``skipped_incomplete_attribute`` partition.
    """
    from tcip_annotation import json_io
    from tcip_annotation.state import bbox_of

    target: dict[str, list] = {"boxes": [], "labels": [], "iscrowd": [], "geometry": []}
    n_unlabeled = 0
    for a in json_io.read_annotations(path):
        if not reads(a.geometry):
            continue
        # Every reads predicate admits a box or a region only (box_derivable or narrower).
        geometry = cast(BBox | Polygon, a.geometry)
        # allow_unlabeled=True: an instance never assessed for `attribute` yet is a soft, expected
        # gap, not a decode bug, must not raise and abort the whole read.
        cid = json_io.target_class_id(a, subject, attribute, id_map, allow_unlabeled=True)
        if cid == json_io.UNLABELED:
            n_unlabeled += 1
            continue
        if cid is None:
            continue
        box = bbox_of(geometry)
        target["boxes"].append([box.x1, box.y1, box.x2, box.y2])
        target["labels"].append(int(cid) + 1)
        target["iscrowd"].append(a.iscrowd)
        target["geometry"].append(geometry)
    return target, n_unlabeled


def ground_truth_shape(ground_truth) -> str:
    """Which of :data:`~tcip_mcp.pipelines.data.selection.GROUND_TRUTH_SHAPES` lives at
    ``ground_truth``: the one sniff the platform makes, over the place a config names.

    What each name it finds there means is
    :func:`~tcip_mcp.pipelines.data.selection.shape_of`, the one rule, asked of the place itself
    and of what the place holds; this function's own job is which names to ask about.

    A ``.csv`` is a table, one row per sample. A directory holding per-image label documents is
    the document shape and one holding ``<stem>.png`` rasters the mask shape. A directory holding
    neither documents nor masks reads as the document shape, whose own admission then names what
    is missing image by image. What each document holds is its reader's to decide: a
    dataset-level COCO sitting among them is refused by the per-image reader when admission reads
    it, naming the import door.
    """
    from tcip_annotation.json_io import prediction_documents
    from tcip_mcp.pipelines.data.selection import DOCUMENT, MASK, TABLE, shape_of

    path = Path(ground_truth)
    if shape_of(str(path), None) == TABLE:
        if not path.is_file():
            raise ValueError(
                f"{path} names a ground-truth table that does not exist; name where this run's "
                "ground truth actually lives.")
        return TABLE
    if not path.is_dir():
        raise ValueError(
            f"{ground_truth!r} is neither a .csv table nor a directory of ground truth: a run "
            "reads one label document per image, one <stem>.png mask per image, or one row of a "
            "table, and this names none of them.")
    if prediction_documents(path):
        return DOCUMENT
    held = {shape_of(str(p), None) for p in path.iterdir() if p.is_file()}
    return MASK if MASK in held else DOCUMENT


def admitted_records(
    records: Mapping[str, tuple[str | None, str | None]], *, labels_dir,
    subject: str | None = None, date, attribute: str | None = None,
    id_map: dict[str, int] | None = None, contradicted_out: set[str] | None = None,
) -> tuple[list[str], dict[str, int]]:
    """The keys of ``records`` the label store accounts for, plus the partition that produced them.

    ``records`` maps a caller's own key to ``(label document path, logical image name)``, the two
    facts every admission verdict below rests on; a ``None`` image name is a key with no image at
    all and a ``None`` document is a key the label store holds nothing for. The one admission, so
    a caller enumerating a directory (:func:`admitted_documents`) and a caller holding samples
    that each name their own two paths (:func:`refuse_inadmissible_samples`) reach the same
    verdict from one implementation rather than one of them rebuilding the other's paths from a
    stem.

    See :func:`admitted_documents` for what each count means and why the branches order the way
    they do; this is that function's body with the directory listing lifted out of it.

    The same pairs answer the confirmed-negative read's own contradiction check, so a positive
    labelled under a name of its own contradicts a stale negative recorded for its image rather
    than being looked for beside that image and quarantined for not being there.
    """
    quarantined: set[str] = set()
    contradicted: set[str] = set()
    negatives = confirmed_negative_names(labels_dir, subject=subject, date=date,
                                         quarantined_out=quarantined,
                                         contradicted_out=contradicted,
                                         label_paths={
                                             image_name: label_path
                                             for label_path, image_name in records.values()
                                             if image_name is not None and label_path is not None
                                         })
    if contradicted_out is not None:
        contradicted_out.update(contradicted)
    counts = {"annotated": 0, "confirmed_negative": 0, "skipped_unannotated": 0,
              "skipped_unconfirmed_empty": 0, "skipped_incomplete_attribute": 0,
              "quarantined_stale_definition": 0}

    incomplete_names: set[str] = set()
    if attribute is not None and id_map is not None:
        # The attribute-completeness rail, through the same reader the loader uses.
        for label_path, image_name in records.values():
            if image_name is None or label_path is None or not Path(label_path).is_file():
                continue
            _target, n_unlabeled = json_det_targets(label_path, subject, attribute, id_map)
            if n_unlabeled:
                incomplete_names.add(image_name)

    keep: list[str] = []
    for key, (label_path, image_name) in records.items():
        if image_name is None or label_path is None:
            counts["skipped_unannotated"] += 1
            continue
        if image_name in incomplete_names:
            # Checked first: an image with incomplete attribute GT is dropped for that reason,
            # never for whichever category its absence downstream happens to resemble.
            counts["skipped_incomplete_attribute"] += 1
            continue
        has_record, holds_subject = _label_record_state(label_path, subject)
        if not has_record:
            counts["skipped_unannotated"] += 1
        elif image_name in quarantined and image_name not in contradicted:
            counts["quarantined_stale_definition"] += 1
        elif holds_subject:
            keep.append(key)
            counts["annotated"] += 1
        elif image_name in negatives:
            keep.append(key)
            counts["confirmed_negative"] += 1
        elif image_name in quarantined:
            counts["quarantined_stale_definition"] += 1
        else:
            counts["skipped_unconfirmed_empty"] += 1
    return keep, counts


def admitted_documents(
    labels_dir, images_dir, members=None, *, subject: str | None = None, date,
    attribute: str | None = None, id_map: dict[str, int] | None = None,
    contradicted_out: set[str] | None = None,
) -> tuple[list[Admitted], dict[str, int]]:
    """The members a directory of per-image label documents admits, resolved, plus the partition
    that produced them.

    A sample is admitted only when the label store actually accounts for it *for this subject*:

    - its document carries ≥1 annotation of ``subject``, whatever geometry that annotation has,
      or
    - it has none and a human marked that image negative for ``subject``
      (``confirmed_negative_names``, the Complete in ``.tcip/state/image_status.json``).

    Which geometries a run can actually read is the selected loader's own question, asked there:
    a document carrying the subject only in a geometry that loader does not read is refused by it
    by name, never admitted here as though the image were empty.

    An image with no label file, or an empty label file nobody confirmed, is unannotated, not a
    negative.

    Returns ``(stems, counts)`` where counts carries ``annotated`` / ``confirmed_negative`` /
    ``skipped_unannotated`` / ``skipped_unconfirmed_empty`` / ``skipped_incomplete_attribute`` /
    ``quarantined_stale_definition`` so a run can record what it dropped.
    ``quarantined_stale_definition`` is distinct from ``skipped_unconfirmed_empty``: it means a
    human did finish the image, complete or negative alike, but the subject's attribute schema has
    since changed and the confirmation can no longer be trusted as-is, a different situation from
    nobody ever having looked, and one a reproduce-a-number chain must be able to tell apart (see
    :func:`confirmed_negative_records`'s and :func:`stale_finished_names`'s shared quarantine
    logic, :func:`_stale_finished`).

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

    A stale-stamped *complete* confirmation is quarantined ahead of the ``has_objects`` branch
    below, never admitted by its real content the way an
    unconfirmed image with the same content would be: a human's assertion under a since-changed
    schema is exactly what this count exists to hold back, whether or not the label file happens
    to carry boxes. One asymmetry follows and stays unresolved by design: a
    selection's ``calibration`` members are re-admitted by the door that builds a loader over them
    (``evaluate_model``, through :func:`refuse_inadmissible_samples`), while a door that only
    resolves an operating point over them measures a stale finished image as it always has.

    ``skipped_incomplete_attribute`` is the whole-image attribute-completeness rail: with
    ``attribute`` set, an image carrying any instance never assessed for it has incomplete ground
    truth for this scope and is dropped entirely, never trained on its labelled subset (which
    would leave its real, unlabelled objects to train as background). It lives here, in the one
    partition that already decides admission, rather than as a second filter over this function's
    output: a filter downstream cannot record *why* a stem left, and applying the rail there
    instead corrupts these counts outright. The rail is applied through
    ``json_det_targets``, the same reader the loader itself uses. ``attribute``/``id_map`` unset
    (every non-attribute run) applies no such rail.

    The verdicts themselves are :func:`admitted_records`; this is its directory-shaped caller,
    pairing each candidate stem with the document that directory actually holds for it and the
    real on-disk image name the listing holds for it. A stem the directory holds no document for
    is paired with ``None``: nothing constructs a candidate path beside an image, so a file whose
    name is reserved for a bucket's own provenance stamp is never read as that image's label.
    """
    from tcip_annotation.json_io import prediction_documents
    from tcip_mcp.pipelines.image_utils import refuse_incomplete_band_group, source_path_of

    sources = list_logical_images(images_dir)
    documents = {p.stem: p for p in prediction_documents(labels_dir)}
    candidates = list(members) if members is not None else sorted(sources)
    keep, counts = admitted_records(
        {stem: (str(documents[stem]) if stem in documents else None,
                logical_image_name(sources[stem]) if stem in sources else None)
         for stem in candidates},
        labels_dir=labels_dir, subject=subject, date=date,
        attribute=attribute, id_map=id_map, contradicted_out=contradicted_out,
    )
    return [
        Admitted(member=stem,
                 source=source_path_of(refuse_incomplete_band_group(sources[stem])),
                 ground_truth=str(documents[stem]))
        for stem in keep
    ], counts


def admission_date(labels_dir) -> str | None:
    """The capture date one labeled directory's admission reads confirmations under, or ``None``
    for a tree that carries no date.

    :func:`~tcip_mcp.dataset_layout.annotation_date`, the declared inverse of the
    ``annotations/<date>/`` layout: pointing a producer at one of those directories is how a
    caller states which date it means, and a tree that is not one of them answers ``None`` rather
    than a segment its path happens to spell. The one resolution, so the directory an admission
    reads and the sample it records name one bucket rather than two keys that can disagree.
    """
    from tcip_mcp.dataset_layout import annotation_date

    return annotation_date(labels_dir)


@dataclass(frozen=True)
class Admitted:
    """One admitted member, as the producer resolved it.

    ``member`` is the name a membership record names it by, the image stem for ground truth that
    is one file per sample and the row key for a table. ``source`` is the image source the
    admission resolved for it, already a path (a ``.bandgroup`` manifest for a grouped capture),
    and ``ground_truth`` the file that answers for it, with ``row_key`` naming its row when one
    file answers for many. The producer resolves both once, here, and hands the record on: nothing
    downstream reconstructs a path from a name or enumerates the image directory again.
    """

    member: str
    source: str
    ground_truth: str
    row_key: str | None = None


def samples_over(
    records: "Sequence[Admitted]", assignment: dict[str, str],
    group_of: "Callable[[str], str]", *, confirmation_bucket: str | None = None,
    digests: Mapping[str, str] | None = None,
) -> list["Sample"]:
    """Admitted records as explicit samples, one per record: the one place anything becomes a
    sample.

    ``assignment`` maps each member to the side it landed on, ``group_of`` gives its group key,
    and ``digests`` its ground-truth digest when the caller computed one. Nothing is resolved or
    reconstructed here: each sample carries the source and the ground truth the admission already
    resolved for that member.

    ``confirmation_bucket`` is the ``image_status.json`` key a document admission read
    (:func:`~tcip_mcp.dataset_layout.status_bucket`), so a selection spanning three capture dates
    says per sample which human confirmations answered for it; it rides only on a sample whose
    ground truth is its own document, since a mask and a row are admitted by existing beside their
    image with no human confirmation store to name.

    The drawn split, the bound run's own record and ``freeze_selection`` all produce their samples
    here, so every route holds one membership shape and reads through the same recorded paths.
    """
    from tcip_mcp.pipelines.data.selection import DOCUMENT, Sample, shape_of

    digests = digests or {}
    return [
        Sample(
            source=record.source, ground_truth=record.ground_truth, row_key=record.row_key,
            group=group_of(record.member), side=assignment[record.member],
            confirmation_bucket=(
                confirmation_bucket
                if shape_of(record.ground_truth, record.row_key) == DOCUMENT else None),
            ground_truth_digest=digests.get(record.member),
        )
        for record in sorted(records, key=lambda r: r.member)
        if record.member in assignment
    ]


def foreground_counts(
    members: "Mapping[str, Sample | Admitted]", scope: "ClassScope",
) -> dict[str, int]:
    """Each admitted member's own foreground count, under the key its caller already holds it by,
    for a draw that balances sides by how much ground truth each member carries.

    A member is an :class:`Admitted` record or the :class:`~tcip_mcp.pipelines.data.selection.Sample`
    it became: each names its own ground truth and, where one ground truth answers for many, the
    row it reads, which is all a count needs, so a caller that has not assigned sides yet counts
    over its records rather than building a sample list it will build again once it has.

    A label document carries a count of its own annotations of ``scope``'s subject (scoped to its
    attribute when one is named), read from the path the sample records rather than from a name
    recomposed under a directory. A mask raster and a table row are admitted by existing beside
    their image, so every admitted one carries ground truth and counts as one: there is no denser
    or sparser member to balance, and counting zero for them would read every member as
    background.

    The caller's own index is the result's index: a lock keyed by member name and a draw keyed by
    source identity each read their counts back under the key they handed in, so no caller
    re-keys this map.
    """
    from tcip_mcp.pipelines.data.selection import DOCUMENT, shape_of
    from tcip_mcp.pipelines.data.splits import count_label_lines

    return {
        key: (count_label_lines(member.ground_truth, subject=scope.subject,
                                attribute=scope.attribute)
              if shape_of(member.ground_truth, member.row_key) == DOCUMENT else 1)
        for key, member in members.items()
    }


def is_mask(path: Path) -> bool:
    """Whether one entry is a mask: a file named ``<stem>.png``, and nothing else.

    The one mask predicate. The admission that enumerates a directory and the re-admission that
    checks a recorded sample's own path both ask it, so a directory named ``a.1.png`` cannot be
    admitted by one and refused by the other.
    """
    return path.is_file() and path.suffix.lower() == ".png"


def admitted_masks(labels_dir, images_dir, members=None) -> tuple[list[Admitted], dict[str, int]]:
    """The members a mask directory admits, resolved, plus the partition that produced them.

    A sample needs a mask, and a mask is exactly ``<stem>.png`` under ``labels_dir``
    (:func:`is_mask`): an all-background mask is an explicit annotation, so the file's existence
    is the rail, and an image with no mask is unannotated rather than a negative. A same-stem
    file in any other format refuses by name rather than leaving its image to train as entirely
    background.
    """
    from tcip_mcp.pipelines.image_utils import refuse_incomplete_band_group, source_path_of

    labels_p = Path(labels_dir)
    entries = list(labels_p.iterdir()) if labels_p.is_dir() else []
    masks = {p.stem: p for p in entries if is_mask(p)}
    sources = list_logical_images(images_dir)
    candidates = list(members) if members is not None else sorted(sources)
    unreadable = sorted(
        p.name for p in entries
        if p.is_file() and not is_mask(p) and p.stem in candidates and p.stem not in masks
    )
    if unreadable:
        raise ValueError(
            f"{labels_p} holds {unreadable} beside an image of the same stem, and a "
            f"mask is read only as <stem>.png; nothing here reads another format, and "
            f"training the image without its mask would train it as entirely background."
        )
    admitted = [
        Admitted(member=stem,
                 source=source_path_of(refuse_incomplete_band_group(sources[stem])),
                 ground_truth=str(masks[stem]))
        for stem in candidates if stem in masks and stem in sources
    ]
    return admitted, {"annotated": len(admitted),
                      "skipped_unannotated": len(candidates) - len(admitted)}


def ground_truth_table(csv_path) -> dict[str, str]:
    """One ground-truth table as ``{row key: value}``, the row key being the image stem its first
    column names and the value its second, both as written.

    The one reader of a ground-truth table: the admission, the loaders and the producer all read
    the rows through it rather than each parsing the file. A key naming more than one row refuses
    by name, since a sample recorded by row key would otherwise read whichever row came last.
    """
    import csv as _csv

    rows: dict[str, str] = {}
    repeated: list[str] = []
    with open(csv_path, newline="") as handle:
        reader = _csv.reader(handle)
        next(reader, None)  # the header row
        for row in reader:
            if len(row) < 2:
                continue
            key = row[0].strip()
            if key in rows:
                repeated.append(key)
            rows[key] = row[1].strip()
    if repeated:
        raise ValueError(
            f"{csv_path} names {sorted(set(repeated))[:5]} on more than one row: a sample "
            "recorded by row key would read whichever row came last. Give each image one row."
        )
    return rows


def holds_row(table: Mapping[str, str], row_key: str | None) -> bool:
    """Whether a ground-truth table holds the row one member names.

    The one row predicate. The admission that reads a whole table, the re-admission that checks a
    recorded sample's own row and the loader that reads its value all ask it, so no two of them
    can spell a key differently and disagree about whether the row is there.
    """
    return row_key is not None and row_key in table


def admitted_rows(csv_path, images_dir, members=None) -> tuple[list[Admitted], dict[str, int]]:
    """The rows a ground-truth table admits, resolved, plus the partition that produced them.

    A row is admitted when the table holds it (:func:`holds_row`) and the image it names exists
    under ``images_dir`` (:func:`~tcip_mcp.pipelines.image_utils.list_logical_images`, so a
    grouped capture is admitted by its ``.bandgroup`` manifest): a row naming no image has nothing
    to train, and an image no row names has no ground truth. The one table admission; the
    re-admission over a recorded sample asks the same table the same question.
    """
    from tcip_mcp.pipelines.image_utils import refuse_incomplete_band_group, source_path_of

    table = ground_truth_table(csv_path)
    sources = list_logical_images(images_dir)
    candidates = list(members) if members is not None else sorted(table)
    admitted = [
        Admitted(member=key,
                 source=source_path_of(refuse_incomplete_band_group(sources[key])),
                 ground_truth=str(csv_path), row_key=key)
        for key in candidates if holds_row(table, key) and key in sources
    ]
    return admitted, {
        "annotated": len(admitted),
        "skipped_no_image": sum(1 for k in candidates if holds_row(table, k) and k not in sources),
        "skipped_no_row": sum(1 for k in candidates if not holds_row(table, k)),
    }


def refuse_inadmissible_samples(
    samples: "Sequence[Sample]", scope: "ClassScope | None" = None,
) -> None:
    """Refuse a recorded sample the platform's own admission would no longer admit, naming which.

    ``scope`` is the class space those samples were admitted under, whole
    (:class:`~tcip_mcp.pipelines.data.selection.ClassScope`, reachable as ``Selection.scope`` and
    ``Admission.scope``): a caller holding one hands it over rather than splitting it into the
    fields this re-admission happens to read, and the normalizing of an empty attribute to "no
    attribute" is the scope's, not each caller's.

    One re-admission over recorded samples, dispatching once on each sample's own shape: the label
    store and its confirmations for a document, the mask's own existence for a mask raster, the
    row's own presence in its table for a table row. Each is the admission that named the sample
    in the first place, re-run over the paths the sample recorded rather than over a directory
    listing, so a sample whose ground truth is named unlike its image is admitted on what it
    carries rather than refused for a file nobody said had to exist.

    A selection records which samples train; it does not freeze what their ground truth says. A
    label that held this subject at draw time can be emptied afterwards, and an empty label nobody
    confirmed is unannotated, never a negative, so training it would put a real object's pixels in
    the background class with no human having said the image is empty. A mask deleted since, or a
    row dropped from its table, reaches the same place. The subject and capture date a document
    sample is checked under come from the confirmation bucket the producer stamped on it
    (:func:`~tcip_mcp.dataset_layout.bucket_subject_date`, the declared inverse), so a selection
    spanning three dates is checked against three buckets rather than one date substituted for all
    of them. Every sample's recorded source is resolved once here
    (:func:`~tcip_mcp.pipelines.image_utils.resolve_source_path`, which also catches a band group
    missing a sibling).

    Membership is never changed here: a sample the admission no longer holds refuses the run by
    name rather than being dropped from it, since a run that quietly trains on fewer samples than
    the record says is a different run than the record describes.
    """
    from tcip_mcp.dataset_layout import bucket_subject_date
    from tcip_mcp.pipelines.data.selection import DOCUMENT, MASK, ClassScope
    from tcip_mcp.pipelines.image_utils import BandGroupIncomplete, resolve_source_path

    scope = scope or ClassScope()
    refused: list[str] = []
    unresolved: list[str] = []
    reasons: dict[str, int] = {}
    tables: dict[str, dict[str, str]] = {}
    documents: dict[tuple[str, str], dict[str, tuple[str, str | None]]] = {}
    for sample in samples:
        try:
            image_name: str | None = logical_image_name(resolve_source_path(sample.source))
        except (FileNotFoundError, BandGroupIncomplete):
            unresolved.append(sample.source)
            continue
        if sample.shape == DOCUMENT:
            bucket = sample.confirmation_bucket
            assert bucket is not None, "a label-document sample always states the bucket it read"
            documents.setdefault(
                (sample.ground_truth_scope, bucket), {}
            )[sample.identity] = (sample.ground_truth, image_name)
        elif sample.shape == MASK:
            if not is_mask(Path(sample.ground_truth)):
                refused.append(sample.identity)
                reasons["ground_truth_gone"] = reasons.get("ground_truth_gone", 0) + 1
        else:
            table = sample.ground_truth
            if table not in tables:
                tables[table] = (ground_truth_table(table)
                                 if Path(table).is_file() else {})
            if not holds_row(tables[table], sample.row_key):
                refused.append(sample.identity)
                reasons["row_gone"] = reasons.get("row_gone", 0) + 1
    for (labels_dir, bucket), records in sorted(documents.items()):
        subject, date = bucket_subject_date(bucket)
        admitted, counts = admitted_records(
            records, labels_dir=labels_dir, subject=subject, date=date,
            attribute=scope.attribute, id_map=scope.id_map,
        )
        for name, value in counts.items():
            if name.startswith(("skipped_", "quarantined_")):
                reasons[name] = reasons.get(name, 0) + value
        refused.extend(sorted(set(records) - set(admitted)))
    if not refused and not unresolved:
        return
    named = ", ".join(f"{name}={value}" for name, value in sorted(reasons.items()) if value)
    sources = (f" {len(unresolved)} name a source that no longer resolves ({unresolved[:5]})."
               if unresolved else "")
    raise ValueError(
        f"{len(refused) + len(unresolved)} of this selection's samples are no longer admissible "
        f"({sorted(refused)[:5] + unresolved[:5]}): {named}.{sources} The data moved under the "
        "selection since it was drawn: a label emptied with nobody confirming that image "
        "negative, a mask or a label file deleted, a row dropped from its table, an image moved, "
        "a confirmation invalidated by a schema edit, or an instance left unassessed for this "
        "run's attribute. Restore what those name, finish the annotation or confirmation, or "
        "draw the selection again over the current data."
    )


def require_admitted(admitted: "Admission") -> None:
    """Refuse an empty admission, naming why nothing was admitted and what would fix it.

    The one refusal for every ground-truth shape, called by whoever needs a non-empty membership:
    filtering to what actually answers for an image can legitimately empty a dataset, and building
    it anyway would train on nothing and report success. A draw spanning several places calls it
    once over what they admitted between them, since one empty date directory beside a full one is
    not an empty draw. The counts are the admission's own partition, read defensively because this
    is the refusal path and a missing key here would replace the explanation with a bare KeyError.
    """
    if admitted.records:
        return
    from tcip_mcp.pipelines.data.selection import DOCUMENT, MASK

    shape, counts = admitted.shape, admitted.counts
    ground_truth, images_dir = admitted.ground_truth, admitted.images_dir
    if shape == MASK:
        raise ValueError(
            f"no trainable samples: none of the {counts.get('skipped_unannotated', 0)} image(s) "
            f"in {images_dir} have a <stem>.png mask in {ground_truth}. An image with no mask "
            f"would train as entirely background, so nothing here admits one. Write the masks, or "
            f"point data.labels_dir at the directory holding them."
        )
    if shape != DOCUMENT:
        raise ValueError(
            f"no trainable samples in {ground_truth}: {counts.get('skipped_no_image', 0)} row(s) "
            f"name an image that is not under {images_dir}. A row naming no image has nothing to "
            f"train. Fix the row keys, or point data.images_dir at the directory holding those "
            f"images."
        )
    quarantined = counts.get("quarantined_stale_definition", 0)
    quarantine_note = (
        f" {quarantined} more were confirmed complete or negative but quarantined because the "
        f"subject's attribute schema changed since, re-confirm them or revert the schema edit."
        if quarantined else ""
    )
    incomplete = counts.get("skipped_incomplete_attribute", 0)
    incomplete_note = (
        f" {incomplete} more carry at least one instance never assessed for this run's attribute, "
        f"so their ground truth is incomplete for this scope and the whole image is held out "
        f"rather than trained on its labelled subset, finish attributing them, or run without "
        f"an attribute scope."
        if incomplete else ""
    )
    raise ValueError(
        f"no trainable samples in {ground_truth}: "
        f"{counts.get('skipped_unannotated', 0)} image(s) "
        f"have no label record and {counts.get('skipped_unconfirmed_empty', 0)} have an empty one "
        f"nobody confirmed. An empty label file is a negative only once a human marks that image "
        f"Complete; until then it reads as unannotated. Annotate some images, or mark the "
        f"genuinely-empty ones Complete.{incomplete_note}{quarantine_note}"
    )


def _label_record_state(label_path: str | Path, subject: str | None) -> tuple[bool, bool]:
    """``(a record exists, it carries ``subject``)`` for one label document.

    The unified file holds every subject, so "annotated" for a given subject's run means it
    carries an annotation of that subject, whatever geometry that annotation has: a box, a
    polygon, a point or no geometry at all. Which of those a run can actually read is the selected
    loader's own question, asked there by name, so admission never reads a document that carries
    the subject as though the image were empty. The predicate is
    :func:`~tcip_mcp.dataset_layout.annotations_hold_subject`, the same one the contradicted-
    negative read asks, so the two cannot disagree about what "carries the subject" means.
    """
    from tcip_annotation import json_io
    from tcip_mcp.dataset_layout import annotations_hold_subject

    path = Path(label_path)
    if not path.is_file():
        return False, False
    anns = json_io.read_annotations(str(path))
    if subject is None:
        return True, bool(anns)
    return True, annotations_hold_subject(anns, subject)


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
    :func:`tcip_mcp.subject_registry._sweep_schema_change`'s count of confirmations a vocabulary
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
    no digest store at all, one whose bytes cannot be decoded, no stamp for an image, no readable
    or existing registry, or no digest for ``subject`` all admit rather than quarantine, since a
    rail must admit valid work, not only reject it; a digest store present but unreadable is
    reported by raising, never admitted.
    """
    import tcip_store

    from tcip_mcp.subject_registry import attribute_schema_digest, read_registry
    from tcip_mcp.dataset_layout import (
        bucket_digest_stamps, image_status_digest_key, is_finished_status,
        status_of, subjects_path,
    )

    finished = {name for name, record in records.items() if is_finished_status(status_of(record))}
    if not finished:
        return set()
    try:
        stamps = tcip_store.read(image_status_digest_key(root), default={})
    except tcip_store.DecodeError:
        stamps = {}
    stamped_by_image = bucket_digest_stamps(stamps, bucket_key)
    if not stamped_by_image:
        return set()
    cp = subjects_path(root)
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
    public reader over a resolved dataset root, for a caller that has one directly rather than
    a labels directory to derive it from (the status route holds ``root``;
    :func:`confirmed_negative_records` already resolves its own and calls :func:`_stale_finished`
    without going through this wrapper, so the partition it feeds
    (:func:`admitted_documents`) still reads each store once, as today).

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
    label_paths: Mapping[str, str] | None = None,
) -> set[str]:
    """Image names a human marked negative (empty + Complete) for this subject.

    The name projection of :func:`confirmed_negative_records`, which it calls rather than reading
    the store a second time, for the admission decisions that need only the names. A name whose
    label file holds the subject, contradicting the stored negative, is excluded from the return
    value the same way a quarantined name is; pass ``contradicted_out`` to learn which.
    ``label_paths`` is forwarded verbatim.
    """
    return set(confirmed_negative_records(
        labels_dir, subject=subject, date=date, quarantined_out=quarantined_out,
        contradicted_out=contradicted_out, label_paths=label_paths))


def _exclude_contradicted(
    records: dict[str, dict[str, str]], subject: str,
    contradicted_out: set[str] | None,
    label_paths: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    """Drops a name whose label document holds ``subject``, the disagreement ``scripts.doctor``'s
    ``check_negatives`` flags, through the same ``annotations_hold_subject`` predicate. Not
    lost: its label document carries real content, so a trainable-stems enumeration over the same
    directory admits it by that content instead.

    ``label_paths`` maps an image name to the label document that answers for it, as the record
    that named both holds them: a positive labelled under a name of its own is read where it
    actually is rather than recomposed beside the image and missed, which would leave a stale
    negative quarantining a sample its own label contradicts. A name the caller named no document
    for is not contradicted by anything, since nothing here goes looking for one.
    """
    from tcip_annotation import json_io
    from tcip_mcp.dataset_layout import annotations_hold_subject

    def _label_holds_subject(name: str) -> bool:
        recorded = label_paths.get(name)
        if recorded is None:
            return False
        label = Path(recorded)
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
    label_paths: Mapping[str, str] | None = None,
) -> dict[str, dict[str, str]]:
    """Image names a human marked negative for this subject, each with the record that says so.

    A record is the store's own ``{status, recorded_by, recorded_at}``, returned whole so a caller
    copying these confirmations into another dataset carries who confirmed them rather than
    re-attributing them to itself.

    Reads the dataset-native store ``image_status_key`` names, keyed by the dataset root the way
    ``subjects.json`` is, so confirmations travel with the dataset rather than living in whichever
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
    :func:`admitted_documents`'s ``quarantined_stale_definition`` count, which counts from this wider
    set. This function's own return value excludes only the stale *negatives* among them, and the
    contradicted ones: :func:`_exclude_contradicted` runs over every original negative, stale or
    not, so a negative both stale-stamped and contradicted lands in ``contradicted_out`` too,
    never swallowed by the stale exclusion first.

    A negative is quarantined, excluded from the return value, only when the dataset's
    ``image_status_digest.json`` sidecar carries an explicit stamp for that image (not merely its
    bucket, a bucket holds every image ever touched under the subject/date, so a bucket-wide stamp
    would be silently overwritten by the next unrelated write and un-quarantine a stale confirmation
    nobody re-reviewed) and it no longer matches the subject's current
    :func:`~tcip_mcp.subject_registry.attribute_schema_digest`: positive, provable evidence the
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

    ``label_paths`` maps an image name to the document that answers for it, as the records that
    named both hold them; see :func:`_exclude_contradicted`, the one reader of it. A caller that
    names none states that no document is known for these names, and no name is then contradicted:
    nothing here composes a path beside an image to go looking for one.
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
            f"silently dropped. Thread the run's subject through the admission that named them."
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
    without_contradicted = _exclude_contradicted(
        negatives, subject, contradicted_out, label_paths or {})
    return {name: r for name, r in without_contradicted.items() if name not in stale}


@dataclass(frozen=True)
class Admission:
    """What one place holding ground truth admits, and the facts every loader over it is built
    from.

    ``shape`` is what that ground truth is, read off the place itself
    (:func:`ground_truth_shape`), never off the task a run states. ``records`` is the admitted
    set as the producer resolved it, each :class:`Admitted` carrying its member name, its image
    source and its own ground truth. :meth:`samples` turns a side assignment over those records
    into explicit samples, the one shape every loader here is built from, so a directory or a
    table a caller names is the producer's interface and never a second membership
    representation, and nothing downstream resolves a source or composes a path again.

    ``subject``/``date``/``id_map`` are the scope a document admission read confirmations and
    class ids under, and are ``None`` for a shape no confirmation store and no registry answers
    for: a mask raster and a table row carry their own classes.
    """

    shape: str
    images_dir: str
    ground_truth: str
    records: list[Admitted]
    counts: dict[str, int]
    subject: str | None = None
    attribute: str | None = None
    date: str | None = None
    id_map: dict[str, int] | None = None

    @property
    def scope(self) -> "ClassScope":
        """The class space these records were admitted under, empty for ground truth no registry
        scopes."""
        from tcip_mcp.pipelines.data.selection import ClassScope

        return ClassScope(subject=self.subject, attribute=self.attribute, id_map=self.id_map)

    def samples(
        self, assignment: dict[str, str], group_of: "Callable[[str], str]",
        digests: Mapping[str, str] | None = None,
    ) -> list["Sample"]:
        """The admitted records this assignment names, as samples on the sides it gives them."""
        from tcip_mcp.dataset_layout import status_bucket

        bucket = status_bucket(self.subject, self.date) if self.subject else None
        return samples_over(self.records, assignment, group_of,
                            confirmation_bucket=bucket, digests=digests)

    def every_sample(self) -> list["Sample"]:
        """Every admitted record as a sample, all on the training side, each its own group.

        What a door measuring or counting over a whole place is built from: it holds out nothing,
        so there is one side, and it groups nothing, so each member is its own group. Stated
        here so an evaluation, a calibration and a preflight cannot spell that assignment three
        ways; a route that does hold a side out states its own assignment through :meth:`samples`.
        """
        return self.samples({record.member: "train" for record in self.records},
                            lambda member: member)

    def one_sample(self) -> "Sample":
        """The single admitted sample, for the within-image route that splits one source's own
        tile lattice. Its group is its member identity: that route draws over tiles, so nothing
        here groups one source with another."""
        from tcip_mcp.pipelines.data.splits import member_identity

        member = self.records[0].member
        return self.samples({member: "train"},
                            lambda stem: member_identity(self.date, stem))[0]


def admit(
    images_dir, ground_truth, *, subject: str | None = None, attribute: str | None = None,
    members: list[str] | None = None, contradicted_out: set[str] | None = None,
) -> Admission:
    """The membership one place holding ground truth admits, through the admission its own shape
    reads.

    The one enumeration of a directory or a table the platform makes: every loader, every
    evaluation door and every draw is built from the samples this returns, so nothing downstream
    enumerates a place a second time. It dispatches once on :func:`ground_truth_shape` and no
    further: the label store and its human confirmations for per-image documents, the mask's own
    existence beside the image for mask rasters, the row's own presence beside a resolvable image
    for a table.

    Raises rather than degrading on anything it cannot read: a place that names no ground truth
    this platform reads, or a directory holding a dataset-level COCO, refuses here by name. An
    empty admission is not one of those, since a draw spanning several places may legitimately
    find one of them empty; whoever needs a non-empty membership says so
    (:func:`require_admitted`).
    """
    from tcip_mcp.pipelines.data.selection import DOCUMENT, MASK

    # An empty subject or attribute is "no subject" and "no attribute", the same fact as unset.
    subject, attribute = subject or None, attribute or None
    shape = ground_truth_shape(ground_truth)
    if shape == MASK:
        records, counts = admitted_masks(ground_truth, images_dir, members)
    elif shape != DOCUMENT:
        records, counts = admitted_rows(ground_truth, images_dir, members)
    else:
        date = admission_date(ground_truth)
        # The single name->id map this run admits under, and the one its loaders read.
        _registry, id_map = resolve_registry_id_map(ground_truth, subject, attribute)
        records, counts = admitted_documents(
            ground_truth, images_dir, members, subject=subject, date=date,
            attribute=attribute, id_map=id_map, contradicted_out=contradicted_out,
        )
        return Admission(
            shape=shape, images_dir=str(images_dir), ground_truth=str(ground_truth),
            records=records, counts=counts, subject=subject, attribute=attribute,
            date=date, id_map=id_map,
        )
    return Admission(shape=shape, images_dir=str(images_dir), ground_truth=str(ground_truth),
                     records=records, counts=counts)
