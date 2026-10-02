"""The producer: where a directory of ground truth or a ground-truth table becomes the samples a
run trains, evaluates or calibrates over.

It reads a dataset's per-image label documents with their completion marks, its ``<stem>.png``
masks or its table of rows, and its ``subjects.json`` registry, admits by the shape the ground
truth itself carries (:func:`ground_truth_shape`), and answers an :class:`Admission`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from tcip_annotation.state import BBox, Polygon, box_derivable

from tcip_mcp.pipelines.image_utils import list_logical_images, logical_image_name

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.selection import ClassScope, Sample


def registered_dataset_root(dataset_dir) -> Path | None:
    """The root of the dataset containing ``dataset_dir`` when it holds a subject registry, else
    ``None``."""
    from tcip_mcp.dataset_layout import dataset_root_of, subjects_path

    root = dataset_root_of(dataset_dir)
    return root if root is not None and subjects_path(root).is_file() else None


def resolve_registry_id_map(labels_dir, scope: "ClassScope"):
    """``(registry, id_map)`` for ``scope``'s subject and attribute from the dataset's
    ``subjects.json``, through :func:`subject_registry.assign_class_ids`. ``scope`` names its
    subject.

    A plain single-class detector (no attribute) needs no registry file: its map is derived from a
    synthesized single-subject registry. Attribute classification needs the registry to order its
    values, and refuses when there is none.
    """
    from tcip_mcp import subject_registry

    subject, attribute = cast(str, scope.subject), scope.attribute
    root = registered_dataset_root(labels_dir)
    if root is not None:
        registry = subject_registry.read_registry(root)
    elif attribute is not None:
        raise ValueError(
            f"attribute {attribute!r} classification needs a subjects.json to order its values, "
            f"but none was found for {labels_dir}.")
    else:
        registry = subject_registry.SubjectRegistry(
            subjects=(subject_registry.Subject(name=subject),))
    return registry, subject_registry.assign_class_ids(registry, subject, attribute)


def stated_scope(labels_dir, subject: str | None, attribute: str | None) -> "ClassScope":
    """A fresh statement of ``subject`` and ``attribute`` over ``labels_dir``, given the map that
    dataset's registry assigns them (:func:`resolve_registry_id_map`); no subject states the empty
    class space and reads no registry."""
    from tcip_mcp.pipelines.data.selection import ClassScope

    stated = ClassScope(subject=subject, attribute=attribute)
    if stated.subject is None:
        return stated
    return ClassScope(subject=subject, attribute=attribute,
                      id_map=resolve_registry_id_map(labels_dir, stated)[1])


def json_det_targets(path, scope: "ClassScope",
                     reads: Callable[[Any], bool] = box_derivable):
    """``(target, n_unlabeled)`` for one image from the name-based per-image JSON, read under
    ``scope``, an admitted document class space.

    ``target`` is the detection target shape, ``{"boxes", "labels", "iscrowd"}`` as parallel lists
    (pixel xyxy, 1-indexed label, crowd flag), and ``"geometry"``, each row's own geometry, which a
    mask is rasterized from. Filters to the scope's subject and the geometry the caller reads
    (``reads``, a loader's own ``reads_geometry``; :func:`~tcip_annotation.state.box_derivable`
    unless stated), then maps each kept annotation to its 0-indexed id via the scope's map, +1 for
    background. An annotation the map cannot decode raises.

    ``n_unlabeled`` counts instances of the subject never assessed for the scope's attribute yet,
    excluded from ``boxes``/``labels`` rather than raising.
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
        cid = json_io.target_class_id(a, cast(str, scope.subject), scope.attribute,
                                      cast(dict, scope.id_map), allow_unlabeled=True)
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
    ``ground_truth``, the place a config names; each name found there is interpreted by
    :func:`~tcip_mcp.pipelines.data.selection.shape_of`.

    A ``.csv`` is a table, one row per sample. A directory holding per-image label documents is the
    document shape and one holding ``<stem>.png`` rasters the mask shape. A directory holding
    neither documents nor masks reads as the document shape, whose own admission then names what is
    missing image by image. A dataset-level COCO sitting among the documents is refused by the
    per-image reader when admission reads it, naming the import door.
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
    records: Mapping[str, tuple[str | None, str | None]], *, scope: "ClassScope",
) -> tuple[list[str], dict[str, int]]:
    """The keys of ``records`` the label store accounts for under ``scope``, an admitted document
    class space, plus the partition that produced them.

    ``records`` maps a caller's own key to ``(label document path, logical image name)``; a
    ``None`` image name is a key with no image at all and a ``None`` document is a key the label
    store holds nothing for. A key is admitted when its document holds the scope's subject, or
    holds none of it and its marks finish it
    (:meth:`~tcip_annotation.json_io.LabelDocument.state`). The counts are ``annotated`` /
    ``confirmed_negative`` / ``skipped_unannotated`` / ``skipped_unconfirmed_empty`` /
    ``skipped_incomplete_attribute``; under a classified scope an image carrying any instance never
    assessed for the attribute is dropped whole, through :func:`json_det_targets`, ahead of every
    other reason.
    """
    from tcip_annotation.json_io import read_label_document

    counts = {"annotated": 0, "confirmed_negative": 0, "skipped_unannotated": 0,
              "skipped_unconfirmed_empty": 0, "skipped_incomplete_attribute": 0}
    keep: list[str] = []
    for key, (label_path, image_name) in records.items():
        if image_name is None or label_path is None or not Path(label_path).is_file():
            counts["skipped_unannotated"] += 1
            continue
        if scope.classified and json_det_targets(label_path, scope)[1]:
            counts["skipped_incomplete_attribute"] += 1
            continue
        state = read_label_document(label_path).state(cast(str, scope.subject))
        reason = {"complete": "annotated", "partial": "annotated",
                  "negative": "confirmed_negative"}.get(state, "skipped_unconfirmed_empty")
        counts[reason] += 1
        if reason != "skipped_unconfirmed_empty":
            keep.append(key)
    return keep, counts


def admitted_documents(
    labels_dir, images_dir, members=None, *, scope: "ClassScope",
) -> tuple[list[Admitted], dict[str, int]]:
    """The members a directory of per-image label documents admits under ``scope``, an admitted
    document class space, resolved, plus the partition that produced them: :func:`admitted_records`
    over each candidate stem paired with the document this directory holds for it (``None`` when
    it holds none) and its real on-disk image name.
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
        scope=scope,
    )
    return [
        Admitted(member=stem,
                 source=source_path_of(refuse_incomplete_band_group(sources[stem])),
                 ground_truth=str(documents[stem]))
        for stem in keep
    ], counts


@dataclass(frozen=True)
class Admitted:
    """One admitted member, as the producer resolved it.

    ``member`` is the name a membership record names it by, the image stem for ground truth that is
    one file per sample and the row key for a table. ``source`` is the image source the admission
    resolved for it, already a path (a ``.bandgroup`` manifest for a grouped capture), and
    ``ground_truth`` the file that answers for it, with ``row_key`` naming its row when one file
    answers for many.
    """

    member: str
    source: str
    ground_truth: str
    row_key: str | None = None


def samples_over(
    records: "Sequence[Admitted]", assignment: dict[str, str],
    group_of: "Callable[[str], str]", *, digests: Mapping[str, str] | None = None,
) -> list["Sample"]:
    """Admitted records as explicit samples, one per record.

    ``assignment`` maps each member to the side it landed on, ``group_of`` gives its group key, and
    ``digests`` its ground-truth digest when the caller computed one. Each sample carries the
    source and the ground truth the admission already resolved for that member.
    """
    from tcip_mcp.pipelines.data.selection import Sample

    digests = digests or {}
    return [
        Sample(
            member=record.member, source=record.source, ground_truth=record.ground_truth, row_key=record.row_key,
            group=group_of(record.member), side=assignment[record.member],
            ground_truth_digest=digests.get(record.member),
        )
        for record in sorted(records, key=lambda r: r.member)
        if record.member in assignment
    ]


def foreground_counts(
    members: "Mapping[str, Sample | Admitted]", scope: "ClassScope",
) -> dict[str, int]:
    """Each admitted member's own foreground count, under the key its caller already holds it by.

    A member is an :class:`Admitted` record or the
    :class:`~tcip_mcp.pipelines.data.selection.Sample` it became. A label document carries a count
    of its own annotations of ``scope``'s subject (scoped to its attribute when one is named), read
    from the path the sample records. A mask raster and a table row count as one each.

    The caller's own index is the result's index.
    """
    from tcip_mcp.pipelines.data.selection import DOCUMENT, shape_of
    from tcip_mcp.pipelines.data.splits import count_label_lines

    return {
        key: (count_label_lines(member.ground_truth, scope)
              if shape_of(member.ground_truth, member.row_key) == DOCUMENT else 1)
        for key, member in members.items()
    }


def is_mask(path: Path) -> bool:
    """Whether one entry is a mask: a file named ``<stem>.png``, and nothing else."""
    return path.is_file() and path.suffix.lower() == ".png"


def admitted_masks(labels_dir, images_dir, members=None) -> tuple[list[Admitted], dict[str, int]]:
    """The members a mask directory admits, resolved, plus the admission counts
    ``{"annotated", "skipped_unannotated"}``.

    A sample needs a mask, and a mask is exactly ``<stem>.png`` under ``labels_dir``
    (:func:`is_mask`): an all-background mask is an explicit annotation, and an image with no mask
    is unannotated rather than a negative. A candidate stem with a file in another format and no
    ``<stem>.png`` refuses by name.
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
    column names and the value its second, both as written. A key naming more than one row refuses
    by name.
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
    """Whether a ground-truth table holds the row one member names."""
    return row_key is not None and row_key in table


def admitted_rows(csv_path, images_dir, members=None) -> tuple[list[Admitted], dict[str, int]]:
    """The rows a ground-truth table admits, resolved, plus the partition that produced them.

    A row is admitted when the table holds it (:func:`holds_row`) and the image it names exists
    under ``images_dir`` (:func:`~tcip_mcp.pipelines.image_utils.list_logical_images`, so a grouped
    capture is admitted by its ``.bandgroup`` manifest).
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


def refuse_inadmissible_samples(samples: "Sequence[Sample]", scope: "ClassScope") -> None:
    """Refuse a recorded sample the platform's own admission would no longer admit, naming which.

    ``scope`` is the class space those samples are read under, whole
    (:class:`~tcip_mcp.pipelines.data.selection.ClassScope`), held to each sample's shape as
    :func:`admit` holds it (``ClassScope.admitted_for``), so a document read under a scope naming
    no subject refuses by name.

    Dispatches once on each sample's own shape: the label document and its marks for a document,
    the mask's own existence for a mask raster, the row's own presence in its table for a table
    row, each over the paths the sample recorded. Every sample's recorded source is resolved once
    (:func:`~tcip_mcp.pipelines.image_utils.resolve_source_path`, which also catches a band group
    missing a sibling).

    Membership is never changed here: a sample the admission no longer holds refuses the run by
    name.
    """
    from tcip_mcp.pipelines.data.selection import DOCUMENT, MASK
    from tcip_mcp.pipelines.image_utils import BandGroupIncomplete, resolve_source_path

    refused: list[str] = []
    unresolved: list[str] = []
    reasons: dict[str, int] = {}
    tables: dict[str, dict[str, str]] = {}
    documents: dict[str, tuple[str | None, str | None]] = {}
    for shape in {s.shape for s in samples}:
        scope.admitted_for(shape, "the class space these samples are read under")
    for sample in samples:
        try:
            image_name: str | None = logical_image_name(resolve_source_path(sample.source))
        except (FileNotFoundError, BandGroupIncomplete):
            unresolved.append(sample.source)
            continue
        if sample.shape == DOCUMENT:
            documents[sample.location] = (sample.ground_truth, image_name)
        elif sample.shape == MASK:
            if not is_mask(Path(sample.ground_truth)):
                refused.append(sample.location)
                reasons["ground_truth_gone"] = reasons.get("ground_truth_gone", 0) + 1
        else:
            table = sample.ground_truth
            if table not in tables:
                tables[table] = (ground_truth_table(table)
                                 if Path(table).is_file() else {})
            if not holds_row(tables[table], sample.row_key):
                refused.append(sample.location)
                reasons["row_gone"] = reasons.get("row_gone", 0) + 1
    if documents:
        admitted, counts = admitted_records(documents, scope=scope)
        for name, value in counts.items():
            if name.startswith("skipped_"):
                reasons[name] = reasons.get(name, 0) + value
        refused.extend(sorted(set(documents) - set(admitted)))
    if not refused and not unresolved:
        return
    named = ", ".join(f"{name}={value}" for name, value in sorted(reasons.items()) if value)
    sources = (f" {len(unresolved)} name a source that no longer resolves ({unresolved[:5]})."
               if unresolved else "")
    raise ValueError(
        f"{len(refused) + len(unresolved)} of this selection's samples are no longer admissible "
        f"({sorted(refused)[:5] + unresolved[:5]}): {named}.{sources} The data moved under the "
        "selection since it was drawn: a label emptied with nobody marking that image complete, "
        "a mask or a label file deleted, a row dropped from its table, an image moved, or an "
        "instance left unassessed for this run's attribute. Restore what those name, finish the "
        "annotation or the mark, or draw the selection again over the current data."
    )


def require_admitted(admitted: "Admission") -> None:
    """Refuse an empty admission, naming why nothing was admitted and what would fix it, for every
    ground-truth shape.
    """
    if admitted.records:
        return
    from tcip_mcp.pipelines.data.selection import DOCUMENT, MASK

    shape, counts = admitted.shape, admitted.tallies
    ground_truth, images_dir = admitted.ground_truth, admitted.images_dir
    if shape == MASK:
        raise ValueError(
            f"no trainable samples: none of the {counts['skipped_unannotated']} image(s) "
            f"in {images_dir} have a <stem>.png mask in {ground_truth}. An image with no mask "
            f"would train as entirely background, so nothing here admits one. Write the masks, or "
            f"point data.labels_dir at the directory holding them."
        )
    if shape != DOCUMENT:
        raise ValueError(
            f"no trainable samples in {ground_truth}: {counts['skipped_no_image']} row(s) "
            f"name an image that is not under {images_dir}. A row naming no image has nothing to "
            f"train. Fix the row keys, or point data.images_dir at the directory holding those "
            f"images."
        )
    incomplete = counts["skipped_incomplete_attribute"]
    incomplete_note = (
        f" {incomplete} more carry at least one instance never assessed for this run's attribute, "
        f"so their ground truth is incomplete for this scope and the whole image is held out "
        f"rather than trained on its labeled subset, finish attributing them, or run without "
        f"an attribute scope."
        if incomplete else ""
    )
    raise ValueError(
        f"no trainable samples in {ground_truth}: "
        f"{counts['skipped_unannotated']} image(s) "
        f"have no label record and {counts['skipped_unconfirmed_empty']} have an empty one "
        f"nobody confirmed. An empty label file is a negative only once a human marks that image "
        f"Complete; until then it reads as unannotated. Annotate some images, or mark the "
        f"genuinely-empty ones Complete.{incomplete_note}"
    )


@dataclass(frozen=True)
class Admission:
    """What one place holding ground truth admits, and the facts every loader over it is built
    from.

    ``shape`` is what that ground truth is, read off the place itself (:func:`ground_truth_shape`),
    never off the task a run states. ``records`` is the admitted set as the producer resolved it,
    each :class:`Admitted` carrying its member name, its image source and its own ground truth.
    :meth:`samples` turns a side assignment over those records into explicit samples.

    ``scope`` and ``date`` are the class space a document admission read class ids under and the
    capture date its documents sit under; the scope is empty and the date ``None`` for a shape no
    registry answers for: a mask raster and a table row carry their own classes.
    """

    shape: str
    images_dir: str
    ground_truth: str
    records: list[Admitted]
    tallies: dict[str, int]
    """How many places the admission kept and skipped, by reason; never a foreground count."""
    scope: "ClassScope"
    date: str | None = None

    def samples(
        self, assignment: dict[str, str], group_of: "Callable[[str], str]",
        digests: Mapping[str, str] | None = None,
    ) -> list["Sample"]:
        """The admitted records this assignment names, as samples on the sides it gives them."""
        return samples_over(self.records, assignment, group_of, digests=digests)

    def every_sample(self) -> list["Sample"]:
        """Every admitted record as a sample, all on the training side, each member grouped by the
        ``stem`` policy's key (:func:`~tcip_mcp.pipelines.data.splits.recorded_group_key_fn`).
        """
        from tcip_mcp.pipelines.data.splits import recorded_group_key_fn

        return self.samples({record.member: "train" for record in self.records},
                            recorded_group_key_fn("stem", date=self.date))


def admit(
    images_dir, ground_truth, *, scope: "ClassScope | None" = None,
    members: list[str] | None = None,
) -> Admission:
    """The membership one place holding ground truth admits, through the admission its own shape
    reads.

    Dispatches once on :func:`ground_truth_shape`: the label documents and their completion marks
    for per-image documents, under ``scope`` and its map; the mask's own existence beside the image for
    mask rasters, and the row's own presence beside a resolvable image for a table, both under an
    empty scope. The admitted scope is held to its shape
    (:meth:`~tcip_mcp.pipelines.data.selection.ClassScope.admitted_for`), so a document scope with
    no subject or no map refuses by name, and a scope naming anything over a mask or a table
    refuses by name, since that ground truth carries its own classes.

    A place that names no ground truth this platform reads, or a directory holding a dataset-level
    COCO, refuses by name. An empty admission is returned as such; :func:`require_admitted` refuses
    it where a non-empty membership is needed.
    """
    from tcip_mcp.pipelines.data.selection import DOCUMENT, MASK, ClassScope

    shape = ground_truth_shape(ground_truth)
    admitted = (scope or ClassScope()).admitted_for(shape, f"the run over {ground_truth}")
    if shape == MASK:
        records, counts = admitted_masks(ground_truth, images_dir, members)
    elif shape != DOCUMENT:
        records, counts = admitted_rows(ground_truth, images_dir, members)
    else:
        from tcip_mcp.dataset_layout import annotation_date

        records, counts = admitted_documents(ground_truth, images_dir, members, scope=admitted)
        return Admission(
            shape=shape, images_dir=str(images_dir), ground_truth=str(ground_truth),
            records=records, tallies=counts, scope=admitted, date=annotation_date(ground_truth),
        )
    return Admission(shape=shape, images_dir=str(images_dir), ground_truth=str(ground_truth),
                     records=records, tallies=counts, scope=admitted)
