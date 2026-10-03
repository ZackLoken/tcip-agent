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

from tcip_mcp.pipelines.image_utils import list_logical_images

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.selection import ClassScope, Sample


def registered_dataset_root(dataset_dir) -> Path | None:
    """The root of the dataset containing ``dataset_dir`` when it holds a subject registry, else
    ``None``."""
    from tcip_mcp.dataset_layout import dataset_root_of, subjects_path

    root = dataset_root_of(dataset_dir)
    return root if root is not None and subjects_path(root).is_file() else None


def registry_scope(labels_dir, subject: str | None) -> "ClassScope":
    """The class space ``subject`` is read under over ``labels_dir``: the subject with every
    attribute its dataset's ``subjects.json`` declares for it, none for a dataset holding no
    registry; no subject is the empty class space, reading no registry. A registry not declaring
    ``subject`` refuses by name."""
    from tcip_mcp import subject_registry
    from tcip_mcp.pipelines.data.selection import ClassScope

    if not subject:
        return ClassScope()
    root = registered_dataset_root(labels_dir)
    if root is None:
        return ClassScope(subject=subject, attributes=())
    declared = subject_registry.read_registry(root).subject(subject)
    if declared is None:
        raise subject_registry.RegistryError(
            f"subject {subject!r} is not in the registry at {root}; declare it there first.")
    return ClassScope(subject=subject, attributes=declared.attributes)


def json_det_targets(path, scope: "ClassScope",
                     reads: Callable[[Any], bool] = box_derivable) -> dict[str, Any]:
    """One image's detection target from the name-based per-image JSON, read under ``scope``, an
    admitted document class space.

    ``{"boxes", "labels", "iscrowd"}`` as parallel lists (pixel xyxy, the subject's label 1, crowd
    flag), ``"geometry"``, each row's own geometry, which a mask is rasterized from, and, when the
    scope declares attributes, ``"attributes"``, an int64 array of rows by attributes holding each
    row's id per attribute in declared order (:func:`~tcip_annotation.json_io.attribute_ids`,
    ``UNASSESSED`` where unassessed). Filters to the scope's subject and the geometry the caller
    reads (``reads``, a loader's own ``reads_geometry``;
    :func:`~tcip_annotation.state.box_derivable` unless stated). A value its attribute does not
    declare refuses.
    """
    import numpy as np
    from tcip_annotation import json_io
    from tcip_annotation.state import bbox_of

    attributes = cast(tuple, scope.attributes)
    target: dict[str, Any] = {"boxes": [], "labels": [], "iscrowd": [], "geometry": []}
    rows: list[list[int]] = []
    for a in json_io.read_annotations(path):
        if not reads(a.geometry):
            continue
        ids = json_io.attribute_ids(a, cast(str, scope.subject), attributes)
        if ids is None:
            continue
        # Every reads predicate admits a box or a region only (box_derivable or narrower).
        geometry = cast(BBox | Polygon, a.geometry)
        box = bbox_of(geometry)
        target["boxes"].append([box.x1, box.y1, box.x2, box.y2])
        target["labels"].append(1)
        target["iscrowd"].append(a.iscrowd)
        target["geometry"].append(geometry)
        rows.append(ids)
    if attributes:
        target["attributes"] = np.asarray(rows, dtype=np.int64).reshape(-1, len(attributes))
    return target


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


ABSENT = "absent"
"""The tally of a sample whose ground truth or image does not exist; every other tally is the
:data:`~tcip_annotation.json_io.SubjectState` its ground truth reads as."""


def admits(tally: str) -> bool:
    """Whether a sample tallied ``tally`` is admitted: its ground truth exists and holds the
    subject or a person finished it."""
    return tally not in (ABSENT, "unannotated")


@dataclass(frozen=True)
class Candidate:
    """One member as a place holding ground truth or a recorded sample names it: its image source
    path (``None`` when no image exists for it), its ground truth (``None`` when the place holds
    none for it) and its row key when one table answers for many."""

    member: str
    source: str | None
    ground_truth: str | None
    row_key: str | None = None


def _candidates(shape: str, ground_truth, images_dir, members,
                tables: dict[str, dict[str, str]]) -> list[Candidate]:
    """Every candidate one place holding ground truth of ``shape`` offers: ``members`` when named,
    else every image (a document or a mask directory) or every row (a table). A mask directory
    holding another format beside an image of the same stem, and no ``<stem>.png``, refuses."""
    from tcip_annotation.json_io import prediction_documents
    from tcip_mcp.pipelines.data.selection import DOCUMENT, MASK
    from tcip_mcp.pipelines.image_utils import source_path_of

    sources = {stem: source_path_of(src) for stem, src in list_logical_images(images_dir).items()}
    if shape == DOCUMENT:
        documents = {p.stem: str(p) for p in prediction_documents(ground_truth)}
        names = list(members) if members is not None else sorted(sources)
        return [Candidate(m, sources.get(m), documents.get(m)) for m in names]
    if shape == MASK:
        labels_p = Path(ground_truth)
        entries = list(labels_p.iterdir()) if labels_p.is_dir() else []
        masks = {p.stem: str(p) for p in entries if is_mask(p)}
        names = list(members) if members is not None else sorted(sources)
        unreadable = sorted(p.name for p in entries if p.is_file() and not is_mask(p)
                            and p.stem in names and p.stem not in masks)
        if unreadable:
            raise ValueError(
                f"{labels_p} holds {unreadable} beside an image of the same stem, and a "
                f"mask is read only as <stem>.png; nothing here reads another format, and "
                f"training the image without its mask would train it as entirely background."
            )
        return [Candidate(m, sources.get(m), masks.get(m)) for m in names]
    table = tables.setdefault(str(ground_truth), ground_truth_table(ground_truth))
    names = list(members) if members is not None else sorted(table)
    return [Candidate(k, sources.get(k), str(ground_truth), row_key=k) for k in names]


def _tally(shape: str, candidate: Candidate, scope: "ClassScope",
           tables: dict[str, dict[str, str]]) -> str:
    """What one candidate's ground truth reads as: its label document's state for ``scope``'s
    subject, ``complete`` for a mask or a row that is there, :data:`ABSENT` for anything missing."""
    from tcip_annotation.json_io import read_label_document
    from tcip_mcp.pipelines.data.selection import DOCUMENT, MASK

    found = candidate.ground_truth
    if candidate.source is None or found is None:
        return ABSENT
    if shape == DOCUMENT:
        return (read_label_document(found).state(cast(str, scope.subject))
                if Path(found).is_file() else ABSENT)
    if shape == MASK:
        return "complete" if is_mask(Path(found)) else ABSENT
    if found not in tables:
        tables[found] = ground_truth_table(found) if Path(found).is_file() else {}
    return "complete" if holds_row(tables[found], candidate.row_key) else ABSENT


def _admission(shape: str, candidates: "Sequence[Candidate]", scope: "ClassScope",
               tables: dict[str, dict[str, str]]) -> tuple[list[Admitted], dict[str, int]]:
    """The candidates :func:`admits` admits by their :func:`_tally`, each with its source resolved
    (a grouped capture missing a band refuses), and how many were tallied each way."""
    from tcip_mcp.pipelines.image_utils import resolve_source_path, source_path_of

    records: list[Admitted] = []
    tallies: dict[str, int] = {}
    for candidate in candidates:
        tally = _tally(shape, candidate, scope, tables)
        if admits(tally):
            source = source_path_of(resolve_source_path(cast(str, candidate.source)))
            records.append(Admitted(candidate.member, source, cast(str, candidate.ground_truth),
                                    candidate.row_key))
        tallies[tally] = tallies.get(tally, 0) + 1
    return records, tallies


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
    of its own annotations of ``scope``'s subject, read from the path the sample records. A mask raster and a table row count as one each.

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


def _refused(tallies: Mapping[str, int]) -> str:
    """The tallies :func:`admits` refuses, as ``name=count`` pairs."""
    return ", ".join(f"{name}={count}" for name, count in sorted(tallies.items())
                     if not admits(name))


def refuse_inadmissible_samples(samples: "Sequence[Sample]", scope: "ClassScope") -> None:
    """Refuse recorded samples the admission no longer admits, naming which and the tallies that
    refused them. Each sample is re-admitted as a :class:`Candidate` through the same evaluation
    :func:`admit` runs, under ``scope`` held to its shape (``ClassScope.admitted_for``, so a
    document read under a scope naming no subject refuses by name). Membership is never changed
    here."""
    refused: list[str] = []
    tallies: dict[str, int] = {}
    tables: dict[str, dict[str, str]] = {}
    for shape in sorted({s.shape for s in samples}):
        held = [s for s in samples if s.shape == shape]
        records, counts = _admission(
            shape, [Candidate(s.location, s.source if Path(s.source).exists() else None,
                              s.ground_truth, s.row_key) for s in held],
            scope.admitted_for(shape, "the class space these samples are read under"), tables)
        refused += sorted({s.location for s in held} - {r.member for r in records})
        for name, count in counts.items():
            tallies[name] = tallies.get(name, 0) + count
    if refused:
        raise ValueError(
            f"{len(refused)} of this selection's samples are no longer admissible "
            f"({refused[:5]}): {_refused(tallies)}. The data moved under the selection since it "
            "was drawn: a label emptied with nobody marking that image complete, a mask or a "
            "label file deleted, a row dropped from its table, or an image moved. Restore what "
            "those name, finish the annotation or the mark, or draw the selection again over the "
            "current data."
        )


def require_admitted(admitted: "Admission") -> None:
    """Refuse an empty admission, naming the tallies that refused every candidate and what would
    fix it for its ground-truth shape."""
    if admitted.records:
        return
    from tcip_mcp.pipelines.data.selection import DOCUMENT, MASK

    fix = {
        MASK: f"An image needs its <stem>.png mask in {admitted.ground_truth}, since one with "
              "no mask would train as entirely background. Write the masks, or point "
              "data.labels_dir at the directory holding them.",
        DOCUMENT: "An empty label file is a negative only once a human marks that image "
                  "Complete; until then it reads as unannotated. Annotate some images, or mark "
                  "the genuinely-empty ones Complete.",
    }.get(admitted.shape, f"A row needs an image under {admitted.images_dir}. Fix the row keys, "
                          "or point data.images_dir at the directory holding those images.")
    raise ValueError(f"no trainable samples in {admitted.ground_truth} over "
                     f"{admitted.images_dir}: {_refused(admitted.tallies)}. {fix}")


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
    """How many places the admission tallied each way (:func:`admits`); never a foreground
    count."""
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
    for per-image documents, under ``scope``; the mask's own existence beside the image for mask
    rasters, and the row's own presence beside a resolvable image for a table, both under an empty
    scope. The admitted scope is held to its shape
    (:meth:`~tcip_mcp.pipelines.data.selection.ClassScope.admitted_for`), so a document scope with
    no subject or no attributes read refuses by name, and a scope naming anything over a mask or a table
    refuses by name, since that ground truth carries its own classes.

    A place that names no ground truth this platform reads, or a directory holding a dataset-level
    COCO, refuses by name. An empty admission is returned as such; :func:`require_admitted` refuses
    it where a non-empty membership is needed.
    """
    from tcip_mcp.dataset_layout import annotation_date
    from tcip_mcp.pipelines.data.selection import DOCUMENT, ClassScope

    shape = ground_truth_shape(ground_truth)
    admitted = (scope or ClassScope()).admitted_for(shape, f"the run over {ground_truth}")
    tables: dict[str, dict[str, str]] = {}
    records, counts = _admission(
        shape, _candidates(shape, ground_truth, images_dir, members, tables), admitted, tables)
    return Admission(shape=shape, images_dir=str(images_dir), ground_truth=str(ground_truth),
                     records=records, tallies=counts, scope=admitted,
                     date=annotation_date(ground_truth) if shape == DOCUMENT else None)
