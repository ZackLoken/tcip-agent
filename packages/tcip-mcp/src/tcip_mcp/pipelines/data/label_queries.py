"""The producer: where a capture's label documents, a directory of masks or a ground-truth table
becomes the samples a run trains, evaluates or calibrates over.

It reads a dataset's per-image label documents with their completion marks by their images' own
keys, its ``<stem>.png`` masks or its table of rows, and its ``subjects.json`` registry, admits by
the shape the ground truth carries (:func:`ground_truth_shape`), and answers an
:class:`Admission`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from tcip_store import Key

from tcip_annotation.state import Annotation, BBox, Polygon, box_derivable

from tcip_mcp.pipelines.image_utils import list_logical_images

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.data.selection import ClassScope, Sample


def registry_scope(images_dir, subject: str | None) -> "ClassScope":
    """The class space ``subject`` is read under over the images ``images_dir`` holds: the subject
    with every attribute their dataset's ``subjects.json`` declares for it, none for a dataset
    holding no registry; no subject is the empty class space, reading no registry. A registry not
    declaring ``subject`` refuses by name."""
    from tcip_mcp import subject_registry
    from tcip_mcp.dataset_layout import dataset_root_of, subjects_path
    from tcip_mcp.pipelines.data.selection import ClassScope

    if not subject:
        return ClassScope()
    root = dataset_root_of(images_dir)
    if root is None or not subjects_path(root).is_file():
        return ClassScope(subject=subject, attributes=())
    declared = subject_registry.read_registry(root).subject(subject)
    if declared is None:
        raise subject_registry.RegistryError(
            f"subject {subject!r} is not in the registry at {root}; declare it there first.")
    return ClassScope(subject=subject, attributes=declared.attributes)


def json_det_targets(annotations: Sequence[Annotation], scope: "ClassScope",
                     reads: Callable[[Any], bool] = box_derivable) -> dict[str, Any]:
    """One image's detection target from its label document's ``annotations``, read under
    ``scope``, an admitted document class space.

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
    for a in annotations:
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
    """Which of :data:`~tcip_mcp.pipelines.data.selection.GROUND_TRUTH_SHAPES` a run's
    ``ground_truth`` names: none names the label documents its images' own keys address, a
    ``.csv`` a table of one row per sample, and a directory holding ``<stem>.png`` rasters masks.
    Anything else refuses (``ValueError``) naming it.
    """
    from tcip_mcp.pipelines.data.selection import DOCUMENT, MASK, TABLE

    if ground_truth is None:
        return DOCUMENT
    path = Path(ground_truth)
    if path.suffix.lower() == ".csv":
        if not path.is_file():
            raise ValueError(
                f"{path} names a ground-truth table that does not exist; name where this run's "
                "ground truth actually lives.")
        return TABLE
    if path.is_dir() and any(is_mask(p) for p in path.iterdir()):
        return MASK
    raise ValueError(
        f"{ground_truth!r} is neither a .csv table nor a directory of <stem>.png masks: a run "
        "names one of those as its ground truth, or names none to read the label documents of "
        "its own images.")


ABSENT = "absent"
"""The tally of a sample whose ground truth or image does not exist; every other tally is the
:data:`~tcip_annotation.json_io.SubjectState` its ground truth reads as."""


def admits(tally: str) -> bool:
    """Whether a sample tallied ``tally`` is admitted: its ground truth exists and holds the
    subject or a person finished it."""
    return tally not in (ABSENT, "unannotated")


@dataclass(frozen=True)
class Candidate:
    """One member as a place holding ground truth or a recorded sample names it: the logical image
    it names (``None`` when no image exists for it), its ground truth (``None`` when the place
    holds none for it) and its row key when one table answers for many."""

    member: str
    source: "Path | BandGroupRef | None"
    ground_truth: Key | str | None
    row_key: str | None = None


_Tables = dict[str, tuple[dict[str, str], Any]]
"""Each ground-truth table one admission read, by path: its rows and what the one read answered
(``tcip_store.Versioned``), no rows for a table that is not there."""


def _table(tables: _Tables, path: str) -> tuple[dict[str, str], Any]:
    """The rows of the table at ``path`` (:func:`parse_ground_truth_table`) and its one read, read
    once per admission."""
    import tcip_store

    if path not in tables:
        stored = tcip_store.read_blob_versioned(Path(path), default=None)
        tables[path] = ({} if stored.version == tcip_store.Version.ABSENT
                        else parse_ground_truth_table(stored.value, path), stored)
    return tables[path]


def _candidates(shape: str, ground_truth, images_dir, members,
                tables: _Tables) -> list[Candidate]:
    """Every candidate one place holding ground truth of ``shape`` offers: ``members`` when named,
    else every image (documents or a mask directory) or every row (a table), an image's label
    document being its own key (:func:`~tcip_mcp.dataset_layout.label_key`). A mask directory
    holding another format beside an image of the same stem, and no ``<stem>.png``, refuses."""
    from tcip_mcp.dataset_layout import label_key_of
    from tcip_mcp.pipelines.data.selection import DOCUMENT, MASK
    from tcip_mcp.pipelines.image_utils import source_path_of

    sources = list_logical_images(images_dir)
    if shape == DOCUMENT:
        names = list(members) if members is not None else sorted(sources)
        return [Candidate(m, sources.get(m),
                          label_key_of(source_path_of(sources[m])) if m in sources else None)
                for m in names]
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
    table, _version = _table(tables, str(ground_truth))
    names = list(members) if members is not None else sorted(table)
    return [Candidate(k, sources.get(k), str(ground_truth), row_key=k) for k in names]


def _acquire(shape: str, ground_truth: "Key | str", row_key: str | None, tables: _Tables, *,
             version: str | None) -> tuple[Any, Any]:
    """The one read of one member's ground truth of ``shape``: what the store answered
    (``tcip_store.Versioned``) and what that decodes to, the label document, the row's value, or
    nothing for a mask; ``(None, None)`` for ground truth that is not there. Each table is read
    once per ``tables``. With ``version``, the token a record carries, ground truth that is not
    there or is at any other version refuses naming both, a document as
    :class:`~tcip_annotation.json_io.UnreadableLabelDocumentError` and anything else as
    ``ValueError``."""
    import tcip_store

    from tcip_annotation.json_io import UnreadableLabelDocumentError, document_at, read_stored
    from tcip_mcp.pipelines.data.selection import DOCUMENT, MASK

    stored: Any = None
    decoded: Any = None
    if shape == DOCUMENT:
        stored = read_stored(cast(Key, ground_truth), default=None)
        if stored.version == tcip_store.Version.ABSENT:
            stored = None
        else:
            decoded = document_at(cast(Key, ground_truth), stored)
    elif shape == MASK:
        if is_mask(Path(cast(str, ground_truth))):
            stored = tcip_store.read_blob_versioned(Path(cast(str, ground_truth)))
    else:
        rows, table = _table(tables, cast(str, ground_truth))
        if holds_row(rows, row_key):
            stored, decoded = table, rows[cast(str, row_key)]
    if version is not None and (stored is None or stored.version.token != version):
        at = "is not there" if stored is None else f"is at version {stored.version.token}"
        refusal = UnreadableLabelDocumentError if shape == DOCUMENT else ValueError
        raise refusal(f"{ground_truth}{f' row {row_key!r}' if row_key else ''} {at}, not "
                      f"{version}, the version recorded")
    return stored, decoded


def _tally(shape: str, candidate: Candidate, scope: "ClassScope",
           tables: _Tables) -> tuple[str, Any, Any]:
    """What one candidate's ground truth reads as, with its one read (:func:`_acquire`): its
    label document's state for ``scope``'s subject, ``complete`` for a mask or a row that is
    there, :data:`ABSENT` (no read) for anything missing."""
    from tcip_mcp.pipelines.data.selection import DOCUMENT

    if candidate.source is None or candidate.ground_truth is None:
        return ABSENT, None, None
    stored, decoded = _acquire(shape, candidate.ground_truth, candidate.row_key, tables,
                               version=None)
    if stored is None:
        return ABSENT, None, None
    return (decoded.state(cast(str, scope.subject)) if shape == DOCUMENT else "complete",
            stored, decoded)


def _admission(shape: str, candidates: "Sequence[Candidate]", scope: "ClassScope",
               tables: _Tables) -> tuple[list[Admitted], dict[str, int]]:
    """The candidates :func:`admits` admits by their :func:`_tally`, each with its source resolved
    (a grouped capture missing a band refuses) and the version its tally read, and how many were
    tallied each way."""
    from tcip_mcp.pipelines.image_utils import refuse_incomplete_band_group

    records: list[Admitted] = []
    tallies: dict[str, int] = {}
    for candidate in candidates:
        tally, stored, read = _tally(shape, candidate, scope, tables)
        if admits(tally):
            image = refuse_incomplete_band_group(cast("Path | BandGroupRef", candidate.source))
            records.append(Admitted(candidate.member, image,
                                    cast("Key | str", candidate.ground_truth),
                                    stored.version.token, read, stored, candidate.row_key))
        tallies[tally] = tallies.get(tally, 0) + 1
    return records, tallies


@dataclass(frozen=True)
class Admitted:
    """One admitted member, as the producer resolved it.

    ``member`` is the name a membership record names it by, the image stem for ground truth that is
    one file per sample and the row key for a table. ``image`` is the logical image the admission
    resolved for it (a ``BandGroupRef`` for a grouped capture), named by :attr:`source`,
    ``ground_truth`` its label document's key or the file that answers for it,
    ``ground_truth_digest`` the version token of the read that admitted it, ``read`` what that
    read decodes to and ``stored`` what the store answered for it
    (:attr:`~tcip_mcp.pipelines.data.selection.Sample.read`), with ``row_key`` naming its row when
    one file answers for many.
    """

    member: str
    image: "Path | BandGroupRef"
    ground_truth: Key | str
    ground_truth_digest: str
    read: Any
    stored: Any
    row_key: str | None = None

    @property
    def source(self) -> str:
        """The path :attr:`image` is named by: a grouped capture's ``.bandgroup`` manifest, a
        plain image's own file."""
        from tcip_mcp.pipelines.image_utils import source_path_of

        return source_path_of(self.image)


def samples_over(
    records: "Sequence[Admitted]", assignment: dict[str, str],
    group_of: "Callable[[str], str]",
) -> list["Sample"]:
    """Admitted records as explicit samples, one per record: ``assignment`` maps each member to
    the side it landed on and ``group_of`` gives its group key. Each sample carries the source,
    the ground truth and its digest the admission resolved for that member.
    """
    from tcip_mcp.pipelines.data.selection import Sample

    return [
        Sample(
            member=record.member, source=record.source, ground_truth=record.ground_truth,
            row_key=record.row_key, group=group_of(record.member),
            side=assignment[record.member], ground_truth_digest=record.ground_truth_digest,
            read=record.read, stored=record.stored, image=record.image,
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
    of its own annotations of ``scope``'s subject, counted in the document its admission read
    (``read``). A mask raster and a table row count as one each.
    """
    from tcip_mcp.pipelines.data.selection import DOCUMENT, shape_of
    from tcip_mcp.pipelines.data.splits import count_label_lines

    return {
        key: (count_label_lines(member.read, scope)
              if shape_of(member.ground_truth, member.row_key) == DOCUMENT else 1)
        for key, member in members.items()
    }


def is_mask(path: Path) -> bool:
    """Whether one entry is a mask: a file named ``<stem>.png``, and nothing else."""
    return path.is_file() and path.suffix.lower() == ".png"


def parse_ground_truth_table(data: bytes, csv_path) -> dict[str, str]:
    """One ground-truth table's bytes as ``{row key: value}``, the row key being the image stem
    its first column names and the value its second, both as written. A key naming more than one
    row refuses naming ``csv_path``.
    """
    import csv as _csv
    import io

    rows: dict[str, str] = {}
    repeated: list[str] = []
    reader = _csv.reader(io.StringIO(data.decode("utf-8-sig"), newline=""))
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


def acquired(samples: "Sequence[Sample]") -> list["Sample"]:
    """``samples``, each carrying its ``read`` and ``stored``: the read this act's admission
    already made, else one read of its ground truth at the version it records
    (:func:`_acquire` at ``ground_truth_digest``, whose refusals propagate), each table read
    once."""
    from dataclasses import replace

    tables: _Tables = {}

    def read(s: "Sample") -> "Sample":
        if s.stored is not None:
            return s
        stored, decoded = _acquire(s.shape, s.ground_truth, s.row_key, tables,
                                   version=s.ground_truth_digest)
        return replace(s, read=decoded, stored=stored)

    return [read(s) for s in samples]


def readmitted_samples(samples: "Sequence[Sample]", scope: "ClassScope") -> list["Sample"]:
    """``samples`` re-admitted, each as a :class:`Candidate` through the same evaluation
    :func:`admit` runs under ``scope`` held to its shape (``ClassScope.admitted_for``, so a
    document read under a scope naming no subject refuses by name), each carrying the version
    that re-admission read as its ``ground_truth_digest``, what it read as its ``read`` and the
    logical image it resolved as its ``image``. A sample the admission no longer admits refuses
    (``ValueError``) naming which and the tallies that refused them; membership is never
    changed."""
    from dataclasses import replace

    from tcip_mcp.pipelines.image_utils import resolve_image_paths

    present = [s.source for s in samples if Path(s.source).exists()]
    logical = dict(zip(present, resolve_image_paths(present)))
    refused: list[str] = []
    tallies: dict[str, int] = {}
    tables: _Tables = {}
    admitted: dict[str, Admitted] = {}
    for shape in sorted({s.shape for s in samples}):
        held = [s for s in samples if s.shape == shape]
        records, counts = _admission(
            shape, [Candidate(s.location, logical.get(s.source), s.ground_truth, s.row_key)
                    for s in held],
            scope.admitted_for(shape, "the class space these samples are read under"), tables)
        admitted.update((r.member, r) for r in records)
        refused += sorted({s.location for s in held} - set(admitted))
        for name, count in counts.items():
            tallies[name] = tallies.get(name, 0) + count
    if refused:
        raise ValueError(
            f"{len(refused)} of this selection's samples are no longer admissible "
            f"({refused[:5]}): {_refused(tallies)}. The data moved under the selection since it "
            "was drawn: a label emptied with nobody marking that image complete, a mask or a "
            "label document deleted, a row dropped from its table, or an image moved. Restore what "
            "those name, finish the annotation or the mark, or draw the selection again over the "
            "current data."
        )
    return [replace(s, ground_truth_digest=admitted[s.location].ground_truth_digest,
                    read=admitted[s.location].read, stored=admitted[s.location].stored,
                    image=admitted[s.location].image)
            for s in samples]


def resolved(samples: "Sequence[Sample]") -> list["Sample"]:
    """``samples``, each carrying its ``image``: the logical image this act's admission already
    resolved, else one resolution of its recorded ``source``
    (:func:`~tcip_mcp.pipelines.image_utils.resolve_image_paths`, each directory scanned once,
    whose refusals propagate)."""
    from dataclasses import replace

    from tcip_mcp.pipelines.image_utils import resolve_image_paths

    unresolved = [s.source for s in samples if s.image is None]
    images = dict(zip(unresolved, resolve_image_paths(unresolved)))
    return [s if s.image is not None else replace(s, image=images[s.source]) for s in samples]


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
        DOCUMENT: "An empty label document is a negative only once a human marks that image "
                  "Complete; until then it reads as unannotated. Annotate some images, or mark "
                  "the genuinely-empty ones Complete.",
    }.get(admitted.shape, f"A row needs an image under {admitted.images_dir}. Fix the row keys, "
                          "or point data.images_dir at the directory holding those images.")
    raise ValueError(f"no trainable samples in the {admitted.shape} ground truth over "
                     f"{admitted.images_dir}: {_refused(admitted.tallies)}. {fix}")


@dataclass(frozen=True)
class Admission:
    """What one place holding ground truth admits, and the facts every loader over it is built
    from.

    ``shape`` is what that ground truth is, read off the place itself (:func:`ground_truth_shape`),
    never off the task a run states. ``records`` is the admitted set as the producer resolved it,
    each :class:`Admitted` carrying its member name, its image source and its own ground truth.
    :meth:`samples` turns a side assignment over those records into explicit samples.

    ``ground_truth`` is the mask directory or table the run named, ``None`` for label documents.
    ``scope`` is the class space a document admission read class ids under, empty for a mask
    raster or a table row, which carry their own classes; ``date`` is the capture the images sit
    under.
    """

    shape: str
    images_dir: str
    ground_truth: str | None
    records: list[Admitted]
    tallies: dict[str, int]
    """How many places the admission tallied each way (:func:`admits`); never a foreground
    count."""
    scope: "ClassScope"
    date: str

    def samples(self, assignment: dict[str, str],
                group_of: "Callable[[str], str]") -> list["Sample"]:
        """The admitted records this assignment names, as samples on the sides it gives them."""
        return samples_over(self.records, assignment, group_of)

    def every_sample(self) -> list["Sample"]:
        """Every admitted record as a sample, all on the training side, each member grouped by the
        ``stem`` policy's key (:func:`~tcip_mcp.pipelines.data.splits.recorded_group_key_fn`).
        """
        from tcip_mcp.pipelines.data.splits import recorded_group_key_fn

        return self.samples({record.member: "train" for record in self.records},
                            recorded_group_key_fn("stem", date=self.date))


def admit(
    images_dir, ground_truth=None, *, scope: "ClassScope | None" = None,
    members: list[str] | None = None,
) -> Admission:
    """The membership the images of ``images_dir`` and their ground truth admit, through the
    admission its own shape reads.

    Dispatches once on :func:`ground_truth_shape`: with no ``ground_truth``, each image's own
    label document and its completion marks, under ``scope``; the mask's own existence beside the
    image for a mask directory, and the row's own presence beside a resolvable image for a table,
    both under an empty scope. The admitted scope is held to its shape
    (:meth:`~tcip_mcp.pipelines.data.selection.ClassScope.admitted_for`), so a document scope with
    no subject or no attributes read refuses by name, and a scope naming anything over a mask or a
    table refuses by name, since that ground truth carries its own classes.

    A ``ground_truth`` this platform does not read, and an ``images_dir`` that is no capture
    (:func:`~tcip_mcp.dataset_layout.parse_capture_dir`), refuse by name. An empty admission is
    returned as such; :func:`require_admitted` refuses it where a non-empty membership is needed.
    """
    from tcip_mcp.dataset_layout import parse_capture_dir
    from tcip_mcp.pipelines.data.selection import ClassScope

    _root, capture = parse_capture_dir(images_dir)
    shape = ground_truth_shape(ground_truth)
    admitted = (scope or ClassScope()).admitted_for(
        shape, f"the run over {ground_truth or images_dir}")
    tables: _Tables = {}
    records, counts = _admission(
        shape, _candidates(shape, ground_truth, images_dir, members, tables), admitted, tables)
    return Admission(shape=shape, images_dir=str(images_dir),
                     ground_truth=None if ground_truth is None else str(ground_truth),
                     records=records, tallies=counts, scope=admitted, date=capture)
