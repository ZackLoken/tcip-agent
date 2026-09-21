"""A selection: which samples train, which validate, which are held back to calibrate on.

A selection lists, per sample, the image source, where that sample's ground truth lives, the group
key that keeps related samples together, and the side the draw put it on. It states no capture
date and no directory scope of its own: every sample names its own source and its own ground
truth, so one selection spans as many capture dates as the draw admitted, and two dates holding a
same-named image are two samples rather than one identity that has to be told apart from itself.

The group key is what a leakage rail rests on: every sample sharing a group key is on one side,
so the tiles cropped from one parent image, or the captures of one tree, cannot be split across
training and validation. A source identity (the source path, and the rect when a sample is a
region of a larger raster) is what a disjointness check rests on: two selections overlap when
they share a group key or a source identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import tcip_store
from tcip_store import RECORD_JSON, Key, StoreDescriptor, register_store
from tcip_store.file_backend import RootedFileLocator

SIDES = ("train", "val", "calibration")
"""The sides a draw assigns. ``train`` and ``val`` build the run's two loaders; ``calibration``
builds neither and is the universe a calibration or an evaluation reads its operating point off,
held out from both training and selection."""

DOCUMENT = "document"
"""Ground truth that is one label document per sample, ``<stem>.json``."""

MASK = "mask"
"""Ground truth that is one mask raster per sample, ``<stem>.png``."""

TABLE = "table"
"""Ground truth that is one row of a table, named by the row's own key."""

GROUND_TRUTH_SHAPES = (DOCUMENT, MASK, TABLE)
"""The three shapes ground truth has here. A shape is a property of the ground truth itself, read
off what a record names rather than decided by the task a run states, so one admission and one
re-admission serve every task."""

SHAPE_DESCRIPTIONS = {
    DOCUMENT: "its own per-image label document",
    MASK: "a <stem>.png mask raster of its own",
    TABLE: "a row of a table, named by its row key",
}
"""How each shape reads in a refusal, stated once so a loader, an admission and a document all
name one shape the same way."""


def shape_of(ground_truth: str, row_key: str | None) -> str:
    """Which of :data:`GROUND_TRUTH_SHAPES` a ground-truth name is, read off the name itself: a
    row key or a ``.csv`` means one row of a table, a ``.png`` means a mask raster of its own, and
    anything else is a label document of its own.

    The one rule, so a loader, a re-admission, a foreground count, a freeze and the sniff over a
    place a config names (:func:`~tcip_mcp.pipelines.data.label_queries.ground_truth_shape`, which
    asks this of what it finds there) cannot disagree about what a name carries, and nothing has
    to carry a second field saying which shape a record already spells. A name, never a file: this
    reads no disk, which is what lets every reader of a record ask it.
    """
    if row_key is not None:
        return TABLE
    suffix = Path(ground_truth).suffix.lower()
    if suffix == ".csv":
        return TABLE
    if suffix == ".png":
        return MASK
    return DOCUMENT


@dataclass(frozen=True)
class Sample:
    """One sample of a selection: where its pixels are, where its ground truth is, what it is
    grouped with, and which side it landed on.

    ``source`` is the image path, or the ``.bandgroup`` manifest path standing in for a grouped
    capture, or the raster path when ``rect`` names a region of it. ``ground_truth`` is the path
    to whatever answers for this sample: a per-image label document, a mask raster, or a table.
    It is never derived from ``source``: ground truth and imagery are two trees, and a selection
    that spans dates cannot recover one from the other. ``row_key`` names this sample's row inside
    a tabular ``ground_truth``, and is ``None`` when the whole file answers for the sample.

    ``confirmation_bucket`` is the ``image_status.json`` key whose human confirmations admitted
    this sample (:func:`~tcip_mcp.dataset_layout.status_bucket` over a subject and a capture
    date). It rides per sample rather than once per selection because a selection spans as many
    capture dates as its draw admitted, and one key for all of them would answer some samples out
    of a bucket nobody wrote them under. It is stated by the producer that admitted the sample,
    never re-derived downstream: the key a writer stated and the date a path happens to spell are
    two different facts. It is ``None`` for a sample no confirmation store answers for: a mask
    raster and a table row are admitted by their own existence beside the image, so there is no
    bucket to name and naming one would claim a human assertion nobody made.

    ``rect`` is the half-open pixel rect ``(x0, y0, x1, y1)`` a within-image draw assigned, or
    ``None`` when the sample is the whole source. ``ground_truth_digest`` is that file's digest at
    draw time, so a reader can say the ground truth moved since without re-reading the draw.
    """

    source: str
    ground_truth: str
    group: str
    side: str
    confirmation_bucket: str | None
    rect: tuple[int, int, int, int] | None = None
    row_key: str | None = None
    ground_truth_digest: str | None = None

    @property
    def identity(self) -> str:
        """The sample's source identity: its source path, and the rect when it is a region.

        Two samples with one identity are the same pixels, whatever they are named or which side
        each landed on, which is what a disjointness check between a training selection and a
        reference selection intersects over.
        """
        if self.rect is None:
            return self.source
        return f"{self.source}[{self.rect[0]},{self.rect[1]},{self.rect[2]},{self.rect[3]}]"

    @property
    def shape(self) -> str:
        """Which of :data:`GROUND_TRUTH_SHAPES` this sample's ground truth is
        (:func:`shape_of`)."""
        return shape_of(self.ground_truth, self.row_key)

    @property
    def member_stem(self) -> str:
        """The name a membership record names this sample by.

        Its row key as written when one ground truth answers for many samples: a row key is
        already the member's own name, so stripping a suffix from it would read ``a.1`` and
        ``a.2`` as one member and collapse two rows into one everywhere a record, a cal/holdout
        lock or a leakage join joins on this. Otherwise the stem of its own ground-truth file,
        which is every sample whose ground truth is one file per sample. The sample states it
        once here rather than each consumer deriving it from whichever of the two fields it
        happens to hold.
        """
        if self.row_key is not None:
            return self.row_key
        return Path(self.ground_truth).stem

    @property
    def ground_truth_scope(self) -> str:
        """Where this sample's ground truth lives: the directory holding its own document, or the
        one document that answers for many samples.

        The container a bare member stem is unique within, so a record listing members per scope
        keeps two directories' same-named images apart. A geometry sample names a document per
        image, so its scope is that document's directory.
        """
        if self.row_key is None:
            return str(Path(self.ground_truth).parent)
        return self.ground_truth


@dataclass(frozen=True)
class ClassScope:
    """The class space a run's ground truth was admitted under.

    ``subject`` is the object class a document admission read confirmations and targets for,
    ``attribute`` the value vocabulary it was scoped to when one was named, and ``id_map`` the
    ``assign_class_ids`` map its loader reads targets under. A mask raster and a table row carry
    their own classes, so a run over them has no scope and every field is ``None``. The one shape
    both a draw's own :class:`~tcip_mcp.pipelines.data.label_queries.Admission` and a recorded
    :class:`Selection` answer with, so nothing downstream restates which facts a class space is.
    """

    subject: str | None = None
    attribute: str | None = None
    id_map: dict[str, int] | None = None

    @classmethod
    def recorded_in(cls, data_cfg: "Mapping[str, Any]") -> "ClassScope":
        """The class space a run's own data config records: the one read of it, for a checkpoint,
        the durable experiment record, a freeze and inference decode alike. A door that reads the
        subject a run admits for reads that key alone, which is not this fact.

        An empty subject, attribute or map reads as "none", the same fact as a missing key, so a
        reader never has to tell one writer's empty form from another's.
        """
        id_map = data_cfg.get("id_map")
        return cls(
            subject=data_cfg.get("subject") or None,
            attribute=data_cfg.get("attribute") or None,
            id_map=({str(name): int(cid) for name, cid in id_map.items()}
                    if isinstance(id_map, dict) and id_map else None),
        )

    def onto(self, data_cfg: "MutableMapping[str, Any]") -> None:
        """Record this class space on a run's own data config, in the one empty form every reader
        of it expects: ``None`` where nothing scopes the run, never an empty map beside a missing
        key. The one write, so a drawn run and a bound run record it the same way."""
        data_cfg["subject"] = self.subject
        data_cfg["attribute"] = self.attribute
        data_cfg["id_map"] = dict(self.id_map) if self.id_map else None


@dataclass(frozen=True)
class Selection:
    """A drawn partition: its samples, the scope they were admitted under, and how they were drawn.

    ``subject``/``attribute``/``id_map`` are the scope the draw admitted through, carried so a run
    binding this selection reads them from here rather than restating them and being checked
    against it. ``id_map`` is the ``assign_class_ids`` map the admission used, so the loader's
    class ids and the draw's are one derivation. All three are empty for a draw over ground truth
    no registry scopes, a mask raster or a table row: the class space a run binding such a
    selection trains in is derived from that ground truth once for the whole run
    (:func:`~tcip_mcp.pipelines.data.split_construction.loader_sizes`).
    """

    samples: tuple[Sample, ...]
    subject: str | None = None
    attribute: str | None = None
    id_map: dict[str, int] = field(default_factory=dict)
    seed: int = 0
    group_by: str = "tile_prefix"
    dataset_fingerprint: str | None = None
    admission_counts: dict[str, int] = field(default_factory=dict)
    realized_ratios: dict[str, float] = field(default_factory=dict)
    origin: dict | None = None

    @property
    def scope(self) -> ClassScope:
        """The class space this selection was drawn under, empty for ground truth no registry
        scopes."""
        return ClassScope(subject=self.subject or None, attribute=self.attribute or None,
                          id_map=dict(self.id_map) if self.id_map else None)

    def on(self, side: str) -> list[Sample]:
        """This selection's samples on one side, in recorded order."""
        return [s for s in self.samples if s.side == side]

    def counts(self) -> dict[str, int]:
        """How many samples each side holds, every side named even at zero."""
        return {side: sum(1 for s in self.samples if s.side == side) for side in SIDES}

    def groups(self) -> set[str]:
        return {s.group for s in self.samples}

    def identities(self) -> set[str]:
        return {s.identity for s in self.samples}


def unscoped_document_issue(selection: Selection, selection_dir: str | Path) -> str | None:
    """Why a selection over per-image label documents that records no subject cannot be read, or
    ``None`` when it can.

    The one statement of that refusal, so a run binding the selection, the preflight that offers
    it and a calibration redrawing its held-out side all say the same thing. A selection over a
    mask raster or a table row records no subject because none scopes it, which is not this
    refusal: only the document shape is subject-scoped.
    """
    if selection.samples and selection.samples[0].shape == DOCUMENT and not selection.subject:
        return (
            f"the selection at {selection_dir} records no subject: a read over per-image label "
            "documents is scoped by the subject the selection was drawn under, and one drawn "
            "without a subject names no class space to read them in."
        )
    return None


def overlap(left: Iterable[Sample], right: Iterable[Sample]) -> dict[str, list[str]]:
    """What two sample sets share: ``{"groups": [...], "identities": [...]}``, each sorted.

    The one disjointness computation. A shared source identity is the same pixels on both sides;
    a shared group key is a sample whose sibling crops, tiles or captures are on the other side,
    which leaks just as surely. Empty lists mean the two are disjoint.
    """
    left, right = list(left), list(right)
    shared_groups = {s.group for s in left} & {s.group for s in right}
    shared_ids = {s.identity for s in left} & {s.identity for s in right}
    return {"groups": sorted(shared_groups), "identities": sorted(shared_ids)}


def refuse_unreadable_samples(samples: Iterable[Sample]) -> None:
    """Refuse a sample no loader here can read whatever its ground truth is, naming what is
    missing rather than letting the loader read something else.

    One field the record carries and no loader honors: ``rect``, a within-image region, which a
    loader would otherwise read as the whole source, so two regions of one raster would train as
    the same repeated image. It refuses here rather than being ignored. The record shape is
    settled either way, so honoring it later changes a loader and not a persisted record. Whether
    a sample's own shape is the one a loader reads is that loader's own declared shape, refused
    once in :meth:`~tcip_mcp.pipelines.data.datasets.BaseImageDataset._init_from_samples`.
    """
    samples = list(samples)
    with_rect = [s.identity for s in samples if s.rect is not None]
    if with_rect:
        raise ValueError(
            f"{len(with_rect)} sample(s) of this selection name a pixel rect "
            f"({with_rect[:5]}): a within-image region is recorded but no loader windows to it "
            "yet, and reading the whole source instead would train two regions as the same "
            "repeated image. Draw a selection of whole sources for this run."
        )


def refuse_crossing_sides(samples: Sequence[Sample]) -> None:
    """Refuse a sample list whose sides are not a partition, naming what crosses.

    Two things are refused: one source identity on more than one side (the same pixels trained on
    and selected on), and one group key on more than one side (crops of one parent, or captures of
    one tree, split across sides, which is the leakage the group key exists to prevent).
    """
    by_identity: dict[str, set[str]] = {}
    by_group: dict[str, set[str]] = {}
    for sample in samples:
        by_identity.setdefault(sample.identity, set()).add(sample.side)
        by_group.setdefault(sample.group, set()).add(sample.side)
    crossing_ids = sorted(i for i, sides in by_identity.items() if len(sides) > 1)
    crossing_groups = sorted(g for g, sides in by_group.items() if len(sides) > 1)
    if crossing_ids:
        raise ValueError(
            f"{len(crossing_ids)} source(s) of this selection are on more than one side "
            f"({crossing_ids[:5]}): the same pixels would be trained on and selected on at once."
        )
    if crossing_groups:
        raise ValueError(
            f"{len(crossing_groups)} group(s) of this selection are on more than one side "
            f"({crossing_groups[:5]}): samples sharing a group key are crops of one parent or "
            "captures of one subject, and splitting them across sides leaks one side into the "
            "other."
        )


# -- the record -----------------------------------------------------------------

_SELECTION_DOC = RootedFileLocator(suffix=".json")
"""A selection directory's own document. The directory is wherever the caller asked the draw to
be written, so no dataset resolver owns its layout."""

SELECTION_STORE = "selection"
_SELECTION_PARTS = ("selection",)
register_store(
    StoreDescriptor(
        name=SELECTION_STORE,
        kind="record",
        key_fields=("document",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        locator=_SELECTION_DOC,
    )
)


def selection_key(selection_dir: str | Path) -> Key:
    """The selection recorded under ``selection_dir``.

    ``last_writer_wins``: a selection is written once, whole, at the end of the draw that produced
    it.
    """
    return Key(SELECTION_STORE, str(Path(selection_dir).absolute()), _SELECTION_PARTS)


def selection_path(selection_dir: str | Path) -> Path:
    """Where a selection's document lives on disk, placed by the store's own locator rather than
    by a second reconstruction of the layout."""
    key = selection_key(selection_dir)
    relative: PurePosixPath = _SELECTION_DOC.relative_path(key.root, key.parts)
    return Path(key.root, *relative.parts)


def _sample_document(sample: Sample) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "source": sample.source, "ground_truth": sample.ground_truth,
        "group": sample.group, "side": sample.side,
    }
    if sample.confirmation_bucket is not None:
        doc["confirmation_bucket"] = sample.confirmation_bucket
    if sample.rect is not None:
        doc["rect"] = list(sample.rect)
    if sample.row_key is not None:
        doc["row_key"] = sample.row_key
    if sample.ground_truth_digest is not None:
        doc["ground_truth_digest"] = sample.ground_truth_digest
    return doc


def selection_document(selection: Selection) -> dict[str, Any]:
    """The JSON shape a selection is written as, built here and read back by :func:`as_selection`,
    so the writer and the reader cannot drift apart on what a selection carries."""
    return {
        "samples": [_sample_document(s) for s in selection.samples],
        "subject": selection.subject,
        "attribute": selection.attribute,
        "id_map": dict(selection.id_map),
        "seed": selection.seed,
        "group_by": selection.group_by,
        "dataset_fingerprint": selection.dataset_fingerprint,
        "admission_counts": dict(selection.admission_counts),
        "realized_ratios": dict(selection.realized_ratios),
        "origin": selection.origin,
    }


def as_selection(document: Any, *, where: str) -> Selection:
    """A recorded document read back as a :class:`Selection`, refusing by name on anything a draw
    never writes.

    ``where`` names the document in every refusal. Refuses a non-mapping, a missing or empty
    ``samples`` list, a sample missing ``source``/``ground_truth``/``group``/``side``, a side
    outside :data:`SIDES`, a malformed ``rect``, a sample whose ground truth is its own label
    document and which names no ``confirmation_bucket`` (the bucket whose human confirmations
    admitted it, which that shape's admission always reads), and a partition whose sides cross
    (:func:`refuse_crossing_sides`).
    """
    if not isinstance(document, dict):
        raise ValueError(
            f"the selection at {where} decodes to a {type(document).__name__}, not the mapping a "
            "draw writes."
        )
    raw_samples = document.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError(
            f"the selection at {where} lists no samples: a selection is its sample list, so an "
            "empty one binds nothing. Draw it again over the current data."
        )
    samples: list[Sample] = []
    for position, raw in enumerate(raw_samples):
        if not isinstance(raw, dict):
            raise ValueError(
                f"sample {position} of the selection at {where} is a "
                f"{type(raw).__name__}, not a mapping.")
        missing = [k for k in ("source", "ground_truth", "group", "side") if not raw.get(k)]
        if missing:
            raise ValueError(
                f"sample {position} of the selection at {where} carries no {missing}: every "
                "sample names its own source, ground truth, group key and side.")
        side = str(raw["side"])
        if side not in SIDES:
            raise ValueError(
                f"sample {position} of the selection at {where} is on side {side!r}, which is "
                f"none of {list(SIDES)}.")
        rect_raw = raw.get("rect")
        rect: tuple[int, int, int, int] | None = None
        if rect_raw is not None:
            if not isinstance(rect_raw, (list, tuple)) or len(rect_raw) != 4:
                raise ValueError(
                    f"sample {position} of the selection at {where} carries rect {rect_raw!r}, "
                    "not a half-open pixel rect (x0, y0, x1, y1).")
            x0, y0, x1, y1 = (int(v) for v in rect_raw)
            rect = (x0, y0, x1, y1)
        row_key = raw.get("row_key")
        bucket = raw.get("confirmation_bucket")
        sample = Sample(
            source=str(raw["source"]), ground_truth=str(raw["ground_truth"]),
            group=str(raw["group"]), side=side,
            confirmation_bucket=str(bucket) if bucket else None, rect=rect,
            row_key=str(row_key) if row_key is not None else None,
            ground_truth_digest=raw.get("ground_truth_digest"),
        )
        if sample.confirmation_bucket is None and sample.shape == DOCUMENT:
            raise ValueError(
                f"sample {position} of the selection at {where} names its own label document "
                "and no confirmation_bucket: that admission reads a human confirmation store, so "
                "the bucket it read is part of the sample. Draw the selection again.")
        samples.append(sample)
    refuse_crossing_sides(samples)
    id_map = document.get("id_map")
    return Selection(
        samples=tuple(samples),
        subject=document.get("subject"),
        attribute=document.get("attribute"),
        id_map=dict(id_map) if isinstance(id_map, dict) else {},
        seed=int(document.get("seed") or 0),
        group_by=str(document.get("group_by") or "tile_prefix"),
        dataset_fingerprint=document.get("dataset_fingerprint"),
        admission_counts=dict(document.get("admission_counts") or {}),
        realized_ratios=dict(document.get("realized_ratios") or {}),
        origin=document.get("origin"),
    )


def write_selection(selection_dir: str | Path, selection: Selection) -> Selection:
    """Write ``selection`` under ``selection_dir`` and answer it back.

    Refuses a crossing partition before anything is written, so a selection on disk is always one
    a reader will accept.
    """
    refuse_crossing_sides(selection.samples)
    out_dir = Path(selection_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tcip_store.replace(selection_key(out_dir), selection_document(selection))
    return selection


def read_selection(selection_dir: str | Path) -> Selection:
    """The selection recorded under ``selection_dir``, the one reader a run, a calibration and the
    data picker all bind through.

    Refuses with ``ValueError`` naming ``selection_dir`` when nothing is recorded there, when the
    record will not decode, or when it fails any of :func:`as_selection`'s shape checks. Lets
    :class:`tcip_store.SchemaVersionRefused` propagate: a record written by a newer platform is a
    real, wrong answer, never the same fact as an absent one.
    """
    document = _read_selection_document(selection_dir)
    if document is None:
        raise ValueError(
            f"no selection recorded under {selection_dir}; run draw_splits first.")
    return as_selection(document, where=str(selection_dir))


def read_selection_checked(
    selection_dir: str | Path,
) -> tuple[Selection | None, str | None]:
    """:func:`read_selection` for a caller listing a candidate directory rather than binding to
    it, which must tell "nothing recorded here" apart from "something is recorded here and it is
    wrong".

    Answers ``(selection, None)``, ``(None, None)`` for an absence, or ``(None, text)`` for a
    record that exists and is refused, a schema-version refusal included.
    """
    from tcip_store import SchemaVersionRefused

    try:
        document = _read_selection_document(selection_dir)
        if document is None:
            return None, None
        return as_selection(document, where=str(selection_dir)), None
    except SchemaVersionRefused as exc:
        return None, str(exc)
    except ValueError as exc:
        return None, str(exc)


def _read_selection_document(selection_dir: str | Path) -> Any | None:
    from tcip_store import DecodeError

    try:
        return tcip_store.read(selection_key(selection_dir), default=None)
    except DecodeError as exc:
        raise ValueError(
            f"the selection at {selection_dir} could not be read: {exc}") from exc


def with_sides(selection: Selection, assignment: dict[str, str]) -> Selection:
    """``selection`` with each sample's side replaced by ``assignment[sample.identity]``, for a
    redraw that repartitions a selection's own members without changing which samples it holds.

    Refuses an identity the assignment does not name, rather than leaving it on the side the
    original draw chose: a redraw states the whole partition it draws.
    """
    missing = sorted(s.identity for s in selection.samples if s.identity not in assignment)
    if missing:
        raise ValueError(
            f"the redraw assigns no side to {len(missing)} of the selection's samples "
            f"({missing[:5]}): a redraw states the whole partition it draws.")
    redrawn = tuple(
        replace(sample, side=assignment[sample.identity]) for sample in selection.samples)
    refuse_crossing_sides(redrawn)
    return replace(selection, samples=redrawn)
