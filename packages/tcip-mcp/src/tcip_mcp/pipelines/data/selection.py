"""A selection: which samples train, which validate, which are held back to calibrate on.

A selection lists, per sample, the image source, where that sample's ground truth lives, the group
key that keeps related samples together, and the side the draw put it on, plus the class space the
draw admitted under. Every sample sharing a group key is on one side, and so is every sample at
one location (the source path, and the rect when a sample is a region of a larger raster).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import tcip_store
from tcip_store import RECORD_JSON, Key, StoreDescriptor, register_store
from tcip_store.file_backend import RootedFileLocator

from tcip_mcp.registry_paths import PathFields, recorded_paths, runtime_paths, within

SIDES = ("train", "val", "calibration", "holdout")
"""The sides a draw assigns. ``train`` and ``val`` build the run's two loaders; ``calibration`` and
``holdout`` build neither and are the reference an assessment fits its operating point on and
checks it against, held out from both training and selection."""

REFERENCE_SIDES = ("calibration", "holdout")
"""The two sides an assessment reads."""

DOCUMENT = "document"
"""Ground truth that is one label document per sample, ``<stem>.json``."""

MASK = "mask"
"""Ground truth that is one mask raster per sample, ``<stem>.png``."""

TABLE = "table"
"""Ground truth that is one row of a table, named by the row's own key."""

GROUND_TRUTH_SHAPES = (DOCUMENT, MASK, TABLE)
"""The three shapes ground truth has here, read off what a record names, never off a run's task."""

SHAPE_DESCRIPTIONS = {
    DOCUMENT: "its own per-image label document",
    MASK: "a <stem>.png mask raster of its own",
    TABLE: "a row of a table, named by its row key",
}
"""How each shape reads in a refusal."""


def shape_of(ground_truth: str, row_key: str | None) -> str:
    """Which of :data:`GROUND_TRUTH_SHAPES` a ground-truth name is, read off the name itself: a row
    key or a ``.csv`` means one row of a table, a ``.png`` means a mask raster of its own, and
    anything else is a label document of its own. Reads no disk.
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

    ``member`` is the name a membership record names this sample by, as its admission resolved it
    (:class:`~tcip_mcp.pipelines.data.label_queries.Admitted`). ``source`` is the image path, or
    the ``.bandgroup`` manifest path standing in for a grouped capture, or the raster path when
    ``rect`` names a region of it. ``ground_truth`` is the path to whatever answers for this
    sample: a per-image label document, a mask raster, or a table; it is never derived from
    ``source``. ``row_key`` names this sample's row inside a tabular ``ground_truth``, and is
    ``None`` when the whole file answers for the sample.

    ``confirmation_bucket`` is the ``image_status.json`` key whose human confirmations admitted
    this sample (:func:`~tcip_mcp.dataset_layout.status_bucket` over a subject and a capture date),
    stated by the producer that admitted it. It is ``None`` for a mask raster or a table row, which
    no confirmation store answers for.

    ``rect`` is the half-open pixel rect ``(x0, y0, x1, y1)`` a within-image draw assigned, or
    ``None`` when the sample is the whole source. ``ground_truth_digest`` is that file's digest at
    draw time.
    """

    member: str
    source: str
    ground_truth: str
    group: str
    side: str
    confirmation_bucket: str | None
    rect: tuple[int, int, int, int] | None = None
    row_key: str | None = None
    ground_truth_digest: str | None = None

    @property
    def location(self) -> str:
        """Where the sample's pixels are: its source path, and the rect when it is a region. The
        key a dataset indexes the sample by; two samples at one location are one sample, whatever
        they are named. Two files at different locations may hold the same pixels, which only
        :func:`source_digests` tells.
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
    """The class space a run's ground truth was admitted under, recorded whole as one ``scope``
    mapping (:func:`dataclasses.asdict`) on a run's data config, a selection and a prediction
    bucket's stamp.

    ``subject`` is the object class a document admission read confirmations and targets for,
    ``attribute`` the value vocabulary it was scoped to when one was named, and ``id_map`` the
    ``assign_class_ids`` map its loader reads targets and decodes predictions under. A mask raster
    and a table row carry their own classes: a run over them records a scope whose every field is
    ``None``.

    An empty subject, attribute or map is ``None``; an attribute with no subject refuses by name.
    """

    subject: str | None = None
    attribute: str | None = None
    id_map: dict[str, int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject", self.subject or None)
        object.__setattr__(self, "attribute", self.attribute or None)
        object.__setattr__(self, "id_map", (
            {str(name): int(cid) for name, cid in self.id_map.items()} if self.id_map else None))
        if self.attribute is not None and self.subject is None:
            raise ValueError(
                f"attribute {self.attribute!r} is stated with no subject: a value with no object "
                "class names nothing a reader could hold it to. State the subject beside it."
            )

    @classmethod
    def of(cls, record: "Mapping[str, Any]") -> "ClassScope":
        """The class space ``record`` carries under ``scope``: a run's data section, a selection
        document or a prediction bucket's stamp. A record carrying no ``scope`` refuses by name.
        """
        if "scope" not in record:
            raise ValueError(
                "this record carries no scope: the class space its ground truth was admitted "
                "under is not recorded, so no reader can hold targets or predictions to it. "
                "Produce it through an admission, which records one."
            )
        return cls(**record["scope"])

    @property
    def classified(self) -> bool:
        """Whether this class space classifies its subject along an attribute."""
        return self.attribute is not None

    @property
    def subjects(self) -> set[str]:
        """The object classes this class space's detections are of: its map's names for a
        detector, its one subject for a classified space or a single-subject read."""
        if self.id_map and not self.classified:
            return set(self.id_map)
        return {self.subject} if self.subject else set()

    def positive_id(self, value: str) -> int | None:
        """The class id this classified space decodes ``value`` under, or ``None`` when it
        classifies no attribute or its map names no such value."""
        if not self.classified or not self.id_map or not value:
            return None
        return self.id_map.get(value)

    def admitted_for(self, shape: str, source: str) -> "ClassScope":
        """This class space, refused by ``source`` when ground truth of ``shape`` cannot be read
        under it: per-image label documents are read for a named subject under its class map, and a
        mask raster or a table row, carrying its own classes, only under the empty class space.
        """
        if shape != DOCUMENT and self != ClassScope():
            raise ValueError(
                f"{source} states a class space ({self}) over {SHAPE_DESCRIPTIONS[shape]}, which "
                "carries its own classes and reads no subject, attribute or map. State the empty "
                "scope for it."
            )
        if shape == DOCUMENT and self.subject is None:
            raise ValueError(
                f"{source} records no subject (data.scope.subject), so nothing names the class "
                "whose records of a per-image label document it reads. State the subject the run "
                "or draw is scoped by."
            )
        if shape == DOCUMENT and self.id_map is None:
            raise ValueError(
                f"{source} records no id_map: per-image label documents are read under the class "
                "map their admission assigned."
            )
        return self


@dataclass(frozen=True)
class Selection:
    """A drawn partition: its samples, the scope they were admitted under, and how they were drawn.

    ``scope`` is the class space the draw admitted through, empty for a draw over ground truth no
    registry scopes, a mask raster or a table row, whose class space a binding run derives from
    that ground truth once (:func:`~tcip_mcp.pipelines.data.split_construction.run_sizes`).
    """

    samples: tuple[Sample, ...]
    seed: int
    group_by: str
    scope: ClassScope
    dataset_fingerprint: str | None = None

    def on(self, side: str) -> list[Sample]:
        """This selection's samples on one side, in recorded order."""
        return [s for s in self.samples if s.side == side]

    def trainable(self) -> list[Sample]:
        """This selection's train-plus-val samples, the pool a redraw draws over."""
        return self.on("train") + self.on("val")

    def counts(self) -> dict[str, int]:
        """How many samples each side holds, every side named even at zero."""
        return {side: sum(1 for s in self.samples if s.side == side) for side in SIDES}


def source_digests(samples: Iterable[Sample]) -> dict[str, str]:
    """Each sample's pixel digest, by its location: sha256 over its source file's bytes (each band
    file of a grouped capture, in band order), each source read once, and its rect when it is a
    region. Two samples with one digest are the same image whatever their files are named."""
    import hashlib

    from tcip_mcp.pipelines.image_utils import BandGroupRef, resolve_source_path

    by_source: dict[str, bytes] = {}
    out: dict[str, str] = {}
    for sample in samples:
        if sample.source not in by_source:
            source = resolve_source_path(sample.source)
            files = list(source.bands.values()) if isinstance(source, BandGroupRef) else [source]
            h = hashlib.sha256()
            for path in files:
                h.update(Path(path).read_bytes())
                h.update(b"\0")
            by_source[sample.source] = h.digest()
        h = hashlib.sha256(by_source[sample.source])
        if sample.rect is not None:
            h.update(repr(sample.rect).encode("utf-8"))
        out[sample.location] = h.hexdigest()
    return out


def refuse_unreadable_samples(samples: Iterable[Sample]) -> None:
    """Refuse a sample no loader here can read whatever its ground truth is, naming what is
    missing: a ``rect`` (a within-image region), which no loader honors.
    """
    samples = list(samples)
    with_rect = [s.location for s in samples if s.rect is not None]
    if with_rect:
        raise ValueError(
            f"{len(with_rect)} sample(s) of this selection name a pixel rect "
            f"({with_rect[:5]}): a within-image region is recorded but no loader windows to it "
            "yet, and reading the whole source instead would train two regions as the same "
            "repeated image. Draw a selection of whole sources for this run."
        )


def refuse_crossing_sides(samples: Sequence[Sample]) -> None:
    """Refuse a sample list whose sides are not a partition, naming what crosses.

    Two things are refused: one location on more than one side (the same pixels trained on and
    selected on), and one group key on more than one side (crops of one parent, or captures of one
    tree, split across sides, which is the leakage the group key exists to prevent).
    """
    by_location: dict[str, set[str]] = {}
    by_group: dict[str, set[str]] = {}
    for sample in samples:
        by_location.setdefault(sample.location, set()).add(sample.side)
        by_group.setdefault(sample.group, set()).add(sample.side)
    crossing_ids = sorted(i for i, sides in by_location.items() if len(sides) > 1)
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
    return Key(SELECTION_STORE, str(Path(selection_dir)), _SELECTION_PARTS)


SAMPLE_PATHS: PathFields = (("source",), ("ground_truth",))
"""The fields of a sample document that name a path."""

SELECTION_PATHS: PathFields = within(("samples", "[]"), SAMPLE_PATHS)
"""The fields of a selection document that name a path."""


def sample_document(sample: Sample) -> dict[str, Any]:
    """The JSON shape one sample is held in, the one :func:`read_sample` reads back."""
    doc: dict[str, Any] = {
        "member": sample.member, "source": sample.source, "ground_truth": sample.ground_truth,
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
    """The JSON shape a selection is held in, the one :func:`as_selection` reads back."""
    return {
        "samples": [sample_document(s) for s in selection.samples],
        "scope": asdict(selection.scope),
        "seed": selection.seed,
        "group_by": selection.group_by,
        "dataset_fingerprint": selection.dataset_fingerprint,
    }


def read_sample(raw: Any, position: int, where: str) -> Sample:
    """One sample document read back as a :class:`Sample`, refusing by name (naming
    ``position`` in the record at ``where``) a non-mapping, a sample missing
    ``member``/``source``/``ground_truth``/``group``/``side``, a side outside :data:`SIDES`, a
    malformed ``rect``, and a sample whose ground truth is its own label document and which names
    no ``confirmation_bucket``."""
    if not isinstance(raw, dict):
        raise ValueError(
            f"sample {position} of the record at {where} is a {type(raw).__name__}, not a mapping.")
    missing = [k for k in ("member", "source", "ground_truth", "group", "side") if not raw.get(k)]
    if missing:
        raise ValueError(
            f"sample {position} of the record at {where} carries no {missing}: every sample names "
            "its own member, source, ground truth, group key and side.")
    side = str(raw["side"])
    if side not in SIDES:
        raise ValueError(
            f"sample {position} of the record at {where} is on side {side!r}, which is none of "
            f"{list(SIDES)}.")
    rect_raw = raw.get("rect")
    rect: tuple[int, int, int, int] | None = None
    if rect_raw is not None:
        if not isinstance(rect_raw, (list, tuple)) or len(rect_raw) != 4:
            raise ValueError(
                f"sample {position} of the record at {where} carries rect {rect_raw!r}, not a "
                "half-open pixel rect (x0, y0, x1, y1).")
        x0, y0, x1, y1 = (int(v) for v in rect_raw)
        rect = (x0, y0, x1, y1)
    row_key = raw.get("row_key")
    bucket = raw.get("confirmation_bucket")
    sample = Sample(
        member=str(raw["member"]), source=str(raw["source"]), ground_truth=str(raw["ground_truth"]),
        group=str(raw["group"]), side=side,
        confirmation_bucket=str(bucket) if bucket else None, rect=rect,
        row_key=str(row_key) if row_key is not None else None,
        ground_truth_digest=raw.get("ground_truth_digest"),
    )
    if sample.confirmation_bucket is None and sample.shape == DOCUMENT:
        raise ValueError(
            f"sample {position} of the record at {where} names its own label document and no "
            "confirmation_bucket: that admission reads a human confirmation store, so the bucket "
            "it read is part of the sample. Draw it again.")
    return sample


def as_selection(document: Any, *, where: str) -> Selection:
    """A held document read back as a :class:`Selection`, refusing by name on anything a draw
    never writes.

    ``where`` names the document in every refusal. Refuses a non-mapping, a missing or empty
    ``samples`` list, a sample missing ``source``/``ground_truth``/``group``/``side``, a side
    outside :data:`SIDES`, a malformed ``rect``, a sample whose ground truth is its own label
    document and which names no ``confirmation_bucket`` (the bucket whose human confirmations
    admitted it, which that shape's admission always reads), a partition whose sides cross
    (:func:`refuse_crossing_sides`), and a scope its samples' shape cannot be read under
    (:meth:`ClassScope.admitted_for`).
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
    samples = [read_sample(raw, position, where) for position, raw in enumerate(raw_samples)]
    refuse_crossing_sides(samples)
    scope = ClassScope.of(document).admitted_for(samples[0].shape, f"the selection at {where}")
    return Selection(
        samples=tuple(samples),
        scope=scope,
        seed=document["seed"],
        group_by=document["group_by"],
        dataset_fingerprint=document["dataset_fingerprint"],
    )


def stored_selection_document(selection: Selection, project: str | Path) -> dict[str, Any]:
    """The document a selection of ``project`` is stored as: :func:`selection_document` with its
    paths stored against ``project`` (:data:`SELECTION_PATHS`)."""
    return recorded_paths(selection_document(selection), SELECTION_PATHS, project)


def write_selection(selection_dir: str | Path, selection: Selection, *,
                    project: str | Path) -> Selection:
    """Write ``selection``, a selection of ``project``, under ``selection_dir``
    (:func:`stored_selection_document`), and answer it back.

    Refuses, before anything is written, whatever :func:`as_selection` refuses, so a selection on
    disk is always one a reader will accept.
    """
    as_selection(selection_document(selection), where=str(selection_dir))
    out_dir = Path(selection_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tcip_store.replace(selection_key(out_dir), stored_selection_document(selection, project))
    return selection


def read_selection(selection_dir: str | Path, *, project: str | Path) -> Selection:
    """The selection of ``project`` recorded under ``selection_dir``.

    Refuses with ``ValueError`` naming ``selection_dir`` when nothing is recorded there, when the
    record will not decode, or when it fails any of :func:`as_selection`'s shape checks. Lets
    :class:`tcip_store.SchemaVersionRefused` propagate.
    """
    document = _read_selection_document(selection_dir, project)
    if document is None:
        raise ValueError(
            f"no selection recorded under {selection_dir}; run draw_splits first.")
    return as_selection(document, where=str(selection_dir))


def read_selection_checked(
    selection_dir: str | Path, *, project: str | Path,
) -> tuple[Selection | None, str | None]:
    """:func:`read_selection` for a caller listing a candidate directory rather than binding to
    it, which must tell "nothing recorded here" apart from "something is recorded here and it is
    wrong".

    Answers ``(selection, None)``, ``(None, None)`` for an absence, or ``(None, text)`` for a
    record that exists and is refused, a schema-version refusal included.
    """
    from tcip_store import SchemaVersionRefused

    try:
        document = _read_selection_document(selection_dir, project)
        if document is None:
            return None, None
        return as_selection(document, where=str(selection_dir)), None
    except SchemaVersionRefused as exc:
        return None, str(exc)
    except ValueError as exc:
        return None, str(exc)


def _read_selection_document(selection_dir: str | Path, project: str | Path) -> Any | None:
    from tcip_store import DecodeError

    try:
        document = tcip_store.read(selection_key(selection_dir), default=None)
    except DecodeError as exc:
        raise ValueError(
            f"the selection at {selection_dir} could not be read: {exc}") from exc
    return None if document is None else runtime_paths(document, SELECTION_PATHS, project)


def digest_bytes(b: bytes) -> str:
    """One ground-truth record's digest from its bytes, ``sha256(bytes)[:16]``."""
    import hashlib

    return hashlib.sha256(b).hexdigest()[:16]


def ground_truth_digest(path: str | Path) -> str:
    """One ground-truth file's own digest (:func:`digest_bytes`), whatever shape the file is. A
    file that is not there raises ``FileNotFoundError`` naming it."""
    return digest_bytes(Path(path).read_bytes())


def ground_truth_digests(paths: Iterable[str]) -> dict[str, str]:
    """Each named file's own digest, keyed by its path and read once per file however many members
    that file answers for."""
    return {path: ground_truth_digest(path) for path in dict.fromkeys(paths)}


def moved_ground_truth(recorded: Mapping[str, str]) -> list[str]:
    """The files among ``recorded`` (each path to the digest recorded for it) that no longer
    digest to it, a file gone since among them, sorted."""
    def moved(path: str, digest: str) -> bool:
        try:
            return ground_truth_digest(path) != digest
        except FileNotFoundError:
            return True

    return sorted(path for path, digest in recorded.items() if moved(path, digest))


def selection_digest(selection: Selection, project: str | Path) -> str:
    """sha256 over the document a selection of ``project`` is stored as
    (:func:`stored_selection_document`)."""
    import hashlib

    return hashlib.sha256(RECORD_JSON.encode(stored_selection_document(selection, project))).hexdigest()


def with_sides(selection: Selection, assignment: dict[str, str]) -> Selection:
    """``selection`` with each sample's side replaced by ``assignment[sample.location]``, for a
    redraw that repartitions a selection's own members without changing which samples it holds.
    Refuses a location the assignment does not name.
    """
    missing = sorted(s.location for s in selection.samples if s.location not in assignment)
    if missing:
        raise ValueError(
            f"the redraw assigns no side to {len(missing)} of the selection's samples "
            f"({missing[:5]}): a redraw states the whole partition it draws.")
    redrawn = tuple(
        replace(sample, side=assignment[sample.location]) for sample in selection.samples)
    refuse_crossing_sides(redrawn)
    return replace(selection, samples=redrawn)
