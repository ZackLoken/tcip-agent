"""A selection: which samples train, which validate, which are held back to calibrate on.

A selection lists, per sample, the image source, where that sample's ground truth lives, the group
key that keeps related samples together, and the side the draw put it on, plus the class space the
draw admitted under. Every sample sharing a group key is on one side, and so is every sample at
one location (the source path, and the rect when a sample is a region of a larger raster).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict, get_args

from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    TypeAdapter,
)

import tcip_store
from tcip_annotation.text import Text
from tcip_mcp.registry_paths import (
    PathFields,
    recorded_paths,
    runtime_paths,
    within,
)
from tcip_mcp.subject_registry import Attribute, named
from tcip_store import Key, encode_record

if TYPE_CHECKING:
    from pydantic import JsonValue

    from tcip_mcp.traits import PositiveState

Side = Literal["train", "val", "calibration", "holdout"]

SIDES: tuple[Side, ...] = get_args(Side)
"""The sides a draw assigns. ``train`` and ``val`` build the run's two loaders; ``calibration`` and
``holdout`` build neither and are the reference an assessment fits its operating point on and
checks it against, held out from both training and selection."""

REFERENCE_SIDES = SIDES[2:]
"""The two reference sides, ``calibration`` and ``holdout``."""

DOCUMENT = "document"
"""Ground truth that is one label document per sample, the record its image's key names."""

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


def shape_of(ground_truth: Key | str, row_key: str | None) -> str:
    """Return which of :data:`GROUND_TRUTH_SHAPES` a sample's ground truth is.

    The shape is read off the value itself: a key is a label document, a row key or a ``.csv``
    path one row of a table, and any other path a mask raster of its own. Reads no disk.
    """
    if isinstance(ground_truth, Key):
        return DOCUMENT
    if row_key is not None or Path(ground_truth).suffix.lower() == ".csv":
        return TABLE
    return MASK


_OMITTED_WHEN_NONE = Field(exclude_if=lambda value: value is None)
"""An optional field of a stored sample, written only when it holds a value."""

_TRANSIENT = Field(exclude=True)
"""A field of a sample this act holds and no record carries."""


@dataclass(frozen=True)
class Sample:
    """One sample of a selection.

    It records where its pixels are, where its ground truth is, what it is grouped with, and
    which side it landed on.

    ``member`` is the name a membership record names this sample by, as its admission resolved it
    (:class:`~tcip_mcp.pipelines.data.label_queries.Admitted`). ``source`` is the image path, or
    the ``.bandgroup`` manifest path standing in for a grouped capture, or the raster path when
    ``rect`` names a region of it. ``ground_truth`` is what answers for this sample: a per-image
    label document's key, or the path of a mask raster or a table. ``row_key`` names this
    sample's row inside a tabular ``ground_truth``, and is ``None`` when the whole file answers
    for the sample.

    ``rect`` is the half-open pixel rect ``(x0, y0, x1, y1)`` a within-image draw assigned, or
    ``None`` when the sample is the whole source. ``ground_truth_digest`` is the ground truth's
    :func:`ground_truth_digest` when it was read, and ``read`` what that one read answered: the
    decoded label document, or the table row's value; ``None`` for a mask and for a sample no
    admission of this act has read; ``stored`` is what the store answered for that read
    (``tcip_store.Versioned``).
    ``image`` is the logical image this act resolved ``source`` to (a
    :class:`~tcip_mcp.pipelines.image_utils.BandGroupRef` for a grouped capture,
    :func:`~tcip_mcp.pipelines.data.label_queries.resolved`). None of the three is ever recorded
    or compared.
    """

    member: Text
    source: Text
    ground_truth: GroundTruth
    group: Text
    side: Side
    rect: Annotated[tuple[int, int, int, int] | None, _OMITTED_WHEN_NONE] = (
        None
    )
    row_key: Annotated[str | None, _OMITTED_WHEN_NONE] = None
    ground_truth_digest: Annotated[str | None, _OMITTED_WHEN_NONE] = None
    read: Annotated[Any, _TRANSIENT] = field(
        default=None, compare=False, repr=False
    )
    stored: Annotated[Any, _TRANSIENT] = field(
        default=None, compare=False, repr=False
    )
    image: Annotated[Any, _TRANSIENT] = field(
        default=None, compare=False, repr=False
    )

    @property
    def location(self) -> str:
        """Where the sample's pixels are: its source path, and the rect when it is a region.

        Two samples at one location are one sample, whatever they are named.
        """
        if self.rect is None:
            return self.source
        return f"{self.source}[{self.rect[0]},{self.rect[1]},{self.rect[2]},{self.rect[3]}]"

    @property
    def shape(self) -> str:
        """Which of :data:`GROUND_TRUTH_SHAPES` this sample's ground truth is.

        The shape is :func:`shape_of`'s.
        """
        return shape_of(self.ground_truth, self.row_key)


@dataclass(frozen=True)
class ClassScope:
    """The class space a run's ground truth was admitted under.

    ``subject`` is the object class a document admission read completion and targets for, and
    ``attributes`` every attribute the registry declares for it, each an
    :class:`~tcip_mcp.subject_registry.Attribute` whose value ids are positions in its declared
    order: ``None`` until the registry is read, empty for a subject that declares none. A mask
    raster and a table row carry their own classes: the scope over them has every field
    ``None``. An empty subject is ``None``.
    """

    subject: str | None = None
    attributes: tuple[Attribute, ...] | None = None
    __pydantic_config__ = ConfigDict(extra="forbid")

    def __post_init__(self) -> None:
        """Hold an empty subject as ``None``."""
        object.__setattr__(self, "subject", self.subject or None)

    @classmethod
    def of(cls, record: Mapping[str, Any]) -> ClassScope:
        """Return the class space ``record`` carries under ``scope``.

        ``record`` is a selection document or a prediction bucket's stamp; its scope is validated
        as a :class:`ClassScope` and read through :meth:`recorded`. A scope naming a key outside
        these fields refuses by name.
        """
        stated = record.get("scope")
        return cls.recorded(
            None if stated is None else _SCOPE.validate_python(stated)
        )

    @staticmethod
    def recorded(scope: ClassScope | None) -> ClassScope:
        """Return ``scope``, the class space a record carries.

        A record carrying none refuses (``ValueError``) by name.
        """
        if scope is None:
            raise ValueError(
                "this record carries no scope: the class space its ground truth was admitted "
                "under is not recorded, so no reader can hold targets or predictions to it. "
                "Produce it through an admission, which records one."
            )
        return scope

    def state_ids(self, state: PositiveState | None) -> tuple[int, int] | None:
        """Return ``(column, value id)`` of ``state`` under this class space.

        That is its attribute's position among the declared attributes and its value's among
        that attribute's values. ``None`` for no state, a space naming no subject, or one whose
        attributes do not list ``state``; a subject's space whose attributes were never read
        refuses (:meth:`document_attributes`).
        """
        if state is None or self.subject is None:
            return None
        attributes = self.document_attributes(f"class space {self}").attributes
        found = named(attributes, state.attribute)
        if found is None or state.value not in found.values:
            return None
        return attributes.index(found), found.values.index(state.value)

    def admitted_for(self, shape: str, source: str) -> ClassScope:
        """Return this class space, refused by ``source`` when ``shape`` cannot be read under it.

        Per-image label documents are read for a named subject under the attributes the registry
        declares for it, and a mask raster or a table row, carrying its own classes, only under
        the empty class space.
        """
        if shape != DOCUMENT and self != ClassScope():
            raise ValueError(
                f"{source} states a class space ({self}) over {SHAPE_DESCRIPTIONS[shape]}, which "
                "carries its own classes and reads no subject or attributes. State the empty "
                "scope for it."
            )
        if shape == DOCUMENT:
            self.document_attributes(source)
        return self

    def document_attributes(self, source: str) -> DocumentScope:
        """Return the subject and attributes per-image label documents are read under here.

        Refused by ``source`` (``ValueError``) when the space names no subject or records no
        attributes.
        """
        if self.subject is None:
            raise ValueError(
                f"{source} records no subject (data.scope.subject), so nothing names the class "
                "whose records of a per-image label document it reads. State the subject the run "
                "or draw is scoped by."
            )
        if self.attributes is None:
            raise ValueError(
                f"{source} records no attributes: per-image label documents are read under every "
                "attribute the registry declares for their subject, which their admission reads."
            )
        return DocumentScope(self.subject, self.attributes)


@dataclass(frozen=True)
class DocumentScope:
    """A class space per-image label documents are read under.

    It is :meth:`ClassScope.document_attributes`' answer: ``subject`` is the object class their
    records are read for and ``attributes`` every attribute the registry declares for it, in
    declared order, empty for a subject that declares none.
    """

    subject: str
    attributes: tuple[Attribute, ...]


_SCOPE = TypeAdapter(ClassScope)


@dataclass(frozen=True)
class Selection:
    """A drawn partition: its samples, the scope they were admitted under, and how it was drawn.

    How it was drawn is its ``seed`` and grouping, ``ratios``, the share each side was requested
    at (``split_construction.check_shares``'s answer), whether it was balanced by foreground
    count (``stratify_foreground``), and ``rationale``, why it was drawn so.

    ``scope`` is the class space the draw admitted through, empty for a draw over ground truth no
    registry scopes, a mask raster or a table row.
    """

    samples: tuple[Sample, ...]
    seed: int
    group_by: str
    scope: ClassScope
    ratios: dict[str, float]
    stratify_foreground: bool
    rationale: str
    dataset_fingerprint: str | None = None

    def on(self, side: str) -> list[Sample]:
        """Return this selection's samples on one side, in recorded order."""
        return [s for s in self.samples if s.side == side]

    def trainable(self) -> list[Sample]:
        """Return this selection's train samples followed by its val samples."""
        return self.on("train") + self.on("val")

    def counts(self) -> dict[str, int]:
        """Return each side's sample count, including zero counts."""
        return {
            side: sum(1 for s in self.samples if s.side == side)
            for side in SIDES
        }


def source_digests(samples: Iterable[Sample]) -> dict[str, str]:
    """Return each sample's pixel digest, by its location.

    The digest is sha256 over its source file's bytes (each band file of a grouped capture, in
    band order), each source read once, and its rect when it is a region. Two samples with one
    digest are the same image whatever their files are named.
    """
    import hashlib

    from tcip_mcp.pipelines.data.label_queries import resolved
    from tcip_mcp.pipelines.image_utils import BandGroupRef

    by_source: dict[str, bytes] = {}
    out: dict[str, str] = {}
    for sample in resolved(list(samples)):
        if sample.source not in by_source:
            source = sample.image
            files = (
                list(source.bands.values())
                if isinstance(source, BandGroupRef)
                else [source]
            )
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
    """Refuse a sample no loader here can read whatever its ground truth is.

    The refusal names what no loader honors: a ``rect`` (a within-image region).
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
    tree, split across sides).
    """
    by_location: dict[str, set[str]] = {}
    by_group: dict[str, set[str]] = {}
    for sample in samples:
        by_location.setdefault(sample.location, set()).add(sample.side)
        by_group.setdefault(sample.group, set()).add(sample.side)
    crossing_ids = sorted(
        i for i, sides in by_location.items() if len(sides) > 1
    )
    crossing_groups = sorted(
        g for g, sides in by_group.items() if len(sides) > 1
    )
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

SELECTION_STORE = "selection"
_SELECTION_PARTS = ("selection",)


def selection_key(selection_dir: str | Path) -> Key:
    """Return the store key of the selection recorded under ``selection_dir``."""
    return Key(SELECTION_STORE, str(Path(selection_dir)), _SELECTION_PARTS)


GROUND_TRUTH_PATHS: PathFields = ((), ("root",))
"""The fields of a stored ground truth (:func:`ground_truth_record`) that name a path: a mask's or
a table's path itself, or a document key's root."""

SAMPLE_PATHS: PathFields = (
    ("source",),
    *within(("ground_truth",), GROUND_TRUTH_PATHS),
)
"""The fields of a sample document that name a path."""

SELECTION_PATHS: PathFields = within(("samples", "[]"), SAMPLE_PATHS)
"""The fields of a selection document that name a path."""


class StoredLabelKey(TypedDict):
    """A label document's key as a record stores it."""

    root: str
    capture: str
    stem: str


def ground_truth_record(ground_truth: Key | str) -> StoredLabelKey | str:
    """Return how a record stores one sample's ground truth.

    A label document's key is stored as its ``root``, ``capture`` and ``stem``, a path as itself.
    """
    if isinstance(ground_truth, Key):
        capture, stem = ground_truth.parts
        return {"root": ground_truth.root, "capture": capture, "stem": stem}
    return ground_truth


def ground_truth_of(stored: StoredLabelKey | str) -> Key | str:
    """Read back a ground truth :func:`ground_truth_record` stored."""
    from tcip_mcp.dataset_layout import label_key

    if isinstance(stored, str):
        return stored
    return label_key(stored["root"], stored["capture"], stored["stem"])


_STORED_GROUND_TRUTH: TypeAdapter[StoredLabelKey | str] = TypeAdapter(
    StoredLabelKey | str
)


def _decoded_ground_truth(stored: object) -> Key | str:
    """Return the ground truth a record stores, validated as :class:`StoredLabelKey` or a path."""
    return ground_truth_of(_STORED_GROUND_TRUTH.validate_python(stored))


GroundTruth = Annotated[
    Key | str,
    BeforeValidator(_decoded_ground_truth),
    PlainSerializer(ground_truth_record),
]
"""A sample's ground truth, a label document's key or a path: read from the form a record stores
it in and written back to that form (:func:`ground_truth_record`)."""


_SELECTION = TypeAdapter(Selection)


def as_selection(document: JsonValue, *, where: str) -> Selection:
    """Read a held document back as a :class:`Selection`, refusing anything a draw never writes.

    The document is validated as the record (``pydantic.ValidationError`` naming each field that
    does not decode, a side outside :data:`SIDES` among them). ``where`` names the document in
    every other refusal: an empty ``samples`` list, a partition whose sides cross
    (:func:`refuse_crossing_sides`), and a scope its samples' shape cannot be read under
    (:meth:`ClassScope.admitted_for`).
    """
    selection = _SELECTION.validate_python(document)
    if not selection.samples:
        raise ValueError(
            f"the selection at {where} lists no samples: a selection is its sample list, so an "
            "empty one binds nothing. Draw it again over the current data."
        )
    refuse_crossing_sides(selection.samples)
    selection.scope.admitted_for(
        selection.samples[0].shape, f"the selection at {where}"
    )
    return selection


def stored_selection_document(
    selection: Selection, project: str | Path
) -> JsonValue:
    """Return the document a selection of ``project`` is stored as.

    That is the selection's record with its paths stored against ``project``
    (:data:`SELECTION_PATHS`).
    """
    return recorded_paths(
        _SELECTION.dump_python(selection, mode="json"),
        SELECTION_PATHS,
        project,
    )


def write_selection(
    selection_dir: str | Path, selection: Selection, *, project: str | Path
) -> Selection:
    """Write ``selection``, a selection of ``project``, under ``selection_dir`` and return it.

    The record is :func:`stored_selection_document`'s.

    Refuses, before anything is written, whatever :func:`as_selection` refuses.
    """
    as_selection(
        _SELECTION.dump_python(selection, mode="json"),
        where=str(selection_dir),
    )
    out_dir = Path(selection_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tcip_store.replace(
        selection_key(out_dir), stored_selection_document(selection, project)
    )
    return selection


def read_selection(
    selection_dir: str | Path, *, project: str | Path
) -> Selection:
    """Read the selection of ``project`` recorded under ``selection_dir``.

    Refuses with ``ValueError`` naming ``selection_dir`` when nothing is recorded there, when the
    record will not decode, or when it fails any of :func:`as_selection`'s shape checks.
    """
    document = _read_selection_document(selection_dir, project)
    if document is None:
        raise ValueError(
            f"no selection recorded under {selection_dir}; run draw_splits first."
        )
    return as_selection(document, where=str(selection_dir))


def read_selection_checked(
    selection_dir: str | Path,
    *,
    project: str | Path,
) -> tuple[Selection | None, str | None]:
    """Return :func:`read_selection`'s selection for ``selection_dir``, or its refusal as text.

    The answer is ``(selection, None)``, ``(None, None)`` for an absence, or ``(None, text)`` for
    a record that exists and is refused.
    """
    try:
        document = _read_selection_document(selection_dir, project)
        if document is None:
            return None, None
        return as_selection(document, where=str(selection_dir)), None
    except ValueError as exc:
        return None, str(exc)


def _read_selection_document(
    selection_dir: str | Path, project: str | Path
) -> JsonValue:
    from tcip_store import DecodeError

    try:
        document = tcip_store.read(selection_key(selection_dir), default=None)
    except DecodeError as exc:
        raise ValueError(
            f"the selection at {selection_dir} could not be read: {exc}"
        ) from exc
    return (
        None
        if document is None
        else runtime_paths(document, SELECTION_PATHS, project)
    )


def ground_truth_digest(ground_truth: Key | str) -> str:
    """Return the sha256 of one ground truth's stored bytes, the store's version token.

    The token is a label document record's (:func:`tcip_store.read_versioned`) or a file's
    (:func:`tcip_store.read_blob_versioned`). One that is absent raises
    ``tcip_store.NotFoundError`` naming it.
    """
    if isinstance(ground_truth, Key):
        return tcip_store.read_versioned(ground_truth).version.token
    return tcip_store.read_blob_versioned(Path(ground_truth)).version.token


def moved_ground_truth(recorded: Mapping[Key | str, str]) -> list[Key | str]:
    """Return the ground truths among ``recorded`` no longer at the digest recorded for each.

    ``recorded`` maps each ground truth to its digest; one gone since counts among the moved, and
    the answer keeps ``recorded``'s order.
    """

    def moved(ground_truth: Key | str, digest: str) -> bool:
        try:
            return ground_truth_digest(ground_truth) != digest
        except tcip_store.NotFoundError:
            return True

    return [gt for gt, digest in recorded.items() if moved(gt, digest)]


def selection_digest(selection: Selection, project: str | Path) -> str:
    """Return the sha256 over the document a selection of ``project`` is stored as.

    The document is :func:`stored_selection_document`'s.
    """
    import hashlib

    return hashlib.sha256(
        encode_record(stored_selection_document(selection, project))
    ).hexdigest()
