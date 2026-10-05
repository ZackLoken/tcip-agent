"""A selection: which samples train, which validate, which are held back to calibrate on.

A selection lists, per sample, the image source, where that sample's ground truth lives, the group
key that keeps related samples together, and the side the draw put it on, plus the class space the
draw admitted under. Every sample sharing a group key is on one side, and so is every sample at
one location (the source path, and the rect when a sample is a region of a larger raster).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence, cast

import tcip_store
from tcip_store import Key, encode_record

from tcip_mcp.registry_paths import PathFields, recorded_paths, runtime_paths, within
from tcip_mcp.subject_registry import Attribute, named

if TYPE_CHECKING:
    from tcip_mcp.traits import PositiveState

SIDES = ("train", "val", "calibration", "holdout")
"""The sides a draw assigns. ``train`` and ``val`` build the run's two loaders; ``calibration`` and
``holdout`` build neither and are the reference an assessment fits its operating point on and
checks it against, held out from both training and selection."""

REFERENCE_SIDES = ("calibration", "holdout")
"""The two sides an assessment reads."""

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
    """Which of :data:`GROUND_TRUTH_SHAPES` a sample's ground truth is, read off the value itself:
    a key is a label document, a row key or a ``.csv`` path one row of a table, and any other path
    a mask raster of its own. Reads no disk."""
    if isinstance(ground_truth, Key):
        return DOCUMENT
    if row_key is not None or Path(ground_truth).suffix.lower() == ".csv":
        return TABLE
    return MASK


@dataclass(frozen=True)
class Sample:
    """One sample of a selection: where its pixels are, where its ground truth is, what it is
    grouped with, and which side it landed on.

    ``member`` is the name a membership record names this sample by, as its admission resolved it
    (:class:`~tcip_mcp.pipelines.data.label_queries.Admitted`). ``source`` is the image path, or
    the ``.bandgroup`` manifest path standing in for a grouped capture, or the raster path when
    ``rect`` names a region of it. ``ground_truth`` is what answers for this sample: a per-image
    label document's key, or the path of a mask raster or a table. ``row_key`` names this
    sample's row inside a tabular ``ground_truth``, and is ``None`` when the whole file answers
    for the sample.

    ``rect`` is the half-open pixel rect ``(x0, y0, x1, y1)`` a within-image draw assigned, or
    ``None`` when the sample is the whole source. ``ground_truth_digest`` is the ground truth's
    :func:`ground_truth_digest` when it was read.
    """

    member: str
    source: str
    ground_truth: Key | str
    group: str
    side: str
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


@dataclass(frozen=True)
class ClassScope:
    """The class space a run's ground truth was admitted under, recorded whole as one ``scope``
    mapping (:func:`dataclasses.asdict`) on a run's data config, a selection and a prediction
    bucket's stamp.

    ``subject`` is the object class a document admission read completion and targets for, and
    ``attributes`` every attribute the registry declares for it, each an
    :class:`~tcip_mcp.subject_registry.Attribute` whose value ids are positions in its declared
    order: ``None`` until the registry is read, empty for a subject that declares none. A mask
    raster and a table row carry their own classes: a run over them records a scope whose every
    field is ``None``. An empty subject is ``None``.
    """

    subject: str | None = None
    attributes: tuple[Attribute, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject", self.subject or None)

    @classmethod
    def of(cls, record: "Mapping[str, Any]") -> "ClassScope":
        """The class space ``record`` carries under ``scope``: a run's data section, a selection
        document or a prediction bucket's stamp, each attribute rebuilt from the mapping
        :func:`dataclasses.asdict` wrote. A record carrying no ``scope`` refuses by name.
        """
        if "scope" not in record:
            raise ValueError(
                "this record carries no scope: the class space its ground truth was admitted "
                "under is not recorded, so no reader can hold targets or predictions to it. "
                "Produce it through an admission, which records one."
            )
        stated = dict(record["scope"])
        if stated.get("attributes") is not None:
            stated["attributes"] = tuple(Attribute(**{**a, "values": tuple(a["values"])})
                                         for a in stated["attributes"])
        return cls(**stated)

    def state_ids(self, state: "PositiveState | None") -> tuple[int, int] | None:
        """``(column, value id)`` of ``state`` under this class space: its attribute's position
        among the declared attributes and its value's among that attribute's values. ``None`` for
        no state, a space naming no subject, or one whose attributes do not list ``state``; a
        subject's space whose attributes were never read refuses (:meth:`admitted_for`)."""
        if state is None or self.subject is None:
            return None
        attributes = cast(tuple, self.admitted_for(DOCUMENT, f"class space {self}").attributes)
        found = named(attributes, state.attribute)
        if found is None or state.value not in found.values:
            return None
        return attributes.index(found), found.values.index(state.value)

    def admitted_for(self, shape: str, source: str) -> "ClassScope":
        """This class space, refused by ``source`` when ground truth of ``shape`` cannot be read
        under it: per-image label documents are read for a named subject under the attributes the
        registry declares for it, and a mask raster or a table row, carrying its own classes, only
        under the empty class space.
        """
        if shape != DOCUMENT and self != ClassScope():
            raise ValueError(
                f"{source} states a class space ({self}) over {SHAPE_DESCRIPTIONS[shape]}, which "
                "carries its own classes and reads no subject or attributes. State the empty "
                "scope for it."
            )
        if shape == DOCUMENT and self.subject is None:
            raise ValueError(
                f"{source} records no subject (data.scope.subject), so nothing names the class "
                "whose records of a per-image label document it reads. State the subject the run "
                "or draw is scoped by."
            )
        if shape == DOCUMENT and self.attributes is None:
            raise ValueError(
                f"{source} records no attributes: per-image label documents are read under every "
                "attribute the registry declares for their subject, which their admission reads."
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

SELECTION_STORE = "selection"
_SELECTION_PARTS = ("selection",)


def selection_key(selection_dir: str | Path) -> Key:
    """The selection recorded under ``selection_dir``, wherever the caller asked the draw to be
    written, once and whole at the end of the draw that produced it."""
    return Key(SELECTION_STORE, str(Path(selection_dir)), _SELECTION_PARTS)


GROUND_TRUTH_PATHS: PathFields = ((), ("root",))
"""The fields of a stored ground truth (:func:`ground_truth_record`) that name a path: a mask's or
a table's path itself, or a document key's root."""

SAMPLE_PATHS: PathFields = (("source",), *within(("ground_truth",), GROUND_TRUTH_PATHS))
"""The fields of a sample document that name a path."""

SELECTION_PATHS: PathFields = within(("samples", "[]"), SAMPLE_PATHS)
"""The fields of a selection document that name a path."""


def ground_truth_record(ground_truth: Key | str) -> str | dict[str, str]:
    """How a record stores one sample's ground truth, the one :func:`ground_truth_of` reads back:
    a label document's key as its ``root``, ``capture`` and ``stem``, a path as itself."""
    if isinstance(ground_truth, Key):
        capture, stem = ground_truth.parts
        return {"root": ground_truth.root, "capture": capture, "stem": stem}
    return ground_truth


def ground_truth_of(stored: Any) -> Key | str:
    """A ground truth :func:`ground_truth_record` stored, read back."""
    from tcip_mcp.dataset_layout import label_key

    if isinstance(stored, dict):
        return label_key(stored["root"], stored["capture"], stored["stem"])
    return str(stored)


def sample_document(sample: Sample) -> dict[str, Any]:
    """The JSON shape one sample is held in, the one :func:`read_sample` reads back."""
    doc: dict[str, Any] = {
        "member": sample.member, "source": sample.source,
        "ground_truth": ground_truth_record(sample.ground_truth),
        "group": sample.group, "side": sample.side,
    }
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
    ``member``/``source``/``ground_truth``/``group``/``side``, a side outside :data:`SIDES`, and
    a malformed ``rect``."""
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
    return Sample(
        member=str(raw["member"]), source=str(raw["source"]),
        ground_truth=ground_truth_of(raw["ground_truth"]),
        group=str(raw["group"]), side=side, rect=rect,
        row_key=str(row_key) if row_key is not None else None,
        ground_truth_digest=raw.get("ground_truth_digest"),
    )


def as_selection(document: Any, *, where: str) -> Selection:
    """A held document read back as a :class:`Selection`, refusing by name on anything a draw
    never writes.

    ``where`` names the document in every refusal. Refuses a non-mapping, a missing or empty
    ``samples`` list, a sample missing ``source``/``ground_truth``/``group``/``side``, a side
    outside :data:`SIDES`, a malformed ``rect``, a partition whose sides cross
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
    record will not decode, or when it fails any of :func:`as_selection`'s shape checks.
    """
    document = _read_selection_document(selection_dir, project)
    if document is None:
        raise ValueError(
            f"no selection recorded under {selection_dir}; run draw_splits first.")
    return as_selection(document, where=str(selection_dir))


def read_selection_checked(
    selection_dir: str | Path, *, project: str | Path,
) -> tuple[Selection | None, str | None]:
    """:func:`read_selection` answering ``(selection, None)``, ``(None, None)`` for an absence, or
    ``(None, text)`` for a record that exists and is refused."""
    try:
        document = _read_selection_document(selection_dir, project)
        if document is None:
            return None, None
        return as_selection(document, where=str(selection_dir)), None
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


def ground_truth_digest(ground_truth: Key | str) -> str:
    """The sha256 of one ground truth's stored bytes, the store's version token: a label document
    record's (:func:`tcip_store.read_versioned`) or a file's
    (:func:`tcip_store.read_blob_versioned`). One that is absent raises ``tcip_store.NotFound``
    naming it."""
    if isinstance(ground_truth, Key):
        return tcip_store.read_versioned(ground_truth).version.token
    return tcip_store.read_blob_versioned(Path(ground_truth)).version.token


def moved_ground_truth(recorded: Mapping[Key | str, str]) -> list[Key | str]:
    """The ground truths among ``recorded`` (each to the digest recorded for it) that no longer
    digest to it, one gone since among them, in ``recorded``'s order."""
    def moved(ground_truth: Key | str, digest: str) -> bool:
        try:
            return ground_truth_digest(ground_truth) != digest
        except tcip_store.NotFound:
            return True

    return [gt for gt, digest in recorded.items() if moved(gt, digest)]


def selection_digest(selection: Selection, project: str | Path) -> str:
    """sha256 over the document a selection of ``project`` is stored as
    (:func:`stored_selection_document`)."""
    import hashlib

    return hashlib.sha256(encode_record(stored_selection_document(selection, project))).hexdigest()


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
