"""The content-digest-keyed label-document parse memo shared by every scan of per-image label
files.

``routes/classes.py``'s registry scan and status derivation, ``routes/dataset.py``'s per-date
subject listing, and ``routes/review.py``'s detection-presence scan all re-parse the same files
under a dataset's ``annotations/`` and ``predictions/`` trees; a memo hit skips the decode and
the parse, not the file read, so the cost of one route's scan is not re-paid by the next.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path

_CACHE_MAX = 4096
_cache: "OrderedDict[str, tuple[bytes, tuple]]" = OrderedDict()


def cached_label_annotations(path: Path) -> tuple:
    """The typed annotation records parsed from ``path``, memoized by a digest of the bytes
    parsed.

    A file's ``(mtime, size)`` cannot tell two writes apart when they land in the same filesystem
    timestamp tick and leave the byte count unchanged (a coordinate rewritten to the same digit
    count, a subject renamed to an equal-length name); this memo reads the file's current bytes
    on every call and keys the hit on their digest instead, so it never answers a same-size
    in-place edit from a parse made before it. Reading the bytes costs a stat's worth more than a
    stat alone would; only the decode and the record construction are skipped on a hit. A miss
    goes through :func:`tcip_annotation.json_io.annotations_from_bytes`, the one path from a
    document's bytes to its records that every reader shares, so this memo and the reader it
    stands in for cannot disagree about whether a document reads.

    A missing file reads as no annotations. A present, unreadable one raises
    :class:`~tcip_annotation.json_io.UnreadableLabelDocument`, uncached: a broken document is
    retried, never remembered as an empty result. That includes a present file the OS refuses to
    open (a permission error): only its absence derives an empty status, never a read failure.

    The returned tuple and its records are the same objects handed to every other caller reading
    this path under one digest: a caller must never mutate a record in place, and takes a copy
    before changing anything.
    """
    from tcip_annotation.json_io import UnreadableLabelDocument, annotations_from_bytes

    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise UnreadableLabelDocument(f"{path} could not be opened: {exc}") from exc
    digest = hashlib.sha256(data).digest()
    key = str(path)
    cached = _cache.get(key)
    if cached is not None and cached[0] == digest:
        _cache.move_to_end(key)
        return cached[1]
    annotations = tuple(annotations_from_bytes(data, source=key))
    _cache[key] = (digest, annotations)
    _cache.move_to_end(key)
    if len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)
    return annotations
