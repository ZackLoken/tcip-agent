"""The content-digest-keyed label-document parse memo shared by every scan of per-image label
files.

A memo hit skips the decode and the parse, not the file read.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path

_CACHE_MAX = 4096
_cache: "OrderedDict[str, tuple[bytes, tuple]]" = OrderedDict()


def cached_label_annotations(path: Path) -> tuple:
    """The typed annotation records parsed from ``path``, memoized by a digest of the bytes parsed.

    Reads the file's current bytes on every call and keys the hit on their digest, so a same-size
    in-place edit is never answered from an earlier parse. A miss goes through
    :func:`tcip_annotation.json_io.annotations_from_bytes`.

    A missing file reads as no annotations. A present, unreadable one raises
    :class:`~tcip_annotation.json_io.UnreadableLabelDocument`, uncached. That includes a present
    file the OS refuses to open (a permission error).

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
