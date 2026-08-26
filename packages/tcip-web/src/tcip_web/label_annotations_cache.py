"""The mtime-and-size-keyed label-document parse memo shared by every scan of per-image label
files.

``routes/classes.py``'s registry scan and status derivation, ``routes/dataset.py``'s per-date
subject listing, and ``routes/review.py``'s detection-presence scan all re-parse the same files
under a dataset's ``annotations/`` and ``predictions/`` trees; a memo hit is a ``stat``, not a
parse, so the cost of one route's scan is not re-paid by the next.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

_CACHE_MAX = 4096
_cache: "OrderedDict[str, tuple[int, int, list]]" = OrderedDict()


def cached_label_annotations(path: Path) -> list:
    """The typed annotation records ``read_annotations`` parses from ``path``, memoized by mtime
    and size: mtime alone collides for two writes close enough together to land in the same
    filesystem timestamp tick, and a real edit almost always changes the byte count too.

    A missing file reads as no annotations. A present, unreadable one raises
    :class:`~tcip_annotation.json_io.UnreadableLabelDocument`, uncached: a broken document is
    retried, never remembered as an empty result. That includes a present file the OS refuses to
    stat (a permission error): only its absence derives an empty status, never a read failure.
    """
    from tcip_annotation.json_io import UnreadableLabelDocument, read_annotations

    try:
        st = path.stat()
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise UnreadableLabelDocument(f"{path} could not be opened: {exc}") from exc
    fingerprint = (st.st_mtime_ns, st.st_size)
    key = str(path)
    cached = _cache.get(key)
    if cached is not None and (cached[0], cached[1]) == fingerprint:
        _cache.move_to_end(key)
        return cached[2]
    annotations = read_annotations(path)
    _cache[key] = (fingerprint[0], fingerprint[1], annotations)
    _cache.move_to_end(key)
    if len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)
    return annotations
