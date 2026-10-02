"""The verdict shard: the log of a person's decisions on a prediction bucket's proposals for one
image, each entry read through :func:`decode_verdict` alone."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, get_args

import tcip_store
from tcip_store import LOG_JSON, Key, StoreDescriptor, register_store
from tcip_store.layout_claims import REVIEW_SHARD_DIRNAME, REVIEW_SHARD_SUFFIX

VerdictAction = Literal["accepted", "rejected"]
VERDICT_ACTIONS: tuple[VerdictAction, ...] = get_args(VerdictAction)

_PATH_HOSTILE = '\\/:*?"<>|'


@dataclass(frozen=True)
class Verdict:
    """One decision: ``proposal`` is the record's index in the bucket's document for the image,
    ``action`` what the person decided, ``by`` who (the label record's own spelling) and ``at``
    when."""

    proposal: int
    action: VerdictAction
    by: str
    at: str


def encode_verdict(verdict: Verdict) -> dict:
    """``verdict`` as the entry its shard stores."""
    return asdict(verdict)


def decode_verdict(entry: Mapping) -> Verdict:
    """One stored entry as a :class:`Verdict`, or ``ValueError`` naming the entry when a key is
    missing, the proposal is not an index, the action is not one of :data:`VERDICT_ACTIONS` or
    the person or time is not a string."""
    try:
        proposal, action, by, at = (entry["proposal"], entry["action"], entry["by"], entry["at"])
    except KeyError as exc:
        raise ValueError(f"verdict entry {dict(entry)!r} states no {exc.args[0]!r}") from exc
    if (isinstance(proposal, bool) or not isinstance(proposal, int) or proposal < 0
            or action not in VERDICT_ACTIONS or not isinstance(by, str) or not by
            or not isinstance(at, str) or not at):
        raise ValueError(f"verdict entry {dict(entry)!r} is not a proposal index, one of "
                         f"{VERDICT_ACTIONS}, a person and a time")
    return Verdict(proposal=proposal, action=action, by=by, at=at)


def _sanitized(key: str) -> str:
    """``key`` with path-hostile characters folded to ``_``, hash-suffixed when that changed it,
    so two keys that fold alike keep distinct files."""
    safe = key
    for ch in _PATH_HOSTILE:
        safe = safe.replace(ch, "_")
    if safe != key:
        safe = f"{safe}.{hashlib.sha1(key.encode('utf-8')).hexdigest()[:8]}"
    return safe


@dataclass(frozen=True)
class _ShardLocator:
    """Places one (bucket, image) shard under ``review/<bucket>/<image>.jsonl``, its parts as
    :func:`verdict_key` spells them."""

    def relative_path(self, scope: str, parts: tuple[str, ...]) -> PurePosixPath:
        bucket, img_name = parts
        return PurePosixPath(REVIEW_SHARD_DIRNAME, bucket, f"{img_name}{REVIEW_SHARD_SUFFIX}")

    def parts_from(self, relative_path: PurePosixPath) -> tuple[str, ...] | None:
        segments = relative_path.parts
        if (len(segments) != 3 or segments[0] != REVIEW_SHARD_DIRNAME
                or not segments[2].endswith(REVIEW_SHARD_SUFFIX)):
            return None
        return (segments[1], segments[2][: -len(REVIEW_SHARD_SUFFIX)])


REVIEW_VERDICTS_STORE = "review_verdicts"
register_store(
    StoreDescriptor(
        name=REVIEW_VERDICTS_STORE,
        kind="log",
        key_fields=("bucket", "image"),
        frozen=True,
        codec=LOG_JSON,
        enumerable=True,
        locator=_ShardLocator(),
    )
)


def verdict_key(state_dir: str | Path, bucket: str, img_name: str) -> Key:
    """One image's verdict shard under one prediction bucket, in a dataset's state directory: each
    part with path-hostile characters folded, so the shard's own path spells its key."""
    return Key(REVIEW_VERDICTS_STORE, str(state_dir), (_sanitized(bucket), _sanitized(img_name)))


def record_verdicts(key: Key, verdicts: list[Verdict]) -> None:
    """Append ``verdicts`` to the shard ``key`` names, each through :func:`encode_verdict`."""
    for verdict in verdicts:
        tcip_store.append(key, encode_verdict(verdict))


def read_verdicts(key: Key) -> list[Verdict]:
    """Every verdict the shard ``key`` names holds, oldest first, each through
    :func:`decode_verdict`; an absent shard holds none. Entries that will not decode, or whose
    schema version this reader does not know, refuse (``ValueError``) naming their positions."""
    page = tcip_store.read_log(key)
    if page.corrupt or page.version_refused:
        raise ValueError(f"verdict shard {key.parts} holds entries at {page.corrupt} that will not "
                         f"decode and at {page.version_refused} of an unknown version")
    return [decode_verdict(entry) for entry in page.records]
