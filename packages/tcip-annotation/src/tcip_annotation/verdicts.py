"""The verdict shard: the log of a person's decisions on a prediction bucket's proposals for one
image, each entry read through :func:`decode_verdict` alone."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, get_args

import tcip_store
from tcip_store import Key

from tcip_annotation.json_io import is_person

VerdictAction = Literal["accepted", "rejected"]
VERDICT_ACTIONS: tuple[VerdictAction, ...] = get_args(VerdictAction)


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
    missing, the proposal is not an index, the action is not one of :data:`VERDICT_ACTIONS`, the
    person is not a person's recorded identity (:func:`~tcip_annotation.json_io.is_person`) or
    the time is not a string."""
    try:
        proposal, action, by, at = (entry["proposal"], entry["action"], entry["by"], entry["at"])
    except KeyError as exc:
        raise ValueError(f"verdict entry {dict(entry)!r} states no {exc.args[0]!r}") from exc
    if (isinstance(proposal, bool) or not isinstance(proposal, int) or proposal < 0
            or action not in VERDICT_ACTIONS or not is_person(by)
            or not isinstance(at, str) or not at):
        raise ValueError(f"verdict entry {dict(entry)!r} is not a proposal index, one of "
                         f"{VERDICT_ACTIONS}, a person and a time")
    return Verdict(proposal=proposal, action=action, by=by, at=at)


REVIEW_VERDICTS_STORE = "review_verdicts"


def verdict_key(dataset_root: str | Path, bucket: str, stem: str) -> Key:
    """The verdict shard of the image of stem ``stem`` under the prediction bucket named
    ``bucket``, under the dataset root the bucket and the image's label document belong to."""
    return Key(REVIEW_VERDICTS_STORE, str(dataset_root), (bucket, stem))


def record_verdicts(txn: tcip_store.Txn, key: Key, verdicts: list[Verdict]) -> None:
    """Append ``verdicts`` to the shard ``key`` names inside ``txn``, each through
    :func:`encode_verdict`, so they commit with whatever else the transaction writes."""
    for verdict in verdicts:
        txn.append(key, encode_verdict(verdict))


def read_verdicts(key: Key) -> list[Verdict]:
    """Every verdict the shard ``key`` names holds, oldest first, each through
    :func:`decode_verdict`; an absent shard holds none. Entries that will not decode refuse
    (``ValueError``) naming their positions."""
    page = tcip_store.read_log(key)
    if page.corrupt:
        raise ValueError(f"verdict shard {key.parts} holds entries at {page.corrupt} that will "
                         "not decode")
    return [decode_verdict(entry) for entry in page.records]
