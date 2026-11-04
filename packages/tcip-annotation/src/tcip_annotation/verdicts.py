"""The verdict shard: the log of a person's decisions on a prediction bucket's proposals.

One shard covers one image, each entry read through :func:`decode_verdict` alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, get_args

from pydantic import (
    AwareDatetime,
    ConfigDict,
    Field,
    StrictInt,
    TypeAdapter,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass

import tcip_store
from tcip_annotation.json_io import Person
from tcip_store import Key

if TYPE_CHECKING:
    from pydantic import JsonValue

type VerdictAction = Literal["accepted", "rejected"]
VERDICT_ACTIONS: tuple[VerdictAction, ...] = get_args(VerdictAction.__value__)


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class Verdict:
    """One decision on one proposal: which, what, who and when.

    ``proposal`` is the record's index in the bucket's document for the image, ``action`` what
    the person decided, ``by`` who (a person's recorded identity, the label record's own
    spelling) and ``at`` when. Construction validates every field (``pydantic.ValidationError``).
    """

    proposal: Annotated[StrictInt, Field(ge=0)]
    action: VerdictAction
    by: Person
    at: AwareDatetime


_VERDICT = TypeAdapter(Verdict)


def encode_verdict(verdict: Verdict) -> dict[str, JsonValue]:
    """Return ``verdict`` as the entry its shard stores."""
    return _VERDICT.dump_python(verdict, mode="json")


def decode_verdict(entry: JsonValue, where: str) -> Verdict:
    """Return one stored entry, which lives at ``where``, as a :class:`Verdict`.

    An entry the model refuses raises ``tcip_store.ReadRefusedError`` behind ``where``.
    """
    return tcip_store.validated(_VERDICT, entry, where)


REVIEW_VERDICTS_STORE = "review_verdicts"


def verdict_key(dataset_root: str | Path, bucket: str, stem: str) -> Key:
    """Return the key of the verdict shard of the image of stem ``stem`` in bucket ``bucket``.

    ``bucket`` names the prediction bucket; the key hangs under the dataset root the bucket and
    the image's label document belong to.
    """
    return Key(REVIEW_VERDICTS_STORE, str(dataset_root), (bucket, stem))


def record_verdicts(
    txn: tcip_store.Txn, key: Key, verdicts: list[Verdict]
) -> None:
    """Append ``verdicts`` to the shard ``key`` names inside ``txn``.

    Each goes through :func:`encode_verdict`, so they commit with whatever else the transaction
    writes.
    """
    for verdict in verdicts:
        txn.append(key, encode_verdict(verdict))


def read_verdicts(key: Key) -> list[Verdict]:
    """Return every verdict the shard ``key`` names holds, oldest first.

    Each goes through :func:`decode_verdict`, naming the shard and the entry's position; an absent
    shard holds none. A shard that will not read, or holds entries that will not decode (named by
    their positions), raises ``tcip_store.ReadRefusedError``.
    """
    shard = f"the verdict shard {'/'.join(key.parts)}"
    with tcip_store.unreadable(shard, tcip_store.StoreError):
        page = tcip_store.read_log(key)
    if page.corrupt:
        raise tcip_store.ReadRefusedError(
            tcip_store.Unreadable(
                message=f"entries at {page.corrupt} will not decode"
            ),
            shard,
        )
    return [
        decode_verdict(entry, f"entry {i} of {shard}")
        for i, entry in enumerate(page.records)
    ]
