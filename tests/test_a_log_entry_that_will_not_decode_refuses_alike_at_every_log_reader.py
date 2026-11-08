"""A log page holding an entry that will not decode refuses alike at the audit log and a shard.

Each log is written by the platform's own producers and its last entry then damaged in the
database; each reader's refusal is compared with the other's, the damaged positions aside.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import tcip_store as ts
from tcip_annotation.verdicts import read_verdicts
from tcip_mcp.audit import audit_entries, audit_log_key
from tcip_mcp.dataset_layout import (
    Decisions,
    Gestures,
    image_dir,
    verdict_key_of,
)
from tests._producer_fixtures import (
    image_label_key,
    label_image,
    staged_box,
    write_image,
)
from tests._record_damage_fixtures import damage_last_log_entry

if TYPE_CHECKING:
    from pathlib import Path


def _unplaced(refused: ts.ReadRefusedError, key: ts.Key) -> str:
    """Return ``refused``'s message with the positions the log ``key`` names left out.

    The message is asserted to name them.
    """
    refusal = refused.served.refusal
    assert isinstance(refusal, ts.Unreadable), refusal
    positions = str(ts.read_log(key).corrupt)
    assert positions in refusal.message, refusal.message
    return refusal.message.replace(positions, "<positions>")


def test_the_audit_log_and_a_verdict_shard_refuse_an_entry_that_will_not_decode_alike(
    tmp_path: Path,
) -> None:
    """The audit log's reader and the verdict shard's refuse a damaged entry in one sentence.

    The shard holds the verdict ``label_image`` recorded and the log the lines of the calls that
    staged and decided it; each refusal is placed at its own log.
    """
    image = write_image(image_dir(tmp_path, "2026-02-11") / "a.jpg", (100, 80))
    bucket = staged_box(tmp_path, image, [40.0, 30.0, 60.0, 50.0])
    label_image(
        image,
        project=tmp_path,
        labels=[],
        gestures=Gestures(
            decide=Decisions(bucket=bucket, verdicts={0: "accepted"})
        ),
    )
    shard = verdict_key_of(image_label_key(image), bucket)
    log = audit_log_key(tmp_path)
    damage_last_log_entry(shard)
    damage_last_log_entry(log)

    with pytest.raises(ts.ReadRefusedError) as audit_refused:
        audit_entries(tmp_path)
    with pytest.raises(ts.ReadRefusedError) as shard_refused:
        read_verdicts(shard)

    assert _unplaced(audit_refused.value, log) == _unplaced(
        shard_refused.value, shard
    )
    assert audit_refused.value.served.where == f"the audit log at {log.root}"
    assert shard_refused.value.served.where == (
        f"the verdict shard {'/'.join(shard.parts)}"
    )
