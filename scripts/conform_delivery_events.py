"""Check a project's stored ``delivery_events`` records against the current
``DeliveryEventRecord`` shape (``tcip_mcp.pipelines.delivery_events_schema``) and name, by
``event_id``, any that no longer validate.

This script conforms almost entirely by checking and naming, never by rewriting: it is a one-off
operator tool per this repo's no-backward-compatibility boundary (CLAUDE.md), and unlike
``conform_view_coverage_viewing.py`` it has almost nothing to reshape. A ``delivery_events`` record
predating the three walked-mapping ``plant_mapping`` disclosure keys (``dates_delivered``,
``images_unattributed``, ``plant_attribution``) or the delivered file's own ``output_sha256``
carries no value for them anywhere on the record: they were never computed for that delivery, so
there is no old shape to map forward, only a gap to name. A refused record is left exactly as
stored; the only remedy is a fresh delivery through the door that writes this record, or removing
the record by hand.

``plant_mapping`` carries one of three disclosure shapes, or ``null`` (``delivery_events_schema.py``):
a walked capture mapping's, or ``deliver_orthomosaic_plant_counts``'s own whole-raster registry or
canopy-segment disclosure. Two write-forwards exist, both applied together when both gaps are
present. The first is ``acknowledged_by``/``acknowledgement_reason``: a record written before this
pair existed carries no acknowledgement of any kind, so ``null`` on both is not a guess but the
true value, derivable from the record's own age rather than from anything it states. The second is
a registry disclosure's ``plants_outside_raster``, added after that form already shipped: its true
value is derivable from the event's own registry (name and digest) and raster identity (width,
height and geotransform) through :func:`~tcip_mcp.pipelines.postprocessing.orthomosaic_mapping.
plants_in_frame`, recomputed here rather than guessed, and refused by name when the named registry
no longer loads or its digest has moved (the value cannot be reconstructed from a registry that is
not the one the delivery read). Every other gap is still only named.

``--plan`` previews what would be written without writing it; without it, an applicable record's
write-forward runs. Every other refused record is unaffected by either mode.

    python scripts/conform_delivery_events.py <project_root> [<project_root> ...]
    python scripts/conform_delivery_events.py --plan <project_root>

Exit codes: 0 when every stored record in every named root validates (or a root holds none, or the
write-forward above conforms it); 2 if any record in any root is refused (in ``--plan`` mode, this
includes a record the write-forward would fix but has not yet), or a named root does not exist or
holds no ``.tcip`` directory to check. Every root named on the command line is still checked and
reported, even after another root's refusal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-mcp" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-store" / "src"))

import tcip_store as ts  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from tcip_mcp.pipelines.delivery_events_schema import (  # noqa: E402
    DeliveryEventRecord,
    validation_error_detail,
)
from tcip_mcp.pipelines.resolution import (  # noqa: E402
    DELIVERY_EVENTS_STORE,
    delivery_events_scope,
)
from tcip_store.binding import bind_default  # noqa: E402


_ACKNOWLEDGEMENT_KEYS = ("acknowledged_by", "acknowledgement_reason")


def _check_record(record: object) -> tuple[bool, str]:
    """Whether ``record`` validates, and the one outcome line naming it."""
    try:
        DeliveryEventRecord.model_validate(record)
    except ValidationError as exc:
        return False, (
            f"refused, {validation_error_detail(exc)}; the disclosure this record lacks was never "
            "computed for this delivery and cannot be reconstructed from it; re-deliver, or "
            "remove the record by hand"
        )
    return True, "validates, unchanged"


def _ack_gap(record: dict) -> bool:
    """Whether ``record`` is missing both acknowledgement keys outright: a delivery from before
    this pair existed, for which ``null`` on both is the true, derivable value."""
    return not any(key in record for key in _ACKNOWLEDGEMENT_KEYS)


def _forward_acknowledgement(record: dict) -> dict:
    """``record`` with ``acknowledged_by``/``acknowledgement_reason`` forwarded to ``null``.
    Call only when :func:`_ack_gap` is true."""
    return {**record, "acknowledged_by": None, "acknowledgement_reason": None}


def _plants_outside_raster_gap(record: dict) -> bool:
    """Whether ``record`` carries a registry (never a canopy-segment) ``plant_mapping``
    disclosure missing ``plants_outside_raster``: the one key that pair predates."""
    pm = record.get("plant_mapping")
    return (
        isinstance(pm, dict) and "plant_registry" in pm and "canopy_segments" not in pm
        and "plants_outside_raster" not in pm
    )


class _RegistryUnrecoverable(Exception):
    """A registry disclosure's ``plants_outside_raster`` gap is real but cannot be recomputed:
    the registry it names no longer loads, or has moved, since this delivery read it."""


def _forward_plants_outside_raster(record: dict, project_root: Path) -> dict:
    """``record`` with its registry disclosure's ``plants_outside_raster`` forwarded, recomputed
    through :func:`~tcip_mcp.pipelines.postprocessing.orthomosaic_mapping.plants_in_frame` from
    the event's own registry (name and digest) and raster identity (width, height and
    geotransform). Call only when :func:`_plants_outside_raster_gap` is true.

    Raises :class:`_RegistryUnrecoverable`, naming why, when the named registry no longer loads,
    when its digest has moved, or when the event's own raster identity carries no geotransform or
    dimensions to project through: the value cannot be reconstructed from a registry, or an
    identity, that is not the one this delivery read.
    """
    from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import (
        GeoTransform, OrthomosaicGeoreference, plants_in_frame,
    )
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        load_registry, read_plant_csvs, registry_csv_entries,
    )

    pm = record["plant_mapping"]
    registry_ref = pm.get("plant_registry") or {}
    name = registry_ref.get("name")
    stored_registry = load_registry(project_root, name) if name else None
    if stored_registry is None:
        raise _RegistryUnrecoverable(
            f"plant registry {name!r} named by this event's plant_mapping no longer loads; "
            "plants_outside_raster cannot be recomputed from a registry that is not the one "
            "this delivery read"
        )
    if stored_registry.get("digest") != registry_ref.get("digest"):
        raise _RegistryUnrecoverable(
            f"plant registry {name!r} named by this event's plant_mapping has moved (delivered "
            f"against digest {registry_ref.get('digest')!r}, now "
            f"{stored_registry.get('digest')!r}); plants_outside_raster cannot be recomputed "
            "from a registry that is not the one this delivery read"
        )
    identity = pm.get("raster_identity") or {}
    geotransform = identity.get("geotransform")
    width, height = identity.get("width"), identity.get("height")
    if not geotransform or width is None or height is None:
        raise _RegistryUnrecoverable(
            "this event's raster_identity carries no geotransform or no dimensions; "
            "plants_outside_raster cannot be recomputed without them"
        )
    plants = read_plant_csvs(Path(e["path"]) for e in registry_csv_entries(stored_registry))
    georef = OrthomosaicGeoreference(GeoTransform(**geotransform))
    _in_frame, outside = plants_in_frame(plants, georef, width=int(width), height=int(height))
    names = sorted(p.plot_name for p in outside)
    return {**record, "plant_mapping": {**pm, "plants_outside_raster": names}}


def _write_forward(
    record: object, project_root: Path,
) -> tuple[dict | None, list[tuple[str, str]], str | None]:
    """Every applicable write-forward composed onto ``record``: ``(forwarded, changes, refusal)``.

    ``changes`` names each applied gap as ``(gap_name, applied_phrase)``, in application order;
    both write-forwards apply together when both gaps are present, since a record can predate
    both. ``forwarded`` is ``None`` when nothing applies or the composed result still does not
    validate; ``refusal`` names why a derivable-in-principle gap (a stored registry disclosure's
    ``plants_outside_raster``) could not be filled.
    """
    if not isinstance(record, dict):
        return None, [], None
    working = record
    changes: list[tuple[str, str]] = []
    if _ack_gap(working):
        working = _forward_acknowledgement(working)
        changes.append((
            "acknowledged_by/acknowledgement_reason",
            "acknowledged_by/acknowledgement_reason to null",
        ))
    if _plants_outside_raster_gap(working):
        try:
            working = _forward_plants_outside_raster(working, project_root)
        except _RegistryUnrecoverable as exc:
            return None, [], str(exc)
        changes.append(("plants_outside_raster", "plants_outside_raster"))
    if not changes:
        return None, [], None
    try:
        DeliveryEventRecord.model_validate(working)
    except ValidationError:
        return None, [], None
    return working, changes, None


def check_root(root: Path, *, plan: bool = False) -> tuple[list[str], bool]:
    """Every outcome line for ``root``'s stored ``delivery_events`` records, and whether any of
    them was refused (as stored, in ``--plan`` mode; after any write-forward, otherwise).

    A ``root`` holding no ``.tcip`` directory is refused by name rather than reported as a project
    with nothing stored: a mistyped path and a real, empty project would otherwise print the same
    "nothing stored" line, and the stated root is a claim the data must positively carry.
    """
    if not (root / ".tcip").is_dir():
        return ["refused, no .tcip directory found; not a project root"], True
    scope = delivery_events_scope(root)
    outcomes: list[str] = []
    refused = False
    for key in ts.keys(DELIVERY_EVENTS_STORE, str(scope)):
        event_id = key.parts[0] if key.parts else "<unknown event>"
        stored = ts.read(key)
        ok, message = _check_record(stored)
        if ok:
            outcomes.append(f"{event_id}: {message}")
            continue
        forwarded, changes, refusal = _write_forward(stored, root)
        if refusal is not None:
            outcomes.append(f"{event_id}: refused, {message} {refusal}")
            refused = True
            continue
        if forwarded is None:
            outcomes.append(f"{event_id}: {message}")
            refused = True
            continue
        names = ", ".join(name for name, _applied in changes)
        if plan:
            outcomes.append(
                f"{event_id}: refused as stored (missing {names}); would write-forward and "
                "validate"
            )
            refused = True
            continue
        with ts.transaction(key) as txn:
            current = txn.read(key, default=None)
            current_forwarded, current_changes, current_refusal = _write_forward(current, root)
            if current_refusal is not None or current_forwarded is None:
                outcomes.append(f"{event_id}: {message}")
                refused = True
                continue
            txn.write(key, current_forwarded)
        applied = ", ".join(phrase for _name, phrase in changes)
        outcomes.append(f"{event_id}: write-forwarded {applied}, validates")
    return outcomes, refused


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument(
        "--plan", action="store_true",
        help="preview only; a fixable record's write-forward is named, not applied")
    args = ap.parse_args()

    bind_default()

    refused_any = False
    for root in args.roots:
        root = root.resolve()
        outcomes, refused = check_root(root, plan=args.plan)
        if refused:
            refused_any = True
        if not outcomes:
            outcomes = ["nothing stored"]
        for line in outcomes:
            print(f"{root}: {line}")

    return 2 if refused_any else 0


if __name__ == "__main__":
    sys.exit(main())
