"""Conform a dataset's stored ``view_coverage`` records to the current ``CoverageRecord`` shape.

A deliberate one-off operator script, per this repo's no-backward-compatibility boundary
(CLAUDE.md): ``CoverageViewing`` now declares a fixed five-key shape with a structured
``stats_source`` and a positional ``display_bounds`` and no longer carries a
``working_scale_bar`` (the bar is derived server-side from the label file, never stored on this
record), and ``CoverageRecord`` carries ``cells_seen_at_scale`` in place of a bare ``cells_swept``
name list. A record still carrying any of the old shapes needs conforming before
``get_coverage``/``post_coverage`` (which refuse a record that does not validate) will read it
again. It never runs as part of any runtime path.

For every bucket and image in a dataset's ``view_coverage`` record that does not already
validate as ``CoverageRecord``, this maps the old string forms of ``stats_source`` (``"none"``,
``"dtype_full_scale"`` and ``"served_array"`` to the read alone; ``overview(scale=...)`` and
``sampled(seed=..., pixel_fraction=...)`` to ``"overview"`` and ``"window_sample"`` with their
numbers) and of ``display_bounds`` (``lo,hi;lo,hi`` to pairs) to the new values, carries
``bands``/``stretch``/``base_served_size`` through unchanged, drops a stored ``working_scale_bar``
key from ``viewing`` outright (the stored bar was always the latest session's, never a bound any
particular cell was actually swept under), and drops a stored ``cells_swept`` name list from the
record, reporting its count: no scale can be anchored to a cell swept under a bar the record no
longer holds, so those names are never fabricated a scale they were never recorded against. A
record that already carries a ``cells_seen_at_scale`` mapping (a half-migrated write already
under the new shape, needing only its ``viewing`` conformed) carries that mapping through
unchanged rather than being blanked back to empty, and a ``viewing`` whose ``stats_source`` is
already the structured mapping (written between the structured source landing and the
``cells_seen_at_scale`` rename, beside an old ``working_scale_bar`` and ``cells_swept``) carries
it through as it stands.

    python scripts/conform_view_coverage_viewing.py <dataset_root> [<dataset_root> ...]
    python scripts/conform_view_coverage_viewing.py --plan <dataset_root>

Every image in a root is validated before anything is written: a record that neither validates
as-is nor reshapes into something that does refuses the whole root by image name and leaves it
untouched, so a root is never left half-conformed. Every other root named on the command line
still runs. The write, when there is one, happens inside the same ``tcip_store.transaction`` lock
the route itself takes on the record's key, so it can never race a concurrent post.

Exit codes: 0 conformed (or nothing to conform) for every root named; 2 if any root was refused.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-mcp" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-store" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-web" / "src"))

import tcip_store as ts  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from tcip_mcp.dataset_layout import view_coverage_key  # noqa: E402
from tcip_store.binding import bind_default  # noqa: E402
from tcip_web.routes._coverage_models import CoverageRecord, StatsSource  # noqa: E402

# working_scale_bar stays a recognized key so an old viewing carrying it does not refuse as an
# unknown key; _reshaped_viewing below drops it unconditionally rather than mapping it forward.
_VIEWING_KEYS = frozenset(
    {"bands", "stretch", "base_served_size", "working_scale_bar", "stats_source", "display_bounds"})
_STATS_SOURCE_LITERALS = ("none", "dtype_full_scale", "served_array")
_OVERVIEW_RE = re.compile(r"^overview\(scale=([^)]+)\)$")
_SAMPLED_RE = re.compile(r"^sampled\(seed=(-?\d+), pixel_fraction=([^)]+)\)$")


def _parse_stats_source(value: object) -> dict | None:
    """The old ``stats_source`` shape (a bare string) mapped to the new structured one, or the
    structured one carried through when the record already holds it (a record the route wrote
    after the structured ``stats_source`` landed but before ``cells_seen_at_scale`` did, whose
    ``viewing`` still needs its other keys conformed).

    Raises ``ValueError`` naming ``value`` when it is neither a mapping ``StatsSource`` accepts,
    nor one of the three flat literals, nor either formatted string shape the route ever emitted.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        try:
            return StatsSource.model_validate(value).model_dump()
        except ValidationError as exc:
            raise ValueError(f"stats_source is a mapping StatsSource does not accept: {exc}") from exc
    if not isinstance(value, str):
        raise ValueError(f"stats_source is not a string: {value!r}")
    if value in _STATS_SOURCE_LITERALS:
        return {"read": value, "seed": None, "pixel_fraction": None, "overview_scale": None}
    overview = _OVERVIEW_RE.match(value)
    if overview:
        return {"read": "overview", "seed": None, "pixel_fraction": None,
                "overview_scale": float(overview.group(1))}
    sampled = _SAMPLED_RE.match(value)
    if sampled:
        return {"read": "window_sample", "seed": int(sampled.group(1)),
                "pixel_fraction": float(sampled.group(2)), "overview_scale": None}
    raise ValueError(f"stats_source does not match a known old shape: {value!r}")


def _parse_display_bounds(value: object) -> list[list[float]] | None:
    """The old ``lo,hi;lo,hi`` joined-string shape mapped to a list of ``[lo, hi]`` pairs."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"display_bounds is not a string: {value!r}")
    pairs = []
    for chunk in value.split(";"):
        parts = chunk.split(",")
        if len(parts) != 2:
            raise ValueError(f"display_bounds pair does not parse: {chunk!r} in {value!r}")
        try:
            pairs.append([float(parts[0]), float(parts[1])])
        except ValueError as exc:
            raise ValueError(f"display_bounds pair does not parse: {chunk!r} in {value!r}") from exc
    return pairs


def _reshaped_viewing(viewing: dict) -> dict:
    """The current five-key ``CoverageViewing`` shape mapped from ``viewing``'s old forms.

    ``bands``, ``stretch`` and ``base_served_size`` already carry the current shape wherever the
    route ever wrote them, so they pass through unchanged (``None`` where the key was never
    present); ``stats_source`` and ``display_bounds`` held a different shape before this change;
    a stored ``working_scale_bar`` (any shape) is dropped outright, never mapped forward, since
    the bar is now derived server-side and a value echoed back from the browser recorded
    nothing. Raises ``ValueError`` naming any key outside the recognized ones rather than
    silently dropping it: a stray key never reaches ``.get()``.
    """
    unknown = sorted(set(viewing) - _VIEWING_KEYS)
    if unknown:
        raise ValueError(f"viewing carries keys the current shape does not declare: {unknown}")
    return {
        "bands": viewing.get("bands"),
        "stretch": viewing.get("stretch"),
        "base_served_size": viewing.get("base_served_size"),
        "stats_source": _parse_stats_source(viewing.get("stats_source")),
        "display_bounds": _parse_display_bounds(viewing.get("display_bounds")),
    }


def _conform_record(record: dict, *, plan: bool) -> tuple[str, dict | None]:
    """One outcome line and, when the record needs conforming, its replacement.

    The replacement is ``None`` (and the outcome says so) when the stored record already
    validates as ``CoverageRecord``: a rerun against an already-conformed dataset is a no-op.
    Raises ``ValueError`` naming what could not be parsed or still does not validate once
    reshaped.
    """
    try:
        CoverageRecord.model_validate(record)
        return "record already validates, unchanged", None
    except ValidationError:
        pass
    viewing = record.get("viewing")
    if not isinstance(viewing, dict):
        raise ValueError(f"viewing is not a mapping: {viewing!r}")
    reshaped_viewing = _reshaped_viewing(viewing)
    old_swept = record.get("cells_swept")
    if old_swept is not None and not isinstance(old_swept, list):
        raise ValueError(f"cells_swept is not a list: {old_swept!r}")
    dropped_count = len(old_swept) if isinstance(old_swept, list) else 0
    existing_seen = record.get("cells_seen_at_scale")
    if existing_seen is not None and not isinstance(existing_seen, dict):
        raise ValueError(f"cells_seen_at_scale is not a mapping: {existing_seen!r}")
    reshaped = {
        "grid": record.get("grid"),
        "cells_served_at_native": record.get("cells_served_at_native") or [],
        "cells_seen_at_scale": existing_seen or {},
        "viewing": reshaped_viewing,
        "updated_at": record.get("updated_at"),
    }
    try:
        CoverageRecord.model_validate(reshaped)
    except ValidationError as exc:
        raise ValueError(f"reshaped record still does not validate: {exc}") from exc
    verb = "would conform" if plan else "conformed"
    if dropped_count:
        name = "swept cell name" if dropped_count == 1 else "swept cell names"
        message = (
            f"{verb} viewing and dropped {dropped_count} old {name}: no scale can be anchored "
            "to a cell swept under a bar the record no longer holds"
        )
    else:
        message = f"{verb} viewing"
    return message, reshaped


def _conform_store(store: dict, *, plan: bool) -> tuple[list[str], dict, bool]:
    """Every outcome line, the store with every conformable record replaced, and whether any
    record was refused (in which case the conformed store must never be written)."""
    outcomes: list[str] = []
    refused = False
    new_store: dict = {}
    for bucket, records in store.items():
        if not isinstance(records, dict):
            new_store[bucket] = records
            continue
        new_records = dict(records)
        for image_name, record in records.items():
            label = f"{bucket}/{image_name}"
            if not isinstance(record, dict):
                outcomes.append(f"{label}: refused, record is not a mapping")
                refused = True
                continue
            try:
                message, replacement = _conform_record(record, plan=plan)
            except ValueError as exc:
                outcomes.append(f"{label}: refused, {exc}")
                refused = True
                continue
            outcomes.append(f"{label}: {message}")
            if replacement is not None:
                new_records[image_name] = replacement
        new_store[bucket] = new_records
    return outcomes, new_store, refused


def conform_root(root: Path, *, plan: bool) -> tuple[list[str], bool]:
    """Every outcome line for ``root``, and whether it was refused.

    In plan mode this reads once, outside a transaction, since nothing is written. Off plan mode
    it reads and writes inside one ``tcip_store.transaction`` on the record's key, the same lock
    ``post_coverage`` itself takes, so the write can never race a concurrent post.
    """
    key = view_coverage_key(root)
    if plan:
        store = ts.read(key, default={})
        if not isinstance(store, dict):
            store = {}
        outcomes, _new_store, refused = _conform_store(store, plan=True)
        return outcomes, refused

    with ts.transaction(key) as txn:
        store = txn.read(key, default={})
        if not isinstance(store, dict):
            store = {}
        outcomes, new_store, refused = _conform_store(store, plan=False)
        if refused:
            return outcomes, True
        txn.write(key, new_store)
    return outcomes, False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--plan", action="store_true", help="show what would change, write nothing")
    args = ap.parse_args()

    bind_default()

    refused_any = False
    for root in args.roots:
        root = root.resolve()
        try:
            outcomes, refused = conform_root(root, plan=args.plan)
        except ts.StoreError as exc:
            print(f"{root}: refused, {exc}")
            refused_any = True
            continue

        if refused:
            refused_any = True
            outcomes = outcomes + [
                "refused: at least one image's viewing failed to parse or validate, so nothing "
                "is written for this root"
            ]
        if not outcomes:
            outcomes = ["nothing to conform"]
        for line in outcomes:
            print(f"{root}: {line}")

    return 2 if refused_any else 0


if __name__ == "__main__":
    sys.exit(main())
