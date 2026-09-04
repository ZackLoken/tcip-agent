"""Rewrite one project's stored plant-mapping records from the old ``plant_csvs`` field to the
new ``plant_registry`` reference, registering the CSV files the old field named under a name the
operator states, and add ``supersedes: null`` (a new field conformed in the same pass so a stored
record moves once, not twice).

A record already carrying ``plant_registry`` (already conformed, or written by the current
door) is reported unchanged, never rewritten twice. Every record this run conforms under one
project shares one registry, since these are pre-registry records that predate the distinction;
an operator whose project's mapping records name genuinely different CSV sets runs this once per
set, naming a different registry each time and passing ``--mapping-name`` to target only the
records built from that set.

    python scripts/conform_plant_mapping_records.py <project_root> <registry_name> \
        --crop <crop> --site <site> [--mapping-name <name>] [--plan]

``--crop``/``--site`` are the expert's facts a registration always carries and are never guessed
here; the operator states them. ``registry_name`` must match
``tcip_store.layout_claims.NAME_SEGMENT`` (lowercase letters, digits, single hyphens), the same
rule ``register_plant_registry`` holds callers to.

Rewriting ``plant_csvs`` to ``plant_registry`` changes the record's own digest (the field this
reader hashes moved), so a delivery event minted before this conform ran cites a digest no stored
record hashes to any more, once it runs. ``--plan`` and a real run both print, at the end, every
delivery event under the project that cites a record this run is about to conform, read through
the one delivery-events reader (``resolution.read_delivery_events``) the panel route and
``plant_mapping._citing_delivery_event_ids`` also read through; the only remedy for one is a fresh
delivery under the rebuilt mapping, or ``supersede_delivery`` naming it withdrawn.

``--plan`` previews every outcome and writes nothing at all, the registry included: it parses the
targeted records' own CSVs (through ``plant_mapping.parse_plant_registry_csvs``, read-only) to
compute the digest a real run would register, without registering it.

Exit codes: 0 when every targeted record conforms (or none is found to conform); 2 when a stored
record names a CSV path that no longer exists (nothing to register from), the named root is not a
project (no ``.tcip`` directory), or ``registry_name`` is not a legal name.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-mcp" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-store" / "src"))

import tcip_store as ts  # noqa: E402
from tcip_store.binding import bind_default  # noqa: E402
from tcip_store.layout_claims import NAME_SEGMENT  # noqa: E402

from tcip_mcp.pipelines.postprocessing import plant_mapping  # noqa: E402


def _stored_records(project_root: Path) -> list[tuple[ts.Key, dict]]:
    """Every legally-named plant-mapping record under this project, in the state as stored."""
    root = str(project_root / ".tcip" / "state")
    keys = ts.keys(plant_mapping.PLANT_MAPPING_STORE, root)
    return [(key, ts.read(key)) for key in keys if NAME_SEGMENT.fullmatch(key.parts[-1])]


def plan_root(
    project_root: Path, registry_name: str, *, crop: str, site: str,
    mapping_name: str | None = None,
) -> tuple[list[str], list[dict]]:
    """Every outcome line for this project, and the writes ``apply_writes`` would then make.

    Writes nothing: the registry's own digest is computed by parsing the targeted CSVs
    read-only (``plant_mapping.parse_plant_registry_csvs``), never by registering them.
    """
    outcomes: list[str] = []
    if not NAME_SEGMENT.fullmatch(registry_name):
        outcomes.append(
            f"refused: registry name {registry_name!r} is not lowercase letters, digits and "
            f"single hyphens ({NAME_SEGMENT.pattern})")
        return outcomes, []

    records = _stored_records(project_root)
    if mapping_name is not None:
        records = [(key, record) for key, record in records if key.parts[-1] == mapping_name]

    to_conform: list[tuple[ts.Key, dict]] = []
    csv_paths: list[Path] = []
    seen_paths: set[str] = set()
    for key, record in records:
        name = key.parts[-1]
        if not isinstance(record, dict) or "plant_csvs" not in record:
            outcomes.append(f"{name}: already conformed (or not a plant_csvs record), unchanged")
            continue
        to_conform.append((key, record))
        for entry in record["plant_csvs"]:
            if entry["path"] not in seen_paths:
                seen_paths.add(entry["path"])
                csv_paths.append(Path(entry["path"]))

    if not to_conform:
        return outcomes, []

    missing = [str(p) for p in csv_paths if not p.is_file()]
    if missing:
        outcomes.append(
            f"refused: {missing} named by a stored record no longer exist; cannot register "
            f"{registry_name!r} without them")
        return outcomes, []

    try:
        _csvs_meta, digest, _n_plants = plant_mapping.parse_plant_registry_csvs(csv_paths)
    except plant_mapping.NoGeoreferencedPlantsRefusal as exc:
        outcomes.append(f"refused: {exc}")
        return outcomes, []

    registry_ref = {"name": registry_name, "digest": digest}

    writes: list[dict] = []
    for key, record in to_conform:
        name = key.parts[-1]
        old_digest = plant_mapping.record_digest(record)
        new_record = {field: value for field, value in record.items() if field != "plant_csvs"}
        new_record["plant_registry"] = registry_ref
        new_record.setdefault("supersedes", None)
        writes.append({
            "key": key, "record": new_record, "name": name, "old_digest": old_digest,
            "registry_name": registry_name, "csv_paths": csv_paths, "crop": crop, "site": site,
        })
        outcomes.append(f"{name}: conformed, plant_registry={registry_name!r}")

    return outcomes, writes


def apply_writes(project_root: Path, writes: list[dict]) -> None:
    """Commit a ``plan_root`` plan for real: register the CSVs (minting ``registered_at`` now),
    then write each targeted record and a fresh receipt naming its own new digest, through the
    same emitter ``persist_mapping`` uses."""
    from tcip_mcp.audit import record_event_or_raise

    if not writes:
        return
    first = writes[0]
    registry = plant_mapping.register_plant_registry_record(
        project_root, first["registry_name"], first["csv_paths"],
        crop=first["crop"], site=first["site"], registered_by="conform_plant_mapping_records")
    registry_ref = {"name": first["registry_name"], "digest": registry["digest"]}

    for write in writes:
        key, name = write["key"], write["name"]
        record = {**write["record"], "plant_registry": registry_ref}
        ts.replace(key, record)
        digest = plant_mapping.record_digest(record)
        record_event_or_raise(
            "plant_mapping_built",
            {
                "name": name, "project_root": str(project_root),
                "dataset_root": record.get("dataset_root"), "built_at": record.get("built_at"),
                "record_sha256": digest,
            },
            scope=project_root,
        )


def _stranded_citation_lines(project_root: Path, writes: list[dict]) -> list[str]:
    """One outcome line per delivery event that cites a record this run is about to conform: once
    it runs, that event's own digest hashes to nothing stored, since the rewrite moves the
    record's digest. Read-only: safe to call whether or not the run actually writes."""
    from tcip_mcp.pipelines.resolution import read_delivery_events

    if not writes:
        return []
    events = read_delivery_events(project_root)
    lines: list[str] = []
    for write in writes:
        name, old_digest = write["name"], write["old_digest"]
        citing = sorted(
            record["event_id"] for record in events
            if isinstance(record.get("plant_mapping"), dict)
            and record["plant_mapping"].get("name") == name
            and record["plant_mapping"].get("record_sha256") == old_digest
        )
        if citing:
            lines.append(
                f"{name}: delivery event(s) {citing} cite record {old_digest} which no stored "
                "record will hash to once this conform runs; re-deliver under the rebuilt "
                "mapping, or supersede_delivery them as withdrawn")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("project_root", type=Path)
    ap.add_argument("registry_name")
    ap.add_argument("--crop", required=True, help="the expert's fact; never guessed")
    ap.add_argument("--site", required=True, help="the expert's fact; never guessed")
    ap.add_argument("--mapping-name", default=None,
                    help="conform only this project's mapping of that name")
    ap.add_argument("--plan", action="store_true", help="preview only; writes nothing")
    args = ap.parse_args()

    bind_default()

    root = args.project_root.resolve()
    if not (root / ".tcip").is_dir():
        print(f"{root}: refused, no .tcip directory found; not a project root")
        return 2

    outcomes, writes = plan_root(
        root, args.registry_name, crop=args.crop, site=args.site,
        mapping_name=args.mapping_name)
    if not outcomes:
        outcomes = ["nothing to conform"]
    for line in outcomes:
        print(f"{root}: {line}")

    refused = any(line.startswith("refused") for line in outcomes)
    if refused:
        return 2
    if not args.plan:
        apply_writes(root, writes)
    for line in _stranded_citation_lines(root, writes):
        print(f"{root}: {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
