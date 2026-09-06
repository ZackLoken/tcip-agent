#!/usr/bin/env python
"""Census of prediction buckets published more than once before the live-bucket refusal.

Read-only. For each project root given, every prediction bucket the project's own records
name (``store_catalogue.project_roots``, the same enumeration ``tcip adopt-store`` and ``tcip
export-store`` walk: the buckets under each registered dataset's ``predictions/`` tree and
each experiment's lineage bucket) is checked two ways. A stamp whose ``image_filenames``
names fewer stems than the bucket holds documents for is a bucket a later run published over
an earlier one, leaving documents the stamp does not vouch for. A validation record whose
``covered_buckets`` names such a bucket is a claim sealed over a mixed-run digest; for an
``operating_point`` claim the sealed digest is compared with the bucket's content now, so the
line says whether the sealed content is still what is on disk. Nothing is repaired; every
finding is printed with the path and the counts.

A project that never registered its own tree as a dataset has no record naming its
``predictions/`` buckets, so they are outside this census; the script prints a note when such
a tree exists. Reading a stamp or a validation log goes through the storage seam, which binds
the process's default backend and opens nothing that is not already a database.

A bucket whose stamp decodes but carries no ``image_filenames`` mapping cannot be checked
against its documents at all (``image_filenames`` is a producer extension,
``STAMP_EXTENSION_KEYS`` in ``pipelines/resolution.py``, not one of the stamp constructor's own
keys): it is reported UNJUDGEABLE rather than read as clean. A clean exit therefore means no
finding among the buckets this census could judge, not that every walked bucket answered clean.

Exit codes: 0 with no finding among the judgeable buckets, 1 with at least one finding (a
double-publish, a named-without-document mismatch, a mixed-run claim, or an unjudgeable
bucket), 2 when a stamp or a validation log could not be read, or when a named root is not a
project (no ``.tcip`` directory); a read refusal is printed and the census continues over the
remaining roots.

    python tools/census_double_published_buckets.py <project_root> [<project_root> ...]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from tcip_store import StoreError
from tcip_store.layout_claims import PREDICTION_BUCKET


@dataclass
class BucketCensus:
    """What one bucket's stamp says against what the bucket holds."""

    bucket: Path
    document_stems: set[str]
    named_stems: set[str] | None
    stamp_read_error: str | None = None
    unjudgeable: bool = False

    @property
    def unnamed(self) -> set[str]:
        """Stems holding a document the stamp does not name: a later publish over an earlier one."""
        if self.named_stems is None:
            return set()
        return self.document_stems - self.named_stems

    @property
    def named_without_document(self) -> set[str]:
        if self.named_stems is None:
            return set()
        return self.named_stems - self.document_stems


@dataclass
class ClaimCensus:
    """One validation row's coverage of a mixed-run bucket."""

    experiment_id: str
    document: str
    bucket: Path
    sealed_digest: str
    content_matches: bool | None


@dataclass
class ProjectCensus:
    project_root: Path
    buckets: list[BucketCensus] = field(default_factory=list)
    claims: list[ClaimCensus] = field(default_factory=list)
    read_errors: list[str] = field(default_factory=list)
    unregistered_tree: bool = False

    @property
    def mixed(self) -> list[BucketCensus]:
        return [b for b in self.buckets if b.unnamed]


def census_bucket(bucket: Path) -> BucketCensus:
    """Read one bucket's documents and its ``operating_point`` stamp through the platform's own
    enumeration and reader."""
    import tcip_store

    from tcip_mcp.pipelines.resolution import sidecar_key
    from tcip_mcp.prediction_buckets import bucket_stems

    stems = bucket_stems(bucket)
    try:
        stamp = tcip_store.read(sidecar_key(bucket, "operating_point"), default=None)
    except StoreError as exc:
        return BucketCensus(bucket, stems, None, stamp_read_error=str(exc))
    if not isinstance(stamp, dict):
        return BucketCensus(bucket, stems, None)
    names = stamp.get("image_filenames")
    if not isinstance(names, dict):
        return BucketCensus(bucket, stems, None, unjudgeable=True)
    return BucketCensus(bucket, stems, set(names))


def census_project(project_root: Path) -> ProjectCensus:
    """Every bucket and every validation row the project's own records name, judged."""
    from tcip_mcp import dataset_layout
    from tcip_mcp.experiments import experiment_ids_with_status, read_validations
    from tcip_mcp.prediction_buckets import bucket_content_digest
    from tcip_mcp.store_catalogue import project_roots

    result = ProjectCensus(project_root=project_root)
    roots = project_roots(project_root)
    for path, layout in roots:
        if layout != PREDICTION_BUCKET:
            continue
        bucket = Path(path)
        if not bucket.is_dir():
            continue
        entry = census_bucket(bucket)
        if entry.stamp_read_error is not None:
            result.read_errors.append(f"{bucket}: {entry.stamp_read_error}")
        if entry.document_stems or entry.named_stems:
            result.buckets.append(entry)

    if dataset_layout.prediction_root(project_root).is_dir():
        registered = {
            Path(path).resolve() for path, layout in roots if layout == PREDICTION_BUCKET
        }
        own = {p.resolve() for p in dataset_layout.prediction_bucket_dirs(project_root)}
        result.unregistered_tree = bool(own) and not own & registered

    mixed_by_path = {b.bucket.resolve(): b for b in result.mixed}
    for experiment_id in experiment_ids_with_status(project_root):
        try:
            rows = read_validations(experiment_id, root=project_root)
        except StoreError as exc:
            result.read_errors.append(f"{experiment_id} validations: {exc}")
            continue
        for row in rows:
            covered = row.get("covered_buckets")
            dataset_root = row.get("dataset_root")
            if not isinstance(covered, dict) or not isinstance(dataset_root, str):
                continue
            for key, sealed in covered.items():
                bucket = (Path(dataset_root) / key).resolve()
                if bucket not in mixed_by_path:
                    continue
                document = str(row.get("document"))
                matches: bool | None = None
                if document == "operating_point" and isinstance(sealed, str):
                    matches = bucket_content_digest(bucket) == sealed
                result.claims.append(ClaimCensus(
                    experiment_id=experiment_id, document=document, bucket=bucket,
                    sealed_digest=str(sealed), content_matches=matches,
                ))
    return result


def render(census: ProjectCensus) -> list[str]:
    """The census as printed lines, one per finding, plus the notes."""
    lines = [f"project {census.project_root}: {len(census.buckets)} bucket(s) with documents or a stamp"]
    if census.unregistered_tree:
        lines.append(
            f"  NOTE {census.project_root} holds a predictions/ tree but registers no dataset, so "
            "no record names those buckets and this census does not walk them"
        )
    for entry in census.buckets:
        if entry.unjudgeable:
            lines.append(
                f"  UNJUDGEABLE {entry.bucket}: {len(entry.document_stems)} document(s), the "
                "stamp records no image_filenames map"
            )
        if entry.unnamed:
            lines.append(
                f"  DOUBLE-PUBLISH {entry.bucket}: {len(entry.document_stems)} document(s), the "
                f"stamp names {len(entry.named_stems or ())} stem(s); unnamed: "
                f"{', '.join(sorted(entry.unnamed))}"
            )
        if entry.named_without_document:
            lines.append(
                f"  NAMED-WITHOUT-DOCUMENT {entry.bucket}: the stamp names "
                f"{', '.join(sorted(entry.named_without_document))} with no document"
            )
    for claim in census.claims:
        state = (
            "digest not compared (the claim digests imagery, not documents)"
            if claim.content_matches is None
            else ("sealed content is still on disk" if claim.content_matches
                  else "content changed since the claim was sealed")
        )
        lines.append(
            f"  MIXED-RUN-CLAIM {claim.experiment_id} ({claim.document}) covers "
            f"{claim.bucket} at digest {claim.sealed_digest}: {state}"
        )
    for error in census.read_errors:
        lines.append(f"  READ-REFUSED {error}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project_root", type=Path, nargs="+")
    args = ap.parse_args(argv)

    from tcip_store.binding import bind_default

    bind_default()

    code = 0
    for root in args.project_root:
        project_root = root.resolve()
        if not (project_root / ".tcip").is_dir():
            print(f"{project_root}: not a project (no .tcip directory)")
            code = max(code, 2)
            continue
        census = census_project(project_root)
        for line in render(census):
            print(line)
        findings = len(census.mixed) + len(census.claims) + sum(
            1 for b in census.buckets if b.named_without_document
        ) + sum(1 for b in census.buckets if b.unjudgeable)
        if census.read_errors:
            code = max(code, 2)
        elif findings:
            code = max(code, 1)
    return code


if __name__ == "__main__":
    sys.exit(main())
