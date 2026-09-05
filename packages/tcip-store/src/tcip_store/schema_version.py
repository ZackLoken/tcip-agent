"""The version-field accept rule every frozen store's reader applies.

Absence in an existing document reads as version 1, the frozen default (``owner-decisions.md``
Part 17 Q1: the field stays lazy, so nothing rewrites an existing document just to stamp it).
A present ``1`` accepts. A present version above ``descriptor.schema_version`` (the ceiling
this reader knows) refuses by name, naming the document's number, the ceiling and the store;
so does a present version that is not a plain integer. The rule applies only to a frozen store
whose documents can carry the field at all: an unstable-by-design store's shape is still
moving, so no ceiling exists to enforce, and a cannot-carry store's documents (raw bytes, a
single text primitive, a heading-parsed markdown file) have no field to inspect in the first
place.
"""

from __future__ import annotations

from typing import Any

from tcip_store.errors import SchemaVersionRefused
from tcip_store.registry import StoreDescriptor

__all__ = ["check_schema_version"]


def check_schema_version(descriptor: StoreDescriptor, doc: Any) -> None:
    """Refuse ``doc`` when its ``schema_version`` is one this store's reader does not know.

    A no-op for a store that is not frozen, for a store whose documents cannot carry the field,
    for a document that is not a mapping (so could never carry the key), and for a mapping with
    no ``schema_version`` key (absence is the frozen default, version 1).
    """
    if not descriptor.frozen or descriptor.cannot_carry_field:
        return
    if not isinstance(doc, dict) or "schema_version" not in doc:
        return
    version = doc["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise SchemaVersionRefused(
            f"{descriptor.name} document carries schema_version={version!r}, which is not a "
            "version number this reader can compare against its ceiling"
        )
    if version > descriptor.schema_version:
        raise SchemaVersionRefused(
            f"{descriptor.name} document is schema_version {version}, above the "
            f"{descriptor.schema_version} this reader knows: a newer writer produced it than "
            "this code understands, or it predates this store's version-1 reset and still "
            "carries a stale 2 that scripts/conform_schema_version_reset.py strips"
        )
