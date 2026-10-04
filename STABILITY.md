# Stability

What an adopter can build against today, independent of the 0.x version number (see
VERSIONING.md); everything not named here is free to change in any release while the shared
version stays below `1.0.0`.

## Persisted formats

No persisted record shape is frozen while the version stays below `1.0.0`: a record is the shape
its one producer writes and its one reader reads, and a change to it regenerates or discards the
records already written. A project's records and logs live in one SQLite database per root
(`<root>/.tcip/store.db`); `tcip dump-store` writes them out as files a person can read.

## The MCP tool surface

`python scripts/list_tools.py` prints the live tool registry; treat its output, not a number in
this or any document, as the current list. Tools may be added in any release. A tool's removal or
rename is announced in CHANGELOG.md. Never state a tool count in shipped prose.

## `tcip-annotation`'s public API

`tcip-annotation` is the one package meant for standalone use outside this platform. Its public
API is exactly what `packages/tcip-annotation/src/tcip_annotation/__init__.py`'s `__all__`
declares: the annotation state types, the canonical per-image JSON read/write path, the
multi-format load/save dispatch, COCO interop, IoU matching, the mask-to-polygon-rings extractor,
the SAM wrapper's `auto_mask` and grid-cell helpers, and the `AnnotationEngine`/`ReviewEngine`
pair. `__all__` is the only boundary.

## The general-capability scripts

Every script listed under `scripts/README.md`'s General capabilities section is written to be
reached for again, with its own docstring and `--help` as the usage contract. A script under
that document's Pilot/incident-bound or One-off conforms sections carries no such commitment and
can be rewritten or removed without notice.

## Not stable

- The web routes (`packages/tcip-web/src/tcip_web/routes/`): the GUI's own, free to change
  alongside the frontend and the MCP server's own panel client (`tcip_mcp.web_client`, which
  posts to `/api/events/<panel>` and documents `/api/state/tab`), which move with them.
- The frontend (`packages/tcip-web/frontend/`) in its entirety.
- Internal functions inside a pipeline module (`packages/tcip-mcp/src/tcip_mcp/pipelines/`) that
  are not themselves an MCP tool or a documented script entry point.
- The prose inside a knowledge document (`packages/tcip-mcp/src/tcip_mcp/knowledge/`): domain
  knowledge that is expected to be refined as the platform and the crops it covers grow.
- The audit log's line shape. What a given line's fields mean is not a promise this document
  extends.
