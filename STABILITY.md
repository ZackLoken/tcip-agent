# Stability

What an adopter can build against today, independent of the 0.x version number (see VERSIONING.md);
everything not named here is free to change in any release while the shared version stays below
`1.0.0`.

## Persisted formats

No persisted record shape is frozen while the version stays below `1.0.0`.

## The MCP tool surface

Tools may be added in any release.

## `tcip-annotation`'s public API

`tcip-annotation` is the one package meant for standalone use outside this platform. Its public API
is exactly what `packages/tcip-annotation/src/tcip_annotation/__init__.py`'s `__all__` declares: the
`Annotation` record and its geometries (`BBox`, `Point`, `Polygon`, and `bbox_of`, a geometry's
enclosing box), the external COCO reader (`parse_coco_annotations`), the point-in-polygon test among
the matcher's geometry helpers (`point_in_polygon`), a binary mask's rings, one per connected region
(`mask_to_polygon_rings`), and a reference-grid cell's name and rect (`cell_fields`). `__all__` is
the only boundary.

## Not stable

- The web routes (`packages/tcip-web/src/tcip_web/routes/`): the GUI's own, free to change alongside
  the frontend and the MCP server's own panel client (`tcip_mcp.web_client`, which posts to
  `/api/events/<panel>`), which move with them.
- The frontend (`packages/tcip-web/frontend/`) in its entirety.
- Internal functions inside a pipeline module (`packages/tcip-mcp/src/tcip_mcp/pipelines/`) that are
  not themselves an MCP tool.
- The prose inside a knowledge document (`packages/tcip-mcp/src/tcip_mcp/knowledge/`): domain
  knowledge that is expected to be refined as the platform and the crops it covers grow.
- The audit log's line shape. What a given line's fields mean is not a promise this document
  extends.
