# packages/tcip-store

The storage seam: one interface for the platform's mutable records, logs and blobs, over a
database backend and a file backend that must mean the same thing. Bottom of the stack,
depending on nothing else in the platform. Loads on top of the root `CLAUDE.md`; invariants and
operating posture there apply here and aren't restated.

## Layout

```
src/tcip_store/
  __init__.py          # the public surface: Key, Store, errors, registry helpers, re-exported
  model.py             # Key, Version and the other identity/value types, identical on every backend
  errors.py            # every refusal the seam raises
  registry.py          # the store catalogue: each store's kind, codec and concurrency policy,
                        #   declared once by the module that owns it
  schema_version.py    # the version-field accept rule every frozen store's reader applies
  store.py             # the public surface's module functions, bound to one backend per process
  binding.py           # which backend a process binds, decided once at its entry point
  file_backend.py      # the filesystem backend: identity to path, atomic replace, file locks,
                        #   logs and blobs
  sqlite_backend.py    # the database backend: one WAL database per root, blobs left as files
  adoption.py          # moving a root's existing record and log files into a database
  export.py            # writing a root's database back out as files
  layout_claims.py     # which store could own which path under a root, shared by the conform
                        #   rail and the adoption planner
  values.py            # what a value must be before a store will carry it: JSON-safe, finite
                        #   numbers
```

## Conventions specific to this package

- Every operation is addressed by a `Key`, never a path; a write acquires that key's lock inside
  the call.
- A store declares its kind, codec, concurrency policy and durability once, in the module that
  owns it; importing that module registers it, so an `UnknownStore` is answered by an import.
- Two backends must answer alike: `binding.py` selects the database (the default) or the file
  layout from the storage-backend environment variable. A root's records move between the two
  only through `adoption.py` and `export.py`, never by binding the other backend directly against
  files the first one produced.
- A frozen store's documents carry a `schema_version` ceiling, checked by `schema_version.py`;
  absence in an existing document reads as version 1.
- No dependency on `tcip-annotation`, `tcip-mcp`, or `tcip-web`.
