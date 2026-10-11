# packages/tcip-store

The storage seam: a root's records and logs in one SQLite database, and the files the platform
writes by path. Bottom of the stack, depending on nothing else in the platform. Loads on top of the
root `CLAUDE.md`; invariants and operating posture there apply here and aren't restated.

## Layout

```text
src/tcip_store/
  __init__.py          # the public surface, re-exported
  model.py             # Key, Version, Versioned, LogPage, canonical_path
  errors.py            # every refusal the seam raises
  values.py            # what a value must be to be stored, and its two spellings: a record
                        #   (encode_record) and a log line (encode_log_line), read by decode_value
  store.py             # the module functions over the one backend a process binds
  sqlite_backend.py    # the database: <root>/.tcip/store.db, WAL at full synchronous, one
                        #   connection per process, thread and root; the transaction handle Txn
  file_backend.py      # a file written atomically by path under its lock, and the lock a root's
                        #   database is created under
```

## Conventions specific to this package

- A record or log is addressed by a `Key` (store, root, parts); a module that owns a store spells
  its key builder and nothing else. A file is addressed by the path its owning module computes.
- A transaction is one commit over one root's records and logs: `txn.write`, `txn.delete` and
  `txn.append` land together or not at all.
- Every process binds the backend once at its entry point with `tcip_store.bind()`.
- `tcip dump-store` writes a project's records and logs out as files for a person to read; nothing
  reads them back.
- No dependency on `tcip-annotation`, `tcip-mcp`, or `tcip-web`.
