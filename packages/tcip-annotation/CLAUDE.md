# packages/tcip-annotation

Headless annotation/review engine: label I/O, IoU matching, SAM wrapper. Loads on top of the root
`CLAUDE.md`; invariants and operating posture there apply here and aren't restated.

## Layout

```
src/tcip_annotation/
  annotation_engine.py   # core annotation read/write
  review_engine.py        # review-verdict logging (accept/edit/reject) and the GT label write-back
                          # (save_gt); the hard-negative partition itself lives in tcip-mcp's pipelines/feedback/materialize.py
  json_io.py               # per-image JSON annotation records
  format_io.py              # format detection/parsing
  matching.py                # IoU matching (GT vs prediction, review vs GT)
  sam_wrapper.py               # SAM-assisted labeling
  state.py                      # engine-local state
  utils.py, viz.py
```

## Conventions specific to this package

- No dependency on `tcip-mcp` or `tcip-web`. The one TCIP dependency it does carry is
  `tcip-store`, the storage seam below all three packages. A private copy of the
  temp-file-plus-replace primitive is not an acceptable substitute. A caller outside TCIP
  addresses a file through `RootedFileLocator` and needs nothing from `tcip-mcp`'s layout.
- Label formats: `{json, coco}` only. VOC, LabelMe, and YOLO annotation label formats are not
  supported; don't reintroduce a format branch without checking `format_io.py` first.
- A negative is empty labels plus an explicit human Complete (see root `CLAUDE.md`'s
  measurement-integrity invariants); this package's read/write paths must not treat an empty label
  file alone as a negative.
