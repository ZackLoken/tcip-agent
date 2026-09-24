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
  format_io.py              # the external COCO reader, for the import only
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
- One label shape: the per-image JSON document. An external dataset-level COCO is read once by
  `format_io.parse_coco_annotations`, on its way into per-image documents through tcip-mcp's
  import door, and never written or trained on. VOC, LabelMe, and YOLO are not read. A record
  carrying `iscrowd` is a region of unseparated objects, never one instance; a run-length mask
  arrives as rings. One per-record decoder (`json_io.annotation_of_record`) reads every record,
  and a supplied value that does not parse refuses rather than reading as absent.
- A negative is empty labels plus an explicit human Complete (see root `CLAUDE.md`'s
  measurement-integrity invariants); this package's read/write paths must not treat an empty label
  file alone as a negative.
