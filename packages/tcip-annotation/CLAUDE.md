# packages/tcip-annotation

Headless annotation library: the label document and its completion marks, the verdict shard, the one
matcher. Loads on top of the root `CLAUDE.md`; invariants and operating posture there apply here and
aren't restated.

## Layout

```text
src/tcip_annotation/
  json_io.py      # the per-image label document: records, completion marks, provenance stamping
  verdicts.py     # the verdict shard: a person's accept/reject decisions on a bucket's proposals
  matching.py     # the one matcher of detections to ground truth, and geometry helpers
  format_io.py    # the external COCO reader, for the import only
  grid.py         # reference-grid cell names and the named-cell lookup
  mask_contours.py  # a binary mask's rings, one per connected region
  state.py        # the Annotation record and its geometries
  utils.py, viz.py
```

## Conventions specific to this package

- No dependency on `tcip-mcp` or `tcip-web`. The one TCIP dependency it does carry is `tcip-store`,
  the storage seam below all three packages. A private copy of the store primitive is not an
  acceptable substitute. A document is a record addressed by the `tcip_store.Key` its caller names;
  a person reads it as a file through `tcip dump-store`.
- One label shape: the per-image label document. An external dataset-level COCO is read once by
  `format_io.parse_coco_annotations`, on its way into per-image documents through tcip-mcp's import
  door, and never written or trained on. VOC, LabelMe, and YOLO are not read. A record carrying
  `iscrowd` is a region of unseparated objects, never one instance; a run-length mask arrives as
  rings. One per-record decoder (`json_io.annotation_of_record`) reads every record, and a supplied
  value that does not parse refuses rather than reading as absent.
- A negative is an empty subject plus a person's completion mark in the same document (see root
  `CLAUDE.md`'s measurement-integrity invariants); `LabelDocument.state` is the one reading of it,
  and an empty label document alone is never a negative.
