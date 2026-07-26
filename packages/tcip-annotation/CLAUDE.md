# packages/tcip-annotation

Headless annotation/review engine — label I/O, IoU matching, SAM wrapper. Loads on top of the root
`CLAUDE.md`; invariants and operating posture there apply here and aren't restated.

## Layout

```
src/tcip_annotation/
  annotation_engine.py   # core annotation read/write
  review_engine.py        # review-verdict processing (accept/edit/reject -> labels/hard negatives)
  json_io.py               # per-image JSON annotation records
  format_io.py              # format detection/parsing
  matching.py                # IoU matching (GT vs prediction, review vs GT)
  sam_wrapper.py               # SAM-assisted labeling
  state.py                      # engine-local state
  utils.py, viz.py
```

## Conventions specific to this package

- **No dependency on `tcip-mcp` or `tcip-web`.** Keep it that way — this is the one package usable
  standalone. If a change here starts requiring an import from either, that's a design smell, not a
  detail to route around.
- **Label formats: `{json, coco}` only.** VOC/LabelMe/YOLO label formats were removed (K13c) along
  with the `_getexif` fallback. Don't reintroduce a format branch without checking whether that
  removal is still current.
- **Label-format "YOLO" is not the YOLO model.** `YoloPredictor`, `KIND_ULTRALYTICS`, and
  `predictor.py` are ultralytics checkpoint support — a different thing that happens to share a
  name. Only the label *format* was removed; don't touch model-side YOLO support on the strength of
  that.
- A negative is empty labels **plus** an explicit human Complete (see root `CLAUDE.md`'s
  measurement-integrity invariants) — this package's read/write paths must not treat an empty label
  file alone as a negative.
