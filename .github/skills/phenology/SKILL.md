---
name: phenology
description: "Compute and deliver bloom phenology — the catkin/pistillate 05/50/95-per-date milestones — as one row per plant, by composing existing pieces instead of re-scripting. Covers the operationalization (dates a plant's validated-classifier elongated fraction crosses 5/50/95%, never a bbox-height proxy), the end-to-end pattern, the pieces to compose (compute_phenology tool, phenology module, plant mapping), and the measurement-integrity guard. Load when computing bloom milestones, building plant mapping, delivering a per-plant phenology CSV, or handling hazelnut catkin or pistillate bloom timing."
---

# Bloom phenology — the 05/50/95-per-date trait

This is the platform's core repeated trait: the dates a plant reaches bloom milestones,
delivered one row per plant. **Compose the pieces below — do not re-script this per
project** (that fragility is exactly what this skill exists to prevent).

## The authoritative trait definition (do not redefine)

Bloom is the **fraction of a plant's detected catkins that are _elongated_.**
"Elongated" is an **expert-defined, visible morphological stage** emitted by a *validated*
2-class classifier (class 1 = elongated). It is a **classification**, never a geometric
proxy — bounding-box height / aspect ratio is scale-, zoom-, and pose-dependent and does
**not** measure elongation. (A prior session invented a bbox-height threshold and shipped
it into a delivered CSV; that was removed. See the CLAUDE.md measurement-integrity
invariant.)

Milestones, per plant, from that plant's elongated-fraction time series:

| Trait | Definition |
|-------|------------|
| `catkin_elongation_date` | date **most** catkins have elongated — the elongated fraction reaches a majority (`crops.yml`: "Date when most catkins have elongated") |
| `catkin_05per_date` | date the elongated fraction crosses **5%** |
| `catkin_50per_date` | date the elongated fraction crosses **50%** |
| `catkin_95per_date` | date the elongated fraction crosses **95%** |

Crossings interpolate linearly between the two neighbouring capture dates. Pistillate
milestones (`pistillate_05/50/95per_date`) are the identical pattern on the pistillate-
flower detector/classifier.

> **Implementation reconciliation needed (code, not definition).** `crops.yml` is the
> authority: `catkin_elongation_date` is a *majority* event ("most catkins have elongated"),
> so it is the majority crossing — near `catkin_50per_date`. The platform's
> `elongation_onset_date` helper still computes *onset* (first date any elongation appears),
> which contradicts this. Reconcile the implementation to the majority crossing and
> **validate** it before delivering `catkin_elongation_date`; the `05/50/95` crossings are
> unaffected.

**Not a count-of-peak, not a sigmoid fit.** Do not normalize catkin *count* to the season
peak and call the crossings bloom — that is an abundance signal and a different (wrong)
trait. There is no wanted abundance-phenology trait; if a stakeholder asks for one, treat
it as a new, separately-named trait and get the definition in writing first.

## The measurement-integrity guard

Because "elongated" is a classifier class, predictions that carry **no** elongation class
cannot yield a valid bloom fraction. Every surface reports `elongation_classified`:
when it is false, the milestones are **not** a measurement — do not deliver them. Run and
*validate* the 2-class elongation classifier first (see the `evaluation` skill).

## End-to-end pattern

```
per date:  images ─► detect catkins ─► classify each elongated vs not (validated 2-class)
                  ─► write YOLO preds (class id = elongation class)
across dates: plant mapping (image → plant_id) ─► per (plant, date) elongated fraction
                  ─► crossings at 5/50/95% + first-elongation date ─► per-plant CSV
                  ─► carry genotype/accession through to the deliverable
```

- **Detection at scale:** `run_inference(tile=True, tile_size=…, overlap=…)` **is** SAHI-
  style tiled sliding-window inference. Use it for high-res ground imagery with many small
  catkins — do **not** re-script tiling.
- **Elongated vs not is a classification step**, distinct from detection. The fraction is
  `n(class==elongated) / n(total detections)` for that plant on that date.
- **Genotype:** carry `accession_name` from the plant mapping into the CSV — breeders
  select on genotype, not `plant_id`.

## The pieces to compose (inventory — so nothing is rediscovered)

| Piece | Where | Role |
|-------|-------|------|
| `build_plant_mapping` (MCP tool) | `tools/phenology_tools.py` | **agent entry point (step 1)** — geolocated images + plant CSVs → persisted `plant_mapping.json` |
| `compute_phenology` (MCP tool) | `tools/phenology_tools.py` | **agent entry point (step 2)** — mapping.json + classified preds → delivered `catkin_phenology.csv`; refuses to write when `elongation_classified` is false |
| `phenology` module | `tcip-mcp .../pipelines/postprocessing/phenology.py` | the **one** canonical milestone implementation: `count_by_class`, `per_plant_phenology`, `crossing_date`, `elongation_onset_date`, `plant_milestones`, `write_phenology_csv` |
| `plant_mapping` module | `tcip-mcp .../pipelines/postprocessing/plant_mapping.py` | image → `plant_id` via sequence-anchored GPS matching; `build_mapping`, `persist_mapping`, `load_mapping` |
| Web Results routes | `tcip-web .../routes/results.py` | `/plant_mapping/build`, `/per_plant_curves`, `/onset_dates` — the human UI; delegates to the **same** shared modules |

**Milestone math lives once**, in the `phenology` module; **plant mapping lives once**, in the
`plant_mapping` module. The MCP tools and the web routes all call them, so a mapping and a
milestone date mean the same thing on both surfaces. If you change a definition, change it
there — never fork a second copy. So the agent composes tools end to end:
`build_plant_mapping` → `run_inference(tile=True)` → (elongation classifier) → `compute_phenology`.

**Don't confuse `run_matching`** (IoU GT-vs-prediction *eval* matching) with plant-GPS
mapping — they are unrelated.

## Plant mapping — why the sequence-anchored matcher

Image GPS (iPhone/handheld EXIF) carries ~5 m error while the plant grid is ~2.8 m between
adjacent plots — so nearest-neighbour GPS alone is ambiguous. The RTK-collected,
GIS-rectified plant grid is accurate; the *image* GPS is the fuzzy side. `plant_mapping.py`
resolves this by ordering each date's images by EXIF capture time (the walker's sequence),
splitting into row runs on large GPS jumps, and assigning along the row. Each assignment
records its `source` (`sequence` / `nearest_neighbour` / `unmapped`) and `distance_m` —
honest, interpretable signals. It deliberately emits **no** fabricated 0–1 "confidence".

## Delivery checklist

1. `elongation_classified` is **true** (predictions carry the elongation class).
2. Every expected plant has a row; genotype/`accession` is carried through.
3. Milestones are chronologically sane (`05per` ≤ `50per` ≤ `95per`; `elongation_date` — the
   majority crossing — lands near `50per`, not before `05per`).
4. Plants that never reach a level have `null` for that milestone (not a fabricated date).
5. The `undated/` image bucket is excluded from the time series (it has no capture date).
