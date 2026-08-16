---
name: phenology
description: "Compute and deliver bloom phenology: the catkin/pistillate 05/50/95-per-date milestones, as one row per plant, by composing existing pieces instead of re-scripting. Covers the operationalization (dates a plant's validated elongated fraction crosses 5/50/95%, never a bbox-height proxy), the end-to-end pattern, the pieces to compose (compute_phenology tool, phenology module, plant mapping), and the measurement-integrity guard. Load when computing bloom milestones, building plant mapping, delivering a per-plant phenology CSV, or handling hazelnut catkin or pistillate bloom timing."
---

# Bloom phenology: the 05/50/95-per-date trait

This is the platform's core repeated trait: the dates a plant reaches bloom milestones,
delivered one row per plant. Compose the pieces below; do not re-script this per
project (that fragility is exactly what this skill exists to prevent).

## The authoritative trait definition (do not redefine)

Bloom is the fraction of a plant's detected catkins that are _elongated_.
"Elongated" is an expert-defined, visible morphological stage: a *validated* per-catkin
elongation call learned from the imagery. It's a *state*, not a dimension: judge it from the
object, not off a bbox's height. (See the CLAUDE.md measurement-integrity invariant.) How that
call is produced (a single multi-class detector, detect-then-classify, …) is a pipeline-design
choice; the trait definition does not fix it.

Milestones, per plant, from that plant's elongated-fraction time series:

| Trait | Definition |
|-------|------------|
| `catkin_elongation_date` | date most catkins have elongated (`crops.yml`: "Date when most catkins have elongated"), operationalized as the 95% majority crossing, i.e. synonymous with `catkin_95per_date` |
| `catkin_05per_date` | date the elongated fraction crosses 5% |
| `catkin_50per_date` | date the elongated fraction crosses 50% |
| `catkin_95per_date` | date the elongated fraction crosses 95% |

Crossings interpolate linearly between the two neighbouring capture dates. Pistillate
milestones (`pistillate_05/50/95per_date`) are the identical pattern on the pistillate-
flower elongation/receptivity call.

> Provisional operationalization (pending breeder confirmation). `crops.yml` is the
> immutable authority ("Date when most catkins have elongated"). The implementation computes
> `catkin_elongation_date` as the 95% majority crossing (= `catkin_95per_date`), the
> current best-guess reading of that text, to be confirmed with the breeders; correct the
> mapping in `phenology.plant_milestones` if they rule otherwise. `positive_onset_date`
> (first date any elongation appears) remains a separate helper, not the delivered trait.

Not a count-of-peak. Do not normalize catkin *count* to the season peak and call the
crossings bloom; that is an abundance signal and a different (wrong) trait. There is no
wanted abundance-phenology trait; if a stakeholder asks for one, treat it as a new,
separately-named trait and get the definition in writing first.

This bans the *quantity*, not an estimator. The crossing is defined on the positive
fraction; how you estimate the date at which that fraction reaches a level (the canonical
implementation interpolates linearly between neighbouring capture dates) is a method
question, and a sparse or irregular capture cadence is exactly the case where it deserves
thought rather than a default.

## The measurement-integrity guard

Because "elongated" is a learned per-catkin call, predictions that carry no elongation
call cannot yield a valid bloom fraction. Every surface reports `positive_class_assessed`:
when it is false, the milestones are not a measurement; do not deliver them. Train and
*validate* whatever model produces the elongation call first, against a reference sized to the
trait: GT annotations, or a breeder-confirmed sample of the model's own outputs
(review-confirmation), not dense GT for every trait (either passes the identical disjoint-split +
count-bias gate). See the `evaluation` skill.

## End-to-end pattern

```
per date:  images ─► detect catkins ─► call each catkin elongated vs not (validated)
                  ─► write per-image JSON preds (carrying the elongation call)
across dates: plant mapping (image → plant_id) ─► per (plant, date) elongated fraction
                  ─► crossings at 5/50/95% (elongation_date = the 95% crossing) ─► per-plant CSV
                  ─► carry genotype/accession through to the deliverable
```

- Detection at scale: `run_inference` already supports tiled sliding-window (SAHI-style)
  inference; compose it, don't re-script tiling. Whether and how to tile (tile size, overlap)
  is a data-derived choice: derive it from the imagery resolution and catkin size at runtime,
  and defer the how to the `pipeline-design` / `evaluation` skills.
- The elongated-vs-not call is a distinct decision from "is this a catkin". The fraction is
  `n(elongated) / n(total detected catkins)` for that plant on that date.
- Genotype: carry `accession_name` from the plant mapping into the CSV; breeders
  select on genotype, not `plant_id`.

## The pieces to compose (inventory, so nothing is rediscovered)

| Piece | Where | Role |
|-------|-------|------|
| `build_plant_mapping` (MCP tool) | `tools/phenology_tools.py` | agent entry point (step 1): geolocated images + plant CSVs → persisted `plant_mapping.json` |
| `compute_phenology` (MCP tool) | `tools/phenology_tools.py` | agent entry point (step 2): mapping.json + classified preds → delivered `catkin_phenology.csv`; refuses to write when `positive_class_assessed` is false |
| `phenology` module | `tcip-mcp .../pipelines/postprocessing/phenology.py` | the one canonical milestone implementation: `count_by_class`, `per_plant_phenology`, `crossing_date`, `positive_onset_date`, `plant_milestones`, `write_phenology_csv` |
| `plant_mapping` module | `tcip-mcp .../pipelines/postprocessing/plant_mapping.py` | image → `plant_id` via sequence-anchored GPS matching; `build_mapping`, `persist_mapping`, `load_mapping` |
| Web Results routes | `tcip-web .../routes/results.py` | `/plant_mapping/build`, `/per_plant_curves`, `/onset_dates`: the human UI; delegates to the same shared modules |

Milestone math lives once, in the `phenology` module; plant mapping lives once, in the
`plant_mapping` module. The MCP tools and the web routes all call them, so a mapping and a
milestone date mean the same thing on both surfaces. If you change a definition, change it
there; never fork a second copy. So the agent composes tools end to end:
`build_plant_mapping` → `run_inference` → (elongation call) → `compute_phenology`.

Once a real localization-kind derivation (from actual GT box geometry) or a real breeder-answered
count objective exists for this trait, persist it with `update_trait_spec_fields(project_root,
trait_name, fields, provenance_entries)`, the one audited write path for a `TraitSpec`'s fields
(`count_objective`, `localization`, `positive_class_name`, ...). It refuses if the trait has no
existing spec file (creating a new trait is a separate, still-manual step) and re-validates the
merged spec against `crops.yml` before writing. Never hand-write the trait's spec YAML directly.

Don't confuse `score_predictions` (IoU GT-vs-prediction *eval* matching) with plant-GPS
mapping; they are unrelated.

## Plant mapping: why the sequence-anchored matcher

Image GPS (iPhone/handheld EXIF) carries ~5 m error while the plant grid is ~2.8 m between
adjacent plots, so nearest-neighbour GPS alone is ambiguous. The RTK-collected,
GIS-rectified plant grid is accurate; the *image* GPS is the fuzzy side. `plant_mapping.py`
resolves this by ordering each date's images by EXIF capture time (the walker's sequence),
splitting into row runs on large GPS jumps, and assigning along the row. Each assignment
records its `source` (`sequence` / `nearest_neighbour` / `unmapped`) and `distance_m`:
honest, interpretable signals. It deliberately emits no fabricated 0–1 "confidence".

## Delivery checklist

1. `positive_class_assessed` is true (predictions carry the elongation call).
2. Every expected plant has a row; genotype/`accession` is carried through.
3. Milestones are chronologically sane (`05per` ≤ `50per` ≤ `95per`; `elongation_date`, the
   majority crossing, equals `95per`).
4. Plants that never reach a level have `null` for that milestone (not a fabricated date).
5. The `undated/` image bucket is excluded from the time series (it has no capture date).
