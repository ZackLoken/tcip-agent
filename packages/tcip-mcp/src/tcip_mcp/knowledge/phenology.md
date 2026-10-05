---
name: phenology
description: "Compute and deliver bloom phenology: the catkin/pistillate 05/50/95-per-date milestones, as one row per plant, by composing existing pieces instead of re-scripting. Covers the operationalization (dates a plant's validated elongated fraction crosses 5/50/95%, never a bbox-height proxy), the end-to-end pattern, the pieces to compose (deliver_phenology_milestones tool, phenology module, plant mapping), and the measurement-integrity guard. Load when computing bloom milestones, building plant mapping, delivering a per-plant phenology CSV, or handling hazelnut catkin or pistillate bloom timing."
---

# Bloom phenology: the 05/50/95-per-date trait

This is the platform's core repeated trait: the dates a plant reaches bloom milestones,
delivered one row per plant. Compose the pieces below; do not re-script this per project.

## The authoritative trait definition (do not redefine)

Bloom is the fraction of a plant's detected catkins that are _elongated_.
"Elongated" is an expert-defined, visible morphological stage: a *validated* per-catkin
elongation call learned from the imagery. It's a *state*, not a dimension: judge it from the
object, not off a bbox's height. (See the CLAUDE.md measurement-integrity invariant.) How that
call is produced (a detector carrying a per-instance head for the state's attribute,
detect-then-classify, …) is a pipeline-design choice; the trait definition does not fix it.

Milestones, per plant, from that plant's elongated-fraction time series:

| Trait | Definition |
|-------|------------|
| `catkin_elongation_date` | date most catkins have elongated (`crops.yml`: "Date when most catkins have elongated"); the trait's confirmed revision states which crossing this maps to (below) |
| `catkin_05per_date` | date the elongated fraction crosses 5% |
| `catkin_50per_date` | date the elongated fraction crosses 50% |
| `catkin_95per_date` | date the elongated fraction crosses 95% |

Crossings interpolate linearly between the two neighboring capture dates. Pistillate
milestones (`pistillate_05/50/95per_date`) are the identical pattern on the pistillate-
flower elongation/receptivity call.

> Which crossing the majority date means. `crops.yml` is the immutable authority ("Date when
> most catkins have elongated"). The crossing `catkin_elongation_date` is computed at is the
> trait entry's `majority_milestone`, and it answers for a delivery only once the breeder has
> confirmed the revision stating it in the Setup tab. A disagreement over which crossing the
> majority date means is corrected in the entry itself, as a new revision proposed with
> `propose_trait`, not in this file. `positive_onset_date` (first date any elongation appears)
> remains a separate helper, not the delivered trait.

Not a count-of-peak. Do not normalize catkin *count* to the season peak and call the
crossings bloom; that is an abundance signal and a different (wrong) trait. There is no
wanted abundance-phenology trait; if a stakeholder asks for one, treat it as a new,
separately-named trait and get the definition in writing first.

This bans the *quantity*, not an estimator. The crossing is defined on the positive
fraction; how you estimate the date at which that fraction reaches a level (the canonical
implementation interpolates linearly between neighboring capture dates) is a method
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
                  ─► publish per-image prediction documents (carrying that call)
across dates: plant mapping (image → plant_id) ─► per (plant, date) elongated fraction
                  ─► crossings at 5/50/95% (and the confirmed majority crossing above) ─► per-plant CSV
                  ─► carry genotype/accession through to the deliverable
```

- Detection at scale: `run_inference` already supports tiled sliding-window (SAHI)
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
| `register_plant_registry` (MCP tool) | `tools/phenology_tools.py` | names a plant-locations CSV set once (`{path, sha256, n_plants}` per file, `crop`, `site`, a content digest), so `build_plant_mapping` and `deliver_orthomosaic_plant_counts` read the same registered version instead of re-asserting file paths; a shapefile is converted first by `tcip shp-to-plant-csv` |
| `build_plant_mapping` (MCP tool) | `tools/phenology_tools.py` | agent entry point (step 1): geolocated images (a registered dataset's own `images/` root) + a `plant_registry` name → a named mapping persisted under the project. A same-name rebuild a delivery event still cites refuses by name unless `supersede=True`, which archives the cited record first (readable by digest, never enumerated) |
| `deliver_phenology_milestones` (MCP tool) | `tools/phenology_tools.py` | agent entry point (step 2): a named mapping + one bucket per date (each at the date its own record states) → delivered `catkin_phenology.csv`; the gate refuses buckets whose scope classifies no positive state |
| `phenology` module | `tcip-mcp .../pipelines/postprocessing/phenology.py` | the one canonical milestone implementation: `count_by_class`, `per_plant_phenology`, `crossing_date`, `positive_onset_date`, `plant_milestones`, `measure_phenology` and the gated delivery door `deliver_phenology` (it writes under the trait's confirmed revision and names it on the delivery event). Every population plant has a point on every mapped date; a date the mapping assigns it no capture on counts against its completeness |
| `plant_mapping` module | `tcip-mcp .../pipelines/postprocessing/plant_mapping.py` | image → `plant_id` via sequence-anchored GPS matching; `build_mapping`, `persist_mapping`, `load_mapping`, `verify_mapping_inputs`, `plant_mapping_names`, `register_plant_registry_record`, `load_registry`. A mapping is project state, named and bound to the dataset it was built over and to its own build receipt: `load_mapping` refuses a record no receipt names, and `verify_mapping_inputs` re-checks the record's dates and plant CSVs (read through the named registry) at delivery time |
| Web Results routes (phenology-specific) | `tcip-web .../routes/results.py` | `/plant_mapping/build`, `/plant_mapping/load`, `/plant_mapping/list`, `/phenology_measurement` (both projections, curve and milestone, from one measurement), `/export_csv` (the door that writes): the human UI; delegates to the same shared modules. Lists only this router's phenology routes; it also carries trait-general routes (the traits and their revision confirmation, delivery events, registered models) not enumerated here |

Milestone math lives once, in the `phenology` module; plant mapping lives once, in the
`plant_mapping` module. The MCP tools and the web routes all call them. If you change a
definition, change it there; never fork a second copy. So the agent composes tools end to end:
`register_plant_registry` → `build_plant_mapping` → `run_inference` → (elongation call) →
`deliver_phenology_milestones`.

Once a real localization-kind derivation (from actual GT box geometry) or a real breeder-answered
count objective exists for this trait, record it with `propose_trait(entry,
rationale)`, the one write path for a trait: it takes the complete entry (the spec fields, such as
`count_objective`, `localization` and `positive_state`, and the operationalization per delivery
kind) and appends it as a new, unconfirmed revision. The positive state names one attribute of
the measured subject and one of its values, each declared in the delivered dataset's own subject
registry, checked when the revision is proposed and again at every delivery. The breeder confirms the whole
revision, spec fields and operationalizations together, in the Setup tab; a delivery ships under
the latest confirmed revision and names it on its event. Never hand-write the trait's record.

Don't confuse `annotation_tools.score_predictions` (IoU GT-vs-prediction *eval* matching, a
library call) with plant-GPS mapping; they are unrelated.

## Plant mapping: why the sequence-anchored matcher

Image GPS (iPhone/handheld EXIF) carries ~5 m error while the plant grid is ~2.8 m between
adjacent plots, so nearest-neighbor GPS alone is ambiguous. The RTK-collected,
GIS-rectified plant grid is accurate; the *image* GPS is the fuzzy side. `plant_mapping.py`
resolves this by ordering each date's images by EXIF capture time (the walker's sequence),
splitting into row runs on large GPS jumps, and assigning along the row. Each assignment
records its `source` (`sequence` / `nearest_neighbor` / `unmapped`) and `distance_m`:
interpretable signals. It records no 0–1 "confidence" value.

## Delivery checklist

1. `positive_class_assessed` is true (predictions carry the elongation call).
2. Every expected plant has a row; genotype/`accession` is carried through.
3. Milestones are chronologically sane (`05per` ≤ `50per` ≤ `95per`; `elongation_date`, the
   majority crossing, equals `95per`).
4. Plants that never reach a level have `null` for that milestone.
5. The `undated/` image bucket is excluded from the time series (it has no capture date).
