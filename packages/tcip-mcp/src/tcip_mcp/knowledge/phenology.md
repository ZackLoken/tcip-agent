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
call is produced (a single multi-class detector, detect-then-classify, …) is a pipeline-design
choice; the trait definition does not fix it.

Milestones, per plant, from that plant's elongated-fraction time series:

| Trait | Definition |
|-------|------------|
| `catkin_elongation_date` | date most catkins have elongated (`crops.yml`: "Date when most catkins have elongated"); see the crossing-unconfirmed operationalization below for what this maps to |
| `catkin_05per_date` | date the elongated fraction crosses 5% |
| `catkin_50per_date` | date the elongated fraction crosses 50% |
| `catkin_95per_date` | date the elongated fraction crosses 95% |

Crossings interpolate linearly between the two neighbouring capture dates. Pistillate
milestones (`pistillate_05/50/95per_date`) are the identical pattern on the pistillate-
flower elongation/receptivity call.

> Crossing-unconfirmed operationalization (pending breeder confirmation). `crops.yml` is the
> immutable authority ("Date when most catkins have elongated"). The implementation computes
> `catkin_elongation_date` as the 95% majority crossing (= `catkin_95per_date`), the
> current best-guess reading of that text, recorded on the trait spec as `majority_milestone`
> and flagged crossing-unconfirmed via `majority_provisional`. That confirmation path is not
> `state_trait_operationalization`, which confirms `state_crossing_dates`' own fields
> (`positive_class_name`, `milestone_on`, `milestone_fractions`) and does not touch this
> mapping; a disagreement over which crossing the majority date means is corrected on the
> trait spec itself, through `revise_trait_spec` (or set at authoring time via
> `author_trait_spec`), not this file. `positive_onset_date`
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
                  ─► crossings at 5/50/95% (see the crossing-unconfirmed operationalization above) ─► per-plant CSV
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
| `register_plant_registry` (MCP tool) | `tools/phenology_tools.py` | names a plant-locations CSV set once (`{path, sha256, n_plants}` per file, `crop`, `site`, a content digest), so `build_plant_mapping` and `deliver_orthomosaic_plant_counts` read the same registered version instead of re-asserting file paths |
| `build_plant_mapping` (MCP tool) | `tools/phenology_tools.py` | agent entry point (step 1): geolocated images (a registered dataset's own `images/` root) + a `plant_registry` name → a named mapping persisted under the project. A same-name rebuild a delivery event still cites refuses by name unless `supersede=True`, which archives the cited record first (readable by digest, never enumerated) |
| `deliver_phenology_milestones` (MCP tool) | `tools/phenology_tools.py` | agent entry point (step 2): a named mapping + classified preds → delivered `catkin_phenology.csv`; refuses to write when `positive_class_assessed` is false |
| `phenology` module | `tcip-mcp .../pipelines/postprocessing/phenology.py` | the one canonical milestone implementation: `count_by_class`, `per_plant_phenology`, `crossing_date`, `positive_onset_date`, `plant_milestones`, and the gated delivery doors `write_phenology_csv` / `write_phenology_curve_csv` (both refuse without a passing operationalization basis) |
| `plant_mapping` module | `tcip-mcp .../pipelines/postprocessing/plant_mapping.py` | image → `plant_id` via sequence-anchored GPS matching; `build_mapping`, `persist_mapping`, `load_mapping`, `verify_mapping_inputs`, `plant_mapping_names`, `register_plant_registry_record`, `load_registry`. A mapping is project state, named and bound to the dataset it was built over and to its own build receipt: `load_mapping` refuses a record no receipt names, and `verify_mapping_inputs` re-checks the record's dates and plant CSVs (read through the named registry) at delivery time |
| Web Results routes (phenology-specific) | `tcip-web .../routes/results.py` | `/plant_mapping/build`, `/plant_mapping/load`, `/plant_mapping/list`, `/phenology_measurement` (both projections, curve and milestone, from one measurement), `/export_csv` (the door that writes): the human UI; delegates to the same shared modules. Lists only this router's phenology routes; it also carries trait-general routes (operationalization records, trait-spec statements, delivery events, registered models) not enumerated here |

Milestone math lives once, in the `phenology` module; plant mapping lives once, in the
`plant_mapping` module. The MCP tools and the web routes all call them. If you change a
definition, change it there; never fork a second copy. So the agent composes tools end to end:
`register_plant_registry` → `build_plant_mapping` → `run_inference` → (elongation call) →
`deliver_phenology_milestones`.

Once a real localization-kind derivation (from actual GT box geometry) or a real breeder-answered
count objective exists for this trait, persist it with `revise_trait_spec(project_root,
trait_name, fields)`, the one audited write path for a `TraitSpec`'s fields (`count_objective`,
`localization`, `positive_class_name`, ...; the positive class must be a value one of the measured
subject's attributes declares in the delivered dataset's own class registry, checked when the
crossing statement is made and again at every delivery). It refuses if the trait has no existing
spec file; register one first with `author_trait_spec(project_root, trait, delivers, rationale,
...)`, which records the breeder's own account of the trait's measurement for their later
confirmation in the GUI. `revise_trait_spec` re-validates the merged spec against
`crops.yml` before writing. Never hand-write the trait's spec YAML directly.

Don't confuse `annotation_tools.score_predictions` (IoU GT-vs-prediction *eval* matching, a
library call) with plant-GPS mapping; they are unrelated.

## Plant mapping: why the sequence-anchored matcher

Image GPS (iPhone/handheld EXIF) carries ~5 m error while the plant grid is ~2.8 m between
adjacent plots, so nearest-neighbour GPS alone is ambiguous. The RTK-collected,
GIS-rectified plant grid is accurate; the *image* GPS is the fuzzy side. `plant_mapping.py`
resolves this by ordering each date's images by EXIF capture time (the walker's sequence),
splitting into row runs on large GPS jumps, and assigning along the row. Each assignment
records its `source` (`sequence` / `nearest_neighbour` / `unmapped`) and `distance_m`:
interpretable signals. It records no 0–1 "confidence" value.

## Delivery checklist

1. `positive_class_assessed` is true (predictions carry the elongation call).
2. Every expected plant has a row; genotype/`accession` is carried through.
3. Milestones are chronologically sane (`05per` ≤ `50per` ≤ `95per`; `elongation_date`, the
   majority crossing, equals `95per`).
4. Plants that never reach a level have `null` for that milestone.
5. The `undated/` image bucket is excluded from the time series (it has no capture date).
