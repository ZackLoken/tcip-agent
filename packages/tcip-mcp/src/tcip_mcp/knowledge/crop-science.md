---
name: crop-science
description: "Cross-cutting domain context for TCIP's six perennial tree-crop breeding crops (hazelnut, chestnut, currant, elderberry, persimmon, black locust): their identity and growth form, the phenophase framing that schedules image capture, and the physical sensing reality that bounds what drone RGB / multispectral / LiDAR, ground and proximal RGB, and lab RGB / NIRS can and cannot observe. Load before phenotyping any crop, choosing a sensing modality, scoping whether a trait is measurable from imagery, reasoning about field-imageable versus lab/destructive traits, or planning the collect-at-phenophase then automate-measurement then per-genotype-selection workflow. Frames sensing as physical constraints on per-trait derivation, never a fixed sensor-to-trait or task-to-trait map. Defers per-crop trait lists to the per-crop documents and crops.yml, and bloom-milestone math to the phenology skill."
---

# Crop science: cross-cutting domain context

General ground for phenotyping the Savanna Institute's six perennial tree crops. This is
the shared context; per-crop detail lives in the per-crop documents, the trait vocabulary in
`packages/tcip-mcp/src/tcip_mcp/knowledge/crops/crops.yml` (the authority), and bloom-milestone
math in the `phenology` skill. The recurring question: given the physics of the sensor, can this
trait even be observed, and if so, at what perspective?

## The six crops at a glance

Not a single taxon: two single-trunk trees, one tall hardwood, and three multi-stem
shrubs. Growth form drives imaging (a hedgerow shrub is not a lone orchard tree), so
identify the crop before scoping a capture. Each row points to its per-crop document for
identity, phenology detail, and the field-imageable-versus-lab partition.

| Crop | Latin / growth form | crops.yml traits | Per-crop document |
|------|---------------------|------------------|----------------|
| Hazelnut | *Corylus americana* × *C. avellana* hybrids, multi-stemmed, clump/thicket-forming shrub (~2–5 m), grown as hedgerow rows | 58 | `packages/tcip-mcp/src/tcip_mcp/knowledge/crops/hazelnut.md` |
| Chestnut | *Castanea* spp., Chinese chestnut *C. mollissima* and American/hybrid (*C. dentata*) material; single-stem deciduous nut tree, rounded spreading crown | 21 | `packages/tcip-mcp/src/tcip_mcp/knowledge/crops/chestnut.md` |
| Currant | *Ribes* spp. (currants, black currant *R. nigrum* and others), multi-stem deciduous shrub/bush (~1–2 m) | 86 | `packages/tcip-mcp/src/tcip_mcp/knowledge/crops/currant.md` |
| Elderberry | *Sambucus nigra* subsp. *canadensis*, suckering, multi-stemmed deciduous shrub (~2–4 m, to ~6 m), hedgerow rows | 69 | `packages/tcip-mcp/src/tcip_mcp/knowledge/crops/elderberry.md` |
| Persimmon | *Diospyros virginiana*, single-trunked, dioecious (occasionally polygamous) deciduous orchard tree | 20 | `packages/tcip-mcp/src/tcip_mcp/knowledge/crops/persimmon.md` |
| Black locust | *Robinia pseudoacacia* (Fabaceae), fast-growing, clonal, N-fixing deciduous hardwood; thorns, suckers | 10 | `packages/tcip-mcp/src/tcip_mcp/knowledge/crops/black-locust.md` |

The per-crop counts sum above the file's 180 unique trait names because many traits (for
example `plant_height`, `flavor_rating`, `bloom_50per_date`) are shared across crops and
counted once per crop that carries them. crops.yml is the source of truth for which crop
carries which trait; verify there, never assert from memory.

## Phenophase framing (schedule; the math lives elsewhere)

Reproductive and vegetative events run in a genotype- and site-specific sequence:
dormancy → budbreak → bloom → fruit/nut set → ripening → harvest → senescence. Timing is
commonly framed with Growing Degree Days (GDD): thermal accumulation above a base
temperature predicts phenophase transitions. GDD is a *scheduling* aid: it tells
you roughly when to fly or walk a row, and it transfers poorly across sites and cultivars; how
much growing-season-onset variance it actually explains is unconfirmed (see Needs expert
confirmation). It does not read a phenophase from an image.

The consequence for capture: a milestone trait is a crossing between two visit dates,
not a single-image reading. Date traits (`catkin_05per_date`, `catkin_50per_date`,
`catkin_95per_date`, `catkin_elongation_date`, `pistillate_05per_date`, `bloom_05per_date`,
`bloom_95per_date`, `fruit_ripe_50per_date`, `burr_drop_date`, `catkin_bloom_date`,
`leaf_out_05per_date`, `leaf_senescence_95per_date`) require repeated captures
bracketing the transition. One flight yields at most a censored bound (an upper bound when the
single observation already meets the target, a lower bound when it does not), never a measured
crossing. The milestone definition and the
crossing math (elongated-fraction, linear interpolation between visits, per-plant delivery)
belong to the `phenology` skill; compose it, do not re-derive it here.

Each phenophase also *exposes* different structure. Leaf-off winter exposes bare
architecture (best window for structural capture: `plant_height`, `plant_volume`,
`stem_count`). Bloom exposes reproductive organs. Ripening exposes color and yield.
Senescence is a change-detection signal across the temporal series.

## Sensing reality: physics that bounds observability

Read this as physical constraints, not a pipeline. A modality's ground sampling
distance (GSD), spectral range/resolution, and standoff/perspective set a *hard bound* on
what is physically observable. That bound is a constraint, not a prescription: it says a
2–3 mm pistillate flower cannot be measured at 2 cm GSD by any model, and that kernel oil
content is unreachable from an RGB photo; it does not say "sensor X is the pipeline for
trait Y." crops.yml omits sensor/task/perspective assignments; do not
reintroduce a fixed sensor→trait or task→trait map. Within the feasible set a modality
permits, the CV approach for each trait is still derived and validated per trait against
real imagery and expert ground truth.

Two rules of thumb recur across the literature:

1. Resolution gate. To detect a structure, effective resolution (GSD or standoff)
   should be roughly 2–3× smaller than the structure. Below that it is physically
   unresolvable; report "not observable by this modality," do not manufacture a value.
   (This ratio, and the GSD/altitude figures below, are literature rules of thumb from
   other crops; confirm the threshold empirically on TCIP structures; see the last
   section.)
2. No modality covers all trait classes. Geometry/color needs calibrated imaging; 3D
   structure needs LiDAR or multi-view SfM; internal chemistry needs spectroscopy. Aerial
   and ground/lab are complementary, not interchangeable.

### Aerial (UAV / airborne): whole-population, top-of-canopy

- Drone RGB (nadir orthomosaic + SfM point cloud; typically ~0.6–1.6 cm/px at orchard
  survey altitude): captures top-of-canopy color, texture, geometry, plant footprint, and
  the 3D envelope. Detects large flowers/fruit *only* when GSD is ~2–3× smaller than the
  object. Blind to interior/under-canopy structure, organs below ~2× GSD, heavy occlusion,
  and all chemistry.
- Multispectral (few discrete bands: blue/green/red/red-edge/NIR): yields vegetation
  indices (NDVI, red-edge) that proxy canopy vigor/biomass/stress. An index is a proxy,
  never the biological trait; it must be validated against ground truth and never delivered
  as the quantity itself. Broad, spaced bands cannot resolve narrow disease spectral
  signatures.
- LiDAR / SfM 3D: can measure canopy height, crown area/volume, and cover of the outer
  canopy envelope directly (the class covering `plant_height`, `plant_max_height`,
  `plant_volume`, `plant_surface_area`, `plant_width_inrow`, `plant_width_betweenrow`, and
  biomass estimated allometrically from volume, `plant_biomass`). Top-of-canopy only:
  internal branch geometry and under-canopy organs are largely unrecoverable from above.
  Note: 3D point-cloud LiDAR/SfM traits are *not built* in the TCIP pipeline today;
  treat them as not-yet-in-pipeline, not available, when scoping.

### Ground / proximal (handheld, tripod, UGV): fine morphology at low throughput

The modality for small structures, fine morphology, and visible disease: hazelnut catkins,
2–3 mm pistillate flowers, on-plant nut clusters (`cluster_nut_count`), burrs (`n_burrs`,
`burrs_density`), and lesions/damage scored close-range (`efb_canker_length`,
`efb_presence`, `efb_damage`, `big_bud_mite_damage`, `weevil_damage`, `borer_damage`).
Enables tiled sliding-window detection of many small objects (compose
`run_inference(tile=True)`; see `phenology`). Constraints: occlusion, variable field
lighting, low area throughput, no chemistry. Frame order is not plant identity:
hedgerow frames span multiple or partial plants, so per-plant attribution needs the spatial
plant mapping (see `phenology`), never an every-Nth-frame rule.

### Lab / harvested-sample: dimensions and chemistry, destructively

- Lab RGB (controlled lighting with a scale reference): precise linear dimensions,
  counts, color, and morphometrics on harvested samples: `inshell_length`, `kernel_weight`,
  `fruit_diameter`, `fruit_color_rgb`. A photo without a scale reference cannot yield an
  absolute dimension; mass is not recoverable from an image.
- NIRS / spectroscopy (point spectrum, typically no image): the only optical route
  to internal chemistry: `kernel_perc_oil`, `kernel_oleic_acid_content`,
  `kernel_dry_matter_perc`, `fruit_anthocyanin_content`, `soluble_tannins`, `total_tannins`.
  Requires a crop- and state-specific calibration model; chemistry cannot be inferred from
  RGB pixels.
- Out of built scope: thermal IR (a water-stress proxy) and ground-penetrating radar
  (below-ground `root_crown_inrow_width` geometry) are edge-of-scope and not in the TCIP
  pipeline today; flag, do not assume.

## The field-imageable vs lab/destructive principle

Every trait falls on a spectrum from field-imageable (observable in situ from aerial or
ground imagery, non-destructively, across the whole population) to lab/destructive
(requires harvest, a calibrated bench, or a spectrometer). The split is a physical fact of
the trait, not a modeling preference: canopy geometry and visible disease are
field-imageable; internal chemistry (`kernel_perc_oil`) and precise mass are not, and
force- or sensory-based traits (`cluster_detachment_force`, `flavor_rating`) and destructive
sample counts (`nut_perc_blanks`) are not imaging traits at all; do not force them into a
pixel pipeline. The exact per-crop partition lives in each per-crop document; this skill only
fixes the principle. When a trait can't be validly measured from the available pixels, report
"not observable by this modality."

## The breeding-program workflow this serves

1. Collect at the right phenophase: GDD-informed scheduling, repeated visits bracketing
   any milestone, matched to the modality that can physically resolve the target structure.
2. Automate the measurement: derive and *validate* a CV measurement per trait within its
   modality's physical limits, against a reference sized to the trait: GT annotations, or a
   breeder-confirmed sample of the model's own outputs (review-confirmation), not dense GT for
   every trait (either reference passes the identical disjoint-split + count-bias gate). Operating
   points (conf, IoU-for-a-hit, NMS, tile size, thresholds) are derived from the data in hand at
   runtime, not pinned.
3. Deliver per-genotype / per-plant values: carry accession/genotype through so breeders
   select on genotype, not `plant_id`.

## Measurement-integrity guards specific to sensing

- Geometry needs a validated mask and physical scale. Area/length/width off a *validated*
  mask (calibrated to real-world scale) is a valid measurement; an *uncalibrated* box height or
  aspect ratio is scale-, zoom-, and pose-dependent, and geometry can't stand in for the visual
  call of a biological *state* (that's a validated classification). See CLAUDE.md.
- A vegetation index is a proxy, not the trait. Validate NDVI/red-edge against expert
  ground truth; never deliver the index as the biological quantity.
- Chemistry is not in RGB pixels. Any RGB-derived oil/protein/moisture/tannin number is
  fabricated; it needs NIRS/hyperspectral with a crop/state-specific calibration.
- Resolution gates feasibility. If standoff/GSD cannot resolve a structure at ~2–3×
  smaller than its size, no model recovers it; report not observable.
- Aerial 3D is top-of-canopy. Do not report internal-structure metrics from a DSM/SfM
  surface.
- Frame order is not plant identity in ground hedgerow imagery; use the spatial plant
  mapping.
- Sensor choice bounds but does not define the measurement. The trait's semantics come
  from the breeder; the operating points come from the data. Do not re-impose a fixed
  sensor→trait mapping that crops.yml intentionally omits.

## Needs expert confirmation

Literature-derived or program-specific; confirm before any of these drives a delivery;
they are not settled facts:

- The 2–3× resolution rule and specific GSD/altitude figures are rules of thumb from
  other crops (citrus, pear, berries, broadacre). The real resolvability threshold for each
  TCIP structure (catkins, 2–3 mm pistillate flowers, `efb_canker_length` lesions) must be
  established empirically on this program's own imagery and sensors.
- Ordinal disease/pest damage scales (`efb_damage`, `big_bud_mite_damage`,
  `weevil_damage`, `borer_damage`) are defined by the breeder's scoring rubric grounded in
  the imagery, not by any pixel measurement; confirm the rubric before operationalizing.
- GDD base/upper thresholds and phenophase calendars are genotype- and site-specific for
  the Savanna Institute plantings; confirm with the breeding team rather than porting generic
  tree-crop models.
- The often-repeated figure that GDD explains only around two-thirds of growing-season-onset
  variance on average is an uncited literature recollection with no source retained; treat it
  as unconfirmed, not a settled number, until a citation is found.
- NIRS chemistry accuracy figures are from published models on related nuts; a
  calibration must be built and validated on TCIP's own crop/matrix before any chemistry
  trait is delivered.
- Presymptomatic hyperspectral disease detection is unverified for TCIP pathogens (EFB,
  chestnut, currant); do not assume figures from other pathosystems transfer.
