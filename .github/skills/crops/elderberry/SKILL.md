---
name: elderberry
description: "Botany, phenology, disease, and imaging reference for American elderberry (Sambucus nigra subsp. canadensis, syn. Sambucus canadensis; the European S. nigra ssp. nigra underperforms in the US Midwest), a multi-stemmed, sucker-forming deciduous shrub that bears perfect (hermaphroditic), insect-pollinated flowers in large flat-topped compound cymes on current-season shoots, opposite pinnately compound leaves, and dense clusters of tiny (~3-6 mm) green-to-purple-black drupes. Load when phenotyping elderberry, working with elderberry imagery, or measuring elderberry traits: it partitions field-imageable traits from lab/destructive chemistry, mass, count, and sensory ones, maps the leaf-out to bloom to fruit-ripening to senescence calendar onto crops.yml date traits, and describes disease/pest symptoms (spotted wing drosophila, shoot/cane borers, eriophyid mites, Japanese beetle, powdery mildew). crops.yml is the trait authority; elderberry bears no catkins, its inflorescence is the compound cyme."
---

# Elderberry: Sambucus nigra subsp. canadensis

> **Grounding note (measurement integrity).** The field/lab partition, phenophase timing, and
> disease agents below are grounded in cited elderberry literature (Prenger et al. 2026;
> University of Missouri / MU Extension; Alabama Extension; MU multi-site phenology) plus
> general *Sambucus* botany, **not** in the Savanna Institute program's own block records.
> Phenology dates lean on Missouri data and are region- and year-dependent; treat everything
> here as an informed default to confirm against the breeding block and derive from data before
> delivery (see **Needs expert confirmation**).

## Identity

**American elderberry** (*Sambucus nigra* subsp. *canadensis*, syn. *S. canadensis*) is a
**multi-stemmed, sucker-forming deciduous shrub** (not a tree), grown in hedgerow-like
breeding rows. In cultivation it is roughly **1.5-3.7 m tall and up to ~3 m wide**, a
clump/thicket of arching canes that arise as **basal shoots (suckers)** from a spreading
crown/rhizome. Young canes are green with prominent **white lenticels** and soft white pith;
older canes are gray-brown and woody. European *S. nigra* ssp. *nigra* is a related crop but
does not reliably perform in Midwest environments. Full production begins ~years 3-5.

American elderberry **flowers and fruits on current-season growth** (terminal cymes on new
shoots), so the block's **cane-management regime** (annual coppice/mow-to-ground vs retained
multi-year canes) strongly changes what cane and basal-shoot imagery shows and must be
confirmed.

**Reproductive biology (imaging-relevant).** Flowers are **perfect (hermaphroditic)**: every
tiny flower carries both stamens and pistil. There are **no separate male/female plants and no
catkins**, so the split staminate/pistillate phenology traits used for hazelnut/chestnut **do
not apply**: elderberry phenology is **whole-cyme bloom, then fruit ripening**. Flowers are
**insect-pollinated** (flies, small bees, beetles) on showy creamy-white cymes. Plants are
self-fertile / partially self-compatible; a single cultivar sets some fruit, but interplanting
two or more genotypes markedly improves fruit set, cyme fill, and yield (the basis of
`plant_self_compatibility` and of the standard practice of mixing cultivars). Some
flowering asynchrony (protandry/protogyny, within and among cymes) is reported; confirm
its degree with the breeder for any pollination-window modeling.

## Trait authority

**crops.yml is the trait authority (69 elderberry traits). Verify every trait there; never
assert one it does not list.** This skill partitions and contextualizes that catalog; it does
not reproduce it. Note some documented elderberry problems have **no** matching crops.yml trait
(elderberry rust, bacterial/fungal leaf spot, Phomopsis-type cane cankers, leaf-footed bug);
do not assume a trait exists for them (see expert flags).

## Field-imageable vs lab / destructive: the operational partition

The most important split for pipeline design: what a camera in the field can measure versus what
needs a bench, an instrument, a taster, destruction of the sample, or a written record. Grounded
in each trait's crops.yml definition. Trait names are verbatim.

### Field-imageable (RGB, ground / close-range / repeat-visit; several need calibration)

- **Phenophase (repeat imaging, dates)**: mapped in the calendar below: `leaf_out_05per_date`,
  `leaf_out_50per_date`, `bloom_05per_date`, `bloom_05per_julian`, `bloom_50per_date`,
  `bloom_50per_julian`, `bloom_95per_date`, `bloom_95per_julian`, `fruit_ripe_05per_date`,
  `fruit_ripe_05per_julian`, `fruit_ripe_50per_date`, `fruit_ripe_50per_julian`,
  `fruit_ripe_95per_date`, `fruit_ripe_95per_julian`, `leaf_senescence_95per_date`.
- **Whole-plant architecture / canopy (ground or aerial RGB; 3-D from photogrammetry):**
  `plant_height`, `plant_height_width_ratio`, `plant_width_inrow`, `plant_width_betweenrow`,
  `plant_volume`, `plant_basal_shoots`.
- **Cyme / inflorescence (close-range RGB; counts under occlusion are proxies, see integrity):**
  `inflorescence_count`, `cyme_diameter`, `cyme_orientation`, `fruit_per_cyme`.
- **Color / gloss (illumination- and white-balance-dependent; need an in-frame calibration
  card):** `bloom_color`, `fruit_color_descriptive`, `fruit_color_rgb`, `fruit_glossiness`,
  `leaf_color_green`, `leaf_color_fall`, `leaf_glossiness`, `stem_shoot_color`.
- **Leaf morphology (calibrated close-up or flatbed scan; needs a scale):** `leaf_length`,
  `leaf_width`, `leaf_length_width_ratio`.
- **Stem descriptor (close-up):** `stem_internode_length`.
- **Berry size (calibrated macro/close-up; resolution-gated, see the `crop-science` skill's
  resolution-gate guidance, effective GSD roughly 2-3x smaller than the ~3-6 mm berry):**
  `fruit_diameter`, `fruit_diameter_range`. Valid only from a *validated mask* with in-frame
  physical-scale calibration: an uncalibrated pixel length is not millimeters, and at this size
  the imaging setup needs to actually resolve the berry, not just calibrate whatever it captures.
- **Disease & abiotic damage (visible symptoms; each an ordinal from a validated model calibrated to the breeder's rubric):**
  `plant_borer`, `plant_eriophyid_mites`, `plant_jbeetle`, `plant_powderymildew_presence`,
  `bloom_frost_tolerance`.

### Cannot be measured from imagery (lab / instrument / destructive / sensory / record)

- **Chemistry & instrument (bench, not pixels):** `fruit_anthocyanin_content` (spectrophotometer),
  `fruit_juice_brix` (refractometer), `fruit_juice_pH` (pH meter), `fruit_juice_TA` (titration),
  `fruit_cyanide_content` (cyanogenic-glycoside assay), `fruit_firmness`. Do **not** infer any of
  these from berry color or darkness; they are not imageable quantities.
- **Mass / yield (kg or g on a scale):** `plant_yield`, `cyme_weight`, `fruit_averageweight`,
  `fruit_batch_weight`, `leaf_25weight_fresh`, `leaf_25weight_dry`, `leaf_aveweight_fresh`,
  `leaf_aveweight_dry`.
- **Destructive size / count:** `fruit_seed_count` (seeds are internal), `leaf_thickness`
  (caliper / cross-section).
- **Physical handling test:** `cyme_shatter_resistance` (berry retention under shaking),
  `fruit_machine_harvest` (mechanical-harvest handling trial).
- **Sensory:** `flavor_rating`, `fruit_flavor_description` (tasting).
- **Pest ground truth (internal to the berry):** `fruit_swd_presence`, `fruit_swd_larvae`;
  external softening/stings are a **weak** field-visible proxy; reliable presence and larval count
  need the destructive salt-flotation / dissection assay.
- **Trial / pedigree record (not a phenotype from an image or a single assay):**
  `plant_self_compatibility` (controlled-pollination trial), `parent_pollen`, `parent_seed`
  (recorded crosses).

## Phenophase calendar

Timing is **approximate, region- and year-dependent, and data-derived**: the windows below are
placeholders for expected order, not fixed values (Savanna Institute WI/IL/MN sites are cooler
and later than the Missouri data these lean on). Defer all bloom-/ripe-fraction crossing math to
the **`phenology`** skill; the fractions here are per-plant fractions of cymes opened / fruit
ripened / leaves unfurled or senesced, exactly its 05/50/95 pattern.

| Phase | Approx. window | crops.yml date traits |
|-------|----------------|-----------------------|
| Dormancy (bare canes) | Nov–Mar | (no date trait; best window to count canes / basal shoots) |
| Budbreak & leaf-out | ~Apr–early May (MO budbreak ~day 64) | `leaf_out_05per_date`, `leaf_out_50per_date` |
| Vegetative cane growth | Apr–Jun | (no date trait; image `plant_basal_shoots`, `stem_internode_length`, canopy size) |
| Bloom (cyme flowering) | late Jun–Jul (MO full bloom ~day 170) | `bloom_05per_date`/`bloom_05per_julian`, `bloom_50per_date`/`bloom_50per_julian`, `bloom_95per_date`/`bloom_95per_julian` |
| Fruit set & green-fruit | Jul–Aug | (no date trait; image `fruit_per_cyme`, `cyme_diameter`) |
| Fruit ripening / harvest | late Aug–Sept (MO peak harvest ~day 231) | `fruit_ripe_05per_date`/`fruit_ripe_05per_julian`, `fruit_ripe_50per_date`/`fruit_ripe_50per_julian`, `fruit_ripe_95per_date`/`fruit_ripe_95per_julian` |
| Leaf senescence & defoliation | Sept–Nov | `leaf_senescence_95per_date` |

Each `*_julian` trait is the **same** milestone as its matching `*_date` expressed as
days-from-January-1 (derived, not an independent observation); deliver it alongside its date.

## Key structures in imagery

- **Cane / stem**: arising as basal shoots from the crown; multi-stem arching clump, ~1.5-3.7 m.
  Young green with white lenticels, older gray-brown. Cane count/architecture is best seen in
  dormancy (`plant_basal_shoots`, `stem_internode_length`, `stem_shoot_color`).
- **Compound leaf**: opposite, pinnately compound with 5-11 (commonly 5-7) serrate lanceolate
  green leaflets; leaf ~15-30 cm (`leaf_length`, `leaf_width`, `leaf_color_green`,
  `leaf_color_fall`, `leaf_glossiness`).
- **Cyme (inflorescence → infructescence)**: large flat-topped to convex compound cyme,
  ~10-25+ cm across, terminal on current-season shoots; creamy-white and showy at bloom, then a
  dense green→purple-black berry head that nods under weight. Bears on `inflorescence_count`,
  `cyme_diameter`, `cyme_orientation`, and `fruit_per_cyme`.
- **Individual flower**: ~5-6 mm, 5 creamy/ivory-white petals, 5 pale-anthered stamens; opens
  roughly synchronously across a cyme (`bloom_color`).
- **Berry (drupe)**: small, ~3-6 mm; green when immature (camouflaged against foliage), ripening
  red then glossy purple-black with a faint waxy bloom; each holds 3-5 internal seeds. Very dense
  and mutually occluding within a cyme (`fruit_color_descriptive`, `fruit_color_rgb`,
  `fruit_glossiness`).

## Diseases & pests

Each existing trait → causal agent → appearance in imagery. Agent specifics are flagged for
confirmation (see expert flags).

| Trait | Causal agent | Appearance in imagery |
|-------|--------------|-----------------------|
| `plant_borer` | Elder shoot borer / spindle worm (*Achatodes zeae*, moth larva in young shoots) and/or elder borer beetle (*Desmocerus palliatus*, larva in canes) | Young shoots (~6-10 in) suddenly wilt, bend over, and blacken at the tip ("flagging"); round entry hole with frass; affected shoots fail to flower. Wilted flags and holes are field-imageable; larvae inside are not |
| `plant_jbeetle` | Japanese beetle (*Popillia japonica*, adult) | Highly imageable: metallic green-and-copper beetles cluster on foliage, cymes, and berries in the sunlit upper canopy; feeding **skeletonizes** leaves (lacy tissue, veins remaining) |
| `plant_eriophyid_mites` | Eriophyid mite (Eriophyidae; species on *Sambucus* poorly documented) | Mites are microscopic (not imageable); infer from symptoms: leaf russeting/bronzing, blistering or erineum patches, distorted growth. Symptom expression on *S. canadensis* is unconfirmed |
| `plant_powderymildew_presence` | Powdery mildew fungi (Erysiphaceae; species on *Sambucus* unconfirmed) | Diffuse white-to-gray powdery coating on (usually upper) leaf surfaces; advanced infection yellows/distorts leaves. Generally a minor issue, confirm economic relevance |
| `fruit_swd_presence`, `fruit_swd_larvae` | Spotted wing drosophila (*Drosophila suzukii*); female oviposits into ripening berries | Ripe berries soften, collapse, weep juice, show tiny oviposition stings, often with secondary mold. External appearance is a weak proxy; larvae are internal white maggots; ground truth needs destructive salt-flotation/dissection |
| `bloom_frost_tolerance` | Abiotic: frost damage to open bloom | Browned/blackened florets and poor set on affected cymes; best imaged during/after a frost event at bloom |

## Annotation challenges

Defer all mechanics (tools, formats, IoU, SAM) to the **`annotation`** skill. Elderberry-specific
difficulties:

- **Elderberry bears no catkins, strigs, racemes, or burrs** (those belong to other TCIP
  crops); its inflorescence/infructescence is the compound cyme. Do not import another crop's
  annotation vocabulary here.
- **Tiny, dense, mutually-occluding berries** (~3-6 mm) make exhaustive per-berry boxing
  occlusion-limited and inconsistent at typical standoff, and a visible per-berry count is a
  proxy for the true count rather than the count itself. What a trait counts, a cyme head or
  individual berries, is its `crops.yml` definition and the breeder's to confirm; note
  `fruit_per_cyme` is a per-berry quantity. How that count is obtained from pixels is yours to
  derive.
- **Low contrast at fruit set**: green immature berries and green cymes blend into foliage; ripe
  purple-black berries contrast strongly, so favor ripe-stage imaging for fruit traits.
- **No separate male/female flowers**: there is nothing to split into staminate vs pistillate;
  one bloom class.
- **Color / gloss traits need an in-frame calibration target** (`fruit_color_rgb`, `bloom_color`,
  `leaf_color_green`, the gloss traits); without it, color is not a valid measurement.
- **An image with no cyme / fruit / symptom is a real observation**; never drop it. It trains as
  a negative only once the breeder confirms it Complete; an empty label file alone reads as
  unannotated (see CLAUDE.md's negative invariant).

## Measurement integrity (highest rule)

Per the **CLAUDE.md** measurement-integrity invariant (never a geometric/pixel proxy; validate
against a reference sized to the trait: GT annotations, or a breeder-confirmed sample of the
model's own outputs (review-confirmation), before any result, not dense GT for every trait; see
the catkin-elongation cautionary tale there). Elderberry-specific traps:

- `cyme_shatter_resistance` is a physical **shake/retention test**, not something a bounding box,
  geometry, or still-image count can measure; do not manufacture a proxy from a static image.
- **Plant-size traits** (`plant_height`, `plant_height_width_ratio`, `plant_width_inrow`,
  `plant_width_betweenrow`) are pixel quantities until an in-frame scale / photogrammetric
  calibration makes them metric. Pixels are not millimeters.
- The fruit-ripening dates (`fruit_ripe_50per_date` and its siblings), "ripened to eating stage,"
  are a **breeder-defined threshold** (color **plus** soluble solids), not "the berry looks dark";
  a pixel-color threshold is a proxy that must be validated against expert scoring before any
  milestone or curve.
- Bloom / leaf-out / senescence dates require an **expert-defined** opened/unfurled/senesced state;
  derive the operating threshold from data and validate it; do not hard-code a guessed pixel rule.
- Mass (`plant_yield`, `cyme_weight`, `fruit_averageweight`, `fruit_batch_weight`) cannot be read
  from pixels; berry-count × assumed-mass is a proxy that must be calibrated and validated.
- Counts under heavy occlusion (`inflorescence_count`, `fruit_per_cyme`) are visible-count proxies
  for the true count and must be calibrated/validated; seeds (`fruit_seed_count`) are internal and
  not imageable at all.
- Berry size (`fruit_diameter`, `fruit_diameter_range`) at ~3-6 mm needs a *validated mask* with
  in-frame physical-scale calibration, not scale alone: a pixel length without a scale is invalid
  science, and a box/count without a validated segmentation is not a size measurement. The imaging
  setup must also actually resolve the berry at this size (see the `crop-science` skill's
  resolution-gate guidance) before calibration is even the binding constraint.
- Chemistry (`fruit_anthocyanin_content`, `fruit_juice_brix`, `fruit_juice_pH`, `fruit_juice_TA`,
  `fruit_cyanide_content`) is not readable from RGB; color may correlate but is not the assay.

## Needs expert confirmation

1. **Upper-Midwest phenology dates.** Timings here lean on Missouri multi-site data (budbreak
   ~day 64, full bloom ~day 170, peak harvest ~day 231); Savanna Institute sites are cooler and
   later. All dates are region- and year-dependent and must be **derived from data, not fixed**.
2. **Which cultivars/genotypes are in the breeding blocks.** Documented Midwest germplasm includes
   Bob Gordon, Wyldewood, Ranch, Ozark, York, Ocoee, Pocahontas, Adams, Johns (*S. canadensis*)
   and Marge (*S. nigra*), but the specific accessions in-program are unconfirmed.
3. **Cane-management regime**: annual coppice/mow-to-ground (primocane fruiting) vs retained
   multi-year canes, which changes `plant_basal_shoots`, `stem_internode_length`, and canopy
   architecture imagery.
4. **`plant_borer` target agent**: elder shoot borer (*Achatodes zeae*, young shoots) vs elder
   borer beetle (*Desmocerus palliatus*, canes) vs both; confirm the intended causal agent.
5. **`plant_eriophyid_mites`**: the exact eriophyid species on *S. canadensis* and its symptom
   expression (russeting vs blistering vs bud distortion) are poorly documented.
6. **`plant_powderymildew_presence`**: causal species on *Sambucus* and whether it is
   economically relevant in the Upper Midwest (may be minor).
7. **Ordinal / test rubrics**: the measurement protocols defining scores for
   `cyme_shatter_resistance`, `fruit_machine_harvest`, `cyme_orientation`, `fruit_glossiness`, and
   `bloom_frost_tolerance`.
8. **Ripeness definition**: the breeder's "eating stage" for the fruit-ripening date traits
   (color threshold vs soluble-solids threshold).
9. **`fruit_cyanide_content`**: confirm the assay and whether it is measured on fresh berries,
   seeds, or processed pulp (cyanogenic glycosides concentrate in seeds/unripe tissue).
10. **Non-phenotype traits**: `parent_pollen` and `parent_seed` are pedigree **metadata**
    (recorded crosses), placed on the non-imageable side only to complete the partition.
11. **Vocabulary gaps**: documented elderberry problems with **no** matching crops.yml trait
    (elderberry rust *Puccinia sambuci/bolleyana*, bacterial/fungal leaf spot, leaf-footed bug,
    Phomopsis-type cane cankers); confirm whether any should map to overall damage tracking rather
    than assuming a trait exists.

## Sources

- Prenger et al. (2026). Developing an understanding of American elderberry (*Sambucus nigra*
  subsp. *canadensis* (L.) Bolli) to support breeding efforts. *Crop Science* 66:e70224.
- University of Missouri Center for Agroforestry / MU Extension, Growing and Marketing
  Elderberries in Missouri (Agroforestry in Action, AF1017).
- Alabama Cooperative Extension System, American Elderberry: Commercial Production Guide
  (ANR-3083).
- Elderberry genotype evaluation and phenology across diverse Missouri environments (budbreak
  ~day 64, full bloom ~day 170, peak harvest ~day 231), NIH PMC4858345.
- University of Missouri IPM, Spindle Worms in Elderberry Shoots (elder shoot borer,
  *Achatodes zeae*).
- Virginia Tech Fruit Entomology, Elderberry borers (*Achatodes zeae*; *Desmocerus palliatus*).
- Midwest Elderberry Cooperative, Cultivation and Midwest cultivars.
- Savanna Institute, Tree Crop Improvement (elderberry among priority agroforestry crops).
