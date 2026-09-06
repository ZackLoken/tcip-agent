---
name: currant
description: "Ribes spp.: currant and gooseberry (black currant Ribes nigrum; red/white/pink currant R. rubrum/sativum; gooseberry R. uva-crispa / R. hirtellum; jostaberry R. nidigrolaria; clove/golden and American currant). A multi-stem deciduous shrub bearing perfect (bisexual) flowers and berries in pendulous racemes called strigs, on thornless currant canes or thorny gooseberry canes; lobed maple-like leaves; overwintering buds along the canes. Load when phenotyping currant, working with currant imagery, or measuring currant traits: bloom, leaf, fruit and strig/berry, cane and bud architecture, plant-size, or disease/pest traits."
---

# Currant (Ribes spp.)

## Identity

Multi-stem deciduous shrub (a bush, not a tree), roughly 1–1.5 m (3–6 ft) tall
and wide, renewed from basal shoots into a framework of 9–12 canes of mixed age;
fruit is borne on 1-, 2-, and 3-year-old wood. Edible currants have thornless,
fairly erect canes; gooseberries arch and bear thorns. In this agroforestry
program the bushes grow in rows/hedgerows, so canopy is captured as in-row vs
between-row width and crown volume.

Reproductively the crop is not dioecious: flowers are perfect
(hermaphroditic) and borne in pendulous racemes ("strigs") near the base of
1-year-old wood and on spurs of older wood. There are no separate male catkins or
pistillate flowers here (unlike hazelnut/chestnut on this platform), so the
phenology target is a single bloom fraction, not two sexes. Most cultivars are
self-fruitful, but insect pollination (bumblebees) raises fruit set, berry size, and
seed count, and a few cultivars are self-incompatible and need a pollenizer. Bloom is
frost-sensitive; late-flowering types are valued for frost avoidance. The breeding
emphasis is black currant selected for white pine blister rust and powdery mildew
resistance, late-frost tolerance, high yield, and high sugar / low acid.

## Trait authority

crops.yml is the trait authority (86 currant traits). Verify every trait there;
never assert one it does not list. This skill does not reproduce the catalog.

## Field-imageable vs lab / destructive

The integrity-positive partition: which traits *can* come from imagery (with the
caveats below) versus those that cannot and require a lab assay, instrument,
sensory panel, or breeding record. Placement in "field-imageable" means a valid CV
path exists in principle; it does not waive validation or scale calibration.

Field-imageable, the CV surface:

- Phenology dates (derived, never single-frame): `bloom_05per_date`,
  `bloom_05per_julian`, `bloom_50per_date`, `bloom_50per_julian`, `bloom_95per_date`,
  `bloom_95per_julian`, `fruit_ripe_05per_date`, `fruit_ripe_05per_julian`,
  `fruit_ripe_50per_date`, `fruit_ripe_50per_julian`, `fruit_ripe_95per_date`,
  `fruit_ripe_95per_julian`, `leaf_out_05per_date`, `leaf_out_50per_date`,
  `leaf_senescence_95per_date`
- Bloom descriptors & counts: `bloom_color`, `bloom_sepal_color`, `bloom_ovary_color`,
  `bloom_length`, `bloom_per_axil`, `inflorescence_count`, `bloom_frost_tolerance`
- Fruit / berry: `fruit_color_descriptive`, `fruit_color_rgb`, `fruit_configuration`,
  `fruit_diameter`, `fruit_diameter_range`, `fruit_glossiness`, `fruit_per_strig`,
  `fruit_calyx_size`, `fruit_sun_scorch`, `fruit_swd_presence`
- Leaf: `leaf_color_green`, `leaf_color_fall`, `leaf_glossiness`, `leaf_length`,
  `leaf_width`, `leaf_length_width_ratio`, `leaf_blade_base`, `leaf_petiole_color`
- Disease / pest scores & photos: `leaf_spot_presence`, `leaf_spot_photo`,
  `plant_powderymildew_presence`, `plant_wpbr_presence`, `plant_wpbr_photo`,
  `plant_dieback_presence`
- Cane / bud architecture: `stem_bud_color`, `stem_bud_length`, `stem_bud_position`,
  `stem_bud_shape`, `stem_shoot_color`, `stem_thorniness`, `stem_internode_length`,
  `plant_basal_shoots`
- Plant size (metric only with calibration): `plant_growth_habit`, `plant_height`,
  `plant_height_width_ratio`, `plant_width_inrow`, `plant_width_betweenrow`,
  `plant_volume`

Lab or destructive, cannot come from field imagery:

- Juice chemistry: `fruit_juice_brix`, `fruit_juice_pH`, `fruit_juice_TA`,
  `fruit_anthocyanin_content`
- Weights & yield: `fruit_averageweight`, `fruit_batch_weight`, `plant_yield`,
  `leaf_25weight_fresh`, `leaf_25weight_dry`, `leaf_aveweight_fresh`,
  `leaf_aveweight_dry`
- Texture / sensory: `fruit_firmness`, `fruit_crispness`, `flavor_rating`,
  `fruit_flavor_description`, `leaf_thickness`
- Counts / assays: `fruit_seed_count`, `fruit_swd_larvae`, `fruit_storage_6week`
- Harvest efficiency: `fruit_hand_harvest`, `fruit_machine_harvest`
- Breeding records / controlled-cross: `parent_pollen`, `parent_seed`,
  `plant_selection_points`, `plant_selection_reason`, `plant_self_compatibility`

Caveats on the field-imageable set:

- Geometric traits (`fruit_diameter`, `fruit_diameter_range`, `bloom_length`,
  `leaf_length`, `leaf_width`, `leaf_length_width_ratio`, `stem_bud_length`,
  `stem_internode_length`) and plant-size traits (`plant_height`,
  `plant_height_width_ratio`, `plant_width_inrow`, `plant_width_betweenrow`,
  `plant_volume`) are pixel quantities until an in-frame scale / photogrammetric
  calibration makes them metric. Pixels are not millimeters.
- Percent-phenology families (`bloom_*`, `fruit_ripe_*`, `leaf_out_*`,
  `leaf_senescence_95per_date`) are imageable only as derived outputs of a *validated*
  per-frame call over a calibrated image time-series, not from a single frame or a raw
  detection count.
- `fruit_swd_presence` sits here only tentatively: late-stage berry collapse is
  visible, but early infestation is essentially invisible (see disease notes).
- `plant_yield` is intentionally in the lab/destructive column: a berry-count ×
  mean-weight estimate is a proxy that must be validated against weighed harvest, not
  a raw imagery measurement.

## Phenophase calendar → date traits

Timing is approximate and strongly region/year/cultivar-dependent (Upper Midwest
reference). The platform derives dates from data and never fixes them here. Bloom-
and ripe-fraction math (the 05/50/95 crossings) is owned by the phenology skill;
defer to it; do not re-derive.

| Phenophase | Approx. timing | crops.yml date traits |
|---|---|---|
| Leaf-out / budbreak | April | `leaf_out_05per_date`, `leaf_out_50per_date` |
| Bloom (frost-sensitive) | May | `bloom_05per_date` / `bloom_05per_julian`, `bloom_50per_date` / `bloom_50per_julian`, `bloom_95per_date` / `bloom_95per_julian` |
| Fruit ripening / harvest | late June–August (black currant ~July) | `fruit_ripe_05per_date` / `fruit_ripe_05per_julian`, `fruit_ripe_50per_date` / `fruit_ripe_50per_julian`, `fruit_ripe_95per_date` / `fruit_ripe_95per_julian` |
| Leaf senescence / fall | September–October | `leaf_senescence_95per_date` |

Two scoring windows carry no date trait but are the best time to capture their
traits: the dormant / bud-swell window (leafless canes; score bud and cane
architecture and `stem_thorniness` before foliage occludes them) and the
summer disease/pest window (June–September; foliar/fruit lesions and cane
dieback are scorable).

## Key structures & imagery appearance

- Bush / cane framework: a fan of arching-to-erect canes from the crown;
  currant canes thornless, gooseberry canes thorny; cane age (1–3 yr) governs bearing
  wood. Whole-plant, ~1–1.5 m.
- Strig (raceme / infructescence): slender pendulous stalk bearing a chain of
  flowers that become a chain of berries; ~10–15 cm, up to ~10–20 flowers / 8–30
  berries. The core detection/annotation unit for counting flowers and berries.
- Flower: small; black currant bell-shaped, pale yellow to greenish-white
  (~12 mm); red currant saucer-shaped, creamy to pinkish (~6 mm). Showy sepal
  lobes larger than the true petals; ovary inferior. Sepal/ovary anthocyanin is a
  scored color descriptor.
- Berry: round, pendulous, retaining a dried calyx at the tip; black currant
  matte brown-purple to black, red/white/pink translucent; gooseberry berries larger
  and borne singly or in 2–3s. Currant ~6–12 mm; ~3–12 small seeds each.
- Leaf: palmately lobed, maple-like, 3–5 toothed lobes; green intensity,
  glossiness, and petiole anthocyanin are scored; black currant foliage is aromatic.
  Blade a few cm to ~10 cm.
- Vegetative bud: overwintering bud at cane nodes; apex shape, position, length,
  and anthocyanin color are UPOV-style descriptors best scored on dormant/early canes.

## Diseases & pests

| Common name | Causal agent | crops.yml trait | Imagery appearance |
|---|---|---|---|
| White pine blister rust | *Cronartium ribicola* (Ribes is the alternate host) | `plant_wpbr_presence`, `plant_wpbr_photo` | Tiny chlorotic spots on the upper leaf surface; orange-yellow blister pustules and later brown hair-like columns on the underside; premature yellowing/drop. Underside close-up is informative. |
| Powdery (American gooseberry) mildew | *Podosphaera mors-uvae* | `plant_powderymildew_presence` | White-to-gray powdery/felty coating on young upper leaf surfaces, shoot tips, and, most damaging, berries; stunted distorted tips; a dried brown felty patch on fruit. |
| Currant leaf spot / anthracnose | *Drepanopeziza ribis* | `leaf_spot_presence`, `leaf_spot_photo` | Numerous small dark brown-to-black spots that enlarge and coalesce; leaves yellow and drop, defoliating from the bottom of the bush upward. |
| Spotted wing drosophila | *Drosophila suzukii* | `fruit_swd_presence`, `fruit_swd_larvae` | Early damage is only a pinprick oviposition scar, nearly invisible; within days the berry softens, sinks, weeps, and collapses. `fruit_swd_larvae` (≤3 mm white maggots) is a dissection/salt-flotation count, not a field-image trait. |
| Cane dieback | Multiple / ambiguous (currant borer *Synanthedon tipuliformis*, fungal canker, WPBR, winter injury) | `plant_dieback_presence` | Individual canes wilt or fail to leaf out while neighbors stay healthy; borer canes show a hollowed/tunneled pith when cut. Scored ordinally as extent across the bush. |

## Annotation challenges

Mechanics live in the annotation skill; currant-specific difficulties:

- Strig-level vs berry/flower-level: decide with the breeder whether the
  detection unit is the whole strig or the individual berry/flower; it changes both
  the annotation target and how a bloom/ripe fraction is computed.
- Small, clustered, overlapping flowers and berries in a pendulous chain, often
  self-occluding; ripe black currants are dark and low-contrast against foliage.
- Uneven ripening within a strig (top ripens first) blurs a single "ripe" call.
- Cryptic disease signal: early SWD is invisible externally; WPBR pustules are on
  the leaf underside; both demand the right view, not just any frame.
- An image with no open flowers on a date is a real observation, not noise, but it trains as
  a negative only once the breeder confirms it Complete; an empty label file alone reads as
  unannotated (see CLAUDE.md's negative invariant).

## Measurement integrity

Per the CLAUDE.md measurement-integrity invariant (never a geometric/pixel proxy; validate
against a reference sized to the trait: GT annotations, or a breeder-confirmed sample of the
model's own outputs (review-confirmation), before any result, not dense GT for every trait; see
the catkin-elongation cautionary tale there). Currant-specific traps:

- Never substitute a raw RGB color threshold for ripeness. `fruit_ripe_*` depends
  on a breeder-defined eating stage (color + firmness + separation); validate a
  ripeness call against expert-scored fruit before emitting any date.
- Never regress chemistry from pixels. A berry-color-to-brix, -pH, -TA, or
  -anthocyanin model is not a valid measurement
  (`fruit_juice_brix`, `fruit_juice_pH`, `fruit_juice_TA`,
  `fruit_anthocyanin_content` are lab assays).
- Disease ordinals map to the breeder's rating scale validated against expert
  scoring, not an arbitrary percent-lesion-area cutoff. WPBR lesion presence in a
  photo is not the same quantity as a cultivar's genetic resistance.
- `fruit_configuration` is a categorical descriptor; do not manufacture a numeric
  surrogate for it.

## Needs expert confirmation

- Species/cultivar scope of this program. Public sources confirm a black-currant-
  focused program; whether red/white currant, gooseberry, jostaberry, and clove/golden
  currant are actively included is inferred from the vocabulary (`stem_thorniness`
  implies gooseberry; `fruit_configuration` implies multiple infructescence types).
- All phenophase timings above are approximate and region/year/cultivar-dependent;
  dates must be derived from data, not the month labels.
- Causal agent behind `plant_dieback_presence` (currant borer vs fungal canker vs
  WPBR-induced dieback vs winter injury); confirm what is scored and whether it is
  agent-specific.
- Value set / rubric for `fruit_configuration` ("Infructescence Type"), likely a
  UPOV-style strig descriptor; exact categories unknown.
- Whether bloom percent-open and fruit-ripe percent are scored per-strig or
  per-flower/per-berry: this changes how the CV proportion is computed.
- Target organism for `leaf_spot_presence`: *Drepanopeziza ribis* specifically,
  or other leaf-spotting pathogens too.
- Whether `plant_powderymildew_presence` targets *Podosphaera mors-uvae*
  specifically vs powdery mildew broadly.
- Imageability of `fruit_swd_presence` from field RGB: early damage is cryptic;
  confirm the expectation (visible collapse vs salt-test-confirmed) before treating it
  as a detection target.
- Method behind `plant_self_compatibility` (controlled selfing trial vs pedigree
  inference); it is not an imagery trait.
- Dichogamy/herkogamy detail in the program's black currant germplasm (for
  pollination-window modeling); reported variably in the literature.