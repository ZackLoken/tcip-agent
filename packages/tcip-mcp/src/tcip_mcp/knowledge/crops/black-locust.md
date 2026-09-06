---
name: black-locust
description: "Domain knowledge for phenotyping black locust (Robinia pseudoacacia L., Fabaceae), a fast-growing, clonal, nitrogen-fixing North American hardwood grown for rot-resistant timber, fodder, and pollinator forage. Its imageable structures are pendant white papilionaceous flower racemes, flat legume pods, paired stipular thorns, deeply furrowed bark, and root-sucker thickets. Flowers are perfect (bisexual), so there is a single bloom event, no separate male/female phenology. Covers crop identity and reproductive biology, the field-imageable versus lab/destructive trait split, the bloom and pod phenophase calendar, the locust borer pest, and annotation pitfalls from clonal suckering. Load this when phenotyping black-locust, working with black-locust imagery, or measuring black-locust traits."
---

# Black locust: Robinia pseudoacacia L. (Fabaceae)

## Identity

A fast-growing, deciduous North American hardwood, typically 12–18 m tall with
a 30–76 cm trunk. Open-grown trees form a short bole dividing into stout branches
with an irregular, often one-sided crown; mature bark is deeply furrowed, thick,
scaly, dark brown-gray. Young stems are smooth and thorny (paired stipular
spines at leaf bases). It is strongly clonal: vigorous root suckers and stump
sprouts form dense multi-stem thickets (root sprouting begins ~4–5 yr; ~4 stems
per rootstalk in the wild), and nitrogen-fixing via *Rhizobium* root nodules.
In Upper-Midwest agroforestry it is grown for rot-resistant timber/fenceposts,
livestock fodder, biomass, and pollinator nectar, managed toward a straight
single-leader timber form on short rotations (borer pressure limits U.S. stand
size). So `plant_growth_habit` spans a real continuum from single-stem upright
timber form to spreading, multi-stem, suckering thicket.

Reproductive biology relevant to imaging:

- Flowers are perfect (bisexual): papilionaceous (pea-type), borne in showy,
  pendant axillary racemes near new-shoot tips. The species is hermaphroditic, not
  monoecious or dioecious. The vocabulary gives black locust a single
  bloom trait (`bloom_50per_date`) and no separate staminate/pistillate or catkin
  phenology (unlike hazelnut): there is one bloom event to detect.
- Insect-pollinated (chiefly bees; a major honey plant). Wind pollination is
  not the mechanism.
- Late-leafing: flowers open *after* leaf emergence, and bloom is brief
  (~1–2 weeks).
- Fruit is a flat, dehiscent legume pod (~7–10 cm) holding ~4–8 hard-coated,
  bean-like seeds (3–5 mm); pods open on the tree and shed seed near the parent.
- Normally diploid (2n=20; some reports 2n=22). Colchicine-induced / European
  timber-breeding tetraploid selections exist, hence `ploidy` is tracked.

Exact program germplasm and mating-system specifics are not asserted here; see
"Needs expert confirmation."

## Trait authority

crops.yml is the trait authority (10 black-locust traits). Verify every trait
there; never assert one it does not list. This skill partitions those traits by how
they can (or cannot) be measured; it does not reproduce the catalog.

## Field-imageable vs lab / destructive

The operationally critical split: which traits can be measured from pixels of a
living plant, and which cannot come from imagery at all. The imaging modality
(sensor, task) is derived and validated per trait, not fixed here.

### Field-imageable: on the living plant

- Bloom phenology (flowering-canopy time series): `bloom_50per_date`. Defer the
  bloom-fraction math to the `phenology` skill.
- Pest cues (external only): `borer_damage`, external signs of the locust
  borer (frass, entry/exit holes, bark scarring, adult beetle). Only the *external*
  face is imageable; internal tunneling severity is not (see below).
- Growth / habit: `plant_growth_habit` (nominal class), `thorns` (ordinal),
  `suckers` (ordinal), `seeds` (binary, inferred from pod presence).
- Structural size, but requires metric-calibrated 3D that TCIP does not yet
  build. CLAUDE.md scope today is 2D imagery, object detection first;
  LiDAR/SfM point clouds are not built. So `plant_height` (m) and `dbh` (m, at
  1.37 m) are field-*sensible* in principle but are not deliverable from current
  2D pipelines: a raw pixel height or width is not meters (see Measurement
  integrity).

### Cannot be measured from imagery: lab / destructive

No valid pixel measurement exists for these; a number regressed from a photo would
be a proxy, not a measurement:

- `bark_thickness` (mm): external RGB shows bark furrow/texture, not mm thickness.
  Needs a bark gauge or destructive coring.
- `ploidy`: cannot be inferred from any pixel or geometric feature. Tetraploids
  trend toward larger leaves/vigor, but leaf size is not ploidy; a valid value
  requires flow cytometry / cytology.

## Phenophase calendar

Timing is approximate, region- and year-dependent, and data-derived: the
windows below are Upper-Midwest orienting ranges, not thresholds. The milestone
date comes from the observed time series, never a fixed calendar. Only one
crops.yml date trait exists for this crop; do not invent dates for phases with no
trait.

| Phase | Approx window | crops.yml date trait |
|-------|---------------|----------------------|
| Dormancy | Nov–Mar | *no trait; best window for pure structural imaging (bare crown)* |
| Budbreak / leaf-out | late Apr–May (late-leafing) | *no black-locust leaf-out trait in crops.yml* |
| Bloom | late May–June (~early June in MN); brief ~1–2 wk | `bloom_50per_date` |
| Pod development & seed set | Jun–Sep (pods green → dark brown) | *relates to `seeds` (presence), not a date trait* |
| Leaf senescence / drop | Sep–Oct (relatively early) | *no black-locust senescence trait in crops.yml* |
| Seed dispersal / pod persistence | Sep–Apr | *no trait; lagging cue only* |

Bloom is brief, so temporal sampling cadence matters: a coarse capture
interval can straddle the whole event. The bloom-fraction definition and the
tooling to compute the crossing live in the `phenology` skill; do not restate
or re-script them here.

## Key structures in imagery

- Raceme (inflorescence): pendant from leaf axils near shoot tips throughout
  the canopy; ~10–14 cm long. Showy drooping cluster of white (cultivars
  pink/purple, e.g. 'Purple Robe') fragrant pea-type flowers, conspicuous against
  blue-green compound foliage. Brief bloom window.
- Legume pod (fruit): pendant in clusters; ~7–10 cm, flat, smooth, thin;
  green when developing, ripening reddish- to dark brown; persists into winter.
  Basis for inferring `seeds`.
- Seeds: 3–5 mm, dark, hard-coated, bean-shaped; enclosed in pods and not
  externally visible until dehiscence; direct seed imaging is largely
  destructive/close-range.
- Stipular thorns (spines): paired at the base of leaf petioles on young
  stems; a few mm to ~1–2 cm; reduced/absent on old bark and in thornless
  cultivars. Basis for `thorns`; needs close-up stem imagery to score.
- Trunk / bole and bark: deeply furrowed, thick, scaly, dark brown-gray at
  maturity (smooth brown when young). Supports `plant_height`, `dbh`, and
  (destructively) `bark_thickness`.
- Root suckers / basal sprouts: clusters of young upright shoots around a
  tree, forming dense clones. Basis for `suckers`; confounds single-tree
  delineation in aerial imagery.
- Compound leaf: pinnately compound, 7–19 blue-green leaflets; a canopy
  texture cue for the leaf-out/senescence phenophases (which have no black-locust
  date trait).

## Diseases & pests

- `borer_damage`: locust borer, *Megacyllene robiniae* (Coleoptera:
  Cerambycidae, a longhorned beetle). Imagery appearance: pale sawdust-like
  frass at the tree base and around round entry/exit holes; wet spots /
  oozing sap in spring; swollen, cracked, or scarred bark; girdling and branch
  dieback; in severe cases honeycombed, wind-broken stems. The adult beetle
  (~12–25 mm) is a slender jet-black longhorn with bright-yellow transverse bands
  (a distinctive W-shaped band across the wing covers) and reddish legs, seen
  feeding on goldenrod in September. Damage is worst on young/stressed trees.
  Borer holes are entry points for heart-rot fungi (*Phellinus*/*Polyporus*), a
  real biological consequence, but heart rot is not a crops.yml trait for black
  locust and must not be asserted as one.

## Annotation challenges

Mechanics live in the `annotation` skill; the crop-specific hard parts:

- Clonal thicket / root suckers: suckers and stump sprouts merge a single
  genet into a multi-stem thicket, making per-tree boundaries genuinely ambiguous
  in aerial imagery. Plant identity comes from the spatial plant mapping, not frame
  order (see `phenology`).
- Tiny paired thorns: a few mm; require close-up stem imagery, easily missed
  at canopy scale; absent on old bark and thornless cultivars.
- Enclosed seeds: `seeds` is inferred from pod presence, but pods hide their
  contents until dehiscence and empty/aborted pods occur; pod presence does not
  guarantee viable seed.
- Brief, occluded bloom: racemes are conspicuous but short-lived and
  intermingled with the leafed-out canopy (a late-leafing species blooms after
  leaf-out), so partial occlusion is expected.

## Measurement integrity

Per the CLAUDE.md measurement-integrity invariant (validate against a reference sized to the
trait: GT annotations, or a breeder-confirmed sample of the model's own outputs (review-confirmation),
not dense GT for every trait; geometry needs a validated mask + physical scale). Black-locust-specific
traps:

- `dbh` / `plant_height`: an uncalibrated pixel width or bounding-box height is
  not a metric diameter or height. These need calibrated 3D (LiDAR/SfM with
  ground control); report no meters without metric scale.
- `borer_damage`: the trait is an ordinal severity of largely internal wood
  damage. Counting external holes or measuring frass area is a proxy that misses
  internal tunneling and is confounded by secondary heart rot; do not equate a
  hole/frass count with the breeder's ordinal rating without validation.
- `plant_growth_habit`: a nominal class in the breeder's vocabulary; an
  aspect-ratio or height/width proxy is not the growth-habit category.
- `bloom_50per_date`: derive the bloom-detection operating point from data and
  validate against expert phenology scoring; do not assume a fixed
  fraction-of-white-pixels threshold.
- `ploidy` / `bark_thickness`: no image-derived value is valid (see above); a
  regressed number would be fabricated.

## Needs expert confirmation

- `seeds` meaning: viable seed set on the tree vs. presence of pods vs. a
  selection-level attribute; confirm the definition before scoring.
- `plant_growth_habit` vocabulary: which nominal classes (upright / spreading
  / weeping? single-stem vs. multi-stem / thicket?); not enumerated in crops.yml.
- Ordinal scale anchors for `borer_damage`, `thorns`, and `suckers` (level
  definitions); not specified in crops.yml.
- Mating system: degree of self-compatibility vs. obligate outcrossing, and
  presence/absence of protandry or dichogamy in *Robinia pseudoacacia* (literature
  is ambiguous).
- Program germplasm: which seedling populations vs. named timber selections
  are grown, and whether thornless selections are present; unconfirmed.
- Tetraploid material: whether it is present and tracked in the program (the
  `ploidy` trait implies it is measured).
- `dbh` modality: whether it is intended as an image/LiDAR-derived structural
  measurement or a physical tape/caliper ground-truth value.
- Phenophase timing: Upper-Midwest bloom timing and all phase windows are
  region/year-dependent approximations; derive per-dataset, never fix.
- `borer_damage` CV framing: object detection (beetle/holes) vs. ordinal
  classification is a CV-design choice to derive and validate per data, not fixed.

## Sources

- USDA Forest Service, Silvics of North America: Black Locust (*Robinia pseudoacacia*). https://research.fs.usda.gov/silvics/black-locust
- USDA Forest Service, Fire Effects Information System (FEIS): *Robinia pseudoacacia*. https://www.fs.usda.gov/database/feis/plants/tree/robpse/all.html
- Savanna Institute, Black Locust program page. https://www.savannainstitute.org/blacklocust/
- The Morton Arboretum, Locust Borer (*Megacyllene robiniae*). https://mortonarb.org/plant-and-protect/tree-plant-care/plant-care-resources/locust-borer/
- Missouri Department of Conservation, Locust Borer field guide. https://mdc.mo.gov/discover-nature/field-guide/locust-borer
- Bugwood Wiki (University of Georgia), Locust Borer (HPIPM). https://wiki.bugwood.org/HPIPM:Locust_Borer
- USDA NRCS Plant Guide, Black Locust (*Robinia pseudoacacia*), symbol ROPS. https://plants.sc.egov.usda.gov/DocumentLibrary/plantguide/pdf/pg_ROPS.pdf
- iForest (2023), Breeding and improvement of black locust with a special focus on Hungary: a review. https://iforest.sisef.org/contents/?id=ifor4254-016
