---
name: persimmon
description: "Diospyros virginiana (American persimmon; cold-hardy 90-chromosome hexaploid cultivars) plus flagged-unconfirmed D. kaki / interspecific hybrids — a functionally dioecious single-trunked deciduous tree with distinctive blocky 'alligator' bark, small inconspicuous cream/greenish flowers, and orange globular berries each subtended by a persistent 4-lobed calyx. Load this skill when phenotyping persimmon, working with persimmon imagery, or measuring persimmon traits (bloom/ripening dates, cropload, fruit drop, calyx retention, fruit size, tree sex, plant height, DBH, disease, and the destructive fruit / tannin / quality samples)."
---

# Persimmon (Diospyros)

## Identity

**Species.** The Savanna Institute Upper-Midwest program is built on cold-hardy named
cultivars of the northern **90-chromosome (hexaploid) race of _Diospyros virginiana_**
(American / common persimmon). Pure Asian persimmon (_D. kaki_) and interspecific
_D. virginiana × D. kaki_ hybrids exist commercially but are generally too cold-tender /
long-season for the region — treat them as not present unless the breeder confirms (see
"Needs expert confirmation").

**Growth form.** Single-trunked, slow-growing deciduous tree (30–80 ft tall, 20–35 ft wide
at maturity; managed shorter in breeding orchards / silvopasture). Distinctive thick
dark-grey blocky "alligator" / checkerboard bark. Prone to **root suckering** — a mother
tree can carry a multi-stem clonal clump, so a "plant" in imagery may be a clonal cluster,
not one stem. Grown as an orchard / silvopasture tree, not a coppiced hedgerow shrub.

**Reproductive biology (imaging-relevant).** Functionally **dioecious**: male and female
flowers are borne on separate trees. A few trees are monoecious or perfect-flowered, and a
tree's expressed sex can shift year to year, so sex cannot be assumed fixed. Insect-
pollinated (bees), not wind-pollinated. Critically, the 90-chromosome race sets fruit
**parthenocarpically** — female trees produce full, essentially seedless fruit with no
pollination — so fruiting marks a female tree, but seed presence/absence is a genetic /
ploidy signal rather than a pollination readout.

## Trait authority

**crops.yml is the trait authority (20 persimmon traits). Verify every trait there; never
assert one it does not list.** This skill grounds the imagery; it does not redefine the
vocabulary. Persimmon is the sparsest TCIP crop and carries only the single aggregate
`overall_disease` trait — no species-specific disease columns.

## Field-imageable vs lab / destructive

This partition replaces any fixed sensor→trait table. Which traits can come from pixels at
all is a science fact, not a config choice.

**Field-imageable** (measurable from imagery, given valid derivation and validation):
`bloom_50per_date`, `fruit_ripe_50per_date`, `cropload`, `fruit_drop`, `calyx`,
`fruit_diameter`, `fruit_height`, `sex`, `plant_height`, `dbh`, `overall_disease`,
`fruit_set`.

- `fruit_diameter` and `fruit_height` are imageable **only** with a real in-frame scale
  reference and an unoccluded fruit silhouette (typically a lab-bench RGB shot, not a
  canopy photo). `fruit_height` is the axial (apex-to-base) dimension, distinct from the
  equatorial `fruit_diameter` — the exact caliper endpoints are the breeder's protocol.
- `fruit_set` is a flower→fruit conversion ratio needing paired counts over time (see
  integrity note); it may in practice be a hand-count field observation — confirm.

**Lab / destructive** (cannot come from external imagery — internal, or requires a cut /
assayed fruit subsample): `seeds`, `ploidy`, `astringency`, `flavor_rating`,
`soluble_tannins`, `total_tannins`, `fruitsample_n`, `fruitsample_weight`. Seeds are
internal (visible only in a cut fruit); tannins are spectrophotometric assays on pulp; the
sensory and quality traits ride on a destructive fruit subsample.

## Phenophase calendar

Timing is **approximate, region- and year-dependent** — the platform derives phenophase
dates from the data in hand and must not freeze these placeholders. The two date traits are
50%-milestone dates; **defer the bloom-fraction / crossing math to the `phenology` skill.**

| Phenophase | Approx. (Upper Midwest) | Imagery | crops.yml date/other traits |
|---|---|---|---|
| Dormancy (leaf-off) | Dec–Mar | Bare blocky-barked branches; on some cultivars fruit persists into winter | `plant_height`, `dbh` (structural, not dates) |
| Budbreak / leaf-out | Apr–mid May | Glossy dark-green oblong leaves unfurl (late-leafing) | no persimmon date trait exists for this phase |
| Bloom | late May–June | Small inconspicuous cream/greenish flowers, easily hidden in foliage; sex is most directly read here | `bloom_50per_date`, `sex`, `fruit_set` |
| Fruit development (green) | June–Sept | Green hard astringent fruit enlarging, each with an enlarging 4-lobed green calyx; heavy occlusion | `cropload`, `fruit_set` |
| Ripening / harvest | Sept–Nov, often post-frost | Fruit yellow → orange → deep orange/red and softens; window for sizing, cropload, drop, and destructive sampling | `fruit_ripe_50per_date`, `cropload`, `fruit_drop`, `calyx`, `fruit_diameter`, `fruit_height`, `astringency`, `flavor_rating`, `seeds`, `soluble_tannins`, `total_tannins`, `fruitsample_n`, `fruitsample_weight` |
| Leaf senescence / drop | Oct–Nov | Leaves yellow to reddish-purple then drop; **fruit persists on a near-bare canopy** — a valuable low-occlusion window for cropload / fruit_drop | `fruit_drop`, `cropload` |

## Key structures + imagery appearance

- **Trunk / bark** — thick dark grey-black bark furrowed into small square blocks
  ("alligator" pattern); a strong species-ID cue and the reference for locating the DBH
  plane (4.5 ft / 1.37 m; mature trunks ~15–45 cm across). Main stem, ground to canopy.
- **Leaf** — simple, alternate, broadly oblong/ovate, pointed, glossy dark green above,
  paler below; fall color yellow to reddish-purple. 5–15 cm long. Throughout canopy.
- **Male flower** — small tubular cream/greenish-yellow flowers in clusters of 2–3, ~8–13
  mm; only on male trees; never develop into fruit. Leaf axils on current-season shoots.
- **Female flower** — solitary, larger (~15–20 mm), urn- / bell-shaped cream/greenish with
  4 recurved petals and a prominent ovary; on female trees; develops into fruit
  (parthenocarpically in 90-chromosome cultivars). Solitary in leaf axils.
- **Fruit (berry)** — globular to slightly flattened fleshy berry, ~2.5–4 cm diameter
  (smaller than _D. kaki_); green → yellow → pale orange → deep orange/red and soft at
  ripeness; often a glaucous waxy bloom; small persistent style "beak" at the apex. One
  fruit per pedicel; persists after leaf drop.
- **Calyx** — persistent, 4-lobed, leaf-like green (browning at senescence) structure
  clasping the base of every fruit (opposite the beak); its retention is the `calyx` trait.
- **Seed** — flattened, oblong, glossy dark-brown, ~1.5–2 cm; absent or few in
  parthenocarpic cultivar fruit; **internal — only visible in a cut fruit (destructive).**

## Diseases / pests

All persimmon disease/pest pressure maps to the single aggregate `overall_disease` rating —
there are no species-specific disease traits in crops.yml.

- **Persimmon wilt (Cephalosporium wilt)** — _Nalanthamala diospyri_ (syn. _Acremonium /
  Cephalosporium diospyri_), a vascular wilt fungus and the most serious disease of
  _D. virginiana_. Imagery: sudden wilting, top-down defoliation, branch dieback, standing
  dead trees. The diagnostic black vascular streaking is **internal and not field-
  imageable** (requires cutting a stem); external imagery shows only canopy collapse.
- **Persimmon leaf spot** — fungal complex (_Cercospora_ / _Colletotrichum_ /
  _Pseudocercospora_ spp.). Dark brown-to-black angular/rounded leaf spots, sometimes
  coalescing, causing premature (Aug–Sept) defoliation. Ground / close-range RGB.
- **Anthracnose** — _Colletotrichum_ spp. Dark sunken lesions on leaves and shoots; black
  irregular sunken spots on fruit. Foliar and fruit symptoms are ground-RGB imageable.
- **Caterpillar / webworm defoliators** (incl. fall webworm, _Hyphantria cunea_) — silken
  webbing / tents over branch tips with skeletonized leaves and localized defoliation;
  visible as pale silk masses in the canopy.
- **Persimmon psyllid** — _Trioza diospyri_. Curling, rolling, puckering of new terminal
  leaves; distorted shoot tips. Lower-confidence for this region.

## Annotation challenges

Defer annotation mechanics and label / format I/O to the `annotation` skill; the crop-
specific difficulties are:

- **Canopy occlusion** — dense foliage hides much of the fruit, so a visible-fruit count
  undercounts. The **leaf-off / post-senescence window** (fruit persists on a bare canopy)
  is far less occluded — prefer it for cropload and drop.
- **Small, inconspicuous flowers**, and male vs female flowers differ (clustered tubular vs
  solitary urn-shaped), making bloom and sex harder than for showy-flowered crops.
- **Clonal sucker clumps** — a mother tree plus root suckers can read as one multi-stem
  "plant"; be explicit about the plant unit before mapping images to plants.
- **Calyx clasps the fruit base** and the style beak sits opposite it — the two are easily
  confused at low resolution.

## Measurement integrity

Per the **CLAUDE.md** measurement-integrity invariant (never a geometric/pixel proxy; validate
against expert-scored ground truth before any result — see the catkin-elongation cautionary tale
there). Persimmon-specific traps:

- `fruit_ripe_50per_date` — American persimmon loses astringency only on **softening**
  (often post-frost), which **lags surface color**. An orange-pixel threshold calls hard,
  astringent fruit "ripe." Validate ripeness against expert-scored eating stage.
- `fruit_set` — a flower→fruit conversion **ratio** needing paired flower-at-bloom and
  fruit-at-set counts on the **same** trees over time; a single-date fruit count is not
  fruit set, and parthenocarpy further decouples it from pollination.
- `cropload` / `fruit_drop` — a visible-fruit bbox count is not total production
  (detectability must be modeled); `fruit_drop` requires partitioning on-tree vs
  ground / dropped fruit, which a bbox count does not give.
- `calyx` — a breeder-defined ordinal **retention** rating; a calyx bounding box or
  presence flag is not the retention score.
- `sex` — fruiting identifies a female tree, but **absence of fruit does not prove male**
  (could be an unpollinated / immature / off-year female or a monoecious tree); do not infer
  sex from fruit presence without flower-level or multi-year evidence.
- `fruit_diameter` / `fruit_height` — valid in mm only with a real in-frame scale reference
  and an unoccluded silhouette; raw bbox pixels are not millimetres.
- `dbh` — a pixel trunk width is not DBH without calibration and correct localization of the
  4.5 ft breast-height plane.
- `overall_disease` — an aggregate whole-tree severity spanning etiologies that look nothing
  alike (wilt canopy-collapse vs foliar leaf spot vs insect defoliation), whose worst
  disease has an internal-only diagnostic sign; a single lesion-area proxy is invalid.
- `bloom_50per_date` — flowers are small and inconspicuous and differ by sex; "50% bloom"
  must use the breeder's definition, not a canopy-greenness or generic-flower proxy.

## Needs expert confirmation

- Exact cultivar list in the Upper-Midwest planting. Documented 90-chromosome hexaploid
  _D. virginiana_ cultivars include Prairie Star, Prairie Sun, Mohler, Early Golden,
  Garretson, Meader, and selection 100-46, but which are actually in the program is
  unconfirmed.
- Whether any _D. kaki_ or interspecific hybrids (e.g. Nikita's Gift, Rosseyanka, JT-02) are
  trialed here — treat as not present unless confirmed.
- Bloom and ripening dates given here are region / year-dependent placeholders; the platform
  derives phenophase dates from data and must not freeze them.
- Nominal levels of `sex` (e.g. male / female / monoecious / perfect) are not enumerated in
  crops.yml — confirm the category set.
- How `ploidy` is recorded (60- vs 90-chromosome race label vs a flow-cytometry value);
  crops.yml lists it as ordinal.
- Whether `fruit_set` is scored from imagery at all or is a hand-count field observation.
- Exact caliper endpoints for `fruit_height` and `fruit_diameter` (whether the style beak is
  included, and where the calyx-end reference sits) are breeder-defined and not in crops.yml.
- Ordinal scale definitions / endpoints for `calyx`, `cropload`, `fruit_drop`,
  `astringency`, `flavor_rating`, and `overall_disease` are breeder-defined and not in
  crops.yml.
- Whether `fruitsample_n` and `fruitsample_weight` are pure destructive-subsample bookkeeping
  supporting the lab tannin / quality assays.
- Method and relationship between `soluble_tannins`, `total_tannins` (both µg/g pulp,
  spectrophotometric) and the sensory `astringency` rating.
- Real Upper-Midwest pest-pressure ranking (psyllid, webworm, deer / wildlife browse, scale)
  and which pressures fold into the single `overall_disease` rating vs are out of scope.
- Confirm that persimmon intentionally carries only the aggregate `overall_disease` trait
  (no per-disease columns) and that individual diseases are not to be scored separately.

## Sources

- ASHS 2017 — Ploidy Level of American Persimmon in Kentucky (90-chromosome hexaploid cultivars, parthenocarpy, race ranges): https://ashs.confex.com/ashs/2017/webprogramarchives/Paper26923.html
- HortScience (peer-reviewed) — Ploidy Level in American Persimmon (_Diospyros virginiana_) Cultivars: https://pdfs.semanticscholar.org/5b6f/63d52e60943414d7f6aaf4fbdde9b2982069.pdf
- Savanna Institute — Persimmon program (Upper-Midwest breeding, 90-chromosome _D. virginiana_): https://www.savannainstitute.org/persimmon/
- NC State Extension Gardener Plant Toolbox — _Diospyros virginiana_ (growth form, dioecious flowers, fruit, calyx / beak, leaf spot, psyllid): https://plants.ces.ncsu.edu/plants/diospyros-virginiana/
- University of Kentucky CCD — American Persimmon crop profile (CCD-CP-001): https://ccd.uky.edu/sites/default/files/2024-11/ccd-cp-001_american-persimmon.pdf
- USDA Forest Service, Silvics of North America (Ag. Handbook 654) — _Diospyros virginiana_ (dioecy, ripening, wilt): https://www.srs.fs.usda.gov/pubs/misc/ag_654/volume_2/diospyros/virginiana.htm
- Florida DACS Plant Pathology Circular 197 — Cephalosporium / Persimmon Wilt (_Acremonium diospyri_): https://ccmedia.fdacs.gov/content/download/11204/file/pp197.pdf
- UF/IFAS ENH390 / ST231 — _Diospyros virginiana_: Common Persimmon (form, leaf spot, wilt susceptibility): https://ask.ifas.ufl.edu/publication/ST231