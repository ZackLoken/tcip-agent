---
name: chestnut
description: "Chestnut (Castanea spp., Chinese chestnut C. mollissima plus interspecific C. dentata × C. mollissima and Euro-Japanese C. sativa × C. crenata hybrids such as 'Colossal'), a monoecious, ambophilous (wind- and insect-pollinated) deciduous nut tree grown in Upper-Midwest agroforestry rows. Covers its duodichogamous two-catkin-type bloom, spiny burrs enclosing 1-3 glossy nuts, and the appearance of chestnut blight, anthracnose, Phytophthora root rot, and weevil damage, plus which chestnut traits are field-imageable versus lab or destructive. Load this skill when phenotyping chestnut, working with chestnut imagery, or measuring chestnut traits (catkin bloom, burr counts/size/density, nut dimensions, disease and weevil ratings)."
---

# Chestnut (Castanea)

## Identity

Chestnut here is primarily Chinese chestnut (*Castanea mollissima*), with interspecific
hybrids in the breeding pool: American × Chinese (*C. dentata* × *C. mollissima*) and
Euro-Japanese cultivars (*C. sativa* × *C. crenata*, e.g. 'Colossal'). It is a single-stem
deciduous tree by default: rounded, spreading crown and low, irregular branching, grown in
orchard/agroforestry rows, not a bush or hedgerow crop like currant or elderberry.

Reproductively it is monoecious and ambophilous, pollinated by both wind and insects (the
strong floral scent, sticky pollen, and nectar-bearing bisexual catkins point to a substantial
insect role), with a duodichogamous bloom that
governs how the floral traits must be imaged:

- Two catkin morphotypes exist: unisexual male catkins and bisexual/androgynous
  catkins (staminate flowers along their length plus 1-3 basal pistillate inflorescences).
- Sequence: unisexual male catkins open first and shed the major pollen pulse; then pistillate
  flowers become receptive; then bisexual catkins release a second, smaller pulse.
- Predominantly self-incompatible, so nut set needs cross-pollination between compatible
  cultivars nearby.

`catkin_sex_ratio` and `catkin_bloom_date` cannot be read off box size or a naive
"male vs female" split (see Measurement integrity).

## Trait authority

`crops.yml` is the trait authority (21 chestnut traits). Verify every trait there; never
assert one it does not list. This skill does not reproduce the catalog.

## Field-imageable vs lab / destructive

Which traits can come from imagery is a per-trait, validated question; there is no fixed
sensor→trait table. The partition below is the starting split; confirm the imaging viewpoint
and scoring basis with the breeder before treating any of these as measured.

Field-imageable (standing tree / ground or canopy views):
`catkin_bloom_date`, `burr_drop_date`, `n_burrs`, `burrs_density`, `burr_size`,
`catkin_sex_ratio`, `cryphonectria_parasitica`, `chestnut_anthracnose`, `plant_height`,
`plant_width_inrow`, `plant_width_betweenrow`, `plant_surface_area`, `plant_volume`.

- Plant-size traits (`plant_height`, `plant_width_inrow`, `plant_width_betweenrow`) are pixel
  quantities until an in-frame scale / photogrammetric calibration makes them metric. Pixels are
  not millimeters.
- `plant_surface_area` and `plant_volume` are defined from a 3D canopy model (SfM/LiDAR).
  TCIP is 2D-only today (3D not built), so these are not achievable yet: a 2D bbox
  height/area is not a valid proxy for crown volume or planimetric area.
- `n_burrs` / `burrs_density` from a single 2D view undercount occluded burrs; a raw count is
  not the true per-tree total without multi-view or occlusion modeling.

Lab or destructive (bench / post-harvest / assay):
`burr_weight`, `burr_yield`, `nut_height`, `nut_length`, `nut_width`, `weevil_damage`,
`flavor_rating`, `phytophthora_cinnamomi`.

- Nut dimensions require the nut extracted from the spiny burr and a reference scale;
  they cannot be read through an intact burr, and a burr box is not a nut dimension.
- `weevil_damage` needs cut-test/float-test on harvested nuts; external appearance does not
  reveal internal larvae before exit holes form.
- `phytophthora_cinnamomi` is a root/crown pathogen; a definitive rating needs root/soil
  assessment. Canopy decline is only a nonspecific above-ground indirect signal.

## Phenophase calendar → date traits

Timing is approximate, region- and year-dependent (values below are Michigan/Upper-Midwest
derived). The platform derives dates from data, never from a fixed calendar. Only two chestnut
phenophases map to a date trait; most map to none.

| Phenophase | Approx. timing | Trait |
|---|---|---|
| Bud break / leaf-out | Late Apr - May | (none) |
| Shoot elongation / pre-bloom | May - mid Jun | (none) |
| Catkin bloom (unisexual male, major pollen) | Late Jun - mid Jul | `catkin_bloom_date` |
| Female receptivity + bisexual (second) pollen phase | Early - mid Jul | (none) |
| Burr set and development | Mid Jul - Sep | (none; feeds `n_burrs`, `burrs_density`, `burr_size`) |
| Burr maturation / dehiscence | Sep - Oct | (none) |
| Burr / nut drop | Sep - Oct | `burr_drop_date` |
| Leaf senescence / fall color | Oct - Nov | (none) |

A date trait is a threshold on a validated multi-date time series, not a single-image
cutoff. Defer the fraction/crossing mechanics to the phenology skill; the biological anchor
for `catkin_bloom_date` (first pollen shed vs general catkin visibility vs a percent-open
threshold) is undefined and must be set by the breeder.

## Key structures and imagery appearance

- Unisexual (male) catkin: near shoot terminals, erect; ~10-25 cm long; slender,
  creamy-white to pale-yellow, densely packed with staminate flowers; makes the canopy look
  pale/fuzzy at peak bloom; strongly scented. Visually dominant and most detectable.
- Bisexual / androgynous catkin: near shoot terminals; similar length or slightly
  shorter; staminate flowers along its length plus 1-3 pistillate inflorescences at the base.
  Telling it apart from the unisexual male catkin is the crux of `catkin_sex_ratio`.
- Pistillate flower / female inflorescence: at the base of bisexual catkins; a few mm; a
  small green spiny cupule with whitish protruding styles; individually inconspicuous and hard
  to resolve at drone distance.
- Burr (involucre): on the canopy, singly or clustered; ~2.5-8 cm diameter; rounded,
  densely covered in long sharp branched spines; green when immature, ripening
  golden/yellow-brown; splits into 2-4 valves at maturity. Spines inflate the apparent outline
  versus the woody body.
- Nut: 1-3 per burr; ~2-4 cm; glossy dark mahogany shell with a paler basal scar and a
  pointed, often pubescent apex; flattened when 2-3 share a burr. Visible only once the burr
  splits or is opened.
- Leaf: throughout canopy; ~10-20 cm, oblong-lanceolate, coarsely serrate with
  bristle-tipped teeth, glossy green; *C. mollissima* has a pubescent underside. Substrate for
  anthracnose and blight flag symptoms.
- Trunk / bark / crown: rounded spreading crown; smooth young bark (blight canker site)
  furrowing with age; lower stem / root crown is the Phytophthora lesion site.

## Diseases and pests

- Chestnut blight (`cryphonectria_parasitica`): *Cryphonectria parasitica* (fungus).
  Sunken or swollen orange-brown cankers on stems/branches, cracked bark, yellow-orange to
  reddish pustules erupting through bark with orange spore tendrils in wet weather, wilted brown
  "flag" leaves that stay attached above the canker, epicormic sprouts below. Ordinal severity.
- Chestnut anthracnose (`chestnut_anthracnose`): leaf/twig blight fungi (*Colletotrichum*
  and/or *Marssonina* spp.; exact agent on chestnut unconfirmed). Dry brown irregular leaf spots
  and blotches (often along veins/margins), leaf curling, premature defoliation, bud/twig
  dieback; worst after cool wet springs. Ordinal severity from leaf imagery.
- Phytophthora root rot / ink disease (`phytophthora_cinnamomi`): *Phytophthora
  cinnamomi* (soilborne oomycete). Root and root-crown rot is below-ground and not directly
  imageable; only indirect above-ground signs show: canopy chlorosis, wilting, thin crowns,
  branch dieback, whole-tree decline, and black-to-rusty bleeding streaks through bark near the
  root crown. Symptoms are nonspecific; a definitive rating needs root/crown/soil assessment.
- Chestnut weevil (`weevil_damage`): *Curculio* spp. snout beetles (lesser chestnut
  weevil, *C. sayi*, is the key Midwest species). ~1-2 mm round exit holes in nut shells,
  cream legless grubs inside cut nuts, frass, premature nut drop. Reliable assessment is on
  harvested/cut nuts (cut or float test), not standing-tree imagery.

## Annotation challenges

Defer annotation mechanics to the annotation skill. Chestnut-specific difficulties:

- Two catkin types look alike. Unisexual male vs bisexual/androgynous catkins are not two
  obvious visual classes; the distinguishing pistillate inflorescences sit at the base and are
  tiny. The breeder must define exactly what `catkin_sex_ratio` counts before any class scheme.
- Spiny burrs blur their own boundary: spines extend the visible outline well past the
  woody body, so any burr box or size measure needs a stated convention (body vs including
  spines) and a reference scale.
- Occlusion and clustering: burrs overlap and hide behind foliage, so counts undercount.
- Nuts are hidden inside intact burrs; nut-level annotation implies split/opened burrs or
  bench imagery.

## Measurement integrity

Per the CLAUDE.md measurement-integrity invariant (never a geometric/pixel proxy; validate
against a reference sized to the trait: GT annotations, or a breeder-confirmed sample of the
model's own outputs (review-confirmation), before any result, not dense GT for every trait; see
the catkin-elongation cautionary tale there). Chestnut-specific traps:

- `catkin_sex_ratio`: a size split or a made-up male/female class fabricates the ratio. The
  breeder defines what is counted (catkin types vs flower counts) and at which pollen phase.
- `burr_size` / `nut_height` / `nut_length` / `nut_width`: a pixel box width is not a mm
  dimension without an extracted structure, a reference scale, and a stated convention.
- `catkin_bloom_date` / `burr_drop_date`: derive the threshold from a validated multi-date time
  series; never freeze a single-image percent-open/percent-drop cutoff.
- `cryphonectria_parasitica` / `chestnut_anthracnose` / `phytophthora_cinnamomi`: the breeder's
  ordinal severity is not a raw lesion-pixel fraction; calibrate and validate against expert
  scores. Phytophthora canopy wilt is a nonspecific proxy for a root pathogen.
- `plant_volume` / `plant_surface_area`: valid only from real 3D reconstruction, which TCIP does
  not have today; no 2D surrogate.
- `weevil_damage`: no valid infestation call from external nut appearance before exit holes;
  needs cut/float-test ground truth.

## Needs expert confirmation

- `plant_surface_area`: is the delivered quantity the *planimetric crown area* (achievable from a
  validated 2D mask + in-frame scale calibration, per CLAUDE.md's measurement-integrity invariant)
  or specifically the *3D-canopy-model derivation* crops.yml's current definition names? The two
  are not the same thing, and only the breeder can redefine their own trait's meaning.
- Exact species/cultivar composition actually planted and bred (Chinese chestnut dominant;
  whether American × Chinese hybrids and Euro-Japanese cultivars like 'Colossal' are included).
- Training system at the sites (single-stem central-leader tree vs coppice/multi-stem);
  affects growth-form and structural-trait assumptions.
- Precise definition of `catkin_sex_ratio`: unisexual-vs-bisexual catkin ratio or a flower-level
  count, and which pollen phase it is imaged in.
- Biological anchor of `catkin_bloom_date` (first pollen shed vs general catkin visibility vs a
  percent-open threshold).
- Local phenology: the timings here are Michigan-derived and region/year-dependent; dates must
  be derived from data, not fixed.
- `weevil_damage` scoring basis (cut test, float test, exit-hole count) and target species.
- `phytophthora_cinnamomi` scoring basis (root/soil assay vs field canopy-decline rating);
  determines whether it is even partially field-imageable.
- `chestnut_anthracnose` causal organism on chestnut (*Colletotrichum* vs *Marssonina* vs other)
  and how it is distinguished from other foliar blights in scoring.
- `flavor_rating` protocol (raw vs roasted, sensory panel design).
- Whether `n_burrs` and `burrs_density` are per whole tree, per canopy-area unit, or per image,
  and the intended imaging viewpoint (ground vs drone).

## Sources

- Savanna Institute, *Chestnut* program overview (Chinese chestnut, cross-pollination).
- University of Missouri Center for Agroforestry, *Descriptions of Chestnut Cultivars for Nut
  Production* (2021).
- *A Roadmap for Participatory Chestnut Breeding for Nut Production in the Eastern United States*
  (Frontiers in Plant Science, 2021).
- *Castanea mollissima: A Chinese Chestnut for the Northeast* (Arnold Arboretum / Arnoldia).
- MSU Extension, *Estimating crop load in edible chestnuts* (nuts per burr, Michigan timing).
- MSU Extension, *Biology and Management of the Lesser Chestnut Weevil in Michigan Chestnut
  Orchards*.
- The American Chestnut Foundation, *Phytophthora Root Rot* fact sheet (2021).
- *Adaptive function of duodichogamy: why do chestnut trees have two pollen emission phases?*
  (American Journal of Botany, 2023).
- *Revisiting pollination mode in chestnut (Castanea spp.): an integrated approach* (Botany
  Letters, 2021).
- *Cryphonectria parasitica, the causal agent of chestnut blight* (Molecular Plant Pathology,
  2018).
