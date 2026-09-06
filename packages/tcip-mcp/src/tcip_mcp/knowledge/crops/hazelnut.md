---
name: hazelnut
description: "Domain knowledge for hazelnut (Corylus americana × Corylus avellana hybrids and pure C. americana selections; family Betulaceae), a multi-stemmed, clump-forming nut shrub grown in Upper-Midwest breeding plantings. Covers its monoecious, wind-pollinated, dichogamous reproductive biology; pendulous male catkins and tiny red-styled pistillate flowers; husk-enclosed nut clusters, in-shell nuts, and kernels; and Eastern Filbert Blight, big-bud mite, and hazelnut weevil signs. Load this when phenotyping hazelnut, working with hazelnut imagery, or measuring hazelnut traits."
---

# Hazelnut (Corylus americana × avellana)

## Identity

Interspecific hybrids of *Corylus americana* Walter (American hazelnut) × *Corylus avellana* L.
(European hazelnut / filbert), plus pure *C. americana* selections; family Betulaceae. The
Upper-Midwest program favors hybrids for cold-hardiness and Eastern Filbert Blight resistance.

Growth form is a multi-stemmed, clump-forming shrub (~1-5 m), arising from a root crown and
spreading by basal suckers/rhizomes into a dense clump or hedgerow, not a single-trunk tree. The
vocabulary tracks multiple stems and a root crown (`stem_count`,
`stem_branching_frequency`, `root_crown_inrow_width`, `root_crown_betweenrow_width`) and separate
in-row vs between-row canopy widths rather than a single-trunk diameter (hazelnut carries no
trunk-diameter trait).

Reproduction is monoecious, wind-pollinated, and dichogamous: on a genotype, pollen shed and
stigma receptivity are offset in time. That offset runs both ways depending on cultivar and
climate: protandrous (pollen first) or protogynous (stigmas first), with protogyny often
predominating in cold, long-winter regions like the Upper Midwest, so do not assume a fixed
male-then-female order. A single-locus sporophytic self-incompatibility system (allele series at one
*S* locus) means a genotype generally needs a bloom-overlapping, compatible pollinizer. Flowers open
on leafless stems in early spring. Consequence for imaging: male (catkin) and female
(pistillate) phenology are tracked as *separate* trait families because dichogamy separates them in
time; derive their overlap per site-year from data rather than assuming an ordering.

## Trait authority

`crops.yml` is the trait authority (58 hazelnut traits). Verify every trait there; never assert one
it does not list. This skill does not reproduce the catalog; it grounds how the traits appear in
imagery and which can come from pixels at all.

## Field-imageable vs lab/destructive

Not every trait is a computer-vision target. "Field-imageable" here spans field/ground/drone/
close-range imagery of the living plant and valid bench morphometry of *intact* harvested nuts;
"lab/destructive" covers weighing, cracking, chemistry/NIRS, sensory panels, force gauges, and
harvest processing that no image can substitute for.

Field-imageable (living plant or bench, with the caveats below):
- Phenology (leafless-season close-range): `catkin_elongation_date`, `catkin_05per_date`,
  `catkin_50per_date`, `catkin_95per_date`, `catkin_05per_julian`, `catkin_50per_julian`,
  `catkin_95per_julian`; `pistillate_flowering_date`, `pistillate_05per_date`,
  `pistillate_50per_date`, `pistillate_95per_date`, `pistillate_05per_julian`,
  `pistillate_50per_julian`, `pistillate_95per_julian`.
- Architecture / canopy: `plant_height`, `plant_max_height`, `plant_max_width`, `plant_min_width`,
  `plant_width_inrow`, `plant_width_betweenrow`, `root_crown_inrow_width`,
  `root_crown_betweenrow_width`, `stem_count`, `stem_branching_frequency`, `stem_internode_length`,
  `stem_vertical_angle`, `terminal_bearing`, `plant_surface_area`, `plant_volume`, `plant_biomass`.
- Disease/pest signs on the plant: `efb_presence`, `efb_damage`, `efb_canker_length`,
  `big_bud_mite_damage`.
- Bench morphometry of intact nuts: `inshell_height`, `inshell_length`, `inshell_width`;
  `weevil_damage` (from exit holes on harvested nuts).

Lab / destructive (never a CV target): `inshell_weight`, `kernel_height`, `kernel_length`,
`kernel_width`, `kernel_weight`, `kernel_dry_matter_perc`, `kernel_fiber`,
`kernel_oleic_acid_content`, `kernel_pellicle`, `kernel_perc_grav`, `kernel_perc_oil`,
`kernel_perc_vol`, `nut_husk_rating`, `nut_perc_blanks`, `cluster_mass`, `cluster_nut_count`,
`cluster_detachment_force`, `ttl_inshell_count`, `ttl_inshell_weight`, `flavor_rating`.

Caveats that move the line (confirm before treating as imageable):
- Plant-size traits (`plant_height`, `plant_max_height`, `plant_max_width`, `plant_min_width`,
  `plant_width_inrow`, `plant_width_betweenrow`, `root_crown_inrow_width`,
  `root_crown_betweenrow_width`) are pixel quantities until an in-frame scale / photogrammetric
  calibration makes them metric. Pixels are not millimeters.
- `plant_surface_area`, `plant_volume`, and `plant_biomass` are defined from a 3D canopy model
  (biomass via an allometric equation). The platform is currently 2D-only with no 3D point-cloud
  pipeline; they may not be computable today.
- `inshell_height/length/width` need a harvested subsample and in-image physical-scale
  calibration; pixels are not mm without it.
- `ttl_inshell_count` and `cluster_nut_count` are harvest/bench counts; on-plant nut counting is
  heavily occluded (see below) and is not a substitute.

## Phenophase calendar

Timing is approximate, region/year/genotype-dependent, and must be derived from imagery each
site-year; never fixed. Upper-Midwest bloom is far later than the Dec-Feb reported for warmer
regions. Defer the elongated-fraction / crossing math to the `phenology` skill.

| Approx. window (Upper Midwest) | Stage | Date traits |
|---|---|---|
| ~April-May, leafless | Catkin elongation & pollen shed (male anthesis) | `catkin_elongation_date`, `catkin_05per_date`, `catkin_50per_date`, `catkin_95per_date` (+ julian counterparts) |
| ~April-May, overlapping catkins (order varies by genotype) | Pistillate receptivity (female anthesis) | `pistillate_flowering_date`, `pistillate_05per_date`, `pistillate_50per_date`, `pistillate_95per_date` (+ julian counterparts) |
| ~May | Leaf-out | none; canopy begins occluding stems/buds/clusters |
| ~June-August | Nut development & sizing | none |
| ~late Aug-Sept (into Oct) | Ripening, husk browning, harvest | none; triggers destructive nut/kernel/cluster measurements |
| ~Oct-Nov | Senescence & dormancy | none; cankers/galls most detectable on bare stems |

The hazelnut vocabulary has no leaf-out, ripening/harvest, or senescence date trait (those
belong to other crops). Do not populate a phenology trait that is not in the hazelnut set.

## Key structures in imagery

- Catkin (male inflorescence): pendulous cylindrical cluster on 1-year-old shoots; overwinters
  short (~1-2 cm), firm, greenish-brown, then elongates to loose ~5-8 cm pale yellow-green/yellow
  clusters that dehisce clouds of yellow pollen. Large, high-contrast against bare stems: the most
  tractable detection target.
- Pistillate flower (female): tiny bud-like structure emitting a tuft of bright red-magenta
  styles (~1-3 mm visible); progresses red-dot → intermediate → full "spider/sunburst." Borne singly
  or clustered in axils/terminals. Very small and low-contrast except for the red styles; needs
  macro/close-range resolution; easily missed at drone scale.
- Nut cluster with involucre (husk): 1-5 nuts, each in a leafy green bracteal husk with ragged
  margins; green in summer, orange-brown at maturity. Nuts are hidden inside the husk on the plant.
- In-shell nut: round-to-ovoid smooth brown shell, ~8-15 mm (hybrids larger); imaged on a bench
  post-harvest.
- Kernel & pellicle: cream seed under a thin brown skin (pellicle); exposed only by cracking.
- Multi-stem clump & root crown: several stems from a basal root crown suckering into a clump;
  the root crown is often occluded by suckers, mulch, or foliage.

## Diseases & pests

- Eastern Filbert Blight: *Anisogramma anomala* (biotrophic ascomycete). Elongated sunken
  perennial cankers on 1+ year branches bearing longitudinal rows of black football-shaped
  stromata erupting through bark; branch flagging/dieback above. Most detectable on dormant,
  defoliated stems. Traits: `efb_presence` (binary), `efb_damage` (1-5 ordinal), `efb_canker_length`
  (cm).
- Big-bud / filbert bud mite: *Phytoptus avellanae* / *Cecidophyopsis vermiformis* (eriophyid
  mites). Mites are microscopic and not imageable; only the sign is: abnormally swollen, rounded
  "big buds" (pseudo-galls) versus normal slender pointed buds, on shoots in late winter-spring.
  Trait: `big_bud_mite_damage` (1-5 ordinal).
- Hazelnut / filbert weevil: *Curculio* spp. (long-snouted weevils). Imageable sign is a round
  ~1.6 mm larval exit hole (plus frass) on harvested nuts; the larva and frass-filled kernel are
  visible only after cracking. Trait: `weevil_damage` (1-5 ordinal).

## Annotation challenges

Defer annotation mechanics to the `annotation` skill. Crop-specific difficulties:
- Catkins are the easy case: large, high-contrast, leafless-season. Pistillate flowers are
  the hard case: ~1-3 mm of red styles that demand macro/close-range imagery and are easily missed.
- Nuts are occluded inside leafy husks and canopy all season; on-plant counting is unreliable.
- EFB cankers and big-bud galls are occluded by foliage; annotate on dormant, defoliated stems.
- The root crown is frequently hidden by suckers, mulch, or foliage.

## Measurement integrity

Per the CLAUDE.md measurement-integrity invariant (validate against a reference sized to the
trait: GT annotations, or a breeder-confirmed sample of the model's own outputs (review-confirmation),
not dense GT for every trait; geometry can't proxy a state or replace the CV step). Hazelnut-specific
traps:

- `catkin_elongation_date`: elongation/anthesis is a breeder-defined *visible morphological
  state* (loosening + pollen shed), established by a validated call against expert scoring: a state,
  not a dimension you read off a bbox. See the `phenology` skill for the elongated-fraction
  definition and its `positive_class_assessed` guard.
- `catkin_*per_*` and `pistillate_*per_*` "% open" mean the fraction of catkins/flowers at the
  breeder-defined anthesis/receptivity state, not "% detected." No date without an expert-scored
  "open" criterion validated against ground truth.
- `efb_damage`, `big_bud_mite_damage`, `weevil_damage`, `terminal_bearing` are breeder-defined
  ordinal rubrics; calibrate the pixel signal to the rubric, never invent the thresholds.
- `efb_canker_length` needs in-image physical-scale calibration and full canker visibility; an
  uncalibrated bbox length is not cm.
- `ttl_inshell_count` / `cluster_nut_count`: nuts are occluded, so a raw detected-nut count is a
  biased undercount, not the true count. No count without a validated occlusion/yield model.
- `plant_biomass` is an allometric estimate from imaged canopy volume, not a direct measurement;
  the equation must be validated against destructively-harvested hybrid-hazelnut biomass.

## Needs expert confirmation

- `plant_surface_area`: is the delivered quantity the *planimetric crown area* (achievable from a
  validated 2D mask + in-frame scale calibration, per CLAUDE.md's measurement-integrity invariant)
  or specifically the *3D-canopy-model derivation* crops.yml's current definition names? The two
  are not the same thing, and only the breeder can redefine their own trait's meaning.
- The program's specific hybrid selections and named releases (general species knowledge is used
  here, not an accession list).
- All Upper-Midwest phenophase timings, approximate and region/year/genotype-dependent; derive from
  data each site-year.
- Direction of dichogamy for the program's genotypes and site (protandrous vs protogynous);
  protogyny may predominate in cold winters; do not assume a male-then-female order.
- Precise ordinal-scale definitions: `efb_damage` (1-5), `big_bud_mite_damage` (1-5),
  `weevil_damage` (1-5), `kernel_fiber` (1-4), `kernel_pellicle` (1-7), `terminal_bearing` (1-5),
  and `nut_husk_rating`.
- Biological criteria for `catkin_elongation_date` ("most catkins elongated") and
  `pistillate_flowering_date` ("most pistillate flowers opened"), and how they relate to the
  05/50/95% traits (e.g., is elongation ≈ pollen shed? is `pistillate_flowering_date` ≈
  `pistillate_50per_date`?).
- The exact "open" definition for catkins (start of elongation vs pollen shed) vs for pistillate
  flowers (style emergence vs full receptivity).
- Whether `inshell_height/length/width` are intended from imagery vs calipers, and how the axes are
  defined relative to nut orientation (tentatively placed as bench-imageable, but the split,
  dimensions imageable, `inshell_weight` not, needs confirmation).
- The allometric equation for `plant_biomass` and whether it is validated for hybrid hazelnut.
- `stem_vertical_angle`: exact definition (a paired pre/post crop-load branch-angle change) and the
  imaging protocol to capture it.
- Whether `cluster_nut_count` and `ttl_inshell_count` are image-estimable at all, or bench/harvest
  counts only.
- `plant_surface_area` and `plant_volume` are defined from a 3D canopy model; under the current
  2D-only scope these may not be computable; confirm the intended data source (drone SfM vs LiDAR)
  and feasibility.
- Which *Curculio* weevil species is the local pest, and whether EFB pressure is present in the
  specific plantings (affects whether `efb_*` traits are active).

## Sources

- Savanna Institute, *Interested in growing hazelnuts?*
- Upper Midwest Hazelnut Development Initiative, *Hazelnuts 101* establishment fact sheet.
- *Yield, quality and genetic diversity of hybrid hazelnut selections in the Upper Midwest* (Experts@Minnesota).
- *Hazelnut floral phenology in southern Ontario* (Canadian Journal of Plant Science).
- *The Reproduction of Hazelnut (Corylus avellana L.): A Review* (ISHS Acta Horticulturae).
- *Anisogramma anomala (eastern filbert blight)*, Bugwood Wiki; UW-Madison PDDC.
- *Hazelnut-Filbert bud mite (Phytoptus avellanae)*, PNW Pest Management Handbooks.
- *Biology, Ecology, and Management of the Hazelnut-Feeding Weevils (Curculio spp.)*, J. Integrated Pest Management.
- *Corylus americana (American Hazelnut)*, Minnesota Wildflowers.
- *Development of a uniform phenology scale (BBCH) in hazelnuts*, Scientia Horticulturae.
