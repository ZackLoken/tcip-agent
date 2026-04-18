---
name: Hazelnut
description: "Hazelnut (Corylus spp.) — most sensor-diverse TCIP crop. 58 traits, 7 sensor types (RGB, LiDAR, SfM, hyperspectral, NIRS, radar). Catkin/pistillate phenology, EFB disease, kernel quality."
---

# Hazelnut — Corylus spp.

58 traits | 7 sensor types | 3 perspectives (aerial, ground, lab) | 10 pipeline groups

## Phenology Calendar

1. **Dormancy** (Nov–Feb) — Catkins visible but closed
2. **Catkin elongation** (Feb–Mar) — Track catkin_05/50/95per_date, catkin_elongation_date
3. **Budbreak** (Mar–Apr) — Leaf emergence
4. **Flowering** (Mar–Apr) — Track pistillate_05/50/95per_date. Female flowers are tiny red tufts at branch tips
5. **Nut development** (May–Aug) — Cluster nut count assessment
6. **Harvest** (Aug–Oct) — Nut drop, inshell weight/count
7. **Senescence** (Oct–Nov) — Defoliation

## Trait Inventory

### Phenology (14 traits)
| Trait | Sensor | ML Task | Format |
|-------|--------|---------|--------|
| `catkin_05per_date` | Ground RGB | Object detection → Change detection | date |
| `catkin_50per_date` | Ground RGB | Object detection → Change detection | date |
| `catkin_95per_date` | Ground RGB | Object detection → Change detection | date |
| `catkin_elongation_date` | Ground RGB | Object detection → Change detection | date |
| `pistillate_05per_date` | Ground RGB | Object detection → Change detection | date |
| `pistillate_50per_date` | Ground RGB | Object detection → Change detection | date |
| `pistillate_95per_date` | Ground RGB | Object detection → Change detection | date |

### Disease (5 traits)
| Trait | Sensor | ML Task | Format |
|-------|--------|---------|--------|
| `efb_damage` | Aerial RGB | Classification | ordinal |
| `efb_presence` | Aerial RGB | Classification | binary |
| `big_bud_mite_damage` | Ground RGB | Classification | ordinal |
| `weevil_damage` | Ground RGB | Classification | ordinal |
| `efb_canker_length` | Ground RGB | Regression | numeric (cm) |

### Morphology (15 traits)
| Trait | Sensor | ML Task |
|-------|--------|---------|
| `plant_biomass` | Aerial LiDAR | Point cloud |
| `plant_height` | Aerial LiDAR | Point cloud |
| `plant_max_height` | Aerial LiDAR | Point cloud |
| `plant_volume` | Aerial LiDAR | Point cloud |
| `stem_branching_frequency` | Aerial LiDAR | Point cloud |
| `stem_vertical_angle` | Aerial LiDAR | Point cloud |
| `stem_count` | Aerial RGB | Object detection |
| `plant_max_width` | Aerial SfM | Point cloud |
| `plant_min_width` | Aerial SfM | Point cloud |
| `plant_surface_area` | Aerial SfM | Point cloud |
| `plant_width_betweenrow` | Aerial SfM | Point cloud |
| `plant_width_inrow` | Aerial SfM | Point cloud |
| `root_crown_betweenrow_width` | Ground Radar | Point cloud |
| `root_crown_inrow_width` | Ground Radar | Point cloud |
| `stem_internode_length` | Ground RGB | Regression |

### Quality (18 traits)
| Trait | Sensor | ML Task |
|-------|--------|---------|
| `inshell_height` | Lab RGB | Instance segmentation |
| `inshell_length` | Lab RGB | Instance segmentation |
| `inshell_width` | Lab RGB | Instance segmentation |
| `kernel_height` | Lab RGB | Instance segmentation |
| `kernel_length` | Lab RGB | Instance segmentation |
| `kernel_width` | Lab RGB | Instance segmentation |
| `kernel_pellicle` | Lab RGB | Classification (ordinal) |
| `inshell_weight` | Lab RGB | Regression |
| `kernel_weight` | Lab RGB | Regression |
| `kernel_perc_grav` | Lab RGB | Regression |
| `kernel_perc_vol` | Lab RGB | Regression |
| `cluster_mass` | Lab RGB | Regression |
| `kernel_perc_oil` | Lab NIRS | Regression |
| `kernel_dry_matter_perc` | Lab NIRS | Regression |
| `kernel_fiber` | Lab NIRS | Regression (ordinal) |
| `kernel_oleic_acid_content` | Lab Hyperspectral | Regression |

### Yield (6 traits)
| Trait | Sensor | ML Task |
|-------|--------|---------|
| `ttl_inshell_weight` | Aerial RGB | Regression |
| `cluster_nut_count` | Ground RGB | Object detection |
| `ttl_inshell_count` | Lab RGB | Object detection |

### Non-Automatable (10)
catkin_05/50/95per_julian, pistillate_05/50/95per_julian, cluster_detachment_force, flavor_rating, nut_husk_rating, nut_perc_blanks

## Annotation Guidance

- **Catkins**: Elongated pendulous structures. Tight bounding boxes aligned to catkin axis. At 5% emergence they're small and partially hidden — high miss rate expected.
- **Pistillate flowers**: Extremely small (2–3mm). Require high-res ground images. Red stigmas emerging from buds — annotators frequently miss early-stage flowers.
- **EFB cankers**: Dark sunken lesions on branches. 1cm to 30cm+. Annotate full lesion extent for canker_length regression.
- **Nut clusters**: Multiple nuts in husks. Count individual nuts, not clusters. Overlapping husks make counting hard.
- **Bush canopy overlap**: Adjacent bushes in hedgerow plantings grow together — instance segmentation boundaries are ambiguous.
