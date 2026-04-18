---
name: Chestnut
description: "Chestnut (Castanea spp.) — 21 traits, 4 sensor types. Focus on blight resistance, bur/nut quality, and disease classification."
---

# Chestnut — Castanea spp.

21 traits (20 automatable, 1 non-automatable) | 4 sensor types | 3 perspectives | 8 pipeline groups

## Phenology Calendar

1. **Dormancy** (Nov–Mar) — Bare canopy
2. **Budbreak** (Apr) — Leaf emergence
3. **Catkin emergence** (May–Jun) — Male catkins elongate
4. **Bloom** (Jun–Jul) — Pollination period
5. **Bur development** (Jul–Sep) — Burs swell, nuts form inside
6. **Harvest** (Sep–Oct) — Burs split, nuts drop
7. **Senescence** (Oct–Nov) — Leaf drop

## Trait Inventory

### Disease (4 traits)
| Trait | Sensor | ML Task | Format |
|-------|--------|---------|--------|
| `chestnut_blight` | Aerial/Ground RGB | Classification | ordinal |
| `phytophthora` | Ground RGB | Classification | ordinal |
| `gall_wasp_damage` | Ground RGB | Classification | ordinal |
| `ambrosia_beetle_damage` | Ground RGB | Classification | ordinal |

### Morphology (5 traits)
| Trait | Sensor | ML Task |
|-------|--------|---------|
| `plant_height` | Aerial LiDAR | Point cloud |
| `plant_volume` | Aerial LiDAR | Point cloud |
| `canopy_diameter` | Aerial SfM | Point cloud |
| `trunk_diameter` | Ground RGB | Regression |
| `branch_architecture` | Aerial LiDAR | Point cloud |

### Quality (6 traits)
| Trait | Sensor | ML Task |
|-------|--------|---------|
| `nut_length` | Lab RGB | Instance segmentation |
| `nut_width` | Lab RGB | Instance segmentation |
| `nut_height` | Lab RGB | Instance segmentation |
| `nut_weight` | Lab RGB | Regression |
| `kernel_perc` | Lab RGB | Regression |
| `pellicle_ease` | Lab RGB | Classification (ordinal) |

### Yield (4 traits)
| Trait | Sensor | ML Task |
|-------|--------|---------|
| `bur_count` | Aerial/Ground RGB | Object detection |
| `nut_count` | Lab RGB | Object detection |
| `total_nut_weight` | Lab RGB | Regression |
| `yield_per_tree` | Aerial RGB | Regression |

### Phenology (2 traits)
| Trait | Sensor | ML Task |
|-------|--------|---------|
| `bloom_date` | Aerial RGB | Change detection |
| `harvest_date` | Aerial RGB | Change detection |

## Annotation Guidance

- **Burs**: Spiny spherical structures. Annotate with bounding boxes. Green burs on tree vs brown split burs on ground.
- **Blight cankers**: Orange-brown bark lesions with fan-shaped mycelial mats. Score 0–5 severity.
- **Nut sizing**: Lab images — individual nuts on calibration background. Instance segmentation for length/width/height.
- **Catkins**: Long pendulous male flowers — similar to hazelnut but larger.
