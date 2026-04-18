---
name: Black Locust
description: "Black locust (Robinia pseudoacacia) — biomass-focused crop with 16 traits. Thorniness classification, growth habit assessment, borer detection, and 3D structure analysis."
---

# Black Locust — Robinia pseudoacacia

16 traits (10 automatable, 6 non-automatable) | 3 sensor types | 2 perspectives (aerial, ground) | 5 pipeline groups

## Phenology Calendar

1. **Dormancy** (Nov–Mar) — Bare branches, thorns visible
2. **Budbreak** (Apr–May) — Late leafing species
3. **Bloom** (May–Jun) — bloom_date. Pendant white racemes, fragrant. Brief bloom period.
4. **Pod development** (Jun–Sep) — Flat brown pods
5. **Senescence** (Oct) — Early leaf drop
6. **Biomass assessment** (Dormant season preferred for LiDAR)

## Trait Inventory

### Morphology (6 traits)
| Trait | Sensor | ML Task |
|-------|--------|---------|
| `plant_height` | Aerial LiDAR | Point cloud |
| `dbh` (diameter at breast height) | Aerial LiDAR | Point cloud |
| `canopy_volume` | Aerial LiDAR | Point cloud |
| `biomass_estimate` | Aerial LiDAR | Point cloud |
| `growth_habit` | Ground RGB | Classification |
| `thorniness` | Ground RGB | Classification |

### Disease (2 traits)
| Trait | Sensor | ML Task |
|-------|--------|---------|
| `locust_borer_damage` | Ground RGB | Object detection / Classification |
| `heart_rot` | Ground RGB | Classification |

### Yield (2 traits)
| Trait | Sensor | ML Task |
|-------|--------|---------|
| `pod_count` | Aerial/Ground RGB | Object detection |
| `bloom_date` | Aerial RGB | Change detection |

### Non-Automatable (6)
wood_density, moisture_content, BTU_content, cellulose_perc, lignin_perc, bark_thickness

## Annotation Guidance

- **Thorniness**: Score 0–3 (none, few, moderate, many). Requires close-up ground images of stems.
- **Growth habit**: Classify as upright, spreading, or weeping. Full-tree ground photos.
- **Locust borer**: Look for exit holes, sawdust frass, and bark damage on trunk/branches.
- **Pods**: Flat brown seed pods, 5-10cm long. Pendant from branches.
- **Biomass focus**: Primary value is wood production — 3D structure (LiDAR) is the main pipeline.

## ML Tasks

| Task | Key Traits |
|------|-----------|
| Classification | thorniness, growth_habit, heart_rot |
| Object detection | pod_count, locust_borer_damage |
| Change detection | bloom_date |
| Point cloud | plant_height, dbh, canopy_volume, biomass_estimate |
