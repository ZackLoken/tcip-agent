---
name: Persimmon
description: "Persimmon (Diospyros spp.) — simplest TCIP crop with 20 traits. Focus on fruit color/sizing, disease classification, and basic phenology."
---

# Persimmon — Diospyros spp.

20 traits (16 automatable, 4 non-automatable) | 3 sensor types | 3 perspectives | 6 pipeline groups

## Phenology Calendar

1. **Dormancy** (Dec–Mar) — Bare branches, persistent fruit on some varieties
2. **Budbreak** (Apr) — Late leafing compared to other TCIP crops
3. **Bloom** (May–Jun) — bloom_date. Small cream/white flowers.
4. **Fruit development** (Jun–Oct) — Slow maturation. Green → orange/red color change.
5. **Harvest** (Oct–Nov) — harvest_date. Fruit color and firmness indicators.
6. **Senescence** (Nov) — Leaf drop, some fruit persist

## Trait Inventory

### Phenology (2 traits)
| Trait | Sensor | ML Task |
|-------|--------|---------|
| `bloom_date` | Aerial RGB | Change detection |
| `harvest_date` | Aerial RGB | Change detection |

### Disease (3 traits)
| Trait | Sensor | ML Task |
|-------|--------|---------|
| `disease_severity` | Ground RGB | Classification (ordinal) |
| `anthracnose` | Ground RGB | Classification |
| `leaf_spot` | Ground RGB | Classification |

### Morphology (3 traits)
| Trait | Sensor | ML Task |
|-------|--------|---------|
| `plant_height` | Aerial LiDAR | Point cloud |
| `canopy_volume` | Aerial LiDAR | Point cloud |
| `cropload` | Aerial/Ground RGB | Classification (ordinal) |

### Quality (5 traits)
| Trait | Sensor | ML Task |
|-------|--------|---------|
| `fruit_length` | Lab RGB | Instance segmentation |
| `fruit_width` | Lab RGB | Instance segmentation |
| `fruit_weight` | Lab RGB | Regression |
| `fruit_color` | Lab RGB | Color analysis |
| `calyx_shape` | Lab RGB | Classification |

### Yield (3 traits)
| Trait | Sensor | ML Task |
|-------|--------|---------|
| `fruit_count` | Aerial/Ground RGB | Object detection |
| `total_yield` | Ground | Regression |
| `sex_determination` | Ground RGB | Classification |

### Non-Automatable (4)
tannin_content, astringency_rating, flesh_texture, flavor_rating

## Annotation Guidance

- **Fruit**: Spherical to slightly elongated. Orange/red at maturity. Clear bounding boxes.
- **Calyx**: Persistent 4-lobed calyx at fruit base. Classify shape (flat, recurved, spreading).
- **Cropload**: Visual estimate of fruit density per tree. Score 1–5 ordinal.
- **Sex determination**: Male vs female vs monoecious flowers. Requires close-up ground images.
- **Simplest crop to annotate** — fruits are large, distinct, and well-separated compared to berries.

## ML Tasks

| Task | Key Traits |
|------|-----------|
| Classification | disease_severity, cropload, calyx_shape, sex |
| Change detection | bloom_date, harvest_date (ripening color) |
| Object detection | fruit_count |
| Instance segmentation | fruit dimensions (lab) |
| Regression | tannins, weight |
| Point cloud | plant_height, canopy_volume |
