---
name: Elderberry
description: "Elderberry (Sambucus spp.) — 68 traits with unique motion_tracking task for cyme shatter resistance. 9 phenology dates, 7 color traits, extensive disease and quality assessment."
---

# Elderberry — Sambucus spp.

68 traits (53 automatable, 15 non-automatable) | 5 sensor types | 3 perspectives | 13 pipeline groups

## Phenology Calendar

1. **Dormancy** (Nov–Feb) — Bare canes
2. **Budbreak** (Mar) — bud_break_date
3. **Leaf out** (Mar–Apr) — leaf_out_date
4. **Bloom** (May–Jun) — bloom_start_date, full_bloom_date, bloom_end_date. Large white cyme inflorescences.
5. **Fruit set** (Jun–Jul) — Green berries forming
6. **Fruit development** (Jul–Aug) — Berry color change (green → purple/black)
7. **Harvest** (Aug–Sep) — harvest_start_date, harvest_end_date. Cyme shatter test.
8. **Senescence** (Sep–Nov) — senescence_date, defoliation_date

## Unique: Motion Tracking

Elderberry is the **only TCIP crop with a motion_tracking task**.

**Cyme shatter resistance**: Measures how easily berries fall off the cyme (fruit cluster) when shaken. Critical for mechanical harvest suitability.
- Video capture of cyme shaking
- Track berry detachment over time
- Quantify shatter rate (berries lost / total berries / shake duration)

## Trait Inventory

### Phenology (9 dates)
bud_break_date, leaf_out_date, bloom_start_date, full_bloom_date, bloom_end_date, harvest_start_date, harvest_end_date, senescence_date, defoliation_date
All via Aerial RGB → Change detection

### Color (7 traits)
berry_color, juice_color, flower_color, stem_color, leaf_color_spring, leaf_color_summer, leaf_color_fall
Via Ground/Lab RGB → Color analysis

### Disease (5+ traits)
elder_shoot_borer, eriophyid_mite, phomopsis, tomato_ringspot_virus, powdery_mildew
Via Ground RGB → Classification

### Morphology (15+ traits)
plant_height, plant_spread, cane_count, cane_diameter, internode_length, leaf_area, cyme_count, cyme_size, berry_count_per_cyme
Via LiDAR/SfM + Ground RGB

### Quality (12+ traits)
berry_weight, brix, pH, anthocyanins, phenolics, vitamin_c, berry_firmness, seed_count
Via Lab RGB/NIRS

### Yield (5 traits)
total_yield, cyme_count, cyme_weight, berry_count, harvest_efficiency

## Annotation Guidance

- **Cymes**: Large flat-topped flower/fruit clusters (10-25cm diameter). Annotate with bounding boxes or polygons.
- **Berries**: Tiny (3-5mm), dark purple/black at maturity. Very dense in cymes — strig-level annotation preferred.
- **Shatter**: Video annotation — track individual berries frame-by-frame during shake test.
- **Shoot borer damage**: Wilted/dead shoot tips. Annotate affected stems.

## ML Tasks

| Task | Key Traits |
|------|-----------|
| Change detection | 9 phenology dates |
| Classification | 5+ diseases, flower_color |
| Object detection | cyme_count, berry_count, cane_count |
| Color analysis | 7 color traits |
| Motion tracking | cyme_shatter_resistance (unique) |
| Instance segmentation | berry/cyme dimensions (lab) |
| Regression | brix, pH, anthocyanins, chemistry |
| Point cloud | plant dimensions |
