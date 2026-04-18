---
name: Currant
description: "Currant (Ribes spp.) — most trait-rich TCIP crop with 85 traits. 11 color traits, berry chemistry, 9 phenology dates, extensive disease and morphology assessment."
---

# Currant — Ribes spp.

85 traits (65 automatable, 20 non-automatable) | 5 sensor types | 3 perspectives | 13 pipeline groups

## Phenology Calendar

1. **Dormancy** (Nov–Feb) — Bare canes
2. **Budbreak** (Mar) — bud_break_date
3. **Leaf out** (Mar–Apr) — leaf_out_date
4. **Bloom** (Apr–May) — bloom_start_date, full_bloom_date, bloom_end_date
5. **Fruit set** (May–Jun) — Green berries forming on strigs
6. **Fruit development** (Jun–Jul) — Berry sizing, color change
7. **Harvest** (Jul–Aug) — harvest_start_date, harvest_end_date
8. **Senescence** (Sep–Nov) — senescence_date, defoliation_date

## Trait Inventory

### Phenology (9 dates)
bud_break_date, leaf_out_date, bloom_start_date, full_bloom_date, bloom_end_date, harvest_start_date, harvest_end_date, senescence_date, defoliation_date
All via Aerial RGB → Change detection (temporal sigmoid fitting)

### Color (11 traits)
berry_color, berry_skin_color, juice_color, flesh_color, stem_color, leaf_color_spring, leaf_color_summer, leaf_color_fall, cane_color_dormant, cane_color_growing, bud_color
Via Ground/Lab RGB → Color analysis (LAB/HSV space)

### Disease (6+ traits)
powdery_mildew, white_pine_blister_rust, anthracnose, leaf_spot, gall_mite, aphid_damage
Via Ground RGB → Classification (ordinal severity)

### Morphology (22 traits)
Including: plant_height, plant_spread, cane_count, cane_diameter, internode_length, leaf_area, leaf_length, leaf_width, strig_length, berry_count_per_strig, berries_per_cluster
Via LiDAR/SfM (point cloud), Ground RGB (detection/segmentation/regression)

### Quality (22 traits)
Including: berry_weight, berry_diameter, brix, pH, titratable_acidity, total_anthocyanins, vitamin_c, berry_firmness, seed_count, juice_yield
Via Lab RGB (instance segmentation), Lab NIRS (regression)

### Yield (5 traits)
total_yield, berry_count, strig_count, cluster_weight, harvest_efficiency
Via Aerial/Ground RGB (detection, regression)

## Annotation Guidance

- **Berries**: Small spherical fruits on strigs (racemes). Very dense — individual berry annotation is tedious. Consider strig-level detection.
- **Strigs**: Elongated fruit clusters hanging from branches. Annotate with bounding boxes.
- **Color traits**: Use calibration target in frame for consistent color measurement.
- **Powdery mildew**: White powdery coating on leaves/berries. Score 0–5 severity.
- **Cane counting**: Dormant season imaging preferred. Canes emerge from crown.

## ML Tasks

| Task | Key Traits |
|------|-----------|
| Change detection | 9 phenology dates |
| Classification | 6+ diseases, bud_color, berry_color |
| Object detection | berry_count, strig_count, cane_count |
| Color analysis | 11 color traits (LAB/HSV) |
| Instance segmentation | berry dimensions (lab) |
| Regression | brix, pH, anthocyanins, chemistry |
| Point cloud | plant dimensions (LiDAR/SfM) |
