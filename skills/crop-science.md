---
name: crop-science
description: "Crop biology, phenology, and breeding context for the 6 supported tree crops. Growth forms, phenological stages, trait profiles, disease identification, and measurement scales."
triggers:
  - crop
  - hazelnut
  - chestnut
  - currant
  - elderberry
  - persimmon
  - black_locust
  - phenology
  - breeding
  - disease
  - trait
modes: [PipelineDesigner, ResultsAnalyzer]
priority: high
max_chars: 4000
---

# Crop Science

## Purpose

Provide crop biology, phenology, and breeding context for the 6 supported tree crops. Guide trait interpretation and pipeline design with domain knowledge.

## Crop Profiles

### Hazelnut (58 traits)
**Growth**: Multi-stemmed bush, 3-5m. **Phenology**: catkin emergence (winter) → leaf out → nut set → harvest (fall). **Key targets**: catkin phenology dates, nut yield, kernel quality, blight resistance. **Imaging**: ground RGB (catkins), aerial RGB (canopy), lab (kernel). **MVP**: catkin_05per_date, catkin_50per_date, catkin_95per_date, catkin_elongation_date.

### Chestnut (21 traits)
**Growth**: Large tree, 15-25m. **Phenology**: bud break → flowering → bur set → harvest. **Key targets**: bur count, nut size, blight resistance, tree form. **Imaging**: ground RGB, aerial RGB, aerial LiDAR.

### Currant (86 traits — most traits)
**Growth**: Small bush, 1-2m. **Phenology**: leaf out → flowering → fruit set → harvest. **Key targets**: yield components (berry count, cluster size), fruit quality (color, Brix), disease resistance. **Imaging**: ground RGB, lab NIRS (Brix/chemistry).

### Elderberry (69 traits)
**Growth**: Multi-stemmed shrub, 2-4m. **Phenology**: leaf out → cyme flowering → berry ripening. **Key targets**: cyme count, berry color stage, anthocyanin content, plant vigor. **Imaging**: ground RGB, aerial RGB, lab NIRS.

### Persimmon (20 traits)
**Growth**: Tree, 5-12m. **Phenology**: leaf out → flowering → fruit set → color change → harvest. **Key targets**: fruit count, color stage, calyx size, astringency. **Imaging**: ground RGB, aerial RGB.

### Black Locust (10 traits — fewest)
**Growth**: Large tree, 15-25m, thorny. **Phenology**: leaf out → raceme flowering → pod set. **Key targets**: flower timing, growth rate, wood quality indicators. **Imaging**: aerial RGB, ground RGB.

## Phenotyping Measurement Scales

| Scale | Examples | ML Task |
|-------|----------|---------|
| Count | berries, catkins, flowers | Detection → count |
| Date | flowering onset, harvest | Detection → temporal sigmoid |
| Ordinal (1-9) | vigor, disease severity | Ordinal classification (CORN) |
| Continuous | diameter, height, Brix | Regression |
| Categorical | color stage, growth form | Classification |
| Binary | presence/absence | Binary classification |

## Disease Identification

**Visual cues**: lesions (brown/black spots), cankers (sunken bark), chlorosis (yellowing), necrosis (dead tissue), mildew (white powder), rust (orange pustules).

**Scoring**: ordinal 0-9 scale where 0=no symptoms, 9=severe. Use CORN/CORAL loss for ordinal traits, NOT standard cross-entropy.

## Key Constraints

- 6 crops only: hazelnut, chestnut, currant, elderberry, persimmon, black_locust
- Always query registry for exact trait definitions — do not assume
- Interpret model results in breeding context (breeders care about ranking genotypes, not absolute accuracy)
- Phenological traits require temporal data — single timepoint is insufficient

## Imaging Perspectives

| Perspective | Coverage | Best For |
|-------------|----------|----------|
| Aerial (88 traits) | Full orchard, 0.5-3 cm/px GSD | Canopy cover, tree counting, vigor, 3D structure |
| Lab (63 traits) | Per-sample, very high res | Fruit/nut quality, NIRS chemistry, morphology |
| Ground (61 traits) | Single tree, very high res | Disease scoring, catkin counting, close-range traits |

## Resolution Requirements

| Trait Category | Min GSD (aerial) | Ground/Lab |
|---------------|------------------|------------|
| Individual flower/catkin | 0.5 cm/px | 0.1 mm/px |
| Fruit/nut counting | 1.0 cm/px | 0.5 mm/px |
| Disease lesion | 0.5 cm/px | 0.2 mm/px |
| Canopy boundary | 3.0 cm/px | N/A |
| Tree location/count | 5.0 cm/px | N/A |
