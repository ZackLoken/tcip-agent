---
name: crop-science
description: "General crop science context for tree crop breeding programs. Covers phenology, sensor types, breeding program workflow, and the 6 TCIP crop species."
---

# Crop Science for Tree Crop Breeding

## TCIP Crops

Six tree crop species, each with unique traits and ML requirements:

| Crop | Traits | Sensors | Key Focus |
|------|--------|---------|-----------|
| Hazelnut | 58 | 7 (RGB, LiDAR, SfM, hyperspectral, NIRS, radar) | Most sensor-diverse; catkin/pistillate phenology, EFB disease |
| Chestnut | 21 | 4 | Blight resistance, bur/nut quality |
| Currant | 85 | 5 | Most trait-rich; 11 color traits, berry chemistry |
| Elderberry | 68 | 5 | Unique motion_tracking task (cyme shatter) |
| Persimmon | 20 | 3 | Simplest crop; fruit color/sizing |
| Black Locust | 16 | 3 | Biomass-focused; thorniness classification |

## Phenology

Phenology = timing of biological events (budbreak, bloom, fruit set, senescence). Key for breeding:
- Measured as calendar dates or growing degree days (GDD)
- Detected via temporal image series (change detection)
- Modeled with sigmoid curves for precise date estimation

## Sensor Types

| Sensor | Resolution | Use Cases |
|--------|-----------|-----------|
| RGB (drone) | 1-5 cm/px | Detection, segmentation, counting |
| RGB (ground) | Sub-mm | Lab measurements, fine morphology |
| LiDAR | Point cloud | 3D structure, volume, height |
| SfM | Point cloud from photos | 3D reconstruction from drone overlap |
| Multispectral | 5 bands | Vegetation indices, stress detection |
| Hyperspectral | 200+ bands | Chemistry, disease spectral signatures |
| NIRS | Spectrum | Nut quality (fat, moisture, protein) |

## Breeding Program Workflow

1. **Data collection**: Flights/ground surveys at key phenological stages
2. **Trait measurement**: Automated via ML pipelines
3. **Genetic analysis**: Breeders use per-plant CSV data for selection decisions
4. **Selection**: Plants with desirable trait combinations advance

Our role: automate step 2 to replace manual measurement.

## Perspectives

- **Aerial**: Drone flights (RGB, multispectral, LiDAR) — isolation at canopy level
- **Ground**: Handheld or tripod cameras — individual plant detail
- **Lab**: Controlled photography/NIRS of harvested samples — fine measurements
