/** The sentences a delivery's plant attribution discloses, each written once for every panel. */

import type { PlantMappingDisclosure } from "@/api/types.generated";

/** The dates a plant mapping delivered and how many of their images it attributed to no plant. */
export function deliveredDatesLine(disclosure: PlantMappingDisclosure): string {
  return (
    `Delivered dates ${disclosure.dates_delivered.join(", ")}: ` +
    `${disclosure.images_unattributed} attributed to no plant ` +
    `(${disclosure.plant_attribution}-level attribution)`
  );
}

/** The registry plants a delivered raster does not cover. */
export function outsideRasterLine(plants: string[]): string {
  return `Outside the raster: ${plants.join(", ")}`;
}
