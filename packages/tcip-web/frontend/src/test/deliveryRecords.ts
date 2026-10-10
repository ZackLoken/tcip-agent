import type { DeliveryEventRecord } from "@/api/inference";
import records from "@/test/deliveryEvents.json";

/**
 * The delivery events `tools/generate_delivery_fixture.py` served from deliveries the platform's
 * own producers made: `validated` a per-image count delivery its assessment validates,
 * `acknowledged_mapping` a phenology delivery through a plant mapping shipped under a breeder's
 * acknowledgment, `archived_mapping` that same record once a superseding rebuild archived the
 * mapping it cites, and `registry` and `canopy` the orthomosaic door's nearest-plant and
 * canopy-segment deliveries.
 */
export const DELIVERY_RECORDS = records as Record<
  "validated" | "acknowledged_mapping" | "archived_mapping" | "registry" | "canopy",
  DeliveryEventRecord
>;
