import type {
  Acknowledgment,
  CanopySegmentDisclosure,
  PlantRegistryDisclosure,
  ServedDeliveryEvent,
  ServedPlantMappingDisclosure,
} from "@/api/types.generated";
import produced from "@/test/deliveryEvents.json";

/** A served delivery event whose disclosure is `D`. */
type Disclosed<D> = Omit<ServedDeliveryEvent, "plant_mapping"> & { plant_mapping: D };

/**
 * The delivery events `tools/generate_delivery_fixture.py` served from deliveries the platform's
 * own producers made: `validated` a per-image count delivery its assessment validates,
 * `acknowledged_mapping` a phenology delivery through a plant mapping shipped under a breeder's
 * acknowledgment, `archived_mapping` that same record once a superseding rebuild archived the
 * mapping it cites, and `registry` and `canopy` the orthomosaic door's nearest-plant and
 * canopy-segment deliveries.
 */
export const DELIVERY_RECORDS = produced.records as {
  validated: Disclosed<null>;
  acknowledged_mapping: Disclosed<ServedPlantMappingDisclosure> & {
    acknowledgment: Acknowledgment;
  };
  archived_mapping: Disclosed<ServedPlantMappingDisclosure>;
  registry: Disclosed<PlantRegistryDisclosure>;
  canopy: Disclosed<CanopySegmentDisclosure>;
};

/** The delivery headers the count-export door answered for the `validated` project's bucket. */
export const COUNT_EXPORT_HEADERS = produced.count_export_headers;
